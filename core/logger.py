"""
core/logger.py

Production-grade structured logger.
Every module imports from here — never uses print() directly.

Design decisions:
  - Structured JSON logs to file  → machine readable, queryable
  - Human readable logs to console → developer experience
  - Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
  - Every log line carries: timestamp, level, module, run_id, message, context
  - run_id ties all logs from one agent run together
  - Single logger instance per process — no duplicate handlers

Usage:
  from core.logger import get_logger
  logger = get_logger(__name__)

  logger.info("jobs_found", count=3, company="Amex")
  logger.warning("title_inflation", title="Director", reason="junior JD")
  logger.error("scrape_failed", company="Amex", error=str(e))

Why __name__:
  get_logger(__name__) passes the module path automatically.
  e.g. agents.discovery, tools.scraper, memory.store
  Every log line shows exactly which module produced it.
  No manual naming needed.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from config.config import LOG_PATH, ENVIRONMENT


# ── Run ID ────────────────────────────────────────────────────────────────────
# Generated once per process start.
# Every log line from this run carries this ID.
# When you have 1000 log lines — filter by run_id to see one run's story.
# In production with multiple users — run_id separates their logs.
RUN_ID = str(uuid.uuid4())[:8]     # Short form — e.g. "a3f8c1d2"


# ── Structured JSON formatter ─────────────────────────────────────────────────
class StructuredFormatter(logging.Formatter):
    """
    Formats log records as JSON lines.
    Each line is a complete, self-describing JSON object.
    Machine readable — can be ingested by CloudWatch, Datadog, ELK.
    This is the production log format used by Netflix, Uber, LinkedIn.

    Output example:
    {
      "timestamp": "2026-05-10T07:23:11",
      "level": "INFO",
      "run_id": "a3f8c1d2",
      "module": "agents.discovery",
      "event": "jobs_found",
      "count": 3,
      "company": "Amex"
    }
    """

    def format(self, record: logging.LogRecord) -> str:
        # Base structure — every log line has these
        log_obj = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level"    : record.levelname,
            "run_id"   : RUN_ID,
            "module"   : record.name,
            "event"    : record.getMessage(),
        }

        # Merge any extra context fields the caller passed
        # e.g. logger.info("jobs_found", count=3) → count appears in JSON
        if hasattr(record, "context") and record.context:
            log_obj.update(record.context)

        # Include exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, default=str)


# ── Human readable formatter ──────────────────────────────────────────────────
class ConsoleFormatter(logging.Formatter):
    """
    Readable format for console output during development.
    Colour coded by level — instant visual scanning.
    Not used in production file logs — too noisy to parse.
    """

    COLOURS = {
        "DEBUG"   : "\033[36m",    # Cyan
        "INFO"    : "\033[32m",    # Green
        "WARNING" : "\033[33m",    # Yellow
        "ERROR"   : "\033[31m",    # Red
        "CRITICAL": "\033[35m",    # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour  = self.COLOURS.get(record.levelname, "")
        reset   = self.RESET
        level   = f"{colour}{record.levelname:8}{reset}"
        module  = f"{record.name:30}"
        message = record.getMessage()

        # Append context fields inline for readability
        context_str = ""
        if hasattr(record, "context") and record.context:
            parts = [f"{k}={v}" for k, v in record.context.items()]
            context_str = "  |  " + "  ".join(parts)

        time_str = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")

        return f"{time_str}  {level}  {module}  {message}{context_str}"


# ── Context-aware logger adapter ──────────────────────────────────────────────
class AgentLogger(logging.LoggerAdapter):
    """
    Wraps standard Logger to support structured context kwargs.

    Standard logger:
      logger.info("Found 3 jobs")              ← unstructured

    AgentLogger:
      logger.info("jobs_found", count=3)       ← structured
      logger.info("jobs_found", count=3, company="Amex", run_id=RUN_ID)

    The kwargs become fields in the JSON log output.
    This is how you build queryable, observable logs.
    """

    def log(self, level: int, msg: str, *args, **kwargs) -> None:
        # Extract context kwargs — everything except standard logging kwargs
        standard_kwargs = {
            "exc_info", "stack_info", "stacklevel", "extra"
        }
        context = {
            k: v for k, v in kwargs.items()
            if k not in standard_kwargs
        }

        # Inject context into the log record via 'extra'
        extra = kwargs.pop("extra", {})
        extra["context"] = context

        # Remove context keys from kwargs before passing to standard logger
        for k in list(context.keys()):
            kwargs.pop(k, None)

        super().log(level, msg, *args, extra=extra, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.log(logging.CRITICAL, msg, *args, **kwargs)


# ── Logger registry ───────────────────────────────────────────────────────────
# Keeps one logger instance per module name.
# Prevents duplicate handlers if get_logger() called multiple times.
_loggers: dict[str, AgentLogger] = {}
_handlers_configured = False


def _configure_root_logger():
    """
    Configures the root logger once per process.
    Called automatically on first get_logger() call.
    Subsequent calls are no-ops — handlers not duplicated.
    """
    global _handlers_configured
    if _handlers_configured:
        return

    # Ensure log directory exists
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()

    # Set level based on environment
    # DEBUG in development — see everything
    # INFO in production — see important events only
    level = logging.DEBUG if ENVIRONMENT == "development" else logging.INFO
    root.setLevel(level)

    # ── File handler — structured JSON ────────────────────────────────────────
    # Rotating would be added in full production:
    # from logging.handlers import RotatingFileHandler
    # handler = RotatingFileHandler(LOG_PATH, maxBytes=10MB, backupCount=5)
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(StructuredFormatter())
    file_handler.setLevel(logging.DEBUG)
    root.addHandler(file_handler)

    # ── Console handler — human readable ─────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ConsoleFormatter())
    console_handler.setLevel(logging.DEBUG)
    root.addHandler(console_handler)

    _handlers_configured = True


def get_logger(name: str) -> AgentLogger:
    """
    Primary entry point. Every module calls this.

    Usage:
      from core.logger import get_logger
      logger = get_logger(__name__)

    __name__ is Python's built-in module path.
    In agents/discovery.py it becomes "agents.discovery".
    In tools/scraper.py it becomes "tools.scraper".
    No manual naming. No typos. Always correct.
    """
    _configure_root_logger()

    if name not in _loggers:
        base_logger = logging.getLogger(name)
        _loggers[name] = AgentLogger(base_logger, extra={})

    return _loggers[name]


# ── Run context helper ────────────────────────────────────────────────────────
def get_run_id() -> str:
    """
    Returns the current run ID.
    Orchestrator uses this to tag all agent runs with same ID.
    Makes tracing a full run across multiple agents possible.
    """
    return RUN_ID


# ── Self test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from config.config import ensure_directories
    ensure_directories()

    # Simulate how different modules would use the logger
    logger = get_logger(__name__)

    print("\n── Logger self test ──────────────────────────────────────")
    print(f"Run ID : {RUN_ID}")
    print(f"Log file: {LOG_PATH}\n")

    logger.debug("debug_event",
                 detail="visible in development only")

    logger.info("agent_started",
                agent="discovery",
                run_id=RUN_ID,
                environment=ENVIRONMENT)

    logger.info("jobs_found",
                agent="discovery",
                company="American Express",
                count=5,
                location="Chennai")

    logger.warning("title_inflation_detected",
                   title="Director of Engineering",
                   reason="JD contains '2-3 years experience'",
                   action="flagged but not saved")

    logger.error("scrape_failed",
                 company="Standard Chartered",
                 error="Connection timeout after 30s",
                 retry="scheduled")

    logger.info("run_complete",
                agent="discovery",
                jobs_saved=3,
                jobs_skipped=2,
                duration_seconds=12.4)

    print(f"\n── Check log file for JSON output ────────────────────────")
    print(f"   {LOG_PATH}")

    # Show what the JSON file looks like
    if LOG_PATH.exists():
        print(f"\n── Last 3 lines of log file ──────────────────────────────")
        lines = LOG_PATH.read_text().strip().split("\n")
        for line in lines[-3:]:
            parsed = json.loads(line)
            print(json.dumps(parsed, indent=2))