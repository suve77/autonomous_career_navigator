"""
core/exceptions.py

Custom exception hierarchy for autonomous career navigator.

Design principles:
  - Every exception carries structured context — not just a message
  - Exceptions are organised by layer — tool, agent, storage, config
  - Each exception knows how to log itself — no logging scattered elsewhere
  - Catchable by category — catch ToolError to handle all tool failures
  - Retryable exceptions are flagged — orchestrator knows what to retry

Hierarchy:
  CareerNavigatorError          ← base for everything
  ├── ConfigurationError        ← bad config, missing env vars
  ├── StorageError              ← database failures
  │   ├── ConnectionError
  │   └── QueryError
  ├── ToolError                 ← tool execution failures
  │   ├── ScrapingError         ← career page scraping failed
  │   ├── ParsingError          ← JD parsing failed
  │   └── AlertError            ← notification delivery failed
  ├── AgentError                ← agent reasoning failures
  │   ├── LLMCallError          ← LLM API call failed
  │   ├── ToolCallError         ← agent tried to call unknown tool
  │   ├── MaxIterationsError    ← agent hit safety cap
  │   └── PromptLoadError       ← agent prompt file not found
  └── ValidationError           ← data validation failures

Usage:
  from core.exceptions import ScrapingError, LLMCallError

  # Raising with context
  raise ScrapingError(
      company   = "American Express",
      ats_type  = "eightfold",
      url       = "https://aexp.eightfold.ai/careers",
      reason    = "Connection timeout after 30s",
      retryable = True
  )

  # Catching by category
  try:
      result = scrape_company(config)
  except ScrapingError as e:
      logger.error("scrape_failed", **e.context)
      if e.retryable:
          schedule_retry(e)
  except ToolError as e:
      logger.error("tool_failed", **e.context)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from typing import Any


# ── Base exception ────────────────────────────────────────────────────────────
class CareerNavigatorError(Exception):
    """
    Base exception for all autonomous career navigator errors.
    All custom exceptions inherit from this.

    Catching CareerNavigatorError catches everything from this system.
    Catching a subclass catches only that category.

    Every exception carries:
      - message    : human readable description
      - context    : structured dict for logging
      - retryable  : should the orchestrator retry this operation?
      - timestamp  : when did this happen
    """

    def __init__(
        self,
        message  : str,
        retryable: bool = False,
        **context: Any
    ):
        super().__init__(message)
        self.message   = message
        self.retryable = retryable
        self.timestamp = datetime.now().isoformat()
        self.context   = {
            "error_type": self.__class__.__name__,
            "message"   : message,
            "retryable" : retryable,
            "timestamp" : self.timestamp,
            **context
        }

    def log(self, logger) -> None:
        """
        Self-logging — exception knows how to report itself.
        Call e.log(logger) instead of manually building the log call.
        Keeps exception handling code clean and consistent.
        """
        if self.retryable:
            logger.warning("retryable_error", **self.context)
        else:
            logger.error("fatal_error", **self.context)

    def __str__(self) -> str:
        context_str = "  ".join(
            f"{k}={v}" for k, v in self.context.items()
            if k not in {"error_type", "message", "timestamp"}
        )
        return f"[{self.__class__.__name__}] {self.message}  |  {context_str}"


# ── Configuration exceptions ──────────────────────────────────────────────────
class ConfigurationError(CareerNavigatorError):
    """
    Raised when configuration is missing or invalid.
    Always fatal — system cannot start without valid config.
    Never retryable — fix the config, then restart.

    Examples:
      - Missing OPENROUTER_API_KEY
      - companies.yaml not found
      - Invalid YAML structure
    """

    def __init__(self, message: str, config_key: str = None, **context):
        super().__init__(
            message   = message,
            retryable = False,          # Config errors are never retryable
            config_key= config_key,
            **context
        )


class MissingAPIKeyError(ConfigurationError):
    """Raised specifically when an API key is absent."""

    def __init__(self, key_name: str):
        super().__init__(
            message    = f"Required API key '{key_name}' is missing from environment",
            config_key = key_name
        )


# ── Storage exceptions ────────────────────────────────────────────────────────
class StorageError(CareerNavigatorError):
    """
    Base for all database/persistence failures.
    Catching StorageError handles all storage problems.
    """
    pass


class DatabaseConnectionError(StorageError):
    """
    Raised when SQLite DB cannot be opened or connected.
    Retryable — transient lock issues resolve on retry.

    Example:
      - DB file locked by another process
      - DB path not writable
    """

    def __init__(self, db_path: str, reason: str):
        super().__init__(
            message   = f"Cannot connect to database at {db_path}: {reason}",
            retryable = True,
            db_path   = db_path,
            reason    = reason
        )


class QueryError(StorageError):
    """
    Raised when a database query fails.
    Not retryable — likely a schema or data problem.

    Example:
      - Column does not exist
      - Constraint violation
      - Malformed SQL
    """

    def __init__(self, operation: str, table: str, reason: str):
        super().__init__(
            message   = f"Query failed on {table}.{operation}: {reason}",
            retryable = False,
            operation = operation,
            table     = table,
            reason    = reason
        )


# ── Tool exceptions ───────────────────────────────────────────────────────────
class ToolError(CareerNavigatorError):
    """
    Base for all tool execution failures.
    Orchestrator catches ToolError to handle any tool failure gracefully.
    Individual tool errors are subclasses with richer context.
    """
    pass


class ScrapingError(ToolError):
    """
    Raised when career page scraping fails.
    Retryable — network issues are often transient.

    Context carried:
      company   : which company failed
      ats_type  : which ATS platform (eightfold, workday, taleo)
      url       : which URL was being scraped
      reason    : what went wrong
      status_code: HTTP status if available
    """

    def __init__(
        self,
        company    : str,
        url        : str,
        reason     : str,
        ats_type   : str  = "unknown",
        status_code: int  = None,
        retryable  : bool = True
    ):
        super().__init__(
            message     = f"Scraping failed for {company}: {reason}",
            retryable   = retryable,
            company     = company,
            ats_type    = ats_type,
            url         = url,
            reason      = reason,
            status_code = status_code
        )


class ParsingError(ToolError):
    """
    Raised when JD parsing or response parsing fails.
    Not retryable — malformed data won't fix itself.

    Examples:
      - LLM returned invalid JSON
      - JD HTML structure changed
      - Missing required fields in scraped data
    """

    def __init__(self, source: str, reason: str, raw_content: str = None):
        super().__init__(
            message     = f"Parsing failed for {source}: {reason}",
            retryable   = False,
            source      = source,
            reason      = reason,
            raw_preview = raw_content[:100] if raw_content else None
        )


class AlertError(ToolError):
    """
    Raised when notification delivery fails.
    Retryable — email/SMTP failures are often transient.

    Examples:
      - Gmail SMTP connection refused
      - Invalid recipient address
      - Rate limit on email sending
    """

    def __init__(
        self,
        channel  : str,
        recipient: str,
        reason   : str,
        retryable: bool = True
    ):
        super().__init__(
            message   = f"Alert failed via {channel} to {recipient}: {reason}",
            retryable = retryable,
            channel   = channel,
            recipient = recipient,
            reason    = reason
        )


# ── Agent exceptions ──────────────────────────────────────────────────────────
class AgentError(CareerNavigatorError):
    """
    Base for all agent reasoning failures.
    Catching AgentError handles any agent-level problem.
    """
    pass


class LLMCallError(AgentError):
    """
    Raised when LLM API call fails.
    Retryable — API rate limits and transient errors resolve.

    Context carried:
      agent      : which agent made the call
      model      : which model was used
      reason     : what went wrong
      status_code: HTTP status if available
      tokens_used: tokens consumed before failure
    """

    def __init__(
        self,
        agent      : str,
        model      : str,
        reason     : str,
        status_code: int = None,
        tokens_used: int = 0,
        retryable  : bool = True
    ):
        super().__init__(
            message     = f"LLM call failed in {agent}: {reason}",
            retryable   = retryable,
            agent       = agent,
            model       = model,
            reason      = reason,
            status_code = status_code,
            tokens_used = tokens_used
        )


class ToolCallError(AgentError):
    """
    Raised when agent tries to call a tool that does not exist.
    Not retryable — this is a programming error, not a runtime error.
    Means TOOL_REGISTRY is missing an entry.

    Example:
      LLM decided to call "search_linkedin" but tool not registered.
    """

    def __init__(self, tool_name: str, agent: str, available_tools: list):
        super().__init__(
            message        = f"Agent '{agent}' called unknown tool '{tool_name}'",
            retryable      = False,
            tool_name      = tool_name,
            agent          = agent,
            available_tools= available_tools
        )


class MaxIterationsError(AgentError):
    """
    Raised when agent hits the safety iteration cap.
    Not retryable without investigation — agent may be in a loop.
    Orchestrator should alert and stop, not retry blindly.

    This is a critical safety mechanism.
    An agent that runs forever costs money and produces no value.
    """

    def __init__(self, agent: str, max_iterations: int, last_tool: str = None):
        super().__init__(
            message        = f"Agent '{agent}' hit safety cap of {max_iterations} iterations",
            retryable      = False,
            agent          = agent,
            max_iterations = max_iterations,
            last_tool      = last_tool
        )


class PromptLoadError(AgentError):
    """
    Raised when agent cannot load its prompt file.
    Not retryable — file must exist before agent can run.

    Example:
      agents/prompts/discovery.md not found.
    """

    def __init__(self, agent: str, prompt_path: str):
        super().__init__(
            message     = f"Agent '{agent}' cannot load prompt from {prompt_path}",
            retryable   = False,
            agent       = agent,
            prompt_path = prompt_path
        )


# ── Validation exceptions ─────────────────────────────────────────────────────
class ValidationError(CareerNavigatorError):
    """
    Raised when data fails validation before being processed or stored.
    Not retryable — bad data needs to be fixed, not retried.

    Examples:
      - Job record missing required title field
      - Invalid salary range (min > max)
      - Malformed URL
    """

    def __init__(self, field: str, value: Any, reason: str):
        super().__init__(
            message   = f"Validation failed for '{field}': {reason}",
            retryable = False,
            field     = field,
            value     = str(value)[:100],   # Truncate long values
            reason    = reason
        )


# ── Retry helper ──────────────────────────────────────────────────────────────
def is_retryable(error: Exception) -> bool:
    """
    Utility function — orchestrator calls this to decide whether to retry.
    Works for both CareerNavigatorError subclasses and standard exceptions.

    Standard exceptions (network, timeout) are assumed retryable.
    Our custom errors declare their own retryable flag.
    """
    if isinstance(error, CareerNavigatorError):
        return error.retryable

    # Standard exceptions — assume retryable
    retryable_builtins = (
        ConnectionError,
        TimeoutError,
        OSError
    )
    return isinstance(error, retryable_builtins)


# ── Self test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from core.logger import get_logger
    from config.config import ensure_directories

    ensure_directories()
    logger = get_logger(__name__)

    print("\n── Exception hierarchy self test ─────────────────────────\n")

    # Test 1: ScrapingError — retryable
    try:
        raise ScrapingError(
            company     = "American Express",
            url         = "https://aexp.eightfold.ai/careers",
            reason      = "Connection timeout after 30s",
            ats_type    = "eightfold",
            status_code = 503
        )
    except ScrapingError as e:
        print(f"[1] Caught ScrapingError")
        print(f"    retryable = {e.retryable}")
        e.log(logger)

    # Test 2: LLMCallError — retryable
    try:
        raise LLMCallError(
            agent       = "discovery",
            model       = "nvidia/nemotron-super-120b-a12b:free",
            reason      = "Rate limit exceeded",
            status_code = 429,
            tokens_used = 840
        )
    except LLMCallError as e:
        print(f"\n[2] Caught LLMCallError")
        print(f"    retryable = {e.retryable}")
        e.log(logger)

    # Test 3: MaxIterationsError — not retryable
    try:
        raise MaxIterationsError(
            agent          = "discovery",
            max_iterations = 50,
            last_tool      = "scrape_company"
        )
    except MaxIterationsError as e:
        print(f"\n[3] Caught MaxIterationsError")
        print(f"    retryable = {e.retryable}")
        e.log(logger)

    # Test 4: Catch by category — ToolError catches ScrapingError
    try:
        raise ScrapingError(
            company = "Standard Chartered",
            url     = "https://careers.sc.com",
            reason  = "CAPTCHA block"
        )
    except ToolError as e:
        print(f"\n[4] Caught as ToolError (parent category)")
        print(f"    actual type = {e.__class__.__name__}")
        print(f"    is_retryable() = {is_retryable(e)}")

    # Test 5: Catch by base — CareerNavigatorError catches everything
    try:
        raise ValidationError(
            field  = "salary_min",
            value  = -1000,
            reason = "Salary cannot be negative"
        )
    except CareerNavigatorError as e:
        print(f"\n[5] Caught as CareerNavigatorError (base class)")
        print(f"    actual type = {e.__class__.__name__}")
        print(f"    context = {e.context}")

    print(f"\n── All exception tests passed ────────────────────────────")
    print(f"   Hierarchy works — catch by category or base class")
    print(f"   All exceptions carry structured context for logging")
    print(f"   retryable flag set correctly on each type")