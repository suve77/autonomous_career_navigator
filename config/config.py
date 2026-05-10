"""
core/config.py

Single source of truth for all configuration.
Every other module imports from here.
Changing model, storage path, or API provider = edit this file only.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env file ────────────────────────────────────────────────────────────
# Looks for .env in the project root (one level above core/)
load_dotenv()


# ── Project paths ─────────────────────────────────────────────────────────────
# Path() gives us OS-independent paths — works on Windows, Mac, Linux
# __file__ = this file's location (core/config.py)
# .parent   = core/
# .parent   = project root/
BASE_DIR    = Path(__file__).parent.parent
CONFIG_DIR  = BASE_DIR / "config"
DB_PATH     = BASE_DIR / "data" / "career_navigator.db"
LOG_PATH    = BASE_DIR / "logs" / "agent.log"


# ── LLM configuration ─────────────────────────────────────────────────────────
# os.getenv reads from .env file (or system environment variables)
# The second argument is the default if the key is not found
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME          = os.getenv("MODEL_NAME", "nvidia/nemotron-super-120b-a12b:free")
ENVIRONMENT         = os.getenv("ENVIRONMENT", "development")

# Max tokens the LLM will return per call
# Keep low during dev — saves your free credits
MAX_TOKENS = 1000


# ── Storage configuration ─────────────────────────────────────────────────────
# SQLite — single file database, built into Python, zero setup
# Lives in data/ folder which we create below


# ── Alert configuration ───────────────────────────────────────────────────────
GMAIL_SENDER    = os.getenv("GMAIL_SENDER", "")
GMAIL_PASSWORD  = os.getenv("GMAIL_PASSWORD", "")   # App password, not login password
GMAIL_RECEIVER  = os.getenv("GMAIL_RECEIVER", "")


# ── Agent configuration ───────────────────────────────────────────────────────
# How often the discovery agent runs (in hours)
SCRAPE_INTERVAL_HOURS = 24

# Minimum fitment score to trigger an alert (0-100)
FITMENT_THRESHOLD = 60

# How many days back to consider a job "new"
JOB_FRESHNESS_DAYS = 7


# ── Validation ────────────────────────────────────────────────────────────────
def validate_config() -> bool:
    """
    Call this at startup to catch missing config early.
    Fails loudly rather than mysteriously later.
    This is called 'fail fast' — a core engineering principle.
    """
    errors = []

    if not OPENROUTER_API_KEY:
        errors.append("OPENROUTER_API_KEY is missing from .env")

    if not GMAIL_SENDER:
        errors.append("GMAIL_SENDER is missing from .env")

    if errors:
        for error in errors:
            print(f"[CONFIG ERROR] {error}")
        return False

    return True


# ── Ensure folders exist ──────────────────────────────────────────────────────
def ensure_directories():
    """
    Creates data/ and logs/ folders if they don't exist.
    exist_ok=True means no error if folder already exists.
    Called once at startup.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


# ── Quick self-test ───────────────────────────────────────────────────────────
# This block runs ONLY when you execute this file directly:
#   python core/config.py
# It does NOT run when another module imports it.
# Use this to verify your config is loading correctly.
if __name__ == "__main__":
    ensure_directories()
    print(f"Environment  : {ENVIRONMENT}")
    print(f"Model        : {MODEL_NAME}")
    print(f"DB path      : {DB_PATH}")
    print(f"Log path     : {LOG_PATH}")
    print(f"Config valid : {validate_config()}")