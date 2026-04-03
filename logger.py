"""
GGTH Predictor - Logger Utility
================================
Drop-in rotating file logger. Replaces print() statements across the project.

Usage:
    from logger import get_logger
    log = get_logger(__name__)

    log.debug("Detailed internal state info")
    log.info("Normal operational messages")
    log.warning("Something unexpected but recoverable")
    log.error("Something failed - still running")
    log.critical("Fatal - system cannot continue")

Log file: ggth_predictor.log (same folder as this file, auto-rotating at 5MB)
Console:  INFO and above only (keeps terminal clean during training)
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# ── Constants ────────────────────────────────────────────────────────────────

_LOG_DIR      = Path(__file__).parent          # same folder as this script
_LOG_FILE     = _LOG_DIR / "ggth_predictor.log"
_MAX_BYTES    = 5 * 1024 * 1024               # 5 MB per file
_BACKUP_COUNT = 3                              # 3 backups → 20 MB total cap

_FILE_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-35s | %(message)s"
)
_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ── Internal registry ─────────────────────────────────────────────────────────

_handlers_installed = False   # ensure root handlers are added exactly once


def _install_root_handlers(log_dir: Optional[Path] = None) -> None:
    """Configure the root logger once. Safe to call multiple times."""
    global _handlers_installed
    if _handlers_installed:
        return

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)   # handlers filter individually

    log_path = (log_dir or _LOG_DIR) / "ggth_predictor.log"

    # ── Rotating file handler (DEBUG+) ──────────────────────────────────────
    fh = RotatingFileHandler(
        log_path,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(fh)

    # ── Console handler (INFO+) ──────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(ch)

    _handlers_installed = True


# ── Public API ────────────────────────────────────────────────────────────────

def get_logger(name: str = "ggth", log_dir: Optional[str] = None) -> logging.Logger:
    """
    Return a named logger backed by the shared rotating file + console handlers.

    Args:
        name:    Logger name. Use __name__ for automatic module-level naming.
                 Common names used in this project:
                     "ggth.predictor"   → UnifiedLSTMPredictor
                     "ggth.models"      → model_builders
                     "ggth.gui"         → ggth_gui
                     "ggth.config"      → config_manager
        log_dir: Override the default log directory (useful in tests).

    Returns:
        Configured logging.Logger instance.

    Example:
        log = get_logger(__name__)
        log.info("MT5 connected successfully")
        log.error("Training failed: %s", exc)        # ← lazy formatting, preferred
        log.debug("Feature shape: %s", df.shape)
    """
    dir_path = Path(log_dir) if log_dir else None
    _install_root_handlers(dir_path)
    return logging.getLogger(name)


# ── Module-level convenience logger ─────────────────────────────────────────
# Import this directly when you don't care about the name:
#   from logger import log
#   log.info("...")

log = get_logger("ggth")


# ── Startup banner helper ─────────────────────────────────────────────────────

def log_startup_banner(version: str = "v8") -> None:
    """Write a separator banner at startup so log files are easy to scan."""
    log.info("=" * 70)
    log.info("  GGTH Predictor %s  —  session start", version)
    log.info("  Log file: %s", _LOG_FILE)
    log.info("=" * 70)


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    _l = get_logger("ggth.test")
    log_startup_banner("test")
    _l.debug("debug message   → file only")
    _l.info("info message    → file + console")
    _l.warning("warning message → file + console")
    _l.error("error message   → file + console")
    print(f"\nLog written to: {_LOG_FILE}")
