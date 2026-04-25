"""
Structured logging for Agentic Morningstar.

All modules should use get_logger(__name__) instead of print().
Logs go to both console (with colour) and a rotating file.
"""
import logging
import logging.handlers
from pathlib import Path

LOG_FILE = Path("./morningstar.log")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Map of emoji prefixes used in old print() calls → log levels
# (preserved so terminal output still looks familiar)
_LEVEL_MAP = {
    "🚀": logging.INFO,
    "✅": logging.INFO,
    "⭐": logging.INFO,
    "🗑️": logging.DEBUG,
    "⚠️": logging.WARNING,
    "❌": logging.ERROR,
}


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-level logger with console + rotating file handlers.

    Usage:
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Starting ingestion")
        logger.warning("No results found")
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(logging.DEBUG)

    # Console handler — INFO and above, human-readable
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(console)

    # Rotating file handler — DEBUG and above, up to 5MB × 3 files
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(file_handler)
    except OSError:
        pass  # Readonly filesystem or permission issue — console-only is fine

    logger.propagate = False
    return logger
