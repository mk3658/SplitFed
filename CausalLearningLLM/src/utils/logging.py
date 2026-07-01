"""Logging utilities for CRA experiments."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create (or retrieve) a named logger with console and optional file output.

    Parameters
    ----------
    name:
        Logger name, typically ``__name__`` of the calling module.
    log_file:
        Optional path to a log file.  The parent directory is created if it
        does not already exist.
    level:
        Logging level (default: ``logging.INFO``).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when the logger is requested multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    return logger


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Convenience wrapper — returns a console-only logger."""
    return setup_logger(name, log_file=None, level=level)
