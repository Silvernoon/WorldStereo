"""Logging configuration utilities."""

import logging
import sys
from typing import Optional


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> None:
    """
    Set up logging configuration.

    Args:
        level: Logging level (default: INFO).
        log_file: Optional file path to write logs to.
        format_string: Custom format string for log messages.
    """
    if format_string is None:
        format_string = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

    # Suppress verbose third-party loggers
    for logger_name in [
        "transformers",
        "diffusers",
        "huggingface_hub",
        "httpx",
        "urllib3",
        "filelock",
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
