"""File system utilities."""

import os
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists.

    Returns:
        Path object for the directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
