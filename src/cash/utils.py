"""Shared utility functions for the cash library.

``normalize_path`` — pure path helper used throughout the codebase.

Notebook-specific I/O and Jupyter Server HTTP discovery
(``get_notebook_path``, ``get_notebook_cells``, etc.) live in
``cash.notebook.server_discovery``.  For backward compatibility these names
are still accessible here, but new code should import them from
``cash.notebook.server_discovery`` directly.
"""

from __future__ import annotations

import logging
import os

from cash.notebook.server_discovery import (  # noqa: F401  (re-exports)
    extract_notebook_path_from_vscode_cell_id,
    get_notebook_cells,
    get_notebook_cells_with_ids,
    get_notebook_path,
    invalidate_notebook_path_cache,
    set_notebook_path,
)

logger = logging.getLogger(__name__)

__all__ = [
    "normalize_path",
    "invalidate_notebook_path_cache",
    "set_notebook_path",
    "extract_notebook_path_from_vscode_cell_id",
    "get_notebook_path",
    "get_notebook_cells",
    "get_notebook_cells_with_ids",
]


def normalize_path(path: str) -> str:
    """Return *path* with all OS-native separators replaced by forward slashes.

    Used to produce portable, platform-independent path strings for cache keys
    and dependency tracking.  On POSIX systems this is a no-op; on Windows it
    converts backslashes to forward slashes.

    Examples::

        normalize_path("C:\\\\Users\\\\foo\\\\bar.csv")  # → "C:/Users/foo/bar.csv"
        normalize_path("/home/foo/bar.csv")              # → "/home/foo/bar.csv"
    """
    return path.replace(os.path.sep, '/')
