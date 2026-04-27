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
    "resolve_file_dep_path",
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


def resolve_file_dep_path(stored_path: str) -> str | None:
    """Resolve a stored file dependency path, trying fallbacks if it doesn't exist.

    When a project is moved (e.g. Google Drive path changes), the absolute path
    stored in cache metadata may no longer be valid.  This function tries to
    locate the file at alternative paths:

    1. The stored path as-is.
    2. The basename resolved relative to the current working directory.
    3. Progressively longer path suffixes relative to the current working directory
       (handles subdirectory structure like ``examples/data.csv``).

    Returns the resolved path if found, or ``None`` if the file cannot be located.
    """
    if os.path.exists(stored_path):
        return stored_path

    basename = os.path.basename(stored_path)

    # Fallback 1: basename in CWD
    cwd_candidate = os.path.join(os.getcwd(), basename)
    if os.path.exists(cwd_candidate):
        return normalize_path(os.path.realpath(cwd_candidate))

    # Fallback 2: try progressively longer path suffixes relative to CWD.
    # E.g. stored = "C:/old/root/project/examples/data.csv"
    #   → try "examples/data.csv" relative to CWD
    parts = stored_path.replace('\\', '/').split('/')
    # Start from the second-to-last component (parent dir + filename)
    for i in range(max(len(parts) - 2, 1), 0, -1):
        suffix = '/'.join(parts[i:])
        candidate = os.path.join(os.getcwd(), suffix)
        if os.path.exists(candidate):
            return normalize_path(os.path.realpath(candidate))

    return None
