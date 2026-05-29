"""File-dependency snapshot utilities used across cache subsystems.

Pure helpers for capturing and unpacking file metadata snapshots
(``{path: {'mtime': float, 'size': int}}``). Consumed by:

- ``src/cash/core.py`` — the decorator subsystem, when recording file deps for
  a cached function call.
- :class:`cash.notebook.Restorer` (``restore.py``) — when validating that
  cached file deps still match.
- :class:`cash.notebook.upstream.VirtualLineage` — when checking file
  freshness during upstream simulation.
- :class:`cash.notebook.statement.CacheFreshnessChecker` — the post-execution
  freshness check for statement-level caching.

These helpers used to live alongside ``CacheFreshnessChecker`` in
``cache_freshness.py``. They were extracted before the ``statement/`` package
was formed (ADR-011) so callers outside the statement subsystem don't end up
reaching into ``cash.notebook.statement.freshness`` for what is really a
pure utility.
"""

from __future__ import annotations

import os
from typing import Any


def snapshot_file_deps(paths: set[str]) -> dict[str, dict[str, float]]:
    """Return ``{path: {'mtime': mtime, 'size': size}}`` for paths that exist."""
    snapshot: dict[str, dict[str, float]] = {}
    for f in paths:
        try:
            st = os.stat(f)
        except OSError:
            continue
        snapshot[f] = {'mtime': st.st_mtime, 'size': st.st_size}
    return snapshot


def split_file_dep_value(value: dict[str, Any]) -> tuple[float, int | None]:
    """Return ``(mtime, size_or_None)`` from a file-dep snapshot dict.

    Snapshots are written as ``{'mtime': float, 'size': int}``; ``size`` may be
    absent for callers that only record mtime, in which case the size check is
    skipped downstream.
    """
    return float(value.get('mtime', 0.0)), value.get('size')
