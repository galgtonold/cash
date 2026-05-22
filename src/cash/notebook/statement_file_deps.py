"""Statement-level file dependency tracking.

Owns the operation "what files did this statement read or inherit, and
how do we hash / propagate / restore those dependencies?"

Two layers:

* :class:`StatementFileDeps` — stateful manager that mutates
  ``executed_file_deps`` and ``executed_file_mtimes`` dicts.  Propagates
  direct file accesses + inherits file deps from input variables.
* Module-level pure helpers ``compute_file_hash_component`` and
  ``read_module_source_hash`` — used both here and by the
  upcoming ``StatementLineageBuilder`` (step B1).

**Anti-god-class rule (load-bearing):** this module owns
**statement-level** file deps (which files contributed to producing
this variable, transitively through its inputs).  It does **not**
intercept file reads at runtime — that is :class:`FileAccessTracker`
in ``file_tracker.py``, a different layer (the import-hook that wraps
``pd.read_csv`` / ``open`` etc.).  Don't conflate them.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
from typing import TYPE_CHECKING, Any

from ..utils import normalize_path
from .server_discovery import get_notebook_path

if TYPE_CHECKING:
    from ._protocols import TrackingState
    from .statement_processor import StatementCacheMetadata

logger = logging.getLogger(__name__)

# Scalar types that DON'T inherit file dependencies — a number derived
# from a DataFrame shouldn't invalidate when the source CSV changes.
_SCALAR_TYPES = (int, float, str, bool, bytes, type(None))


# ---------------------------------------------------------------------------
# Pure helpers (used by both StatementFileDeps and StatementLineageBuilder)
# ---------------------------------------------------------------------------

def compute_file_hash_component(accessed_files: set[str]) -> str:
    """Compute a hash component from accessed file paths and their stats."""
    notebook_dir = None
    try:
        notebook_path = get_notebook_path()
        if notebook_path:
            notebook_dir = os.path.dirname(os.path.realpath(notebook_path))
    except (OSError, ValueError):
        logger.debug("[PROCESSOR] Failed to get notebook directory for file hash")

    file_components = []
    for f in sorted(accessed_files):
        if os.path.exists(f):
            canonical_path = normalize_path(os.path.realpath(f))
            display_path = canonical_path
            if notebook_dir:
                try:
                    rel_path = normalize_path(os.path.relpath(canonical_path, notebook_dir))
                    if not rel_path.startswith('../../../'):
                        display_path = rel_path
                except (ValueError, OSError):
                    pass  # Cross-drive relpath fails on Windows; fall back to absolute
            try:
                stat = os.stat(canonical_path)
                file_components.append(f"{display_path}:{stat.st_mtime}:{stat.st_size}")
            except OSError:
                pass  # File may have been removed between exists() and stat()
    if file_components:
        component = ":" + hashlib.sha256(",".join(file_components).encode('utf-8')).hexdigest()
        logger.debug("[FILE_HASH] Final hash component: %s...", component[:50])
        return component
    return ""


def read_module_source_hash(mod_file: str, dep_files: set[str] | None = None) -> str | None:
    """Read a module file (and optional dependency files) and return their combined SHA256 hash."""
    try:
        hasher = hashlib.sha256()
        with open(mod_file, 'rb') as mf:
            hasher.update(mf.read())
        if dep_files:
            for dep_path in sorted(dep_files):
                if os.path.isfile(dep_path):
                    try:
                        with open(dep_path, 'rb') as df:
                            hasher.update(df.read())
                    except OSError:
                        logger.debug("[MODULE_HASH] Could not read dependency file: %s", dep_path)
        return hasher.hexdigest()
    except OSError:
        logger.debug("[MODULE_HASH] Could not read module file: %s", mod_file)
        return None


# ---------------------------------------------------------------------------
# StatementFileDeps
# ---------------------------------------------------------------------------

class StatementFileDeps:
    """Stateful per-statement file-dependency tracker.

    Holds aliased dict references to ``executed_file_deps`` (from
    :class:`TrackingState`) and ``executed_file_mtimes`` (owned by
    ``StatementProcessor``).  Mutations are visible to all sharers via
    the aliased references.
    """

    def __init__(
        self,
        tracking_state: 'TrackingState',
        executed_file_mtimes: dict[str, dict[str, float]],
        debug: bool = False,
    ) -> None:
        self.executed_file_mtimes = executed_file_mtimes
        self.debug = debug
        self.set_tracking_state(tracking_state)

    def set_tracking_state(self, state: 'TrackingState') -> None:
        """Re-wire ``executed_file_deps`` from *state* (the TrackingState dict)."""
        self.executed_file_deps = state.executed_file_deps

    def update_for_var(
        self,
        var_name: str,
        accessed_files: set[str] | None,
        inputs: set[str],
        value: Any,
    ) -> None:
        """Record direct and inherited file dependencies for *var_name*.

        Two sources of file deps are handled here so that the logic is not
        duplicated:

        1. **Direct deps** — files that were read during the current
           statement's execution (``accessed_files``).
        2. **Inherited deps** — file deps already carried by input variables
           (e.g. ``df = df.sort_values()`` inherits ``df``'s source file so
           downstream cells still invalidate when that file changes).
           Scalar outputs are excluded from inheritance because a scalar
           derived from a DataFrame (``n_rows = len(df)``) should not be
           invalidated when the source CSV changes.
        """
        # 1. Direct file dependencies from this statement's execution.
        if accessed_files:
            if var_name not in self.executed_file_deps:
                self.executed_file_deps[var_name] = set()
            self.executed_file_deps[var_name].update(accessed_files)
            if var_name not in self.executed_file_mtimes:
                self.executed_file_mtimes[var_name] = {}
            for fpath in accessed_files:
                with contextlib.suppress(OSError):  # File may have been deleted between execution and capture
                    self.executed_file_mtimes[var_name][fpath] = os.path.getmtime(fpath)
        # 2. Propagate file dependencies from input variables (unless output is scalar).
        self.inherit_from_inputs(var_name, inputs, value)

    def inherit_from_inputs(self, var_name: str, inputs: set[str], value: Any) -> None:
        """Propagate file deps from *inputs* to *var_name*, skipping scalar outputs."""
        is_scalar = isinstance(value, _SCALAR_TYPES)
        if not is_scalar:
            for input_var in inputs:
                if input_var not in self.executed_file_deps:
                    continue
                if var_name not in self.executed_file_deps:
                    self.executed_file_deps[var_name] = set()
                self.executed_file_deps[var_name].update(self.executed_file_deps[input_var])
                if input_var in self.executed_file_mtimes:
                    if var_name not in self.executed_file_mtimes:
                        self.executed_file_mtimes[var_name] = {}
                    self.executed_file_mtimes[var_name].update(self.executed_file_mtimes[input_var])
                if self.debug:
                    logger.debug(
                        "[CACHE DEBUG] Propagated file deps from '%s' to '%s': %s",
                        input_var, var_name, self.executed_file_deps[input_var],
                    )
        elif self.debug and any(iv in self.executed_file_deps for iv in inputs):
            logger.debug(
                "[FILE_DEPS] Skipping file dep propagation for scalar '%s' (type: %s)",
                var_name, type(value).__name__,
            )

    def restore_from_metadata(
        self,
        restored_vars: dict,
        metadata: 'StatementCacheMetadata | None',
    ) -> None:
        """Propagate file deps from cached metadata back into the tracking dicts."""
        if not metadata:
            return
        file_deps = metadata.get('file_dependencies', {})
        if not file_deps:
            return
        # Local import to avoid a circular dep with cache_freshness — both
        # this module and cache_freshness need split_file_dep_value, but
        # cache_freshness imports nothing from us.
        from .cache_freshness import split_file_dep_value

        # ``executed_file_mtimes`` historically holds {path: float}; flatten
        # the new {'mtime': ..., 'size': ...} form back to a bare mtime here.
        mtime_map = {
            path: split_file_dep_value(stored)[0]
            for path, stored in file_deps.items()
        }
        for var_name in restored_vars:
            self.executed_file_deps.setdefault(var_name, set()).update(file_deps.keys())
            self.executed_file_mtimes.setdefault(var_name, {}).update(mtime_map)
