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

from ...utils import normalize_path
from ..server_discovery import get_notebook_path

if TYPE_CHECKING:
    from .._protocols import TrackingState
    from ._metadata import StatementCacheMetadata

logger = logging.getLogger(__name__)

# Scalar types that DON'T inherit file dependencies — a number derived
# from a DataFrame shouldn't invalidate when the source CSV changes.
_SCALAR_TYPES = (int, float, str, bool, bytes, type(None))


# ---------------------------------------------------------------------------
# Pure helpers (used by both StatementFileDeps and StatementLineageBuilder)
# ---------------------------------------------------------------------------

def compute_file_hash_component(
    accessed_files: set[str],
    accessed_remote: set[str] | None = None,
) -> str:
    """Compute a hash component from accessed file paths and their stats.

    *accessed_remote* carries URLs the statement read from object storage. They
    contribute the validator their store maintains — an ETag, a version id, a
    GCS generation — rather than a stat, because there is nothing local to stat
    (see :class:`cash.remote_source.RemoteFileDataSource`).

    Folding the token straight into the key is the right shape *here*, and
    differs from the decorator path deliberately. A statement's file state is
    already part of its lineage, so a changed object simply yields a different
    key: there is no entry to re-validate and therefore no way to serve a stale
    one. The decorator, whose key is fixed before the call runs, has to record
    the token and re-check it on a hit instead.

    Remote URLs are deliberately kept out of ``executed_file_deps``: that set is
    ``stat``-ed and ``getmtime``-d by its consumers, so a URL there contributes
    nothing at best. The key component alone is sufficient. See CAS-237.
    """
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

    for url in sorted(accessed_remote or ()):
        # The URL is the identity and the token is the state, exactly as
        # display_path/mtime/size are for a local file. Both are facts about the
        # object rather than about this machine, so unlike the local component
        # this one is identical on every machine -- a statement reading object
        # storage keys the same for a teammate (CAS-233's portability problem,
        # which for remote data simply does not arise).
        from cash.remote_source import RemoteFileDataSource

        file_components.append(f"{url}:{RemoteFileDataSource(url).state_token()}")

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

    Stateless apart from a ``debug`` flag.  All :class:`TrackingState`
    access happens through the ``tracking_state`` method parameter, which
    owns both ``executed_file_deps`` and ``executed_file_mtimes``.
    """

    def __init__(
        self,
        debug: bool = False,
    ) -> None:
        self.debug = debug

    def update_for_var(
        self,
        tracking_state: 'TrackingState',
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
        executed_file_deps = tracking_state.executed_file_deps
        executed_file_mtimes = tracking_state.executed_file_mtimes
        # 1. Direct file dependencies from this statement's execution.
        if accessed_files:
            if var_name not in executed_file_deps:
                executed_file_deps[var_name] = set()
            executed_file_deps[var_name].update(accessed_files)
            if var_name not in executed_file_mtimes:
                executed_file_mtimes[var_name] = {}
            for fpath in accessed_files:
                with contextlib.suppress(OSError):  # File may have been deleted between execution and capture
                    executed_file_mtimes[var_name][fpath] = os.path.getmtime(fpath)
        # 2. Propagate file dependencies from input variables (unless output is scalar).
        self.inherit_from_inputs(tracking_state, var_name, inputs, value)

    def inherit_from_inputs(self, tracking_state: 'TrackingState', var_name: str, inputs: set[str], value: Any) -> None:
        """Propagate file deps from *inputs* to *var_name*, skipping scalar outputs."""
        executed_file_deps = tracking_state.executed_file_deps
        executed_file_mtimes = tracking_state.executed_file_mtimes
        is_scalar = isinstance(value, _SCALAR_TYPES)
        if not is_scalar:
            for input_var in inputs:
                if input_var not in executed_file_deps:
                    continue
                if var_name not in executed_file_deps:
                    executed_file_deps[var_name] = set()
                executed_file_deps[var_name].update(executed_file_deps[input_var])
                if input_var in executed_file_mtimes:
                    if var_name not in executed_file_mtimes:
                        executed_file_mtimes[var_name] = {}
                    executed_file_mtimes[var_name].update(executed_file_mtimes[input_var])
                if self.debug:
                    logger.debug(
                        "[CACHE DEBUG] Propagated file deps from '%s' to '%s': %s",
                        input_var, var_name, executed_file_deps[input_var],
                    )
        elif self.debug and any(iv in executed_file_deps for iv in inputs):
            logger.debug(
                "[FILE_DEPS] Skipping file dep propagation for scalar '%s' (type: %s)",
                var_name, type(value).__name__,
            )

    def restore_from_metadata(
        self,
        tracking_state: 'TrackingState',
        restored_vars: dict,
        metadata: 'StatementCacheMetadata | None',
    ) -> None:
        """Propagate file deps from cached metadata back into the tracking dicts."""
        if not metadata:
            return
        file_deps = metadata.file_dependencies or {}
        if not file_deps:
            return
        from ..file_dep_snapshot import split_file_dep_value

        # ``executed_file_mtimes`` historically holds {path: float}; flatten
        # the new {'mtime': ..., 'size': ...} form back to a bare mtime here.
        mtime_map = {
            path: split_file_dep_value(stored)[0]
            for path, stored in file_deps.items()
        }
        executed_file_deps = tracking_state.executed_file_deps
        executed_file_mtimes = tracking_state.executed_file_mtimes
        for var_name in restored_vars:
            executed_file_deps.setdefault(var_name, set()).update(file_deps.keys())
            executed_file_mtimes.setdefault(var_name, {}).update(mtime_map)
