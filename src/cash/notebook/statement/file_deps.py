"""Statement-level file dependency tracking.

Owns the operation "what files did this statement read or inherit, and
how do we hash / propagate / restore those dependencies?"

Two layers:

* :class:`StatementFileDeps` — stateful manager that mutates the
  ``executed_file_deps`` dict.  Propagates
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

import hashlib
import logging
import os
from typing import TYPE_CHECKING, Any

from ..._paths import normalize_path
from ...remote_source import RemoteFileDataSource
from ...source_norm import module_identity
from ...tracking.file_dep_snapshot import realpath_of_read_this_run
from ...value_types import IMMUTABLE_PRIMS
from ..server_discovery import get_notebook_path

if TYPE_CHECKING:
    from .._protocols import TrackingState
    from ._metadata import StatementCacheMetadata

logger = logging.getLogger(__name__)

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
    # Relative to the notebook, per directory: a folder's files share one.
    rel_dirs: dict[str, str | None] = {}
    for f in sorted(accessed_files):
        try:
            # Through the directory, with the lstat that shows the file is not
            # a link as its stat: a statement over a frame read from 5,000
            # files resolved every path again per lineage (round 25, r25s4).
            resolved, stat = realpath_of_read_this_run(f)
            canonical_path = normalize_path(resolved)
            if stat is None:
                stat = os.stat(canonical_path)
        except (OSError, ValueError):
            continue  # gone, or never a file
        display_path = canonical_path
        if notebook_dir:
            head, _, name = canonical_path.rpartition("/")
            if head not in rel_dirs:
                try:
                    # A drive or filesystem root keeps its separator: `C:` alone
                    # means the current directory on C:.
                    root = head + "/" if (not head or head.endswith(":")) else head
                    rel_dirs[head] = normalize_path(os.path.relpath(root, notebook_dir))
                except (ValueError, OSError):
                    rel_dirs[head] = None  # Cross-drive relpath fails on Windows; keep absolute
            rel_dir = rel_dirs[head]
            if rel_dir is not None:
                rel_path = name if rel_dir == "." else f"{rel_dir}/{name}"
                if not rel_path.startswith("../../../"):
                    display_path = rel_path
        file_components.append(f"{display_path}:{stat.st_mtime}:{stat.st_size}")

    for url in sorted(accessed_remote or ()):
        # The URL is the identity and the token is the state, exactly as
        # display_path/mtime/size are for a local file. Both are facts about the
        # object rather than about this machine, so unlike the local component
        # this one is identical on every machine -- a statement reading object
        # storage keys the same for a teammate (CAS-233's portability problem,
        # which for remote data simply does not arise).
        file_components.append(f"{url}:{RemoteFileDataSource(url).state_token()}")

    if file_components:
        component = ":" + hashlib.sha256(",".join(file_components).encode("utf-8")).hexdigest()
        logger.debug("[FILE_HASH] Final hash component: %s...", component[:50])
        return component
    return ""


def read_module_source_hash(mod_file: str, dep_files: set[str] | None = None) -> str | None:
    """Combined identity hash of a module file and its dependency files.

    See `cash.source_norm.module_identity` for what "identity" covers and why
    it is not the file's bytes.
    """
    own = module_identity(mod_file)
    if own is None:
        logger.debug("[MODULE_HASH] Could not read module file: %s", mod_file)
        return None
    if not dep_files:
        return own
    hasher = hashlib.sha256()
    hasher.update(own.encode("utf-8"))
    for dep_path in sorted(dep_files):
        dep = module_identity(dep_path)
        if dep is None:
            logger.debug("[MODULE_HASH] Could not read dependency file: %s", dep_path)
            continue
        hasher.update(dep.encode("utf-8"))
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# StatementFileDeps
# ---------------------------------------------------------------------------


class StatementFileDeps:
    """Stateful per-statement file-dependency tracker.

    Stateless apart from a ``debug`` flag.  All :class:`TrackingState`
    access happens through the ``tracking_state`` method parameter, which
    owns ``executed_file_deps``.
    """

    def __init__(
        self,
        debug: bool = False,
    ) -> None:
        self.debug = debug

    def update_for_var(
        self,
        tracking_state: "TrackingState",
        var_name: str,
        accessed_files: set[str] | None,
        inputs: set[str],
        value: Any,
        rebind: bool = False,
    ) -> None:
        """Record direct and inherited file dependencies for *var_name*.

        *rebind* -- the statement bound a fresh value to the name (it is not
        also one of the statement's inputs): the files the OLD value came from
        are forgotten. Merged instead, ``for f in files: d = pd.read_csv(f)``
        left ``d`` depending on every file read so far, and each iteration's
        save snapshotted all of them -- quadratic in the files (round 23, r23s2:
        865,265 hashes in one cell). An in-place change keeps what it had.

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
        if rebind:
            executed_file_deps.pop(var_name, None)
        # 1. Direct file dependencies from this statement's execution.
        if accessed_files:
            if var_name not in executed_file_deps:
                executed_file_deps[var_name] = set()
            executed_file_deps[var_name].update(accessed_files)
        # 2. Propagate file dependencies from input variables (unless output is scalar).
        self.inherit_from_inputs(tracking_state, var_name, inputs, value)

    def inherit_from_inputs(self, tracking_state: "TrackingState", var_name: str, inputs: set[str], value: Any) -> None:
        """Propagate file deps from *inputs* to *var_name*, skipping scalar outputs."""
        executed_file_deps = tracking_state.executed_file_deps
        # A scalar does not inherit file dependencies: a number derived from a
        # DataFrame shouldn't invalidate when the source CSV changes.
        is_scalar = isinstance(value, IMMUTABLE_PRIMS)
        if not is_scalar:
            for input_var in inputs:
                if input_var not in executed_file_deps:
                    continue
                if var_name not in executed_file_deps:
                    executed_file_deps[var_name] = set()
                executed_file_deps[var_name].update(executed_file_deps[input_var])
                if self.debug:
                    logger.debug(
                        "[CACHE DEBUG] Propagated file deps from '%s' to '%s': %s",
                        input_var,
                        var_name,
                        executed_file_deps[input_var],
                    )
        elif self.debug and any(iv in executed_file_deps for iv in inputs):
            logger.debug(
                "[FILE_DEPS] Skipping file dep propagation for scalar '%s' (type: %s)",
                var_name,
                type(value).__name__,
            )

    def restore_from_metadata(
        self,
        tracking_state: "TrackingState",
        restored_vars: dict,
        metadata: "StatementCacheMetadata | None",
    ) -> None:
        """Propagate file deps from cached metadata back into the tracking dict."""
        if not metadata:
            return
        file_deps = metadata.file_dependencies or {}
        if not file_deps:
            return
        executed_file_deps = tracking_state.executed_file_deps
        # A restored value is exactly the entry's value: its files are the
        # entry's, not those of whatever the name held before.
        for var_name in restored_vars:
            executed_file_deps[var_name] = set(file_deps.keys())
