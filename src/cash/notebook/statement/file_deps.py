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

import ast
import contextlib
import hashlib
import logging
import os
from typing import TYPE_CHECKING, Any

from ...source_norm import drop_docstrings, stat_has_settled
from ...utils import normalize_path
from ..file_dep_snapshot import realpath_of_read_this_run
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
    stats_out: dict[str, os.stat_result] | None = None,
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

    *stats_out*, when given, receives each local file's stat, so the caller
    need not stat the same files again (a folder of 5,030 files was stat-ed
    three times after the read, round 25).
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
            head, _, name = canonical_path.rpartition('/')
            if head not in rel_dirs:
                try:
                    # A drive or filesystem root keeps its separator: `C:` alone
                    # means the current directory on C:.
                    root = head + '/' if (not head or head.endswith(':')) else head
                    rel_dirs[head] = normalize_path(os.path.relpath(root, notebook_dir))
                except (ValueError, OSError):
                    rel_dirs[head] = None  # Cross-drive relpath fails on Windows; keep absolute
            rel_dir = rel_dirs[head]
            if rel_dir is not None:
                rel_path = name if rel_dir == '.' else f"{rel_dir}/{name}"
                if not rel_path.startswith('../../../'):
                    display_path = rel_path
        if stats_out is not None:
            stats_out[f] = stat
        file_components.append(f"{display_path}:{stat.st_mtime}:{stat.st_size}")

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


#: ``{path: (mtime_ns, size, identity_digest)}``. One entry per file, replaced
#: when it moves. The identity below parses the file, which is far too much to
#: repeat per statement -- and even the plain read it replaces was one file
#: read per statement per module.
_IDENTITY_CACHE: dict[str, tuple[int, int, str]] = {}


def _module_identity(raw: bytes) -> bytes:
    """What a module file says, with what it merely looks like removed.

    The digest of this lands in the lineage of every name bound from the
    module and in the key of every statement that reads one, so anything it
    covers re-runs work when it moves. Hashing the FILE meant a comment, a
    blank line or a reformat re-ran everything built on the module: measured
    2026-09-21, adding one comment to a module re-executed a 1.2 s call that
    used a function the edit did not touch. Round 27 r27s2 reported the same
    thing at scale -- editing one helper re-read all 10,000 of their ticket
    files, 48.7 s against a 17.3 s control, later 9.1x.

    So: the module re-rendered from its AST, which drops comments and
    normalises formatting, and carries no line or column numbers -- without
    that last part, inserting a comment at the top would still move every
    node below it and nothing would be gained.

    ``ast.unparse`` rather than ``ast.dump``: both drop what we want dropped,
    but ``dump`` prints the AST's own field names, so a Python release that
    adds a field moves every module's digest. Generated source moves less.
    Neither is a promise across versions, and cache keys are already not
    portable across machines (see docs/how-it-works/cache-keys-and-lineage.md),
    but there is no reason to add a reason.

    Docstrings go too, the module's own included (``strip_docstrings``): they
    are prose, the same as comments.

    And ``@cash:`` directives stay, because they are instructions TO cash --
    ``# @cash:assume-safe`` on a line waives a purity check, and cash's own
    diagnostic says "@cash: directives are part of its source identity".

    Each is kept as its WHOLE line, in source order, which anchors it to the
    code it annotates: moving ``# @cash:assume-safe`` from one function to
    another moves the digest, where keeping only the directive text would
    have made that invisible -- the unsafe direction. The cost is that
    reformatting a line that carries a directive still re-runs work. That is
    a narrow class and it errs toward invalidating.

    Matched with ``annotations.ANNOTATION_PATTERN`` itself, so the set of
    directives that counts here cannot drift from the set cash parses.

    Falls back to the raw bytes for anything that will not decode or parse --
    a data file among the dependencies, a module written for a different
    Python. That is exactly the previous behaviour, so nothing that works
    today can be made worse by this.
    """
    try:
        text = raw.decode('utf-8')
        tree = ast.parse(text)
        drop_docstrings(tree, module=True)
        rendered = ast.unparse(tree)
    except (UnicodeDecodeError, SyntaxError, ValueError, AttributeError, RecursionError):
        return raw
    from ..annotations import ANNOTATION_PATTERN
    parts = [rendered]
    parts.extend(
        line.strip() for line in text.splitlines()
        if ANNOTATION_PATTERN.search(line)
    )
    return "\n".join(parts).encode('utf-8')


def _identity_digest(path: str) -> str | None:
    """:func:`_module_identity` of *path*, memoised on its stat."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    cached = _IDENTITY_CACHE.get(path)
    if cached is not None and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
        return cached[2]
    settled = stat_has_settled(st)
    try:
        with open(path, 'rb') as fh:
            raw = fh.read()
    except OSError:
        return None
    # Keyed on the same signal `FunctionTracker.check_tracked_modules` uses to
    # notice a module changed at all, so a change this memo would miss is one
    # cash would not have reloaded for either -- once the file has settled,
    # since a same-size save inside one mtime tick keeps that stat too.
    digest = hashlib.sha256(_module_identity(raw)).hexdigest()
    if settled:
        _IDENTITY_CACHE[path] = (st.st_mtime_ns, st.st_size, digest)
    return digest


def read_module_source_hash(mod_file: str, dep_files: set[str] | None = None) -> str | None:
    """Combined identity hash of a module file and its dependency files.

    See :func:`_module_identity` for what "identity" covers and why it is not
    the file's bytes.
    """
    own = _identity_digest(mod_file)
    if own is None:
        logger.debug("[MODULE_HASH] Could not read module file: %s", mod_file)
        return None
    if not dep_files:
        return own
    hasher = hashlib.sha256()
    hasher.update(own.encode('utf-8'))
    for dep_path in sorted(dep_files):
        dep = _identity_digest(dep_path)
        if dep is None:
            logger.debug("[MODULE_HASH] Could not read dependency file: %s", dep_path)
            continue
        hasher.update(dep.encode('utf-8'))
    return hasher.hexdigest()


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
        rebind: bool = False,
        stats: dict[str, os.stat_result] | None = None,
    ) -> None:
        """Record direct and inherited file dependencies for *var_name*.

        *stats* are the files' stats already taken for the lineage component
        (``compute_file_hash_component``'s *stats_out*), reused for the mtimes.

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
        executed_file_mtimes = tracking_state.executed_file_mtimes
        if rebind:
            executed_file_deps.pop(var_name, None)
            executed_file_mtimes.pop(var_name, None)
        # 1. Direct file dependencies from this statement's execution.
        if accessed_files:
            if var_name not in executed_file_deps:
                executed_file_deps[var_name] = set()
            executed_file_deps[var_name].update(accessed_files)
            if var_name not in executed_file_mtimes:
                executed_file_mtimes[var_name] = {}
            for fpath in accessed_files:
                known = stats.get(fpath) if stats else None
                if known is not None:
                    executed_file_mtimes[var_name][fpath] = known.st_mtime
                    continue
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
        # A restored value is exactly the entry's value: its files are the
        # entry's, not those of whatever the name held before.
        for var_name in restored_vars:
            executed_file_deps[var_name] = set(file_deps.keys())
            executed_file_mtimes[var_name] = dict(mtime_map)
