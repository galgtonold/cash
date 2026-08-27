"""Canonical loop-split derivation, shared by the simulator and the runtime.

A loop that neither caching mechanism covers -- too few iterations for the
single-unit heuristic, too cheap per call for ``call_unit`` -- caches nothing
while still paying per-iteration decomposition overhead on every pass. At
n=124 that made cash SLOWER than not using cash (0.1ms body: 22ms off vs
215ms on). Such a loop is learned on one run and thereafter executed as two
statements: a short decomposed head and its remainder as one unit.

**The simulator is what makes that happen.** This is the single most
important fact about this module, and three reverted attempts came from not
knowing it. ``upstream/`` does not merely predict what the runtime will do
for metrics or key parity -- the re-execution planner executes *the
statements the simulator modelled*. Split the simulator's model and the
runtime follows; split only the runtime and the planner re-runs the whole
loop against entries written for halves, which is a silent stale value:

    result = np.zeros(N)          # N edited 10 -> 20 upstream
    for i in range(100):
        result = result + 1
    total = result.sum()

    split applied on BOTH sides   -> total = 2000  (correct)
    split applied at runtime only -> total = 1000  (STALE)

Hence: one derivation, here, imported by both sides, pinned by a
derivation test. Two properties are load-bearing:

1. **The tail's source is a pure function of** ``(loop node, k)``. It slices
   the ORIGINAL iterable expression rather than binding materialised items to
   a temp name -- a content digest of live data would be uncomputable by the
   simulator, which has source and no values.
2. ``k`` **is persisted, not re-measured.** Only the runtime can time a loop;
   the simulator has no clock. Timing jitter would otherwise move the split
   between runs, changing the tail's source and so its key.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import logging
import os

from cash.utils import replace_with_retry

logger = logging.getLogger(__name__)

_STORE_FILENAME = "_loop_split.json"
_STORE_VERSION = 1


def loop_source_hash(node: ast.AST) -> str:
    """Identity of a loop for split purposes: sha256 of its unparsed source.

    ``ast.unparse`` rather than raw cell text, so formatting and comments
    cannot change a loop's identity -- and so the runtime (holding an AST
    node) and the simulator (parsing cell source) agree without either
    needing the other's representation.
    """
    return hashlib.sha256(ast.unparse(node).encode("utf-8")).hexdigest()


def is_split_half(node: ast.AST) -> bool:
    """Whether *node* is already a half produced by :func:`split_nodes`.

    Guards against re-splitting a half, which would recurse. Structural
    rather than a marker attribute, because the simulator re-parses source
    and would lose any attribute set on the runtime's node.
    """
    return (isinstance(node, ast.For)
            and isinstance(node.iter, ast.Subscript)
            and isinstance(node.iter.slice, ast.Slice))


def split_nodes(node: ast.For, k: int) -> tuple[ast.For, ast.For]:
    """Return ``(head, tail)`` for-nodes covering ``[:k]`` and ``[k:]``.

    Together they iterate exactly what *node* iterates, in the same order,
    provided the iterable is sliceable and the header is safe to evaluate
    twice -- the caller's responsibility to have checked.

    ``orelse`` must be empty: a ``for ... else`` has one completion point and
    a split loop has none, so there is nowhere correct to put it.
    """
    if node.orelse:
        raise ValueError("cannot split a for/else loop")

    def _half(lower: int | None, upper: int | None) -> ast.For:
        half = ast.For(
            target=node.target,
            iter=ast.Subscript(
                value=node.iter,
                slice=ast.Slice(
                    lower=None if lower is None else ast.Constant(value=lower),
                    upper=None if upper is None else ast.Constant(value=upper),
                ),
                ctx=ast.Load(),
            ),
            body=list(node.body),
            orelse=[],
            type_comment=None,
        )
        return ast.fix_missing_locations(ast.copy_location(half, node))

    return _half(None, k), _half(k, None)


def split_sources(node: ast.For, k: int) -> tuple[str, str]:
    """:func:`split_nodes` unparsed -- what the simulator keys.

    Routing both sides through one function is what makes them identical.
    """
    head, tail = split_nodes(node, k)
    return ast.unparse(head), ast.unparse(tail)


class LoopSplitStore:
    """Persisted ``source_hash -> k`` verdicts, read by both sides.

    Mirrors ``statement/miss_guard.py``: loaded lazily once per session,
    written only when a verdict is added, atomic via ``os.replace``, and
    best-effort throughout -- a missing, unreadable, corrupt or
    future-versioned store leaves it empty, which means "no loop is split",
    which is exactly the pre-split behaviour. The failure mode must be "no
    optimisation", never "wrong answer".
    """

    def __init__(self, cache_dir: str | None) -> None:
        self._path = os.path.join(cache_dir, _STORE_FILENAME) if cache_dir else None
        self._splits: dict[str, int] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._path:
            return
        try:
            with open(self._path, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            logger.debug("[LOOP_SPLIT] no readable store at %s", self._path)
            return
        if not isinstance(doc, dict) or doc.get("version") != _STORE_VERSION:
            return
        splits = doc.get("splits")
        if not isinstance(splits, dict):
            return
        for source_hash, k in splits.items():
            if isinstance(source_hash, str) and isinstance(k, int) and k > 0:
                self._splits[source_hash] = k

    def get(self, source_hash: str) -> int | None:
        """The persisted ``k`` for this loop, or ``None`` if it is not split."""
        self._ensure_loaded()
        return self._splits.get(source_hash)

    def record(self, source_hash: str, k: int) -> None:
        """Persist a split verdict. No-op if one already exists.

        Never rewrites: a ``k`` that moved between runs would change the
        tail's source and therefore its key, which is the failure this store
        exists to prevent.
        """
        self._ensure_loaded()
        if source_hash in self._splits:
            return
        self._splits[source_hash] = k
        self._persist()

    def _persist(self) -> None:
        if not self._path:
            return
        doc = {"version": _STORE_VERSION, "splits": dict(sorted(self._splits.items()))}
        tmp_path = f"{self._path}.{os.getpid()}.tmp"
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(doc, fh)
            # Not a bare os.replace: on Windows the call is DENIED, not
            # delayed, while any handle has the destination open -- and the
            # except below swallows that at debug level, so the verdict
            # vanished from disk while staying in memory. The next session
            # then loads a store without it, does not split the loop, and
            # keys the tail differently: a cache miss and a real recompute,
            # which is precisely what this store exists to prevent. Measured
            # susceptible with a single reader handle held open (#74).
            replace_with_retry(tmp_path, self._path)
        except OSError:
            logger.debug("[LOOP_SPLIT] could not persist to %s", self._path)
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)


# One store per cache dir, process-wide.
#
# NOT an optimisation -- a correctness requirement. The store loads from disk
# once per session, so two independent instances diverge the moment a verdict
# is recorded: the runtime's holds it in memory while the simulator's, built
# earlier and already marked loaded, never sees it. The runtime would then be
# recording a split the simulator does not apply. Sharing the instance makes
# "recorded" mean the same thing on both sides at the same instant.
_STORES: dict[str | None, LoopSplitStore] = {}


def get_store(cache_dir: str | None) -> LoopSplitStore:
    """The shared :class:`LoopSplitStore` for *cache_dir*."""
    store = _STORES.get(cache_dir)
    if store is None:
        store = LoopSplitStore(cache_dir)
        _STORES[cache_dir] = store
    return store


def store_for_backend(backend) -> LoopSplitStore | None:
    """Shared store for *backend*'s cache dir, or ``None`` if unresolvable.

    The one place both sides resolve a store, so they cannot disagree about
    which directory they are reading. Returns ``None`` rather than raising:
    an unresolvable store means "no loop is split".
    """
    try:
        from .statement.miss_guard import resolve_cache_dir
        return get_store(resolve_cache_dir(backend))
    except Exception:  # noqa: BLE001 - splitting is an optimisation
        logger.debug("[LOOP_SPLIT] could not resolve a store", exc_info=True)
        return None


def _reset_stores_for_tests() -> None:
    """Drop cached stores. Tests only -- each tmp_path is a fresh session."""
    _STORES.clear()
