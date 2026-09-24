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

from cash.backends.cache_dir import LOOP_SPLIT_FILENAME

from .cache_key import statement_source_hash
from .versioned_json_store import StoreRegistry, VersionedJsonStore

_STORE_FILENAME = LOOP_SPLIT_FILENAME
_STORE_VERSION = 1


def loop_source_hash(node: ast.AST) -> str:
    """Identity of a loop for split purposes: the statement source hash
    (``statement_source_hash``) of its unparsed source.

    ``ast.unparse`` rather than raw cell text, so formatting and comments
    cannot change a loop's identity -- and so the runtime (holding an AST
    node) and the simulator (parsing cell source) agree without either
    needing the other's representation.
    """
    return statement_source_hash(ast.unparse(node))


def is_split_half(node: ast.AST) -> bool:
    """Whether *node* is already a half produced by :func:`split_nodes`.

    Guards against re-splitting a half, which would recurse. Structural
    rather than a marker attribute, because the simulator re-parses source
    and would lose any attribute set on the runtime's node.
    """
    return isinstance(node, ast.For) and isinstance(node.iter, ast.Subscript) and isinstance(node.iter.slice, ast.Slice)


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


class LoopSplitStore(VersionedJsonStore[int]):
    """Persisted ``source_hash -> k`` verdicts, read by both sides.

    Loaded lazily once per session, written only when a verdict is added,
    and best-effort throughout (see :mod:`.versioned_json_store`) -- a missing, unreadable, corrupt or
    future-versioned store leaves it empty, which means "no loop is split",
    which is exactly the pre-split behaviour. The failure mode must be "no
    optimisation", never "wrong answer".
    """

    FILENAME = _STORE_FILENAME
    VERSION = _STORE_VERSION
    FIELD = "splits"
    LOG_TAG = "LOOP_SPLIT"

    def _load_value(self, value: object) -> int | None:
        return value if isinstance(value, int) and value > 0 else None

    def get(self, source_hash: str) -> int | None:
        """The persisted ``k`` for this loop, or ``None`` if it is not split."""
        self._ensure_loaded()
        return self._items.get(source_hash)

    def record(self, source_hash: str, k: int) -> None:
        """Persist a split verdict. No-op if one already exists.

        Never rewrites: a ``k`` that moved between runs would change the
        tail's source and therefore its key, which is the failure this store
        exists to prevent. A verdict that cannot be written stays in memory
        for this session; the next one does not split the loop.
        """
        self._ensure_loaded()
        if source_hash in self._items:
            return
        self._items[source_hash] = k
        self._write()


# One store per cache dir, process-wide.
#
# NOT an optimisation -- a correctness requirement. The store loads from disk
# once per session, so two independent instances diverge the moment a verdict
# is recorded: the runtime's holds it in memory while the simulator's, built
# earlier and already marked loaded, never sees it. The runtime would then be
# recording a split the simulator does not apply. Sharing the instance makes
# "recorded" mean the same thing on both sides at the same instant.
_STORES: StoreRegistry[LoopSplitStore] = StoreRegistry(LoopSplitStore)


def get_store(cache_dir: str | None) -> LoopSplitStore:
    """The shared :class:`LoopSplitStore` for *cache_dir*."""
    return _STORES.get(cache_dir)


def store_for_backend(backend) -> LoopSplitStore | None:
    """Shared store for *backend*'s cache dir, or ``None`` if unresolvable.

    The one place both sides resolve a store, so they cannot disagree about
    which directory they are reading. ``None`` means "no loop is split".
    """
    return _STORES.for_backend(backend)


def _reset_stores_for_tests() -> None:
    """Drop cached stores. Tests only -- each tmp_path is a fresh session."""
    _STORES.reset()
