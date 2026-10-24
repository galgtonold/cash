"""Loops whose iterations cost enough to be worth reusing one by one.

The single-unit rule (:mod:`.control_structures.single_unit_policy`) sends a
long loop to one cache entry because per-iteration machinery costs about 8 ms
per body statement, a guess made without seeing the loop run. For a loop whose
iterations each take far longer than that, the guess is wrong in the other
direction: one entry is all-or-nothing, so extending ``range(100)`` or editing
the last statement of the body re-runs every iteration, where per-iteration
entries would serve the unchanged work.

This store keeps what a loop's iterations MEASURED, and the handler reads it
back on later runs. Like the loop split it only ever picks between two
execution modes that are both correct, so a missing, stale or wrong entry costs
time, never a value.

A loop is remembered by its HEADER (target and iterable, with numbers blanked),
not its source: the point is to carry the verdict over the edits that follow.
Under one header each measurement keeps the body it was taken on, and a loop
reads the one whose body shares at least half its statements (numbers blanked
too): an edit to part of the body finds its loop, and a different loop with
the same header (``for k in range(130)`` around a slow call, and a cheap
``for k in range(3000)``) has its own. Shared, the cheap loop ran pass by pass
on the slow one's measurement, and its own measurement then sent the slow one
back to one unit.
"""

from __future__ import annotations

import ast

from cash.analysis.ast_util import copy_tree
from cash.backends.cache_dir import HEAVY_LOOPS_FILENAME

from .cache_key import statement_source_hash
from .control_structures.single_unit_policy import PER_STMT_OVERHEAD_SEC
from .versioned_json_store import StoreRegistry, VersionedJsonStore

_STORE_FILENAME = HEAVY_LOOPS_FILENAME
_STORE_VERSION = 2

#: Measurements kept under one header, the latest last.
_PER_HEADER = 8

#: An iteration is heavy when its work is at least this many times what
#: decomposing it costs (``PER_STMT_OVERHEAD_SEC`` per body statement). At 4 the
#: first, cold run pays at most about a quarter more than running it whole.
HEAVY_FACTOR = 4

#: ...and never below this, whatever the body's size.
MIN_HEAVY_ITER_SEC = 0.01


class _BlankNumbers(ast.NodeTransformer):
    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return ast.copy_location(ast.Constant(value=0), node)
        return node


def header_identity(node: ast.For) -> str:
    """Identity of a loop's header: target and iterable, numbers blanked, so
    ``range(100)`` and ``range(180)`` are the same loop."""
    header = _BlankNumbers().visit(copy_tree(ast.Module(body=[ast.Expr(node.iter)], type_ignores=[])))
    return statement_source_hash(f"{ast.unparse(node.target)} in {ast.unparse(header)}")


def body_identity(node: ast.For) -> tuple[str, ...]:
    """Each statement of a loop's body, numbers blanked: which measurement
    under its header is this loop's (`HeavyLoopStore`)."""
    return tuple(
        statement_source_hash(ast.unparse(_BlankNumbers().visit(copy_tree(ast.Module(body=[stmt], type_ignores=[])))))
        for stmt in node.body
    )


def _shared(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    """Statements two bodies share, or 0 when that is under half the longer."""
    shared = len(set(a) & set(b))
    return shared if shared * 2 >= max(len(a), len(b)) else 0


def is_heavy(body_statements: int, seconds_per_iteration: float) -> bool:
    """Whether iterations of this cost are worth one cache entry each."""
    needed = max(MIN_HEAVY_ITER_SEC, HEAVY_FACTOR * body_statements * PER_STMT_OVERHEAD_SEC)
    return seconds_per_iteration >= needed


#: One measurement: seconds per iteration, the source hash of the loop as it
#: ran as one unit, and the body it was taken on (`body_identity`).
Measurement = tuple[float, str, tuple[str, ...]]


class HeavyLoopStore(VersionedJsonStore[list[Measurement]]):
    """Persisted ``header identity -> measurements``, each the last measured
    run of one loop under that header. The source hash is that of the loop as
    it ran as one unit: run again unchanged it finds that unit's entry, which
    is restored whole. Best-effort like every store here: empty means "every
    loop follows the static rule"."""

    FILENAME = _STORE_FILENAME
    VERSION = _STORE_VERSION
    FIELD = "loops"
    LOG_TAG = "HEAVY_LOOPS"

    def _load_value(self, value: object) -> list[Measurement] | None:
        if not isinstance(value, list):
            return None
        kept = []
        for item in value:
            if not (isinstance(item, list) and len(item) == 3):
                return None
            seconds, unit, body = item
            if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds < 0:
                return None
            if not (isinstance(unit, str) and isinstance(body, list) and all(isinstance(b, str) for b in body)):
                return None
            kept.append((float(seconds), unit, tuple(body)))
        return kept or None

    def _dump_value(self, value: list[Measurement]) -> list:
        return [[seconds, unit, list(body)] for seconds, unit, body in value]

    def _find(self, node: ast.For) -> tuple[str, tuple[str, ...], list[Measurement], int | None]:
        """The header's measurements, and the index of *node*'s among them."""
        self._ensure_loaded()
        identity = header_identity(node)
        body = body_identity(node)
        found = self._items.get(identity, [])
        best, at = 0, None
        for i, (_seconds, _unit, other) in enumerate(found):
            shared = _shared(body, other)
            if shared and shared >= best:  # the latest of equals
                best, at = shared, i
        return identity, body, found, at

    def get(self, node: ast.For) -> tuple[float, str] | None:
        """*node*'s measurement: seconds per iteration and its unit's source hash."""
        _identity, _body, found, at = self._find(node)
        return None if at is None else found[at][:2]

    def record(self, node: ast.For, seconds_per_iteration: float, unit_hash: str | None = None) -> None:
        """Keep the latest measurement; written only when it moved enough to
        change a verdict somewhere (a quarter), or the unit's source or the
        body changed. *unit_hash* None keeps the one recorded: an
        iteration-by-iteration run has no unit entry of its own."""
        identity, body, found, at = self._find(node)
        old = None if at is None else found[at]
        if unit_hash is None:
            unit_hash = old[1] if old else ""
        if (
            old is not None
            and old[1] == unit_hash
            and old[2] == body
            and abs(old[0] - seconds_per_iteration) <= 0.25 * max(old[0], seconds_per_iteration)
        ):
            return
        kept = [m for i, m in enumerate(found) if i != at]
        kept.append((seconds_per_iteration, unit_hash, body))
        self._items[identity] = kept[-_PER_HEADER:]
        self._write()


_STORES: StoreRegistry[HeavyLoopStore] = StoreRegistry(HeavyLoopStore)


def get_store(cache_dir: str | None) -> HeavyLoopStore:
    return _STORES.get(cache_dir)


def store_for_backend(backend) -> HeavyLoopStore | None:
    return _STORES.for_backend(backend)
