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
Two loops sharing a header share an entry, and the later measurement wins.
"""

from __future__ import annotations

import ast

from cash.analysis.ast_util import copy_tree
from cash.backends.cache_dir import HEAVY_LOOPS_FILENAME

from .cache_key import statement_source_hash
from .control_structures.single_unit_policy import PER_STMT_OVERHEAD_SEC
from .versioned_json_store import StoreRegistry, VersionedJsonStore

_STORE_FILENAME = HEAVY_LOOPS_FILENAME
_STORE_VERSION = 1

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


def is_heavy(body_statements: int, seconds_per_iteration: float) -> bool:
    """Whether iterations of this cost are worth one cache entry each."""
    needed = max(MIN_HEAVY_ITER_SEC, HEAVY_FACTOR * body_statements * PER_STMT_OVERHEAD_SEC)
    return seconds_per_iteration >= needed


class HeavyLoopStore(VersionedJsonStore[float]):
    """Persisted ``header identity -> seconds per iteration`` of the last
    measured run. Best-effort like every store here: empty means "every loop
    follows the static rule"."""

    FILENAME = _STORE_FILENAME
    VERSION = _STORE_VERSION
    FIELD = "loops"
    LOG_TAG = "HEAVY_LOOPS"

    def _load_value(self, value: object) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value) if value >= 0 else None

    def get(self, identity: str) -> float | None:
        self._ensure_loaded()
        return self._items.get(identity)

    def record(self, identity: str, seconds_per_iteration: float) -> None:
        """Keep the latest measurement; written only when it moved enough to
        change a verdict somewhere (a quarter), not on every run."""
        self._ensure_loaded()
        old = self._items.get(identity)
        if old is not None and abs(old - seconds_per_iteration) <= 0.25 * max(old, seconds_per_iteration):
            return
        self._items[identity] = seconds_per_iteration
        self._write()


_STORES: StoreRegistry[HeavyLoopStore] = StoreRegistry(HeavyLoopStore)


def get_store(cache_dir: str | None) -> HeavyLoopStore:
    return _STORES.get(cache_dir)


def store_for_backend(backend) -> HeavyLoopStore | None:
    return _STORES.for_backend(backend)
