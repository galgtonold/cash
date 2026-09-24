"""When a ``for`` loop is learned as a split (a short head, then one unit).

The single-unit constants in :mod:`.single_unit_policy` are a STATIC guess --
an assumed per-statement cost times an iteration count, decided without
seeing the loop run. That leaves a band where the guess says "decompose" and
reality disagrees: n below the single-unit threshold, every call below
``call_unit._COST_FLOOR_S``, so neither mechanism caches anything while
per-iteration machinery is charged on every pass. Measured at n=124 on a warm
rerun against a cash-off arm, cash was SLOWER than not using cash: 0.1ms body
22ms off vs 215ms on; 2.5ms body 320ms vs 617ms.

This policy MEASURES such a loop and records a verdict; the handler reads the
verdict back on later runs. Executing a split is the handler's job and the
simulator's (``upstream/virtual_lineage.py``), and the two must agree -- see
``notebook/loop_split.py`` for why a runtime-only split is a stale-value bug.
"""

from __future__ import annotations

import ast
import logging
from typing import TYPE_CHECKING, Any

from ..loop_split import LoopSplitStore, is_split_half, loop_source_hash, store_for_backend
from .single_unit_policy import has_file_io_calls, header_safe_to_reevaluate

if TYPE_CHECKING:
    from ..statement import StatementProcessor

logger = logging.getLogger(__name__)

# Iterations measured before judging, and the split point thereafter.
# Small on purpose: the head re-runs on every warm pass and per-iteration
# overhead is exactly what the split removes, so a long head keeps the
# cost it is meant to eliminate (k=10 measured ~25ms warm on a 0.1ms body
# against 22ms for cash-off -- no gain at all; k=5 roughly halves it).
PROBE_ITERS = 5

_UNSET = object()


class LoopSplitPolicy:
    """Judges loops for splitting and remembers the verdicts.

    Holds the statement processor only to reach the cash instance: its
    backend picks the verdict store, its config the thresholds
    (``loop_split_max_iter_seconds``, ``loop_split_min_remaining_seconds``).
    """

    def __init__(self, statement_processor: StatementProcessor):
        self.statement_processor = statement_processor
        self._store: LoopSplitStore | None | object = _UNSET

    def store(self) -> LoopSplitStore | None:
        """Shared split store, or ``None`` if unresolvable (means: learn nothing)."""
        if self._store is _UNSET:
            self._store = store_for_backend(self.statement_processor.cash_instance.backend)
        return self._store  # type: ignore[return-value]

    def eligible(self, node: ast.For, iterable: Any, user_ns: dict[str, Any]) -> int | None:
        """Iteration count if this loop could be split, else ``None``.

        Checked before measuring, so an ineligible loop pays only these
        predicates. Each exclusion is load-bearing:

        * **Already a half** -- splitting a half recurses.
        * **No** ``len()`` -- a generator cannot be sliced for the tail, and
          re-iterating it drains an exhausted source.
        * **Not sliceable** -- the tail's source is ``<iter expr>[k:]``;
          ``set``/``dict`` are sized but cannot form it.
        * **Header unsafe to re-evaluate** -- both halves evaluate it. Reuses
          the single-unit path's own guard, so there is one rule rather than
          two that can drift.
        * ``break``/``continue`` -- head+tail is NOT equivalent when a break
          in the head must skip the tail.
        * ``for ... else`` -- ``else`` has one completion point; a split has
          none.
        * **File I/O in the body** -- needs per-iteration dep tracking.
        """
        if node.orelse or is_split_half(node):
            return None
        try:
            n = len(iterable)
        except TypeError:
            return None
        if n <= PROBE_ITERS:
            return None
        try:
            iterable[0:0]
        except Exception:  # noqa: BLE001 - a user __getitem__ can raise anything: not sliceable
            logger.debug("[LOOP_SPLIT] iterable is not sliceable", exc_info=True)
            return None
        if not header_safe_to_reevaluate(node.iter, iterable, user_ns):
            return None
        if has_file_io_calls(node.body):
            return None
        for sub in ast.walk(ast.Module(body=list(node.body), type_ignores=[])):
            if isinstance(sub, (ast.Break, ast.Continue)):
                return None
        return n

    def should_split(self, elapsed: float, done: int, n: int) -> bool:
        """Judge from MEASURED cost whether the remainder is worth one unit.

        *elapsed* covers ``done`` iterations including per-iteration
        decomposition overhead -- correct, because that overhead is most of
        what a split recovers for a cheap body.
        """
        if done <= 0:
            return False
        # Read on every decision, so ``cash.configure(...)`` takes effect at
        # once. They are settings because the judgement is a wall-clock
        # measurement: a kernel descheduled mid-probe measures a cheap body as
        # an expensive one, and only the threshold can be moved far enough to
        # absorb that.
        config = self.statement_processor.cash_instance.config
        max_iter = config.loop_split_max_iter_seconds
        min_remaining = config.loop_split_min_remaining_seconds
        per_iter = elapsed / done
        if per_iter >= max_iter:
            return False
        split = (n - done) * per_iter >= min_remaining
        logger.debug(
            "[LOOP_SPLIT] per_iter=%.2fms remaining=%.0fms n=%d -> %s",
            per_iter * 1000,
            (n - done) * per_iter * 1000,
            n,
            split,
        )
        return split

    def record_verdict(self, node: ast.For, elapsed: float, n: int) -> None:
        """Record that this loop should be split on later runs.

        Recording only -- this never executes a split.
        """
        if not self.should_split(elapsed, PROBE_ITERS, n):
            return
        store = self.store()
        if store is None:
            return
        try:
            store.record(loop_source_hash(node), PROBE_ITERS)
            logger.debug("[LOOP_SPLIT] recorded k=%d; splits from next run", PROBE_ITERS)
        except (OSError, ValueError, RecursionError):
            # The store swallows its own write failures; this is the source
            # hash (``ast.unparse`` of a very deep loop) and a cache dir that
            # cannot be made. Learning must never break execution.
            logger.debug("[LOOP_SPLIT] could not record a verdict", exc_info=True)

    def recorded_k(self, node: ast.For, iterable: Any, user_ns: dict[str, Any]) -> int | None:
        """Persisted ``k`` for this loop, or ``None`` if it must not be split.

        Eligibility is re-checked against the LIVE iterable even when a
        verdict exists: the verdict was recorded for this loop's source, but
        the value bound to its iterable can change between runs (a list
        becoming a generator, say), and a tail is only derivable for a sized,
        sliceable one.
        """
        store = self.store()
        if store is None:
            return None
        if self.eligible(node, iterable, user_ns) is None:
            return None
        try:
            return store.get(loop_source_hash(node))
        except (OSError, ValueError, RecursionError):
            logger.debug("[LOOP_SPLIT] verdict lookup failed", exc_info=True)
            return None
