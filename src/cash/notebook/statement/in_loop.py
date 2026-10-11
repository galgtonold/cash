"""What the statements of a running loop change, as the loop runs.

Two records, both owned by the loop that runs, never by one statement:

* :class:`~cash.notebook.loop_watch.LoopWatch`, per loop source: the names
  the loop hands to code whose effect its text cannot say, fingerprinted
  around the loop the first time its source runs, and the verdict recorded
  for the loop's lineage update, a restore and the simulation.
* :class:`IterationOverlay`, per loop decomposed into its iterations: a body
  statement that changes a name in place (``history.append(loss * lr)``)
  moves that name's lineage on, from the lineage it had and the statement's
  key, so a statement of a later iteration that reads it
  (``loss = score(history, epoch)``) is keyed on what the earlier ones did.
  The moves are taken back when the loop ends, where the loop gives every
  name it changed its lineage as a whole, as the simulation does.
"""

from __future__ import annotations

import ast
import hashlib
import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ...analysis.mutation_effects import is_module_name
from ..loop_watch import LoopWatch, loop_watch_hash, watch_candidates

if TYPE_CHECKING:
    from ..tracking_state import TrackingState
    from .records import StatementRecords

__all__ = ["InLoopChanges", "IterationOverlay"]

logger = logging.getLogger(__name__)


@dataclass
class IterationOverlay:
    """The lineages the statements of one decomposed loop moved, and what
    each was before (None: it had none)."""

    #: Each loop target, with the names its loop takes its values from: a
    #: change to the target is a change to what those hold.
    roots: Mapping[str, Iterable[str]]
    saved: dict[str, str | None] = field(default_factory=dict)


class InLoopChanges:
    """The loop watches and iteration overlays of the loops running now."""

    def __init__(self, shell, tracking_state: TrackingState, records: StatementRecords) -> None:
        self.shell = shell
        self.tracking_state = tracking_state
        self._records = records
        #: The open watch of each loop source running now.
        self._watches: dict[str, LoopWatch] = {}
        #: The overlay of each decomposed loop running now, innermost last.
        self._overlays: list[IterationOverlay] = []

    # -- Loop watches -----------------------------------------------------

    def open_watch(self, node: ast.AST, code: str, resolve_source: Callable[[str], str | None]) -> LoopWatch | None:
        """Start watching the loop *node* (*code*) before it runs; the watch
        already open for it when there is one (the rest of a loop run as one
        unit is the same loop). None for a loop that hands nothing to code
        whose effect is unknown."""
        if not isinstance(node, (ast.For, ast.While)):
            return None
        open_ = self._watches.get(code)
        if open_ is not None:
            return open_
        user_ns = self.shell.user_ns
        try:
            candidates = watch_candidates(node, user_ns, lambda name: is_module_name(name, user_ns), resolve_source)
        except Exception:
            logger.debug("[CONTROL] loop watch candidates failed", exc_info=True)
            return None
        if not candidates:
            return None
        verdict = self._verdict(loop_watch_hash(code))
        watch = LoopWatch.start(code, candidates, verdict, user_ns)
        self._watches[code] = watch
        return watch

    def watch(self, code: str) -> LoopWatch | None:
        """The watch open for the loop *code*, if any."""
        return self._watches.get(code)

    def finish_watch(self, code: str) -> frozenset[str]:
        """What the loop *code*, which ran, changed through code its text
        cannot decide; recorded when it was being learned. The watch stays
        open until :meth:`close_watch`."""
        watch = self._watches.get(code)
        if watch is None:
            return frozenset()
        learning = watch.learning and watch.seen is None
        seen = watch.finish(self.shell.user_ns)
        if learning:
            self.tracking_state.mutation_verdicts[watch.key] = set(seen)
            self._records.persist_mutation_verdict(watch.key, set(seen))
        return seen

    def close_watch(self, code: str, ran: bool) -> frozenset[str]:
        """Close the loop *code*'s watch: what it changed (:meth:`finish_watch`)
        when it *ran*, else what a restored loop is taken to have changed
        (the record, or every candidate)."""
        watch = self._watches.get(code)
        if watch is None:
            return frozenset()
        try:
            if ran or watch.seen is not None:
                return self.finish_watch(code)
            return watch.assumed()
        finally:
            self._watches.pop(code, None)

    def drop_watches(self) -> None:
        """Forget every open watch: a structure that failed leaves none."""
        self._watches.clear()

    def _verdict(self, key: str) -> frozenset[str] | None:
        verdict = self.tracking_state.mutation_verdicts.get(key)
        if verdict is None:
            verdict = self._records.recorded_mutation_verdict(key)
            if verdict is not None:
                self.tracking_state.mutation_verdicts[key] = set(verdict)
        return frozenset(verdict) if verdict is not None else None

    # -- Iteration overlays -------------------------------------------------

    def push_overlay(self, roots: Mapping[str, Iterable[str]]) -> IterationOverlay:
        """A decomposed loop starts its iterations."""
        overlay = IterationOverlay(roots)
        self._overlays.append(overlay)
        return overlay

    def pop_overlay(self, overlay: IterationOverlay) -> set[str]:
        """The loop of *overlay* ended: give every name its statements moved
        the lineage it had before the loop. Returns those names."""
        if self._overlays and self._overlays[-1] is overlay:
            self._overlays.pop()
        else:
            try:
                self._overlays.remove(overlay)
            except ValueError:
                pass
        lineage = self.tracking_state.lineage
        for name, before in overlay.saved.items():
            if before is None:
                lineage.discard(name)
            else:
                lineage.reset_to(name, before)
        return set(overlay.saved)

    def moved(self, names: Iterable[str], cache_key: str | None) -> None:
        """A statement of the innermost decomposed loop, keyed *cache_key*,
        ran and changed *names* in place: move each on (a loop target, the
        names it is taken from)."""
        if not self._overlays or not cache_key:
            return
        overlay = self._overlays[-1]
        targets: set[str] = set()
        for name in names:
            roots = overlay.roots.get(name)
            if roots is not None:
                targets.update(roots)
            else:
                targets.add(name)
        if not targets:
            return
        lineage = self.tracking_state.lineage
        current = self.tracking_state.variable_lineage
        for name in sorted(targets):
            if name not in self.shell.user_ns:
                continue
            before = current.get(name)
            overlay.saved.setdefault(name, before)
            moved = hashlib.sha256(f"in-loop:{before}:{cache_key}".encode()).hexdigest()
            lineage.reset_to(name, moved)
