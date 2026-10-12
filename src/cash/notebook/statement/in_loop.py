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

from ...analysis.mutation_effects import WATCHED_STRUCTURES, is_module_name
from ..loop_watch import LoopWatch, branch_changes, loop_watch_hash, watch_candidates
from .derivation_edges import bump_derived_lineages

if TYPE_CHECKING:
    from ..tracking_state import TrackingState
    from .records import StatementRecords

__all__ = ["InLoopChanges", "IterationOverlay", "part_bindings"]

logger = logging.getLogger(__name__)


def part_bindings(tree: ast.Module) -> dict[str, str]:
    """Each name *tree*'s one statement binds to a part of another variable,
    or to it: ``r = rows[i]``, ``w = model['w']``, ``cfg = self.cfg``,
    ``y = x``, with that variable. Not a call's result (``r =
    rows[i].copy()``), which may be a new object."""
    if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.Assign, ast.AnnAssign)):
        return {}
    stmt = tree.body[0]
    value = stmt.value
    if value is None:
        return {}
    node = value
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    if not isinstance(node, ast.Name):
        return {}
    targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
    return {t.id: node.id for t in targets if isinstance(t, ast.Name) and t.id != node.id}


@dataclass
class IterationOverlay:
    """The lineages the statements of one decomposed loop moved, and what
    each was before (None: it had none)."""

    #: Each loop target, with the names its loop takes its values from: a
    #: change to the target is a change to what those hold.
    roots: Mapping[str, Iterable[str]]
    saved: dict[str, str | None] = field(default_factory=dict)
    #: Names a statement of its body bound to a part of another variable
    #: (``r = rows[i]``, ``w = model['w']``), with that variable: a change
    #: through the name is a change to what the variable holds.
    aliases: dict[str, set[str]] = field(default_factory=dict)
    #: Every name its statements moved, here or, for a name an enclosing
    #: loop owns, in that loop's overlay.
    moved: set[str] = field(default_factory=set)


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
        #: Verdict keys no earlier kernel recorded.
        self._unrecorded: set[str] = set()
        #: Per loop verdict, the states it was seen running on this session
        #: (:meth:`open_watch`).
        self._observed: dict[str, set[str]] = {}

    # -- Loop watches -----------------------------------------------------

    def open_watch(
        self,
        node: ast.AST,
        code: str,
        resolve_source: Callable[[str], str | None],
        cache_key: str | None = None,
    ) -> LoopWatch | None:
        """Start watching the loop *node* (*code*) before it runs; the watch
        already open for it when there is one (the rest of a loop run as one
        unit is the same loop). None for a loop that hands nothing to code
        whose effect is unknown.

        Once a verdict is recorded, the candidates it does not hold are
        fingerprinted again whenever the loop runs on a state not observed
        before in this session (*cache_key*, the unit's key when it runs as
        one, and the lineage of every variable): a loop that first ran on no
        items changed nothing, and that is no verdict on its next items. On
        a state observed before, the same code on the same inputs does what
        it did then."""
        if not isinstance(node, WATCHED_STRUCTURES):
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
        key = loop_watch_hash(code)
        verdict = self._verdict(key)
        rewatch = False
        if verdict is None or candidates - verdict:
            state = self._observed_state(cache_key)
            observed = self._observed.setdefault(key, set())
            rewatch = state not in observed
            if len(observed) >= 16:
                observed.clear()
            observed.add(state)
        watch = LoopWatch.start(code, candidates, verdict, user_ns, rewatch=rewatch)
        self._watches[code] = watch
        return watch

    def _observed_state(self, cache_key: str | None) -> str:
        """A digest of what a loop runs on: its key, when it has one, and
        the lineage of every variable."""
        digest = hashlib.sha256((cache_key or "").encode())
        for name, lineage in sorted(self.tracking_state.variable_lineage.items()):
            digest.update(f"{name}={lineage};".encode())
        return digest.hexdigest()

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
        if learning or (watch.verdict is not None and seen - watch.verdict):
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

    def branch_verdict_changes(self, node: ast.AST, resolve_source: Callable[[str], str | None]) -> set[str]:
        """What the statements, loops and ``with`` blocks in the branches of
        the ``if`` or ``try`` *node* change through calls the text cannot
        decide, as recorded (``branch_changes``); the simulation reads the
        same records."""
        if not isinstance(node, (ast.If, ast.Try)):
            return set()
        user_ns = self.shell.user_ns
        return branch_changes(node, self._verdict, user_ns, lambda name: is_module_name(name, user_ns), resolve_source)

    def _verdict(self, key: str) -> frozenset[str] | None:
        verdict = self.tracking_state.mutation_verdicts.get(key)
        if verdict is None and key not in self._unrecorded:
            verdict = self._records.recorded_mutation_verdict(key)
            if verdict is not None:
                self.tracking_state.mutation_verdicts[key] = set(verdict)
            else:
                # Asked once per session: a verdict learned later lands in
                # ``mutation_verdicts`` first.
                self._unrecorded.add(key)
        return frozenset(verdict) if verdict is not None else None

    # -- Iteration overlays -------------------------------------------------

    def push_overlay(self, roots: Mapping[str, Iterable[str]]) -> IterationOverlay:
        """A decomposed loop starts its iterations."""
        overlay = IterationOverlay(roots)
        self._overlays.append(overlay)
        return overlay

    def pop_overlay(self, overlay: IterationOverlay) -> set[str]:
        """The loop of *overlay* ended: give every name its statements moved
        the lineage it had before the loop. Returns those names.

        Inside an enclosing decomposed loop, the moves are kept in the
        outermost loop's overlay (:meth:`moved`) and taken back when it
        ends: a statement of the enclosing loop's pass that reads what the
        inner loop changed (``for r in rows: for k in ks: r.append(k)``,
        then ``s = total(rows)``) is keyed on it."""
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
        return set(overlay.saved) | overlay.moved

    def note_bindings(self, tree: ast.Module | None) -> None:
        """A statement of the innermost decomposed loop ran: remember each
        name it bound to a part of another variable (:func:`part_bindings`)."""
        if not self._overlays or tree is None:
            return
        aliases = self._overlays[-1].aliases
        for name, root in part_bindings(tree).items():
            aliases.setdefault(name, set()).add(root)

    def moved(self, names: Iterable[str], cache_key: str | None) -> None:
        """A statement of the innermost decomposed loop, keyed *cache_key*,
        ran and changed *names* in place: move each on, with what it is
        part of -- the names a loop target is taken from, in this loop or an
        enclosing one; the variable a name was bound to a part of
        (``r = rows[i]``); and every variable recorded as sharing its object
        (``w = model['w']`` before the loop, `bump_derived_lineages`)."""
        if not self._overlays or not cache_key:
            return
        targets = self._reached(names)
        if not targets:
            return
        # Kept in the outermost overlay: taken back when the outermost loop
        # ends, where that loop's update takes over, as the simulation sees it.
        outer = self._overlays[0]
        self._overlays[-1].moved.update(targets)
        lineage = self.tracking_state.lineage
        current = self.tracking_state.variable_lineage
        user_ns = self.shell.user_ns
        present = {name for name in targets if name in user_ns}
        for name in sorted(present):
            before = current.get(name)
            outer.saved.setdefault(name, before)
            moved = hashlib.sha256(f"in-loop:{before}:{cache_key}".encode()).hexdigest()
            lineage.reset_to(name, moved)
        edges = self.tracking_state.derivation_edges
        if edges and present:

            def record(target: str, lineage_hash: str) -> None:
                outer.saved.setdefault(target, current.get(target))
                lineage.reset_to(target, lineage_hash)

            bumped = bump_derived_lineages(
                edges, current, present, present, record=record, present=lambda name: name in user_ns
            )
            self._overlays[-1].moved.update(bumped)

    def _reached(self, names: Iterable[str]) -> set[str]:
        """*names* with what each is part of (:meth:`moved`); a loop target
        stands for what its loop takes its values from."""
        targets: set[str] = set()
        pending = list(names)
        seen: set[str] = set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            roots = next((o.roots[name] for o in reversed(self._overlays) if name in o.roots), None)
            if roots is not None:
                pending.extend(roots)
                continue
            targets.add(name)
            for overlay in self._overlays:
                pending.extend(overlay.aliases.get(name, ()))
        return targets
