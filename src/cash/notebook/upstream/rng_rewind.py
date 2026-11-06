"""Where the random stream a cell draws from stands before it runs.

A seed or a draw binds no variable, so the plan of the upstream check never
links a draw to the seed above it. :class:`RngRewind` supplies that link: before
each statement the plan re-runs that draws, and before the cell itself, it puts
each global stream where a run from the top has it there.

It works on statements at their place in the run order, never on cells or on
source text: three ``torch.manual_seed(1)`` cells are three seeds. And it never
runs a statement again for the stream alone when that statement does anything
else. In order of preference, the stream is

1. left alone, when it already stands where a run from the top has it;
2. put back to a position recorded in this kernel under the same lineage
   (``TrackingState.rng_states_by_lineage``);
3. put back to where the statement above that last moved it left it, read
   from that statement's cache entry when the check restored it;
4. set by running that statement again, only when it is a bare seed
   (``np.random.seed(0)``), which changes nothing but the stream;
5. none of these: the draw runs from where the stream stands, and cash says
   so (``RANDOM-STREAM-UNRESTORED``).

A stream nothing above the draw seeds is unseeded there, whatever a cell below
seeded on an earlier pass: it has no position a run from the top fixes, so only
the cell's own earlier start is put back (the frozen-draw rule).
"""

from __future__ import annotations

import ast
import logging
import types
from dataclasses import dataclass
from typing import Any

from ..._user_attrs import attribute_or
from ...analysis.annotations import get_statement_annotations
from ...analysis.ast_util import walk_tree
from ...analysis.code_analyzer import CodeAnalyzer, clean_cell_source, parse_cell_source, statement_code
from ...diagnostics import warn_diagnostic
from ...source_norm import exact_source_digest
from ...tracking.randomness import (
    CashRandomnessWarning,
    drawing_modules_now,
    get_drawing_rng_modules,
    get_seeding_rng_modules,
    restore_rng_state,
    rng_lineage_fingerprint,
    rng_virtual_var,
    seed_cells_not_yet_run,
    seeding_modules_now,
)
from ...tracking.randomness.detect import get_value_seeding_rng_modules
from ...value_types import BUILTIN_NAMES
from .._protocols import ShellProtocol
from ..callee_reach import helper_seeded_modules
from ..lineage_formula import UNSEEDED, seeded_unseen_by
from ..statement.randomness import position_after
from ..tracking_state import TrackingState

__all__ = ["RngRewind"]

logger = logging.getLogger(__name__)

#: The miss reason a statement the plan runs only to set a stream carries.
RERUN_REASON = "re-run to restore the random stream"


@dataclass(frozen=True, slots=True)
class _Stmt:
    """One top-level statement above the cell, at its place in the run order."""

    cell: int
    code: str
    #: Its place among the statements of its cell with the same text.
    occurrence: int
    #: The global streams it moves, by drawing, seeding or setting them.
    moves: frozenset[str]
    #: The streams it draws from.
    draws: frozenset[str]
    #: The streams it sets and nothing else: a bare seed with a value
    #: (``np.random.seed(0)``), empty for any other statement.
    bare_seed: frozenset[str]


def _module_of(var: str) -> str:
    return var[len(rng_virtual_var("")) :]


def _with_companions(state: dict, modules: set[str]) -> dict:
    """The part of *state* for *modules*: torch's device streams go with it."""
    wanted = set(modules)
    if "torch" in wanted:
        wanted.add("torch.cuda")
    return {m: s for m, s in state.items() if m in wanted}


def _bare_seed_modules(code: str) -> frozenset[str]:
    """The streams *code* seeds when it is a bare seed call and nothing else.

    ``np.random.seed(0)``, ``torch.manual_seed(SEED)``: one call, through a
    dotted name, whose arguments are constants or names it only reads. Running
    such a statement again changes the stream and nothing a variable holds.
    A seed from entropy (``seed()``) asks for another stream each run, so it
    is not one.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return frozenset()
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        return frozenset()
    call = tree.body[0].value
    if not isinstance(call, ast.Call) or sum(isinstance(n, ast.Call) for n in walk_tree(call)) != 1:
        return frozenset()

    def plain(node: ast.AST) -> bool:
        if isinstance(node, (ast.Constant, ast.Name)):
            return True
        if isinstance(node, ast.Attribute):
            return plain(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return plain(node.operand)
        return False

    if not plain(call.func) or not all(plain(a) for a in call.args) or not all(plain(k.value) for k in call.keywords):
        return frozenset()
    seeded = seeding_modules_now(code)
    if not seeded or get_value_seeding_rng_modules(code) != seeded or drawing_modules_now(code):
        return frozenset()
    return frozenset(seeded)


def _imports_only(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return bool(tree.body) and all(isinstance(n, (ast.Import, ast.ImportFrom)) for n in tree.body)


def _constant_binding(code: str) -> bool:
    """``n = 2``: an assignment of an immutable literal to plain names,
    which gives them the same value wherever it runs."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        return False
    node = tree.body[0]
    for t in node.targets:
        if not all(isinstance(e, ast.Name) for e in (t.elts if isinstance(t, ast.Tuple) else [t])):
            return False
    if any(isinstance(n, (ast.List, ast.Dict, ast.Set)) for n in walk_tree(node.value)):
        return False  # a value a call could change in place
    try:
        ast.literal_eval(node.value)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False
    return True


def _rebinds(code: str, name: str) -> bool:
    try:
        return name in CodeAnalyzer.analyze_code_block(code)[1]
    except (SyntaxError, ValueError, TypeError):
        return True


def _binder(name: str, at: int, stmts: list[_Stmt]) -> int | None:
    """The nearest statement above *at* that binds *name*."""
    for p in range(at - 1, -1, -1):
        try:
            if name in CodeAnalyzer.analyze_code_block(stmts[p].code)[1]:
                return p
        except (SyntaxError, ValueError, TypeError):
            continue
    return None


class RngRewind:
    """The upstream check's handling of the random stream."""

    def __init__(self, shell: ShellProtocol, tracking_state: TrackingState) -> None:
        self.shell = shell
        self.tracking_state = tracking_state
        self._reset()

    def _reset(self) -> None:
        #: Plan index -> the stream states to put in place before it runs.
        self._before: dict[int, dict] = {}
        #: The stream states to put in place before the cell runs.
        self._cell_state: dict = {}
        #: Plan index -> the lineage each stream put back there has in a run.
        self._before_lineages: dict[int, dict[str, str]] = {}
        #: The lineage each stream put back before the cell has in a run.
        self._cell_lineages: dict[str, str] = {}
        #: The streams the cell draws whose position this check settled; the
        #: others (unseeded, or nothing above moves them) take the frozen rule.
        self._cell_settled: set[str] = set()
        #: Whether :meth:`with_rng_chain` planned the streams of this check.
        self._planned = False

    # ------------------------------------------------------------------
    # The plan
    # ------------------------------------------------------------------

    def with_rng_chain(
        self,
        cell_code: str,
        notebook_cells: list[str],
        current_cell_idx: int | None,
        statements: list[str],
        trace: list | None = None,
        restored_states: dict[str, dict] | None = None,
    ) -> tuple[list[str], set[str]]:
        """*statements*, with the bare seeds that set a stream a re-run draw
        or the cell draws from put in at their places; and those seeds.

        *trace* is the simulation's (each entry's hidden lineages are the
        streams' lineages a statement reads), *restored_states* the stream
        states the entries the check restored left, by statement. What is put
        back instead of re-run goes in place through :meth:`before_statement`
        and :meth:`restore_position_rng_state`.
        """
        self._reset()
        if current_cell_idx is None:
            return statements, set()
        self._planned = True
        try:
            return self._plan(cell_code, notebook_cells[:current_cell_idx], statements, trace, restored_states or {})
        except (AttributeError, IndexError, TypeError, ValueError, SyntaxError) as e:  # pragma: no cover - defensive
            logger.debug("[RNG_REWIND] planning failed: %r", e, exc_info=True)
            self._reset()
            return statements, set()

    def _plan(
        self,
        cell_code: str,
        upstream: list[str],
        statements: list[str],
        trace: list | None,
        restored_states: dict[str, dict],
    ) -> tuple[list[str], set[str]]:
        cell_modules = set() if self._opts_out_of_rng_rewind(cell_code) else self._cell_draws_before_seeding(cell_code)
        simulated = self.tracking_state.simulated_lineage
        # A stream the simulation reads unseeded (`UNSEEDED`, no lineage) has
        # no position a run from the top fixes: the frozen rule takes it.
        seeded_at_cell = {m for m in cell_modules if simulated.get(rng_virtual_var(m)) not in (None, UNSEEDED)}
        if not any(self._draws(code) for code in statements):
            # Nothing re-run draws: only the cell's streams can need moving,
            # and none does where the plan leaves it and the kernel holds it
            # at the lineage a run has there. Read without the notebook.
            moved = {m for code in statements for m in self._draws(code) | seeding_modules_now(code)}
            live = self.tracking_state.variable_lineage
            off = {
                m
                for m in seeded_at_cell
                if m in moved or live.get(rng_virtual_var(m)) != simulated.get(rng_virtual_var(m))
            }
            if not off:
                self._cell_settled |= seeded_at_cell
                return statements, set()
        stmts = self._statements(upstream)
        positions = self._positions(statements, stmts)
        in_plan = {p for p in positions if p is not None}
        reads = self._trace_reads(trace)
        simulated_inputs = self._trace_inputs(trace)
        codes_seen: dict[str, int] = {}
        for s in stmts:
            codes_seen[s.code] = codes_seen.get(s.code, 0) + 1

        inserts: dict[int, None] = {}  # positions of statements put in, to run at their places
        restores: dict[tuple, dict] = {}  # ("plan", i) / ("pos", p) / ("cell",) -> states before it
        # The same keys -> the lineage each stream put back holds there in a
        # run. The kernel's lineage of a stream names where it stood before
        # the rewind moved it: the statement drawing next keys on it and
        # records the position it leaves under it, so a later rewind took a
        # position that statement left with other inputs on an earlier run.
        named: dict[tuple, dict[str, str]] = {}

        def put_back(key: tuple, module: str, state: dict, lineage: str | None) -> None:
            restores.setdefault(key, {}).update(state)
            if lineage:
                named.setdefault(key, {})[rng_virtual_var(module)] = lineage
            self.tracking_state.rng_unplaced.discard(module)

        def lineage_read(pos: int, module: str) -> tuple[bool, str | None]:
            """Whether the statement at *pos* reads a seeded *module*, as
            far as the simulation tells, and the lineage it reads."""
            target = stmts[pos]
            read = reads.get((target.cell, target.code, target.occurrence))
            if read is None:
                return True, None  # not simulated: its lineage is not known
            lineage = read.get(rng_virtual_var(module))
            if lineage == UNSEEDED:
                return False, None  # drawn unseeded there: nothing fixes its position
            return lineage is not None, lineage

        def inputs_at(pos: int) -> dict[str, str]:
            """The lineages the simulation gives the statement at *pos*'s
            inputs; empty when it was not simulated."""
            target = stmts[pos]
            return simulated_inputs.get((target.cell, target.code, target.occurrence), {})

        def left_under_these_inputs(after: tuple, pos: int) -> bool:
            """Whether the position kept under *after* was left by the
            statement at *pos* reading its inputs at the lineages a run
            has there: an input both name at another lineage (an edit
            above not run yet) can change how far it moves the stream."""
            kept = self.tracking_state.rng_position_inputs.get(after, {})
            wanted = inputs_at(pos)
            return all(kept[name] == lineage for name, lineage in wanted.items() if name in kept)

        def settle(module: str, target_pos: int, target_lineage: str | None, key: tuple) -> bool:
            """Put *module* where a run has it before the statement at
            *target_pos* (the cell when it is ``len(stmts)``). False when
            nothing above moves it: left to the frozen rule."""
            above = next((p for p in range(target_pos - 1, -1, -1) if module in stmts[p].moves), None)
            logger.debug(
                "[RNG_REWIND] %s before %s: moved last by %s; reads %s, kernel at %s",
                module,
                stmts[target_pos].code[:40] if target_pos < len(stmts) else "the cell",
                None if above is None else stmts[above].code[:40],
                str(target_lineage)[:12],
                str(self.tracking_state.variable_lineage.get(rng_virtual_var(module)))[:12],
            )
            if above is None:
                return False
            if above in in_plan or above in inserts:
                return True  # it runs again, at its place, before the target
            var = rng_virtual_var(module)
            moved_by_plan = any(p < target_pos and module in stmts[p].moves for p in (*in_plan, *inserts))
            source = stmts[above]
            # Where a run left it right after the statement that moved it last,
            # read from where that statement read: a lineage after a hidden
            # draw seen for the first time can name another place.
            seeded, came_from = lineage_read(above, module)
            if seeded and came_from is not None:
                after = position_after(source.code, var, came_from)
                recorded = self.tracking_state.rng_states_by_lineage.get(after)
                if recorded is not None and left_under_these_inputs(after, above):
                    # Named as the run that left it there named it: the
                    # simulation's lineage can name where an earlier run
                    # left the stream (a memo filled then, a seed in a call
                    # skipped now), and a draw keyed on it is served that
                    # run's value.
                    put_back(key, module, recorded, self.tracking_state.rng_position_names.get(after))
                    return True
            if target_lineage is not None:
                # A recorded state is the stream itself; the live lineage is
                # only a name for it and can match while the stream does not.
                recorded = self.tracking_state.rng_states_by_lineage.get((var, target_lineage))
                if recorded is not None:
                    put_back(key, module, recorded, target_lineage)
                    return True
                if (
                    not moved_by_plan
                    and module not in self.tracking_state.rng_unplaced
                    and self.tracking_state.variable_lineage.get(var) == target_lineage
                ):
                    return True  # the stream stands there already
            if key == ("cell",) and target_lineage is not None:
                own = self.tracking_state.rng_pre_states.get(exact_source_digest(cell_code))
                if own is not None and dict(own[1]).get(module) == target_lineage:
                    # Where this cell started last time, under the same lineage.
                    put_back(key, module, _with_companions(own[0], {module}), target_lineage)
                    return True
            entry = restored_states.get(source.code)
            if entry is not None and module in entry and codes_seen.get(source.code) == 1:
                put_back(key, module, _with_companions(entry, {module}), target_lineage)
                return True
            if module in source.bare_seed and self._inputs_ready(
                above, stmts, in_plan, inserts, target_pos, inputs_at(above)
            ):
                inserts[above] = None
                self.tracking_state.rng_unplaced.discard(module)
                return True
            if self._draws_and_binds_only(above, stmts, target_pos) and self._inputs_ready(
                above, stmts, in_plan, inserts, target_pos, inputs_at(above)
            ):
                # It binds names nothing between rebinds or changes, from
                # inputs nothing between rebinds: run again, it gives them the
                # values they hold in a run, from where its own draws start.
                inserts[above] = None
                for drawn in sorted(source.draws):
                    seeded, lineage = lineage_read(above, drawn)
                    if seeded:
                        settle(drawn, above, lineage, ("pos", above))
                return True
            self._warn_not_restored(module, source.code)
            self.tracking_state.rng_unplaced.add(module)
            return True

        for i, pos in enumerate(positions):
            if pos is None:
                continue
            for module in sorted(stmts[pos].draws):
                seeded, lineage = lineage_read(pos, module)
                if seeded:  # else unseeded there, or seeded first: nothing fixes its position
                    settle(module, pos, lineage, ("plan", i))

        for module in sorted(seeded_at_cell):  # an unseeded one takes the frozen rule
            if settle(module, len(stmts), simulated[rng_virtual_var(module)], ("cell",)):
                self._cell_settled.add(module)

        new_plan, index_of = self._merge(statements, positions, sorted(inserts), stmts)
        for key, state in restores.items():
            if key[0] == "cell":
                self._cell_state.update(state)
                self._cell_lineages.update(named.get(key, {}))
            else:
                self._before.setdefault(index_of[key], {}).update(state)
                self._before_lineages.setdefault(index_of[key], {}).update(named.get(key, {}))
        added = {stmts[p].code for p in inserts}
        if added:
            logger.debug("[UPSTREAM] Setting the random stream with %d statement(s) above", len(added))
        return new_plan, added

    @staticmethod
    def _merge(
        statements: list[str], positions: list[int | None], inserts: list[int], stmts: list[_Stmt]
    ) -> tuple[list[str], dict[tuple, int]]:
        """*statements* with the statements at *inserts* put in at their
        places; and where each one ended up (``("plan", i)``, ``("pos", p)``)."""
        out: list[str] = []
        index_of: dict[tuple, int] = {}
        pending = list(inserts)
        for i, stmt in enumerate(statements):
            pos = positions[i]
            while pending and pos is not None and pending[0] < pos:
                index_of[("pos", pending[0])] = len(out)
                out.append(stmts[pending.pop(0)].code)
            index_of[("plan", i)] = len(out)
            out.append(stmt)
        for p in pending:
            index_of[("pos", p)] = len(out)
            out.append(stmts[p].code)
        return out, index_of

    def _draws_and_binds_only(self, at: int, stmts: list[_Stmt], end: int) -> bool:
        """Whether running the statement at *at* again changes nothing but
        its draws' streams and names that hold what a run gives them there.

        ``a = np.random.rand(n)``: an assignment to plain names, from a
        draw that calls only through modules and builtins; none of its
        names is read by it (``df = df.sample(...)`` shuffles ``df`` twice),
        and no statement between it and *end* rebinds or changes them or
        the names it reads. A call on a user's object (``gs.fit(X, y)``), a
        loop, a ``def``: never.
        """
        source = stmts[at]
        if not source.draws or source.draws - set(drawing_modules_now(source.code)):
            return False  # a draw only observed happens inside a call on something
        try:
            tree = ast.parse(source.code)
        except SyntaxError:
            return False
        node = tree.body[0] if len(tree.body) == 1 else None
        if not isinstance(node, ast.Assign):
            return False
        targets: set[str] = set()
        for t in node.targets:
            elts = t.elts if isinstance(t, ast.Tuple) else [t]
            if not all(isinstance(e, ast.Name) for e in elts):
                return False
            targets |= {e.id for e in elts}
        read = {n.id for n in walk_tree(node.value) if isinstance(n, ast.Name)}
        if targets & read:
            return False
        live = self.shell.user_ns
        for call in (n for n in walk_tree(node.value) if isinstance(n, ast.Call)):
            root = call.func
            while isinstance(root, ast.Attribute):
                root = root.value
            if not isinstance(root, ast.Name):
                return False
            if self._is_module(root.id, at, stmts):
                continue
            if root is call.func and root.id in BUILTIN_NAMES and root.id not in live:
                continue
            return False
        guarded = targets | {name for name in read if not self._is_module(name, at, stmts)}
        for q in range(at + 1, end):
            if self._changes(stmts[q].code, guarded):
                return False
        return True

    @staticmethod
    def _changes(code: str, names: set[str]) -> bool:
        """Whether *code* may rebind or change in place one of *names*."""
        try:
            _inputs, outputs = CodeAnalyzer.analyze_code_block(code)
        except (SyntaxError, ValueError, TypeError):
            return True
        if outputs & names:
            return True
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return True
        for n in walk_tree(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                base = n.func.value
                while isinstance(base, (ast.Attribute, ast.Subscript)):
                    base = base.value
                if isinstance(base, ast.Name) and base.id in names:
                    return True  # a method call on it may change it
            if isinstance(n, (ast.Attribute, ast.Subscript)) and isinstance(n.ctx, (ast.Store, ast.Del)):
                base = n.value
                while isinstance(base, (ast.Attribute, ast.Subscript)):
                    base = base.value
                if isinstance(base, ast.Name) and base.id in names:
                    return True
            if isinstance(n, ast.Call) and any(isinstance(a, ast.Name) and a.id in names for a in n.args):
                if not isinstance(n.func, ast.Name) or n.func.id not in BUILTIN_NAMES:
                    return True  # handed to a function, which may change it
        return False

    def _is_module(self, name: str, at: int, stmts: list[_Stmt]) -> bool:
        """Whether *name* names a module at *at*: bound to one in the kernel,
        or, unbound there (after a restart), by an import above."""
        live = self.shell.user_ns
        if name in live:
            return isinstance(live[name], types.ModuleType)
        binder = _binder(name, at, stmts)
        return binder is not None and _imports_only(stmts[binder].code)

    def _inputs_ready(
        self,
        at: int,
        stmts: list[_Stmt],
        in_plan: set[int],
        inserts: dict[int, None],
        end: int,
        simulated: dict[str, str] | None = None,
    ) -> bool:
        """Whether every name the statement at *at* reads holds, when it runs
        again, what it held there in a run: bound in the kernel and changed
        by nothing between it and *end*, at the lineage the simulation gives
        it there (*simulated*) when it gives one, or bound by a statement the
        plan runs before it, or by an import above it, which is then run first."""
        try:
            inputs, _ = CodeAnalyzer.analyze_code_block(stmts[at].code)
        except (SyntaxError, ValueError, TypeError):
            return False
        live = self.shell.user_ns
        needed: dict[int, None] = {}
        for name in inputs:
            if name in BUILTIN_NAMES and name not in live and name not in self.tracking_state.variable_lineage:
                continue
            binder = _binder(name, at, stmts)
            if binder is not None and (binder in in_plan or binder in inserts):
                continue
            if name in live:
                if not isinstance(live[name], types.ModuleType) and any(
                    self._changes(stmts[q].code, {name}) for q in range(at + 1, end)
                ):
                    return False
                held = self.tracking_state.variable_lineage.get(name)
                if simulated and name in simulated and held is not None and held != simulated[name]:
                    return False  # an edit above, not run yet, gives it another value there
                continue
            if binder is None:
                return False
            if _imports_only(stmts[binder].code):
                needed[binder] = None
                continue
            # An immutable literal bound above and rebound by nothing below
            # it: run again at its place, it leaves the name as a run does.
            if not _constant_binding(stmts[binder].code) or any(
                _rebinds(stmts[q].code, name) for q in range(binder + 1, len(stmts))
            ):
                return False
            needed[binder] = None
        inserts.update(needed)
        return True

    def before_statement(self, index: int) -> None:
        """Put the streams where a run has them before the plan's statement
        at *index* runs (``StatementReplay.reexecute``)."""
        state = self._before.get(index)
        if state:
            restore_rng_state(state, self.tracking_state.rng_live_states)
            self._name_positions(self._before_lineages.get(index, {}))

    def _name_positions(self, lineages: dict[str, str]) -> None:
        """Give each stream just put back the lineage it has there in a run."""
        for var, lineage in lineages.items():
            self.tracking_state.lineage.reset_to(var, lineage)

    # ------------------------------------------------------------------
    # Reading the notebook
    # ------------------------------------------------------------------

    def _statements(self, upstream: list[str]) -> list[_Stmt]:
        """Every top-level statement of *upstream*, in run order, with the
        streams it moves."""
        ns = self.shell.user_ns
        ledger = self.tracking_state.observed_rng_statement_draws
        observed_cells = self.tracking_state.observed_rng_cells
        out: list[_Stmt] = []
        for idx, cell in enumerate(upstream):
            tree = parse_cell_source(cell)
            if tree is None:
                continue
            clean = clean_cell_source(cell)
            seen: dict[str, int] = {}
            first = len(out)
            for node in tree.body:
                try:
                    code = statement_code(node, clean)
                except (ValueError, TypeError, RecursionError):
                    continue
                occurrence = seen.get(code, 0)
                seen[code] = occurrence + 1
                digest = exact_source_digest(code)
                draws = set(drawing_modules_now(code)) | set(ledger.get(digest, ()))
                moves = set(draws) | set(seeding_modules_now(code))
                try:
                    moves |= helper_seeded_modules(code, ns)
                except Exception:  # noqa: BLE001 - a helper that cannot be read seeds nothing known
                    pass
                for lineages in (seeded_unseen_by(self.tracking_state, code) or {}).values():
                    moves |= {_module_of(var) for var in lineages}
                out.append(_Stmt(idx, code, occurrence, frozenset(moves), frozenset(draws), _bare_seed_modules(code)))
            # A stream the cell was seen to move that no statement of it shows
            # moving is moved somewhere in it: its last statement stands for it.
            unseen = set(observed_cells.get(exact_source_digest(cell), ())) - {m for s in out[first:] for m in s.moves}
            unseen.discard("torch.cuda")
            if unseen and len(out) > first:
                last = out[-1]
                out[-1] = _Stmt(last.cell, last.code, last.occurrence, last.moves | unseen, last.draws, frozenset())
        return out

    @staticmethod
    def _positions(statements: list[str], stmts: list[_Stmt]) -> list[int | None]:
        """Each planned statement's place among *stmts*: the next one with
        its text, in order (the plan keeps notebook order)."""
        out: list[int | None] = []
        cursor = 0
        for code in statements:
            found = next((p for p in range(cursor, len(stmts)) if stmts[p].code == code), None)
            if found is None:
                found = next((p for p in range(len(stmts)) if stmts[p].code == code), None)
            else:
                cursor = found + 1
            out.append(found)
        return out

    @staticmethod
    def _trace_reads(trace: list | None) -> dict[tuple[int, str, int], dict[str, str | None]]:
        """(cell, statement, occurrence) -> the streams' lineages it reads
        in the simulation."""
        out: dict[tuple[int, str, int], dict[str, str | None]] = {}
        seen: dict[tuple[int, str], int] = {}
        for entry in trace or ():
            cell = attribute_or(entry, "cell", -1)
            code = attribute_or(entry, "stmt_code", None)
            if code is None:
                continue
            k = seen.get((cell, code), 0)
            seen[(cell, code)] = k + 1
            hidden = attribute_or(attribute_or(entry, "input_hashes"), "hidden_lineages") or {}
            out[(cell, code, k)] = dict(hidden)
        return out

    @staticmethod
    def _trace_inputs(trace: list | None) -> dict[tuple[int, str, int], dict[str, str]]:
        """(cell, statement, occurrence) -> the lineages of its inputs in
        the simulation."""
        out: dict[tuple[int, str, int], dict[str, str]] = {}
        seen: dict[tuple[int, str], int] = {}
        for entry in trace or ():
            cell = attribute_or(entry, "cell", -1)
            code = attribute_or(entry, "stmt_code", None)
            if code is None:
                continue
            k = seen.get((cell, code), 0)
            seen[(cell, code)] = k + 1
            out[(cell, code, k)] = dict(attribute_or(entry, "input_hashes") or {})
        return out

    def _draws(self, code: str) -> set[str]:
        """The streams the statement *code* draws from, as written or as seen."""
        seen = self.tracking_state.observed_rng_statement_draws.get(exact_source_digest(code), ())
        return set(drawing_modules_now(code)) | set(seen)

    def _cell_draws_before_seeding(self, cell_code: str) -> set[str]:
        """The streams the cell draws from before seeding them itself: one
        it seeds first draws only what its own seed sets, wherever the
        stream stood."""
        drawing = self._current_cell_drawing_modules(cell_code)
        tree = parse_cell_source(cell_code)
        if tree is None or not drawing:
            return drawing
        clean = clean_cell_source(cell_code)
        settled: set[str] = set()
        out: set[str] = set()
        for node in tree.body:
            try:
                code = statement_code(node, clean)
            except (ValueError, TypeError, RecursionError):
                continue
            draws = self._draws(code) & drawing
            seeds = set(get_value_seeding_rng_modules(code)) & drawing
            out |= draws - settled
            settled |= seeds - draws
        # Seen moved by the cell but by no statement of it: drawn first.
        return out | (drawing - settled - out)

    def _warn_not_restored(self, module: str, above: str) -> None:
        first = above.strip().splitlines()[0] if above.strip() else above
        if len(first) > 60:
            first = first[:57] + "..."
        warn_diagnostic(
            CashRandomnessWarning,
            "RANDOM-STREAM-UNRESTORED",
            f"cash cannot put the `{module}` stream back where a run from the top has it here: "
            f"`{first}` above moved it last, and running it again would change more than the stream. "
            "The draw continues from where the stream stands now.",
            "run the notebook from the top for the numbers a run from the top gives, or seed the stream "
            "in the cell that draws.",
            location=("<cash>", 0),
        )

    # ------------------------------------------------------------------
    # The badge
    # ------------------------------------------------------------------

    @staticmethod
    def label_rerun_metrics(executed_metrics: list, rng_rerun: set[str]) -> None:
        """Explain, on the badge, why a seed was re-run.

        A seed pulled into the plan only to set the random stream would
        otherwise render as a bare COMPUTED row in the UPSTREAM section with
        no attribution. Stamp its ``miss_reason`` — the badge's "why did this
        re-run?" field. Matched on stripped source; never fatal.
        """
        if not rng_rerun:
            return
        wanted = {s.strip() for s in rng_rerun}
        for m in executed_metrics:
            try:
                if m.get("code", "").strip() in wanted:
                    m["miss_reason"] = RERUN_REASON
            except (AttributeError, TypeError):  # pragma: no cover - defensive
                continue

    # ------------------------------------------------------------------
    # The cell
    # ------------------------------------------------------------------

    def _current_cell_drawing_modules(self, cell_code: str) -> set[str]:
        """RNG modules this cell draws from — statically OR by prior observation.

        Static analysis sees ``np.random.rand()`` in the cell; the observed set
        adds modules a call like ``model.fit()`` changed at runtime, so
        an indirect draw is treated like a direct one on re-run. So does a
        statement seen drawing in another cell text: an edit elsewhere in
        the cell (``hidden = 8`` above ``model = Model(hidden)``) leaves the
        cell's own record behind, not the draw.
        """
        modules = set(get_drawing_rng_modules(cell_code))
        digest = exact_source_digest(cell_code)
        modules |= self.tracking_state.observed_rng_cells.get(digest, set())
        ledger = self.tracking_state.observed_rng_statement_draws
        tree = parse_cell_source(cell_code) if ledger else None
        if tree is not None:
            clean = clean_cell_source(cell_code)
            for node in tree.body:
                try:
                    modules |= set(ledger.get(exact_source_digest(statement_code(node, clean)), ()))
                except (ValueError, TypeError, RecursionError):
                    continue
        modules.discard("torch.cuda")
        return modules

    @staticmethod
    def _opts_out_of_rng_rewind(cell_code: str) -> bool:
        """True if a statement in *cell_code* carries ``# @cash:no-cache``.

        The rewind is what freezes an unseeded value: the statement re-executes
        but lands on the same stream position, so it redraws the same number.
        Caching is not involved -- a cheap draw is under the persistence floor
        and is never stored in the first place.

        ``no-cache`` is the documented way to say "run this for real every
        time", and the warning cash prints on a frozen draw names it directly.
        So it has to switch the REWIND off, not just caching; otherwise the
        statement dutifully re-executes, redraws the identical value, and the
        one escape hatch users are told to reach for silently does nothing.

        Read the way the statement's own run reads it, so the directive counts
        on a line of its own above the statement and at the end of its line.
        """
        if "@cash:" not in cell_code:
            return False
        tree = parse_cell_source(cell_code)
        if tree is None:
            return False
        clean = clean_cell_source(cell_code)
        return any(get_statement_annotations(clean, node).no_cache for node in tree.body)

    def cell_touches_rng(self, src: str) -> bool:
        """True if *src* seeds or draws — statically or by prior observation."""
        if get_seeding_rng_modules(src) or get_drawing_rng_modules(src):
            return True
        digest = exact_source_digest(src)
        return bool(self.tracking_state.observed_rng_cells.get(digest))

    def restore_position_rng_state(
        self,
        cell_code: str,
        notebook_cells: list[str],
        current_cell_idx: int | None,
    ) -> None:
        """Put the streams the cell draws from where they stand before it in
        a run from the top.

        A stream :meth:`with_rng_chain` settled goes where it found; the
        others (unseeded at the cell, or moved by nothing above it) go back
        to where this cell itself started last time, while the seeds that
        position was taken under still hold -- what freezes an unseeded
        draw. On a cache HIT the statement restores its own post-state
        afterwards, so this is harmless.
        """
        try:
            drawing = self._current_cell_drawing_modules(cell_code)
            if not drawing:
                return
            if self._opts_out_of_rng_rewind(cell_code):
                return
            if self._planned:
                if self._cell_state:
                    restore_rng_state(self._cell_state, self.tracking_state.rng_live_states)
                    self._name_positions(self._cell_lineages)
                drawing -= self._cell_settled
                if not drawing:
                    return
            self._restore_own_start(cell_code, notebook_cells, current_cell_idx, drawing)
        except (AttributeError, IndexError, TypeError):  # pragma: no cover - defensive
            return
        finally:
            self._reset()

    def _restore_own_start(
        self, cell_code: str, notebook_cells: list[str], current_cell_idx: int | None, drawing: set[str]
    ) -> None:
        """Put *drawing* back where this cell started last time, or, never
        started under these seeds, where the nearest cell above that moved a
        stream left it."""
        end = len(notebook_cells) if current_cell_idx is None else current_cell_idx
        # An edited seed above that has not run makes every position recorded
        # below it belong to the old seed.
        upstream = notebook_cells[:end]
        if seed_cells_not_yet_run(drawing, upstream, self.tracking_state.executed_cell_source_hashes):
            return
        own = self.tracking_state.rng_pre_states.get(exact_source_digest(cell_code))
        if own is not None:
            own_state, own_fingerprint = own
            if own_fingerprint == rng_lineage_fingerprint(self.tracking_state.variable_lineage, drawing):
                restore_rng_state(_with_companions(own_state, drawing), self.tracking_state.rng_live_states)
                return
        post_states = self.tracking_state.rng_post_states
        for idx in range(end - 1, -1, -1):
            src = notebook_cells[idx]
            if not self.cell_touches_rng(src):
                continue
            state: Any = post_states.get(exact_source_digest(src))
            if state is not None:
                restore_rng_state(_with_companions(state, drawing), self.tracking_state.rng_live_states)
            return  # nearest predecessor only, whether or not it was recorded
