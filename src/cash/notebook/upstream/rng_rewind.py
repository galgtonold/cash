"""Where the random stream a cell draws from stands before it runs.

A seed or a draw binds no variable, so the simulation never links a draw to
the seed it depends on. :class:`RngRewind` supplies that link for the upstream
check: it re-runs a stale seed and the draws after it, brings in the
chain ahead of a re-executed draw, and puts the stream back to the position a
top-to-bottom run gives just before the cell.
"""

from __future__ import annotations

import ast
import logging

from ...analysis.annotations import get_statement_annotations
from ...analysis.ast_util import parse_cached
from ...analysis.code_analyzer import CodeAnalyzer, clean_cell_source, parse_cell_source
from ...diagnostics import warn_diagnostic
from ...source_norm import exact_source_digest
from ...tracking.randomness import (
    CashRandomnessWarning,
    get_drawing_rng_modules,
    get_seeding_rng_modules,
    hidden_lineage_reads,
    observed_rng_reads,
    restore_rng_state,
    rng_lineage_fingerprint,
    rng_virtual_var,
    seed_cells_not_yet_run,
    seed_epochs,
)
from ...value_types import BUILTIN_NAMES
from .._protocols import ShellProtocol
from ..tracking_state import TrackingState

__all__ = ["RngRewind"]

logger = logging.getLogger(__name__)


class RngRewind:
    """The upstream check's handling of the random stream."""

    def __init__(self, shell: ShellProtocol, tracking_state: TrackingState) -> None:
        self.shell = shell
        self.tracking_state = tracking_state

    def with_rng_chain(
        self, cell_code: str, notebook_cells: list[str], current_cell_idx: int, statements: list[str]
    ) -> tuple[list[str], set[str]]:
        """*statements*, with what re-establishes the random stream the cell
        draws from in front; and which of those were added for that alone.

        A bare ``np.random.seed(N)`` binds no variable, so the
        simulator never links it to a downstream draw. An upstream seed cell
        edited but not re-run is re-run first -- re-running a seed is
        idempotent -- and a draw re-executed because an ORDINARY input changed
        still runs from its top-to-bottom stream position.
        """
        before = set(statements)
        statements = self._prepend_stale_seed_cells(cell_code, notebook_cells, statements, current_cell_idx)
        statements = self._prepend_rng_chain_for_reexecuted_draws(notebook_cells, statements, current_cell_idx)
        return statements, {s for s in statements if s not in before and self.cell_touches_rng(s)}

    def _prepend_stale_seed_cells(
        self,
        cell_code: str,
        notebook_cells: list[str],
        statements: list[str],
        current_cell_idx: int | None,
    ) -> list[str]:
        """Schedule an edited-but-not-rerun seed cell ahead of a draw.

        When the current cell draws from a global RNG and an UPSTREAM cell seeds
        that module with source that has not executed this session (an edited or
        never-run seed), that seed's side effect must be re-established before the
        draw. But re-seeding alone is not enough when draws sit between
        the seed and the current cell: those intervening draws advanced the stream
        under the OLD seed, so the current draw would run from the new seed's
        position 0 instead of the position the chain holds top-to-bottom
        (combined case). So when a seed is stale, re-run the whole RNG
        chain from the earliest stale seed to the current cell, in order: the
        re-seed updates the global epoch, which makes each intervening draw miss
        and recompute under the new seed, advancing the stream correctly.

        Returns *statements* with those cells prepended in notebook order,
        de-duplicated against what is already scheduled. A no-op when the cell
        does not draw or no seed is stale, so warm draws and the plain re-run
        path are untouched. Only cells strictly BEFORE the current one count.
        """
        try:
            drawing = self._current_cell_drawing_modules(cell_code)
            if not drawing:
                return statements
            # Restrict to genuine upstream cells; a None index means "treat all
            # as upstream" (the checker's own fallback), so scan everything then.
            upstream = notebook_cells if current_cell_idx is None else notebook_cells[:current_cell_idx]
            executed = self.tracking_state.executed_cell_source_hashes
            stale = seed_cells_not_yet_run(drawing, upstream, executed)
            if not stale:
                return statements
            # Rebuild the RNG chain from the earliest stale seed forward: every
            # random cell (seed or draw) from there to the current cell, in order.
            earliest = min(idx for _module, idx in stale)
            already = set(statements)
            prepend: list[str] = []
            for idx in range(earliest, len(upstream)):
                src = upstream[idx]
                if not self.cell_touches_rng(src):
                    continue
                if src not in already and src not in prepend:
                    prepend.append(src)
            if not prepend:
                return statements
            inputs = self._input_definitions_of_cells(prepend, upstream)
            rest = [stmt for stmt in statements if stmt not in inputs]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[UPSTREAM] Rebuilding RNG chain (%d cells) before draw", len(prepend))
            return inputs + prepend + rest
        except (AttributeError, IndexError, TypeError, ValueError):  # pragma: no cover - defensive
            return statements

    def _input_definitions_of_cells(self, cells: list[str], upstream: list[str]) -> list[str]:
        """The upstream statements, outside *cells*, that bind what *cells*
        read and the kernel does not hold, in source order: after a restart
        the chain's ``def temp_seed`` cell reads ``contextlib`` from an
        import above it, which must run first (`_with_input_definitions`)."""
        all_stmts: list[str] = []
        chain: set[int] = set()
        for cell in upstream:
            tree = parse_cached(clean_cell_source(cell))
            if tree is None:
                continue
            for node in tree.body:
                try:
                    stmt = ast.unparse(node)
                except (ValueError, TypeError):
                    continue
                if cell in cells:
                    chain.add(len(all_stmts))
                all_stmts.append(stmt)
        extra = self._with_input_definitions(chain, all_stmts) - chain
        return [all_stmts[idx] for idx in sorted(extra)]

    def _prepend_rng_chain_for_reexecuted_draws(
        self,
        notebook_cells: list[str],
        statements: list[str],
        current_cell_idx: int | None,
    ) -> list[str]:
        """Re-establish the RNG stream before a *re-executed* upstream draw.

        The RNG state is a side-effect dependency a draw consumes, but it binds no
        variable, so the lineage graph carries no edge from a draw back to its
        seed. When reconstruction re-executes a draw because one of its ORDINARY
        inputs changed (e.g. ``arr = np.random.rand(3) * MULT`` after editing
        ``MULT``), the unchanged upstream ``seed()`` is not scheduled, so the draw
        re-runs from wherever the live stream was left. The result matches neither
        a cache-off run nor a clean top-to-bottom run -- a silently wrong value.

        Fix: re-run the RNG chain that PRECEDES the earliest re-executed draw --
        the seed plus any draws ahead of it in source order that aren't already
        scheduled -- so that re-executed draw lands at the stream position it
        holds top-to-bottom. Statements AT or AFTER the earliest re-executed draw
        stay in the plan (the re-executed draws run there, in order; an unchanged
        later draw keeps its cached value, whose position is unaffected when the
        edit does not change how many values the re-executed draws consume).

        Chain statements come with the definitions they READ -- see
        :meth:`_with_input_definitions`. Selecting purely on "touches an RNG
        module" would schedule ``base = np.random.randn(n)`` without the
        ``n = 500`` beside it, and the reconstruction would raise ``NameError``.
        Its cell-granular sibling never had this problem: whole cells carry
        their siblings along.

        Statement-granular sibling of :meth:`_prepend_stale_seed_cells`, triggered
        by a re-executed DRAW rather than a stale seed. A no-op on the warm path
        (nothing re-executes), when the seed is already scheduled, and when no
        upstream RNG statement precedes the earliest re-executed draw.
        """
        try:
            drawn: set[str] = set()
            seeded_in_plan: set[str] = set()
            for stmt in statements:
                drawn |= self._draws(stmt)
                seeded_in_plan |= self._seeds(stmt, surely=True)
            # Only modules whose draw re-executes but whose seed is NOT already
            # being re-run need their chain re-established.
            missing = drawn - seeded_in_plan
            if not missing:
                return statements
            upstream = notebook_cells if current_cell_idx is None else notebook_cells[:current_cell_idx]
            already = set(statements)

            # Every upstream statement in source order, so a chain statement's
            # own inputs stay locatable, plus the positions of those touching a
            # missing module, tagged with whether they draw.
            all_stmts: list[str] = []
            rng_positions: list[tuple[int, bool]] = []
            for cell in upstream:
                tree = parse_cached(clean_cell_source(cell))
                if tree is None:
                    continue
                for node in tree.body:
                    try:
                        stmt = ast.unparse(node)
                    except (ValueError, TypeError):
                        continue
                    all_stmts.append(stmt)
                    draws = self._draws(stmt) & missing
                    seeds = self._seeds(stmt) & missing
                    if draws or seeds:
                        rng_positions.append((len(all_stmts) - 1, bool(draws)))

            # The earliest re-executed draw is the boundary: everything strictly
            # before it re-runs to advance the stream; it and everything after
            # stay in the plan.
            boundary = next(
                (k for k, (idx, is_draw) in enumerate(rng_positions) if is_draw and all_stmts[idx] in already),
                None,
            )
            if boundary is None:
                return statements

            chain = {idx for idx, _is_draw in rng_positions[:boundary] if all_stmts[idx] not in already}
            if not chain:
                return statements
            selected = self._with_input_definitions(chain, all_stmts)

            prepend: list[str] = []
            for idx in sorted(selected):
                stmt = all_stmts[idx]
                if stmt not in prepend:
                    prepend.append(stmt)
            if not prepend:
                return statements
            # A definition the plan already scheduled moves up with the chain
            # that reads it (see `_with_input_definitions`).
            rest = list(statements)
            for stmt in prepend:
                if stmt in rest:
                    rest.remove(stmt)
            statements = rest
            logger.debug(
                "[UPSTREAM] Re-establishing RNG chain (%d stmts) before a re-executed draw",
                len(prepend),
            )
            return prepend + statements
        except (AttributeError, IndexError, TypeError, ValueError):  # pragma: no cover - defensive
            return statements

    def _draws(self, stmt: str) -> set[str]:
        """The global streams *stmt* draws from: those it spells, and those a
        run of it was seen to draw from inside a call (``df.sample(...)``,
        ``model.fit()``: the observed-draw ledger). A ``def`` or ``class``
        draws when it is called."""
        if _only_defines(stmt):
            return set()
        prefix = rng_virtual_var("")
        observed = {var[len(prefix) :] for var in observed_rng_reads(self.tracking_state, stmt)}
        return get_drawing_rng_modules(stmt) | observed

    def _seeds(self, stmt: str, *, surely: bool = False) -> set[str]:
        """The global streams *stmt* seeds: by a seed it spells, or through a
        notebook function it calls (``set_seed(0)``). With *surely*, a call
        through a memo (``@cache``) counts for nothing: a memo that serves
        does not run its body, so the seed in it may not run again."""
        from ..callee_reach import helper_seeded_modules

        spelled = get_seeding_rng_modules(stmt)
        if surely and _calls_a_memo(stmt, self.shell.user_ns):
            return spelled
        if not surely and not stmt.lstrip().startswith(("import ", "from ")):
            # Seen to seed when it last ran, through code cash does not read
            # as seeding (``mylib.setup(4)``, ``Exp(3)`` seeding in __init__).
            # Not an import: run again, it finds its module loaded and seeds
            # nothing.
            adopted = self.tracking_state.adopted_rng_after.get(exact_source_digest(stmt))
            if adopted:
                prefix = rng_virtual_var("")
                spelled = spelled | {var[len(prefix) :] for var in adopted[1] if var.startswith(prefix)}
        try:
            return spelled | set(helper_seeded_modules(stmt, self.shell.user_ns))
        except (SyntaxError, ValueError, AttributeError, RecursionError, TypeError):
            return spelled

    def _with_input_definitions(self, chain: set[int], all_stmts: list[str]) -> set[int]:
        """Widen an RNG chain to include the definitions its statements read.

        The chain is chosen by whether a statement touches an RNG module, which
        says nothing about what it *reads*. ``base = np.random.randn(n)`` is a
        draw and gets prepended; the ``n = 500`` beside it in the same cell is
        not a draw and does not, so the reconstruction raises ``NameError``.

        For every free name a chain statement reads, pull in the nearest
        PRECEDING upstream statement that binds it, then repeat for that
        statement's own inputs. Returns the widened index set; the caller
        re-sorts, so source order is preserved and a definition always lands
        ahead of its reader.

        A name already bound in the live namespace needs no statement of its
        own: re-deriving it would re-run work the kernel already holds, which is
        cheap for ``n = 500`` and not cheap for ``n = load_config()``.

        A definition the plan already schedules is selected too, and the
        caller moves it into the prepend: left where it was, it ran after the
        chain, and after a restart ``np.random.seed(0)`` ran ahead of the
        scheduled ``import numpy as np`` and failed. Moving it earlier keeps it
        ahead of its other consumers, which all come later in the plan.
        """
        live = self.shell.user_ns
        # name -> indices binding it, built once and only if a name goes missing.
        definers: dict[str, list[int]] | None = None

        selected = set(chain)
        queue = sorted(chain)
        while queue:
            idx = queue.pop()
            try:
                inputs, _outputs = CodeAnalyzer.analyze_code_block(all_stmts[idx])
            except (SyntaxError, ValueError, TypeError):
                continue
            for name in inputs:
                # A builtin name the user never bound needs no statement; one
                # they did (`format = "csv"`) is found like any other.
                if name in live or (name in BUILTIN_NAMES and name not in self.tracking_state.variable_lineage):
                    continue
                if definers is None:
                    definers = {}
                    for j, stmt in enumerate(all_stmts):
                        try:
                            _in, outs = CodeAnalyzer.analyze_code_block(stmt)
                        except (SyntaxError, ValueError, TypeError):
                            continue
                        for out in outs:
                            definers.setdefault(out, []).append(j)
                nearest = next((j for j in reversed(definers.get(name, [])) if j < idx), None)
                if nearest is None or nearest in selected:
                    continue
                selected.add(nearest)
                queue.append(nearest)
        return selected

    def warn_redrawn(self, statements: list[str], current_cell_idx: int | None) -> None:
        """RANDOM-REDRAWN for each statement in the plan *statements* that
        draws from a seeded global stream no statement before it in the plan
        seeds again (by a seed it spells or a notebook function it calls).

        Such a draw continues from wherever the stream stands now, not from
        the position a run from the top gives it: a value it makes can differ
        from a plain run. Said, never silent; once per statement.
        """
        try:
            seeded = {rng_virtual_var(module) for module in seed_epochs()}
            if not seeded:
                return
            reset: set[str] = set()
            for stmt in statements:
                tree = parse_cached(stmt)
                if tree is not None and all(
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in tree.body
                ):
                    continue  # its body seeds and draws when it is called
                try:
                    spelled = {rng_virtual_var(module) for module in get_seeding_rng_modules(stmt)}
                except (SyntaxError, ValueError, AttributeError, RecursionError, TypeError):
                    spelled = set()
                called = {rng_virtual_var(module) for module in self._seeds(stmt)} - spelled
                if tree is not None and len(tree.body) == 1 and isinstance(tree.body[0], (ast.With, ast.AsyncWith)):
                    # ``with temp_seed(123):`` seeds before its body draws.
                    from ..callee_reach import temp_seeded_modules

                    for item in tree.body[0].items:
                        context = ast.unparse(item.context_expr)
                        spelled |= {
                            rng_virtual_var(m)
                            for m in self._seeds(context) | temp_seeded_modules(context, self.shell.user_ns)
                        }
                # A draw after a seed the statement spells reads that seed.
                drawn = (hidden_lineage_reads(stmt) | observed_rng_reads(self.tracking_state, stmt)) & seeded
                unset = drawn - reset - spelled
                if unset:
                    self._warn_redrawn(stmt, sorted(unset)[0], current_cell_idx)
                reset |= spelled | called
        except (AttributeError, TypeError, ValueError):  # pragma: no cover - defensive
            return

    @staticmethod
    def _warn_redrawn(stmt: str, var: str, current_cell_idx: int | None) -> None:
        first = stmt.strip().splitlines()[0] if stmt.strip() else stmt
        if len(first) > 60:
            first = first[:57] + "..."
        module = var[len(rng_virtual_var("")) :]
        cell = "this cell" if current_cell_idx is None else f"cell {current_cell_idx + 1}"
        warn_diagnostic(
            CashRandomnessWarning,
            "RANDOM-REDRAWN",
            f"to bring {cell}'s inputs up to date, cash ran `{first}` above it again, and it draws from "
            f"`{module}` again from where the stream stands now, not from where a run from the top has it. "
            "What it made can differ from a plain run.",
            "for the numbers a run from the top gives, run the notebook from the top; or seed the stream in "
            "the cell that draws.",
            location=("<cash>", 1),
        )

    @staticmethod
    def label_rerun_metrics(executed_metrics: list, rng_rerun: set[str]) -> None:
        """Explain, on the badge, why an RNG statement was re-executed (Stage 2 UX).

        A seed/draw pulled into the plan only to re-establish the random stream
        would otherwise render as a bare COMPUTED row in the UPSTREAM section
        with no attribution. Stamp its ``miss_reason`` — the badge's "why did
        this re-run?" field — so the reason reads alongside the Stage-1 random
        pill the row already carries. Matched on stripped source; never fatal.
        """
        if not rng_rerun:
            return
        wanted = {s.strip() for s in rng_rerun}
        for m in executed_metrics:
            try:
                if m.get("code", "").strip() in wanted:
                    m["miss_reason"] = "re-run to restore the random stream"
            except (AttributeError, TypeError):  # pragma: no cover - defensive
                continue

    def _current_cell_drawing_modules(self, cell_code: str) -> set[str]:
        """RNG modules this cell draws from — statically OR by prior observation.

        Static analysis sees ``np.random.rand()`` in the cell; the observed set
        adds modules a call like ``model.fit()`` changed at runtime, so
        an indirect draw is treated like a direct one on re-run.
        """
        modules = set(get_drawing_rng_modules(cell_code))
        digest = exact_source_digest(cell_code)
        modules |= self.tracking_state.observed_rng_cells.get(digest, set())
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
        """Restore the RNG to the state it holds just before this cell.

        If the current cell draws, find the nearest UPSTREAM cell that touched the
        RNG (seed or draw) and whose post-state we recorded this session, and
        restore that state. A re-executed draw then continues from the correct
        stream position instead of the last-left live state. A no-op
        unless the cell draws and such a predecessor exists; on a cache HIT the
        statement restores its own post-state afterwards, so this is harmless.
        """
        try:
            drawing = self._current_cell_drawing_modules(cell_code)
            if not drawing:
                return
            if self._opts_out_of_rng_rewind(cell_code):
                return
            end = len(notebook_cells) if current_cell_idx is None else current_cell_idx
            # If an upstream seed is stale (edited-not-rerun), the whole chain of
            # recorded post-states below it is stale too — restoring one would
            # apply the OLD seed's position. Defer to the reseed path
            # (_prepend_stale_seed_cells) instead of using a stale snapshot.
            upstream = notebook_cells[:end]
            if seed_cells_not_yet_run(drawing, upstream, self.tracking_state.executed_cell_source_hashes):
                return
            # PRIMARY: the position this cell itself started from last time.
            # That is precisely what re-executing its draw needs, and it is exact
            # rather than inferred -- the upstream scan below reaches the same
            # value only indirectly, via the NEAREST predecessor's post-state
            # (equal by construction when nothing RNG-touching sits between).
            # Safe to prefer because the fingerprint expires it as soon as the
            # seed behind it changes, the same lineage check that invalidates any
            # other value.
            own = self.tracking_state.rng_pre_states.get(exact_source_digest(cell_code))
            if own is not None:
                own_state, own_fingerprint = own
                if own_fingerprint == rng_lineage_fingerprint(
                    self.tracking_state.variable_lineage,
                    drawing,
                ):
                    restore_rng_state(own_state, self.tracking_state.rng_live_states)
                    return
                # Seed changed since it was recorded: the saved position belongs
                # to the old seed. Fall through rather than apply it.
            # FALLBACK: nearest upstream cell that touched RNG. Still needed when
            # this cell has no recorded start of its own (never run this session)
            # or its seed moved on, and it stays fresher than a stale own-position
            # when an upstream cell re-ran more recently than this one.
            post_states = self.tracking_state.rng_post_states
            for idx in range(end - 1, -1, -1):
                src = notebook_cells[idx]
                if not self.cell_touches_rng(src):
                    continue
                digest = exact_source_digest(src)
                state = post_states.get(digest)
                if state is not None:
                    restore_rng_state(state, self.tracking_state.rng_live_states)
                return  # nearest predecessor only, whether or not it was recorded
        except (AttributeError, IndexError, TypeError):  # pragma: no cover - defensive
            return


def _only_defines(stmt: str) -> bool:
    """Whether *stmt* only defines functions and classes."""
    tree = parse_cached(stmt)
    return (
        tree is not None
        and bool(tree.body)
        and all(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in tree.body)
    )


def _calls_a_memo(stmt: str, namespace: dict) -> bool:
    """Whether *stmt* calls a name bound to a wrapped function (a memo)."""
    tree = parse_cached(stmt)
    if tree is None:
        return False
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and hasattr(namespace.get(node.func.id), "__wrapped__")
        for node in ast.walk(tree)
    )
