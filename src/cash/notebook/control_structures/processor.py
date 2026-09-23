"""
Control Structure Processing for Statement-Level Caching

This module provides handlers for processing control structures (for, while, if,
with, try).

**For loops** are decomposed per-iteration: each body statement is passed to the
statement processor individually, with an iteration context marker in the code.
This ensures:
- Expensive pure computations inside loops are cached per-iteration.
- Statements that mutate external variables (e.g., ``a.append(x)``) are detected
  by the statement processor's mutation detection and executed directly (no cache).
- If a loop iteration was cached previously and nothing changed, it restores
  instantly.

**If statements** and **try/except blocks** are decomposed per-statement:
each branch statement is processed individually with a control-context marker.
This gives correct per-statement caching, badge display, and ensures that
side-effect statements like ``print()`` always execute.

**All other control structures** (while, with) are executed as single cacheable
units — the entire code is passed to the statement processor, which handles
cache key computation, mutation detection, and side-effect checking.

Loops containing ``break`` or ``continue`` are also executed as single units,
because decomposing them per-iteration is not possible (those statements must
execute inside a loop context).
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import logging
import random
import sys
import types
from typing import TYPE_CHECKING, Any

from ...analysis.cacheability import (
    callee_global_mutations,
    statement_calls_user_writer,
    statement_writes_files,
)
from ...analysis.code_analyzer import CodeAnalyzer
from ..cache_key import called_function_globals, control_outcome_key
from ..cache_status import CacheStatus
from ..lineage_formula import statement_environment_reads
from ..statement.file_deps import compute_file_hash_component
from ..write_observer import observe_writes
from . import helpers as _helpers
from .common import (
    ControlStructureResult,
    contains_break_or_continue,
    get_control_structure_type,
)
from .for_handler import ForLoopHandler
from .if_handler import IfHandler
from .try_handler import TryHandler

if TYPE_CHECKING:
    from ...analysis.annotations import CacheAnnotation
    from ..statement import ProcessResult

__all__ = ["ControlStructureProcessor"]

logger = logging.getLogger(__name__)


def _global_rng_fingerprint() -> tuple:
    """The state of ``random``'s and numpy's global generators, comparable with ``==``."""
    np = sys.modules.get("numpy")
    numpy_state: tuple | None = None
    if np is not None:
        try:
            kind, keys, pos, has_gauss, gauss = np.random.get_state()
            numpy_state = (kind, keys.tobytes(), pos, has_gauss, gauss)
        except Exception:  # noqa: BLE001 - an unreadable state is not the same state
            numpy_state = (object(),)
    return (random.getstate(), numpy_state)


def _status(metric: Any) -> Any:
    status = metric.get("status") if isinstance(metric, dict) else getattr(metric, "status", None)
    return CacheStatus(status) if isinstance(status, str) and status in CacheStatus.__members__ else status


def _entry_lineages(
    reads: set[str],
    lineage: dict[str, str],
    simulated: dict[str, str] | None,
) -> dict[str, str]:
    """What each name this structure reads was worth when it ran.

    The runtime's own lineage wherever it has one, and the simulation's where
    it does not. A name bound in the same cell as ``%cash_on`` is in the
    second group forever: cash was not listening when that cell started, so
    nothing recorded what ``DATA = Path(...)`` produced. The simulation reads
    that cell out of the .ipynb and has a lineage for it like any other.

    Recording the runtime's silence for such a name left this dict SHORT of a
    key the simulation carries, so ``recorded[0] == input_hashes`` in
    ``VirtualLineage._simulate_one_control_unit`` was false every time, the
    loop's recorded outcome was never adopted, and the loop re-ran with
    everything below it after every restart -- round 23's symptom, still live
    for this one shape. Measured 2026-09-20 on the same eight-iteration loop:
    0.94 s re-running the chain with ``DATA`` in the ``%cash_on`` cell against
    0.07 s and ``5 upstream steps not re-run`` with it one cell lower.

    Filling the gap from the simulation rather than inventing a value is what
    keeps the comparison honest. Edit that cell and the simulated lineage
    moves, so the recorded outcome stops matching -- the same way a tracked
    name behaves, and the reason "just compare on the keys we happen to have"
    was rejected: that would have trusted the outcome across such an edit.

    The simulated lineage is a WEAKER witness than the runtime's own, which
    folds in what the statement actually read. It is the same witness the
    upstream check already trusts for every name it models -- and for these
    names the alternative is not caution but the wrong answer: with nothing
    recorded, a loop reading one kept its table when that cell was edited to
    name a different file (`test_editing_the_cash_on_cell_still_invalidates
    _the_loop`). Filling the gap is strictly better than leaving it.

    *simulated* may be from an earlier cell if no upstream check ran for this
    one. Harmless in the direction that matters: a name whose simulated
    lineage has moved since produces a mismatch, which is what the code did
    unconditionally before.
    """
    entry = {n: lineage[n] for n in reads if n in lineage}
    if simulated:
        for name in reads:
            if name not in entry and name in simulated:
                entry[name] = simulated[name]
    return entry


def _holds_rng_state(value: Any) -> bool:
    """A generator object: drawing from it inside a loop changes it in place."""
    if isinstance(value, random.Random):
        return True
    module = type(value).__module__ or ""
    return module.startswith("numpy.random") and type(value).__name__ in ("Generator", "RandomState")


class ControlStructureProcessor:
    """
    Processor for control structures.

    **For loops** are decomposed per-iteration.  Each body statement is passed
    to the statement processor with an iteration-context comment injected into
    the code.  The statement processor's existing mutation detection decides
    per-statement whether caching is safe:

    - Statements that only *read* external variables and assign to local
      outputs are cached per-iteration (e.g. ``stats = expensive(...)``).
    - Statements that *mutate* external variables in-place (e.g.
      ``a.append(x)``, ``d[k] = v``) get ``skip_cache=True`` from the
      mutation detector and are executed directly every time.

    **If and try/except** are decomposed per-statement for correct caching
    and output handling.  **Other control structures** (while, with) are
    executed as single cacheable units through the statement processor.
    """

    def __init__(
        self,
        shell,
        statement_processor,  # The StatementProcessor instance
        debug: bool = False,
    ):
        self.shell = shell
        self.statement_processor = statement_processor
        self.debug = debug
        # Per-strategy handlers — constructed once.  Each owns the
        # strategy-specific logic; the orchestrator stays thin.
        self._for_handler = ForLoopHandler(shell, statement_processor, debug, dispatcher=self)
        self._if_handler = IfHandler(shell, statement_processor, debug, dispatcher=self)
        self._try_handler = TryHandler(shell, statement_processor, debug, dispatcher=self)

    def process(
        self,
        node: ast.AST,
        ttl: int | None = None,
        silent: bool = False,
        parent_context: dict[str, Any] | None = None,
        raw_cell: str | None = None,
        inherited_annotation: "CacheAnnotation | None" = None,
        prev_node: ast.stmt | None = None,
    ) -> ControlStructureResult:
        """
        Process a control structure node.

        For loops are decomposed per-iteration; other control structures are
        executed as single units.

        Args:
            node: The AST node representing the control structure
            ttl: Time-to-live for cache entries
            silent: Suppress output
            parent_context: Iteration context from an enclosing loop (for nesting)
            raw_cell: The cell's original source. Required to honour ``@cash:``
                directives inside the structure — ``ast.unparse`` drops comments,
                so a body statement's directive can only be recovered from the
                original text. ``None`` disables annotation handling,
                which is the pre-existing behaviour and keeps direct callers
                (tests constructing handlers with mock deps) working unchanged.
            inherited_annotation: Directives from enclosing structures, already
                resolved, to merge into everything within this one.
            prev_node: The immediately-preceding top-level statement in the same
                cell, or ``None``. Threaded down to ``ForLoopHandler`` (CAS-259
                follow-up), which needs the ``out = []`` seed that sits right
                before the loop to compute ``force_outputs`` for its cost-based
                single-unit branch — see ``cacheability.cacheable_accumulator_loop``
                and ``for_handler.py``'s single-unit branch for why. Additive and
                default-``None`` so nested / direct callers (which have no notion
                of a preceding sibling) are unchanged.

        Returns:
            ControlStructureResult with metrics
        """
        state = getattr(self.statement_processor, "tracking_state", None)
        outcomes = getattr(state, "control_outcomes", None)
        if parent_context is not None or not isinstance(outcomes, dict):
            return self._dispatch(node, ttl, silent, parent_context, raw_cell, inherited_annotation, prev_node)
        # Record what this structure left behind, for the simulation -- see
        # TrackingState.control_outcomes.
        lineage = state.variable_lineage
        code = ast.unparse(node)
        try:
            reads, writes = CodeAnalyzer.analyze_code_block(code)
        except (SyntaxError, ValueError, TypeError):
            reads, writes = set(), set()
        entry = _entry_lineages(reads, lineage, getattr(state, "simulated_lineage", None))
        before = dict(lineage)
        reads_before = dict(state.statement_file_reads)
        rng_before = _global_rng_fingerprint() if isinstance(node, ast.For) else None

        sp = self.statement_processor
        begin_cost = getattr(sp, "begin_structure_cost", None)
        if begin_cost is not None:
            begin_cost()
        result = None
        try:
            with observe_writes() as written:
                result = self._dispatch(node, ttl, silent, parent_context, raw_cell, inherited_annotation, prev_node)
        finally:
            if begin_cost is not None:
                changed = (
                    {v for v, h in lineage.items() if before.get(v) != h} | set(writes)
                    if result is not None and result.success
                    else set()
                )
                sp.end_structure_cost(reads, changed, result is not None and result.success)
        if result.success:
            self._record_writes(code, reads, written)
            left = {v: h for v, h in lineage.items() if before.get(v) != h or v in writes}
            # A file the body read changes nothing above, so the entry lineages
            # cannot see it: keep the files behind what it left, and their state.
            files: set[str] = set()
            for var in left:
                files.update(state.executed_file_deps.get(var, ()))
            for key, (local, _remote) in state.statement_file_reads.items():
                if reads_before.get(key, (None,))[0] is not local:
                    files.update(local)

            outcome = (entry, left, frozenset(files), compute_file_hash_component(files))
            outcomes[hashlib.sha256(code.encode("utf-8")).hexdigest()] = outcome
            # Judged only on a run that restored nothing: a restored statement
            # puts back the RNG state it was stored with, so a loop that draws
            # nothing still moves the generators when it hits in a new kernel.
            if not any(_status(m) == CacheStatus.RESTORED for m in result.metrics):
                self._persist_outcome(node, code, reads, before, outcome, rng_before)
                # What the loop read, for the planner of a later kernel -- as a
                # statement's reads are kept (``persist_read_provenance``). A
                # restored iteration records no read, hence the same condition.
                # ``for f in files: pd.read_csv(f)`` names no path a reader can
                # resolve, so without it the read set of every cell below was
                # unknown after a restart, no writer could be ruled out as
                # unread, and a table cell under a chart cell re-drew the charts
                # with everything they read (round 23, r23s2).
                self.statement_processor.persist_read_provenance(code, files)
        return result

    def _record_writes(self, code: str, reads, written: set[str]) -> None:
        """A structure that wrote files is a writer, as the simulation sees it.

        The simulation plans a loop as one statement, so that is where the
        planner looks for a writer's provenance. Each body statement ran on
        its own and knew its writes, but ``for kind in KINDS: save_chart(kind)``
        as a whole had none, and after a restart it was re-fired -- with
        everything it reads (round 23, r23s3: a 263 s sweep, to redraw four
        charts already on disk).
        """
        sp = self.statement_processor
        try:
            written = sp.user_written_paths(written)
            if not written:
                return
            sp.tracking_state.executed_write_stmt_codes.add(code)
            sp.persist_write_provenance(code, set(reads), None, written)
        except Exception:  # noqa: BLE001 - never let bookkeeping break the user's loop
            logger.debug("[CONTROL] write provenance failed", exc_info=True)

    def _persist_outcome(self, node, code, reads, before, outcome, rng_before) -> None:
        """Keep a loop's outcome for the simulation of a later kernel.

        ``control_outcomes`` dies with the kernel, and without it the
        simulation's lineages for what a loop built disagreed with the
        entries written from them: after a restart nothing downstream of a
        loop restored, and the loop ran again (round 23; see
        ``control_outcome_key``). A record is trusted instead of a replay, so
        it is written only for a loop whose outcome is all it did
        (``_persistable_callees``), together with the lineages of what its
        callees read. A loop that no longer qualifies deletes its record.
        Best-effort both ways: without a record the loop is replayed, as it
        always was.
        """

        sp = self.statement_processor
        backend = getattr(getattr(sp, "cash_instance", None), "backend", None)
        restorer = getattr(sp, "_stmt_restorer", None)
        if backend is None or restorer is None:
            return
        key = control_outcome_key(code)
        written = self.__dict__.setdefault("_outcomes_written", {})
        try:
            callees = self._persistable_callees(node, code, reads, before, rng_before)
            if callees is None:
                if written.get(key, True) is not None:
                    backend.delete(key)
                    written[key] = None
                return
            entry, left, files, file_component = outcome
            record = {
                "entry": entry,
                "callees": callees,
                "left": left,
                "files": sorted(files),
                "file_component": file_component,
            }
            if written.get(key) == record:
                return
            restorer.persist_metadata_only(backend, key, {"control_outcome": True, "code": code, "ttl": None, **record})
            written[key] = record
        except Exception:  # noqa: BLE001 - never let bookkeeping break the user's loop
            logger.debug("[CONTROL] control-outcome persistence failed", exc_info=True)

    def _persistable_callees(self, node, code, reads, before, rng_before) -> dict[str, str] | None:
        """What a later kernel must find unchanged to trust this loop's record, or None.

        None -- replay, never trust -- unless the loop's outcome is all it did:

        * a ``for`` loop (the shape that is expensive to replay);
        * the global RNG where it was: a draw is the loop's effect on every
          draw after it, and a record would skip it;
        * no file written, by its text or by a function it calls: skipping
          the loop would skip the write;
        * no clock or uuid read, in it or in a function it calls: its
          outcome is not a function of its inputs;
        * no environment read, in it or in a function it calls: a statement
          folds the value into its key and lineage, which a trusted record
          would skip;
        * no global mutated in place by a function it calls, and no RNG
          object read: effects the entry lineages do not show.

        Otherwise the lineages of every global its callees read, transitively
        -- an edited helper, even one called through another, has a new
        lineage, which the entry lineages alone do not name.
        """
        if not isinstance(node, ast.For) or rng_before is None:
            return None
        if rng_before != _global_rng_fingerprint():
            return None

        user_ns = self.shell.user_ns
        if statement_writes_files(code) or statement_calls_user_writer(code, user_ns):
            return None
        if any(_holds_rng_state(user_ns.get(name)) for name in reads):
            return None
        callee_names = called_function_globals(reads, user_ns)
        resolve = getattr(self.statement_processor, "_resolve_live_function_source", None)
        if resolve is None:
            return None
        for name in set(reads) | callee_names:
            if not isinstance(user_ns.get(name), types.FunctionType):
                continue
            source = resolve(name)
            if (
                source is None
                or CodeAnalyzer.scan_for_forbidden_functions(source, user_ns)
                or statement_environment_reads(source, user_ns)
            ):
                return None
        if CodeAnalyzer.scan_for_forbidden_functions(code, user_ns) or statement_environment_reads(code, user_ns):
            return None
        if callee_global_mutations(ast.parse(code), resolve):
            return None
        return {name: before.get(name, "ABSENT") for name in sorted(callee_names)}

    def _dispatch(
        self,
        node: ast.AST,
        ttl: int | None,
        silent: bool,
        parent_context: dict[str, Any] | None,
        raw_cell: str | None,
        inherited_annotation: "CacheAnnotation | None",
        prev_node: ast.stmt | None,
    ) -> ControlStructureResult:
        if isinstance(node, ast.For):
            # For loops with break/continue must be executed as single units
            if contains_break_or_continue(node.body):
                if self.debug:
                    logger.debug("[CONTROL] Loop contains break/continue, executing as single unit")
                return self.execute_as_single_unit(
                    node,
                    ttl,
                    silent,
                    raw_cell,
                    inherited_annotation,
                )
            return self._for_handler.process(
                node,
                ttl,
                silent,
                parent_context,
                raw_cell,
                inherited_annotation,
                prev_node,
            )
        if isinstance(node, ast.If):
            return self._if_handler.process(
                node,
                ttl,
                silent,
                raw_cell,
                inherited_annotation,
            )
        if isinstance(node, ast.Try):
            return self._try_handler.process(
                node,
                ttl,
                silent,
                raw_cell,
                inherited_annotation,
            )
        return self.execute_as_single_unit(
            node,
            ttl,
            silent,
            raw_cell,
            inherited_annotation,
        )

    # ------------------------------------------------------------------
    # Single-unit execution (for while/with and break/continue loops)
    # ------------------------------------------------------------------

    def execute_as_single_unit(
        self,
        node: ast.AST,
        ttl: int | None,
        silent: bool,
        raw_cell: str | None = None,
        inherited_annotation: "CacheAnnotation | None" = None,
        force_outputs: set[str] | None = None,
    ) -> ControlStructureResult:
        """
        Execute an entire control structure as a single unit.

        Used for while loops, with statements, and for loops that contain
        break/continue. The whole code is passed to the statement processor,
        which handles caching decisions (mutation detection, side-effect
        checks, etc.).

        The unit is ONE cache entry, so a ``@cash:`` directive anywhere inside
        it scopes to the whole thing — there is no finer entry for it to attach
        to. That is why this resolves the node's whole range rather than a
        per-statement annotation.

        *force_outputs* names extra variables the statement processor must
        capture/restore and treat as expected writes — the accumulator + leaked
        loop variable of an accumulator-loop fast path. ``None`` for every
        other single-unit structure, which keeps their behaviour unchanged.
        """
        try:
            code = ast.unparse(node)

            if self.debug:
                cs_type = get_control_structure_type(node)
                logger.debug("[CONTROL] Processing %s as single unit: %s...", cs_type, code[:80])

            annotation = _helpers.resolve_unit_annotation(
                raw_cell,
                node,
                inherited_annotation,
            )
            metrics = self.statement_processor.process_statement(
                code,
                ttl,
                silent,
                annotation=annotation,
                stream_output=True,
                force_outputs=force_outputs,
            )
            return self._finalize_single_unit(node, code, metrics)
        except Exception as e:  # noqa: BLE001 - broad fallback wrapping arbitrary user code executed as a unit
            # Handed back to the cell, which raises it: logged at ERROR it printed
            # the traceback a second time, through cash (round 25, r25s2/r25s3).
            logger.debug("[CONTROL] Error executing control structure as single unit: %s", e, exc_info=True)
            return ControlStructureResult(success=False, metrics=[], error=e)

    async def process_await_unit(
        self,
        node: ast.AST,
        ttl: int | None = None,
        silent: bool = False,
        raw_cell: str | None = None,
        inherited_annotation: "CacheAnnotation | None" = None,
    ) -> ControlStructureResult:
        """Run a control structure whose body contains a top-level ``await`` as
        ONE awaited unit.

        The per-iteration and sync single-unit paths compile body statements
        with an unflagged ``compile()``, which raises ``SyntaxError: 'await'
        outside function``.  This routes the whole structure through
        :meth:`StatementProcessor.process_statement_async`, whose compile carries
        ``PyCF_ALLOW_TOP_LEVEL_AWAIT`` and awaits the resulting coroutine on
        IPython's live loop — the same primitive the regular top-level-await
        statement path already uses.

        A whole-structure unit (never per-iteration) is the correct granularity
        here: an ``await`` is I/O, so per-iteration caching is inappropriate
        anyway, and the sync ``ControlStructureProcessor`` never reaches its
        break/continue-style per-iteration decomposition for these.
        """
        try:
            code = ast.unparse(node)
            if self.debug:
                cs_type = get_control_structure_type(node)
                logger.debug("[CONTROL] Processing %s as awaited single unit: %s...", cs_type, code[:80])

            annotation = _helpers.resolve_unit_annotation(
                raw_cell,
                node,
                inherited_annotation,
            )
            metrics = await self.statement_processor.process_statement_async(
                code,
                ttl,
                silent,
                annotation=annotation,
                stream_output=True,
            )
            return self._finalize_single_unit(node, code, metrics)
        except Exception as e:  # noqa: BLE001 - broad fallback wrapping arbitrary user code executed as a unit
            # Handed back to the cell, which raises it: logged at ERROR it printed
            # the traceback a second time, through cash (round 25, r25s2/r25s3).
            logger.debug("[CONTROL] Error executing awaited control structure as single unit: %s", e, exc_info=True)
            return ControlStructureResult(success=False, metrics=[], error=e)

    def _finalize_single_unit(
        self,
        node: ast.AST,
        code: str,
        metrics: "ProcessResult",
    ) -> ControlStructureResult:
        """Shared post-execution bookkeeping for a single-unit control structure.

        Called by BOTH the sync (:meth:`execute_as_single_unit`) and awaited
        (:meth:`process_await_unit`) paths so their lineage update, badge
        annotation, and clean-traceback line offset can never drift — the drift
        between a flagged and an unflagged compile path is exactly what produced it.
        """
        # After execution, update lineage for mutated variables
        if metrics.get("status") in (CacheStatus.COMPUTED, CacheStatus.RESTORED):
            _helpers.update_lineage_after_execution(
                self.shell,
                self.statement_processor,
                node,
                code,
                debug=self.debug,
            )

        # Annotate metrics with control structure body statements
        # so the badge can show individual statements instead of the
        # entire block as one opaque line.
        cs_type = get_control_structure_type(node)
        metrics["control_type"] = cs_type
        body_stmts = _helpers.extract_body_statements(node)
        if body_stmts:
            metrics["body_statements"] = body_stmts

        # Extract error and annotate with line info for clean traceback.
        # For single-unit control structures, the <cash> frame has a line
        # number relative to the unparsed code.  We need to offset it by
        # the node's starting line in the cell so show_clean_error points
        # to the correct cell line.
        error = metrics.get("error") if metrics.get("status") == CacheStatus.ERROR else None
        if error is not None:
            # Try to extract the actual error line from the <cash> traceback
            cash_lineno = _helpers.extract_cash_frame_lineno(error)
            if cash_lineno is not None:
                # ast.unparse produces code starting at line 1;
                # the node in the cell starts at node.lineno.
                cell_lineno = getattr(node, "lineno", 1) + cash_lineno - 1
                with contextlib.suppress(AttributeError, TypeError):
                    error._cash_error_lineno = cell_lineno

        return ControlStructureResult(
            success=metrics.get("status") != CacheStatus.ERROR,
            metrics=[metrics],
            error=error,
            total_iterations=1,
            cached_iterations=1 if metrics.get("status") == CacheStatus.RESTORED else 0,
            computed_iterations=1 if metrics.get("status") == CacheStatus.COMPUTED else 0,
        )
