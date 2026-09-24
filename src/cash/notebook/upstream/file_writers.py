"""Upstream statements that write files, and whether what they wrote is current.

A file write binds no variable, so the backward scan never schedules a writer.
:class:`FileWriterScheduler` finds the writers whose output is out of date --
edited or never run this session, or fed by something the plan re-runs -- and
schedules them for the files the checked cell depends on.
"""

from __future__ import annotations

import ast
import logging
import os
import stat
import textwrap
import types
from typing import TYPE_CHECKING

from ..._paths import resolve_file_dep_path
from ...analysis.ast_util import called_names
from ...analysis.cacheability import statement_writes_files
from ...analysis.file_effects import REPEATABILITY_ACCUMULATING, REPEATABILITY_REPLACING, statement_write_repeatability
from ...analysis.namespace_effects import (
    resolve_literal_path,
    statement_calls_user_writer,
    statement_written_paths,
)
from ...tracking.file_dep_snapshot import snapshot_is_fresh
from .._trace import trace_event
from ..cache_key import called_function_globals, write_provenance_key
from ..cache_status import CacheStatus
from ..carrier_history import carrier_history_fingerprint
from .virtual_lineage import key_lineages

if TYPE_CHECKING:
    from .virtual_lineage import VirtualLineage

__all__ = ["FileWriterScheduler"]

logger = logging.getLogger(__name__)


def _literal_path_bindings(simulation_trace: list | None) -> dict[str, str]:
    """``{name: path}`` for names the notebook binds to one literal path.

    ``OUT = Path('report')``, ``EXPORTS = BASE / 'exports'``: what a writer's
    ``OUT / 'chart.png'`` means when the kernel does not hold ``OUT`` yet.
    A name bound more than once, or by anything else, is left out.
    """

    bound: dict[str, str | None] = {}
    for entry in simulation_trace or ():
        outputs = entry.outputs
        if not outputs:
            continue
        value = None
        try:
            node = ast.parse(entry.stmt_code).body
        except (SyntaxError, ValueError, TypeError):
            node = []
        if (
            len(node) == 1
            and isinstance(node[0], ast.Assign)
            and len(node[0].targets) == 1
            and isinstance(node[0].targets[0], ast.Name)
        ):
            known = {k: v for k, v in bound.items() if v is not None}
            value = resolve_literal_path(node[0].value, known)
        for name in outputs:
            bound[name] = value if name not in bound or bound[name] == value else None
    return {name: path for name, path in bound.items() if path is not None}


def _only_defines(code: str) -> bool:
    """True when *code* only defines functions or classes."""
    try:
        body = ast.parse(textwrap.dedent(code)).body
    except (SyntaxError, ValueError, TypeError):
        return False
    return bool(body) and all(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in body)


class FileWriterScheduler:
    """Schedules the upstream file writers whose output is out of date."""

    def __init__(self, virtual_lineage: VirtualLineage) -> None:
        self.virtual_lineage = virtual_lineage
        self.tracking_state = virtual_lineage.tracking_state
        #: ``(read paths, their count, index)`` -- see ``_read_path_index``.
        self._read_index: tuple[set[str], int, tuple[set[str], list[str], list[str]]] | None = None
        #: ``(trace, its length, defs)`` -- see ``_trace_defs``. Holds the
        #: trace itself, so an identity match cannot be a reused id.
        self._trace_defs_memo: tuple[list, int, dict] | None = None

    def _user_ns(self) -> dict:
        return self.virtual_lineage.shell.user_ns

    def schedule(
        self,
        stmts_to_run_indices: list[int],
        simulation_trace: list,
        restored_statements_info: list[dict],
        broken_vars: set[str] | None = None,
        virtual_lineage: dict | None = None,
        relevant_read_paths: set[str] | None = None,
        relevant_read_paths_known: bool = True,
        stale_exports: list | None = None,
    ) -> tuple[list[int], list[dict]]:
        """Schedule upstream file-WRITING statements whose effect is stale.

        File writes have no variable edge, so the backward scan never
        schedules them: editing a writer cell and re-running only the reader
        served the stale pre-edit file; and when a writer cell's
        sibling statements DID re-run, the side-effect-only write statement
        was still skipped and the reader's freshness stayed decided against
        the pre-run file state.

        A writer statement is scheduled when

        * its code was never executed in this session (edited or new — the
          runtime records every executed file-writing statement's code in
          ``executed_write_stmt_codes``), or
        * any of its inputs is produced by an already-scheduled statement
          (its payload changed).

        Restored statements positioned after the first scheduled writer whose
        variables carry file dependencies are PROMOTED to re-execution: their
        plan-time restore was validated against the pre-write file state, and
        re-executing them in trace order (after the writer) overwrites the
        stale restore with a fresh read.

        A writer that is not scheduled but whose file is out of date anyway
        is added to *stale_exports* (see :meth:`find_stale_file_writer_indices`).
        """
        scheduled = set(stmts_to_run_indices)
        scheduled_outputs: set[str] = set()
        for idx in scheduled:
            scheduled_outputs.update(simulation_trace[idx].outputs)
        # A writer whose input is BROKEN is also stale, even when nothing in
        # the variable plan demands that input (the write may be the only
        # consumer — e.g. an edited payload that exists just to be dumped).
        changed_inputs = scheduled_outputs | set(broken_vars or ())

        runtime_lineage = self.tracking_state.variable_lineage
        # Live kernel namespace, to tell "provably unchanged" apart from
        # "absent". A writer input that is gone from user_ns (kernel restart,
        # ``del``, or an isolated re-run whose producer never ran this session)
        # has no runtime lineage -- but neither does an unchanged one, so the
        # lineage comparison below reads absent-as-unchanged and never schedules
        # the input's producer. That left ``df.to_csv(path)`` scheduled WITHOUT
        # its producer, so it ran against a missing ``df`` and raised NameError
        # post-restart, poisoning the whole notebook.
        user_ns = self._user_ns()

        def _input_changed(name: str) -> bool:
            if name in changed_inputs:
                return True
            if user_ns is not None and name not in user_ns:
                # Genuinely absent from the live namespace: schedule its producer
                # so the writer runs against a re-materialised (cache-restored or
                # recomputed) input rather than crashing -- exactly what run_all
                # already does for the same notebook.
                return True
            if virtual_lineage is None:
                return False
            virt = virtual_lineage.get(name)
            run = runtime_lineage.get(name)
            return virt is not None and run is not None and virt != run

        writer_indices = self.find_stale_file_writer_indices(
            simulation_trace,
            scheduled_outputs=changed_inputs,
            skip=scheduled,
            virtual_lineage=virtual_lineage,
            relevant_read_paths=relevant_read_paths,
            relevant_read_paths_known=relevant_read_paths_known,
            stale_exports=stale_exports,
        )
        if not writer_indices:
            return stmts_to_run_indices, restored_statements_info

        writer_indices = self._whole_cell_writers(simulation_trace, writer_indices, scheduled)
        scheduled.update(writer_indices)
        first_writer = min(writer_indices)

        def _rebound_after(name: str, w: int) -> int | None:
            for j in range(len(simulation_trace) - 1, w, -1):
                if name in simulation_trace[j].outputs:
                    return j
            return None

        # Re-materialise a scheduled writer's changed inputs: their producers
        # may be missing from the plan when the write is the only consumer.
        #
        # An input the trace binds AGAIN after the writer is changed too, for
        # this writer: the live value is the later binding. A chart cell that
        # reuses `fig, axes = plt.subplots(...)` for a second figure re-ran the
        # first figure's `save(fig, ...)` and its draws on the second figure's
        # axes. Its producer before the writer re-runs, and so
        # does the last one, so the name ends bound as the cell leaves it.
        pending = list(writer_indices)
        while pending:
            w = pending.pop()
            for v in self._writer_inputs(simulation_trace[w].inputs, simulation_trace):
                later = _rebound_after(v, w)
                if later is None and not _input_changed(v):
                    continue
                for prod in range(w - 1, -1, -1):
                    if v in simulation_trace[prod].outputs:
                        if prod not in scheduled:
                            scheduled.add(prod)
                            pending.append(prod)
                            logger.debug(
                                "[UPSTREAM] Scheduling producer [%s] of writer input '%s'",
                                prod,
                                v,
                            )
                        break
                if later is not None and later not in scheduled:
                    scheduled.add(later)
                    pending.append(later)

        # Every statement AFTER a scheduled writer whose variables carry file
        # dependencies must re-execute: its cached value (whether restored at
        # plan time or simply sitting untouched in memory) was validated
        # against the PRE-write file state. Re-execution happens in trace
        # order, after the writer, so its freshness is decided against the
        # freshly written file.
        executed_file_deps = self.tracking_state.executed_file_deps
        # Only what depends on a file a scheduled writer WRITES. "Any file
        # dependency" promoted nearly everything after the writer, because
        # recorded dependencies include inherited ones: saving a cleaned copy of
        # the data re-ran the backtest and the forecast behind it for a cell that
        # read only the copy.
        # A writer whose path does not resolve keeps the broad rule.
        written_forms = self._written_path_forms(simulation_trace, writer_indices)

        def _reads_written(outputs) -> bool:
            deps = set()
            for v in outputs:
                dep = executed_file_deps.get(v)
                if dep:
                    deps.update(dep.keys() if hasattr(dep, "keys") else dep)
            if not deps:
                return False
            if written_forms is None:
                return True
            return any(self._normalize_path_forms(d) & written_forms for d in deps)

        promoted: set[int] = set()
        promoted_outputs: set[str] = set()
        for i in range(first_writer + 1, len(simulation_trace)):
            if i in scheduled:
                continue
            outputs, inputs = simulation_trace[i].outputs, simulation_trace[i].inputs
            # ...and, in trace order, whatever consumes a re-read value.
            if _reads_written(outputs) or (written_forms is not None and set(inputs) & promoted_outputs):
                scheduled.add(i)
                promoted.add(i)
                promoted_outputs.update(outputs)
                logger.debug(
                    "[UPSTREAM] Promoting file-reader [%s] to re-exec (writer scheduled at [%s]): %s",
                    i,
                    first_writer,
                    simulation_trace[i].stmt_code[:40],
                )

        if promoted:
            promoted_codes = {simulation_trace[i].stmt_code for i in promoted}
            restored_statements_info = [
                info for info in restored_statements_info if info.get("code") not in promoted_codes
            ]

        return sorted(scheduled), restored_statements_info

    def _whole_cell_writers(self, simulation_trace: list, writer_indices, scheduled) -> list[int]:
        """*writer_indices* plus every other file writer in the same cells.

        A cell's writes are replayed together or not at all. Re-running only
        the stale ones left states no run order produces: a report folder
        whose grid came from the new models and whose ROC chart from the old,
        or -- when ``shutil.rmtree`` re-ran and the loop that refills the
        folder did not -- charts that were simply gone.
        Running the cell writes every one of its files; so does this.

        Only writes that provably REPLACE their file are pulled in: they land
        the same bytes when repeated. A write that is not stale itself and may
        append -- ``os.write(fd, ...)`` on a descriptor opened elsewhere -- is
        left to run when its own cell runs; re-firing it here duplicated a
        counter line on every replay. The exception is a cell
        whose replay already re-fires such a write -- a ``shutil.rmtree`` --
        where everything but a provable append follows it, or ``PACK.mkdir()``
        stays behind and the next write finds no folder.
        """

        cells = {simulation_trace[w].cell for w in writer_indices} - {-1}
        if not cells:
            return list(writer_indices)
        # Cells whose replay already re-fires a write that is not provably
        # replacing (``rmtree``, ``mkdir``): the rest of their writes follow it.
        destructive = {
            simulation_trace[w].cell
            for w in writer_indices
            if statement_write_repeatability(simulation_trace[w].stmt_code) != REPEATABILITY_REPLACING
        }
        extra = []
        chosen = set(writer_indices)
        for j, entry in enumerate(simulation_trace):
            if j in chosen or j in scheduled or entry.cell not in cells:
                continue
            code = entry.stmt_code
            if not self._is_file_writer(code, simulation_trace):
                continue
            verdict = statement_write_repeatability(code)
            if verdict == REPEATABILITY_ACCUMULATING:
                continue
            if verdict != REPEATABILITY_REPLACING and entry.cell not in destructive:
                continue
            extra.append(j)
            logger.debug(
                "[UPSTREAM] Scheduling same-cell file-writer [%s] with its cell's stale writers: %s",
                j,
                code[:60],
            )
        return sorted(chosen | set(extra))

    def find_stale_file_writer_indices(
        self,
        simulation_trace: list,
        scheduled_outputs: frozenset | set = frozenset(),
        skip: frozenset | set = frozenset(),
        virtual_lineage: dict | None = None,
        relevant_read_paths: set[str] | None = None,
        relevant_read_paths_known: bool = True,
        stale_exports: list | None = None,
    ) -> list[int]:
        """Trace indices of file-WRITING statements whose effect is stale.

        A writer is stale when its code was never executed in this session
        (edited/new — the runtime records executed file-writing statements'
        code in ``executed_write_stmt_codes``) or when one of its inputs is
        produced by an already-scheduled statement. Also used by the
        simulator to gate its no-broken-vars early return, so the common
        path stays cheap: a textual marker pre-filter runs before any AST
        analysis.

        Scoped to the current cell: a writer whose resolvable
        output path is read by NO consumer relevant to this reconstruction
        (``relevant_read_paths``) is an unrelated / terminal side-effect and is
        never scheduled — re-firing it can only repeat an external write (a
        non-idempotent ``mode='a'`` append corrupts the file) without helping
        reconstruct any value the current cell needs. The gate applies only when
        the read set is fully known and the writer's own path resolves; every
        uncertain case falls through to the prior (conservative) behaviour.

        A writer the scope gate leaves alone although what it writes has
        changed -- an input's lineage drifted from the one it was written
        with -- is appended to *stale_exports* as ``(index, paths)``: the
        file on disk is now out of date, and the badge must not call it
        current.
        """
        executed_writes = self.tracking_state.executed_write_stmt_codes
        runtime_lineage = self.tracking_state.variable_lineage

        def _input_lineage_drifted(name: str) -> bool:
            # The writer's payload changed even though nothing in the variable
            # plan demands it: the simulated (current-code) lineage differs
            # from the lineage last seen at runtime.
            if virtual_lineage is None:
                return False
            virt = virtual_lineage.get(name)
            run = runtime_lineage.get(name)
            return virt is not None and run is not None and virt != run

        writer_indices: list[int] = []
        for i, entry in enumerate(simulation_trace):
            if i in skip:
                continue
            stmt_code, _outputs, inputs = entry.stmt_code, entry.outputs, entry.inputs
            if not self._is_file_writer(stmt_code, simulation_trace):
                continue
            inputs = self._writer_inputs(inputs, simulation_trace)
            # Scope gate: skip a writer whose output file no relevant consumer
            # reads. Its write runs when the user runs its own
            # cell; reconstruction of an unrelated cell must never re-fire it.
            unread = self._writer_output_unread(
                stmt_code,
                relevant_read_paths,
                relevant_read_paths_known,
                simulation_trace,
            )
            trace_event(
                "writer_considered",
                stmt=stmt_code[:80],
                unread=unread,
                read_paths_known=relevant_read_paths_known,
                read_paths=sorted(relevant_read_paths or ())[:20],
            )
            if unread:
                if stale_exports is not None:
                    self._note_if_stale(
                        stale_exports, i, stmt_code, inputs, virtual_lineage, runtime_lineage, simulation_trace
                    )
                logger.debug(
                    "[UPSTREAM] File-writer output read by no relevant consumer; not scheduling (scope): %s",
                    stmt_code[:60],
                )
                continue
            # Repeatability gate. The scope gate above keys on
            # RELEVANCE -- "does a relevant consumer read this file?" -- not on
            # whether repeating the write is safe. Those coincide in the cases
            # it was built for, which is why it reads as a correctness
            # guarantee; they come apart for an APPEND whose file is also read.
            # There the gate steps aside and the re-fire silently duplicates the
            # payload on disk, damage a kernel restart cannot undo.
            #
            # Not re-firing is also what the user's own kernel does: the write
            # runs when they run its own cell, and they did not run it here.
            #
            # ONLY a PROVABLE append is refused. The evidence also suggests refusing
            # UNKNOWN repeatability, and that was measured rather than reasoned
            # about: the two failure modes are not symmetric. Refusing a write
            # that SHOULD re-fire leaves the reader on stale data every time,
            # silently, on a common path -- the measurement broke two
            # tests exactly that way. Leaving an unprovable append re-firing
            # costs a duplicated line in an uncommon shape. Narrow is the right
            # side of that trade; see the recorded measurement.
            if statement_write_repeatability(stmt_code) == REPEATABILITY_ACCUMULATING:
                logger.debug(
                    "[UPSTREAM] File-writer is a non-idempotent append; not re-firing (repeatability): %s",
                    stmt_code[:60],
                )
                continue
            changed = stmt_code not in executed_writes
            scheduled_inputs = set(inputs) & scheduled_outputs
            drifted = any(_input_lineage_drifted(v) for v in inputs)
            inputs_changed = bool(scheduled_inputs) or drifted
            # ``changed`` fires for every writer after a kernel restart because
            # ``executed_write_stmt_codes`` is session-scoped and starts empty.
            # Before re-firing such a writer (which would re-run its
            # non-idempotent side effect and re-derive stale data), check its
            # persisted provenance: if the payload is unchanged AND the output
            # file is still fresh on disk, the effect is already applied — do
            # NOT schedule it.
            #
            # An input whose producer is scheduled is answered the same way. A
            # producer is scheduled to REBUILD a value as often as to change it
            # -- after a restart, everything the cell needs is -- and the
            # provenance tells the two apart: the lineage the input had when
            # the file was written against the lineage the simulation gives it
            # now, which an upstream edit changes. Without this every writer
            # above a restarted cell re-fired, with everything it reads.
            # A lineage that drifted from the
            # runtime's (an unsaved edit) always re-runs.
            if (
                (changed or scheduled_inputs)
                and not drifted
                and self._writer_output_already_fresh(
                    stmt_code,
                    inputs,
                    virtual_lineage,
                    runtime_lineage,
                    must_cover=scheduled_inputs,
                    simulation_trace=simulation_trace,
                    index=i,
                )
            ):
                changed = inputs_changed = False
                logger.debug(
                    "[UPSTREAM] File-writer effect already fresh on disk; not re-firing: %s",
                    stmt_code[:60],
                )
            trace_event(
                "writer_decided",
                stmt=stmt_code[:80],
                refire=changed or inputs_changed,
                changed=changed,
                scheduled_inputs=sorted(scheduled_inputs),
                drifted=drifted,
            )
            if changed or inputs_changed:
                writer_indices.append(i)
                logger.debug(
                    "[UPSTREAM] File-writer scheduling: [%s] %s (changed=%s, inputs_changed=%s)",
                    i,
                    stmt_code[:40],
                    changed,
                    inputs_changed,
                )
        return writer_indices

    def _note_if_stale(
        self,
        stale_exports: list,
        i: int,
        stmt_code: str,
        inputs,
        virtual_lineage: dict | None,
        runtime_lineage: dict,
        simulation_trace: list,
    ) -> None:
        """Add writer *i* to *stale_exports* when what it recorded as it wrote
        says its data has changed since: an input's lineage, or for a figure
        what was drawn into it. Not the drift of its inputs at the end of the
        simulation: that missed a chart drawn through ``for ax in axes`` --
        ``fig`` itself never changes -- and every chart after a restart, which
        has no runtime lineage to drift from. A
        folder has no content to be out of date."""
        reason = self._writer_not_fresh_because(
            stmt_code, inputs, virtual_lineage, runtime_lineage, simulation_trace=simulation_trace, index=i
        )
        if reason not in self._DATA_CHANGED:
            return
        paths = sorted(p for p in (self._writer_paths(stmt_code, simulation_trace) or ()) if not os.path.isdir(p))
        if paths:
            stale_exports.append((i, paths))

    def note_stale_exports(
        self,
        simulation_trace: list,
        stmts_to_run_indices: list[int],
        restored_statements_info: list[dict],
        stale_exports: list[tuple[int, list[str]]],
    ) -> list[dict]:
        """Mark the writers this repair left out of date (see
        :meth:`find_stale_file_writer_indices`) as such, in place of the
        "already current" skipped row they would otherwise get."""
        stale = {i: paths for i, paths in stale_exports if i not in set(stmts_to_run_indices)}
        if not stale:
            return restored_statements_info
        kept = [
            m
            for m in restored_statements_info
            if m.get("position") not in stale or str(m.get("status")) != str(CacheStatus.SKIPPED)
        ]
        for i, paths in sorted(stale.items()):
            trace_event("stale_export", stmt=simulation_trace[i].stmt_code[:80], paths=paths)
            kept.append(
                {
                    "code": simulation_trace[i].stmt_code,
                    "status": CacheStatus.SKIPPED,
                    "saved_time": 0.0,
                    "is_upstream": True,
                    "source": "Skipped",
                    "position": i,
                    "has_cache": False,
                    "stale_export": True,
                    "written_paths": paths,
                }
            )
        return kept

    @staticmethod
    def _called_names(code: str) -> set[str]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set()
        return set(called_names(tree))

    def _trace_defs(self, simulation_trace: list | None) -> dict:
        """``{name: trace entry}`` of the last ``def`` binding each name.

        After a kernel restart a helper is not defined yet, so its body can
        only be read from the notebook -- the ``def`` statement in the trace,
        whose inputs are the globals the body reads."""
        memo = self._trace_defs_memo
        if memo is not None and memo[0] is simulation_trace and memo[1] == len(simulation_trace or ()):
            return memo[2]
        defs: dict = {}
        self._trace_defs_memo = (simulation_trace, len(simulation_trace or ()), defs)
        for entry in simulation_trace or ():
            code = entry.stmt_code.lstrip()
            if not code.startswith(("def ", "async def ", "@")):
                continue
            try:
                node = ast.parse(entry.stmt_code).body[0]
            except (SyntaxError, IndexError):
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs[node.name] = entry
        return defs

    def _unbound_helpers(self, names, defs: dict) -> list:
        """The ``def`` entries for *names* that are not live functions."""
        user_ns = self._user_ns() or {}
        return [defs[n] for n in names if n in defs and not isinstance(user_ns.get(n), types.FunctionType)]

    def _is_file_writer(self, stmt_code: str, simulation_trace: list | None = None) -> bool:
        """A statement that writes files -- in its own text, or through a user
        function it calls (``save_png(kind, path)``, whose ``savefig`` sits in
        the helper). A replay that re-ran a cell's inline writes but not its
        helper's left the report folder half old, half new."""

        if _only_defines(stmt_code):
            # ``def save_page(...)`` writes nothing when it runs; its callers do,
            # and they are writers above. Taken for one, it had no provenance to
            # vouch for it after a restart, was re-fired, and pulled every write
            # of its cell along.
            return False
        if statement_writes_files(stmt_code):
            return True
        if statement_calls_user_writer(stmt_code, self._user_ns()) is not None:
            return True
        defs = self._trace_defs(simulation_trace)
        seen: set[str] = set()
        pending = self._unbound_helpers(self._called_names(stmt_code), defs)
        while pending:
            entry = pending.pop()
            code = entry.stmt_code
            if code in seen:
                continue
            seen.add(code)
            if statement_writes_files(code) and statement_write_repeatability(code) != REPEATABILITY_ACCUMULATING:
                return True
            pending.extend(self._unbound_helpers(self._called_names(code), defs))
        return False

    def _writer_inputs(self, inputs, simulation_trace: list | None = None) -> set[str]:
        """A writer's inputs plus the notebook globals its callees read: the
        helper that plots ``scores`` depends on ``scores`` though the call
        site never names it."""

        user_ns = self._user_ns()
        names = set(inputs)
        if user_ns:
            names |= called_function_globals(names, user_ns)
        defs = self._trace_defs(simulation_trace)
        pending = self._unbound_helpers(names, defs)
        seen: set[str] = set()
        while pending:
            entry = pending.pop()
            if entry.stmt_code in seen:
                continue
            seen.add(entry.stmt_code)
            new = set(entry.inputs) - names
            names |= new
            pending.extend(self._unbound_helpers(new, defs))
        return names

    def _written_path_forms(self, simulation_trace: list, writer_indices) -> set[str] | None:
        """Comparable forms of every path the writers write, or ``None`` if any
        writer's output does not resolve."""
        forms: set[str] = set()
        for w in writer_indices:
            written = self._writer_paths(simulation_trace[w].stmt_code, simulation_trace)
            if not written:
                return None
            for p in written:
                forms |= self._normalize_path_forms(p)
        return forms

    def _writer_paths(self, stmt_code: str, simulation_trace: list | None = None) -> set[str] | None:
        """The paths a writer writes: from its code, else from the record it
        left when it last ran.

        After a kernel restart ``OUT`` in ``OUT / 'table.csv'`` is not bound
        yet, so the code alone no longer resolves -- and every writer looked
        like it might feed the cell being run. A name the notebook binds to a
        literal path (``OUT = Path('report')``) resolves from *simulation_trace*;
        a folder removed by ``shutil.rmtree(OUT)`` leaves no record to fall back on.
        """
        user_ns = self._user_ns()
        namespace = {**_literal_path_bindings(simulation_trace), **(user_ns or {})}
        written = statement_written_paths(stmt_code, namespace=namespace)
        if written:
            return written
        backend = self.virtual_lineage.backend()
        if backend is None:
            return None
        try:
            record = backend.get_metadata(write_provenance_key(stmt_code))
        except (OSError, TypeError, ValueError, AttributeError):
            return None
        if not record or not record.get("write_provenance") or not record.get("paths"):
            return None
        return set(record["paths"])

    @staticmethod
    def _normalize_path_forms(path: str) -> set[str]:
        """Comparable forms of a file path: resolved-abspath (normcased) + basename.

        Both sides of the writer-output vs relevant-reads comparison are reduced
        to this set so a cwd-relative write (``'audit.log'``) and an absolute read
        of the same file compare equal, while a basename match keeps the gate
        biased toward NOT suppressing (a false "read" only foregoes a suppression;
        a false "unread" would wrongly drop a needed write).
        """
        forms: set[str] = set()
        try:
            resolved = resolve_file_dep_path(path)
        except (OSError, ValueError, TypeError):
            resolved = None
        for candidate in (resolved, path):
            if not candidate:
                continue
            try:
                forms.add(os.path.normcase(os.path.abspath(candidate)))
                forms.add(os.path.normcase(os.path.basename(candidate)))
            except (OSError, ValueError, TypeError):
                continue
        return forms

    def _read_path_index(self, relevant_read_paths) -> tuple[set[str], list[str], list[str]]:
        """``(comparable forms, folders listed, places)`` of the paths read, once
        per set of read paths rather than once per writer: each is a resolve and
        an ``isdir``, and 12 writers over 1,312 read files made 44,000 of them
        before one restarted cell. The simulator and the planner ask
        with the same set in one check, so it is kept across passes too; one
        notebook's 10,000 documents were indexed twice per cell, 1.8 s, and each resolved
        twice."""
        cached = self._read_index
        if cached is not None and cached[0] is relevant_read_paths and cached[1] == len(relevant_read_paths):
            return cached[2]
        read_forms: set[str] = set()
        read_dirs: list[str] = []
        read_places: list[str] = []
        for rp in relevant_read_paths:
            # One stat answers both "is it there" and "is it a folder" for a
            # path that has not moved; only a missing one takes the relocation
            # fallbacks and a second look.
            try:
                is_dir = stat.S_ISDIR(os.stat(rp).st_mode)
                resolved = rp
            except (OSError, ValueError, TypeError):
                try:
                    resolved = resolve_file_dep_path(rp) or rp
                except (OSError, ValueError, TypeError):
                    resolved = rp
                is_dir = resolved != rp and os.path.isdir(resolved)
            for candidate in {resolved, rp}:
                try:
                    read_forms.add(os.path.normcase(os.path.abspath(candidate)))
                    read_forms.add(os.path.normcase(os.path.basename(candidate)))
                except (OSError, ValueError, TypeError):
                    continue
            read_places.append(os.path.normcase(os.path.abspath(resolved)))
            if is_dir:
                # A listed / globbed folder: whatever is written inside it is
                # read by the next listing.
                read_dirs.append(os.path.normcase(os.path.abspath(resolved)) + os.sep)
        index = (read_forms, read_dirs, read_places)
        self._read_index = (relevant_read_paths, len(relevant_read_paths), index)
        return index

    def _writer_output_unread(
        self,
        stmt_code: str,
        relevant_read_paths: set[str] | None,
        relevant_read_paths_known: bool,
        simulation_trace: list | None = None,
    ) -> bool:
        """True when a writer's output file is read by no relevant consumer.

        The scope gate. Conservative in every uncertain
        case — returns ``False`` (do not suppress) unless the read set is fully
        known AND the writer's output path(s) all resolve statically AND none of
        them match any relevant read path. Then the writer is an unrelated /
        terminal side-effect for this cell and must not be re-fired.
        """
        if not relevant_read_paths_known or relevant_read_paths is None:
            return False
        written = self._writer_paths(stmt_code, simulation_trace)
        if not written:
            return False  # unresolvable target -> stay conservative
        read_forms, read_dirs, read_places = self._read_path_index(relevant_read_paths)
        for wp in written:
            if self._normalize_path_forms(wp) & read_forms:
                return False  # this output IS read by a relevant consumer
            where = os.path.normcase(os.path.abspath(resolve_file_dep_path(wp) or wp))
            if any(where.startswith(d) for d in read_dirs):
                return False
            # A folder made or removed (``OUT.mkdir()``): read by whatever
            # reads a file inside it.
            if any(place.startswith(where + os.sep) for place in read_places):
                return False
        return True

    def _writer_output_already_fresh(
        self,
        stmt_code: str,
        inputs,
        virtual_lineage: dict | None,
        runtime_lineage: dict,
        must_cover: set[str] | frozenset[str] = frozenset(),
        simulation_trace: list | None = None,
        index: int | None = None,
    ) -> bool:
        """True when a writer's effect is already on disk and provably current.

        *must_cover*: inputs the record has to have a lineage for -- ones
        whose producer is being re-run, which it can vouch for only if it
        knows what they were.

        *simulation_trace* / *index*: where the writer is in the simulation. Its
        inputs are compared with the lineages they have THERE, which is what
        the runtime recorded; the end of the simulation is later, and a
        ``plt.close(fig)`` or a second chart after the write had moved ``fig``
        on by then. A figure is compared by its drawing history instead of its
        lineage (``carrier_history``).

        Consulted only for a writer that looks ``changed`` purely because its
        code was never seen THIS session (the post-restart case). Returns True —
        meaning "do not re-fire" — only when the persisted provenance for this
        exact writer source is present, EVERY recorded output path is still fresh
        (:func:`snapshot_is_fresh`), AND every recorded input lineage still
        matches the writer's current (simulated / runtime) lineage.

        Conservative in every uncertain case: no backend, missing provenance, an
        unreadable / stale output file, or a drifted input lineage all return
        False, so the writer is scheduled exactly as before.
        """
        return (
            self._writer_not_fresh_because(
                stmt_code, inputs, virtual_lineage, runtime_lineage, must_cover, simulation_trace, index
            )
            is None
        )

    #: Reasons `_writer_not_fresh_because` gives that mean the file on disk was
    #: written from data that has since changed -- the rest mean only that
    #: nothing vouches for it.
    _DATA_CHANGED = frozenset({"input changed", "figure drawn differently"})

    def _writer_not_fresh_because(
        self,
        stmt_code: str,
        inputs,
        virtual_lineage: dict | None,
        runtime_lineage: dict,
        must_cover: set[str] | frozenset[str] = frozenset(),
        simulation_trace: list | None = None,
        index: int | None = None,
    ) -> str | None:
        """Why a writer's file is not provably current, or ``None`` when it is.
        See :meth:`_writer_output_already_fresh`."""

        def stale(reason: str, **detail) -> str:
            trace_event("writer_not_fresh", stmt=stmt_code[:80], reason=reason, **detail)
            return reason

        backend = self.virtual_lineage.backend()
        if backend is None:
            return "no backend"
        try:
            record = backend.get_metadata(write_provenance_key(stmt_code))
        except (OSError, TypeError, ValueError, AttributeError):
            return "no provenance"
        if not record or not record.get("write_provenance"):
            return stale("no provenance")
        paths = record.get("paths") or []
        file_deps = record.get("file_deps") or {}
        if not paths or not file_deps:
            return stale("no files recorded")
        missing = [path for path in paths if not file_deps.get(path)]
        if missing:
            return stale("file not recorded", path=missing[0])
        fresh, changed = snapshot_is_fresh({path: file_deps[path] for path in paths})
        if not fresh:
            return stale("file changed", path=changed.path)
        # The output on disk is only the writer's CURRENT output if its inputs
        # still carry the lineage they had when it was written. A drift means
        # the file was produced from a now-stale payload.
        stored_lineages = record.get("input_lineages") or {}
        if not set(must_cover) <= set(stored_lineages):
            return stale("input not recorded", inputs=sorted(set(must_cover) - set(stored_lineages)))
        entry = simulation_trace[index] if simulation_trace is not None and index is not None else None
        histories = record.get("carrier_histories") or {}
        for var, stored_lineage in stored_lineages.items():
            if var in histories:
                if entry is None or histories[var] != self._carrier_history_at(simulation_trace, index, var):
                    return stale("figure drawn differently", var=var)
                continue
            current = None
            if entry is not None:
                # After the writer ran: what the runtime recorded.
                current = (
                    entry.produced_lineages.get(var)
                    if var in entry.outputs
                    else key_lineages(entry.input_hashes).get(var)
                )
            if current is None:
                current = (virtual_lineage or {}).get(var)
            if current is None:
                current = runtime_lineage.get(var)
            if current != stored_lineage:
                return stale("input changed", var=var)
        return None

    @staticmethod
    def _carrier_history_at(simulation_trace: list, index: int, carrier: str) -> str | None:
        """The history fingerprint of figure *carrier* at the writer at *index*,
        from the statements of the writer's cell that come before it -- the same
        span the runtime took (``StatementProcessor._carrier_histories``)."""
        cell = simulation_trace[index].cell
        if cell == -1:
            return None
        first = index
        while first > 0 and simulation_trace[first - 1].cell == cell:
            first -= 1
        return carrier_history_fingerprint(
            [(entry.stmt_code, key_lineages(entry.input_hashes)) for entry in simulation_trace[first:index]],
            carrier,
        )
