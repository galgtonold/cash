"""Which files the cell being checked reads, directly or through what it needs.

A file-writing statement upstream is re-run only when a file it writes is one
of those (:meth:`ReadScope.relevant_read_paths`); a writer whose output nothing
relevant reads is a side effect the check must not fire again.
"""

from __future__ import annotations

import ast
import collections
import logging

from ...analysis.code_analyzer import CodeAnalyzer, clean_cell_source, parse_cell_source, statement_code
from ...analysis.namespace_effects import resolve_literal_path, resolve_path_list, statement_read_paths
from .._protocols import ShellProtocol, TrackingState
from .._trace import trace_event
from ..cache_key import read_provenance_key
from .virtual_lineage import VirtualLineage

__all__ = ["ReadScope"]

logger = logging.getLogger(__name__)


def _statement_codes(cell_source: str) -> list[str]:
    """The cell's top-level statements as the runtime keys them (unparsed,
    with an expression's trailing ``;`` kept); the raw text if it does not parse."""
    try:
        clean = clean_cell_source(cell_source)
        tree = parse_cell_source(cell_source)
    except (ValueError, TypeError):
        return [cell_source]
    if tree is None:
        return [cell_source]
    return [statement_code(node, clean) for node in tree.body]


def _bind_literal_paths(stmt: str, bound: dict, namespace) -> None:
    """Record in *bound* a name *stmt* binds to a path or a list of paths.

    ``TF = [Path('other.csv')]`` binds ``TF``; any other binding of a name
    drops it, so a later statement never reads a stale value from here.
    """

    try:
        tree = ast.parse(CodeAnalyzer.strip_magics(stmt))
    except (SyntaxError, ValueError, TypeError):
        return
    node = tree.body[0] if len(tree.body) == 1 else None
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        name = node.targets[0].id
        value = resolve_path_list(node.value, namespace)
        if value is None:
            value = resolve_literal_path(node.value, namespace)
        if value is not None:
            bound[name] = value
            return
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            bound[n.id] = None


class ReadScope:
    """The file paths the cell being checked depends on."""

    def __init__(self, shell: ShellProtocol, tracking_state: TrackingState, virtual_lineage: VirtualLineage) -> None:
        self.shell = shell
        self.tracking_state = tracking_state
        self.virtual_lineage = virtual_lineage

    def _persisted_reads(self, code: str) -> set[str] | None:
        """Files *code* read when it last ran, from the backend, or ``None``."""
        backend = self.virtual_lineage.backend()
        if backend is None:
            return None

        try:
            record = backend.get_metadata(read_provenance_key(code))
        except (OSError, TypeError, ValueError, AttributeError):
            return None
        if not record or not record.get("read_provenance"):
            return None
        return set(record.get("paths") or ())

    @staticmethod
    def _statements_the_cell_depends_on(
        required_inputs: set[str] | None,
        simulation_trace: list,
    ) -> set[int]:
        """Trace positions whose outputs the current cell's inputs derive from.

        Only these can make a file the cell's reconstruction reads. Before this
        scope, one unresolvable read anywhere above -- a helper's
        ``pd.read_parquet(path)`` -- let every stale writer in the notebook
        re-fire, and with them the fits feeding their charts: a sanity-check
        cell reading only the loaded frame took 309 s.
        """
        if required_inputs is None:
            return set(range(len(simulation_trace)))
        needed = set(required_inputs)
        relevant: set[int] = set()
        for i in range(len(simulation_trace) - 1, -1, -1):
            outputs, inputs = simulation_trace[i].outputs, simulation_trace[i].inputs
            if outputs & needed:
                relevant.add(i)
                needed |= set(inputs)
        return relevant

    def _defs_whose_callers_recorded_reads(
        self,
        simulation_trace: list,
        relevant: set[int],
        efd: dict,
    ) -> set[int]:
        """Relevant ``def`` statements whose reads are already known elsewhere.

        Defining a function reads nothing; its body reads when a statement
        calls it, and the tracker records that against the caller's outputs
        (or, after a restart, the caller's persisted reads). A path the body
        leaves unresolvable (``pd.read_csv(path)``) then says nothing unknown.
        A caller that is itself such a ``def`` counts when it is covered.
        """
        defs: dict[int, str] = {}
        for i in relevant:
            code = simulation_trace[i].stmt_code
            if not code.lstrip().startswith(("def ", "async def ", "@")):
                continue
            try:
                body = ast.parse(code).body
            except SyntaxError:
                continue
            if len(body) == 1 and isinstance(body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs[i] = body[0].name

        def recorded(i: int) -> bool:
            outputs = simulation_trace[i].outputs
            if outputs and all(efd.get(o) for o in outputs):
                return True
            return self._persisted_reads(simulation_trace[i].stmt_code) is not None

        covered: set[int] = set()
        changed = True
        while changed:
            changed = False
            for i, name in defs.items():
                if i in covered:
                    continue
                callers = [j for j in relevant if j > i and name in simulation_trace[j].inputs]
                if callers and all((j in covered) if j in defs else recorded(j) for j in callers):
                    covered.add(i)
                    changed = True
        return covered

    def relevant_read_paths(
        self,
        required_inputs: set[str] | None,
        simulation_trace: list,
        notebook_cells: list[str] | None,
        current_cell_idx: int | None,
    ) -> tuple[set[str], bool]:
        """File paths this cell's reconstruction actually READS.

        A file-writer is only worth re-firing during reconstruction when a
        consumer relevant to the current cell reads the file it writes. This
        collects those consumed paths from three sources:

        * the recorded file-deps of the current cell's required inputs (the
          within-session, already-propagated read edges), and
        * every file READ statically named by an upstream trace statement, and
        * every file READ statically named by the current cell itself (the only
          signal that survives a kernel restart, when the tracking dicts are
          empty and the reader is the cell the user ran).

        Returns ``(paths, fully_known)``. ``fully_known`` is ``False`` when any
        recognised read target could not be statically resolved (an f-string /
        computed path) — the caller must then suppress no writer, since it cannot
        prove the writer's output is unread. A writer whose resolvable output
        path is in none of these paths is an unrelated / terminal side-effect and
        must not be re-fired for THIS cell.
        """

        paths: set[str] = set()
        fully_known = True
        user_ns = self.shell.user_ns

        efd = self.tracking_state.executed_file_deps
        for v in required_inputs or ():
            dep = efd.get(v)
            if not dep:
                continue
            # Recorded file deps are usually {path: snapshot} but some code paths
            # store a plain set/list of paths -- accept either shape.
            paths.update(dep.keys() if hasattr(dep, "keys") else dep)

        def _collect(src: str, outputs=(), namespace=None) -> None:
            nonlocal fully_known
            try:
                clean = CodeAnalyzer.strip_magics(src.replace("\r\n", "\n"))
            except (ValueError, TypeError):
                return
            if not clean.strip():
                return
            try:
                r = statement_read_paths(clean, namespace=user_ns if namespace is None else namespace)
            except (SyntaxError, ValueError, TypeError):
                r = None
            if r is None and outputs and all(o in efd for o in outputs):
                # Not resolvable from the code (``pd.read_csv(f)`` over a glob
                # result), but the statement ran this session and the tracker
                # recorded what fed its outputs -- a superset of what it read.
                # Without this one comprehension switched the scope gate off
                # for the whole notebook, and a chart nothing reads was re-drawn
                # for every downstream cell.
                r = set()
                for o in outputs:
                    dep = efd[o]
                    r.update(dep.keys() if hasattr(dep, "keys") else dep)
            if r is None:
                # After a restart the session record is empty; what the
                # statement read when it last ran was persisted for this.
                r = self._persisted_reads(src)
            if r is None:
                trace_event("read_path_unknown", stmt=src[:90])
                fully_known = False
            else:
                paths.update(r)

        relevant = self._statements_the_cell_depends_on(required_inputs, simulation_trace)
        covered_defs = self._defs_whose_callers_recorded_reads(simulation_trace, relevant, efd)
        for i, entry in enumerate(simulation_trace):
            if i not in relevant:
                continue
            code = entry.stmt_code
            if i not in covered_defs and ("read" in code or "open(" in code or "load" in code):
                _collect(code, entry.outputs)
            # What the tracker recorded behind this statement's outputs counts
            # too, whatever the code looks like: a reader static analysis does
            # not recognise (``PIL.Image.open(p)``) must not make its file look
            # unread now that more write paths resolve.
            for o in entry.outputs:
                dep = efd.get(o)
                if dep:
                    paths.update(dep.keys() if hasattr(dep, "keys") else dep)

        if notebook_cells and current_cell_idx is not None and 0 <= current_cell_idx < len(notebook_cells):
            # One statement at a time, keyed as the runtime keys them, so a
            # statement's persisted read record is found after a restart (the
            # whole cell's text is no statement's key).
            # The cell has not run yet, so a path its own earlier statement
            # binds (``TF = [Path('other.csv')]``) is in no namespace; resolve
            # it from the code.
            bound: dict = {}
            for stmt in _statement_codes(notebook_cells[current_cell_idx]):
                _collect(stmt, namespace=collections.ChainMap(bound, user_ns or {}))
                _bind_literal_paths(stmt, bound, collections.ChainMap(bound, user_ns or {}))

        return paths, fully_known
