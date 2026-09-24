"""What a statement leaves on record beyond its value: the provenance a
restart needs, and this cell's statement log.

Everything written here is best-effort metadata under its own key
(``cache_key.*_key``). Each record is written only when it changed this
session, so re-running a notebook costs no extra writes.
"""

from __future__ import annotations

import ast
import base64
import importlib.util
import logging
import marshal
import os
import types
from typing import TYPE_CHECKING, Any

from cash.analysis.code_analyzer import CodeAnalyzer
from cash.analysis.mutation_effects import live_function_source
from cash.analysis.namespace_effects import statement_written_paths
from cash.notebook.cache_key import (
    called_function_dependencies,
    called_function_globals,
    import_bindings_key,
    mutation_verdict_key,
    read_provenance_key,
    write_provenance_key,
)
from cash.notebook.carrier_history import FIGURE_KINDS, carrier_history_fingerprint
from cash.notebook.stateful_carriers import stateful_carrier_kind
from cash.tracking.file_dep_snapshot import snapshot_file_deps

if TYPE_CHECKING:
    from cash.notebook._protocols import CashInstanceProtocol, ShellProtocol
    from cash.notebook.tracking_state import TrackingState
    from cash.tracking.function_tracker import FunctionTracker

logger = logging.getLogger(__name__)

_LOG_PROCESSOR = "[PROCESSOR]"

__all__ = ["StatementRecords"]


class StatementRecords:
    """Provenance records a statement persists, and the cell's statement log."""

    def __init__(
        self,
        shell: ShellProtocol,
        tracking_state: TrackingState,
        cash_instance: CashInstanceProtocol | None,
        function_tracker: FunctionTracker | None,
    ) -> None:
        self.shell = shell
        self.tracking_state = tracking_state
        self.cash_instance = cash_instance
        self.function_tracker = function_tracker
        # This cell's statements so far, each with the lineages it read, for a
        # chart writer's provenance (``carrier_history``).
        self._cell_stmt_log: list[tuple[str, dict[str, str]]] = []
        # What was last written this session, per record, so an unchanged
        # record is not written again.
        self._import_bindings_written: dict[str, dict[str, dict[str, Any]]] = {}
        self._mutation_verdicts_written: dict[str, list[str]] = {}
        self._read_provenance_written: dict[str, list[str]] = {}

    def begin_cell(self) -> None:
        """Start this cell's statement log, before its statements run."""
        self._cell_stmt_log = []

    def _resolve_source(self, name: str) -> str | None:
        return live_function_source(name, self.shell.user_ns)

    #: Statements logged per cell for ``carrier_history``; a longer cell (a loop's
    #: iterations) stops logging, and its charts are judged as before.
    _MAX_CELL_STMT_LOG = 5000

    def log_statement_reads(self, code: str, inputs: set[str]) -> None:
        """Record *code* with the lineages it reads, as the simulation keys them:
        its inputs, and the globals its callees read."""
        log = self._cell_stmt_log
        if len(log) >= self._MAX_CELL_STMT_LOG:
            return
        log.append((code, self.lineages_read(inputs)))

    def lineages_read(self, inputs: set[str]) -> dict[str, str]:
        read = {}
        for dep in called_function_dependencies(
            sorted(inputs), self.shell.user_ns, self.tracking_state.variable_lineage, None
        ):
            name, _, lineage = dep.partition(":")
            if lineage != "ABSENT":
                read[name] = lineage
        read.update(
            {n: self.tracking_state.variable_lineage[n] for n in inputs if n in self.tracking_state.variable_lineage}
        )
        return read

    def begin_control_log(self, code: str):
        """Start logging control structure *code* as ONE statement, the way the
        upstream simulation traces it: with the lineages it reads as it starts.
        Its body's statements log themselves per iteration, and a figure drawn
        through ``for ax in axes`` then had a history the simulation could never
        reproduce -- a chart drawn in a loop was never known to be current, nor
        known to be stale. Pass the result to `end_control_log`."""
        try:
            inputs, _outputs = CodeAnalyzer.analyze_code_block(
                code, resolve_source=self._resolve_source, user_ns=self.shell.user_ns
            )
            return len(self._cell_stmt_log), (code, self.lineages_read(inputs))
        except Exception:  # noqa: BLE001 - a history is optional; none means "cannot vouch"
            return len(self._cell_stmt_log), (code, {})

    def end_control_log(self, mark) -> None:
        """Replace what the body of the control structure begun at *mark* logged
        with the structure itself."""
        start, entry = mark
        del self._cell_stmt_log[start:]
        if len(self._cell_stmt_log) < self._MAX_CELL_STMT_LOG:
            self._cell_stmt_log.append(entry)

    def persist_import_bindings(self, code: str, tree: ast.Module | None) -> None:
        """Record, across restarts, what a ``from X import Y`` statement bound.

        See :func:`~cash.notebook.cache_key.import_bindings_key`. For each name:
        whether it is a module, and for a callable the source digest the
        lineage and key take from it, and a plain function's code (marshal,
        tagged with the bytecode magic) for the callee walk. Written only when
        it changed this session; best-effort.
        """
        if "import" not in code:
            return
        try:
            nodes = [n for n in (tree or ast.parse(code)).body if isinstance(n, ast.ImportFrom)]
        except SyntaxError:
            return
        if not nodes:
            return

        user_ns = self.shell.user_ns
        bindings: dict[str, dict[str, Any]] = {}
        for node in nodes:
            for alias in node.names:
                name = alias.asname or alias.name
                if name == "*" or name not in user_ns:
                    continue
                value = user_ns[name]
                entry: dict[str, Any] = {"module": isinstance(value, types.ModuleType)}
                if callable(value) and not entry["module"] and self.function_tracker is not None:
                    try:
                        entry["digest"] = self.function_tracker.get_function_source_hash(value)
                    except Exception:  # noqa: BLE001 - no digest is a smaller record, not an error
                        entry["digest"] = None
                    entry["is_class"] = isinstance(value, type)
                    func_code = getattr(value, "__code__", None)
                    if isinstance(func_code, types.CodeType) and not entry["is_class"]:
                        try:
                            entry["code"] = base64.b64encode(marshal.dumps(func_code)).decode("ascii")
                        except ValueError:
                            pass
                bindings[name] = entry
        if not bindings:
            return
        written = self._import_bindings_written
        if written.get(code) == bindings:
            return
        backend = self.cash_instance.backend if self.cash_instance else None
        if backend is None:
            return

        try:
            backend.set_metadata_only(
                import_bindings_key(code),
                {
                    "import_bindings": True,
                    "bindings": bindings,
                    "code": code,
                    "ttl": None,
                    "magic": importlib.util.MAGIC_NUMBER.hex(),
                },
            )
            written[code] = bindings
        except (OSError, TypeError, ValueError, AttributeError):
            logger.debug("%s import-binding persistence failed", _LOG_PROCESSOR)

    def persist_mutation_verdict(self, source_hash: str, receivers: set[str]) -> None:
        """Record, across restarts, which receivers this bare method call mutated.

        See :func:`~cash.notebook.cache_key.mutation_verdict_key`. Written only
        when the verdict is new this session. Best-effort: without it the
        simulation after a restart assumes the call mutates, as it always did.
        """
        verdict = sorted(receivers)
        written = self._mutation_verdicts_written
        if written.get(source_hash) == verdict:
            return
        backend = self.cash_instance.backend if self.cash_instance else None
        if backend is None:
            return

        try:
            backend.set_metadata_only(
                mutation_verdict_key(source_hash),
                {"mutation_verdict": True, "receivers": verdict, "ttl": None},
            )
            written[source_hash] = verdict
        except (OSError, TypeError, ValueError, AttributeError):
            logger.debug("%s mutation-verdict persistence failed", _LOG_PROCESSOR)

    def persist_read_provenance(self, code: str, accessed_files: set[str]) -> None:
        """Record, across restarts, which files this statement read.

        See :func:`~cash.notebook.cache_key.read_provenance_key`. Written only
        when the statement's read set is new this session, so a notebook re-run
        costs no extra writes. Best-effort: a failure leaves the read set
        unknown after a restart, which is the conservative old behaviour.
        """
        paths = sorted(accessed_files)
        written = self._read_provenance_written
        if written.get(code) == paths:
            return
        backend = self.cash_instance.backend if self.cash_instance else None
        if backend is None:
            return
        try:
            backend.set_metadata_only(
                read_provenance_key(code),
                {"read_provenance": True, "paths": paths, "code": code, "ttl": None},
            )
            written[code] = paths
        except (OSError, TypeError, ValueError, AttributeError):
            logger.debug("%s read-provenance persistence failed", _LOG_PROCESSOR)

    #: A writer that produced more files than this gets no provenance.
    _MAX_PROVENANCE_FILES = 1000

    def user_written_paths(self, paths) -> frozenset[str]:
        """*paths* without cash's own storage (its cache directories)."""
        if not paths:
            return frozenset()
        roots = set()
        backend = self.cash_instance.backend if self.cash_instance else None
        root = backend.local_dir if backend is not None else None
        if isinstance(root, (str, os.PathLike)):
            roots.add(os.path.normcase(os.path.abspath(os.fspath(root))) + os.sep)
        return frozenset(p for p in paths if not any(os.path.normcase(p).startswith(r) for r in roots))

    def persist_write_provenance(
        self,
        code: str,
        inputs: set[str],
        tree: ast.Module | None,
        written: frozenset[str] | set[str] = frozenset(),
    ) -> None:
        """Record what file(s) a just-executed writer statement produced.

        Persists ``{paths, file_deps snapshot, input lineages}`` to the backend
        under a writer-specific key derived from the statement source, so a
        post-restart isolated downstream reader can short-circuit an
        already-fresh writer (:meth:`FileWriterScheduler._writer_output_already_fresh`)
        instead of re-firing a non-idempotent side effect and re-deriving stale
        data. Best-effort and CONSERVATIVE: a writer whose output paths neither
        resolve from the code nor were *written* as it ran records nothing,
        and keeps re-firing as before.

        *written* is what the statement was seen writing (``write_observer``,
        cash's own storage removed). A path it wrote that is gone again -- a
        temporary file renamed into place -- is dropped; a path the code names
        must exist. The input lineages cover the globals the statement's
        callees read too: an edited helper is a new payload.
        """
        try:
            raw_paths = statement_written_paths(code, tree, self.shell.user_ns) or set()
            named = {os.path.abspath(p) for p in raw_paths}
            seen = {p for p in written if os.path.exists(p)} - named
            if not named and not seen:
                return  # nothing resolvable, nothing observed -> stay conservative
            if len(named) + len(seen) > self._MAX_PROVENANCE_FILES:
                return  # snapshotting would cost more than a re-fire saves
            paths = sorted(named | seen)
            file_deps = snapshot_file_deps(set(paths))
            # Every recorded path must be readable now, else there is nothing to
            # vouch for (and a later freshness check would fail anyway).
            if any(p not in file_deps for p in paths):
                return
            names = set(inputs) | called_function_globals(inputs, self.shell.user_ns)
            input_lineages = {
                v: self.tracking_state.variable_lineage[v] for v in names if v in self.tracking_state.variable_lineage
            }
            record = {
                "write_provenance": True,
                "paths": paths,
                "file_deps": file_deps,
                "input_lineages": input_lineages,
                "code": code,
                "ttl": None,  # provenance must not expire out from under a reader
            }
            histories = self._carrier_histories(code, inputs)
            if histories:
                record["carrier_histories"] = histories
            backend = self.cash_instance.backend if self.cash_instance else None
            if backend is not None:
                backend.set_metadata_only(
                    write_provenance_key(code),
                    record,
                )
        except (OSError, TypeError, ValueError, AttributeError):
            logger.debug("%s write-provenance persistence failed", _LOG_PROCESSOR)

    def _carrier_histories(self, code: str, inputs: set[str]) -> dict[str, str]:
        """``{figure name: history fingerprint}`` for the figures writer *code* reads.

        A figure's own lineage cannot vouch for it after a restart (see
        ``carrier_history``); the history that drew it, taken from this cell's
        statements before the write, can.
        """

        log = self._cell_stmt_log
        end = next((k for k in range(len(log) - 1, -1, -1) if log[k][0] == code), None)
        if end is None:
            return {}
        histories = {}
        for name in inputs:
            if stateful_carrier_kind(self.shell.user_ns.get(name)) not in FIGURE_KINDS:
                continue
            fingerprint = carrier_history_fingerprint(log[:end], name)
            if fingerprint is not None:
                histories[name] = fingerprint
        return histories
