"""Keep and serve the plot runs of a cell (see :mod:`cash.notebook.plot_outputs`).

The cell executor tells :class:`PlotRuns` where each top-level statement
starts and what it reported when it ended; this records the runs of
statements that went from no open figure to none, keeps what they showed,
and on a later run of the same statements with the same inputs offers that
output in place of running them.
"""

from __future__ import annotations

import ast
import logging
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cash._user_attrs import attribute_or

from ..._clock import perf_counter as _perf_counter
from ...analysis.cacheability_decision import has_missing_lineage
from ...analysis.code_analyzer import parse_cell_source
from ...analysis.mutation_effects import statement_effects
from ...tracking.env_reads import environment_unchanged
from ..cache_key import CacheKeyContext, compute_cache_key
from ..cache_status import CacheStatus
from ..plot_outputs import (
    PLOT_FACTS,
    PLOT_RECORD_CAP,
    StoredEcho,
    figure_settings_digest,
    first_read_after,
    inline_backend_closes_figures,
    is_pyplot_reason,
    mutated_names,
    plot_shapes_key,
    record_size,
    rich_records,
)
from ..pyplot_state import pyplot_state
from ..server_discovery import get_notebook_cells
from ..statement.capture import capture_output, replay_outputs
from ..statement.run import ECHO_FIELD

if TYPE_CHECKING:
    from ..statement import ProcessResult, StatementProcessor
    from ..upstream.checker import UpstreamChecker

logger = logging.getLogger(__name__)

__all__ = ["PlotRuns", "ServedStatement"]

#: The most plot runs kept for one first statement.
_SHAPES_KEPT = 8

#: Seconds below which a plot run is not worth keeping.
_KEEP_FLOOR_S = 0.01

#: A cell line IPython runs as a magic or a shell command.
_MAGIC_LINE = re.compile(r"^\s*[%!]", re.MULTILINE)

_LOG = "[PLOT]"


#: What pyplot's state reads as with no figure open (`pyplot_state`).
_EMPTY = (None, ())


def _changes_its_own(reason: str, bound: set[str]) -> bool:
    """Whether *reason* says only that the statement changed in place what
    the plot run bound before it (``ax.plot(...)`` after ``fig, ax = ...``)."""
    names = mutated_names(reason)
    return names is not None and bool(names) and names <= bound


def _no_figures() -> bool:
    return pyplot_state() == _EMPTY


@dataclass
class ServedStatement:
    """One statement of a served plot run: its badge row, and what it
    showed, part by part (a loop's passes are parts of one statement)."""

    metrics: dict
    parts: list[tuple[str, str, list[dict]]]


@dataclass
class _Open:
    """A plot run being recorded."""

    start: int
    occurrence: int
    settings: str
    lineage: dict[str, str]
    started: float
    codes: list[str] = field(default_factory=list)
    statements: list[dict] = field(default_factory=list)
    bound: set[str] = field(default_factory=set)
    env: dict[str, str] = field(default_factory=dict)
    drew: bool = False
    shown: bool = False
    refused: str | None = None
    echo: Any = None


class PlotRuns:
    """The plot runs of the cell running now, and the kept ones."""

    def __init__(self, shell: Any, statement_processor: StatementProcessor, upstream_checker: UpstreamChecker) -> None:
        self.shell = shell
        self._processor = statement_processor
        self._checker = upstream_checker
        self._open: _Open | None = None
        self._cell: Any = None
        self._parsed: dict[str, list[tuple[str, ast.stmt | None]]] = {}

    # -- the cell ----------------------------------------------------------

    def begin_cell(self, cell: Any) -> None:
        self._open = None
        self._cell = cell

    @property
    def recording(self) -> bool:
        return self._open is not None

    def interrupt(self) -> None:
        """A statement ran in a way this does not see (a planned skip): what
        is being recorded is not kept."""
        if self._open is not None:
            self._open.refused = "a statement was skipped by the cell's plan"

    # -- recording ---------------------------------------------------------

    def before(self, index: int, occurrence: int) -> None:
        """Statement *index* of the cell is about to run."""
        if self._open is not None or not self._watching():
            return
        settings = figure_settings_digest(self.shell)
        if settings is None:
            return
        lineage = dict(self._processor.tracking_state.variable_lineage)
        self._open = _Open(index, occurrence, settings, lineage, _perf_counter())

    def after(self, index: int, code: str, display: str | None, rows: list[ProcessResult]) -> None:
        """Statement *index* (*code*) ran, reporting *rows*."""
        run = self._open
        facts = [row.pop(PLOT_FACTS, None) for row in rows]
        if run is None:
            return
        run.codes.append(code)
        parts = []
        seconds = 0.0
        for row, fact in zip(rows, facts, strict=True):
            status = CacheStatus.parse(row.get("status"))
            if status is CacheStatus.ERROR:
                run.refused = "a statement raised"
            elif fact is None:
                if status is CacheStatus.RESTORED:
                    run.bound |= set(row.get("restored_vars") or ())
                else:
                    run.refused = f"cash did not see what {code[:40]!r} did"
            else:
                if fact.blockers:
                    run.refused = f"{code[:40]!r} {fact.blockers[0]}"
                # Drawing on a figure or axes the run made changes it in place.
                other = [r for r in fact.reasons if not is_pyplot_reason(r) and not _changes_its_own(r, run.bound)]
                if other:
                    run.refused = run.refused or f"{code[:40]!r} is not stored: {other[0]}"
                run.drew = run.drew or any(is_pyplot_reason(r) for r in fact.reasons)
                run.bound |= fact.bound
                run.env.update(fact.env)
            rich = rich_records(row.get("rich_outputs") or ())
            if rich is None:
                run.refused = "it showed an output that is not plain data"
                rich = []
            run.shown = run.shown or bool(rich)
            parts.append((str(row.get("stdout") or ""), str(row.get("stderr") or ""), rich))
            seconds += float(row.get("execution_time") or 0.0)
            if ECHO_FIELD in row:
                run.echo = row[ECHO_FIELD]
        run.statements.append({"code": code, "display": display, "parts": parts, "seconds": seconds})
        if not _no_figures():
            return
        self._open = None
        if run.drew:
            self._keep(run, flush=None)
        elif run.start != index:
            logger.debug("%s statements %d-%d drew nothing pyplot shows", _LOG, run.start, index)

    def end_cell(self, last_index: int) -> None:
        """The cell's statements ran: a plot run still open ends here, where
        the inline backend shows the open figures."""
        run, self._open = self._open, None
        if run is None:
            return
        if not run.drew or run.refused is not None or not inline_backend_closes_figures():
            logger.debug("%s open at the end of the cell, not kept: %s", _LOG, run.refused or "drew nothing")
            return
        flush = sys.modules.get("matplotlib_inline.backend_inline")
        flush = attribute_or(flush, "flush_figures", None)
        if flush is None:
            return
        try:
            with capture_output(stdout=False, stderr=False, display=True) as captured:
                flush()
        except Exception:
            logger.debug("%s could not render the open figures", _LOG, exc_info=True)
            return
        shown = list(captured.outputs)
        if shown:
            self._show_after_cell(shown)
        if run.start + len(run.codes) - 1 != last_index or not _no_figures():
            return
        rich = rich_records(shown)
        if rich is None:
            return
        self._keep(run, flush=rich)

    def _watching(self) -> bool:
        return inline_backend_closes_figures() and _no_figures()

    def _keep(self, run: _Open, flush: list[dict] | None) -> None:
        """Store *run*, which ended with no figure open."""
        try:
            self._store(run, flush)
        except Exception:
            logger.debug("%s could not keep the plot run", _LOG, exc_info=True)

    def _store(self, run: _Open, flush: list[dict] | None) -> None:
        if run.refused is not None:
            logger.debug("%s not kept: %s", _LOG, run.refused)
            return
        if not run.shown and not flush:
            # Showed nothing: not a plot anyone sees (a ``def`` of a helper that draws).
            logger.debug("%s not kept: it showed nothing", _LOG)
            return
        if not environment_unchanged(run.env):
            logger.debug("%s not kept: an environment variable it read changed", _LOG)
            return
        seconds = _perf_counter() - run.started
        if seconds < _KEEP_FLOOR_S:
            logger.debug("%s not kept: it took %.4fs", _LOG, seconds)
            return
        if figure_settings_digest(self.shell) != run.settings:
            logger.debug("%s not kept: it changed matplotlib's settings", _LOG)
            return
        code = "\n".join(run.codes)
        inputs = self._inputs(code)
        if inputs is None or inputs & run.bound:
            logger.debug("%s not kept: it binds a name it reads", _LOG)
            return
        lineage = self._processor.tracking_state.variable_lineage
        if any(run.lineage.get(name) != lineage.get(name) for name in inputs):
            logger.debug("%s not kept: an input changed as it ran", _LOG)
            return
        if has_missing_lineage(inputs, self.shell.user_ns, lineage):
            logger.debug("%s not kept: an input has no lineage", _LOG)
            return
        echo = None
        if run.echo is not None:
            echo = self._format_echo(run.echo)
            if echo is None:
                return
        size = sum(record_size(p[2]) + len(p[0]) + len(p[1]) for s in run.statements for p in s["parts"])
        size += record_size(flush or ())
        if size > PLOT_RECORD_CAP:
            return
        key = self._key(code, inputs, run.occurrence, run.settings)
        if key is None:
            return
        backend = self._backend()
        if backend is None:
            return
        bound = sorted(run.bound | (_HISTORY_NAMES if echo is not None else set()))
        payload = {"statements": run.statements, "flush": flush, "echo": echo}
        metadata = {
            "plot_outputs": True,
            "ttl": attribute_or(self._cell, "ttl", None),
            "execution_time": seconds,
            "size": size,
            "bound": bound,
            "env": run.env,
            "force_persist": bool(self._processor.persist_all),
        }
        backend.set(key, payload, metadata)
        self._note_shape(backend, run.codes, run.occurrence)
        logger.debug(
            "%s kept %d statement(s) under %s (inputs %s, settings %s)",
            _LOG,
            len(run.codes),
            key[:24],
            sorted(inputs),
            run.settings[:12],
        )

    # -- serving -----------------------------------------------------------

    def serve(
        self,
        index: int,
        occurrence: int,
        code_at: Callable[[int], str | None],
        body: list[ast.stmt],
        source: str,
        *,
        blocked: Callable[[int], bool] = lambda _k: False,
    ) -> dict[int, ServedStatement] | None:
        """The statements from *index* on, served from a kept plot run, or
        None when none can be. *code_at(k)* is the text of statement *k*;
        a statement *blocked(k)* says is not to be skipped is in none."""
        if self._open is not None or not self._watching():
            return None
        backend = self._backend()
        if backend is None:
            return None
        first = code_at(index)
        if first is None:
            return None
        try:
            record = backend.get_metadata(plot_shapes_key(first, occurrence))
        except Exception:  # noqa: BLE001
            return None
        shapes = (record or {}).get("shapes") or []
        settings = None
        for shape in shapes:
            last = index + len(shape) - 1
            if last >= len(body) or any(blocked(k) for k in range(index + 1, last + 1)):
                continue
            if any(code_at(index + k) != text for k, text in enumerate(shape)):
                continue
            if settings is None:
                settings = figure_settings_digest(self.shell)
                if settings is None:
                    return None
            served = self._served(backend, shape, index, occurrence, settings, body, source)
            if served is not None:
                return served
        return None

    def _served(
        self,
        backend: Any,
        shape: list[str],
        index: int,
        occurrence: int,
        settings: str,
        body: list[ast.stmt],
        source: str,
    ) -> dict[int, ServedStatement] | None:
        code = "\n".join(shape)
        inputs = self._inputs(code)
        lineage = self._processor.tracking_state.variable_lineage
        if inputs is None or has_missing_lineage(inputs, self.shell.user_ns, lineage):
            return None
        key = self._key(code, inputs, occurrence, settings)
        if key is None:
            return None
        try:
            metadata, payload = backend.get(key)
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(payload, dict) or metadata is None:
            return None
        last = index + len(shape) - 1
        if payload.get("flush") is not None and last != len(body) - 1:
            return None
        if payload.get("echo") is not None and last != len(body) - 1:
            return None
        if not environment_unchanged(metadata.get("env") or {}):
            logger.debug("%s not served: an environment variable it read changed", _LOG)
            return None
        bound = frozenset(metadata.get("bound") or ())
        read = self._read_after(bound, body, last, source)
        if read is not False:
            logger.debug("%s not served: %s", _LOG, read)
            return None
        statements = payload.get("statements") or []
        if len(statements) != len(shape):
            return None
        served: dict[int, ServedStatement] = {}
        for k, stored in enumerate(statements):
            row = {
                "code": shape[k],
                "display_code": stored.get("display"),
                "status": CacheStatus.RESTORED,
                "execution_time": 0.0,
                "total_time": 0.0,
                "saved_time": float(stored.get("seconds") or 0.0),
                "restored_vars": [],
                "uncacheable_reasons": [],
                "plot_output": True,
            }
            served[index + k] = ServedStatement(row, [tuple(p) for p in stored.get("parts") or []])
        echo = payload.get("echo")
        if echo is not None:
            served[last].metrics[ECHO_FIELD] = StoredEcho(echo[0], echo[1])
        if payload.get("flush"):
            self._show_after_cell(payload["flush"])
        user_ns = self.shell.user_ns
        for name in bound - _HISTORY_NAMES:
            # Not drawn this time: what a run bound is gone, as after a restart.
            user_ns.pop(name, None)
            self._processor.forget_variable(name)
        logger.debug("%s served %d statement(s) from %s", _LOG, len(shape), key[:24])
        return served

    def _read_after(self, bound: frozenset[str], body: list[ast.stmt], last: int, source: str) -> str | bool:
        """The name of *bound* the code after statement *last* of the cell
        reads before binding it, ``"unknown"`` when that cannot be told, or
        False when none does."""
        if not bound:
            return False
        later = self._cells_after()
        if later is None:
            return "the cells below are not known"
        statements: list[tuple[str, ast.stmt | None]] = []
        for node in body[last + 1 :]:
            statements.append((ast.get_source_segment(source, node) or ast.unparse(node), node))
        for text in later:
            statements.extend(self._statements(text))
        name = first_read_after(bound, statements)
        return f"'{name}' is read below" if name is not None else False

    def _cells_after(self) -> list[str] | None:
        """The notebook's cells below the one running, or None when it is
        not known which cell that is."""
        raw = attribute_or(self._cell, "raw_cell", None)
        if raw is None:
            return None
        try:
            cells = get_notebook_cells()
        except Exception:  # noqa: BLE001
            return None
        if not cells:
            return None
        found = attribute_or(self._checker, "last_found_cell", None)
        if found is not None and found[0] == raw and found[1] < len(cells) and cells[found[1]].strip() == raw.strip():
            return cells[found[1] + 1 :]
        matches = [k for k, text in enumerate(cells) if text.strip() == raw.strip()]
        if len(matches) != 1:
            return None
        return cells[matches[0] + 1 :]

    def _statements(self, text: str) -> list[tuple[str, ast.stmt | None]]:
        """*text*, a cell, as statements to read (`first_read_after`)."""
        kept = self._parsed.get(text)
        if kept is not None:
            return kept
        statements: list[tuple[str, ast.stmt | None]] = []
        if _MAGIC_LINE.search(text):
            # A magic or a shell command is read as text, first.
            magics = [line for line in text.splitlines() if _MAGIC_LINE.match(line)]
            statements.append(("\n".join(magics), None))
        tree = parse_cell_source(text)
        if tree is None:
            statements.append((text, None))
        else:
            statements.extend((ast.get_source_segment(text, node) or "", node) for node in tree.body)
        if len(self._parsed) > 512:
            self._parsed.clear()
        self._parsed[text] = statements
        return statements

    # -- shared ------------------------------------------------------------

    def _inputs(self, code: str) -> set[str] | None:
        """What the plot run *code* reads that it does not bind first."""
        try:
            tree = ast.parse(code)
            effects = statement_effects(
                code,
                tree,
                namespace=self.shell.user_ns,
                resolve_source=self._processor.resolve_live_function_source,
            )
        except Exception:  # noqa: BLE001 - not understood, not kept
            return None
        return set(effects.inputs)

    def _key(self, code: str, inputs: set[str], occurrence: int, settings: str) -> str | None:
        """The key of the plot run *code* with *inputs*, under matplotlib's
        *settings*: ``compute_cache_key`` in the ``plot`` namespace, the
        settings a line of its text."""
        try:
            result = compute_cache_key(
                f"{code}\n# cash: matplotlib settings {settings}",
                inputs,
                ctx=CacheKeyContext(
                    variable_lineage=self._processor.tracking_state.variable_lineage,
                    user_ns=self.shell.user_ns,
                    function_tracker=self._processor.function_tracker,
                    compute_hash_fn=self._processor.compute_hash,
                    traced=True,
                ),
                occurrence_index=occurrence,
                namespace="plot",
            )
        except Exception:
            logger.debug("%s could not key the plot run", _LOG, exc_info=True)
            return None
        return None if result.matches_nothing else result.cache_key

    def _backend(self) -> Any:
        try:
            cash = self._processor.get_cash_instance()
            return cash.backend if cash is not None else None
        except Exception:  # noqa: BLE001
            return None

    def _note_shape(self, backend: Any, codes: list[str], occurrence: int) -> None:
        key = plot_shapes_key(codes[0], occurrence)
        try:
            record = backend.get_metadata(key) or {}
        except Exception:  # noqa: BLE001
            record = {}
        shapes = [s for s in (record.get("shapes") or []) if s != codes]
        shapes.insert(0, list(codes))
        backend.set_metadata_only(key, {"plot_shapes": True, "shapes": shapes[:_SHAPES_KEPT], "ttl": None})

    def _format_echo(self, value: Any) -> tuple[dict, dict] | None:
        formatter = attribute_or(self.shell, "display_formatter", None)
        if formatter is None:
            return None
        try:
            data, metadata = formatter.format(value)
        except Exception:  # noqa: BLE001
            return None
        rich = rich_records([{"data": data, "metadata": metadata}])
        return (rich[0]["data"], rich[0]["metadata"]) if rich else None

    def _show_after_cell(self, outputs: list) -> None:
        """Show *outputs* where the inline backend shows a cell's figures:
        after the cell, its echoed value included."""
        events = attribute_or(self.shell, "events", None)
        if events is None:
            replay_outputs(rich=outputs)
            return

        def show(*_args: Any) -> None:
            try:
                events.unregister("post_execute", show)
            except ValueError:
                pass
            replay_outputs(rich=outputs)

        events.register("post_execute", show)


#: IPython's output history, which a served echo puts a stand-in into.
_HISTORY_NAMES = frozenset({"_", "__", "___", "Out", "_oh"})
