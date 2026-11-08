"""Keep and serve the plot runs of a cell (see :mod:`cash.notebook.plot_outputs`).

The cell executor tells :class:`PlotRuns` where each top-level statement
starts and what it reported when it ended; this records the runs of
statements that went from no open figure to none, keeps what they showed,
and on a later run of the same statements with the same inputs offers that
output in place of running them.
"""

from __future__ import annotations

import ast
import hashlib
import io
import logging
import pickle
import re
import sys
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cash._user_attrs import attribute_or

from ..._clock import perf_counter as _perf_counter
from ...analysis.cacheability_decision import has_missing_lineage
from ...analysis.code_analyzer import parse_cell_source
from ...analysis.mutation_effects import statement_effects
from ...tracking.env_reads import environment_unchanged
from ...tracking.randomness import rng_virtual_var
from .. import helper_memos
from ..cache_key import CacheKeyContext, compute_cache_key
from ..cache_status import CacheStatus
from ..plot_outputs import (
    PLOT_FACTS,
    PLOT_RECORD_CAP,
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

#: The version of what a kept plot run holds: one without it carries no
#: random streams or memos, which a run of it may have moved or filled,
#: and is not one of the runs its statements' record lists (`_condition`),
#: or keeps what its last line echoed as text, not as the value, or may
#: have unbound a name or bound ``_`` itself.
_FORMAT = 6


def _condition(env: dict, rng_reads: dict, memo_records: list) -> str:
    """What decides where a kept run of a plot run's statements can stand
    in for a run: the environment variables it read, the random streams it
    drew from, and what it found in the helper memos it called.

    The same statements may run under several: a tutorial calls
    ``run(tsp, random_cities(10))`` again further down, where the memo
    already holds the cities the first call computed. Kept as one, each
    place's run replaced the other's, and neither place was ever served.
    So each condition's run is an entry of its own, and the statements'
    record (its key without a condition) lists them, most recently used
    first."""
    text = repr((sorted(env.items()), sorted(rng_reads.items()), memo_records))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
    #: Where its run left the random streams it moved, and the ones it seeded.
    rng_state: dict = field(default_factory=dict)
    rng_reseeded: frozenset[str] = frozenset()
    #: What puts the answers its run added back into the helper memos.
    memo_puts: list[Callable[[], None]] = field(default_factory=list)


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
    #: The random streams a statement drew from before one of the run seeded them.
    rng_read: set[str] = field(default_factory=set)
    #: The streams a statement of the run seeded before drawing from them.
    rng_placed: set[str] = field(default_factory=set)
    #: The counters of the memos `helper_memos` does not find, as it began.
    hidden: dict = field(default_factory=dict)
    #: The names bound in the notebook's namespace as it began.
    names: frozenset[str] = frozenset()
    #: Whether `hidden` was read after a look through every object
    #: (`helper_memos.hidden_scan`): a memo not in it was built by the run.
    fresh: bool = False


class PlotRuns:
    """The plot runs of the cell running now, and the kept ones."""

    def __init__(self, shell: Any, statement_processor: StatementProcessor, upstream_checker: UpstreamChecker) -> None:
        self.shell = shell
        self._processor = statement_processor
        self._checker = upstream_checker
        self._open: _Open | None = None
        #: What a served plot run still does once it takes its place
        #: (`take_place`): the outputs shown after the cell, and the names
        #: its run bound, to drop.
        self._pending: tuple[list | None, frozenset[str], dict[str, Any]] | None = None
        #: ``(key of its record, condition)`` of the run being served:
        #: noted as served once it takes its place (`_used_first`).
        self._promote: tuple[str, str] | None = None
        #: ``key of a record -> {condition: when}`` of the runs served in
        #: this session, which the next keep lists first (`_note_variant`):
        #: no write per hit, which a remote tier pays a round trip for.
        self._served_last: dict[str, dict[str, int]] = {}
        self._serves = 0
        self._cell: Any = None
        #: ``(cell text, statement)`` where a plot run that drew began in
        #: this session: a run beginning there again first looks through
        #: every object for the memos cash does not find (`before`).
        self._draw_starts: set[tuple[str, int]] = set()
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
        self._adopt_unseen_seeds()
        lineage = dict(self._processor.tracking_state.variable_lineage)
        self._open = _Open(index, occurrence, settings, lineage, _perf_counter())
        self._open.names = frozenset(self.shell.user_ns)
        try:
            if self._start_key(index) in self._draw_starts:
                # Where a plot drew before: the memos it may empty are read
                # now, so one emptied by it does not pass for one never
                # called (`hidden_moved`). One look through every object
                # per such run, not per statement.
                self._open.hidden = helper_memos.hidden_scan(self.shell.user_ns)
                self._open.fresh = True
            else:
                self._open.hidden = helper_memos.hidden_counters()
        except Exception:
            logger.debug("%s could not read the memos cash does not find", _LOG, exc_info=True)

    def _start_key(self, index: int) -> tuple[str, int]:
        return (str(attribute_or(self._cell, "raw_cell", None) or ""), index)

    def _note_drew(self, run: _Open) -> None:
        if len(self._draw_starts) > 4096:
            self._draw_starts.clear()
        self._draw_starts.add(self._start_key(run.start))

    def after(self, index: int, code: str, display: str | None, rows: list[ProcessResult]) -> None:
        """Statement *index* (*code*) ran, reporting *rows*."""
        run = self._open
        facts = [row.pop(PLOT_FACTS, None) for row in rows]
        if run is None:
            return
        run.codes.append(code)
        # What it may bind, read from its text: the rows of a loop's passes
        # name no loop variable, and a branch not taken binds nothing now.
        binds = _statement_binds(code)
        if binds is None:
            run.refused = f"cash could not read what {code[:40]!r} binds"
        else:
            run.bound |= binds
        parts = []
        seconds = 0.0
        rng_state: dict = {}
        reseeded: set[str] = set()
        memos: list = []
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
                run.rng_read |= set(fact.rng_read) - run.rng_placed
                run.rng_placed |= set(fact.rng_seeded_first)
                rng_state.update(fact.rng_state)
                reseeded |= set(fact.rng_reseeded)
                if fact.memos is not None:
                    memos.append(list(fact.memos))
            rich = rich_records(row.get("rich_outputs") or ())
            if rich is None:
                run.refused = "it showed an output that is not plain data"
                rich = []
            run.shown = run.shown or bool(rich)
            parts.append((str(row.get("stdout") or ""), str(row.get("stderr") or ""), rich))
            seconds += float(row.get("execution_time") or 0.0)
            if ECHO_FIELD in row:
                run.echo = row[ECHO_FIELD]
        run.statements.append(
            {
                "code": code,
                "display": display,
                "parts": parts,
                "seconds": seconds,
                "rng_state": rng_state,
                "rng_reseeded": sorted(reseeded),
                "memos": memos,
            }
        )
        if not _no_figures():
            return
        self._open = None
        if run.drew:
            self._note_drew(run)
            self._keep(run, flush=None)
        elif run.start != index:
            logger.debug("%s statements %d-%d drew nothing pyplot shows", _LOG, run.start, index)

    def end_cell(self, last_index: int) -> None:
        """The cell's statements ran: a plot run still open ends here, where
        the inline backend shows the open figures."""
        run, self._open = self._open, None
        if run is None:
            return
        if run.drew:
            self._note_drew(run)
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
        modules = _bound_modules(run.bound, self.shell.user_ns)
        if run.bound - set(modules):
            # Served, it would draw nothing and so have no figure, axes or
            # value to bind them to: a cell run later, in the notebook or
            # not, would find them unbound where plain Python has them. A
            # module it imported is bound again to the module (`take_place`).
            # ``_ = plt.plot(...)`` too: IPython's history then leaves the
            # user's ``_`` alone, and served, it would hold an older echo.
            names = ", ".join(sorted(run.bound - set(modules)))
            logger.debug("%s not kept: it binds %s", _LOG, names)
            return
        gone = run.names - self.shell.user_ns.keys()
        if gone:
            # ``del tmp``: served, the name would stay bound.
            logger.debug("%s not kept: it unbinds %s", _LOG, ", ".join(sorted(gone)))
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
        rng_reads = {rng_virtual_var(module): run.lineage.get(rng_virtual_var(module)) for module in run.rng_read}
        if any(value is None for value in rng_reads.values()):
            # Unseeded: a run from the top draws other numbers each time.
            logger.debug("%s not kept: it draws from an unseeded random stream", _LOG)
            return
        echo = None
        if run.echo is not None:
            # Kept as the value, which a hit shows and binds to ``_`` and
            # ``Out`` as a run does; one holding a figure, axes or other
            # artist of matplotlib's cannot be, and the run is not kept.
            echo = _echo_bytes(run.echo)
            if echo is None:
                logger.debug("%s not kept: what its last line echoed cannot be kept", _LOG)
                return
        size = sum(record_size(p[2]) + len(p[0]) + len(p[1]) for s in run.statements for p in s["parts"])
        size += record_size(flush or ()) + len(echo or b"")
        if size > PLOT_RECORD_CAP:
            return
        variants_key = self._key(code, inputs, run.occurrence, run.settings)
        if variants_key is None:
            return
        try:
            hidden = helper_memos.hidden_moved(run.hidden, self.shell.user_ns, fresh=run.fresh)
        except Exception:  # noqa: BLE001 - not known to call none: not kept
            hidden = "?"
        if hidden is not None:
            # What it left in that memo is in no record: served, the memo
            # stays as it was, and a later call computes (prints, seeds a
            # stream) where a run finds the answer.
            logger.debug("%s not kept: it may call the memo %s, which cash cannot find", _LOG, hidden)
            return
        backend = self._backend()
        if backend is None:
            return
        memo_records = [[records for records, _changes in stored["memos"]] for stored in run.statements]
        condition = _condition(run.env, rng_reads, memo_records)
        key = self._key(code, inputs, run.occurrence, run.settings, condition)
        if key is None:
            return
        payload = {"statements": run.statements, "flush": flush, "echo": echo}
        metadata = {
            "plot_outputs": True,
            "ttl": attribute_or(self._cell, "ttl", None),
            "execution_time": seconds,
            "size": size,
            "bound": sorted(run.bound | (_HISTORY_NAMES if echo is not None else set())),
            "env": run.env,
            "force_persist": bool(self._processor.persist_all),
            "rng_reads": rng_reads,
            "memo_records": memo_records,
            "modules": modules,
            "flushes": flush is not None,
            "echoes": echo is not None,
            "plot_format": _FORMAT,
        }
        backend.set(key, payload, metadata)
        kept = self._note_variant(
            backend,
            variants_key,
            condition,
            size,
            lambda c: self._key(code, inputs, run.occurrence, run.settings, c),
            self._served_last.get(variants_key),
        )
        self._note_shape(backend, run.codes, run.occurrence)
        logger.debug(
            "%s kept %d statement(s) under %s (inputs %s, settings %s, %d condition(s))",
            _LOG,
            len(run.codes),
            key[:24],
            sorted(inputs),
            run.settings[:12],
            kept,
        )

    @staticmethod
    def _note_variant(
        backend: Any,
        variants_key: str,
        condition: str,
        size: int,
        key_of: Callable[[str], str | None],
        served: Mapping[str, int] | None = None,
    ) -> int:
        """List *condition*, the run just kept (of *size*), first in the
        statements' record at *variants_key*, and drop from it the runs
        that can no longer be served: those no tier holds any more, and the
        least recently used ones the size a plot run may show leaves no
        room for. *served*: ``condition -> when`` of the runs served in
        this session (`_used_first`), listed before the others, the latest
        first. Gives how many it lists."""
        listed = _listed_variants(backend, variants_key)
        if served:
            order = {entry[0]: k for k, entry in enumerate(listed)}
            listed.sort(key=lambda entry: (-served.get(entry[0], -1), order[entry[0]]))
        kept: list[list] = [[condition, size]]
        total = size
        for other, other_size in listed:
            if other == condition:
                continue
            other_key = key_of(other)
            if other_key is None or backend.peek_metadata(other_key) is None:
                continue
            if total + other_size > PLOT_RECORD_CAP:
                # Too big beside the newer ones; a smaller older one may fit.
                _drop(backend, other_key)
                continue
            total += other_size
            kept.append([other, other_size])
        backend.set_metadata_only(variants_key, {"plot_variants": True, "variants": kept, "ttl": None})
        return len(kept)

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
        except Exception:  # noqa: BLE001 - a store that cannot answer: the plot is drawn
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
        variants_key = self._key(code, inputs, occurrence, settings)
        if variants_key is None:
            return None
        last = index + len(shape) - 1
        listed = _listed_variants(backend, variants_key)
        recent = self._served_last.get(variants_key)
        if recent:
            order = {entry[0]: k for k, entry in enumerate(listed)}
            listed.sort(key=lambda entry: (-recent.get(entry[0], -1), order[entry[0]]))
        # Each kept run is looked at without counting a use, and only one
        # that fits the memos as they stand: none is filled to fit, as which
        # of a memo's answers a run read is not recorded.
        for condition, _size in listed:
            key = self._key(code, inputs, occurrence, settings, condition)
            if key is None or not self._fits(backend.peek_metadata(key), shape, last, body, source):
                continue
            served = self._read_variant(backend, key, shape, index, last)
            if served is not None:
                self._promote = (variants_key, condition)
                return served
        return None

    def _read_variant(
        self, backend: Any, key: str, shape: list[str], index: int, last: int
    ) -> dict[int, ServedStatement] | None:
        """The statements of *shape* served from the run kept at *key*, or
        None when it cannot be read or put back here."""
        try:
            metadata, payload = backend.get(key)
        except Exception:  # noqa: BLE001 - an entry that cannot be read: the next is tried
            return None
        if not isinstance(payload, dict) or metadata is None or metadata.get("plot_format") != _FORMAT:
            return None
        statements = payload.get("statements") or []
        if len(statements) != len(shape):
            return None
        echo = _NO_ECHO
        if payload.get("echo") is not None:
            try:
                echo = pickle.loads(payload["echo"])
            except Exception:
                logger.debug("%s not served: its echoed value cannot be rebuilt", _LOG, exc_info=True)
                return None
        puts: list[list[Callable[[], None]]] = []
        for stored in statements:
            put = self._memo_puts(stored.get("memos") or [])
            if isinstance(put, str):
                logger.debug("%s not served: %s", _LOG, put)
                return None
            puts.append(put)
        return self._serve_variant(payload, metadata, puts, echo, shape, index, last, key)

    def _fits(self, metadata: Any, shape: list[str], last: int, body: list[ast.stmt], source: str) -> bool:
        """Whether the kept run whose metadata is *metadata* can stand in
        for a run of *shape*, ending at statement *last* of the cell, here."""
        if not isinstance(metadata, dict) or metadata.get("plot_format") != _FORMAT:
            return False
        if (metadata.get("flushes") or metadata.get("echoes")) and last != len(body) - 1:
            return False
        if not environment_unchanged(metadata.get("env") or {}):
            logger.debug("%s not served: an environment variable it read changed", _LOG)
            return False
        records = metadata.get("memo_records")
        if not isinstance(records, list) or len(records) != len(shape):
            return False
        bound = frozenset(metadata.get("bound") or ())
        modules = metadata.get("modules") or {}
        if bound - _HISTORY_NAMES - set(modules) or not all(name in sys.modules for name in modules.values()):
            # A module it imported that is not imported now: binding it
            # would import it, which a hit does not do.
            return False
        read = self._read_after(bound, body, last, source)
        if read is not False:
            logger.debug("%s not served: %s", _LOG, read)
            return False
        why = self._rng_moved_on(metadata.get("rng_reads") or {})
        if why is not None:
            logger.debug("%s not served: %s", _LOG, why)
            return False
        flat = [record for statement in records for memo in statement for record in memo or ()]
        try:
            why = helper_memos.unfilled(flat, self.shell.user_ns, refill=False)
        except Exception:  # noqa: BLE001 - a record this cannot read: the plot is drawn
            return False
        if why is not None:
            logger.debug("%s not served: %s", _LOG, why)
            return False
        return True

    def _serve_variant(
        self,
        payload: dict,
        metadata: dict,
        puts: list[list[Callable[[], None]]],
        echo: Any,
        shape: list[str],
        index: int,
        last: int,
        key: str,
    ) -> dict[int, ServedStatement]:
        """The statements of *shape* from *index* on, served from the kept
        run *payload* (with *metadata*; `_memo_puts` gave *puts*), its
        last line echoing *echo* unless that is `_NO_ECHO`."""
        statements = payload["statements"]
        bound = frozenset(metadata.get("bound") or ())
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
            served[index + k] = ServedStatement(
                row,
                [tuple(p) for p in stored.get("parts") or []],
                dict(stored.get("rng_state") or {}),
                frozenset(stored.get("rng_reseeded") or ()),
                puts[k],
            )
        if echo is not _NO_ECHO:
            served[last].metrics[ECHO_FIELD] = echo
        modules = {name: sys.modules[module] for name, module in (metadata.get("modules") or {}).items()}
        self._pending = (payload.get("flush") or None, bound, modules)
        logger.debug("%s serving %d statement(s) from %s", _LOG, len(shape), key[:24])
        return served

    def take_place(self, served: dict[int, ServedStatement], texts: list[tuple[str, int]]) -> bool:
        """The plot run *served* (from `serve`, its statements' code and
        occurrence in *texts*, in order) takes its place, before any of it
        is shown: the random streams and memos its run changed are left as
        that run left them, the RNG ledgers move on as after it
        (`settle_served_plot_run`), what it showed after the cell is queued
        and the names it bound are dropped. False when that cannot be done:
        nothing is changed then, and its statements are to run."""
        pending, self._pending = self._pending, None
        items = [served[k] for k in sorted(served)]
        if pending is None or len(items) != len(texts):
            self._promote = None
            return False
        settled = self._processor.settle_served_plot_run(
            [(code, occ, item.rng_state, item.rng_reseeded) for (code, occ), item in zip(texts, items, strict=True)]
        )
        if not settled:
            self._promote = None
            return False
        for item in items:
            for put in item.memo_puts:
                put()
        flush, _bound, modules = pending
        if flush:
            self._show_after_cell(flush)
        # The only names a kept run binds besides IPython's history: the
        # modules it imported, bound as its import statement binds them.
        self.shell.user_ns.update(modules)
        self._used_first()
        logger.debug("%s served %d statement(s)", _LOG, len(items))
        return True

    def _used_first(self) -> None:
        """Note the kept run just served as the most recently used of its
        statements' record, for the next keep to list first."""
        promote, self._promote = self._promote, None
        if promote is None:
            return
        variants_key, condition = promote
        self._serves += 1
        if len(self._served_last) > 4096:
            self._served_last.clear()
        self._served_last.setdefault(variants_key, {})[condition] = self._serves

    def _rng_moved_on(self, reads: dict[str, str]) -> str | None:
        """Why the random streams a kept plot run drew from (*reads*, the
        RNG variable and its lineage then) no longer stand where they did,
        or None."""
        if not reads:
            return None
        self._adopt_unseen_seeds()
        state = self._processor.tracking_state
        for var, lineage in reads.items():
            if state.variable_lineage.get(var) != lineage:
                return f"the random stream {var} stands elsewhere"
            if any(rng_virtual_var(module) == var for module in state.rng_unplaced):
                return f"the random stream {var} could not be placed"
        return None

    def _memo_puts(self, memos: list) -> list[Callable[[], None]] | str:
        """What puts back what one statement of a kept plot run added to
        the helper memos (*memos*, each ``[records, changes]``), or why
        that cannot be done: a memo it read lacks what it read then."""
        user_ns = self.shell.user_ns
        puts: list[Callable[[], None]] = []
        for records, changes in memos:
            try:
                why = helper_memos.unfilled(records, user_ns, refill=False)
                put = why if why is not None else helper_memos.put_back(records, changes, user_ns)
            except Exception:  # noqa: BLE001 - a record this cannot read: the plot is drawn
                return "its helper memos cannot be read back"
            if isinstance(put, str):
                return put
            puts.append(put)
        return puts

    def _adopt_unseen_seeds(self) -> None:
        try:
            self._processor.adopt_unseen_seeds()
        except Exception:
            logger.debug("%s could not adopt an unseen seed", _LOG, exc_info=True)

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
        except Exception:  # noqa: BLE001 - cells that cannot be read: nothing is served
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

    def _key(
        self, code: str, inputs: set[str], occurrence: int, settings: str, condition: str | None = None
    ) -> str | None:
        """The key of the plot run *code* with *inputs*, under matplotlib's
        *settings*: ``compute_cache_key`` in the ``plot`` namespace, the
        settings a line of its text. Without a *condition* (`_condition`),
        the key of the record listing the runs kept of it; with one, the
        key of the run kept under it, the condition another line."""
        text = f"{code}\n# cash: matplotlib settings {settings}"
        if condition is not None:
            text += f"\n# cash: plot condition {condition}"
        try:
            result = compute_cache_key(
                text,
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
        except Exception:  # noqa: BLE001 - no cache: nothing is kept
            return None

    def _note_shape(self, backend: Any, codes: list[str], occurrence: int) -> None:
        key = plot_shapes_key(codes[0], occurrence)
        try:
            record = backend.get_metadata(key) or {}
        except Exception:  # noqa: BLE001 - a store that cannot answer: no shapes known
            record = {}
        shapes = [s for s in (record.get("shapes") or []) if s != codes]
        shapes.insert(0, list(codes))
        backend.set_metadata_only(key, {"plot_shapes": True, "shapes": shapes[:_SHAPES_KEPT], "ttl": None})

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


#: IPython's output history, which a served echo binds as a run does: what
#: the cells below are checked for (`first_read_after`, which counts a read
#: of a numbered ``_<n>`` as one of them). The input history (``In``,
#: ``_i``, ``_i<n>``) is written for a served cell as for any other.
_HISTORY_NAMES = frozenset({"_", "__", "___", "Out", "_oh"})


def _listed_variants(backend: Any, variants_key: str) -> list[list]:
    """``[condition, size]`` of each run kept of a plot run's statements,
    most recently used first, as their record at *variants_key* lists them,
    read without counting a use."""
    try:
        record = backend.peek_metadata(variants_key) or {}
    except Exception:  # noqa: BLE001 - a store that cannot answer: none is listed
        return []
    listed = record.get("variants") if record.get("plot_variants") else None
    return [
        [entry[0], entry[1]]
        for entry in listed or ()
        if isinstance(entry, (list, tuple)) and len(entry) == 2 and isinstance(entry[0], str)
    ]


def _drop(backend: Any, key: str) -> None:
    try:
        backend.delete(key)
    except Exception:  # left to the store's own eviction
        logger.debug("%s could not drop %s", _LOG, key[:24], exc_info=True)


#: A served run whose last line echoed nothing.
_NO_ECHO = object()

#: The packages whose objects an echoed value may not hold: a figure, axes
#: or artist of theirs is the live figure a run drew, which a kept copy is
#: not (and a copy of a figure registers itself with pyplot).
_DRAWING_PACKAGES = frozenset({"matplotlib", "mpl_toolkits"})


class _HoldsAFigure(Exception):
    pass


class _EchoPickler(pickle.Pickler):
    def persistent_id(self, obj: Any) -> None:
        module = attribute_or(type(obj), "__module__", None) or ""
        if module.partition(".")[0] in _DRAWING_PACKAGES:
            raise _HoldsAFigure(module)
        return None


def _echo_bytes(value: Any) -> bytes | None:
    """*value*, what the last line of a plot run echoed, as the bytes a
    hit rebuilds it from, or None when it cannot be kept: it holds an
    object of matplotlib's, or it cannot be pickled."""
    buffer = io.BytesIO()
    try:
        _EchoPickler(buffer, protocol=pickle.HIGHEST_PROTOCOL).dump(value)
    except Exception:  # noqa: BLE001 - a figure, or a value pickle refuses
        return None
    return buffer.getvalue()


def _bound_modules(names: set[str], user_ns: Mapping[str, Any]) -> dict[str, str]:
    """``name -> module name`` of each of *names* bound to a module, as
    ``import pandas as pd`` binds one."""
    out: dict[str, str] = {}
    for name in names:
        value = user_ns.get(name)
        module = attribute_or(value, "__name__", None) if isinstance(value, types.ModuleType) else None
        if isinstance(module, str) and sys.modules.get(module) is value:
            out[name] = module
    return out


def _statement_binds(code: str) -> set[str] | None:
    """The names the top-level statement *code* may bind or unbind in the
    notebook's namespace, in any branch or pass, or None when it cannot be
    parsed: its targets, its imports, the functions and classes it defines,
    ``except ... as`` and ``match`` captures, ``del``, and a walrus, also
    one inside a comprehension. Not what a function or class body binds:
    a helper setting a global is caught when it runs."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    names: set[str] = set()

    def visit(node: ast.AST, in_comprehension: bool) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            for sub in [*node.decorator_list, *attribute_or(node, "bases", ()), *attribute_or(node, "keywords", ())]:
                visit(sub, in_comprehension)
            if not isinstance(node, ast.ClassDef):
                for default in [*node.args.defaults, *[d for d in node.args.kw_defaults if d is not None]]:
                    visit(default, in_comprehension)
            return
        if isinstance(node, ast.Lambda):
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for sub in ast.iter_child_nodes(node):
                visit(sub, True)
            return
        if isinstance(node, ast.comprehension):
            # Its own loop variable is the comprehension's.
            visit(node.iter, in_comprehension)
            for sub in node.ifs:
                visit(sub, in_comprehension)
            return
        if isinstance(node, ast.NamedExpr):
            names.add(node.target.id)
            visit(node.value, in_comprehension)
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and not in_comprehension:
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name != "*":
                    names.add((alias.asname or alias.name).split(".")[0])
                else:
                    names.add("*")
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
        for sub in ast.iter_child_nodes(node):
            visit(sub, in_comprehension)

    for statement in tree.body:
        visit(statement, False)
    return names
