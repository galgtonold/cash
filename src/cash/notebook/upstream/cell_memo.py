"""One cell's simulation, kept for the next check to reuse.

The simulation cache (:class:`~._types.SimulationCache`) keeps a snapshot
per cell, but only for the cells above the cell last checked, and only up
to the first one that changed: running a cell near the top dropped every
snapshot below it, and an edit near the top re-simulated every cell below
the edit, though most of them read nothing it changed. Each simulated
statement costs a few milliseconds, so a check grew with the notebook.

A :class:`CellMemo` is what simulating one cell did: what it read of the
simulation state it started from (each name whose lineage it looked up, by
a recording mapping, :class:`RecordingLineage`) and what it changed. The
next simulation of the same cell text at the same place, starting from a
state that agrees on everything it read, applies the change instead of
simulating the cell again. Whatever the cell's simulation saw as a whole
(the module names, the share edges, the loop and stale-file sets) must be
equal as a whole; a memo that saw the whole lineage map is never reused.

The memo keeps the guards the cache's own scan keeps (text, magics, the
statement a failed run stopped at, the environment, the files, the share
records), and is dropped when a statement of its cell runs, when the
notebook's functions or a module change, and when a repair re-records a
lineage. ``CASH_VERIFY_INCREMENTAL_SIM`` also simulates each cell a memo
stands in for, from the same state, and fails when the two differ
(:func:`differences`).
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._types import SimulationCacheEntry, SimulationResult, TraceEntry

__all__ = [
    "CellMemo",
    "RecordingLineage",
    "differences",
    "namespace_token",
    "verify_requested",
]

#: Set to simulate each cell a memo stands in for as well, and fail when the
#: two differ (the test suites run with it).
VERIFY_ENV = "CASH_VERIFY_INCREMENTAL_SIM"
#: A file each difference is appended to, for a kernel the test cannot see into.
VERIFY_LOG_ENV = "CASH_VERIFY_INCREMENTAL_SIM_LOG"


def verify_requested() -> bool:
    """Whether ``CASH_VERIFY_INCREMENTAL_SIM`` asks to check each memo used."""
    return os.environ.get(VERIFY_ENV, "") not in ("", "0")


class RecordingLineage(dict):
    """The virtual lineage map, noting what one cell's simulation reads of it.

    A name looked up before the cell wrote it is a read (*reads*); one it
    set or deleted is a write (*written*, in the order of the last write).
    Anything that sees the map as a whole (iterating it, its length, a copy,
    a comparison) sets *whole*: a memo of that cell is never reused.
    """

    __slots__ = ("reads", "written", "whole", "recording", "empty")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reads: dict[str, Any] = {}
        self.written: dict[str, None] = {}
        self.whole = False
        self.recording = False
        #: Whether the map was empty when the cell started, where the cell
        #: asked (``virtual_lineage or {}``); None where it did not.
        self.empty: bool | None = None

    def start(self) -> None:
        """Begin recording one cell."""
        self.reads = {}
        self.written = {}
        self.whole = False
        self.empty = None
        self.recording = True

    def stop(self) -> None:
        self.recording = False

    def _read(self, key: Any) -> None:
        if self.recording and key not in self.written and key not in self.reads:
            self.reads[key] = dict.get(self, key, _MISSING)

    def _wrote(self, key: Any) -> None:
        if self.recording:
            self.written.pop(key, None)
            self.written[key] = None

    def _all(self) -> None:
        if self.recording:
            self.whole = True

    # Reads of one name.
    def __getitem__(self, key: Any) -> Any:
        self._read(key)
        return dict.__getitem__(self, key)

    def get(self, key: Any, default: Any = None) -> Any:
        self._read(key)
        return dict.get(self, key, default)

    def __contains__(self, key: object) -> bool:
        self._read(key)
        return dict.__contains__(self, key)

    # Writes.
    def __setitem__(self, key: Any, value: Any) -> None:
        self._wrote(key)
        dict.__setitem__(self, key, value)

    def __delitem__(self, key: Any) -> None:
        self._read(key)
        self._wrote(key)
        dict.__delitem__(self, key)

    def pop(self, key: Any, *default: Any) -> Any:
        self._read(key)
        self._wrote(key)
        return dict.pop(self, key, *default)

    def setdefault(self, key: Any, default: Any = None) -> Any:
        self._read(key)
        self._wrote(key)
        return dict.setdefault(self, key, default)

    def update(self, *args: Any, **kwargs: Any) -> None:
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def __ior__(self, other: Any) -> RecordingLineage:
        self.update(other)
        return self

    # Reads of the whole map.
    def __iter__(self):
        self._all()
        return dict.__iter__(self)

    def __len__(self) -> int:
        self._all()
        return dict.__len__(self)

    def __bool__(self) -> bool:
        size = dict.__len__(self)
        if self.recording:
            if not self.written:
                if self.empty is None:
                    self.empty = size == 0
            elif not any(dict.__contains__(self, key) for key in self.written):
                self._all()  # emptied or not by what it deleted: unknown
        return size > 0

    def __reversed__(self):
        self._all()
        return dict.__reversed__(self)

    def keys(self):
        self._all()
        return dict.keys(self)

    def values(self):
        self._all()
        return dict.values(self)

    def items(self):
        self._all()
        return dict.items(self)

    def copy(self) -> dict:
        self._all()
        return dict(dict.items(self))

    def __eq__(self, other: object) -> bool:
        self._all()
        return dict.__eq__(self, other)

    def __ne__(self, other: object) -> bool:
        self._all()
        return dict.__ne__(self, other)

    __hash__ = None  # type: ignore[assignment]

    def __or__(self, other: Any) -> dict:
        self._all()
        return dict.__or__(dict(dict.items(self)), other)

    def __ror__(self, other: Any) -> dict:
        self._all()
        return dict.__or__(dict(other), dict(dict.items(self)))

    def __reduce_ex__(self, protocol: Any) -> Any:
        self._all()
        return (dict, (dict(dict.items(self)),))

    def __repr__(self) -> str:
        self._all()
        return dict.__repr__(self)

    def popitem(self) -> tuple[Any, Any]:
        self._all()
        key, value = dict.popitem(self)
        self._wrote(key)
        return key, value

    def clear(self) -> None:
        self._all()
        dict.clear(self)

    def plain(self) -> dict:
        """The map as a plain dict, not noted as a read."""
        return dict(dict.items(self))


def namespace_token(value: Any) -> Any:
    """What the simulation may have seen of the kernel's *value*: its
    identity for code and modules (a function's body is analysed), else
    its type. ``_MISSING`` when the name is unbound."""
    if value is _MISSING:
        return _MISSING
    kind = type(value)
    if callable(value) or kind.__name__ in _BY_IDENTITY or hasattr(kind, "__next__"):
        return (kind, id(value))
    return kind


#: Types whose instances the simulation looks into (modules, generators,
#: partials, methods), compared by identity.
_BY_IDENTITY = frozenset({"module", "generator", "coroutine", "async_generator", "partial", "method"})


class _Missing:
    def __repr__(self) -> str:
        return "<missing>"


_MISSING: Any = _Missing()


def missing() -> Any:
    """The token of an unbound name or an absent lineage."""
    return _MISSING


@dataclass(slots=True)
class CellMemo:
    """What simulating one cell read and did (see the module docstring)."""

    entry: SimulationCacheEntry
    """The snapshot the simulation appended for the cell."""
    reads: dict[str, tuple[Any, Any, Any]]
    """Each name whose virtual lineage the cell looked up before writing it:
    ``(virtual lineage, live lineage, namespace token)`` at the cell's start."""
    empty: bool | None
    """Whether the virtual lineage was empty at the cell's start, where the
    cell asked; None where it did not."""
    watched: dict[str, tuple[Any, Any]]
    """Each other name its statements read or wrote: ``(live lineage,
    namespace token)`` when it was simulated."""
    names: frozenset[str]
    """The names the cell can mention (``mentioned_names``), which scope
    the module names and loop targets of *before* and *after*."""
    functions: dict[str, str]
    """The source of each notebook function among *names* (the simulation
    resolves the functions a statement calls from the cell text)."""
    before: tuple
    """The whole-state fields at the cell's start (``state_of``, scoped)."""
    after: tuple
    """The same fields after it."""
    written: dict[str, Any]
    """Each name it set, in the order of its last write, with the lineage it
    left; ``_MISSING`` for a name it deleted."""
    lookup_times: dict[str, float]
    """The statement lookup times it set."""
    trace: list[TraceEntry]
    """Its trace entries."""
    generations: tuple
    """``(module generation, magic generation)``."""
    statements: frozenset[str] = field(default_factory=frozenset)
    """Every statement of the cell, nested ones too (``statements_of``)."""


def state_of(sim: SimulationResult, names: frozenset[str] | None = None) -> tuple:
    """The fields of *sim* a cell's simulation is compared on as a whole.

    With *names*, the module names and loop targets only as far as they are
    among *names*: a cell asks those sets about the names it mentions, and
    an import added above it changes nothing it can see."""

    def scoped(values: set[str]) -> frozenset[str]:
        return frozenset(values) if names is None else frozenset(values & names)

    return (
        scoped(sim.virtual_modules),
        None if sim.share_edges is None else {src: frozenset(t) for src, t in sim.share_edges.items()},
        frozenset(sim.vars_with_stale_files),
        frozenset(sim.vars_mutated_by_loops),
        frozenset(sim.vars_maybe_changed_by_loops),
        scoped(sim.loop_target_vars),
        dict(sim.vars_left_by_failed_runs),
    )


def put_state(sim: SimulationResult, before: tuple, after: tuple) -> None:
    """Make in *sim* what a cell changed from *before* to *after* (``state_of``,
    the module names and loop targets scoped, the rest whole)."""
    modules, edges, stale, looped, maybe, targets, left = after
    sim.virtual_modules.difference_update(before[0] - modules)
    sim.virtual_modules.update(modules - before[0])
    sim.loop_target_vars.difference_update(before[5] - targets)
    sim.loop_target_vars.update(targets - before[5])
    sim.share_edges = None if edges is None else {src: set(t) for src, t in edges.items()}
    sim.vars_with_stale_files.clear()
    sim.vars_with_stale_files.update(stale)
    sim.vars_mutated_by_loops.clear()
    sim.vars_mutated_by_loops.update(looped)
    sim.vars_maybe_changed_by_loops.clear()
    sim.vars_maybe_changed_by_loops.update(maybe)
    sim.vars_left_by_failed_runs.clear()
    sim.vars_left_by_failed_runs.update(left)


_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")


def mentioned_names(cell_code: str, functions: Mapping[str, str]) -> frozenset[str]:
    """Every identifier in *cell_code*, and in the source of each notebook
    function (*functions*, name to source) one of them reaches, transitively.
    A superset of the names the cell's simulation asks the module and loop
    target sets about."""
    names = set(_IDENTIFIER.findall(cell_code))
    todo = [name for name in names if name in functions]
    seen: set[str] = set()
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        for found in _IDENTIFIER.findall(functions[name]):
            if found not in names:
                names.add(found)
                if found in functions:
                    todo.append(found)
    return frozenset(names)


def apply_written(virtual_lineage: dict, written: Mapping[str, Any]) -> None:
    """Make in *virtual_lineage* the writes a memo recorded, in their order."""
    for name, lineage in written.items():
        if lineage is _MISSING:
            virtual_lineage.pop(name, None)
        else:
            virtual_lineage[name] = lineage


# --- Verify mode -----------------------------------------------------------


def _trace_row(entry: TraceEntry) -> tuple:
    hashes = entry.input_hashes
    extra = (
        dict(getattr(hashes, "callee_lineages", None) or {}),
        dict(getattr(hashes, "hidden_lineages", None) or {}),
    )
    return (
        entry.stmt_code,
        frozenset(entry.outputs),
        frozenset(entry.inputs),
        dict(hashes),
        extra,
        dict(entry.produced_lineages),
        bool(entry.files_stale),
        entry.cell,
        dict(getattr(entry, "stored_lineages", None) or {}),
    )


def _entry_row(entry: SimulationCacheEntry) -> tuple:
    return (
        entry.cell_code_hash,
        dict(entry.virtual_lineage),
        frozenset(entry.virtual_modules),
        [_trace_row(t) for t in entry.trace_segment],
        frozenset(entry.vars_mutated_by_loops),
        frozenset(entry.vars_with_stale_files),
        dict(entry.cell_file_deps),
        entry.cell_environment,
        entry.stopped_at,
        entry.magic_generation,
        dict(entry.vars_left_by_failed_runs or {}),
        None if entry.share_edges is None else {k: frozenset(v) for k, v in entry.share_edges.items()},
        entry.statement_hashes,
        frozenset(entry.vars_maybe_changed_by_loops or ()),
    )


def differences(
    incremental: SimulationResult,
    reference: SimulationResult,
    incremental_entries: list[SimulationCacheEntry],
    reference_entries: list[SimulationCacheEntry],
) -> list[str]:
    """Where a simulation reusing memos differs from one that does not:
    lineages (rng positions are lineages), modules, share edges, trace (the
    taint and the plan are read from it), stale-file and loop sets, and the
    snapshots (their file dependencies). Empty when they agree."""
    found: list[str] = []

    def compare(what: str, a: Any, b: Any) -> None:
        if a != b:
            found.append(f"{what}: incremental={a!r} reference={b!r}")

    vl_a, vl_b = dict(incremental.virtual_lineage), dict(reference.virtual_lineage)
    if vl_a != vl_b:
        diff = {k: (vl_a.get(k), vl_b.get(k)) for k in vl_a.keys() | vl_b.keys() if vl_a.get(k) != vl_b.get(k)}
        found.append(f"virtual_lineage differs: {diff!r}")
    compare("virtual_modules", set(incremental.virtual_modules), set(reference.virtual_modules))
    compare("share_edges", incremental.share_edges, reference.share_edges)
    compare("vars_with_stale_files", incremental.vars_with_stale_files, reference.vars_with_stale_files)
    compare("vars_mutated_by_loops", incremental.vars_mutated_by_loops, reference.vars_mutated_by_loops)
    compare(
        "vars_maybe_changed_by_loops", incremental.vars_maybe_changed_by_loops, reference.vars_maybe_changed_by_loops
    )
    compare("loop_target_vars", incremental.loop_target_vars, reference.loop_target_vars)
    compare("vars_left_by_failed_runs", incremental.vars_left_by_failed_runs, reference.vars_left_by_failed_runs)
    compare("stmt_lookup_times keys", set(incremental.stmt_lookup_times), set(reference.stmt_lookup_times))
    trace_a = [_trace_row(t) for t in incremental.trace]
    trace_b = [_trace_row(t) for t in reference.trace]
    if trace_a != trace_b:
        for i, (a, b) in enumerate(zip(trace_a, trace_b, strict=False)):
            if a != b:
                found.append(f"trace[{i}]: incremental={a!r} reference={b!r}")
                break
        else:
            found.append(f"trace length: incremental={len(trace_a)} reference={len(trace_b)}")
    if len(incremental_entries) != len(reference_entries):
        found.append(f"snapshots: incremental={len(incremental_entries)} reference={len(reference_entries)}")
    for i, (a, b) in enumerate(zip(incremental_entries, reference_entries, strict=False)):
        row_a, row_b = _entry_row(a), _entry_row(b)
        if row_a != row_b:
            fields = [n for n, (x, y) in enumerate(zip(row_a, row_b, strict=True)) if x != y]
            found.append(f"snapshot {i} differs in fields {fields}: {[(row_a[n], row_b[n]) for n in fields]!r}")
    return found
