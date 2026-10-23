"""Run a notebook or decorator shape under cash and under plain Python, and
compare what each run printed and holds.

A shape is a small repro: notebook cells (with optional edits between
runs), or a function with a call sequence. `check` runs it under cash in
each config (`CONFIGS`) and once in plain Python, through the same run
sequence, and returns a `Report` of every place the two differ, with
cash's status for each statement. See README.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["CONFIGS", "RUNS", "Outcome", "Report", "Shape", "check", "decorator", "notebook"]

#: The run sequence: Run All three times, a restart, Run All twice.
RUNS: tuple[str, ...] = ("run1", "run2", "run3", "restart+run1", "restart+run2")

#: Config name -> environment for the cash side. ``floors0`` also sets
#: ``call_unit._COST_FLOOR_S = 0`` in the kernel (`kernel.FLOORS0_SNIPPET`).
CONFIGS: dict[str, dict[str, str]] = {
    "trace": {},
    "notrace": {"CASH_TRACE": "0"},
    "floors0": {
        "CASH_MIN_EXECUTION_TIME_TO_CACHE_SECONDS": "0",
        "CASH_CALL_COST_FLOOR_SECONDS": "0",
    },
}


@dataclass
class Shape:
    """One repro. *kind* is ``"notebook"`` or ``"decorator"``."""

    name: str
    kind: str
    cells: list[str] = field(default_factory=list)
    #: Before run index -> {cell index: new source}; runs are indexed as in `RUNS`.
    edits: dict[int, dict[int, str]] = field(default_factory=dict)
    #: Variables compared after each run; None compares every top-level name
    #: the cells assign (not functions, classes or modules).
    names: list[str] | None = None
    setup: str = ""
    steps: list[str] = field(default_factory=list)
    source: str = ""

    def cells_for(self, run: int) -> list[str]:
        """The cells as they stand for run *run*, every earlier edit applied."""
        cells = list(self.cells)
        for at in sorted(self.edits):
            if at <= run:
                for index, code in self.edits[at].items():
                    cells[index] = code
        return cells


def notebook(
    name: str,
    cells: list[str],
    *,
    edits: dict[int, dict[int, str]] | None = None,
    names: list[str] | None = None,
    source: str = "",
) -> Shape:
    return Shape(name, "notebook", cells=list(cells), edits=dict(edits or {}), names=names, source=source)


def decorator(name: str, setup: str, steps: list[str], *, source: str = "") -> Shape:
    """*setup* defines the functions, decorated with the bare name ``cache``
    (cash's decorator, or nothing in plain Python); *steps* are statements
    run in order, and a bare expression's value is compared."""
    return Shape(name, "decorator", setup=setup, steps=list(steps), source=source)


@dataclass
class Outcome:
    """What one side saw: run -> {what: fingerprint}, and per run and cell
    the statuses of its statements (cash only)."""

    seen: dict[str, dict[str, str]] = field(default_factory=dict)
    statuses: dict[str, dict[int, list[str]]] = field(default_factory=dict)

    def without_paths(self, work: Any) -> Outcome:
        """Each side runs in its own folder: its path, as a module's or a
        file's repr shows it, reads ``<work>``."""
        import os

        paths = sorted({str(work), os.path.realpath(work)}, key=len, reverse=True)

        def clean(text: str) -> str:
            for path in paths:
                text = text.replace(path, "<work>")
            return text

        self.seen = {run: {where: clean(v) for where, v in seen.items()} for run, seen in self.seen.items()}
        return self


@dataclass
class Report:
    shape: str
    #: (config, run, where, cash, plain): each place cash and plain differ.
    mismatches: list[tuple[str, str, str, str, str]] = field(default_factory=list)
    #: The same, where cash's value is what a fresh kernel's Run All of the
    #: same cells gives: allowed where cash rebuilds what the cells hold
    #: (a generator, a module's state) rather than drawing on it further.
    rebuilt: list[tuple[str, str, str, str, str]] = field(default_factory=list)
    #: config -> run -> cell index -> statuses of its statements.
    statuses: dict[str, dict[str, dict[int, list[str]]]] = field(default_factory=dict)
    #: (config, error): a side that could not run at all.
    errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Nothing differs, or only as a fresh kernel's Run All would."""
        return not self.mismatches and not self.errors

    @property
    def verdict(self) -> str:
        return "BAD" if not self.ok else "REBUILT" if self.rebuilt else "OK"

    def text(self, show_statuses: bool = False) -> str:
        lines = [f"{self.verdict:7} {self.shape}"]
        for config, error in self.errors:
            lines.append(f"    [{config}] could not run: {error}")
        for config, run, where, cash, plain in self.mismatches:
            lines.append(f"    [{config}] {run} {where}: cash={cash!r} plain={plain!r}")
        for config, run, where, cash, plain in self.rebuilt:
            lines.append(f"    [{config}] {run} {where}: cash={cash!r} plain={plain!r} (as a fresh kernel)")
        if show_statuses or not self.ok:
            for config, runs in self.statuses.items():
                for run, cells in runs.items():
                    shown = "  ".join(f"c{i}:{','.join(s) or '-'}" for i, s in sorted(cells.items()))
                    lines.append(f"    [{config}] {run} statuses  {shown}")
        return "\n".join(lines)

    def assert_ok(self, *, allow_rebuilt: bool = True) -> None:
        assert self.ok and (allow_rebuilt or not self.rebuilt), self.text()


def _fresh_runs(shape: Shape) -> dict[str, str]:
    """Run -> the plain run that is a fresh kernel's Run All of the same
    cells: the first, or the first after the restart. A run whose cells
    neither has (an edit in between) has none."""
    fresh = {}
    for index, run_name in enumerate(RUNS):
        for first in (0, RUNS.index("restart+run1")):
            if first <= index and shape.cells_for(index) == shape.cells_for(first):
                fresh[run_name] = RUNS[first]
    return fresh


def check(shape: Shape, work_dir: Any, configs: list[str] | None = None) -> Report:
    """Run *shape* in each of *configs* (default: all of `CONFIGS`) and in
    plain Python, under *work_dir* (a fresh folder per side is made in it)."""
    from pathlib import Path

    work = Path(work_dir)
    report = Report(shape.name)
    if shape.kind == "notebook":
        from tools.plain_checker.kernel import run_notebook as run
    else:
        from tools.plain_checker.deco import run_decorator as run
    try:
        plain = run(shape, work / "plain", None)
    except Exception as exc:
        report.errors.append(("plain", repr(exc)))
        return report
    fresh = _fresh_runs(shape)
    for config in configs or list(CONFIGS):
        try:
            got = run(shape, work / config, config)
        except Exception as exc:
            report.errors.append((config, repr(exc)))
            continue
        report.statuses[config] = got.statuses
        for run_name in RUNS:
            mine, theirs = got.seen.get(run_name, {}), plain.seen.get(run_name, {})
            first = plain.seen.get(fresh[run_name], {}) if run_name in fresh else {}
            for where in sorted(set(mine) | set(theirs), key=str):
                if mine.get(where) == theirs.get(where):
                    continue
                found = (config, run_name, where, str(mine.get(where)), str(theirs.get(where)))
                if where in first and mine.get(where) == first[where]:
                    report.rebuilt.append(found)
                else:
                    report.mismatches.append(found)
    return report
