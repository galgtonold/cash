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


@dataclass
class Report:
    shape: str
    #: (config, run, where, cash, plain): each place cash and plain differ.
    mismatches: list[tuple[str, str, str, str, str]] = field(default_factory=list)
    #: config -> run -> cell index -> statuses of its statements.
    statuses: dict[str, dict[str, dict[int, list[str]]]] = field(default_factory=dict)
    #: (config, error): a side that could not run at all.
    errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches and not self.errors

    def text(self, show_statuses: bool = False) -> str:
        lines = [f"{'OK ' if self.ok else 'BAD'} {self.shape}"]
        for config, error in self.errors:
            lines.append(f"    [{config}] could not run: {error}")
        for config, run, where, cash, plain in self.mismatches:
            lines.append(f"    [{config}] {run} {where}: cash={cash!r} plain={plain!r}")
        if show_statuses or not self.ok:
            for config, runs in self.statuses.items():
                for run, cells in runs.items():
                    shown = "  ".join(f"c{i}:{','.join(s) or '-'}" for i, s in sorted(cells.items()))
                    lines.append(f"    [{config}] {run} statuses  {shown}")
        return "\n".join(lines)

    def assert_ok(self) -> None:
        assert self.ok, self.text()


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
    for config in configs or list(CONFIGS):
        try:
            got = run(shape, work / config, config)
        except Exception as exc:
            report.errors.append((config, repr(exc)))
            continue
        report.statuses[config] = got.statuses
        for run_name in RUNS:
            mine, theirs = got.seen.get(run_name, {}), plain.seen.get(run_name, {})
            for where in sorted(set(mine) | set(theirs), key=str):
                if mine.get(where) != theirs.get(where):
                    report.mismatches.append((config, run_name, where, str(mine.get(where)), str(theirs.get(where))))
    return report
