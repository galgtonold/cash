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

__all__ = ["CONFIGS", "PLAN", "RUNS", "Outcome", "Report", "Shape", "check", "decorator", "notebook"]

#: The default plan: Run All three times, a restart, Run All twice. A
#: shape can give its own: "all", "restart", or a cell's index to run alone.
PLAN: tuple[str | int, ...] = ("all", "all", "all", "restart", "all", "all")

#: The default plan's runs, by name.
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
    #: Before run index -> {cell index: new source}; the runs of the plan
    #: (each step but a restart) are indexed from 0.
    edits: dict[int, dict[int, str]] = field(default_factory=dict)
    #: The steps, as `PLAN`; None runs `PLAN`.
    plan: list[str | int] | None = None
    #: Variables compared after each run; None compares every top-level name
    #: the cells assign (not functions, classes or modules).
    names: list[str] | None = None
    #: Variables that read module state a statement writes (a counter a
    #: function bumps). There cash answers as a restart and a Run All would
    #: (documented: the write is put back on a hit), not as the same kernel
    #: run again would; a difference on them that a fresh kernel's Run All
    #: gives is accepted (``AS-RESTART``).
    module_writes: list[str] = field(default_factory=list)
    #: With a plan: a cell re-run alone answers as the Run All before it
    #: (cash re-runs the cells above a draw's tail), so a difference there
    #: that the Run All gave counts as one on a fresh kernel; only when the
    #: run shows the rebuild (`_shows_a_rebuild`), as a stale hit gives the
    #: Run All's value too.
    alone_as_run_all: bool = False
    setup: str = ""
    steps: list[str] = field(default_factory=list)
    source: str = ""

    def runs(self) -> list[tuple[str, bool, list[int] | None]]:
        """Each run of the plan: its name, whether a restart comes first,
        and the cells it runs (None: all of them)."""
        out: list[tuple[str, bool, list[int] | None]] = []
        restarts, alls, restart_first = 0, 0, False
        for step in self.plan or PLAN:
            if step == "restart":
                restarts, alls, restart_first = restarts + 1, 0, True
                continue
            prefix = "" if not restarts else "restart+" if restarts == 1 else f"restart{restarts}+"
            if step == "all":
                alls += 1
                out.append((f"{prefix}run{alls}", restart_first, None))
            else:
                out.append((f"{prefix}step{len(out) + 1}", restart_first, [int(step)]))
            restart_first = False
        return out

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
    plan: list[str | int] | None = None,
    module_writes: list[str] | None = None,
    alone_as_run_all: bool = False,
    source: str = "",
) -> Shape:
    return Shape(
        name,
        "notebook",
        cells=list(cells),
        edits=dict(edits or {}),
        names=names,
        plan=plan,
        module_writes=list(module_writes or ()),
        alone_as_run_all=alone_as_run_all,
        source=source,
    )


def decorator(
    name: str, setup: str, steps: list[str], *, module_writes: list[str] | None = None, source: str = ""
) -> Shape:
    """*setup* defines the functions, decorated with the bare name ``cache``
    (cash's decorator, or nothing in plain Python); *steps* are statements
    run in order, and a bare expression's value is compared (``"step 3
    value"`` is what *module_writes* names for step 3)."""
    return Shape(
        name, "decorator", setup=setup, steps=list(steps), module_writes=list(module_writes or ()), source=source
    )


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
    #: same cells gives, on a variable the shape does not name in
    #: ``module_writes``: by design where cash rebuilds a generator a
    #: variable holds, but a stale value looks the same, so each needs a read.
    rebuilt: list[tuple[str, str, str, str, str]] = field(default_factory=list)
    #: The same, on a ``module_writes`` variable: what a restart and a Run
    #: All give, the reference for module state a statement writes.
    as_restart: list[tuple[str, str, str, str, str]] = field(default_factory=list)
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
        if not self.ok:
            return "BAD"
        return "REBUILT" if self.rebuilt else "AS-RESTART" if self.as_restart else "OK"

    def text(self, show_statuses: bool = False) -> str:
        lines = [f"{self.verdict:10} {self.shape}"]
        for config, error in self.errors:
            lines.append(f"    [{config}] could not run: {error}")
        for config, run, where, cash, plain in self.mismatches:
            lines.append(f"    [{config}] {run} {where}: cash={cash!r} plain={plain!r}")
        for config, run, where, cash, plain in self.rebuilt:
            lines.append(f"    [{config}] {run} {where}: cash={cash!r} plain={plain!r} (as a fresh kernel: read it)")
        for config, run, where, cash, plain in self.as_restart:
            lines.append(f"    [{config}] {run} {where}: cash={cash!r} plain={plain!r} (module state: as a restart)")
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
    fresh: dict[str, str] = {}
    if shape.plan is not None:
        if shape.alone_as_run_all and not shape.edits:
            last = None
            for run_name, _restart, cells in shape.runs():
                if cells is None:
                    last = run_name
                elif last is not None:
                    fresh[run_name] = last
        return fresh  # otherwise a plan of its own: no fresh Run All to compare with
    for index, run_name in enumerate(RUNS):
        for first in (0, RUNS.index("restart+run1")):
            if first <= index and shape.cells_for(index) == shape.cells_for(first):
                fresh[run_name] = RUNS[first]
    return fresh


def _shows_a_rebuild(statuses: dict[str, dict[int, list[str]]], run_name: str, run_all: str) -> bool:
    """Whether the cells *run_name* re-ran alone report more statements than
    they did in the Run All *run_all*: cash re-ran cells above with them,
    where a stale hit reports only their own."""
    alone, before = statuses.get(run_name, {}), statuses.get(run_all, {})
    return bool(alone) and all(len(ran) > len(before.get(cell, ())) for cell, ran in alone.items())


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
        for run_name in dict.fromkeys([*plain.seen, *got.seen]):
            mine, theirs = got.seen.get(run_name, {}), plain.seen.get(run_name, {})
            first = plain.seen.get(fresh[run_name], {}) if run_name in fresh else {}
            if shape.plan is not None and first and not _shows_a_rebuild(got.statuses, run_name, fresh[run_name]):
                first = {}
            for where in sorted(set(mine) | set(theirs), key=str):
                if mine.get(where) == theirs.get(where):
                    continue
                found = (config, run_name, where, str(mine.get(where)), str(theirs.get(where)))
                if where in first and mine.get(where) == first[where]:
                    (report.as_restart if where in shape.module_writes else report.rebuilt).append(found)
                else:
                    report.mismatches.append(found)
    return report
