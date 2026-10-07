"""Measurement core of the speed set (``speed_set.py``).

A scenario times one unit of work twice: as plain Python and with cash. Its
result is the RATIO cash / plain, both measured in this process, back to back,
so a faster or slower machine moves both sides alike and the ratio stays
comparable across machines and runs.

How a scenario is measured:

* one untimed warm-up sample per side (imports, first-use caches);
* then ``repeats`` rounds, each taking one plain and one cash sample, the order
  alternating (plain-cash, cash-plain, ...) so drift hits both sides equally;
* ``gc.collect()`` before every sample, so one side does not pay for the
  other's garbage;
* the reported time per side is the median of its samples, in CPU time
  (``time.process_time``: what this process spent, all threads, not what
  other processes on the machine took) where the CPU clock is fine-grained,
  as on Linux and macOS. On Windows it advances in 15.6 ms steps, too coarse
  for samples of a few milliseconds, so there the wall clock is used
  (:func:`default_clock`). The row records which.

A sample returns one or more named metrics (a notebook run can time its
early and its late cells), and each metric becomes a row of the report.
"""

from __future__ import annotations

import contextlib
import gc
import io
import json
import shutil
import statistics
import sys
import tempfile
import time
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SIDES = ("plain", "cash")


@dataclass(frozen=True)
class Timing:
    cpu: float
    wall: float

    def __add__(self, other: Timing) -> Timing:
        return Timing(self.cpu + other.cpu, self.wall + other.wall)

    def get(self, clock: str) -> float:
        return self.cpu if clock == "cpu" else self.wall


ZERO = Timing(0.0, 0.0)


@contextlib.contextmanager
def stopwatch() -> Iterator[list[Timing]]:
    """``with stopwatch() as out:`` leaves the block's Timing in ``out[0]``."""
    out: list[Timing] = []
    c0, w0 = time.process_time(), time.perf_counter()
    try:
        yield out
    finally:
        out.append(Timing(time.process_time() - c0, time.perf_counter() - w0))


def total(timings: list[Timing]) -> Timing:
    acc = ZERO
    for t in timings:
        acc = acc + t
    return acc


def cpu_clock_step() -> float:
    """The smallest step ``time.process_time`` actually advances by.

    ``time.get_clock_info`` reports the API's unit (100 ns on Windows), not
    the tick the OS updates the counter on, so measure it."""
    steps = []
    for _ in range(3):
        start = time.process_time()
        deadline = time.perf_counter() + 0.1
        now = start
        while now == start and time.perf_counter() < deadline:
            now = time.process_time()
        steps.append(now - start if now != start else 0.1)
    return min(steps)


_CLOCK: list[str] = []


def default_clock() -> str:
    """``"cpu"`` where the CPU clock steps by under 0.1 ms, else ``"wall"``."""
    if not _CLOCK:
        _CLOCK.append("cpu" if cpu_clock_step() < 1e-4 else "wall")
    return _CLOCK[0]


@dataclass
class Context:
    """What a scenario's sample function gets: a scratch folder that lives
    for the scenario, a counter for values never seen before (cache misses),
    and a place for notes that end up in the report."""

    tmp: Path
    quick: bool = False
    notes: list[str] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    _counter: int = 0

    def fresh(self) -> int:
        self._counter += 1
        return self._counter

    def note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)


SampleFn = Callable[[str, Context], dict[str, Timing]]


@dataclass(frozen=True)
class Scenario:
    name: str
    group: str
    what: str
    sample: SampleFn
    # "cpu", "wall", or "auto" (default_clock()).
    clock: str = "auto"
    needs: tuple[str, ...] = ()
    # Metrics per sample, in report order, with a short description each.
    metrics: tuple[tuple[str, str], ...] = (("", ""),)


@dataclass
class Row:
    """One reported number: a scenario's metric."""

    name: str
    group: str
    what: str
    clock: str
    plain_s: float
    cash_s: float
    ratio: float
    # Noise: the median absolute deviation of the per-round ratios
    # (cash sample i / plain sample i) over their median.
    spread: float
    plain_samples: list[float]
    cash_samples: list[float]
    notes: list[str] = field(default_factory=list)
    skipped: str | None = None

    def to_json(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Row:
        return cls(**data)


def _row_name(scenario: Scenario, metric: str) -> str:
    return f"{scenario.name}/{metric}" if metric else scenario.name


def _missing(needs: tuple[str, ...]) -> str | None:
    import importlib

    for mod in needs:
        try:
            importlib.import_module(mod)
        except ImportError:
            return f"needs {mod}"
    return None


def _clock(scenario: Scenario) -> str:
    return default_clock() if scenario.clock == "auto" else scenario.clock


def skipped_rows(scenario: Scenario, reason: str) -> list[Row]:
    return [
        Row(
            _row_name(scenario, m),
            scenario.group,
            desc or scenario.what,
            _clock(scenario),
            0.0,
            0.0,
            float("nan"),
            0.0,
            [],
            [],
            skipped=reason,
        )
        for m, desc in scenario.metrics
    ]


def measure(scenario: Scenario, repeats: int, quick: bool = False) -> list[Row]:
    """Run one scenario: warm-up, then ``repeats`` interleaved rounds."""
    reason = _missing(scenario.needs)
    if reason:
        return skipped_rows(scenario, reason)
    clock = _clock(scenario)
    tmp = Path(tempfile.mkdtemp(prefix=f"cash-speed-{scenario.name}-"))
    ctx = Context(tmp=tmp, quick=quick)
    samples: dict[str, dict[str, list[float]]] = {s: {} for s in SIDES}
    try:
        for side in SIDES:
            _one(scenario, side, ctx)
        for r in range(repeats):
            for side in SIDES if r % 2 == 0 else SIDES[::-1]:
                for metric, timing in _one(scenario, side, ctx).items():
                    samples[side].setdefault(metric, []).append(timing.get(clock))
    finally:
        with quiet():
            cleanup = ctx.state.get("cleanup")
            if cleanup:
                cleanup()
        shutil.rmtree(tmp, ignore_errors=True)
    rows = []
    for metric, desc in scenario.metrics:
        p, c = samples["plain"].get(metric, []), samples["cash"].get(metric, [])
        if not p or not c:
            raise RuntimeError(f"{scenario.name}: no samples for metric {metric!r}")
        per_round = [ci / pi for ci, pi in zip(c, p) if pi > 0]
        med_p, med_c = statistics.median(p), statistics.median(c)
        ratio = med_c / med_p if med_p > 0 else float("inf")
        mid = statistics.median(per_round) if per_round else 0.0
        spread = statistics.median(abs(r - mid) for r in per_round) / mid if mid > 0 else 0.0
        rows.append(
            Row(
                _row_name(scenario, metric),
                scenario.group,
                desc or scenario.what,
                clock,
                med_p,
                med_c,
                ratio,
                spread,
                p,
                c,
                notes=list(ctx.notes),
            )
        )
    return rows


def _one(scenario: Scenario, side: str, ctx: Context) -> dict[str, Timing]:
    gc.collect()
    with quiet():
        result = scenario.sample(side, ctx)
    if not isinstance(result, dict):
        raise TypeError(f"{scenario.name}: a sample returns {{metric: Timing}}, got {type(result).__name__}")
    return result


@contextlib.contextmanager
def quiet() -> Iterator[None]:
    """Swallow what cash and the cells print and warn: the badge, the
    %cash_on banner, advisory warnings. The scenarios check results
    themselves; the console is for the report."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield


# --------------------------------------------------------------------------
# Notebook sessions
# --------------------------------------------------------------------------


def write_ipynb(path: Path, cells: list[str]) -> None:
    nb = {
        "cells": [
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}
            for src in cells
        ],
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")


class Notebook:
    """One kernel's life, in process: a fresh IPython shell, with or without
    ``%cash_on``, running ``cells`` one at a time and timing each.

    The cells are saved as a real ``.ipynb`` and the shell is told its path
    the way VS Code tells a kernel (``__vsc_ipynb_file__``), so cash's upstream
    check reads the notebook from disk as it does in a real session. What a
    real kernel adds on top (the ZMQ round trip, the frontend rendering the
    badge) is not measured; it is not cash's work.
    """

    def __init__(self, cells: list[str], cash_on: bool, workdir: Path):
        from IPython.core.interactiveshell import InteractiveShell

        import cash

        self.cells = list(cells)
        self.cash_on = cash_on
        workdir.mkdir(parents=True, exist_ok=True)
        self.workdir = workdir
        self.path = workdir / "speed.ipynb"
        write_ipynb(self.path, self.cells)
        InteractiveShell.clear_instance()
        self.shell = InteractiveShell.instance()
        cash.reset_session()
        _forget_notebook_path()
        self.shell.user_ns["__vsc_ipynb_file__"] = str(self.path)
        self.magics = None
        if cash_on:
            from cash.core import Cash
            from cash.notebook.ipython.magics import CashMagics

            instance = Cash(cache_dir=str(workdir / ".cash"), register_magic=False)
            self.magics = CashMagics(shell=self.shell, cash_instance=instance)
            self.shell.register_magics(self.magics)
            self.magics.cash_on("")

    def run(self, numbers: list[int] | None = None) -> list[Timing]:
        """Run cells (1-based ``numbers``, default all, in order); a Timing each."""
        out = []
        for n in numbers or range(1, len(self.cells) + 1):
            source = self.cells[n - 1]
            with stopwatch() as t:
                result = self.shell.run_cell(source)
            err = result.error_in_exec or result.error_before_exec
            if err is not None:
                raise RuntimeError(f"cell {n} failed ({'cash' if self.cash_on else 'plain'}): {err!r}\n{source}")
            out.append(t[0])
        if self.cash_on and not self.magics.tracking_state.variable_lineage:
            raise RuntimeError("cash was on but tracked no variable: the cell transform was not active")
        return out

    def peek(self, expr: str) -> Any:
        return eval(expr, self.shell.user_ns)

    def close(self) -> None:
        from IPython.core.interactiveshell import InteractiveShell

        InteractiveShell.clear_instance()


def _forget_notebook_path() -> None:
    """Drop cash's memo of the notebook path, so a session finds its own
    ``.ipynb`` instead of the previous session's (attributes looked up
    defensively: an older cash under A/B comparison may not have them)."""
    try:
        from cash.notebook import server_discovery as sd
    except ImportError:
        return
    for name, value in (
        ("_cached_notebook_path", None),
        ("_cached_notebook_path_time", 0.0),
        ("_negative_cache_time", 0.0),
    ):
        if hasattr(sd, name):
            setattr(sd, name, value)


@contextlib.contextmanager
def on_sys_path(folder: Path, modules: tuple[str, ...]) -> Iterator[None]:
    """Make ``folder`` importable and forget ``modules`` before and after, so
    each session imports its own copy as a fresh kernel would."""
    for m in modules:
        sys.modules.pop(m, None)
    sys.path.insert(0, str(folder))
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(str(folder))
        for m in modules:
            sys.modules.pop(m, None)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def format_table(rows: list[Row]) -> str:
    head = f"{'scenario':<34} {'plain ms':>10} {'cash ms':>10} {'ratio':>8} {'noise':>7}  note"
    lines = [head, "-" * len(head)]
    group = None
    for r in rows:
        if r.group != group:
            group = r.group
            lines.append(f"[{group}]")
        if r.skipped:
            lines.append(f"{r.name:<34} {'-':>10} {'-':>10} {'-':>8} {'-':>7}  skipped: {r.skipped}")
            continue
        lines.append(
            f"{r.name:<34} {r.plain_s * 1e3:>10.3f} {r.cash_s * 1e3:>10.3f} {r.ratio:>7.2f}x {r.spread:>6.0%}  {'; '.join(r.notes)}"
        )
    return "\n".join(lines)
