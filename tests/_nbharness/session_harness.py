"""Replay a user's working session on a small notebook, checking every step.

Human-style testing found what the regression suite could not:
bugs that need a SEQUENCE -- edit a cell, look at another one first, restart,
a file re-delivered overnight -- or two conditions at once (a read inside a
helper AND a morning restore). Each session here scripts what one user did
on their real project, shrunk to a small dataset, as a list of steps:

    Run("export")                  run one cell, the way a user jumps to it
    RunAll() / RestartAndRunAll()  the whole notebook
    Restart()                      new kernel; runs the setup cell, as every user does
    Edit("features", new_source)   change a cell (a str, or a function of the old source)
    AddFile / ReplaceFile / RemoveFile / ClearDir    what arrives on disk

After every run the notebook is checked against a plain top-to-bottom run of
itself as it stands at that moment (``session_oracle.py``, no cash, the current
input files):

* the cell prints what the plain run's cell prints, and displays a value only
  where Jupyter would (its last expression);
* every file cash wrote in this step holds what a plain run leaves there;
* every file the cell itself writes in a plain run is on disk, byte-identical
  (a chart that was never written is as wrong as a wrong number);
* for a whole-notebook run: every file a plain run writes is on disk, identical.

A step may also state what it must recompute (``calls={"backtest": 0}``): the
sessions' expensive functions append their name to ``calls.log``. A fix that
re-runs everything passes the correctness checks; this is the check it fails.

Failures are collected, not raised at the first one, so a report shows the
whole session -- but a step after a wrong one may be wrong only because of it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Union

from tests._nbharness.replay_harness import _calls, _cell_stdout, _snapshot, _strip_cash

Content = Union[str, bytes]
_ORACLE = Path(__file__).with_name("session_oracle.py")


# --- steps -------------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    cell: str
    calls: dict[str, int] | None = None  # expensive steps this run must recompute, exactly
    # A cost the step is known to miss today, and why. Reported, not failed --
    # and failed the day it is met, so the note is removed with the fix.
    gap: str | None = None


@dataclass(frozen=True)
class RunAll:
    calls: dict[str, int] | None = None
    gap: str | None = None


@dataclass(frozen=True)
class RestartAndRunAll:
    calls: dict[str, int] | None = None
    gap: str | None = None


@dataclass(frozen=True)
class Restart:
    """Restart the kernel and run the setup cell (``%cash_on``), nothing else."""


@dataclass(frozen=True)
class Edit:
    cell: str
    source: Union[str, Callable[[str], str]]


@dataclass(frozen=True)
class AddFile:
    path: str
    content: Content


ReplaceFile = AddFile  # same operation; the name says what the user did


@dataclass(frozen=True)
class RemoveFile:
    path: str


@dataclass(frozen=True)
class ClearDir:
    """Delete a folder of OUTPUTS (``out/``) before producing them again."""

    path: str


Step = Union[Run, RunAll, RestartAndRunAll, Restart, Edit, AddFile, RemoveFile, ClearDir]


@dataclass(frozen=True)
class Session:
    name: str
    cells: tuple[tuple[str, str], ...]  # (label, source); the first turns cash on
    files: tuple[tuple[str, Content], ...]  # input files at the start
    steps: tuple[Step, ...]


# --- the player --------------------------------------------------------------


def _describe(step: Step) -> str:
    """A step in one short line: a file's path, not its content."""
    if isinstance(step, AddFile):
        return f"AddFile({step.path})"
    if isinstance(step, Edit):
        return f"Edit({step.cell})"
    if isinstance(step, Run):
        return f"Run({step.cell})" + (f" calls={step.calls}" if step.calls else "")
    return type(step).__name__ + (f" calls={step.calls}" if getattr(step, "calls", None) else "")


def _write(work: Path, rel: str, content: Content) -> None:
    path = work / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    # A delivered file lands later than the one it replaces, even on a
    # coarse-mtime filesystem.
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))


@dataclass
class _Oracle:
    cells: list[dict]
    final: dict[str, str]


@dataclass
class Player:
    session: Session
    runner: object  # NotebookTestRunner, not started
    persist: bool = False
    failures: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.labels = [label for label, _ in self.session.cells]
        self.sources = dict(self.session.cells)
        self.inputs = dict(self.session.files)
        self.work = Path(self.runner.work_dir)
        self._oracles: dict[tuple, _Oracle] = {}

    # -- oracle
    def _oracle(self, upto: int) -> _Oracle:
        cells = [_strip_cash(self.sources[label]) for label in self.labels[:upto]]
        key = (tuple(cells), tuple(sorted((k, v if isinstance(v, str) else v.hex()) for k, v in self.inputs.items())))
        if key not in self._oracles:
            with tempfile.TemporaryDirectory(prefix="session-oracle-") as tmp:
                for rel, content in self.inputs.items():
                    _write(Path(tmp), rel, content)
                spec, result = Path(tmp) / "_spec.json", Path(tmp) / "_result.json"
                spec.write_text(json.dumps({"work": tmp, "cells": cells}), encoding="utf-8")
                env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"}
                env.pop("CASH_TRACE_FILE", None)
                run = subprocess.run(
                    [sys.executable, str(_ORACLE), str(spec), str(result)],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if run.returncode != 0:
                    raise RuntimeError(
                        f"{self.session.name}: the plain run itself failed "
                        f"-- the session is broken, not cash:\n{run.stderr[-3000:]}"
                    )
                data = json.loads(result.read_text(encoding="utf-8"))
                for c in data["cells"]:
                    for rel in ("_spec.json", "_result.json"):
                        c["written"].pop(rel, None)
                data["final"].pop("_spec.json", None)
                self._oracles[key] = _Oracle(data["cells"], data["final"])
        return self._oracles[key]

    # -- helpers
    def _idx(self, label: str) -> int:
        return self.labels.index(label) + 1

    def _fail(self, step_no: int, step: Step, message: str) -> None:
        self.failures.append(f"step {step_no} {_describe(step)}: {message}")

    def _run_cell(self, i: int) -> str | None:
        try:
            self.runner.run_cell(i)
        except Exception as exc:  # a cell error is a finding, not a crash
            name = getattr(exc, "ename", None) or type(exc).__name__
            value = getattr(exc, "evalue", None) or str(exc)[-400:]
            return f"cell {self.labels[i - 1]!r} raised {name}: {value}"
        if i == 1 and self.persist:
            # `%cash_persist on` needs cash on, and a restart forgets it.
            self.runner.enable_persist()
        return None

    def _check_displays(self, step_no: int, step, i: int, want: int) -> None:
        """The values the kernel displayed for cell *i*, cash's badges aside,
        against the one a plain Jupyter run displays (its last expression)."""
        shown = []
        for o in self.runner.nb.cells[i - 1].get("outputs", []):
            if o.get("output_type") not in ("execute_result", "display_data"):
                continue
            plain = o.get("data", {}).get("text/plain", "")
            plain = "".join(plain) if isinstance(plain, list) else plain
            if plain.startswith("<IPython.core.display.HTML object>"):
                continue  # the badge
            shown.append(plain)
        if len(shown) != want:
            self._fail(
                step_no,
                step,
                f"cell {self.labels[i - 1]!r} displayed {len(shown)} "
                f"value(s), Jupyter displays {want}: {[s[:60] for s in shown[:4]]}",
            )

    def _check_cost(self, step_no: int, step, calls_before: int) -> None:
        expected = getattr(step, "calls", None)
        if not expected:
            return
        ran = _calls(self.work)[calls_before:]
        misses = [
            f"recomputed {name} {ran.count(name)}x, expected {n}x"
            for name, n in expected.items()
            if ran.count(name) != n
        ]
        if step.gap and misses:
            self.gaps.append(
                f"step {step_no} {step.cell if isinstance(step, Run) else step}: "
                f"{'; '.join(misses)} (known: {step.gap})"
            )
        elif step.gap:
            self._fail(step_no, step, f"the known gap no longer happens ({step.gap}) -- drop the note from the session")
        for miss in [] if step.gap else misses:
            self._fail(step_no, step, miss)

    def _check_files(self, step_no, step, before, oracle_final, must_exist) -> None:
        after = _snapshot(self.work)
        for rel, v in after.items():
            if before.get(rel) != v and rel not in self.inputs:
                if rel not in oracle_final:
                    self._fail(step_no, step, f"wrote {rel}, which a plain run never writes")
                elif oracle_final[rel] != v[2]:
                    self._fail(step_no, step, f"wrote {rel}, which differs from a plain run")
        for rel, sha in must_exist.items():
            if rel in self.inputs:
                continue
            if rel not in after:
                self._fail(step_no, step, f"{rel} is missing (a plain run writes it)")
            elif after[rel][2] != sha:
                self._fail(step_no, step, f"{rel} differs from a plain run")

    # -- steps
    def play(self) -> list[str]:
        self.runner.create_notebook([src for _, src in self.session.cells])
        for rel, content in self.session.files:
            _write(self.work, rel, content)
        self.runner.start_kernel()
        for step_no, step in enumerate(self.session.steps, start=1):
            self.log.append(f"{step_no}: {_describe(step)}")
            if isinstance(step, Edit):
                old = self.sources[step.cell]
                new = step.source(old) if callable(step.source) else step.source
                assert new != old, f"step {step_no}: the edit changes nothing"
                self.sources[step.cell] = new
                self.runner.set_cell_source(self._idx(step.cell), new)
            elif isinstance(step, AddFile):
                self.inputs[step.path] = step.content
                _write(self.work, step.path, step.content)
            elif isinstance(step, RemoveFile):
                self.inputs.pop(step.path, None)
                (self.work / step.path).unlink()
            elif isinstance(step, ClearDir):
                shutil.rmtree(self.work / step.path, ignore_errors=True)
            elif isinstance(step, Restart):
                self.runner.restart()
                error = self._run_cell(1)
                if error:
                    self._fail(step_no, step, error)
            elif isinstance(step, Run):
                self._play_run(step_no, step)
            elif isinstance(step, (RunAll, RestartAndRunAll)):
                self._play_run_all(step_no, step)
            else:
                raise TypeError(step)
        return self.failures

    def _play_run(self, step_no: int, step: Run) -> None:
        i = self._idx(step.cell)
        before, calls_before = _snapshot(self.work), len(_calls(self.work))
        error = self._run_cell(i)
        if error:
            self._fail(step_no, step, error)
            return
        oracle = self._oracle(i)
        got, want = _cell_stdout(self.runner, i), oracle.cells[-1]["stdout"].rstrip("\n")
        if got != want:
            self._fail(step_no, step, f"printed {got!r}, a plain run prints {want!r}")
        self._check_displays(step_no, step, i, oracle.cells[-1]["displays"])
        self._check_files(step_no, step, before, oracle.final, oracle.cells[-1]["written"])
        self._check_cost(step_no, step, calls_before)

    def _play_run_all(self, step_no: int, step) -> None:
        if isinstance(step, RestartAndRunAll):
            self.runner.restart()
        before, calls_before = _snapshot(self.work), len(_calls(self.work))
        oracle = self._oracle(len(self.labels))
        written = {}
        for i in range(1, len(self.labels) + 1):
            error = self._run_cell(i)
            if error:
                self._fail(step_no, step, error)
                return
            got, want = _cell_stdout(self.runner, i), oracle.cells[i - 1]["stdout"].rstrip("\n")
            if i > 1 and got != want:  # cell 1 prints cash's own banner
                self._fail(step_no, step, f"cell {self.labels[i - 1]!r} printed {got!r}, a plain run prints {want!r}")
            if i > 1:
                self._check_displays(step_no, step, i, oracle.cells[i - 1]["displays"])
            written.update(oracle.cells[i - 1]["written"])
        self._check_files(step_no, step, before, oracle.final, written)
        self._check_cost(step_no, step, calls_before)

    def report(self) -> str:
        return (
            f"{self.session.name} ({'persist all' if self.persist else 'default'}): "
            f"{len(self.failures)} problem(s)\n  "
            + "\n  ".join(self.failures)
            + "\n known gaps:\n  "
            + "\n  ".join(self.gaps or ["none"])
            + "\n steps:\n  "
            + "\n  ".join(self.log)
        )


__all__ = [
    "Session",
    "Player",
    "Run",
    "RunAll",
    "RestartAndRunAll",
    "Restart",
    "Edit",
    "AddFile",
    "ReplaceFile",
    "RemoveFile",
    "ClearDir",
]
