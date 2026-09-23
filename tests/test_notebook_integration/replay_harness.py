"""Edit one cell, run a later one: does cash give what a plain top-to-bottom run gives?

The acceptance harness for upstream replay. A scenario is a
notebook, an edit a user would make, and the cell they run next. The oracle is
the same notebook, edited, executed top to bottom without cash in a fresh
process. Two properties are checked:

* the target cell prints what the oracle's target cell prints (stdout only --
  a no-cash run has no badge and no rich display);
* every file cash WROTE during the target run is byte-identical to the file
  the oracle wrote. A file cash did not touch may legitimately be older: a
  write nothing reads is a terminal side effect that replay does not re-fire
  (the reconstruction scope gate). A file cash writes must never hold
  something no real run would produce -- such as a blank chart.

What was re-run for other cells is recorded, not asserted: it is the cost
side, reported next to the correctness verdict.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

Content = Union[str, bytes]


@dataclass(frozen=True)
class Edit:
    """What the user changes between the first run and the target run."""

    cell: int | None = None  # 1-based cell whose source is replaced
    source: str | None = None  # ... by this
    files: tuple[tuple[str, Content | None], ...] = ()  # relpath -> new content (None = delete)

    def apply_to_cells(self, cells: list[str]) -> list[str]:
        out = list(cells)
        if self.cell is not None:
            out[self.cell - 1] = self.source
        return out

    def apply_to_dir(self, work: Path) -> None:
        for rel, content in self.files:
            path = work / rel
            if content is None:
                path.unlink()
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                path.write_bytes(content)
            else:
                path.write_text(content, encoding="utf-8")
            # A re-delivered file lands later than the one it replaces; make
            # that visible even on a coarse-mtime filesystem.
            st = path.stat()
            os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))


@dataclass(frozen=True)
class Scenario:
    notebook: str
    name: str
    cells: tuple[str, ...]  # cell 1 must turn cash on
    files: tuple[tuple[str, Content], ...]
    edit: Edit
    target: int  # 1-based
    restart: bool = False  # restart + run cell 1 before the edit
    # Cells run after the edit and before the target: the user looks at
    # another cell first. Its replay refreshes part of what the target needs
    # and must not make the rest look fresh.
    first: tuple[int, ...] = ()

    @property
    def id(self) -> str:
        path = "".join(f"->c{c}" for c in (*self.first, self.target))
        return f"{self.notebook}:{self.name}{'+restart' if self.restart else ''}{path}"


@dataclass
class Result:
    scenario: str
    stdout: str
    oracle_stdout: str
    written: dict[str, str] = field(default_factory=dict)  # relpath -> sha (cash)
    oracle_files: dict[str, str] = field(default_factory=dict)  # relpath -> sha (oracle)
    rerun: list[str] = field(default_factory=list)
    seconds: float = 0.0
    recomputed: list[str] = field(default_factory=list)  # expensive steps that really ran

    @property
    def stdout_ok(self) -> bool:
        return self.stdout == self.oracle_stdout

    @property
    def bad_files(self) -> dict[str, str]:
        return {
            rel: ("not written by a real run" if rel not in self.oracle_files else "differs from a real run")
            for rel, sha in self.written.items()
            if self.oracle_files.get(rel) != sha
        }

    @property
    def ok(self) -> bool:
        return self.stdout_ok and not self.bad_files

    def explain(self) -> str:
        lines = [
            f"{self.scenario}: {'OK' if self.ok else 'WRONG'}  ({self.seconds:.1f}s, "
            f"{len(self.rerun)} statements re-run for other cells, "
            f"recomputed: {', '.join(self.recomputed) or 'nothing expensive'})"
        ]
        if not self.stdout_ok:
            lines += [
                "  stdout (cash):",
                *("    " + x for x in self.stdout.splitlines()),
                "  stdout (no cash):",
                *("    " + x for x in self.oracle_stdout.splitlines()),
            ]
        for rel, why in self.bad_files.items():
            lines.append(f"  file {rel}: {why}")
        if self.rerun:
            lines.append("  re-ran: " + " | ".join(s.splitlines()[0][:60] for s in self.rerun))
        return "\n".join(lines)


# ---------------------------------------------------------------------------

_SKIP_PARTS = {".cash", ".ipynb_checkpoints", "__pycache__"}

#: The corpus's expensive functions append their name here, so a scenario can
#: report what REALLY recomputed -- a statement scheduled for re-execution may
#: still be served from the statement cache. Bookkeeping, not output: never
#: compared with the oracle.
CALLS_LOG = "calls.log"


def _calls(work: Path) -> list[str]:
    p = work / CALLS_LOG
    return p.read_text(encoding="utf-8").split() if p.exists() else []


def _snapshot(work: Path) -> dict[str, tuple[int, int, str]]:
    snap = {}
    for p in work.rglob("*"):
        if not p.is_file() or _SKIP_PARTS & set(p.relative_to(work).parts) or p.suffix == ".ipynb":
            continue
        if p.name.endswith(".cashtrace.jsonl") or p.name == CALLS_LOG:
            continue
        st = p.stat()
        snap[p.relative_to(work).as_posix()] = (st.st_mtime_ns, st.st_size, _sha(p))
    return snap


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _write_files(work: Path, files) -> None:
    for rel, content in files:
        path = work / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")


def _strip_cash(source: str) -> str:
    keep = [ln for ln in source.splitlines() if not ln.lstrip().startswith("%") and ln.strip() != "import cash"]
    return "\n".join(keep)


_MARK = "@@REPLAY-CELL@@"


def oracle(scenario: Scenario) -> tuple[str, dict[str, str]]:
    """The edited notebook run top to bottom without cash: (target stdout, files)."""
    cells = scenario.edit.apply_to_cells(list(scenario.cells))[: scenario.target]
    with tempfile.TemporaryDirectory(prefix="replay-oracle-") as tmp:
        work = Path(tmp)
        _write_files(work, scenario.files)
        scenario.edit.apply_to_dir(work)
        before = _snapshot(work)
        script = "\n".join(
            f"print({_MARK + str(i)!r}, flush=True)\n{_strip_cash(c)}" for i, c in enumerate(cells, start=1)
        )
        env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"}
        env.pop("CASH_TRACE_FILE", None)
        run = subprocess.run(
            [sys.executable, "-c", script], cwd=work, env=env, capture_output=True, text=True, timeout=600
        )
        if run.returncode != 0:
            raise RuntimeError(f"oracle for {scenario.id} failed:\n{run.stderr[-3000:]}")
        target = run.stdout.split(f"{_MARK}{scenario.target}\n", 1)[1]
        after = _snapshot(work)
        keep = os.environ.get("CASH_REPLAY_KEEP")
        if keep:  # for looking at what differs
            import shutil

            shutil.copytree(work, Path(keep) / "oracle", dirs_exist_ok=True)
        files = {rel: v[2] for rel, v in after.items() if before.get(rel) != v}
        return target.rstrip("\n"), files


def _cell_stdout(runner, cell: int) -> str:
    outs = runner.nb.cells[cell - 1].get("outputs", [])
    text = "".join(
        "".join(o.get("text", "")) if isinstance(o.get("text"), list) else o.get("text", "")
        for o in outs
        if o.get("output_type") == "stream" and o.get("name") == "stdout"
    )
    return text.rstrip("\n")


def run_with_cash(scenario: Scenario, runner, trace_path: str) -> Result:
    """Run the scenario in *runner* (a NotebookTestRunner not yet started)."""
    work = Path(runner.work_dir)
    _write_files(work, scenario.files)
    prev = os.environ.get("CASH_TRACE_FILE")
    os.environ["CASH_TRACE_FILE"] = trace_path
    try:
        runner.create_notebook(list(scenario.cells))
        runner.start_kernel()
        runner.run_all()
        if scenario.restart:
            # The next session: a fresh kernel, the setup cell, then straight to
            # the cell the user wants.
            runner.restart()
            runner.run_cell(1)
        with open(trace_path, "a", encoding="utf-8") as fh:
            fh.write('{"event": "__edit__"}\n')
        edited = scenario.edit.apply_to_cells(list(scenario.cells))
        if scenario.edit.cell is not None:
            runner.set_cell_source(scenario.edit.cell, edited[scenario.edit.cell - 1])
        scenario.edit.apply_to_dir(work)
        for cell in scenario.first:
            runner.run_cell(cell)
        before = _snapshot(work)
        calls_before = len(_calls(work))
        t0 = time.perf_counter()
        runner.run_cell(scenario.target)
        seconds = time.perf_counter() - t0
        after = _snapshot(work)
        recomputed = _calls(work)[calls_before:]
    finally:
        if prev is None:
            os.environ.pop("CASH_TRACE_FILE", None)
        else:
            os.environ["CASH_TRACE_FILE"] = prev
    records = [json.loads(line) for line in open(trace_path, encoding="utf-8") if line.strip()]
    split = next(i for i, r in enumerate(records) if r.get("event") == "__edit__")
    rerun = [r["stmt"] for r in records[split:] if r.get("event") == "schedule_reexec"]
    written = {rel: v[2] for rel, v in after.items() if before.get(rel) != v}
    o_stdout, o_files = oracle(scenario)
    return Result(
        scenario.id, _cell_stdout(runner, scenario.target), o_stdout, written, o_files, rerun, seconds, recomputed
    )


def fresh_trace_file() -> str:
    fd, path = tempfile.mkstemp(suffix=".cashtrace.jsonl")
    os.close(fd)
    return path


def scenarios_from(
    notebook: str,
    cells,
    files,
    edits: dict[str, Edit],
    targets: dict[str, tuple[int | tuple[int, ...], ...]],
    restart: tuple[str, ...] = (),
) -> list[Scenario]:
    """One scenario per (edit, target); edits named in *restart* also get a
    variant that restarts the kernel between the first run and the edit.

    A target given as a tuple is a path: ``(7, 9)`` runs cell 7 after the
    edit, then checks cell 9.
    """
    out = []
    for name, edit in edits.items():
        for path in targets[name]:
            *first, target = path if isinstance(path, tuple) else (path,)
            for again in (False, True) if name in restart else (False,):
                out.append(
                    Scenario(
                        notebook, name, tuple(cells), tuple(files), edit, target, restart=again, first=tuple(first)
                    )
                )
    return out


__all__ = ["Edit", "Scenario", "Result", "oracle", "run_with_cash", "fresh_trace_file", "scenarios_from"]
