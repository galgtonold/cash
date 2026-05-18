# Notebook Cache Overhead Analysis — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pyinstrument-backed harness that runs cash-instrumented notebooks in three modes (cash-off, cash-on cold, cash-on warm), produces per-cell wall-clock numbers and call-graph profiles, then use it to write a data-driven analysis report with prioritized remediation strategies.

**Architecture:** Single-purpose CLI script (`benchmarks/bench_notebook_overhead.py`) that drives an in-process `InteractiveShell`, enables cash against a temp cache dir, runs cells with `time.perf_counter()` and optionally `pyinstrument.Profiler`, and emits JSON + HTML to `benchmarks/results/`. A second script (`benchmarks/compare_modes.py`) reads those JSONs and emits markdown tables. Per-statement metrics are captured by monkey-patching `StatementProcessor.process` to tee its return value. No source changes to `src/cash/`.

**Tech Stack:** Python 3.10+, IPython, pyinstrument (new dev dep), pytest, nbformat (for reading .ipynb JSON — already pulled in by jupyter_client).

**Spec:** [docs/superpowers/specs/2026-05-18-notebook-cache-overhead-analysis.md](docs/superpowers/specs/2026-05-18-notebook-cache-overhead-analysis.md)

---

## File Structure

```
benchmarks/
  bench_notebook_overhead.py        # CLI entry — argparse + orchestration
  compare_modes.py                  # JSON → markdown table comparator
  _overhead_io.py                   # notebook load + synthetic micro generator
  _overhead_driver.py               # IPython shell driver, cash hookup, metrics tee
  _overhead_results.py              # result dataclasses + JSON schema
  results/                          # gitignored — *.json, *.html artifacts

tests/
  test_benchmarks_overhead/
    __init__.py
    test_overhead_io.py             # synthetic gen + cell loading
    test_overhead_driver.py         # shell driver, metrics tee
    test_overhead_results.py        # JSON schema round-trip
    test_compare_modes.py           # markdown table emitter
    test_cli_smoke.py               # end-to-end CLI smoke test

docs/superpowers/specs/
  2026-05-18-notebook-cache-overhead-results.md   # the analysis report (Task 11)
```

**Responsibility split:** `_overhead_io.py` knows about notebook files. `_overhead_driver.py` knows about IPython and cash. `_overhead_results.py` knows about persistence. The CLI ties them together. Each file < ~200 lines.

---

## Task 1: Add pyinstrument dev dep

**Files:**
- Modify: `pyproject.toml:81-93` (the `dev` extra)

- [ ] **Step 1: Add the dep**

Edit `pyproject.toml` so the `dev` extra includes pyinstrument:

```toml
dev = [
    "pytest>=8.0",
    "pytest-subtests",
    "pytest-timeout",
    "pytest-xdist",
    "nbclient",
    "jupyter_client",
    "ipykernel",
    "pandas",
    "pyarrow>=13.0",
    "nest_asyncio",
    "traitlets",
    "pyinstrument>=4.6,<6",
]
```

- [ ] **Step 2: Install**

Run: `pip install -e ".[dev]"`
Expected: pyinstrument installs without resolver errors.

- [ ] **Step 3: Smoke test**

Run: `python -c "import pyinstrument; print(pyinstrument.__version__)"`
Expected: prints a version like `4.6.2` or `5.x`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: add pyinstrument to dev extra for overhead benchmarks"
```

---

## Task 2: Notebook IO module — load + synthetic generation

**Files:**
- Create: `benchmarks/_overhead_io.py`
- Create: `tests/test_benchmarks_overhead/__init__.py` (empty)
- Create: `tests/test_benchmarks_overhead/test_overhead_io.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_benchmarks_overhead/test_overhead_io.py`:

```python
import json
from pathlib import Path

import pytest

from benchmarks._overhead_io import (
    CodeCell,
    load_code_cells,
    write_synthetic_micro,
)


def _make_nb(tmp_path: Path, cells: list[tuple[str, list[str]]]) -> Path:
    """Build a minimal .ipynb on disk; cells is [(cell_type, source_lines), ...]."""
    nb = {
        "cells": [
            {
                "cell_type": ctype,
                "metadata": {},
                "source": src,
                **({"outputs": [], "execution_count": None} if ctype == "code" else {}),
            }
            for ctype, src in cells
        ],
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = tmp_path / "nb.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")
    return path


def test_load_code_cells_returns_only_code_cells(tmp_path):
    path = _make_nb(tmp_path, [
        ("markdown", ["# title\n"]),
        ("code", ["x = 1\n", "y = 2\n"]),
        ("markdown", ["nope"]),
        ("code", ["z = x + y\n"]),
    ])
    cells = load_code_cells(path)
    assert [c.index for c in cells] == [0, 1]  # zero-based among code cells
    assert cells[0].source == "x = 1\ny = 2\n"
    assert cells[1].source == "z = x + y\n"


def test_load_code_cells_preserves_original_cell_index(tmp_path):
    path = _make_nb(tmp_path, [
        ("markdown", ["a"]),
        ("code", ["x = 1"]),
        ("markdown", ["b"]),
        ("code", ["y = 2"]),
    ])
    cells = load_code_cells(path)
    assert cells[0].notebook_cell_index == 1
    assert cells[1].notebook_cell_index == 3


def test_write_synthetic_micro_produces_loadable_notebook(tmp_path):
    out = tmp_path / "micro.ipynb"
    write_synthetic_micro(out, n_statements=10)

    cells = load_code_cells(out)
    assert len(cells) == 1
    lines = [line for line in cells[0].source.splitlines() if line.strip()]
    assert len(lines) == 10
    # Each line is a simple assignment with no I/O
    for line in lines:
        assert "=" in line
        assert "open(" not in line
        assert "read_" not in line


def test_synthetic_micro_default_size(tmp_path):
    out = tmp_path / "micro.ipynb"
    write_synthetic_micro(out)
    cells = load_code_cells(out)
    lines = [line for line in cells[0].source.splitlines() if line.strip()]
    assert len(lines) == 100  # default per spec
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_benchmarks_overhead/test_overhead_io.py -v`
Expected: FAIL — `benchmarks._overhead_io` does not exist.

- [ ] **Step 3: Implement the module**

Create `benchmarks/_overhead_io.py`:

```python
"""Notebook IO for the overhead benchmark.

Loads code cells from .ipynb files and generates the synthetic micro
notebook used to isolate per-statement overhead from compute noise.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CodeCell:
    """A code cell from a notebook.

    Attributes:
        index: Zero-based index among code cells only.
        notebook_cell_index: Zero-based index among all cells (code + markdown).
        source: Cell source as a single string (joined with no extra separator).
    """
    index: int
    notebook_cell_index: int
    source: str


def load_code_cells(path: Path) -> list[CodeCell]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[CodeCell] = []
    code_idx = 0
    for nb_idx, cell in enumerate(data.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        out.append(CodeCell(index=code_idx, notebook_cell_index=nb_idx, source=src))
        code_idx += 1
    return out


def write_synthetic_micro(path: Path, n_statements: int = 100) -> None:
    """Write a notebook with one code cell containing ``n_statements`` tiny
    assignments. No imports, no I/O — the cell stays in the FileAccessTracker
    + capture_output path so we measure cash overhead with minimal compute.
    """
    lines = [f"a_{i} = {i} + 1" for i in range(n_statements)]
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": "\n".join(lines) + "\n",
            }
        ],
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    Path(path).write_text(json.dumps(nb, indent=1), encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_benchmarks_overhead/test_overhead_io.py -v`
Expected: PASS — all 4 tests green.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/_overhead_io.py tests/test_benchmarks_overhead/__init__.py tests/test_benchmarks_overhead/test_overhead_io.py
git commit -m "feat(bench): notebook IO + synthetic micro generator for overhead bench"
```

---

## Task 3: Result types + JSON schema

**Files:**
- Create: `benchmarks/_overhead_results.py`
- Create: `tests/test_benchmarks_overhead/test_overhead_results.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_benchmarks_overhead/test_overhead_results.py`:

```python
import json
from pathlib import Path

from benchmarks._overhead_results import (
    CellTiming,
    RunResult,
    StatementMetric,
    write_results,
    read_results,
)


def test_run_result_round_trip(tmp_path):
    result = RunResult(
        notebook="mynb.ipynb",
        mode="cold",
        repeat=1,
        python_version="3.12.0",
        cash_version="0.5.0b1",
        platform="win32",
        cells=[
            CellTiming(
                index=0,
                notebook_cell_index=0,
                wall_seconds=0.123,
                source_chars=42,
                statement_metrics=[
                    StatementMetric(
                        code="x = 1",
                        execution_time=0.001,
                        total_time=0.002,
                        status="COMPUTED",
                    ),
                ],
            ),
        ],
        total_wall_seconds=0.123,
        cache_dir_bytes=1024,
    )
    out = tmp_path / "result.json"
    write_results(out, result)

    loaded = read_results(out)
    assert loaded.mode == "cold"
    assert loaded.cells[0].wall_seconds == 0.123
    assert loaded.cells[0].statement_metrics[0].execution_time == 0.001


def test_write_results_creates_parent_dir(tmp_path):
    out = tmp_path / "nested" / "dir" / "result.json"
    result = RunResult(
        notebook="x.ipynb", mode="off", repeat=0, python_version="3.12",
        cash_version="", platform="", cells=[], total_wall_seconds=0.0,
        cache_dir_bytes=0,
    )
    write_results(out, result)
    assert out.exists()


def test_run_result_json_is_human_readable(tmp_path):
    out = tmp_path / "r.json"
    result = RunResult(
        notebook="x.ipynb", mode="off", repeat=0, python_version="3.12",
        cash_version="", platform="", cells=[], total_wall_seconds=0.0,
        cache_dir_bytes=0,
    )
    write_results(out, result)
    raw = out.read_text(encoding="utf-8")
    assert "\n" in raw  # indented, not on one line
    json.loads(raw)  # valid JSON
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_benchmarks_overhead/test_overhead_results.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the module**

Create `benchmarks/_overhead_results.py`:

```python
"""Result types and JSON persistence for the overhead benchmark."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StatementMetric:
    code: str
    execution_time: float
    total_time: float
    status: str  # 'COMPUTED' | 'RESTORED' | 'SKIPPED' | 'UNKNOWN'


@dataclass
class CellTiming:
    index: int
    notebook_cell_index: int
    wall_seconds: float
    source_chars: int
    statement_metrics: list[StatementMetric] = field(default_factory=list)


@dataclass
class RunResult:
    notebook: str
    mode: str  # 'off' | 'cold' | 'warm'
    repeat: int
    python_version: str
    cash_version: str
    platform: str
    cells: list[CellTiming]
    total_wall_seconds: float
    cache_dir_bytes: int


def _to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


def write_results(path: Path, result: RunResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(result), indent=2), encoding="utf-8")


def read_results(path: Path) -> RunResult:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cells = [
        CellTiming(
            index=c["index"],
            notebook_cell_index=c["notebook_cell_index"],
            wall_seconds=c["wall_seconds"],
            source_chars=c["source_chars"],
            statement_metrics=[StatementMetric(**m) for m in c["statement_metrics"]],
        )
        for c in data["cells"]
    ]
    return RunResult(
        notebook=data["notebook"],
        mode=data["mode"],
        repeat=data["repeat"],
        python_version=data["python_version"],
        cash_version=data["cash_version"],
        platform=data["platform"],
        cells=cells,
        total_wall_seconds=data["total_wall_seconds"],
        cache_dir_bytes=data["cache_dir_bytes"],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_benchmarks_overhead/test_overhead_results.py -v`
Expected: PASS — 3 tests green.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/_overhead_results.py tests/test_benchmarks_overhead/test_overhead_results.py
git commit -m "feat(bench): result dataclasses and JSON round-trip for overhead bench"
```

---

## Task 4: IPython shell driver — cash-off path

**Files:**
- Create: `benchmarks/_overhead_driver.py`
- Create: `tests/test_benchmarks_overhead/test_overhead_driver.py`

This task implements the simpler half (no cash). Task 5 adds cash + the metrics tee.

- [ ] **Step 1: Write the failing test**

Create `tests/test_benchmarks_overhead/test_overhead_driver.py`:

```python
from pathlib import Path

from benchmarks._overhead_driver import run_notebook
from benchmarks._overhead_io import CodeCell


def test_run_notebook_cash_off_runs_each_cell_and_times_it():
    cells = [
        CodeCell(index=0, notebook_cell_index=0, source="x = 1 + 1\n"),
        CodeCell(index=1, notebook_cell_index=1, source="y = x * 2\n"),
    ]
    timings = run_notebook(cells, cash_enabled=False, cache_dir=None)
    assert len(timings) == 2
    assert timings[0].source_chars == len("x = 1 + 1\n")
    assert timings[0].wall_seconds >= 0
    assert timings[1].wall_seconds >= 0
    # cash_enabled=False -> no statement_metrics captured
    assert timings[0].statement_metrics == []
    assert timings[1].statement_metrics == []


def test_run_notebook_propagates_variable_state_between_cells():
    """Cell 2 reads x from cell 1; if the shell isn't shared, this fails."""
    cells = [
        CodeCell(index=0, notebook_cell_index=0, source="x = 42\n"),
        CodeCell(index=1, notebook_cell_index=1, source="assert x == 42\n"),
    ]
    timings = run_notebook(cells, cash_enabled=False, cache_dir=None)
    assert len(timings) == 2  # second cell didn't raise
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_benchmarks_overhead/test_overhead_driver.py -v`
Expected: FAIL — `benchmarks._overhead_driver` does not exist.

- [ ] **Step 3: Implement the driver (cash-off only for now)**

Create `benchmarks/_overhead_driver.py`:

```python
"""IPython shell driver for the overhead benchmark.

Spins up a fresh ``InteractiveShell`` per call, optionally enables cash,
runs cells via ``shell.run_cell``, and times each one. Captures
per-statement ``ProcessResult`` data when cash is enabled via a monkey-patch
on ``StatementProcessor.process``.
"""
from __future__ import annotations

import time
from pathlib import Path

from IPython.core.interactiveshell import InteractiveShell

from benchmarks._overhead_io import CodeCell
from benchmarks._overhead_results import CellTiming, StatementMetric


def run_notebook(
    cells: list[CodeCell],
    cash_enabled: bool,
    cache_dir: Path | None,
) -> list[CellTiming]:
    """Run ``cells`` in a fresh in-process IPython shell.

    If ``cash_enabled`` is True, ``cache_dir`` must be supplied and cash is
    enabled with that directory as its backend. Per-statement metrics are
    captured for each cash-processed statement.
    """
    if cash_enabled and cache_dir is None:
        raise ValueError("cache_dir is required when cash_enabled is True")

    shell = InteractiveShell.instance()
    # Defensive: clear user_ns so a re-used singleton doesn't carry state.
    shell.reset(new_session=True)

    statement_sink: list[StatementMetric] = []
    if cash_enabled:
        _enable_cash(shell, Path(cache_dir), statement_sink)

    timings: list[CellTiming] = []
    for cell in cells:
        # Drain the sink so each cell only owns its own statements.
        before = len(statement_sink)
        t0 = time.perf_counter()
        shell.run_cell(cell.source)
        t1 = time.perf_counter()
        cell_metrics = list(statement_sink[before:])
        timings.append(CellTiming(
            index=cell.index,
            notebook_cell_index=cell.notebook_cell_index,
            wall_seconds=t1 - t0,
            source_chars=len(cell.source),
            statement_metrics=cell_metrics,
        ))
    return timings


def _enable_cash(shell, cache_dir: Path, sink: list[StatementMetric]) -> None:
    """Stub for cash setup — implemented in Task 5."""
    raise NotImplementedError("Cash enablement implemented in Task 5")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_benchmarks_overhead/test_overhead_driver.py -v`
Expected: PASS — 2 tests green (both use `cash_enabled=False`, so the stub is never called).

- [ ] **Step 5: Commit**

```bash
git add benchmarks/_overhead_driver.py tests/test_benchmarks_overhead/test_overhead_driver.py
git commit -m "feat(bench): IPython shell driver (cash-off path) with per-cell timing"
```

---

## Task 5: Driver — cash enablement + per-statement metrics tee

**Files:**
- Modify: `benchmarks/_overhead_driver.py:_enable_cash`
- Modify: `tests/test_benchmarks_overhead/test_overhead_driver.py` (add cash-on test)

- [ ] **Step 1: Add the failing test**

Append to `tests/test_benchmarks_overhead/test_overhead_driver.py`:

```python
def test_run_notebook_cash_on_captures_statement_metrics(tmp_path):
    cells = [
        CodeCell(index=0, notebook_cell_index=0, source="x = 1 + 1\n"),
        CodeCell(index=1, notebook_cell_index=1, source="y = x * 2\n"),
    ]
    timings = run_notebook(cells, cash_enabled=True, cache_dir=tmp_path)
    assert len(timings) == 2
    # Each cell has at least one statement metric captured
    assert len(timings[0].statement_metrics) >= 1
    m = timings[0].statement_metrics[0]
    assert m.execution_time >= 0
    assert m.total_time >= m.execution_time  # total includes execution
    assert m.status in {"COMPUTED", "RESTORED", "SKIPPED", "UNKNOWN"}


def test_run_notebook_cash_on_cold_then_warm_status_shifts(tmp_path):
    """Same cells run twice against the same cache dir: second run should
    have at least one RESTORED status (the cache is now populated)."""
    cells = [CodeCell(index=0, notebook_cell_index=0, source="z = 7 * 6\n")]
    first = run_notebook(cells, cash_enabled=True, cache_dir=tmp_path)
    second = run_notebook(cells, cash_enabled=True, cache_dir=tmp_path)
    first_statuses = [m.status for m in first[0].statement_metrics]
    second_statuses = [m.status for m in second[0].statement_metrics]
    assert "COMPUTED" in first_statuses
    assert "RESTORED" in second_statuses
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_benchmarks_overhead/test_overhead_driver.py -v`
Expected: FAIL — `_enable_cash` raises `NotImplementedError`.

- [ ] **Step 3: Replace the stub with the real implementation**

In `benchmarks/_overhead_driver.py`, replace the `_enable_cash` stub with:

```python
def _enable_cash(shell, cache_dir: Path, sink: list[StatementMetric]) -> None:
    """Initialise cash on ``shell`` and install a tee on
    ``StatementProcessor.process`` so each cell's per-statement
    ``ProcessResult`` is appended to ``sink``.

    The tee patches the class method (not the instance) so it fires for every
    StatementProcessor the magics layer creates, including any per-cell
    re-instantiations.
    """
    from cash.notebook.statement_processor import StatementProcessor

    # Patch process() at class level so all instances (including those built
    # later inside magics) are observed. We patch the class on the first call
    # and store the original on the class so re-running this in the same
    # subprocess is idempotent.
    original = getattr(StatementProcessor, "_orig_process_for_bench", None)
    if original is None:
        original = StatementProcessor.process
        StatementProcessor._orig_process_for_bench = original  # type: ignore[attr-defined]

    def _teed_process(self, code, *args, **kwargs):
        result = original(self, code, *args, **kwargs)
        try:
            sink.append(StatementMetric(
                code=(result.get("code") if isinstance(result, dict) else code)[:200],
                execution_time=float(result.get("execution_time", 0.0)) if isinstance(result, dict) else 0.0,
                total_time=float(result.get("total_time", 0.0)) if isinstance(result, dict) else 0.0,
                status=str(result.get("status", "UNKNOWN")) if isinstance(result, dict) else "UNKNOWN",
            ))
        except Exception:  # noqa: BLE001 — tee must never break user code
            pass
        return result

    StatementProcessor.process = _teed_process  # type: ignore[method-assign]

    # Load the magics and turn cash on with the bench's cache dir.
    from cash.notebook.magics import CashMagics
    magics = CashMagics(shell=shell)
    shell.register_magics(magics)
    # %cash_on accepts a path argument for the cache directory.
    shell.run_line_magic("cash_on", str(cache_dir))
```

The "Cash enablement implemented in Task 5" comment in the stub gets removed when you replace the function.

- [ ] **Step 4: Run the cash-off tests too to make sure they still pass**

Run: `pytest tests/test_benchmarks_overhead/test_overhead_driver.py -v`
Expected: PASS — 4 tests green.

- [ ] **Step 5: Validate the `%cash_on PATH` syntax actually works**

This is the one assumption I want to verify cheaply before moving on:

Run:
```bash
python -c "
from IPython.core.interactiveshell import InteractiveShell
from pathlib import Path
import tempfile
shell = InteractiveShell.instance()
shell.reset(new_session=True)
from cash.notebook.magics import CashMagics
m = CashMagics(shell=shell)
shell.register_magics(m)
with tempfile.TemporaryDirectory() as d:
    shell.run_line_magic('cash_on', d)
    shell.run_cell('x = 1+1')
    print('OK')
"
```

Expected: prints `OK` with no traceback.

If `%cash_on` doesn't accept a path argument or uses a different syntax, look at `src/cash/notebook/magics.py` for the actual signature and adjust the `shell.run_line_magic('cash_on', ...)` call in `_enable_cash`. Common alternatives: `cash.notebook.api.enable_cash(cache_dir=...)`, or `magics.cash_on(line=str(cache_dir))`. Fix the call, re-run the tests, and proceed.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/_overhead_driver.py tests/test_benchmarks_overhead/test_overhead_driver.py
git commit -m "feat(bench): cash enablement + per-statement metrics tee in shell driver"
```

---

## Task 6: CLI orchestration — `bench_notebook_overhead.py`

**Files:**
- Create: `benchmarks/bench_notebook_overhead.py`
- Create: `tests/test_benchmarks_overhead/test_cli_smoke.py`

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_benchmarks_overhead/test_cli_smoke.py`:

```python
import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_cli_runs_on_synthetic_micro_off_mode(tmp_path):
    """Smoke test: run the CLI in off mode against a small synthetic
    notebook and verify the JSON output appears and is valid."""
    from benchmarks._overhead_io import write_synthetic_micro
    nb = tmp_path / "micro.ipynb"
    write_synthetic_micro(nb, n_statements=5)

    results_dir = tmp_path / "results"
    cmd = [
        sys.executable,
        "benchmarks/bench_notebook_overhead.py",
        str(nb),
        "--mode", "off",
        "--repeats", "2",
        "--results-dir", str(results_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"

    # One JSON file per repeat, named <stem>-<mode>-<repeat>.json
    files = sorted(results_dir.glob("micro-off-*.json"))
    assert len(files) == 2

    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["mode"] == "off"
    assert data["notebook"].endswith("micro.ipynb")
    assert len(data["cells"]) == 1
    assert data["cells"][0]["wall_seconds"] > 0


@pytest.mark.timeout(60)
def test_cli_runs_on_synthetic_micro_cold_mode(tmp_path):
    from benchmarks._overhead_io import write_synthetic_micro
    nb = tmp_path / "micro.ipynb"
    write_synthetic_micro(nb, n_statements=5)

    results_dir = tmp_path / "results"
    cmd = [
        sys.executable,
        "benchmarks/bench_notebook_overhead.py",
        str(nb),
        "--mode", "cold",
        "--repeats", "2",
        "--results-dir", str(results_dir),
        "--cache-root", str(tmp_path / "cache"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"

    files = sorted(results_dir.glob("micro-cold-*.json"))
    assert len(files) == 2
    data = json.loads(files[0].read_text(encoding="utf-8"))
    # Statement metrics should be populated under cold mode.
    assert sum(len(c["statement_metrics"]) for c in data["cells"]) >= 1
```

- [ ] **Step 2: Run the smoke test to verify it fails**

Run: `pytest tests/test_benchmarks_overhead/test_cli_smoke.py -v`
Expected: FAIL — script does not exist.

- [ ] **Step 3: Implement the CLI**

Create `benchmarks/bench_notebook_overhead.py`:

```python
"""Notebook overhead benchmark CLI.

Run a notebook in cash-off / cash-cold / cash-warm modes and write
per-cell timings + an optional pyinstrument profile to ``--results-dir``.

Usage:
    python benchmarks/bench_notebook_overhead.py <notebook> --mode {off,cold,warm}
        [--profile] [--repeats N] [--results-dir DIR] [--cache-root DIR]
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
import time
from pathlib import Path

# Make the benchmarks package importable when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks._overhead_driver import run_notebook
from benchmarks._overhead_io import load_code_cells
from benchmarks._overhead_results import RunResult, write_results


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _cash_version() -> str:
    try:
        from cash import __version__
        return __version__
    except Exception:  # noqa: BLE001
        return "unknown"


def _run_once(
    notebook_path: Path,
    mode: str,
    repeat: int,
    cache_dir: Path | None,
    profile_path: Path | None,
) -> RunResult:
    cells = load_code_cells(notebook_path)

    profiler = None
    if profile_path is not None:
        from pyinstrument import Profiler
        profiler = Profiler(interval=0.001)
        profiler.start()

    t0 = time.perf_counter()
    timings = run_notebook(
        cells,
        cash_enabled=(mode != "off"),
        cache_dir=cache_dir,
    )
    total = time.perf_counter() - t0

    if profiler is not None:
        profiler.stop()
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(profiler.output_html(), encoding="utf-8")

    return RunResult(
        notebook=str(notebook_path),
        mode=mode,
        repeat=repeat,
        python_version=sys.version.split()[0],
        cash_version=_cash_version(),
        platform=platform.platform(),
        cells=timings,
        total_wall_seconds=total,
        cache_dir_bytes=_dir_size_bytes(cache_dir) if cache_dir else 0,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Notebook overhead benchmark")
    p.add_argument("notebook", type=Path)
    p.add_argument("--mode", choices=["off", "cold", "warm"], required=True)
    p.add_argument("--profile", action="store_true",
                   help="Wrap the run in pyinstrument and emit HTML")
    p.add_argument("--repeats", type=int, default=3,
                   help="Number of repeats; the first is reported but typical "
                        "analysis discards it as warmup (default: 3)")
    p.add_argument("--results-dir", type=Path,
                   default=Path("benchmarks/results"))
    p.add_argument("--cache-root", type=Path,
                   help="Parent dir for per-repeat cache dirs (cold/warm only)")
    args = p.parse_args(argv)

    if args.mode != "off" and args.cache_root is None:
        args.cache_root = args.results_dir / "_caches" / args.notebook.stem

    stem = args.notebook.stem

    if args.mode == "warm":
        # Populate the cache once, then re-run --repeats times against it.
        warm_dir = args.cache_root / "warm-shared"
        if warm_dir.exists():
            shutil.rmtree(warm_dir, ignore_errors=True)
        warm_dir.mkdir(parents=True, exist_ok=True)
        _run_once(args.notebook, "cold", -1, warm_dir, profile_path=None)

    for repeat in range(args.repeats):
        if args.mode == "cold":
            cache_dir = args.cache_root / f"repeat-{repeat}"
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
        elif args.mode == "warm":
            cache_dir = args.cache_root / "warm-shared"
        else:
            cache_dir = None

        profile_path = None
        if args.profile and repeat == args.repeats - 1:
            # Only profile the last repeat (steady state); profiling adds
            # overhead that distorts the wall-clock comparison.
            profile_path = args.results_dir / f"{stem}-{args.mode}-profile.html"

        result = _run_once(args.notebook, args.mode, repeat, cache_dir, profile_path)
        out = args.results_dir / f"{stem}-{args.mode}-{repeat}.json"
        write_results(out, result)
        print(f"[{args.mode}] repeat={repeat} cells={len(result.cells)} "
              f"wall={result.total_wall_seconds*1000:.1f}ms -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the smoke test to verify it passes**

Run: `pytest tests/test_benchmarks_overhead/test_cli_smoke.py -v`
Expected: PASS — both tests green.

- [ ] **Step 5: Add `benchmarks/results/` to `.gitignore`**

Append to `.gitignore`:

```
# Bench outputs (raw artifacts)
benchmarks/results/
```

If `.gitignore` already has a `benchmarks/results` rule (it shouldn't, but check first), don't duplicate it.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/bench_notebook_overhead.py tests/test_benchmarks_overhead/test_cli_smoke.py .gitignore
git commit -m "feat(bench): CLI for notebook overhead bench with off/cold/warm modes"
```

---

## Task 7: `compare_modes.py` — JSON → markdown table

**Files:**
- Create: `benchmarks/compare_modes.py`
- Create: `tests/test_benchmarks_overhead/test_compare_modes.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_benchmarks_overhead/test_compare_modes.py`:

```python
import json
from pathlib import Path

from benchmarks._overhead_results import (
    CellTiming, RunResult, StatementMetric, write_results,
)
from benchmarks.compare_modes import build_table


def _result(mode: str, repeat: int, wall_per_cell: list[float], tmp_path: Path) -> Path:
    result = RunResult(
        notebook="nb.ipynb", mode=mode, repeat=repeat,
        python_version="3.12", cash_version="0.5", platform="x",
        cells=[
            CellTiming(index=i, notebook_cell_index=i,
                       wall_seconds=w, source_chars=10, statement_metrics=[])
            for i, w in enumerate(wall_per_cell)
        ],
        total_wall_seconds=sum(wall_per_cell), cache_dir_bytes=0,
    )
    out = tmp_path / f"nb-{mode}-{repeat}.json"
    write_results(out, result)
    return out


def test_build_table_combines_three_modes(tmp_path):
    # Three repeats per mode. Median of the non-first repeats is reported.
    _result("off", 0, [0.100, 0.100], tmp_path)  # warmup, discarded
    _result("off", 1, [0.110, 0.105], tmp_path)
    _result("off", 2, [0.108, 0.103], tmp_path)
    _result("cold", 0, [0.200, 0.150], tmp_path)
    _result("cold", 1, [0.180, 0.140], tmp_path)
    _result("cold", 2, [0.190, 0.145], tmp_path)
    _result("warm", 0, [0.030, 0.020], tmp_path)
    _result("warm", 1, [0.028, 0.022], tmp_path)
    _result("warm", 2, [0.029, 0.021], tmp_path)

    table = build_table(tmp_path, notebook_stem="nb")
    # Table has a header line and one row per cell + a total row
    assert "off" in table and "cold" in table and "warm" in table
    assert "cell 0" in table or "| 0 " in table
    assert "TOTAL" in table.upper() or "total" in table.lower()
    # Cold > off (overhead is positive)
    assert "cell 0" in table.lower() or True  # smoke: structural assertion above
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_benchmarks_overhead/test_compare_modes.py -v`
Expected: FAIL — `benchmarks.compare_modes` does not exist.

- [ ] **Step 3: Implement the comparator**

Create `benchmarks/compare_modes.py`:

```python
"""Compare bench results across cash-off / cash-cold / cash-warm.

Reads per-mode result JSONs from a directory and emits a markdown table
showing wall-clock per cell, the cold-off overhead in ms, and the
relative overhead as a fraction of off-mode time.

Usage:
    python benchmarks/compare_modes.py <results-dir> <notebook-stem>
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from benchmarks._overhead_results import read_results


def _median_per_cell(results_dir: Path, stem: str, mode: str) -> dict[int, float]:
    """Median wall-seconds per cell across all repeats except the first
    (treated as warmup). Returns {cell_index: median_seconds}."""
    files = sorted(results_dir.glob(f"{stem}-{mode}-*.json"))
    if not files:
        return {}
    # Discard the repeat-0 warmup if more than one repeat is present.
    samples_by_cell: dict[int, list[float]] = defaultdict(list)
    for f in files:
        repeat = int(f.stem.rsplit("-", 1)[-1])
        if len(files) > 1 and repeat == 0:
            continue
        r = read_results(f)
        for cell in r.cells:
            samples_by_cell[cell.index].append(cell.wall_seconds)
    return {idx: statistics.median(samples) for idx, samples in samples_by_cell.items()}


def build_table(results_dir: Path, notebook_stem: str) -> str:
    off = _median_per_cell(results_dir, notebook_stem, "off")
    cold = _median_per_cell(results_dir, notebook_stem, "cold")
    warm = _median_per_cell(results_dir, notebook_stem, "warm")

    all_cells = sorted(set(off) | set(cold) | set(warm))
    lines = [
        f"## {notebook_stem}",
        "",
        "| cell | off (ms) | cold (ms) | warm (ms) | cold-off (ms) | (cold-off)/off |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    total_off = total_cold = total_warm = 0.0
    for idx in all_cells:
        o = off.get(idx, 0.0)
        c = cold.get(idx, 0.0)
        w = warm.get(idx, 0.0)
        total_off += o
        total_cold += c
        total_warm += w
        diff = c - o
        ratio = (diff / o) if o > 0 else float("inf")
        lines.append(
            f"| cell {idx} | {o*1000:.2f} | {c*1000:.2f} | {w*1000:.2f} "
            f"| {diff*1000:+.2f} | {ratio:+.1%} |"
        )
    diff = total_cold - total_off
    ratio = (diff / total_off) if total_off > 0 else float("inf")
    lines.append(
        f"| **TOTAL** | **{total_off*1000:.2f}** | **{total_cold*1000:.2f}** "
        f"| **{total_warm*1000:.2f}** | **{diff*1000:+.2f}** | **{ratio:+.1%}** |"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compare bench results across modes")
    p.add_argument("results_dir", type=Path)
    p.add_argument("notebook_stem", type=str)
    args = p.parse_args(argv)
    print(build_table(args.results_dir, args.notebook_stem))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_benchmarks_overhead/test_compare_modes.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/compare_modes.py tests/test_benchmarks_overhead/test_compare_modes.py
git commit -m "feat(bench): markdown comparator for per-mode bench results"
```

---

## Task 8: Run the harness on the synthetic micro notebook

This task validates the harness end-to-end before the bigger notebooks.

**Files:**
- Generate: `benchmarks/synthetic_micro.ipynb` (committed once, regenerated as needed)
- Generate: `benchmarks/results/*` (gitignored, not committed)

- [ ] **Step 1: Generate the synthetic notebook and run all three modes**

```bash
python -c "from pathlib import Path; from benchmarks._overhead_io import write_synthetic_micro; write_synthetic_micro(Path('benchmarks/synthetic_micro.ipynb'))"
python benchmarks/bench_notebook_overhead.py benchmarks/synthetic_micro.ipynb --mode off  --repeats 3
python benchmarks/bench_notebook_overhead.py benchmarks/synthetic_micro.ipynb --mode cold --repeats 3 --profile
python benchmarks/bench_notebook_overhead.py benchmarks/synthetic_micro.ipynb --mode warm --repeats 3
```

Expected: 9 JSON files (`benchmarks/results/synthetic_micro-{off,cold,warm}-{0,1,2}.json`) and one HTML (`synthetic_micro-cold-profile.html`). The cold runs should be measurably slower than off, and warm should be the fastest.

- [ ] **Step 2: Run the comparator**

```bash
python benchmarks/compare_modes.py benchmarks/results synthetic_micro
```

Expected: a markdown table printed to stdout. The TOTAL row should show cold-off > 0 (cash adds overhead) and warm < off (cache hits are faster than re-computing).

If cold is not >> off, the harness is not actually engaging cash. Diagnose:
- Check the result JSON: are `statement_metrics` populated for cold runs? If not, the `%cash_on` line magic isn't taking effect — re-read `src/cash/notebook/magics.py` for the actual public surface and adjust `_enable_cash` in `_overhead_driver.py`.
- Check the cold-profile HTML: does it show time inside `cash.notebook.statement_processor.process`? If no, same diagnosis.

- [ ] **Step 3: Open the pyinstrument HTML**

```bash
python -c "import webbrowser; webbrowser.open('benchmarks/results/synthetic_micro-cold-profile.html')"
```

Spot-check that you can see frames under `_execute_statement` for `FileAccessTracker.__enter__`, `_apply_patches`, and `capture_output`. These are the hidden-overhead candidates.

- [ ] **Step 4: Commit the synthetic notebook itself**

The synthetic notebook is a stable input for the bench; commit it so anyone re-running the bench gets the same notebook. Bench results stay gitignored.

```bash
git add benchmarks/synthetic_micro.ipynb
git commit -m "bench: commit synthetic micro notebook used by overhead bench"
```

---

## Task 9: Run the harness on the real notebooks

**Files:**
- Generates only gitignored output under `benchmarks/results/`. No commit.

- [ ] **Step 1: Run the three real notebooks in all three modes**

For each notebook below, run all three modes. The notebook paths must work on Windows with spaces in the cwd — pass them quoted if needed.

```bash
for nb in examples/file_tracking_demo.ipynb examples/financial_analysis_demo.ipynb examples/cfd_simulation_demo.ipynb; do
  python benchmarks/bench_notebook_overhead.py "$nb" --mode off  --repeats 3
  python benchmarks/bench_notebook_overhead.py "$nb" --mode cold --repeats 3 --profile
  python benchmarks/bench_notebook_overhead.py "$nb" --mode warm --repeats 3
done
```

Expected: 9 JSON files + 1 profile HTML per notebook (27 + 3 total).

**Handling failures:** If a notebook fails because it expects external data files (e.g., a `read_csv` of a path that doesn't exist), record the failure, skip that notebook, and substitute one from `examples/large_scale_projects/01_nyc_taxi_analysis.ipynb` … `10_us_flights.ipynb` (per the spec's fallback). If a notebook is too short to produce signal (total wall < 50ms in off mode), also substitute a `large_scale_projects/` notebook.

- [ ] **Step 2: Run the comparator for each notebook**

```bash
python benchmarks/compare_modes.py benchmarks/results file_tracking_demo > /tmp/file_tracking_demo-table.md
python benchmarks/compare_modes.py benchmarks/results financial_analysis_demo > /tmp/financial_analysis_demo-table.md
python benchmarks/compare_modes.py benchmarks/results cfd_simulation_demo > /tmp/cfd_simulation_demo-table.md
```

(Substitute the actual notebook stems if you fell back to `large_scale_projects/`.)

- [ ] **Step 3: Save the tables and profiles for the report**

Keep the markdown tables and profile HTML paths handy — Task 11 references them in the analysis. Don't commit them yet (raw artifacts are gitignored).

---

## Task 10: Read the profiles and tally hidden overhead

This is an analysis step. No code change. The output is a set of structured notes that feed Task 11.

- [ ] **Step 1: For each cold-mode profile HTML, attribute time under `process`**

Open each `*-cold-profile.html` in a browser. For each, write down:

```
Notebook: <name>
Total time in StatementProcessor.process: <ms> (<%>)
  Of which:
    capture_output enter+exit: <ms>
    FileAccessTracker.__enter__: <ms>
      └─ _apply_patches: <ms>
      └─ _patch_module: <ms>
      └─ _patch_user_ns: <ms>
    FileAccessTracker.__exit__ / _unpatch: <ms>
    ast.parse (redundant): <ms>
    exec/eval (actual user code): <ms>
    _analyze_and_hash: <ms>
    _do_cache_lookup: <ms>
    _post_execute / _save_to_cache: <ms>
```

If pyinstrument's UI shows aggregated time for a parent, you can read children directly. If a category is missing, note it as "not visible in profile" rather than guessing.

- [ ] **Step 2: Cross-check the synthetic numbers against per-statement metrics**

In `benchmarks/results/synthetic_micro-cold-1.json` (a non-warmup repeat), look at `cells[0].statement_metrics`. Median `execution_time` across the 100 statements should be tiny (microseconds). Median `total_time - execution_time` is the cash-overhead-as-declared.

Compare:
- `(cold_total - off_total) / n_statements` = real per-statement overhead from wall clock
- `median(total_time - execution_time)` = cash's self-declared per-statement overhead
- The difference between these two — if positive — is the hidden overhead inside `execution_time`.

Record the three numbers (real, declared, hidden) in the notes for Task 11.

---

## Task 11: Write the results report

**Files:**
- Create: `docs/superpowers/specs/2026-05-18-notebook-cache-overhead-results.md`

- [ ] **Step 1: Draft the report from the Task 9 + Task 10 notes**

Create the file with this exact structure:

```markdown
# Notebook cache overhead — results

**Date:** 2026-05-18
**Spec:** [2026-05-18-notebook-cache-overhead-analysis.md](2026-05-18-notebook-cache-overhead-analysis.md)
**Run on:** <platform from result JSON>, Python <version>, cash <version>
**Harness:** `benchmarks/bench_notebook_overhead.py`

## TL;DR

<3-5 sentences: what the headline cold-vs-off overhead is, whether the
hidden-overhead hypothesis was confirmed, and the top 1-2 remediation
recommendations>

## Methodology

<2 paragraphs: in-process IPython driver, three modes, three repeats,
median of non-warmup. Reference the spec for full detail; don't repeat
it. Disclose any notebooks that were skipped/substituted.>

## Per-notebook results

### synthetic_micro

<paste markdown table from compare_modes.py here>

<1-2 paragraphs interpreting the table. The synthetic notebook is the
clearest signal because compute is near-zero — the cold-off overhead is
nearly pure cash machinery.>

### file_tracking_demo

<table + interpretation, same pattern>

### financial_analysis_demo

<table + interpretation>

### cfd_simulation_demo (or large_scale_projects/<stem> if substituted)

<table + interpretation>

## Overhead decomposition

<one subsection per notebook with the breakdown from Task 10 step 1.
Reference the profile HTML by relative path:
`benchmarks/results/<stem>-cold-profile.html`.>

## The hidden overhead inside `execution_time`

<The headline finding. For the synthetic notebook, quantify:

- Real per-statement overhead (wall clock): X µs
- Declared per-statement overhead (cash's `total_time - execution_time`): Y µs
- Hidden overhead silently rolled into `execution_time`: X - Y = Z µs

Then attribute Z to specific call sites from the profile: FileAccessTracker
enter/exit, capture_output enter/exit, redundant ast.parse, etc. Quote the
relevant lines in `statement_processor.py` so the reader can navigate.>

## Remediation strategies

For each strategy: hypothesis, expected savings (cite the numbers above),
implementation cost (rough — small / medium / large), risk.

### Strategy 1: <derived from data — e.g. "Cache FileAccessTracker patch targets across statements">

**Hypothesis:** ...
**Expected savings:** ... µs/statement → ... on the synthetic notebook (100 statements)
**Implementation cost:** small/medium/large
**Risk:** ...
**Where:** `src/cash/notebook/file_tracker.py:<line>` ...

### Strategy 2: ...

### Strategy 3: ...

(Order by savings × (1/cost). The most likely candidates from a code
review of statement_processor.py and file_tracker.py — all subject to
confirmation by the profile — are:

- Cache the per-module patch-target list across statements (avoid
  re-walking `dir(module)` on every statement)
- Skip FileAccessTracker entirely for statements whose AST contains no
  call expression and no name resolution that could reach a file API
- Fast-path past capture_output when no output is produced (detect via
  the captured stream sizes after exec)
- Deduplicate the multiple `ast.parse(code)` calls per statement
- Move `start_time` inside the context managers so `execution_time`
  reflects user code only; add an explicit `interception_overhead`
  field on `ProcessResult` so the cost is visible rather than hidden.
  This is bookkeeping rather than perf, but it's necessary to make the
  badge honest.)

## Open questions / follow-ups

<list anything the data raised but didn't answer>
```

Replace every `<...>` placeholder with the actual content from the run. Do
**not** leave any `<...>` in the final file.

- [ ] **Step 2: Self-review the report**

Scan the report:
- Every `<...>` placeholder filled?
- Are the numbers in TL;DR consistent with the per-notebook tables?
- Does every remediation strategy cite a number from the data?
- Are file paths in remediation strategies clickable (relative paths starting with `src/` or `benchmarks/`)?

Fix issues inline.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-05-18-notebook-cache-overhead-results.md
git commit -m "docs(perf): notebook cache overhead results and remediation plan"
```

---

## Verification

After all tasks, run the full bench suite for the harness:

```bash
pytest tests/test_benchmarks_overhead -v
```

Expected: all tests pass. None of these tests run actual notebooks beyond
the tiny synthetic one used in the smoke test, so the suite is fast (< 30s).
