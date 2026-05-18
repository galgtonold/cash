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
