import csv
import subprocess
import sys

import pytest


@pytest.mark.timeout(120)
def test_quick_run_writes_every_case_with_its_machine(tmp_path):
    out = tmp_path / "call_cost.csv"
    cmd = [sys.executable, "benchmarks/measure_call_cost.py", "--quick", "--rounds", "1", "--out", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"

    with out.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    cases = {(r["tool"], r["argument"], r["outcome"]) for r in rows}
    # lru_cache cannot take an array; every other pairing is measured.
    assert len(cases) == 10, sorted(cases)
    assert all(float(r["seconds_per_call"]) > 0 for r in rows)
    assert all(r["cpu"] and r["python"] and r["cash_version"] for r in rows)
    assert "| `@cash.cache` |" in proc.stdout
    assert "not supported" in proc.stdout


def test_seconds_are_shown_in_the_unit_the_page_uses():
    from benchmarks.measure_call_cost import format_seconds

    assert format_seconds(0.00000004) == "0.04 µs"
    assert format_seconds(0.0000055) == "5.5 µs"
    assert format_seconds(0.000110) == "110 µs"
    assert format_seconds(0.00104) == "1.0 ms"
    assert format_seconds(0.0123) == "12 ms"
