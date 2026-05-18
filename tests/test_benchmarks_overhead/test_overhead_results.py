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
