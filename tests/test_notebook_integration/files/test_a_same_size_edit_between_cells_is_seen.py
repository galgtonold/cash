"""A same-size edit made right after a cell run is seen by the next run.

Cells run without history (``shell.run_cell(code)``, an agent, a frontend
extension) leave the kernel's ``execution_count`` where it was. The statement
freshness check kept its per-file answers for as long as that count stayed
put, up to two seconds, so the next cell reused the previous cell's "fresh":
a file rewritten with the same byte count within that window was served
stale, and ``data`` came back as the old text.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.files, pytest.mark.timeout(120)]


def test_a_same_size_edit_right_after_a_history_less_run_is_seen(nb_runner, tmp_path):
    data = tmp_path / "d.txt"
    data.write_text("hello world", encoding="utf-8")
    path = str(data).replace("\\", "/")
    nb_runner.create_notebook(
        [
            f"data = open('{path}').read()",
            "def slow(n):\n    import time\n    time.sleep(0.3)\n    return n * 2",
            "y = slow(len(data))",
        ]
    )
    nb_runner.start_kernel()
    for _ in range(2):
        for cell in (1, 2, 3):
            nb_runner.run_cell_without_history(cell)
    assert nb_runner.peek("data") == repr("hello world")

    data.write_text("HELLO WORLD", encoding="utf-8")  # same byte count, no pause
    for cell in (1, 2, 3):
        nb_runner.run_cell_without_history(cell)
    assert nb_runner.peek("data") == repr("HELLO WORLD"), "a same-size edit was served stale"


def test_an_unchanged_file_stays_cached(nb_runner, tmp_path):
    """Positive control: with nothing edited the read is still restored."""
    data = tmp_path / "d.txt"
    data.write_text("hello world", encoding="utf-8")
    path = str(data).replace("\\", "/")
    nb_runner.create_notebook(["import cash\n%cash_badge print", f"data = open('{path}').read()\nprint('read')"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_cell(2)
    assert "CACHED" in nb_runner.get_raw_output(2), nb_runner.get_raw_output(2)
    assert nb_runner.peek("data") == repr("hello world")
