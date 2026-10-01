"""A same-size edit made just after a cell run is seen by the next cell run.

The statement freshness check keeps one answer per file for the rest of a
cell run (up to two seconds), keyed on the shell's ``execution_count``. That
count only moves for a cell run with history: ``shell.run_cell(code)`` and a
frontend's history-less execute leave it where it was, so the next cell run
reused the last one's answer, and a file rewritten with the same byte count
within that window was served stale ('hello world' after it became
'HELLO WORLD'). The answers now last for one cell run, whatever the count.

``MockShell`` has no ``execution_count``, which made every lookup check afresh;
these tests give it a fixed one, as a shell that runs cells without history
has.
"""

from __future__ import annotations

from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell

CELLS = ["data = open('d.txt').read()", "y = len(data) * 2"]


def _run_twice(magics) -> None:
    for _ in range(2):
        for cell in CELLS:
            run_cash_cell(magics, cell)


def test_a_same_size_edit_right_after_a_run_is_seen(cash_magics, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cash_magics.shell.execution_count = 7  # a shell that runs cells without history
    data = tmp_path / "d.txt"
    data.write_text("hello world", encoding="utf-8")
    ns = cash_magics.shell.user_ns
    _run_twice(cash_magics)
    assert ns["data"] == "hello world"

    data.write_text("HELLO WORLD", encoding="utf-8")  # same byte count, no pause
    for cell in CELLS:
        run_cash_cell(cash_magics, cell)
    assert ns["data"] == "HELLO WORLD", "a same-size edit made right after a run was served stale"


def test_an_unchanged_file_is_still_served_from_the_cache(cash_magics, tmp_path, monkeypatch):
    """Positive control: the read is checked again, and still restored."""
    monkeypatch.chdir(tmp_path)
    cash_magics.shell.execution_count = 7
    (tmp_path / "d.txt").write_text("hello world", encoding="utf-8")
    _run_twice(cash_magics)

    captured: list[list[dict]] = []
    real_render = cash_magics.badges.render

    def capture(metrics, **kw):
        captured.append(list(metrics))
        return real_render(metrics, **kw)

    cash_magics.badges.render = capture  # type: ignore[assignment]
    try:
        run_cash_cell(cash_magics, CELLS[0])
    finally:
        cash_magics.badges.render = real_render  # type: ignore[assignment]
    assert captured and captured[-1][-1]["status"] == CacheStatus.RESTORED, captured
    assert cash_magics.shell.user_ns["data"] == "hello world"
