"""A ``def`` is not reported as a statement cash could not cache.

Four ``^NOT CACHED: def draw_roc(ax): ... __main__
function/class output(s) draw_roc are unrestorable by value; statement
re-executes (lineage persists)`` rows in a report cell's badge. A definition
always re-runs at no cost and its lineage persists either way; the reason is for
``f = make_fn()``. Most defs never reached it -- they are below the too-cheap
floor -- but a def whose body reads a frame loaded from a file inherits the
file dependency, which skips the floor.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell


@pytest.fixture
def magics(cash_magics, mock_shell, monkeypatch):
    # The notebook's module is ``__main__``, the name the skip reason reports.
    mock_shell.user_ns["__name__"] = "__main__"
    m = cash_magics
    seen: list = []
    original = m._update_last_cell_metrics
    monkeypatch.setattr(
        m, "_update_last_cell_metrics", lambda metrics, total: (seen.append(list(metrics)), original(metrics, total))
    )
    return m, seen


def test_a_def_reading_file_loaded_data_has_no_skip_reason(magics, tmp_path):
    m, seen = magics
    data = tmp_path / "a.txt"
    data.write_text("1\n2\n3\n")
    run_cash_cell(m, f"import time\nrows = open(r'{data}').read().split()\ntime.sleep(0.02)")
    run_cash_cell(m, "def draw(ax):\n    return len(rows)")
    [row] = [x for x in seen[-1] if str(x.get("code", "")).startswith("def draw")]
    assert not row.get("skipped_reason"), row.get("skipped_reason")
    assert m.shell.user_ns["draw"](None) == 3


def test_a_function_made_by_a_call_keeps_its_reason(magics, tmp_path):
    m, seen = magics
    data = tmp_path / "a.txt"
    data.write_text("1\n2\n3\n")
    run_cash_cell(m, f"import time\nrows = open(r'{data}').read().split()\ntime.sleep(0.02)")
    run_cash_cell(m, "def make():\n    def inner():\n        return len(rows)\n    return inner")
    run_cash_cell(m, "import time\nf = (time.sleep(0.02), make())[1]")
    [row] = [x for x in seen[-1] if str(x.get("code", "")).startswith("f = ")]
    assert "unrestorable by value" in (row.get("skipped_reason") or ""), row
