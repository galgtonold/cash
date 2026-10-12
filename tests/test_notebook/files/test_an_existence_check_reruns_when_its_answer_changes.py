"""A statement that only checks whether a path exists runs again once the
answer changes, and so does what is built on its value."""

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SLEEP = f"time.sleep({ABOVE_PERSISTENCE_FLOOR_S})"


def test_a_flag_check_follows_the_flag(cash_magics, mock_shell, tmp_path):
    flag = tmp_path / "flag.txt"
    flag.write_text("x", encoding="utf-8")
    path = str(flag).replace("\\", "/")
    run_cash_cell(cash_magics, "import os, time")
    code = f"ready = os.path.exists('{path}') and {SLEEP} is None"
    run_cash_cell(cash_magics, code)
    assert mock_shell.user_ns["ready"] is True
    run_cash_cell(cash_magics, code)
    assert mock_shell.user_ns["ready"] is True
    flag.unlink()
    run_cash_cell(cash_magics, code)
    assert mock_shell.user_ns["ready"] is False
    flag.write_text("x", encoding="utf-8")
    run_cash_cell(cash_magics, code)
    assert mock_shell.user_ns["ready"] is True


def test_a_pending_list_and_what_is_built_on_it_follow_the_outputs(cash_magics, mock_shell, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "a.txt").write_text("1", encoding="utf-8")
    folder = str(out).replace("\\", "/")
    run_cash_cell(
        cash_magics,
        "import os, time\ndef pending(names):\n    "
        + SLEEP
        + "\n    return [n for n in names if not os.path.isfile(f'"
        + folder
        + "/{n}.txt')]",
    )
    cells = ["todo = pending(['a', 'b'])", f"labels = [n.upper() for n in todo if {SLEEP} is None]"]
    for code in cells:
        run_cash_cell(cash_magics, code)
    assert mock_shell.user_ns["labels"] == ["B"]
    (out / "a.txt").unlink()
    for code in cells:
        run_cash_cell(cash_magics, code)
    assert mock_shell.user_ns["todo"] == ["a", "b"]
    assert mock_shell.user_ns["labels"] == ["A", "B"]
    (out / "b.txt").write_text("2", encoding="utf-8")
    for code in cells:
        run_cash_cell(cash_magics, code)
    assert mock_shell.user_ns["labels"] == ["A"]
