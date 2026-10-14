"""A statement that reads back a file a statement of this session wrote, and
runs again because the write changed it, is named once
(``NOTEBOOK-FILE-ROUNDTRIP``).

Every write gives such a file new bytes (an Excel file's zip timestamps), so
the read, and whatever is computed from it, can never be reused. The fix is
the user's: keep the data in a variable and write the file once, at the end.
Silent when the reread is cheap, when nothing in this session wrote the file
(a checkpoint from an earlier kernel, an input), and after the first time.
"""

from __future__ import annotations

import warnings

from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell

CODE = "NOTEBOOK-FILE-ROUNDTRIP"
#: Above the hint's one-second floor, with room for a slow machine's timer.
SLOW = 1.1


def _cells(path) -> dict[str, str]:
    p = str(path).replace("\\", "/")
    return {
        "imports": f"import pathlib, time\nP = {p!r}",
        "write": "pathlib.Path(P).write_text(str(time.time_ns()))",
        "read": f"data = [pathlib.Path(P).read_text(), time.sleep({SLOW})][0]",
        "cheap": "data = pathlib.Path(P).read_text()",
        "use": f"n = [len(data), time.sleep({SLOW})][0]",
        "other": f"m = [1, time.sleep({SLOW})][0]",
    }


def _run(magics, *cells: str) -> list[warnings.WarningMessage]:
    """Run *cells* in turn; the round-trip warnings they raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for cell in cells:
            run_cash_cell(magics, cell)
    return [w for w in caught if getattr(w.message, "code", None) == CODE]


def _status(magics, code: str) -> str:
    rows = magics.cash_status("dict")["last_cell"]["statements"]
    return next(r["status"] for r in rows if code.startswith(r.get("code") or "\0"))


def test_a_costly_reread_of_a_file_written_this_session_is_named(cash_magics, tmp_path):
    c = _cells(tmp_path / "report.txt")
    assert not _run(cash_magics, c["imports"], c["write"], c["read"]), "a first read has nothing to reuse"

    found = _run(cash_magics, c["write"], c["read"])

    assert _status(cash_magics, c["read"]) == CacheStatus.COMPUTED
    assert len(found) == 1, found
    text = str(found[0].message)
    assert "report.txt" in text
    assert "`pathlib.Path(P).write_text(str(time.time_ns()))`" in text, "names the writing statement"
    assert "`data = [pathlib.Path(P).read_text(), time.sleep(1.1)][0]`" in text, "names the reading statement"
    assert "Fix: keep the data in a variable" in text and "write the file once, as the last step" in text


def test_it_is_named_once_per_file(cash_magics, tmp_path):
    c = _cells(tmp_path / "report.txt")
    _run(cash_magics, c["imports"], c["write"], c["read"])

    found = _run(cash_magics, c["write"], c["read"], c["write"], c["read"])

    assert _status(cash_magics, c["read"]) == CacheStatus.COMPUTED, "control: it ran again"
    assert len(found) == 1, found


def test_a_cheap_reread_is_not_named(cash_magics, tmp_path):
    c = _cells(tmp_path / "report.txt")
    _run(cash_magics, c["imports"], c["write"], c["cheap"])

    found = _run(cash_magics, c["write"], c["cheap"])
    assert _status(cash_magics, c["cheap"]) == CacheStatus.COMPUTED, "control: it ran again"
    found += _run(cash_magics, c["other"])

    assert not found, "a costly statement that does not use what was read is no part of it"


def test_a_cheap_reread_with_costly_work_on_it_is_named(cash_magics, tmp_path):
    c = _cells(tmp_path / "report.txt")
    _run(cash_magics, c["imports"], c["write"], c["cheap"], c["use"])

    found = _run(cash_magics, c["write"], c["cheap"], c["use"])

    assert len(found) == 1, found
    assert "read back by `data = pathlib.Path(P).read_text()`" in str(found[0].message), "names the read"


def test_a_file_this_session_did_not_write_is_not_named(cash_magics, tmp_path):
    path = tmp_path / "input.txt"
    path.write_text("1", encoding="utf-8")
    c = _cells(path)
    _run(cash_magics, c["imports"], c["read"])
    path.write_text("22", encoding="utf-8")

    found = _run(cash_magics, c["read"])

    assert _status(cash_magics, c["read"]) == CacheStatus.COMPUTED, "control: the change made it run"
    assert not found


def _restart(magics) -> None:
    """What a kernel restart leaves: no session state, none of the variables."""
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in ("pathlib", "time", "P", "data"):
        magics.shell.user_ns.pop(name, None)


def test_a_file_an_earlier_session_wrote_is_not_named(cash_magics, tmp_path):
    """The positive control is the test above it: the same runs without the
    restart between the last write and the read are named."""
    c = _cells(tmp_path / "checkpoint.txt")
    _run(cash_magics, c["imports"], c["write"], c["read"], c["write"])
    _restart(cash_magics)

    found = _run(cash_magics, c["imports"], c["read"])

    assert _status(cash_magics, c["read"]) == CacheStatus.COMPUTED, "control: the rewrite made it run"
    assert not found


def test_the_same_runs_without_the_restart_are_named(cash_magics, tmp_path):
    c = _cells(tmp_path / "checkpoint.txt")
    _run(cash_magics, c["imports"], c["write"], c["read"], c["write"])

    found = _run(cash_magics, c["imports"], c["read"])

    assert len(found) == 1, found


def test_a_file_written_again_after_a_restart_is_named(cash_magics, tmp_path):
    """After a restart the writer runs again and the stored read is thrown
    away for it: a round trip of this session, though the read has not run
    in it before."""
    c = _cells(tmp_path / "report.txt")
    _run(cash_magics, c["imports"], c["write"], c["read"])
    _restart(cash_magics)

    found = _run(cash_magics, c["imports"], c["write"], c["read"])

    assert _status(cash_magics, c["read"]) == CacheStatus.COMPUTED
    assert len(found) == 1, found
