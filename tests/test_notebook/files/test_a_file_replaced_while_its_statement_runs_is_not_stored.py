"""A file another process replaces while the statement reading it still runs.

The statement's entry fingerprinted its files when it was stored, after it had
run: the NEW file was recorded next to a value computed from the old one, and
every later run found the entry fresh and served the old value. The decorator
refuses such a store (``STORE-INPUT-CHANGED``); the notebook now does too.
"""

from __future__ import annotations

import os
import threading
import warnings

import pytest

from tests._cell_driver import run_cash_cell

#: The longest the statement and the replacing thread wait for each other.
WAIT_S = 30.0

#: Set by the statement once it has read the file, and by the thread once it
#: has replaced it: the replace lands between the read and the store however
#: slow the machine. A fixed 0.4 s window landed before the read on a slow
#: Windows runner, and the cell read the new file.
_READ = threading.Event()
_REPLACED = threading.Event()


def handoff() -> None:
    """Called by the statement right after its read: the file is replaced
    before the statement goes on."""
    _READ.set()
    assert _REPLACED.wait(WAIT_S), "the file was not replaced"


@pytest.fixture
def stores_everything(cash_instance):
    """Statements and calls are stored however cheap."""
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0


@pytest.fixture
def data(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("1", encoding="utf-8")
    return path


def _replace_soon(path, text):
    """Replace *path* from a thread of the test's own, as another process
    would, once the statement has read it: nothing of the statement's sees
    the write."""
    _READ.clear()
    _REPLACED.clear()

    def later():
        if not _READ.wait(WAIT_S):
            return
        tmp = str(path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
        _REPLACED.set()

    thread = threading.Thread(target=later, daemon=True)
    thread.start()
    return thread


def _run(magics, code):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        run_cash_cell(magics, code)
    return [getattr(w.message, "code", None) for w in rec]


def test_a_statement_whose_file_was_replaced_mid_run_is_not_served(cash_magics, stores_everything, data):
    path = str(data).replace("\\", "/")
    run_cash_cell(cash_magics, f"from {__name__} import handoff")
    statement = f"text = [open({path!r}).read(), handoff()][0]"
    ns = cash_magics.shell.user_ns
    thread = _replace_soon(data, "2")
    codes = _run(cash_magics, statement)
    thread.join()
    assert ns["text"] == "1", "the statement must read the file before it is replaced"
    assert "STORE-INPUT-CHANGED" in codes, codes

    run_cash_cell(cash_magics, statement)
    assert ns["text"] == "2"


def test_a_call_whose_file_was_replaced_mid_run_is_not_served(cash_magics, stores_everything, data):
    """The same through a function the call cache holds (`df = load(p)`)."""
    path = str(data).replace("\\", "/")
    run_cash_cell(
        cash_magics,
        f"from {__name__} import handoff\ndef load(p):\n    with open(p) as fh:\n        text = fh.read()\n"
        "    handoff()\n    return text",
    )
    ns = cash_magics.shell.user_ns
    thread = _replace_soon(data, "2")
    codes = _run(cash_magics, f"text = load({path!r})")
    thread.join()
    assert ns["text"] == "1", "the call must read the file before it is replaced"
    assert "STORE-INPUT-CHANGED" in codes, codes

    run_cash_cell(cash_magics, f"text = load({path!r})")
    assert ns["text"] == "2"


def test_a_statement_below_is_not_served_from_the_replaced_run(cash_magics, stores_everything, data):
    """What a statement below computed from the unsettled value is not served
    once the file has settled: the value's lineage said "the new file"."""
    path = str(data).replace("\\", "/")
    run_cash_cell(
        cash_magics,
        f"from {__name__} import handoff\ndef load(p):\n    with open(p) as fh:\n        text = fh.read()\n"
        "    handoff()\n    return text",
    )
    ns = cash_magics.shell.user_ns
    thread = _replace_soon(data, "2")
    _run(cash_magics, f"text = load({path!r})")
    thread.join()
    run_cash_cell(cash_magics, "t = int(text) * 10")
    assert ns["t"] == 10, "the statement must read the file before it is replaced"

    run_cash_cell(cash_magics, f"text = load({path!r})")
    run_cash_cell(cash_magics, "t = int(text) * 10")
    assert ns["t"] == 20


def test_a_settled_file_is_stored_and_served(cash_magics, stores_everything, data):
    """Control: with nothing replacing the file, the call is stored and hits."""
    path = str(data).replace("\\", "/")
    run_cash_cell(
        cash_magics,
        "import os, time\ndef load(p):\n    with open(p) as fh:\n        text = fh.read()\n"
        "    os.write(RUNS_FD, b'x')\n    return text",
    )
    fd = os.open(data.parent / "runs", os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    try:
        cash_magics.shell.user_ns["RUNS_FD"] = fd
        codes = _run(cash_magics, f"text = load({path!r})")
        assert "STORE-INPUT-CHANGED" not in codes
        run_cash_cell(cash_magics, f"text = load({path!r})")
    finally:
        os.close(fd)
    assert cash_magics.shell.user_ns["text"] == "1"
    assert (data.parent / "runs").read_bytes() == b"x", "the second run should be a hit"
