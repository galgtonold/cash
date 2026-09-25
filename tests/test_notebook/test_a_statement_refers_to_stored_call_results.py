"""A statement's entry refers to call results already in the cache.

Every project's cache held its expensive results twice -- the dict a
statement gathered from per-call cached fits, and each fit under its call's key
(~1.2 of 2.7 GiB in one project, ~1.3 of 4 GiB in another). The statement's entry still
restores the value on its own (a later cell or a restart needs no call
arguments rebuilt); it just points at the call entries for the parts they hold.
"""

from __future__ import annotations

import pytest

from cash.backends import InMemoryBackend
from cash.notebook.call_refs import CallRef
from tests._cell_driver import run_cash_cell


@pytest.fixture
def nb(cash_magics, mock_shell, clean_backend):
    # The notebook's module is ``__main__``, as in a kernel.
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, "import time\ndef fit(k):\n    time.sleep(0.12)\n    return list(range(k * 1000))")
    return cash_magics, mock_shell, clean_backend


def _stored(backend, needle):
    for meta in backend.list_entries() or ():
        if needle in str(meta.get("code", "")):
            return backend.get(meta["key"])[1]["variables"]
    raise AssertionError(f"no entry for {needle!r}")


def _statuses(magics, shell, code, name):
    shell.user_ns.pop(name, None)
    run_cash_cell(magics, code)
    return shell.user_ns[name]


def test_a_dict_of_call_results_is_stored_as_references(nb):
    magics, shell, backend = nb
    code = "models = {k: fit(k) for k in range(1, 4)}"
    run_cash_cell(magics, code)
    stored = _stored(backend, code)["models"]
    assert all(isinstance(v, CallRef) for v in stored.values()), stored

    shell.user_ns.pop("models")
    run_cash_cell(magics, code)  # restored: the references are read back
    assert shell.user_ns["models"] == {k: list(range(k * 1000)) for k in range(1, 4)}


def test_a_result_changed_after_the_call_is_stored_by_value(nb):
    """The ``__iadd__`` sleeps so the statement is stored for its own work
    (a statement whose own work is cheap keeps no value beside its call's)."""
    magics, shell, backend = nb
    code = "m = fit(2).__iadd__([time.sleep(0.12) or -1])"
    run_cash_cell(magics, code)
    stored = _stored(backend, code)["m"]
    assert not isinstance(stored, CallRef) and stored[-1] == -1, type(stored)
    shell.user_ns.pop("m")
    run_cash_cell(magics, code)
    assert shell.user_ns["m"][-1] == -1 and len(shell.user_ns["m"]) == 2001
    run_cash_cell(magics, "f = fit(2)")
    assert shell.user_ns["f"] == list(range(2000)), "the change reached the call's entry"


def test_a_method_changing_a_call_result_in_place_leaves_its_entry_alone(nb):
    """``fit(2)`` inside ``fit(2).__iadd__(...)`` is a call unit; the method
    runs on what the call returned, and the entry must not see the change."""
    magics, shell, backend = nb
    code = "m = fit(2).__iadd__([-1])"
    for _ in range(3):
        shell.user_ns.pop("m", None)
        run_cash_cell(magics, code)
        assert shell.user_ns["m"][-1] == -1 and len(shell.user_ns["m"]) == 2001
    run_cash_cell(magics, "f = fit(2)")
    assert shell.user_ns["f"] == list(range(2000))


def test_a_statement_whose_call_entry_is_gone_recomputes(nb):
    magics, shell, backend = nb
    code = "models = {k: fit(k) for k in range(1, 3)}"
    run_cash_cell(magics, code)
    for meta in list(backend.list_entries() or ()):
        if str(meta.get("key", "")).startswith("call:"):
            backend.delete(meta["key"])
    shell.user_ns.pop("models")
    run_cash_cell(magics, code)
    assert shell.user_ns["models"] == {k: list(range(k * 1000)) for k in range(1, 3)}


def test_a_reference_to_an_entry_holding_something_else_is_a_miss():
    from cash.notebook.call_refs import resolve_call_refs

    backend = InMemoryBackend()
    backend.set("call:a", [1, 2], {"value_digest": "d1"})
    payload = {"variables": {"models": {1: CallRef("call:a", "d1")}}, "stdout": ""}
    assert resolve_call_refs(payload, backend)["variables"] == {"models": {1: [1, 2]}}
    backend.set("call:a", ["other"], {"value_digest": "d2"})
    assert resolve_call_refs(payload, backend) is None
    backend.delete("call:a")
    assert resolve_call_refs(payload, backend) is None
