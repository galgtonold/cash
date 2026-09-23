"""A statement's entry refers to call results already in the cache.

Every project's cache held its expensive results twice -- the dict a
statement gathered from per-call cached fits, and each fit under its call's key
(~1.2 of 2.7 GiB in one project, ~1.3 of 4 GiB in another). The statement's entry still
restores the value on its own (a later cell or a restart needs no call
arguments rebuilt); it just points at the call entries for the parts they hold.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from traitlets.config import Configurable

from cash.backends import InMemoryBackend
from cash.core import Cash
from cash.notebook.call_refs import CallRef
from cash.notebook.ipython.magics import CashMagics


class _Shell(Configurable):
    def __init__(self):
        super().__init__()
        self.user_ns = {"__name__": "__main__"}
        self.input_transformers_cleanup = []
        self.run_cell = MagicMock()
        self.events = MagicMock()
        self.ast_transformers = []
        self.user_global_ns = self.user_ns
        self.display_pub = type("Pub", (), {"publish": MagicMock()})()


@pytest.fixture
def nb():
    backend = InMemoryBackend()
    shell = _Shell()
    magics = CashMagics(shell, Cash(backend=backend, register_magic=False))
    magics._auto_cache_enabled = True
    magics.cash("", "import time\ndef fit(k):\n    time.sleep(0.12)\n    return list(range(k * 1000))")
    yield magics, shell, backend
    backend.clear()


def _stored(backend, needle):
    for meta in backend.list_entries() or ():
        if needle in str(meta.get("code", "")):
            return backend.get(meta["key"])[1]["variables"]
    raise AssertionError(f"no entry for {needle!r}")


def _statuses(magics, shell, code, name):
    shell.user_ns.pop(name, None)
    magics.cash("", code)
    return shell.user_ns[name]


def test_a_dict_of_call_results_is_stored_as_references(nb):
    magics, shell, backend = nb
    code = "models = {k: fit(k) for k in range(1, 4)}"
    magics.cash("", code)
    stored = _stored(backend, code)["models"]
    assert all(isinstance(v, CallRef) for v in stored.values()), stored

    shell.user_ns.pop("models")
    magics.cash("", code)  # restored: the references are read back
    assert shell.user_ns["models"] == {k: list(range(k * 1000)) for k in range(1, 4)}


def test_a_result_changed_after_the_call_is_stored_by_value(nb):
    magics, shell, backend = nb
    code = "m = fit(2).__iadd__([-1])"
    magics.cash("", code)
    stored = _stored(backend, code)["m"]
    assert not isinstance(stored, CallRef) and stored[-1] == -1, type(stored)
    shell.user_ns.pop("m")
    magics.cash("", code)
    assert shell.user_ns["m"][-1] == -1 and len(shell.user_ns["m"]) == 2001


def test_a_statement_whose_call_entry_is_gone_recomputes(nb):
    magics, shell, backend = nb
    code = "models = {k: fit(k) for k in range(1, 3)}"
    magics.cash("", code)
    for meta in list(backend.list_entries() or ()):
        if str(meta.get("key", "")).startswith("call:"):
            backend.delete(meta["key"])
    shell.user_ns.pop("models")
    magics.cash("", code)
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
