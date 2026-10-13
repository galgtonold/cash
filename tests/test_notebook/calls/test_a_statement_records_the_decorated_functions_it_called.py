"""A notebook statement's entry names the ``@cash.cache`` functions it called.

CACHE-NOT-WORTH-BYTES told a statement over a decorated call (p17's
``df = crlib.build_features(...)``, p27's ``X_train = F.design_matrix(...)``)
to put "`@cash.cache` on the function", which it already had. The store had no
way to know: the statement's metadata did not say which decorated functions it
called. It does now (``decorated_calls``), and a function cash wrapped itself
(an intercepted call) is not one of them, so its statement keeps the advice.
"""

from __future__ import annotations

import os
import sys

import pytest

from cash._active import default_cash, set_default_cash
from cash.backends import InMemoryBackend
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

LIB = (
    "import cash\n"
    "import time\n"
    "@cash.cache\n"
    "def make(n):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return list(range(n))\n"
    "def plain(n):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return list(range(n))\n"
)


@pytest.fixture
def shared(cash_magics, cash_instance, mock_shell, tmp_path, monkeypatch):
    """The notebook and the decorator over one backend, with a module ``lib``."""
    ram = InMemoryBackend()
    cash_instance.backend = ram
    previous = default_cash()
    set_default_cash(cash_instance)
    name = f"_decorated_calls_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(LIB, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, f"import {name} as lib")
    yield cash_magics, ram
    set_default_cash(previous)
    sys.modules.pop(name, None)


def _statement(ram, code):
    found = [m for m in ram.list_entries() or () if m.get("code") == code]
    assert len(found) == 1, [m.get("code") for m in ram.list_entries() or ()]
    return found[0]


def test_a_statement_over_a_decorated_call_names_it(shared):
    magics, ram = shared
    run_cash_cell(magics, "b = lib.make(1000) + [1]")
    assert [n.rsplit(".", 1)[-1] for n in _statement(ram, "b = lib.make(1000) + [1]")["decorated_calls"]] == ["make"]


def test_a_statement_over_an_intercepted_call_names_none(shared):
    """The statement does work of its own (the sleep), so it keeps an entry
    beside the intercepted call's."""
    magics, ram = shared
    code = f"c = lib.plain(1000) + [time.sleep({ABOVE_PERSISTENCE_FLOOR_S})]"
    run_cash_cell(magics, "import time")
    run_cash_cell(magics, code)
    assert not _statement(ram, code).get("decorated_calls")
