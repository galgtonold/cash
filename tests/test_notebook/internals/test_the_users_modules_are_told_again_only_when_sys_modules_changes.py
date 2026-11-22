"""Which loaded modules are the user's is told again only when
``sys.modules`` holds other module objects than when last asked.

Every statement asked for the classes and functions of the user's modules
up to 9 times, and each time went over every loaded module (some 1,400 in
a kernel): 2-3 ms each, most of what a statement like ``k = len(records)``
cost beside plain Python's 0.002 s.
"""

from __future__ import annotations

import sys
import types

import pytest

from cash.notebook import held_functions


def _users(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    exec(f"class Order:\n    __module__ = {name!r}\n", vars(module))
    return module


@pytest.fixture
def asked(monkeypatch):
    """The modules `_users_module` is asked about; a module counts as the
    user's when its name says so."""
    names: list[str] = []
    real = held_functions._users_module

    def counting(module):
        names.append(getattr(module, "__name__", None))
        return real(module)

    monkeypatch.setattr(held_functions, "_users_module", counting)
    # Verdicts given under the stand-in below are not kept past the test.
    monkeypatch.setattr(held_functions, "_USERS_MODULE", {})
    monkeypatch.setattr(held_functions, "_USERS_MODULES", [[], []])
    monkeypatch.setattr(held_functions, "is_local_module", lambda module: module.__name__.startswith("_users_mod"))
    return names


def test_the_modules_are_not_gone_over_again_while_none_is_loaded_or_replaced(monkeypatch, asked):
    monkeypatch.setitem(sys.modules, "_users_mod_a", _users("_users_mod_a"))
    first = held_functions.users_module_classes()
    assert asked
    asked.clear()
    assert held_functions.users_module_classes() == first
    assert asked == []


def test_a_module_loaded_since_is_read(monkeypatch, asked):
    held_functions.users_module_classes()
    module = _users("_users_mod_b")
    monkeypatch.setitem(sys.modules, "_users_mod_b", module)
    assert module.Order in held_functions.users_module_classes()


def test_a_module_replaced_under_its_name_is_read(monkeypatch, asked):
    monkeypatch.setitem(sys.modules, "_users_mod_c", types.ModuleType("_users_mod_c"))
    monkeypatch.setitem(sys.modules, "_users_mod_c_other", types.ModuleType("not_the_users"))
    assert held_functions.users_module_classes() == []
    module = _users("_users_mod_c_other")
    # Same count, one object in another's place.
    monkeypatch.setitem(sys.modules, "_users_mod_c_other", module)
    assert module.Order in held_functions.users_module_classes()
