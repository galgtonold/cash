"""A search up the referrers of what a library object holds stops without
reading the heap once a module is found holding what it climbs through.

E18's seaborn ``FacetGrid`` holds ~900 objects pyplot's figure registry and
the like hold too. Their referrers took one read of the heap; the second
level, another (~1 s), only to meet a module's globals and give up, as it
does for an object a module keeps. That level is now answered by reading
the modules' globals for the objects first: the same answer, with no read
of the heap when a module holds one.
"""

from __future__ import annotations

import sys
import types

import pytest

import cash.notebook.shared_objects as shared_objects

COUNT = 300  # past `_REFERRERS_BY_CALL`: a level read off the heap


@pytest.fixture
def module(monkeypatch):
    made = types.ModuleType("cash_test_holding_module")
    monkeypatch.setitem(sys.modules, made.__name__, made)
    return made


def _climb(inside, user_ns, monkeypatch, *, shortcut):
    reads = []
    real = shared_objects._referrers_of
    monkeypatch.setattr(shared_objects, "_referrers_of", lambda objs, stop: reads.append(len(objs)) or real(objs, stop))
    if not shortcut:
        monkeypatch.setattr(shared_objects, "_held_by_a_module", lambda objs, namespaces: False)
    value_types = shared_objects.VALUE_TYPES + shared_objects.library_value_types()
    found = shared_objects._variables_holding(
        {id(obj): obj for obj in inside}, user_ns, set(), {id(inside)}, value_types
    )
    monkeypatch.undo()
    return found, reads


@pytest.mark.parametrize("shortcut", [True, False])
def test_what_a_module_holds_past_the_first_level_is_given_up_on(module, monkeypatch, shortcut):
    inside = [[n] for n in range(COUNT)]
    holders = [[obj] for obj in inside]
    for n, holder in enumerate(holders):
        setattr(module, f"holder{n}", holder)
    found, reads = _climb(inside, {}, monkeypatch, shortcut=shortcut)
    assert found is None
    assert reads == ([COUNT] if shortcut else [COUNT, COUNT])


@pytest.mark.parametrize("shortcut", [True, False])
def test_a_variable_holding_them_is_found_as_before(monkeypatch, shortcut):
    inside = [[n] for n in range(COUNT)]
    holders = [[obj] for obj in inside]
    found, _reads = _climb(inside, {"rows": holders}, monkeypatch, shortcut=shortcut)
    assert found == {"rows"}
