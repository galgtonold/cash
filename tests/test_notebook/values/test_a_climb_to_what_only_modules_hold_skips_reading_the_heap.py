"""A search up the referrers of what a library object holds answers a level
that only modules' globals hold without reading the heap.

A match object holds its compiled pattern, which ``re``'s caches hold too.
Each statement that left a match climbed from the pattern to those caches,
then read the heap again for the caches' own referrers, only to meet the
``re`` module's globals, where the climb ends (60 ms each read, student_7).
Their reference counts already tell that nothing else holds them. The
answer is the same: none for a library's own object, given up on for the
notebook's own object a module keeps, and a variable holding one is still
found by the read of the heap.
"""

from __future__ import annotations

import re
import sys
import types

import pytest

import cash.notebook.shared_objects as shared_objects


@pytest.fixture
def heap_reads(monkeypatch):
    reads = []
    real = shared_objects._referrers

    def counted(objs, stop):
        reads.append(len(objs))
        return real(objs, stop)

    monkeypatch.setattr(shared_objects, "_referrers", counted)
    return reads


@pytest.fixture
def module(monkeypatch):
    made = types.ModuleType("cash_test_registry_module")
    monkeypatch.setitem(sys.modules, made.__name__, made)
    return made


def _climb(inside, user_ns, ignore=()):
    """`_variables_holding` of *inside* (by id, as its caller passes them),
    with what *ignore* holds left out as the caller's own references."""
    value_types = shared_objects.VALUE_TYPES + shared_objects.library_value_types()
    return shared_objects._variables_holding(inside, user_ns, set(), {id(obj) for obj in ignore}, value_types)


def test_a_pattern_re_caches_hold_reads_the_heap_once(heap_reads):
    pattern = re.compile(r"Action_(\d+) \((\d{2})/(\d{2})/(\d{2})\) cash-test")
    match = pattern.match("Action_3 (15/10/19) cash-test")
    assert _climb({id(pattern): pattern}, {"match": match}, ignore=[match]) == set()
    assert heap_reads == [1]


def test_the_notebooks_object_a_module_keeps_is_given_up_on(heap_reads, module):
    rows = [1, 2]
    module.registry = [rows]
    assert _climb({id(rows): rows}, {}) is None
    assert heap_reads == [1]


def test_a_variable_holding_it_too_is_still_found(heap_reads, module):
    rows = [1, 2]
    held = [rows]
    module.registry = held
    assert _climb({id(rows): rows}, {"kept": [held]}) is None
    assert heap_reads == [1, 1]


def test_a_library_object_a_variable_also_holds_is_named(heap_reads, module):
    pattern = re.compile(r"User(\d+) cash-test-two")
    keep = [pattern]
    module.cache = keep
    found = _climb({id(pattern): pattern}, {"patterns": [keep]})
    assert found == {"patterns"}
    assert len(heap_reads) >= 2


def test_counts_are_none_when_anything_else_holds_one():
    namespaces = [vars(module) for module in list(sys.modules.values()) if module is not None]
    held = [re._cache]
    assert shared_objects._module_references([re._cache], namespaces) is None
    del held
    assert shared_objects._module_references([re._cache], namespaces) == [1]
