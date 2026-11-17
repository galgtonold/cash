"""Where a module holds the class a method names as its owner is looked up
in every loaded module once, not once per method, while no module is loaded.

A purity report is stored under the path that leads back to its function.
For a method no other name leads to (a class a package re-exports under
another ``__module__``, scanpy's ``Preset`` and its properties), that path
is found by asking every loaded module for the class. A cached call over
such a package asked it for each method in every process, hits included:
a few thousand namespaces a time, a tenth of the cost of a hit. The answer
is kept until another module is loaded; each module it names is checked
again when asked, so a module that stopped holding the class is not
answered, and one loaded since is found.
"""

from __future__ import annotations

import sys
import types

import pytest

from cash.analysis import report_store


class _CountedModule(types.ModuleType):
    """A module that counts how often its namespace is read."""

    reads = 0

    @property
    def __dict__(self):  # type: ignore[override]
        type(self).reads += 1
        return super().__dict__


def _class_in(module_name: str, class_name: str) -> type:
    cls = type(class_name, (), {"a": lambda self: 1, "b": lambda self: 2})
    cls.__module__ = module_name
    cls.a.__qualname__ = f"{class_name}.a"
    cls.b.__qualname__ = f"{class_name}.b"
    return cls


@pytest.fixture
def counted():
    name = "_cash_test_counted_namespace"
    module = _CountedModule(name)
    sys.modules[name] = module
    _CountedModule.reads = 0
    yield module
    sys.modules.pop(name, None)


def test_the_methods_of_one_class_scan_the_loaded_modules_once(counted):
    holder = types.ModuleType("_cash_test_holder_a")
    cls = _class_in(holder.__name__, "HeldOnceA")
    holder.HeldOnceA = cls
    sys.modules[holder.__name__] = holder
    try:
        first = report_store._holder_homes(cls.a)
        reads = _CountedModule.reads
        second = report_store._holder_homes(cls.b)
        assert first == [(holder.__name__, ("HeldOnceA", "a"))]
        assert second == [(holder.__name__, ("HeldOnceA", "b"))]
        assert reads >= 1
        # The second method of the class did not read every module again.
        assert _CountedModule.reads == reads
    finally:
        sys.modules.pop(holder.__name__, None)


def test_a_module_loaded_since_is_found(counted):
    cls = _class_in("_cash_test_holder_b", "HeldLaterB")
    assert report_store._holder_homes(cls.a) == []
    holder = types.ModuleType("_cash_test_holder_b")
    holder.HeldLaterB = cls
    sys.modules[holder.__name__] = holder
    try:
        assert report_store._holder_homes(cls.b) == [(holder.__name__, ("HeldLaterB", "b"))]
    finally:
        sys.modules.pop(holder.__name__, None)


def test_a_module_that_no_longer_holds_the_class_is_not_answered(counted):
    holder = types.ModuleType("_cash_test_holder_c")
    cls = _class_in(holder.__name__, "HeldOnceC")
    holder.HeldOnceC = cls
    sys.modules[holder.__name__] = holder
    try:
        assert report_store._holder_homes(cls.a) == [(holder.__name__, ("HeldOnceC", "a"))]
        del holder.HeldOnceC
        assert report_store._holder_homes(cls.b) == []
        holder.HeldOnceC = int  # not a class defined there
        assert report_store._holder_homes(cls.b) == []
    finally:
        sys.modules.pop(holder.__name__, None)
