"""Which classes of the user's a ``with`` item may enter and exit
(`handed_callables.manager_classes`): what a factory returns, read through
its parameters, locals, defaults, a classmethod's ``cls``, a container, a
conditional and a ``@contextmanager`` body. What a library hands back adds
nothing, and every manager class of the user's counts only where the item
may give an object of the user's that cannot be read further.
"""

from __future__ import annotations

import ast
import importlib
import sys
import textwrap

import pytest

from cash.analysis.handed_callables import manager_classes

SETUP = textwrap.dedent(
    """
    import contextlib, threading
    timings = []
    hits = []

    class Mgr:
        def __enter__(self):
            hits.append(1)
            return self
        def __exit__(self, *a):
            return False
        @classmethod
        def open(cls):
            return cls()

    class AMgr:
        async def __aenter__(self):
            hits.append(1)
        async def __aexit__(self, *a):
            return False

    class Timer:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            timings.append(1)

    shared = Mgr()
    KINDS = {'m': Mgr}
    LOCKS = {'a': threading.Lock()}

    def factory():
        return Mgr()
    def local():
        m = Mgr()
        return m
    def glob():
        return shared
    def inner():
        return Mgr()
    def two_deep():
        return inner()
    def handed(kind):
        return kind()
    def default(kind=Mgr):
        return kind()
    def alias():
        make = Mgr
        return make()
    def registry(name):
        return KINDS[name]()
    def locked(name):
        return LOCKS[name]
    def ahanded(kind):
        return kind()
    def nested():
        def other():
            return Timer()
        return Mgr()
    def star(*args):
        return args[0]()

    @contextlib.contextmanager
    def wrapping():
        with Mgr() as m:
            yield m
    """
)

#: ``with`` item -> the classes it may enter.
CASES = {
    "factory()": {"Mgr"},
    "local()": {"Mgr"},
    "glob()": {"Mgr"},
    "two_deep()": {"Mgr"},
    "handed(Mgr)": {"Mgr"},
    "handed(kind=Mgr)": {"Mgr"},
    "default()": {"Mgr"},
    "alias()": {"Mgr"},
    "Mgr.open()": {"Mgr"},
    "registry('m')": {"Mgr"},
    "ahanded(AMgr)": {"AMgr"},
    "nested()": {"Mgr"},
    "wrapping()": {"Mgr"},
    "factory() if hits else contextlib.nullcontext()": {"Mgr"},
    # A library object handed back, and a library call: nothing of the user's.
    "locked('a')": set(),
    "contextlib.nullcontext()": set(),
    "threading.Lock()": set(),
    # What a call is handed that cannot be read: any manager of the user's.
    "star(*[Mgr])": {"Mgr", "AMgr", "Timer"},
}


@pytest.fixture(scope="module")
def namespace(tmp_path_factory):
    folder = tmp_path_factory.mktemp("managers")
    (folder / "managers_probe_mod.py").write_text(SETUP)
    sys.path.insert(0, str(folder))
    try:
        module = importlib.import_module("managers_probe_mod")
        yield vars(module)
    finally:
        sys.path.remove(str(folder))
        sys.modules.pop("managers_probe_mod", None)


@pytest.mark.parametrize("item", list(CASES))
def test_the_classes_an_item_may_enter(namespace, item):
    expr = ast.parse(item, mode="eval").body
    assert {cls.__name__ for cls in manager_classes(expr, namespace)} == CASES[item]
