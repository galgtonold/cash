"""Objects implemented in C are never the user's code.

A C type or callable whose ``__module__`` cannot be followed fell into the
fallback meant for exec'd and notebook code, and counted as user code whose
code could not be hashed. ``sys.stdout.write`` has no ``__module__``,
``_thread.lock`` is not reachable as ``_thread.lock``, and up to Python 3.11
``_io`` calls itself ``io``. So a cached function reading a module-level
logger whose handler holds a queue (and its ``threading.Condition``) warned
KEY-OPAQUE-CALLABLE about ``lock.acquire`` and ``lock.release``; in a Jupyter
kernel ``logging.basicConfig()`` alone was enough.
"""

from __future__ import annotations

import _io
import importlib.machinery
import io
import logging
import logging.handlers
import queue
import re
import sys
import threading
import types
import warnings

import pytest

from cash import Cash, FileBackend
from cash.decorator.code_identity import is_user_code_module, is_user_code_object

log = logging.getLogger("cash-test.c-objects.work")


def _step(n):
    log.info("step %d", n)
    return n + 1


class _OpaqueCallable:
    """Control: code of the user's whose behaviour has no Python code."""

    __call__ = staticmethod(abs)


def _takes(fn):
    return 1


def _opaque_warnings(caught):
    return [str(w.message) for w in caught if "KEY-OPAQUE-CALLABLE" in str(w.message)]


@pytest.fixture
def queue_handler():
    handler = logging.handlers.QueueHandler(queue.Queue())
    log.addHandler(handler)
    yield handler
    log.removeHandler(handler)


def test_a_logger_with_a_queue_handler_read_by_a_cached_function_does_not_warn(tmp_path, queue_handler):
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))
    step = c.cache(_step)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert step(1) == 2
        assert step(1) == 2

    assert _opaque_warnings(caught) == []
    assert step.cache_info()["hits"] == 1


def test_control_user_code_that_cannot_be_hashed_still_warns(tmp_path):
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))
    takes = c.cache(_takes)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        takes(_OpaqueCallable())

    assert len(_opaque_warnings(caught)) == 1, _opaque_warnings(caught)


def test_a_built_in_module_is_judged_by_its_spec_not_its_name():
    # Up to Python 3.11 _io names itself "io", which a name check would take for
    # a module with no file of its own; 3.12 and later name it "_io".
    assert is_user_code_module(sys.modules["_io"]) is False

    # The same case on every version: a module under a name no built-in has.
    renamed = types.ModuleType("_cash_test_renamed_builtin")
    assert is_user_code_module(renamed), "control: a module with no file and no spec is a notebook's"
    for origin in ("built-in", "frozen"):
        renamed.__spec__ = importlib.machinery.ModuleSpec(renamed.__name__, None, origin=origin)
        assert is_user_code_module(renamed) is False, origin


_LOCK = threading.Lock()


@pytest.mark.parametrize(
    "obj",
    [
        _io.TextIOWrapper,
        io.StringIO,
        type(_LOCK),
        _LOCK.acquire,
        io.StringIO().write,
        re.compile("a").search,
        str.join,
        object().__str__,
        object.__init__,
    ],
    ids=[
        "TextIOWrapper",
        "StringIO",
        "lock",
        "lock.acquire",
        "StringIO.write",
        "Pattern.search",
        "str.join",
        "method-wrapper",
        "slot-wrapper",
    ],
)
def test_c_types_and_callables_are_not_user_code(obj):
    assert is_user_code_object(obj) is False


def test_code_the_fallback_is_for_still_counts():
    """Exec'd and notebook code still counts, and so does a Python method of
    a user's subclass of a C type -- but not the C method it inherits."""
    ns: dict = {}
    exec("class A:\n    def f(self):\n        pass\ndef g():\n    pass\n", ns)
    assert is_user_code_object(ns["A"]) and is_user_code_object(ns["g"])
    assert is_user_code_object(ns["A"]().f)

    mod = types.ModuleType("_cash_test_c_objects_nb")
    exec("import io\nclass Buffer(io.StringIO):\n    def extra(self):\n        pass\n", mod.__dict__)
    sys.modules[mod.__name__] = mod
    try:
        buffer = mod.Buffer()
        assert is_user_code_object(mod.Buffer)
        assert is_user_code_object(buffer.extra)
        assert is_user_code_object(buffer.write) is False
    finally:
        del sys.modules[mod.__name__]
