"""A cached function that logs is keyed by its code, not by the whole process.

The search for user code held in a global looks through library objects. From
a logger it went on through the handler, the stream, and -- in a kernel --
ipykernel's output thread, session and shell, to the notebook's namespace and
cash's own objects, taking every function defined inside a library function
(``WeakSet.__init__.<locals>._remove``) for the user's. Whatever it met moved
the key: every call of a function that logged missed ("global log changed").
Loggers, handlers, streams, threads and locks hold no code a result depends
on and are not searched; a function or class found inside a library object is
the user's only when its module is.
"""

from __future__ import annotations

import logging
import threading
import types
import weakref

import pytest

from cash import Cash

pytestmark = [pytest.mark.core]

#: Stands for the notebook's namespace, which the kernel's output stream
#: reaches through its session and shell.
NAMESPACE: dict = {}


class _Stream:
    """A log stream like ipykernel's: it holds the machinery around it."""

    def __init__(self):
        self.shell = types.SimpleNamespace(user_ns=NAMESPACE)
        self.watcher = threading.Thread(target=lambda: None)

    def write(self, text):
        pass

    def flush(self):
        pass


log = logging.getLogger("cash-test-logs")
log.handlers[:] = [logging.StreamHandler(_Stream())]
log.propagate = False


def step(n):
    log.warning("step %d", n)
    return n + 1


def test_defining_another_function_does_not_invalidate_a_function_that_logs():
    c = Cash()
    cached = c.cache(step)
    cached(1)
    NAMESPACE["later"] = lambda: 2  # a new cell defines a function
    try:
        cached(1)
    finally:
        NAMESPACE.clear()
    assert [call["cache_hit"] for call in c.drain_decorator_calls()] == [False, True]


def test_a_library_object_s_own_functions_are_not_taken_for_user_code():
    c = Cash()
    held = weakref.WeakSet()  # its ``_remove`` is defined inside ``WeakSet.__init__``
    found = list(c._code_args.iter_code_carriers(types.SimpleNamespace(held=held)))
    assert found == []
