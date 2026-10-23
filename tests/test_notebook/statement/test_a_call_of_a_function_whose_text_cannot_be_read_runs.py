"""A statement calling a function of the user's whose text cannot be read
runs every time, unless its bytecode shows it changes nothing.

``exec("def inner(x): ... getattr(LOG, 'append')(x) ...")`` defines a function
with no source. The statement's analysis reads a callee's writes off its text
and took one with none for a builtin, so ``v = inner(3)`` was served without
traces (``CASH_TRACE=0``, Python before 3.12) and ``LOG`` missed the item
plain Python appends.
"""

from __future__ import annotations

import pytest

from cash.analysis.annotations import CacheAnnotation
from cash.notebook.cache_status import CacheStatus

_PERSIST = CacheAnnotation(persist=True)


def _run(processor, code):
    return processor.process_statement(code, annotation=_PERSIST)


def _second_run(processor, define):
    _run(processor, "import time\nLOG = []")
    _run(processor, define)
    _run(processor, "v = inner(3)")
    return _run(processor, "v = inner(3)")


@pytest.mark.parametrize(
    "define",
    [
        "exec('def inner(x):\\n    time.sleep(0.01)\\n    getattr(LOG, \"append\")(x)\\n    return x * 10')",
        "inner = eval('lambda x: LOG.append(x) or x * 10')",
        "exec('def helper(x):\\n    LOG.append(x)')\ndef inner(x):\n    helper(x)\n    return x * 10",
    ],
    ids=["exec-def", "lambda-from-a-string", "readable-calling-unreadable"],
)
def test_it_runs_every_time(statement_processor, mock_shell, define):
    second = _second_run(statement_processor, define)
    assert second["status"] != CacheStatus.RESTORED
    assert any("text cannot be read" in r for r in second["uncacheable_reasons"])


def test_a_readable_one_that_changes_nothing_is_served(statement_processor):
    """Control."""
    second = _second_run(statement_processor, "def inner(x):\n    return x * 10")
    assert second["status"] == CacheStatus.RESTORED


def _second_method_run(processor, define, call):
    _run(processor, "import time\nLOG = []")
    _run(processor, define)
    _run(processor, call)
    return _run(processor, call)


EXEC_CLASS = (
    "exec('class K:\\n    def go(self, x):\\n        LOG.append(x)\\n        return x * 10\\n"
    "    @staticmethod\\n    def st(x):\\n        LOG.append(x)\\n        return x\\n"
    "    @classmethod\\n    def cm(cls, x):\\n        LOG.append(x)\\n        return x')\nk = K()"
)


@pytest.mark.parametrize("call", ["v = k.go(3)", "v = K.st(3)", "v = K.cm(3)", "v = k.cm(3)"])
def test_a_method_of_a_class_an_exec_defined_runs_every_time(statement_processor, call):
    """``k.go(3)`` on an object of a class ``exec`` defined: only names bound
    to a function were followed, and the method's write was served away."""
    second = _second_method_run(statement_processor, EXEC_CLASS, call)
    assert second["status"] != CacheStatus.RESTORED
    assert any("text cannot be read" in r for r in second["uncacheable_reasons"])


def test_a_method_of_a_readable_class_is_served(statement_processor):
    """Control: a method whose text is read is the statement's analysis's."""
    second = _second_method_run(
        statement_processor, "class K:\n    def go(self, x):\n        return x * 10\nk = K()", "v = k.go(3)"
    )
    assert second["status"] == CacheStatus.RESTORED
