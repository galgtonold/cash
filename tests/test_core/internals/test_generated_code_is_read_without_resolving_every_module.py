"""Code compiled from no file -- a dataclass's generated ``__init__``,
``exec`` source -- is answered "no source" without ``inspect.getmodule``.

``inspect`` looks for such code's module by resolving the file of every
loaded module (a ``realpath`` each): with pandas, scikit-learn and scipy
loaded, 1,700 modules and 20,000 ``stat`` calls, paid on the first key of
every script run that hands a cached function a dataclass. The answer it
reaches is fixed by the name alone: a name in angle brackets that
``linecache`` does not hold and no file has has no source. What must hold:
the same answers as ``inspect`` (no source, no file to memoise on), the
lines ``inspect`` gives where ``linecache`` does hold the name, and no
``getmodule`` call for the first.
"""

from __future__ import annotations

import dataclasses
import inspect
import linecache

import pytest

from cash import source_reading


@dataclasses.dataclass(frozen=True)
class _Config:
    alpha: float = 0.1


def _inspect_answer(code):
    try:
        return inspect.getsourcelines(code)
    except Exception as exc:  # whatever inspect raises, raised alike
        return type(exc)


def _cash_answer(code):
    try:
        return source_reading.getsourcelines(code)
    except Exception as exc:  # whatever inspect raises, raised alike
        return type(exc)


@pytest.fixture
def getmodule_calls(monkeypatch):
    calls: list = []
    real = inspect.getmodule

    def spy(obj, _filename=None):
        calls.append(obj)
        return real(obj, _filename)

    monkeypatch.setattr(inspect, "getmodule", spy)
    return calls


def test_a_dataclass_s_generated_code_has_no_source_and_no_module_lookup(getmodule_calls):
    code = _Config.__init__.__code__
    assert code.co_filename.startswith("<"), "control: generated code names no file"
    expected = _inspect_answer(code)
    assert expected is OSError, "control: inspect finds no source for it"
    getmodule_calls.clear()

    assert source_reading.settled_source_version(code) is None
    assert _cash_answer(code) is OSError
    assert getmodule_calls == []


def test_exec_source_linecache_holds_is_read_as_inspect_reads_it(getmodule_calls):
    name = "<cash-test-held-source>"
    text = "def held(x):\n    return x + 1\n"
    namespace: dict = {}
    exec(compile(text, name, "exec"), namespace)
    code = namespace["held"].__code__
    linecache.cache[name] = (len(text), None, text.splitlines(True), name)
    try:
        expected = inspect.getsourcelines(code)
        assert expected[0] == text.splitlines(True), "control: inspect reads the held lines"
        assert source_reading.getsourcelines(code) == expected
    finally:
        linecache.cache.pop(name, None)


def test_exec_source_nobody_holds_has_no_source_as_inspect_says(getmodule_calls):
    namespace: dict = {}
    exec(compile("def gone(x):\n    return x\n", "<cash-test-gone>", "exec"), namespace)
    code = namespace["gone"].__code__
    assert _inspect_answer(code) is OSError
    getmodule_calls.clear()
    assert _cash_answer(code) is OSError
    assert getmodule_calls == []
