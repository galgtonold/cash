"""A strict traced call works out what each code is once, not per start.

A traced call that reaches a dynamic import (``__import__(name)``) is
strict: every start of a method that binds an attribute of its arguments
(``self._val = ...``) stays armed, so the recorder notes each object it was
bound on (`TraceUnit.attr_params`). lmfit's ``Parameter._getval`` starts
about 5000 times per fit, and each start looked the code up in three weak
sets and dictionaries (a weak reference made for each look) and noted every
object again. The recorder now keeps what a code is (`_start_kinds`),
worked out again only after code joins one of those sets, and notes an
object it already holds for that unit once.
"""

from __future__ import annotations

import importlib
import sys
import textwrap

import pytest

from cash import Cash
from cash.decorator import code_trace

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(sys.version_info < (3, 12), reason="sys.monitoring is Python 3.12+"),
]

_MODULE = """
class P:
    def __init__(self, v):
        self._val = v
        self._expr = None

    def loader(self, name):
        return load(name)

    def _getval(self):
        if self._expr is not None:
            self._val = 0
        return self._val

    @property
    def value(self):
        return self._getval()


def load(name):
    return __import__(name)


def work(ps, n):
    t = 0
    for _ in range(n):
        for p in ps:
            t += p.value
    return t
"""


@pytest.fixture
def mod(tmp_path, monkeypatch):
    name = f"start_kind_mod_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(_MODULE), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module(name)
    yield module
    sys.modules.pop(name, None)


def test_a_strict_call_notes_each_object_and_works_out_each_code_once(mod, tmp_path, monkeypatch):
    recorder = code_trace._recorder()
    if recorder is None:
        pytest.skip(code_trace.recorder_or_reason().reason)
    cls = type(recorder)
    worked_out, units = [], []
    real_kind, real_note = cls._start_kind, cls._note_bound

    def kind(self, code):
        worked_out.append(code)
        return real_kind(self, code)

    def note(self, code, frame, params, is_site):
        units.extend(u for u in self.active if u not in units)
        return real_note(self, code, frame, params, is_site)

    monkeypatch.setattr(cls, "_start_kind", kind)
    monkeypatch.setattr(cls, "_note_bound", note)
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    ran = []

    @c.cache
    def run(ps, n):
        ran.append(n)
        return mod.work(ps, n)

    ps = [mod.P(i) for i in range(10)]
    assert run(ps, 200) == 200 * 45
    getval = mod.P._getval.__code__
    starts_of_getval = sum(1 for code in worked_out if code is getval)
    assert starts_of_getval <= 3, f"worked out again on {starts_of_getval} of 2000 starts"
    held = [u.attr_params[getval]["self"] for u in units if getval in u.attr_params]
    assert held, "the strict call noted no start of the method binding an attribute"
    assert held[0] == {id(p) for p in ps}, "an object the method was bound on was not noted"
    assert run(ps, 200) == 200 * 45
    assert ran == [200], "the call bound only its own argument's attributes, yet was not a hit"
