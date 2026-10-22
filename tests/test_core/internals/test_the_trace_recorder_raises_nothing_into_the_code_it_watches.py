"""The trace recorder raises nothing into the code it watches.

An exception in a ``sys.monitoring`` callback is raised in the frame that
started, as if the user's line had raised it. ``isinstance(self, type)`` on
an object whose ``__getattribute__`` raises for ``__class__`` raised
KeyError from ``Cls.__init__``: a cached function building one failed where
plain Python runs it. Now no callback lets an exception out, and a trace it
cut short is not checked against (``trace_failed``).
"""

from __future__ import annotations

import sys

import pytest

from cash import Cash

pytestmark = pytest.mark.skipif(sys.version_info < (3, 12), reason="traces are recorded on Python 3.12+")

MODULE = """
class Cls:
    def __init__(self, v):
        object.__setattr__(self, 'v', v)
    def __getattribute__(self, k):
        if k == '__class__':
            raise KeyError(k)
        return object.__getattribute__(self, k)
def build(n):
    return Cls(n).v + 1
"""


@pytest.fixture
def hostile(tmp_path, monkeypatch):
    (tmp_path / "hostile_class.py").write_text(MODULE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "hostile_class", raising=False)
    import hostile_class

    return hostile_class


def test_a_class_hiding_its_class_runs_under_the_trace(tmp_path, hostile):
    build = Cash(cache_dir=str(tmp_path / "cache"), register_magic=False).cache(hostile.build)
    assert build(4) == build(4) == 5


def test_a_callback_that_raises_marks_the_trace_unusable(tmp_path, hostile, monkeypatch):
    from cash.decorator import code_trace

    def broken(self, *args):
        raise RuntimeError("cash's own bug")

    monkeypatch.setattr(code_trace._recorder_class(), "_seen_start", broken)
    build = Cash(cache_dir=str(tmp_path / "cache"), register_magic=False).cache(hostile.build)
    assert build(4) == 5
