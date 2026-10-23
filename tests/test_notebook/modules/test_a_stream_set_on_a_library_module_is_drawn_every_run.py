"""A draw from an iterator the notebook set on a library's module runs
every time.

``textwrap.cnt = itertools.count()``, then ``a = next(textwrap.cnt)`` and
``b = next(textwrap.cnt)``: once the value set on ``textwrap`` was folded
into the readers' keys, a run again served ``a`` without drawing, and ``b``,
keyed on the counter where the first draw had not moved it, drew 0 where
plain Python draws 1 (Python 3.12, where the counter pickles). The read of
a stream set on a library's module is a draw, as one from a variable is.
"""

from __future__ import annotations

import pytest

from cash.notebook import callee_reach
from tests._cell_driver import run_cash_cell

#: Iterators that pickle on every Python, and the counter.
STREAMS = {"a_counter": "itertools.count()", "a_list_iterator": "iter(range(100))"}


@pytest.mark.parametrize("stream", list(STREAMS))
def test_each_run_draws_as_plain_python_does(cash_magics, stream, monkeypatch):
    monkeypatch.setattr(callee_reach, "_SET_ON_LIBRARIES", {})
    cells = [
        "import time, itertools\nimport textwrap",
        f"textwrap.cnt = {STREAMS[stream]}",
        "a = (time.sleep(0.05), next(textwrap.cnt))[1]",
        "b = (time.sleep(0.3), next(textwrap.cnt))[1]",
    ]
    ns = cash_magics.shell.user_ns
    seen = []
    try:
        for _ in range(3):
            for cell in cells:
                run_cash_cell(cash_magics, cell, cells=cells)
            seen.append((ns["a"], ns["b"]))
    finally:
        import textwrap

        textwrap.__dict__.pop("cnt", None)
    assert seen == [(0, 1)] * 3


def test_the_draw_is_named(monkeypatch):
    import itertools
    import textwrap

    monkeypatch.setattr(callee_reach, "_SET_ON_LIBRARIES", {"textwrap": {"cnt"}})
    monkeypatch.setattr(textwrap, "cnt", itertools.count(), raising=False)
    ns = {"textwrap": textwrap, "tw": textwrap}
    assert callee_reach.library_streams_read("a = next(textwrap.cnt)", ns) == ["textwrap.cnt"]
    assert callee_reach.library_streams_read("a = next(getattr(tw, 'cnt'))", ns) == ["textwrap.cnt"]
    assert callee_reach.library_streams_read("from textwrap import cnt\na = next(cnt)", ns) == ["textwrap.cnt"]
    assert callee_reach.library_streams_read("a = textwrap.dedent('x')", ns) == []
