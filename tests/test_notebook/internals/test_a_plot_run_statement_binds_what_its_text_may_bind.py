"""`plot_runs._statement_binds`: what a statement of a plot run may bind or
unbind, read from its text.

A served plot run skips its statements, so one that may bind or unbind a
name is never kept. The rows a statement reports do not always say: a
loop's passes name no loop variable, and a branch not taken binds nothing
on this run but may on the next. The text says what any run may bind.
"""

from __future__ import annotations

import pytest


def _statement_binds(code):
    from cash.notebook.ipython.plot_runs import _statement_binds as binds

    return binds(code)


CASES = {
    "plain call": ("plt.plot(x)", set()),
    "assignment": ("fig, ax = plt.subplots()", {"fig", "ax"}),
    "underscore": ("_ = plt.plot(x)", {"_"}),
    "augmented": ("n += 1", {"n"}),
    "loop": ("for i in range(3):\n    plt.plot([i])", {"i"}),
    "branch": ("if flag:\n    lines = plt.plot(x)", {"lines"}),
    "del": ("del tmp", {"tmp"}),
    "with": ("with ctx() as fh:\n    plt.plot(x)", {"fh"}),
    "except": ("try:\n    plt.plot(x)\nexcept Exception as e:\n    pass", {"e"}),
    "walrus": ("plt.plot(y) if (y := x) else None", {"y"}),
    "walrus in comprehension": ("plt.plot([last := v for v in x])", {"last"}),
    "comprehension variable": ("plt.plot([v * 2 for v in x])", set()),
    "lambda": ("plt.plot(list(map(lambda v: v, x)))", set()),
    "def": ("def helper(v):\n    w = v\n    return w", {"helper"}),
    "import": ("import json.tool", {"json"}),
    "import as": ("import numpy as np", {"np"}),
    "star import": ("from pylab import *", {"*"}),
    "match": ("match x:\n    case [a, *rest]:\n        pass", {"a", "rest"}),
}


@pytest.mark.parametrize("code,binds", list(CASES.values()), ids=list(CASES))
def test_what_a_statement_may_bind(code, binds):
    assert _statement_binds(code) == binds


def test_a_statement_that_cannot_be_read_is_none():
    assert _statement_binds("plt.plot(") is None
