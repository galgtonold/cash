"""A call that changes an argument in place is not stored when that
argument shares memory with another argument or with a value the function
reads outside its arguments.

The key hashes each argument by its content alone, so ``f(a, a[::-1])``
keys like ``f(a, b)`` with ``b`` an equal, independent array, though the
body sees ``b`` change as it writes ``a``. A stored result then came back
for the independent pair, and a hit writing ``a`` back left the view as no
plain run does. The same holds for a global the function reads that is,
or shares memory with, the argument it changes.
"""

from __future__ import annotations

import warnings

import numpy as np


def _view_sum(a, b):
    a += 1
    return float(b.sum())


def _sorted_head(a, b):
    a.sort()
    return float(b[0])


GLOBAL = np.array([3.0, 1.0, 2.0])


def _reads_global(a):
    a.sort()
    return float(GLOBAL[0])


def _quiet(f, *args):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*args)


def test_arguments_sharing_memory_give_what_plain_runs_give(disk_cash):
    """An independent pair first, then the shared ones; each call checked
    against the plain function on fresh data."""
    cases = [
        (_view_sum, lambda: (lambda a: (a, a[::-1]))(np.arange(3.0))),
        (_view_sum, lambda: (lambda a: (a, a[::-1].copy()))(np.arange(3.0))),
        (_sorted_head, lambda: (lambda a: (a, a[:2]))(np.array([3.0, 1.0, 2.0]))),
        (_sorted_head, lambda: (lambda a: (a, a[:2].copy()))(np.array([3.0, 1.0, 2.0]))),
    ]
    cached = {body: disk_cash.cache(body) for body in (_view_sum, _sorted_head)}
    for body, make in cases * 2:
        plain_args = make()
        plain = body(*plain_args)
        args = make()
        got = _quiet(cached[body], *args)
        assert got == plain, (body.__name__, got, plain)
        assert [x.tolist() for x in args] == [x.tolist() for x in plain_args]


def test_a_global_that_is_the_changed_argument_is_read_as_a_plain_run_reads_it(disk_cash):
    """A copy of the global first, stored; then the global itself, which
    keys alike."""
    cached = disk_cash.cache(_reads_global)
    seen, plain = [], []
    for use_global in (False, True, True):
        for out, f in ((plain, _reads_global), (seen, lambda a: _quiet(cached, a))):
            GLOBAL[:] = [3.0, 1.0, 2.0]
            out.append(f(GLOBAL if use_global else GLOBAL.copy()))
    GLOBAL[:] = [3.0, 1.0, 2.0]
    assert plain == [3.0, 1.0, 1.0]
    assert seen == plain


def test_the_warning_says_the_call_runs_every_time(disk_cash):
    """Not the line saying a hit writes the argument back: no hit does."""
    cached = disk_cash.cache(_sorted_head)
    a = np.array([3.0, 1.0, 2.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cached(a, a[:2])
    text = " ".join(str(w.message) for w in caught)
    assert "runs every time" in text, text
    assert "a hit writes it back" not in text, text
