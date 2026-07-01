"""Hidden mutation through a custom dunder method must reset on isolated re-run
(CAS-70). An operation that implicitly invokes a user-defined
``__setitem__`` / ``__delitem__`` / ``__getitem__`` / ``__call__`` whose body
mutates hidden state (a module free variable or ``self``) accumulates on an
isolated re-run, because cash treats the operation as a builtin-container op or
pure and never attributes the mutation.

Each operation runs ONCE, so the correct value is identical on the first run and
every re-run.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream]


def _rerun(nb_runner, setup, cell, expect):
    nb_runner.create_notebook([setup, cell])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert expect in nb_runner.get_output(2), f"first: {nb_runner.get_output(2)!r}"
    nb_runner.run_cell(2)
    assert expect in nb_runner.get_output(2), f"re-run: {nb_runner.get_output(2)!r}"


# --- must reset ---------------------------------------------------------------

def test_setitem_free_var(nb_runner):
    _rerun(nb_runner,
           "log = []\nclass Store:\n    def __setitem__(self, k, v):\n        log.append((k, v))\ns = Store()",
           "s['k'] = 1\nprint('log', len(log))", "log 1")


def test_delitem_free_var(nb_runner):
    _rerun(nb_runner,
           "log = []\nclass Store:\n    def __delitem__(self, k):\n        log.append(k)\ns = Store()",
           "del s['a']\nprint('log', len(log))", "log 1")


def test_getitem_mutates_self(nb_runner):
    # a subscript READ that mutates self.hits (anti-pattern, but must be correct).
    _rerun(nb_runner,
           "class Counter:\n    def __init__(self):\n        self.hits = []\n    def __getitem__(self, k):\n        self.hits.append(k)\n        return k\nc = Counter()",
           "v = c['q']\nprint('hits', len(c.hits))", "hits 1")


def test_call_mutates_self(nb_runner):
    _rerun(nb_runner,
           "class Accum:\n    def __init__(self):\n        self.calls = []\n    def __call__(self, x):\n        self.calls.append(x)\n        return x\na = Accum()",
           "r = a('z')\nprint('calls', len(a.calls))", "calls 1")


def test_setitem_mutates_self(nb_runner):
    _rerun(nb_runner,
           "class Bag:\n    def __init__(self):\n        self.keys = []\n    def __setitem__(self, k, v):\n        self.keys.append(k)\nb = Bag()",
           "b['x'] = 9\nprint('keys', len(b.keys))", "keys 1")


# --- must NOT over-invalidate (pure) ------------------------------------------

def test_builtin_dict_setitem_not_over_invalidated(nb_runner):
    # the COMMON case: a plain dict subscript-assign must keep its cache and NOT
    # be treated as hidden-state mutation of anything but the dict.
    _rerun(nb_runner,
           "d = {}",
           "d['k'] = 1\nprint('v', d['k'])", "v 1")


def test_pure_call_not_over_invalidated(nb_runner):
    _rerun(nb_runner,
           "class Doubler:\n    def __call__(self, x):\n        return x * 2\nf = Doubler()",
           "r = f(5)\nprint('r', r)", "r 10")


def test_pure_getitem_not_over_invalidated(nb_runner):
    _rerun(nb_runner,
           "class Squares:\n    def __getitem__(self, k):\n        return k * k\nsq = Squares()",
           "r = sq[4]\nprint('r', r)", "r 16")
