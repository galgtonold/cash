"""Hidden mutation through a decorator wrapper or a no-arg classmethod must reset
on isolated re-run (CAS-71, extends CAS-68). Calling a decorated function runs
the wrapper (not the resolved original ``def``), whose captured-var mutation is
missed; a no-arg classmethod mutating a class variable slips through the
method-receiver path.

Each helper appends/increments ONCE, so the correct value is identical on the
first run and every re-run.
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

def test_decorator_wrapper_free_var(nb_runner):
    _rerun(nb_runner,
           "calls = []\ndef logged(f):\n    def wrap(*a, **k):\n        calls.append('x')\n        return f(*a, **k)\n    return wrap\n@logged\ndef work():\n    return 42",
           "work()\nprint('calls', len(calls))", "calls 1")


def test_reassignment_decorator_free_var(nb_runner):
    _rerun(nb_runner,
           "hits = {'n': 0}\ndef counting(f):\n    def wrap(*a, **k):\n        hits['n'] += 1\n        return f(*a, **k)\n    return wrap\ndef g():\n    return 1\ng = counting(g)",
           "g()\nprint('n', hits['n'])", "n 1")


def test_noarg_classmethod_class_var(nb_runner):
    _rerun(nb_runner,
           "class Registry:\n    log = []\n    @classmethod\n    def record(cls):\n        cls.log.append('r')",
           "Registry.record()\nprint('len', len(Registry.log))", "len 1")


# --- must NOT over-invalidate (pure) ------------------------------------------

def test_pure_decorator_not_over_invalidated(nb_runner):
    _rerun(nb_runner,
           "import functools\ndef trace(f):\n    @functools.wraps(f)\n    def wrap(*a, **k):\n        return f(*a, **k)\n    return wrap\n@trace\ndef square(x):\n    return x * x",
           "r = square(6)\nprint('r', r)", "r 36")


def test_classmethod_with_arg_still_resets(nb_runner):
    # the existing passing form (CAS-68) — guard against regression.
    _rerun(nb_runner,
           "class Reg:\n    items = []\n    @classmethod\n    def add(cls, x):\n        cls.items.append(x)",
           "Reg.add(1)\nprint('len', len(Reg.items))", "len 1")
