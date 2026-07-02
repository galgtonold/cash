"""Hidden mutation reached through the harder / indirect object-protocol channels
must reset on isolated re-run (CAS-80). Each needs a bespoke resolver: an
ExitStack.enter_context, a context-manager FACTORY binding, an aliased or
class-based decorator, a functools memoizer, or the iterator protocol.

Each helper mutates ONCE, so the correct value is identical on the first run and
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

def test_exitstack_enter_context(nb_runner):
    _rerun(nb_runner,
           "import contextlib\nlog = []\nclass CM:\n    def __enter__(self):\n        log.append(1)\n        return self\n    def __exit__(self, *a):\n        return False\ncm = CM()",
           "with contextlib.ExitStack() as stack:\n    stack.enter_context(cm)\nprint('len', len(log))", "len 1")


def test_with_factory_binding(nb_runner):
    _rerun(nb_runner,
           "hits = []\nclass Mgr:\n    def __enter__(self):\n        hits.append(1)\n        return len(hits)\n    def __exit__(self, *a):\n        return False\ndef cm():\n    return Mgr()",
           "with cm() as x:\n    v = x\nprint('hits', len(hits), 'v', v)", "hits 1 v 1")


def test_aliased_decorated_function(nb_runner):
    _rerun(nb_runner,
           "log = []\ndef counting(f):\n    def wrap(*a, **k):\n        log.append(1)\n        return f(*a, **k)\n    return wrap\ndef g():\n    return 1\ng = counting(g)\nh = g",
           "h()\nprint('log', len(log))", "log 1")


def test_class_based_decorator(nb_runner):
    _rerun(nb_runner,
           "class Counter:\n    def __init__(self, f):\n        self.f = f\n        self.n = 0\n    def __call__(self, *a, **k):\n        self.n += 1\n        return self.f(*a, **k)\n@Counter\ndef task():\n    return 7",
           "task()\nprint('n', task.n)", "n 1")


def test_lru_cache_side_effect(nb_runner):
    _rerun(nb_runner,
           "import functools\nseen = []\n@functools.lru_cache(maxsize=None)\ndef f(x):\n    seen.append(x)\n    return x * x",
           "f(3)\nf(3)\nprint('seen', len(seen))", "seen 1")


def test_iterator_next_hidden_cursor(nb_runner):
    _rerun(nb_runner,
           "cursor = [0]\ndata = [10, 20, 30]\nclass It:\n    def __iter__(self):\n        return self\n    def __next__(self):\n        i = cursor[0]\n        cursor[0] += 1\n        return data[i]\nit = It()",
           "v = next(it)\nprint('v', v, 'cursor', cursor[0])", "v 10 cursor 1")


# --- must NOT over-invalidate (pure) ------------------------------------------

def test_pure_lru_cache_not_over_invalidated(nb_runner):
    # a memoized PURE function (no side effect) must stay correct on re-run.
    _rerun(nb_runner,
           "import functools\n@functools.lru_cache(maxsize=None)\ndef sq(x):\n    return x * x",
           "r = sq(6)\nprint('r', r)", "r 36")


def test_pure_factory_cm_not_over_invalidated(nb_runner):
    _rerun(nb_runner,
           "class Quiet:\n    def __enter__(self):\n        return 42\n    def __exit__(self, *a):\n        return False\ndef make():\n    return Quiet()",
           "with make() as v:\n    r = v\nprint('r', r)", "r 42")
