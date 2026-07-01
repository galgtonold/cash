"""Hidden mutation through a context manager (``with``) must reset on isolated
re-run (CAS-69). A ``with`` statement invokes ``__enter__`` / ``__exit__`` (or a
``@contextlib.contextmanager`` generator) whose body mutates hidden state (a
captured free variable or the context-manager object itself); on an isolated
re-run the mutation accumulates because nothing in the cell text names it.

Each context body runs ONCE, so the correct value is identical on the first run
and every re-run.
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

def test_contextmanager_generator_free_var(nb_runner):
    # @contextlib.contextmanager generator mutates a captured module list.
    _rerun(nb_runner,
           "import contextlib\nlog = []\n@contextlib.contextmanager\ndef track():\n    log.append('enter')\n    yield\n    log.append('exit')",
           "with track():\n    pass\nprint('log', len(log))", "log 2")


def test_context_object_enter_mutates_self(nb_runner):
    # context-manager OBJECT whose __enter__ mutates self.n.
    _rerun(nb_runner,
           "class Counter:\n    def __init__(self):\n        self.n = 0\n    def __enter__(self):\n        self.n += 1\n        return self\n    def __exit__(self, *a):\n        return False\ncm = Counter()",
           "with cm:\n    pass\nprint('n', cm.n)", "n 1")


def test_context_object_exit_mutates_self(nb_runner):
    # mutation happens in __exit__ rather than __enter__.
    _rerun(nb_runner,
           "class Tracker:\n    def __init__(self):\n        self.closes = 0\n    def __enter__(self):\n        return self\n    def __exit__(self, *a):\n        self.closes += 1\n        return False\nt = Tracker()",
           "with t:\n    pass\nprint('closes', t.closes)", "closes 1")


# --- must NOT over-invalidate (pure) ------------------------------------------

def test_pure_context_object_not_over_invalidated(nb_runner):
    # a context manager that does NOT mutate persistent state.
    _rerun(nb_runner,
           "class Noop:\n    def __enter__(self):\n        return 5\n    def __exit__(self, *a):\n        return False\ncm = Noop()",
           "with cm as v:\n    r = v * 2\nprint('r', r)", "r 10")


def test_suppress_body_mutation_still_resets(nb_runner):
    # contextlib.suppress with a mutating BODY already resets (CAS-57/66) — guard.
    _rerun(nb_runner,
           "import contextlib\ndata = []",
           "with contextlib.suppress(ValueError):\n    data.append(1)\nprint('len', len(data))", "len 1")
