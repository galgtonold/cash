"""Which calls of a notebook function count as a bare seed.

``set_seed(5)`` with ``def set_seed(s): random.seed(s); np.random.seed(s)``
changes the streams and nothing a variable holds, so the upstream check may
run it again to put a stream back, as it runs ``np.random.seed(5)`` again.
A function that does anything else is not run again for the stream alone.
"""

from __future__ import annotations

import pytest

MODULES = {"random", "np", "os", "torch"}


def _modules(body: str, call: str = "set_seed(5)", extra: dict[str, str] | None = None) -> frozenset[str]:
    from cash.notebook.callee_reach import seed_only_call_modules

    defs = {"set_seed": "def set_seed(s):\n" + body, **(extra or {})}
    return seed_only_call_modules(call, {}, defs.get, MODULES.__contains__)


@pytest.mark.parametrize(
    "body",
    [
        "    random.seed(s)\n    np.random.seed(s)",
        '    """Seed everything."""\n    random.seed(s)\n    np.random.seed(int(s))',
        "    import random\n    random.seed(s)\n    np.random.seed(s)\n    return",
        "    random.seed(s)\n    np.random.seed(s)\n    os.environ['PYTHONHASHSEED'] = str(s)",
        "    random.seed(s)\n    np.random.seed(s)\n    torch.backends.cudnn.deterministic = True",
        "    random.seed(s)\n    np.random.seed(s)\n    if torch.cuda.is_available():\n        torch.cuda.manual_seed_all(s)",
    ],
)
def test_a_function_that_only_seeds(body):
    assert {"random", "numpy.random"} <= _modules(body)


def test_a_function_that_seeds_through_another():
    inner = {"seed_numpy": "def seed_numpy(s):\n    np.random.seed(s)"}
    assert _modules("    random.seed(s)\n    seed_numpy(s)", extra=inner) == {"random", "numpy.random"}


@pytest.mark.parametrize(
    "body",
    [
        "    print('seeding', s)\n    np.random.seed(s)",  # shows something
        "    LOG.append(s)\n    np.random.seed(s)",  # changes a value of the notebook
        "    global SEED\n    SEED = s\n    np.random.seed(s)",  # binds a name
        "    np.random.seed(s)\n    return np.random.rand()",  # draws
        "    np.random.seed()",  # from entropy: another stream each run
        "    np.random.seed(make(s))",  # a call among the arguments
        "    np.random.seed(s)\n    set_seed(s)",  # calls itself
        "    np.random.seed(s)\n    other(s)",  # a function cash cannot read
        "    for m in (random, np.random):\n        m.seed(s)",  # a loop
        "    cfg.seed = s\n    np.random.seed(s)",  # a setting on a value, not a module
        "    if np.random.rand() > 0.5:\n        np.random.seed(s)",  # the test draws
    ],
)
def test_a_function_that_does_more_is_not_one(body):
    assert _modules(body) == frozenset()


def test_a_decorated_function_is_not_one():
    from cash.notebook.callee_reach import seed_only_call_modules

    defs = {"set_seed": "@functools.cache\ndef set_seed(s):\n    np.random.seed(s)"}
    assert seed_only_call_modules("set_seed(5)", {}, defs.get, MODULES.__contains__) == frozenset()


@pytest.mark.parametrize("call", ["x = set_seed(5)", "set_seed(make())", "obj.set_seed(5)", "set_seed(5); set_seed(6)"])
def test_only_a_bare_call_with_plain_arguments(call):
    assert _modules("    np.random.seed(s)", call=call) == frozenset()
