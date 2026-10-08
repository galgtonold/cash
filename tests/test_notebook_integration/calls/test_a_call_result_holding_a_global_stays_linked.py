"""An expensive notebook function that hands back an object it keeps -- a
global, an item of one, a class attribute, its own lazily built singleton, a
bound method of a global, a dict holding a global -- still hands back that
very object on the next Run All, and after a restart.

The call cache used to serve such a call as a deserialised copy: ``m = best()``
returning ``MODELS[1]`` and then ``m['w'] = 9`` left ``MODELS`` as it was on
the second Run All, and ``b['model'] is MODEL`` turned False (bug-hunt-5
AL-02, CC-6). Plain Jupyter is the oracle: every check reads kernel state out
of band with ``peek``, since a cached cell replays its stdout.
"""

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

SETUP = "import cash\n%cash_on\n%cash_badge print\nimport time"

CASES = {
    "lazy-singleton": (
        [
            "_DATA = None\ndef get_data():\n    global _DATA\n    if _DATA is None:\n"
            "        time.sleep(0.3)\n        _DATA = {'rows': [1, 2]}\n    return _DATA",
            "d = get_data()",
            "d['rows'].append(3)",
        ],
        {"get_data()['rows']": "[1, 2, 3]", "d is get_data()": "True"},
    ),
    "item-of-a-global": (
        ["MODELS = [{'w': 0}, {'w': 0}]", "def best():\n    time.sleep(0.3)\n    return MODELS[1]", "m = best()", "m['w'] = 9"],
        {"MODELS": "[{'w': 0}, {'w': 9}]", "m is MODELS[1]": "True"},
    ),
    "class-attribute": (
        ["class Config:\n    items = []", "def items():\n    time.sleep(0.3)\n    return Config.items", "it = items()", "it.append('x')"],
        {"Config.items": "['x']"},
    ),
    "bound-method": (
        ["LOG = []", "def logger():\n    time.sleep(0.3)\n    return LOG.append", "log = logger()", "log('hello')"],
        {"LOG": "['hello']"},
    ),
    "dict-holding-a-global": (
        [
            "MODEL = {'w': 1}\ndef build(tag):\n    time.sleep(0.3)\n    return {'tag': tag, 'model': MODEL}",
            "b = build('x')",
            "MODEL['w'] = 2",
        ],
        {"b['model']['w']": "2", "b['model'] is MODEL": "True"},
    ),
}


@pytest.mark.parametrize("case", list(CASES))
def test_a_second_run_all_hands_back_the_global_itself(nb_runner, case):
    cells, expected = CASES[case]
    nb_runner.create_notebook([SETUP, *cells])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    assert {expr: nb_runner.peek(expr) for expr in expected} == expected


@pytest.mark.fresh_kernel
def test_a_restart_hands_back_the_global_itself(nb_runner):
    """After a restart the statement entry must not restore a copy either."""
    nb_runner.create_notebook(
        [SETUP, "REG = {'k': 1}", "def get():\n    time.sleep(0.3)\n    return REG", "c = get()", "c['k'] = 2"]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.restart()
    nb_runner.run_all()
    assert nb_runner.peek("REG") == "{'k': 2}"
    assert nb_runner.peek("c is REG") == "True"


def test_a_function_that_only_reads_a_global_is_still_served(nb_runner):
    """Control: a new object built from a global is still a call-cache hit."""
    nb_runner.create_notebook(
        [SETUP, "MODELS = [1, 2]", "def total():\n    time.sleep(0.3)\n    return {'sum': sum(MODELS)}", "t = total()"]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    assert nb_runner.peek("t") == "{'sum': 3}"
    out = nb_runner.get_output(4)
    assert shows_cached(out) or "1/1 hit" in out, out
