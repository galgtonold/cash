"""A cell with a control structure leaves what plain Python would leave.

Cash runs ``for``/``if``/``try`` statement by statement (and a learned loop
split as a head and a tail), so Python's control flow is re-implemented in
the handlers rather than inherited. Each case here is a short run of cells,
executed once with ``exec`` and once through ``run_cash_cell``; after every
cell the two namespaces and the exception type must match. The cells run
twice, so the second pass checks the cache hits as well.

Loop splits are learned on the first pass (the thresholds below make every
eligible loop split), so a loop over a list longer than the split's head runs
split on the second pass.
"""

from __future__ import annotations

import types

import pytest

from tests._cell_driver import run_cash_cell

CASES = {
    # try / except / else / finally
    "finally runs when no handler matches": [
        "done = False\ntry:\n    1/0\nexcept KeyError:\n    handled = True\nfinally:\n    done = True\n",
    ],
    "finally runs when the handler raises": [
        "done = False\ntry:\n    1/0\nexcept ZeroDivisionError:\n    raise ValueError('h')\nfinally:\n    done = True\n",
    ],
    "finally runs when else raises, and the handlers do not catch it": [
        "done = False\ntry:\n    a = 1\nexcept ValueError:\n    handled = True\nelse:\n"
        "    raise ValueError('else')\nfinally:\n    done = True\n",
    ],
    "an exception raised in finally replaces the one in flight": [
        "try:\n    1/0\nfinally:\n    raise KeyError('fin')\n",
    ],
    "KeyboardInterrupt reaches its handler and finally": [
        "try:\n    raise KeyboardInterrupt\nexcept KeyboardInterrupt:\n    stopped = True\nfinally:\n    fin = True\n",
    ],
    "a bare except catches a BaseException": [
        "try:\n    raise SystemExit(3)\nexcept:\n    stopped = True\n",
    ],
    "an unmatched BaseException still runs finally": [
        "try:\n    raise KeyboardInterrupt\nexcept Exception:\n    handled = True\nfinally:\n    fin = True\n",
    ],
    "the except name is unbound after the handler": [
        "e = 'before'\ntry:\n    1/0\nexcept ZeroDivisionError as e:\n    msg = str(e)\n",
    ],
    "a bare raise in a handler re-raises the caught exception": [
        "try:\n    {}['k']\nexcept KeyError:\n    seen = True\n    raise\n",
    ],
    "else runs only after a clean try body": [
        "log = []\ntry:\n    log.append('body')\nexcept ValueError:\n    log.append('handler')\nelse:\n"
        "    log.append('else')\nfinally:\n    log.append('finally')\n",
    ],
    "nested try": [
        "log = []\ntry:\n    try:\n        1/0\n    except KeyError:\n        log.append('inner')\n    finally:\n"
        "        log.append('inner finally')\nexcept ZeroDivisionError:\n    log.append('outer')\nfinally:\n"
        "    log.append('outer finally')\n",
    ],
    # if
    "an if condition's error keeps its type for an enclosing handler": [
        "d = {}\ntry:\n    if d['k']:\n        x = 1\nexcept KeyError:\n    caught = True\n",
    ],
    "an if condition's error keeps its type at the top level": [
        "d = {}\nif d['k']:\n    x = 1\n",
    ],
    "an error from a condition's truth value keeps its type": [
        "class Unsure:\n    def __bool__(self):\n        raise KeyError('bool')\n",
        "try:\n    if Unsure():\n        x = 1\nexcept KeyError:\n    caught = True\n",
    ],
    "an elif condition's error keeps its type": [
        "d = {}\nif False:\n    x = 1\nelif d['k']:\n    x = 2\n",
    ],
    # for / while
    "for-else runs after a loop that finished": [
        "tot = 0\nfor i in range(3):\n    tot = tot + i\nelse:\n    fe = 'ran'\nafter = 1\n",
    ],
    "for-else is skipped after a break": [
        "for i in range(3):\n    if i == 1:\n        break\nelse:\n    fe = 'ran'\n",
    ],
    "for-else is skipped when the body raises": [
        "for i in range(3):\n    if i == 1:\n        raise ValueError(i)\nelse:\n    fe = 'ran'\n",
    ],
    "for-else over an empty iterable": [
        "for i in []:\n    x = i\nelse:\n    fe = 'ran'\n",
    ],
    "a for-else nested in a loop": [
        "log = []\nfor i in range(3):\n    for j in range(i):\n        log.append(j)\n    else:\n        log.append('e')\n",
    ],
    "a split loop that raises in its head skips the tail": [
        "boom = False\nseen = []",
        "for i in range(20):\n    seen.append(i)\n    if boom and i == 2:\n        raise ValueError('boom')\n",
        "boom = True\nseen = []",
        "for i in range(20):\n    seen.append(i)\n    if boom and i == 2:\n        raise ValueError('boom')\n",
    ],
    "a split loop that raises in its tail": [
        "boom = False\nseen = []",
        "for i in range(20):\n    seen.append(i)\n    if boom and i == 12:\n        raise ValueError('boom')\n",
        "boom = True\nseen = []",
        "for i in range(20):\n    seen.append(i)\n    if boom and i == 12:\n        raise ValueError('boom')\n",
    ],
}


def _comparable(value):
    if isinstance(value, type):
        return f"class {value.__name__}"
    if isinstance(value, (types.FunctionType, types.GeneratorType)):
        return type(value).__name__
    return value


def _user_names(ns: dict, baseline: set[str]) -> dict:
    return {k: _comparable(v) for k, v in ns.items() if k not in baseline and not k.startswith("_")}


def _run(fn) -> type | None:
    try:
        fn()
    except BaseException as exc:  # noqa: BLE001 - the error type is what the test compares
        return type(exc)
    return None


@pytest.fixture
def magics(cash_magics, cash_instance):
    # Every eligible loop learns a split on its first run (see the module docstring).
    cash_instance.config.loop_split_max_iter_seconds = 1.0
    cash_instance.config.loop_split_min_remaining_seconds = 0.0
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    return cash_magics


@pytest.mark.parametrize("cells", list(CASES.values()), ids=list(CASES))
def test_a_cell_leaves_what_python_leaves(magics, mock_shell, cells):
    plain: dict = {}
    baseline = set(mock_shell.user_ns)
    for run in (1, 2):
        for n, cell in enumerate(cells, 1):
            expected_error = _run(lambda cell=cell: exec(cell, plain))
            error = _run(lambda cell=cell: run_cash_cell(magics, cell))
            where = f"run {run}, cell {n}"
            assert error is expected_error, f"{where}: raised {error}, Python raised {expected_error}"
            assert _user_names(mock_shell.user_ns, baseline) == _user_names(plain, set()), where
