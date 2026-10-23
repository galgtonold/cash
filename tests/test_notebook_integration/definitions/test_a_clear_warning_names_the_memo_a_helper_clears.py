"""A statement whose helper clears a memo and fills another names the memo
it clears, on every Run All.

Run All defines the memos again, so the one cleared is empty before the
clear and its counters only grow, as the one filled beside it does. The
warning named the memo filled (``other``); it names the one whose
``cache_clear`` the helper calls.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]

CELLS = [
    "%cash_badge print\nimport functools, time",
    "@functools.lru_cache(maxsize=None)\ndef random_cities(n):\n    return tuple(range(n))\n"
    "@functools.lru_cache(maxsize=None)\ndef other(n):\n    return n * 2\n"
    "@functools.lru_cache(maxsize=None)\ndef shortest_segment(a, b):\n    return abs(a - b)\n"
    "def held_karp(c):\n    shortest_segment.cache_clear(); time.sleep(0.2)\n"
    "    return sum(shortest_segment(c[i], c[i + 1]) for i in range(len(c) - 1))\n"
    "def mixed(c):\n    other(len(c) + 100)\n    return held_karp(c)",
    "cities = random_cities(6); x = other(1)",
    "tour = mixed(cities)",
]


def test_the_memo_the_helper_clears_is_named(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    for run in range(2):
        nb_runner.run_all()
        shown = nb_runner.get_raw_output(4)
        assert "helper memo shortest_segment " in shown, (run, shown)
