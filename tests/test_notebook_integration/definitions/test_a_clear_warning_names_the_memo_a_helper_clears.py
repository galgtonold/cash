"""A statement whose helper clears a memo and fills another names the memo
it clears when it runs every time, on every Run All, and is served as a
run leaves the memos when it can be.

Run All defines the memos again, so the one cleared is empty before the
clear and its counters only grow, as the one filled beside it does. A
helper that also reads the memo's counters (``cache_info``), which a hit
does not give back, runs every time: the warning named the memo filled
(``other``); it names the one whose ``cache_clear`` the helper calls. One
that does not is stored with what it put into both memos, and a hit leaves
them holding what a run in plain Python leaves.
"""

from __future__ import annotations

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]

HELPERS = (
    "@functools.lru_cache(maxsize=None)\ndef random_cities(n):\n    return tuple(range(n))\n"
    "@functools.lru_cache(maxsize=None)\ndef other(n):\n    return n * 2\n"
    "@functools.lru_cache(maxsize=None)\ndef shortest_segment(a, b):\n    return abs(a - b)\n"
    "def held_karp(c):\n    shortest_segment.cache_clear(); time.sleep(0.2)\n"
    "    return sum(shortest_segment(c[i], c[i + 1]) for i in range(len(c) - 1))\n"
    "def mixed(c):\n    other(len(c) + 100)\n    return held_karp(c)"
)
COUNTED = HELPERS.replace("    return held_karp(c)", "    return held_karp(c), shortest_segment.cache_info().currsize")
#: The memos' sizes and answers, which a hit puts back, not their counters;
#: the sizes again after the calls, which find every answer.
MEMOS = (
    "([m.cache_info().currsize for m in (random_cities, other, shortest_segment)], tour, "
    "sorted(other(n) for n in (1, 106)), [random_cities(6)], [shortest_segment(i, i + 1) for i in range(5)], "
    "[m.cache_info().currsize for m in (random_cities, other, shortest_segment)])"
)


def _cells(helpers: str) -> list[str]:
    return [
        "%cash_badge print\nimport functools, time",
        helpers,
        "cities = random_cities(6); x = other(1)",
        "tour = mixed(cities)",
    ]


def _plain(cells: list[str]) -> str:
    ns: dict = {}
    for cell in cells:
        exec(cell.replace("%cash_badge print\n", ""), ns)
    return repr(eval(MEMOS, ns))


def test_the_memo_the_helper_clears_is_named(nb_runner):
    nb_runner.create_notebook(_cells(COUNTED))
    nb_runner.start_kernel()
    for run in range(2):
        nb_runner.run_all()
        shown = nb_runner.get_raw_output(4)
        assert "helper memo shortest_segment " in shown, (run, shown)


def test_a_helper_that_clears_without_reading_counters_is_served_with_the_memos_as_plain(nb_runner):
    cells = _cells(HELPERS)
    plain = _plain(cells)
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek(MEMOS) == plain
    nb_runner.run_all()
    assert shows_cached(nb_runner.get_output(4)), nb_runner.get_output(4)
    assert nb_runner.peek(MEMOS) == plain
