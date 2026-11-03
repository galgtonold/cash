"""The TSP notebook's Held-Karp cell is served on the next Run All and after
a restart, with ``shortest_segment`` holding what a run leaves in it.

``held_karp`` empties ``shortest_segment`` before any call of it and fills
it again: 11,264 answers for 12 cities (245,760 for 16), more than cash
checks in a memo it did not see emptied. Such a cell ran on every pass. A
memo emptied first is stored however big, its keys being plain values a
pickle gives back equal; a hit empties the live memo and puts them in, so a
later call finds every answer, and the tour is the memo's answer itself, as
after a run.
"""

from __future__ import annotations

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = "import functools, random, time"
CITIES = (
    "@functools.cache\n"
    "def random_cities(n, seed=1234):\n"
    "    random.seed(n * seed)\n"
    "    return frozenset(complex(round(random.uniform(0, 300)), round(random.uniform(0, 200))) for _ in range(n))\n"
    "def tour_length(t):\n"
    "    return sum(abs(t[i] - t[i - 1]) for i in range(len(t)))"
)
EXACT = (
    "@functools.cache\n"
    "def shortest_segment(A, Bs, C):\n"
    "    if not Bs:\n"
    "        return [A, C]\n"
    "    return min(\n"
    "        (shortest_segment(A, Bs - {B}, B) + [C] for B in Bs), key=lambda s: tour_length(s) - abs(s[0] - s[-1])\n"
    "    )\n"
    "def held_karp(c):\n"
    "    A = min(c, key=abs)\n"
    "    shortest_segment.cache_clear()\n"
    "    return min((shortest_segment(A, c - {A, C}, C) for C in c - {A}), key=tour_length)\n"
    "exact = (held_karp,)"
)
CITY_COUNT = 12
CELLS = [
    "%cash_badge print\n" + SETUP,
    CITIES,
    EXACT,
    # Leaves the memo holding answers, so the clear below is seen as the
    # first. Cities of its own: one more ``random_cities`` call above the
    # served cell is another matter (its reseed is not replayed there).
    "small = held_karp(frozenset({1 + 1j, 4 + 2j, 2 + 5j, 6 + 6j, 3 + 3j}))",
    f"t = ([f(random_cities({CITY_COUNT})) for f in exact], time.sleep(0.3))[0][0]",
    "x = random.random()",
]
HELD_KARP_CELL = 5
#: Whether the tour is one of the memo's answers, the very object.
TOUR_IS_AN_ANSWER = (
    f"(lambda c: (lambda A: any(shortest_segment(A, c - {{A, C}}, C) is t for C in c - {{A}}))"
    f"(min(c, key=abs)))(random_cities({CITY_COUNT}))"
)


def _plain() -> dict:
    """What a restart and Run All in plain Python leave."""
    import random

    state = random.getstate()
    ns: dict = {"__name__": "__plain__"}
    try:
        for cell in CELLS:
            exec(cell.replace("%cash_badge print\n", ""), ns)
        memo = ns["shortest_segment"]
        return {
            "t": ns["t"],
            "x": ns["x"],
            "size": memo.cache_info().currsize,
            "shared": eval(TOUR_IS_AN_ANSWER, ns),
        }
    finally:
        random.setstate(state)


def _check(nb_runner, plain: dict, when: str) -> None:
    assert shows_cached(nb_runner.get_output(HELD_KARP_CELL)), (when, nb_runner.get_output(HELD_KARP_CELL))
    assert nb_runner.peek("t") == repr(plain["t"]), when
    assert float(nb_runner.peek("x")) == plain["x"], when
    assert nb_runner.peek("shortest_segment.cache_info().currsize") == repr(plain["size"]), when
    misses = nb_runner.peek("shortest_segment.cache_info().misses")
    # The tour is the memo's answer, and every call of it below finds one.
    assert plain["shared"] is True
    assert nb_runner.peek(TOUR_IS_AN_ANSWER) == "True", when
    assert nb_runner.peek("shortest_segment.cache_info().misses") == misses, when


def test_the_held_karp_cell_is_served_on_the_next_run_all_and_after_a_restart(nb_runner):
    plain = _plain()
    assert plain["size"] > 4096  # more than cash checks in a memo not seen emptied
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    _check(nb_runner, plain, "second Run All")
    nb_runner.settle_writes()
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()
    _check(nb_runner, plain, "after a restart")
