"""A cached function that reads a variable of its enclosing function answers
for the value that variable holds now, also when it holds an earlier value
again.

``threshold = 1`` ... ``threshold = 9`` ... ``threshold = 1`` around repeated
calls of ``above(5)``: a write to a closure cell is seen by no watcher, so
the repeated-hit memo compares the cells it remembered. The hit for 9 was
remembered beside the cells of 1, and served once the variable was 1 again.
The same call is made a few times per value, as a loop or a service makes
it: the memo only serves from the second hit on.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.core

REPEATS = 4


def test_a_loop_variable_of_the_enclosing_function(disk_cash):
    threshold = 1

    @disk_cash.cache
    def above(x):
        return x > threshold

    seen = []
    for threshold in (1, 9, 1, 9, 3, 1):
        seen.append([above(5) for _ in range(REPEATS)])
    assert seen == [[5 > t] * REPEATS for t in (1, 9, 1, 9, 3, 1)]


def test_a_helper_whose_captured_variable_a_setter_sets(disk_cash):
    def make():
        k = 2

        def scaled(x):
            return x * k

        def configure(value):
            nonlocal k
            k = value

        return scaled, configure

    scaled, configure = make()

    @disk_cash.cache
    def by_helper(x):
        return scaled(x)

    seen = []
    for k in (2, 7, 2, 7):
        configure(k)
        seen.append([by_helper(3) for _ in range(REPEATS)])
    assert seen == [[3 * k] * REPEATS for k in (2, 7, 2, 7)]


def test_two_argument_tuples_each_follow_the_variable(disk_cash):
    offset = 10

    @disk_cash.cache
    def shifted(x):
        return x + offset

    seen = []
    for offset in (10, 20, 10):
        seen.append([(shifted(1), shifted(2)) for _ in range(REPEATS)])
    assert seen == [[(1 + o, 2 + o)] * REPEATS for o in (10, 20, 10)]
