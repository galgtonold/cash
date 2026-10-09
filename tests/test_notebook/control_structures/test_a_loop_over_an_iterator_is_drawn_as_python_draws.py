"""A loop over an iterator draws from it exactly as plain Python does.

A loop whose header only names an iterator of unknown length
(``it = filter(...)``, ``g = gen()``, then ``for x in it:``) runs its first
passes one by one and, once it is long enough, the rest as one unit from
source, which draws the rest from the same iterator. These pin that nothing
about the drawing changes: the order of the generator's own work against the
body's, where it stops on an error, what is left in it, and what a loop over
a name holding an open file or a re-iterable object does.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

GEN = "def gen(n):\n    for i in range(n):\n        log.append(('g', i))\n        yield i\n"


@pytest.mark.parametrize("n", [5, 300])
def test_the_generator_and_the_body_interleave_as_in_python(cash_magics, n):
    ns = cash_magics.shell.user_ns
    run_cash_cell(cash_magics, "log = []\n" + GEN)
    run_cash_cell(cash_magics, f"g = gen({n})\nfor x in g:\n    log.append(('b', x))\n    y = x * 2")
    assert ns["log"] == [step for i in range(n) for step in (("g", i), ("b", i))]
    assert ns["x"] == n - 1 and ns["y"] == 2 * (n - 1)
    assert list(ns["g"]) == []


def test_an_error_stops_the_drawing_where_python_stops(cash_magics):
    ns = cash_magics.shell.user_ns
    run_cash_cell(cash_magics, "log = []\n" + GEN)
    with pytest.raises(ValueError, match="stop at 250"):
        run_cash_cell(
            cash_magics,
            "g = gen(300)\nfor x in g:\n    log.append(('b', x))\n    if x == 250:\n        raise ValueError('stop at 250')",
        )
    assert ns["log"] == [step for i in range(251) for step in (("g", i), ("b", i))]
    assert next(ns["g"]) == 251


def test_a_filter_gives_the_items_plain_python_gives(cash_magics):
    ns = cash_magics.shell.user_ns
    run_cash_cell(cash_magics, "data = [('u%d' % i, [1] * (i % 3)) for i in range(2000)]")
    run_cash_cell(
        cash_magics,
        "it = filter(lambda x: len(x[1]) != 1, data)\nempty = []\nfor (name, act) in it:\n"
        "    if (len(act) == 0):\n        empty.append(name)",
    )
    assert ns["empty"] == ["u%d" % i for i in range(2000) if i % 3 == 0]
    assert ns["name"] == "u1998"
    assert list(ns["it"]) == []
    # Run again over new data: nothing is served from before.
    run_cash_cell(cash_magics, "data = [('v%d' % i, []) for i in range(400)]")
    run_cash_cell(
        cash_magics,
        "it = filter(lambda x: len(x[1]) != 1, data)\nempty = []\nfor (name, act) in it:\n"
        "    if (len(act) == 0):\n        empty.append(name)",
    )
    assert ns["empty"] == ["v%d" % i for i in range(400)]


def test_a_named_open_file_is_read_once_and_left_open(cash_magics, tmp_path):
    ns = cash_magics.shell.user_ns
    path = tmp_path / "lines.txt"
    path.write_text("".join(f"line {i}\n" for i in range(3000)), encoding="utf-8")
    run_cash_cell(cash_magics, f"f = open({str(path)!r})\nfirst = next(f)\nn = 0\nfor line in f:\n    n += 1")
    assert ns["first"] == "line 0\n" and ns["n"] == 2999
    assert not ns["f"].closed
    assert ns["f"].read() == ""
    ns["f"].close()


def test_a_named_reiterable_object_is_iterated_in_full(cash_magics):
    ns = cash_magics.shell.user_ns
    run_cash_cell(
        cash_magics,
        "class Bag:\n    def __init__(self):\n        self.calls = 0\n    def __iter__(self):\n"
        "        self.calls += 1\n        return iter(range(500))\nbag = Bag()",
    )
    run_cash_cell(cash_magics, "tot = 0\nfor v in bag:\n    tot += v")
    assert ns["tot"] == sum(range(500))
