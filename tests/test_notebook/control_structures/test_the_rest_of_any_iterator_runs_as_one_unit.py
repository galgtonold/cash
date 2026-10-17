"""A long loop over an iterator of unknown length runs its rest as one unit.

``for m in re.finditer(...)``, a generator, ``itertools.product(...)``: an
iterator with no length, which only a bare name (``for x in it:``) had its
rest continued as one unit for. Any other header ran every pass through the
per-statement machinery: 10,000 matches took 18 s against 2 ms plain. Now
the unit is handed the very iterator the first passes drew from, so each item
is still drawn once, in order.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

GEN = "drawn = []\ndef gen(n):\n    for i in range(n):\n        drawn.append(i)\n        yield i * 2\n"


def _statements(cash_magics, code):
    processor = cash_magics._statement_processor
    seen = []
    real = processor.process_statement

    def counted(*args, **kwargs):
        seen.append(1)
        return real(*args, **kwargs)

    processor.process_statement = counted
    try:
        run_cash_cell(cash_magics, code)
    finally:
        processor.process_statement = real
    return len(seen)


def test_a_generator_is_drawn_once_and_its_rest_runs_whole(cash_magics):
    ns = cash_magics.shell.user_ns
    run_cash_cell(cash_magics, GEN)
    ran = _statements(cash_magics, "tot = 0\nfor v in gen(3000):\n    tot += v")
    assert ns["tot"] == 2 * sum(range(3000))
    assert ns["drawn"] == list(range(3000))
    assert ran < 500, ran


@pytest.mark.parametrize(
    ("setup", "loop", "name", "expected"),
    [
        ("import re\ntext = 'ab ' * 3000", "n = 0\nfor m in re.finditer('ab', text):\n    n += 1", "n", 3000),
        (
            "import itertools",
            "best = 0\nfor a, b in itertools.product(range(60), range(60)):\n    best = max(best, a * b + a)",
            "best",
            59 * 59 + 59,
        ),
    ],
)
def test_the_whole_loop_is_run_again_and_after_an_edit(cash_magics, setup, loop, name, expected):
    ns = cash_magics.shell.user_ns
    run_cash_cell(cash_magics, setup)
    for _ in range(2):
        run_cash_cell(cash_magics, loop)
        assert ns[name] == expected


def test_a_rerun_after_an_input_changed_reads_the_new_input(cash_magics):
    ns = cash_magics.shell.user_ns
    loop = "import re\nn = 0\nfor m in re.finditer('ab', text):\n    n += len(m.group())"
    run_cash_cell(cash_magics, "text = 'ab ' * 3000")
    run_cash_cell(cash_magics, loop)
    assert ns["n"] == 6000
    run_cash_cell(cash_magics, "text = 'ab ' * 2000")
    run_cash_cell(cash_magics, loop)
    assert ns["n"] == 4000


def test_a_named_iterator_inside_the_header_is_left_drained(cash_magics):
    """``enumerate(it)``: the unit draws the rest of ``it``, and a later run of
    the loop is not served past an iterator a hit could not advance."""
    ns = cash_magics.shell.user_ns
    for _ in range(2):
        run_cash_cell(cash_magics, "it = iter(range(3000))")
        run_cash_cell(cash_magics, "tot = 0\nfor i, v in enumerate(it):\n    tot += v")
        run_cash_cell(cash_magics, "left = list(it)")
        assert ns["tot"] == sum(range(3000))
        assert ns["left"] == []
