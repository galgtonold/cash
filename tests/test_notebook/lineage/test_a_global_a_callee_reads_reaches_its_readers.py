"""A global the called function reads is in the statement's key, and in what
its outputs carry to their readers.

``res = summarize(raw)`` reads ``CFG`` through ``summarize``. Changing
``CFG["round"]`` in place re-keyed ``res`` (it ran again), but ``res`` kept
its lineage, so ``report = render(res)`` below kept its key and returned the
report of the old ``res``.
"""

from __future__ import annotations

from tests._cell_driver import run_cash_cell

DEFS = (
    "def summarize(xs):\n    return [round(x, CFG['round']) for x in xs]\n\n"
    "def render(r):\n    return ', '.join(str(x) for x in r)"
)
USE = "res = summarize(raw)\nreport = render(res)"


def test_a_change_in_place_to_a_global_a_callee_reads_reaches_the_reader(cash_magics):
    cash_magics.cash_persist("on")
    for cell in ["CFG = {'round': 2}\nraw = [1.234, 2.345]", DEFS, USE]:
        run_cash_cell(cash_magics, cell)
    assert cash_magics.shell.user_ns["report"] == "1.23, 2.35"
    run_cash_cell(cash_magics, "CFG['round'] = 1")
    run_cash_cell(cash_magics, USE)
    assert cash_magics.shell.user_ns["res"] == [1.2, 2.3]
    assert cash_magics.shell.user_ns["report"] == "1.2, 2.3"


def test_an_unchanged_global_keeps_the_lineage(cash_magics):
    """The lineage moves only when what the callee reads moves."""
    cells = ["CFG = {'round': 2}\nraw = [1.234, 2.345]", DEFS, USE]
    lineages = []
    for _ in range(2):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        lineages.append(cash_magics.tracking_state.variable_lineage["res"])
    assert lineages[0] == lineages[1]


def test_a_global_read_in_a_generator_expression_is_in_the_key(cash_magics):
    """A generator expression, a lambda or (before Python 3.12) a
    comprehension is code of its own inside the function; the globals it
    reads were left out of the key, and ``y`` was served from before the
    change."""
    cash_magics.cash_persist("on")
    run_cash_cell(cash_magics, "K = 2\ndef scaled(xs):\n    return sum(x * K for x in xs)")
    run_cash_cell(cash_magics, "y = scaled([1, 2])")
    assert cash_magics.shell.user_ns["y"] == 6
    run_cash_cell(cash_magics, "K = 3")
    run_cash_cell(cash_magics, "y = scaled([1, 2])")
    assert cash_magics.shell.user_ns["y"] == 9
