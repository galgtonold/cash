"""Which code after a kept plot run reads a name it bound (``first_read_after``),
and which of a statement's reasons say only that it draws.

Served, a plot run binds nothing: a later read of its ``fig`` or ``ax``
before binding it again would find no figure, so the run is drawn instead.
"""

from __future__ import annotations

import ast

import pytest

from cash.notebook.plot_outputs import first_read_after, is_pyplot_reason, mutated_names
from cash.notebook.pyplot_state import CHANGED_PYPLOT_REASON


def _statements(*cells: str):
    out = []
    for cell in cells:
        out.extend((ast.get_source_segment(cell, node) or "", node) for node in ast.parse(cell).body)
    return out


NAMES = frozenset({"fig", "ax"})


@pytest.mark.parametrize(
    "below",
    [
        "fig.savefig('a.png')",
        "print(ax.get_title())",
        "def f():\n    return fig\n",
        "del ax",
        "for a in [ax]:\n    pass",
        "if cond:\n    ax = 1\nax.plot([1])",
        "fig2 = fig",
        "eval('fig')",
        "x = [a for a in (fig,)]",
    ],
)
def test_a_read_before_binding_again_is_found(below):
    assert first_read_after(NAMES, _statements(below)) in NAMES


@pytest.mark.parametrize(
    "below",
    [
        "fig, ax = plt.subplots()\nax.plot([1])\nfig.savefig('a.png')",
        "for ax in axes.flat:\n    ax.hist(x)",
        "x = [ax for ax in axes]",
        "def f(ax):\n    return ax\n",
        "plt.xlabel('later')\nplt.show()",
        "fig = plt.figure()\nax = fig.add_subplot()\nprint(ax)",
    ],
)
def test_code_that_binds_first_or_reads_nothing_is_no_read(below):
    assert first_read_after(NAMES, _statements(below)) is None


def test_the_cells_are_read_in_order():
    """A cell binding ``fig`` again hides a later cell's read of it."""
    assert first_read_after(NAMES, _statements("fig, ax = plt.subplots()", "fig.savefig('a.png')")) is None
    assert first_read_after(NAMES, _statements("fig.savefig('a.png')", "fig, ax = plt.subplots()")) == "fig"


def test_a_statement_that_cannot_be_parsed_is_read_as_text():
    assert first_read_after(NAMES, [("%timeit fig.savefig('a.png')", None)]) == "fig"
    assert first_read_after(NAMES, [("%timeit other()", None)]) is None


def test_a_read_of_the_output_history_is_found_when_the_run_echoed():
    names = NAMES | {"_", "Out"}
    assert first_read_after(names, _statements("print(_5)")) is not None
    assert first_read_after(names, _statements("Out[3].set_title('x')")) == "Out"
    assert first_read_after(NAMES, _statements("print(_5)")) is None


def test_only_the_reasons_that_say_it_draws_are_pyplot_reasons():
    assert is_pyplot_reason(CHANGED_PYPLOT_REASON)
    assert is_pyplot_reason("Side effect: plt.plot (display)")
    assert is_pyplot_reason(
        "Calls run(), which reads or changes pyplot's current figure through run: it runs every time"
    )
    assert not is_pyplot_reason("Side effect: df.to_csv (file_write)")
    assert not is_pyplot_reason("Input variable missing lineage")


def test_an_in_place_change_names_what_it_changed():
    assert mutated_names("In-place mutation on: ax, fig (statement re-executes)") == {"ax", "fig"}
    assert mutated_names("In-place mutation on: ax (receiver lineage bumped; statement re-executes)") == {"ax"}
    assert mutated_names("Side effect: plt.plot (display)") is None
