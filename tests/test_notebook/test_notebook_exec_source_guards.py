"""Guards for compiling a statement's ORIGINAL source instead of the unparsed form.

Task 1 changed what a %cash_on cell executes: the user's own text, comments and
all, rather than `ast.unparse` output. The cache key still comes from the
unparsed text. These tests pin the consequences -- the two that must NOT change,
and the one that deliberately does.
"""
from __future__ import annotations

import tempfile

import pytest

pytest.importorskip("IPython")

from cash import Cash
from cash.notebook.ipython.magics import CashMagics
from cash.source_norm import source_identity_digest
from tests.conftest import MockShell


@pytest.fixture
def cell_runner():
    shell = MockShell()
    cash = Cash(cache_dir=tempfile.mkdtemp(), register_magic=False)
    magics = CashMagics(shell, cash)
    magics.cash_on("")
    magics._badge_mode = "off"
    shell.user_ns["c"] = cash

    def run(cell: str):
        magics.cash("", cell)
        return shell.user_ns

    return run


def test_an_ordinary_comment_does_not_change_a_function_s_identity():
    """Reworded prose must not invalidate a cell-defined cached function."""
    base = "def f(n):\n    return n * 2\n"
    commented = "def f(n):\n    # explaining myself\n    return n * 2\n"
    assert source_identity_digest(base) == source_identity_digest(commented)


def test_a_cash_directive_does_change_it():
    """DELIBERATE. A waiver changes how the function is treated, so it changes
    what the function IS. A miss after adding one is correct -- do not 'fix' it."""
    base = "def f(n):\n    return n * 2\n"
    waived = "def f(n):\n    return n * 2  # @cash:assume-safe\n"
    assert source_identity_digest(base) != source_identity_digest(waived)


def test_a_control_body_still_executes(cell_runner):
    """A control structure has no original segment, so it falls back to the
    unparsed text. It must still run."""
    ns = cell_runner("total = 0\nfor i in range(3):\n    total += i\n")
    assert ns["total"] == 3


def test_trailing_semicolon_still_suppresses_the_repr(cell_runner):
    """The semicolon fixup interacts with the text handed to compile()."""
    ns = cell_runner("vals = [1, 2, 3]\nlen(vals);\n")
    assert ns["vals"] == [1, 2, 3]


def test_an_annotated_cell_defined_function_keeps_its_original_source(cell_runner):
    """The fix reached linecache: a directive-carrying function is compiled from
    the user's own text, so its comments survive `inspect.getsource`.

    This is the most direct evidence the fix works at all -- the marker can only
    appear if linecache holds the original, not `ast.unparse` output.
    """
    import inspect

    ns = cell_runner(
        "def marked(n):\n"
        "    value = n * 2  # @cash:assume-safe - a distinctive marker\n"
        "    return value\n"
    )
    src = inspect.getsource(ns["marked"])
    assert "a distinctive marker" in src, (
        f"linecache holds normalised source, not the user's:\n{src}"
    )


def test_an_unannotated_cell_defined_function_does_not(cell_runner):
    """The other half of the narrowing, pinned so it is a decision and not a
    silent gap.

    A `def` with no `@cash:` directive is deliberately still compiled from the
    unparsed form. The reason is measured: the upstream checker/restorer
    recompiles a def on a path that never threads the original source, so
    recovering it here unconditionally gave one unedited function two textual
    representations, which hashed differently, moved its CAS-243 call-cache key,
    and re-ran its body on the next same-session re-run. Gating on the directive
    keeps every unannotated function on exactly the path it used before.

    If a future change makes this assertion fail, that is not automatically a
    bug -- but it must be a deliberate choice that re-checks
    `test_callee_global_capture.py::test_a_same_session_rerun_neither_freezes_nor_accumulates`.
    """
    import inspect

    ns = cell_runner(
        "def plain(n):\n"
        "    value = n * 2  # an ordinary comment, no directive\n"
        "    return value\n"
    )
    assert "an ordinary comment" not in inspect.getsource(ns["plain"])
