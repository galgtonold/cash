"""CAS-163: a multi-line ``%``-format ``print(...)`` in a cell must not disable
cache restore for that cell.

The magic-stripper used by the upstream simulator dropped any physical line
that began with ``%``/``!``.  A valid multi-line statement whose continuation
line starts with the ``%`` (string-format) operator::

    print("Asian call = %.4f\\n"
          "European   = %.4f"
          % (a, b))

was therefore mangled into unparseable code, making the simulator raise a
*fictional* SyntaxError that silently aborted the replay and recomputed the
whole cell on every run.

These tests put genuinely-expensive work in the same cell as such a print and
assert the cell RESTORES from cache on an isolated re-run.  The f-string
variant is a control that does byte-identical work and must also restore.
"""
import pytest

pytestmark = [pytest.mark.upstream, pytest.mark.timeout(120)]

SETUP = (
    "import numpy as np\n"
    "import cash\n"
    "%cash_on\n"
    "%cash_badge print"
)

# Deterministic, comfortably above the ~10 ms cost floor so it is cached.
BASE = "data = np.linspace(0.0, 1.0, 2_000_000)"
EXPENSIVE = "result = float((data.reshape(2000, 1000) @ data.reshape(1000, 2000)).sum())"

# The regression trigger: the ``%`` operator opens a continuation line.
MULTILINE_PCT_PRINT = (
    'print("Asian call = %.4f\\n"\n'
    '      "European   = %.4f"\n'
    '      % (a, b))'
)

# Control: byte-identical work, f-string formatting (never touched the magic bug).
FSTRING_PRINT = 'print(f"Asian call = {a:.4f}\\nEuropean   = {b:.4f}")'


def _expensive_cell(printer: str) -> str:
    return (
        EXPENSIVE + "\n"
        "a = result\n"
        "b = result / 2.0\n"
        + printer
    )


def test_multiline_percent_print_cell_restores(nb_runner):
    """Expensive cell + multi-line ``%``-format print restores on isolated re-run.

    Without the fix, ``strip_magics`` deletes the ``      % (a, b))`` line, the
    simulator sees a fictional SyntaxError, and the cell recomputes every run
    (never RESTORED).
    """
    nb_runner.create_notebook([SETUP, BASE, _expensive_cell(MULTILINE_PCT_PRINT)])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "Asian call =" in nb_runner.get_output(3), "cell 3 print did not run"

    # Isolated re-run: inputs unchanged -> must be a cache hit.
    nb_runner.run_cell(3)
    out = nb_runner.get_output(3)
    assert "RESTORED" in out, (
        "cell with a multi-line %-format print did not restore from cache "
        f"(CAS-163). Badge output:\n{out}"
    )
    # And the formatted output is still produced.
    assert "Asian call =" in out


def test_fstring_print_cell_restores_control(nb_runner):
    """Control: the byte-identical f-string variant also restores."""
    nb_runner.create_notebook([SETUP, BASE, _expensive_cell(FSTRING_PRINT)])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "Asian call =" in nb_runner.get_output(3)

    nb_runner.run_cell(3)
    out = nb_runner.get_output(3)
    assert "RESTORED" in out, f"f-string control did not restore. Badge output:\n{out}"
    assert "Asian call =" in out
