"""An error raised in a loop body is reported once, as the user's own.

One KeyError printed a ~100-line traceback three
times, through cash's internals (for_handler.py, call_unit.py). The control
structure handlers logged the exception they then hand back to the cell at
ERROR level with its traceback; the cell raises it and IPython prints it
again.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.loops]


@pytest.mark.parametrize(
    "body",
    [
        "out = []\nfor r in rows:\n    out.append(score(r))",
        "out = []\nif rows:\n    out.append(score(rows[0]))",
        "out = []\ntry:\n    out.append(score(rows[0]))\nfinally:\n    pass",
    ],
    ids=["for", "if", "try"],
)
def test_the_error_is_not_logged_through_cash(nb_runner, body):
    nb_runner.create_notebook(
        ["import cash\n%cash_on", "def score(d):\n    return d['missing']\nrows = [{'a': 1}, {'a': 2}]", body]
    )
    nb_runner.start_kernel()
    nb_runner.run_cell(1)
    nb_runner.run_cell(2)
    with pytest.raises(Exception) as raised:
        nb_runner.run_cell(3)
    assert getattr(raised.value, "ename", None) == "KeyError", raised.value
    shown = nb_runner.get_raw_output(3) + str(raised.value)
    assert "[CONTROL]" not in shown, shown
    assert "for_handler.py" not in shown and "if_handler.py" not in shown, shown


def test_the_traceback_shows_no_cash_frames(nb_runner):
    """The traceback IPython prints goes from the cell to the user's function:
    cash's call wrapper frames sat between them."""
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on",
            "def score(d):\n    return d['missing']\nrows = [{'a': 1}]",
            "vals = [score(r) for r in rows]",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_cell(1)
    nb_runner.run_cell(2)
    with pytest.raises(Exception):
        nb_runner.run_cell(3)
    import re

    tb = re.sub(
        r"\x1b\[[0-9;]*m",
        "",
        "\n".join(
            line
            for o in nb_runner.nb.cells[2].outputs
            if o.get("output_type") == "error"
            for line in o.get("traceback", [])
        ),
    )
    assert "return d['missing']" in tb, tb
    assert "call_unit.py" not in tb, tb
