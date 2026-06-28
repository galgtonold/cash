"""Editing an upstream cell must propagate to downstream consumers on re-run.

Core to cash's value: re-running a downstream cell after an upstream edit must
reflect the edit (re-deriving intermediate cells as needed), exactly as a clean
run-from-top would. Found solid by the 2026-06-28 correctness sweep.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream]


def _last(out: str) -> str:
    return out.strip().splitlines()[-1].strip() if out.strip() else ""


def test_function_body_edit_propagates(nb_runner):
    """Edit a function body upstream; a downstream caller reflects it."""
    nb_runner.create_notebook([
        "def f(x):\n    return x * 2",
        "y = f(10)\nprint(y)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _last(nb_runner.get_output(2)) == "20"
    nb_runner.set_cell_source(1, "def f(x):\n    return x * 3")
    nb_runner.run_cell(2)
    assert _last(nb_runner.get_output(2)) == "30", nb_runner.get_output(2)


def test_multihop_chain_edit_propagates(nb_runner):
    """a -> b -> c. Edit a, run ONLY c: must re-derive b then c."""
    nb_runner.create_notebook([
        "a = 2",
        "b = a * 10",
        "c = b + 1\nprint(c)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _last(nb_runner.get_output(3)) == "21"
    nb_runner.set_cell_source(1, "a = 5")
    nb_runner.run_cell(3)
    assert _last(nb_runner.get_output(3)) == "51", nb_runner.get_output(3)


def test_constant_edit_propagates_through_dataframe(nb_runner):
    nb_runner.create_notebook([
        "import pandas as pd\nscale = 2\ndf = pd.DataFrame({'v':[1,2,3]})",
        "df['s'] = df['v'] * scale",
        "print(df['s'].sum())",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _last(nb_runner.get_output(3)) == "12"
    nb_runner.set_cell_source(1, "import pandas as pd\nscale = 10\ndf = pd.DataFrame({'v':[1,2,3]})")
    nb_runner.run_cell(2)
    nb_runner.run_cell(3)
    assert _last(nb_runner.get_output(3)) == "60", nb_runner.get_output(3)
