"""An edit that makes two names one object re-runs the change made through it.

``d2 = df.copy()`` edited to ``d2 = df``, with ``d2['b'] = ...`` unchanged
below it: running only a cell that reads ``df`` left the change out, as
``d2['b'] = ...`` had changed only ``d2`` when it ran. The edited statement's
text says it binds ``df``'s object, and the simulation now links the two
there.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(300)]

SETUP = "import cash\n%cash_on\nimport time\nimport pandas as pd\ndef ok(v):\n    time.sleep(0.2)\n    return v"

CASES = {
    "frame": (
        "df = pd.DataFrame({'a': [1, 2]})",
        "d2 = df.copy()",
        "d2 = df",
        "d2['b'] = d2['a'] * 10",
        "print(df.columns.tolist())",
        "['a', 'b']",
    ),
    "list": ("x = [1]", "y = list(x)", "y = x", "y.append(2)", "print(x)", "[1, 2]"),
    "returns_its_argument": ("x = [1]", "y = list(x)", "y = ok(x)", "y.append(2)", "print(x)", "[1, 2]"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_cell_below_sees_the_change_through_the_new_alias(nb_runner, case):
    make, before, after, change, read, expected = CASES[case]
    nb_runner.create_notebook([SETUP, make, before, change, read])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(3, after)

    nb_runner.run_cells([5])

    assert expected in nb_runner.get_output(5), nb_runner.get_output(5)
    assert nb_runner.peek("d2 is df" if case == "frame" else "y is x") == "True"
