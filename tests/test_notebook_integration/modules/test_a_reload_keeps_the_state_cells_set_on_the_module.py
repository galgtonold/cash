"""Editing a local module keeps the values the notebook's cells set on it.

``helper.SCALE = 5`` and ``helper.REGISTRY['a'] = 10`` in one cell; editing
an unrelated function in ``helper.py`` made cash reload the module, which put
both back to the file's defaults, and the next cell computed ``(3, {})``.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]

HELPER = (
    "import time\nSCALE = 1\nREGISTRY = {{}}\n"
    "def compute(v):\n    time.sleep(0.3)\n    return (v * SCALE, dict(REGISTRY))\n"
    "def unrelated():\n    return {n}\n"
)


def test_editing_an_unrelated_function_keeps_the_settings(nb_runner, tmp_path):
    path = tmp_path / "statehelper.py"
    path.write_text(HELPER.format(n=1), encoding="utf-8")
    nb_runner.create_notebook(
        [
            "import statehelper",
            "statehelper.SCALE = 5\nstatehelper.REGISTRY['a'] = 10",
            "z = statehelper.compute(3)\nprint('Z', z)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "Z (15, {'a': 10})" in nb_runner.get_output(3)

    path.write_text(HELPER.format(n=2), encoding="utf-8")
    nb_runner.run_cell(3)

    assert nb_runner.peek("z") == "(15, {'a': 10})", nb_runner.get_raw_output(3)


@pytest.mark.parametrize(
    "cells",
    [
        ["import statehelper", "k = 5", "statehelper.SCALE = k", "k = 9"],
        ["import statehelper, numpy as np\nrng = np.random.default_rng(0)", "statehelper.SCALE = int(rng.integers(100))"],
    ],
    ids=["an_input_rebound_since", "a_draw_in_the_setting"],
)
def test_a_setting_whose_inputs_moved_on_keeps_the_value_it_set(nb_runner, tmp_path, cells):
    """Run again after the reload, the setting read the ``k`` a later cell
    bound (``z`` came out 27), or drew anew from the generator."""
    path = tmp_path / "statehelper.py"
    path.write_text(HELPER.format(n=1), encoding="utf-8")
    cells = [*cells, "z = statehelper.compute(3)\nprint('Z', z)"]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = nb_runner.peek("z")

    path.write_text(HELPER.format(n=2), encoding="utf-8")
    nb_runner.run_cell(len(cells))
    assert nb_runner.peek("z") == first, nb_runner.get_raw_output(len(cells))
