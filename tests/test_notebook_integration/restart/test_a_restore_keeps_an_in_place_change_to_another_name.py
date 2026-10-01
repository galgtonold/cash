"""Restoring a cell after a restart keeps what it changed in another name.

``b = a.astype(np.float64, copy=False)`` hands back ``a`` itself, so
``b += 1`` changes ``a`` too. A restore that jumps to the saved last ``b`` and
skips those two statements would leave ``a`` at zeros. Only calls known to
copy may start a run a restore jumps over.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]


def _cells(first):
    return [
        "import cash\n%cash_on",
        "import numpy as np, time\ndef slow(x):\n    time.sleep(0.5)\n    return x * 2",
        "a = np.zeros(3)",
        f"b = {first}\nb += 1\nb = slow(b)",
        "print('A', a.tolist(), 'B', b.tolist())",
    ]


@pytest.mark.parametrize(
    "first",
    ["a.astype(np.float64, copy=False)", "np.asarray(a)"],
)
def test_an_alias_written_in_place_still_changes_its_source(nb_runner, first):
    nb_runner.create_notebook(_cells(first))
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "A [1.0, 1.0, 1.0] B [2.0, 2.0, 2.0]" in nb_runner.get_output(5)

    nb_runner.restart()
    nb_runner.run_all()
    out = nb_runner.get_output(5)
    assert "A [1.0, 1.0, 1.0] B [2.0, 2.0, 2.0]" in out, out
    assert nb_runner.peek("a.tolist()") == "[1.0, 1.0, 1.0]"


def test_a_copy_written_in_place_leaves_its_source_alone(nb_runner):
    """Positive control: with a real copy, ``a`` stays at zeros either way."""
    nb_runner.create_notebook(_cells("a.copy()"))
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.restart()
    nb_runner.run_all()
    out = nb_runner.get_output(5)
    assert "A [0.0, 0.0, 0.0] B [2.0, 2.0, 2.0]" in out, out
