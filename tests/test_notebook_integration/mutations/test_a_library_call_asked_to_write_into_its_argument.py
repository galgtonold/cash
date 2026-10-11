"""``np.median(a, overwrite_input=True)`` sorts ``a`` part way in place, and
a cached run of it must leave ``a`` sorted as a plain run does.

numpy's reductions are known to only read their arguments, so the statement
was cached without watching ``a``: the second Run All's hit skipped the
partial sort and the cell below printed the old order, on every later Run
All too. The call asks for the write (``overwrite_input=True``), so ``a`` is
watched and the statement changes it each run. Bare or with its result kept.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = "import cash\n%cash_on\nimport numpy as np\na = np.random.default_rng(0).random(4_000_000)"
SHOW = "print('A', a[:4].tolist())"


def _plain_order():
    np = pytest.importorskip("numpy")
    a = np.random.default_rng(0).random(4_000_000)
    np.median(a, overwrite_input=True)
    return f"A {a[:4].tolist()}"


@pytest.mark.parametrize(
    "call", ["np.median(a, overwrite_input=True)", "m = np.median(a, overwrite_input=True)"], ids=["bare", "kept"]
)
def test_run_all_again_keeps_the_partial_sort(nb_runner, call):
    expected = _plain_order()
    nb_runner.create_notebook([SETUP, call, SHOW])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert expected in nb_runner.get_output(3), nb_runner.get_raw_output(3)
    for _ in range(2):
        nb_runner.run_all()
        assert expected in nb_runner.get_output(3), (
            "a Run All replayed the median without its in-place sort:\n" + nb_runner.get_raw_output(3)
        )
        assert nb_runner.peek("a[:4].tolist()") == expected.removeprefix("A ")
