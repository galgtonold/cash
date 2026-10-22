"""A categorical of objects that a list also holds keeps them on a hit.

``df = pd.DataFrame({'ck': pd.Categorical([ks[i % 3] for i in range(n)])})``
holds ``ks``'s objects among its categories, which the share check did not
read: no holder was found, and a hit in the same kernel handed back a frame
whose categories were copies, so ``ks[1].v.append(9)`` no longer showed in
``df``. ``ks`` is now found by its objects' counts and takes the restored
categories back at its own places.
"""

import pytest

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SLEEP = ABOVE_PERSISTENCE_FLOOR_S * 2
SETUP = "class K:\n    def __init__(self, v):\n        self.v = [v]\nks = [K(i) for i in range(3)]"


@pytest.mark.parametrize(
    "make, read",
    [
        ("df = pd.DataFrame({'ck': pd.Categorical([ks[i % 3] for i in range(100_000)])})", "df['ck']"),
        ("s = pd.Series(pd.Categorical([ks[i % 3] for i in range(100_000)]))", "s"),
    ],
    ids=["frame", "series"],
)
def test_the_served_categorical_holds_the_lists_objects(nb_runner, make, read):
    cells = [
        "import cash\n%cash_on\nimport time\nimport pandas as pd",
        SETUP,
        f"{make}\ntime.sleep({SLEEP})",
        f"ks[1].v.append(9)\nprint('R', {read}.iloc[1] is ks[1], {read}.iloc[1].v)",
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    for restart in (False, False, True):
        if restart:
            nb_runner.restart()
        nb_runner.run_all()
        assert "R True [1, 9]" in nb_runner.get_output(4), nb_runner.get_output(4)
