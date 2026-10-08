"""A cached statement restored on a later Run All hands back objects of the
class the notebook defines now, not of the one the first run defined.

The class cell is cheap and re-runs on every Run All, defining a new class
object under the same name; the statement building the objects keeps its key
and is restored from RAM. Plain Jupyter re-runs both, so ``isinstance``,
``==`` and pickling work, and a frame of such objects is new each run.
"""

import pytest

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = [pytest.mark.core]


@pytest.mark.timeout(60)
def test_a_restored_object_is_an_instance_of_the_class_defined_now(nb_runner):
    nb_runner.create_notebook(
        [
            "import time, dataclasses, pickle\nimport cash",
            "@dataclasses.dataclass\nclass Fit:\n    w: list",
            f"def train(n):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S * 2})\n    return Fit([1.0] * n)\n\n"
            f"@cash.cache\ndef train_dec(n):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S * 2})\n    return Fit([1.0] * n)",
            "m = train(3)\nd = train_dec(3)",
            "ok = (isinstance(m, Fit), m == Fit([1.0] * 3), isinstance(d, Fit), len(pickle.dumps(m)) > 0)",
        ]
    )
    nb_runner.start_kernel()
    for run in range(2):
        nb_runner.run_all()
        assert nb_runner.peek("ok") == "(True, True, True, True)", f"Run All {run + 1}"


@pytest.mark.timeout(60)
def test_changes_to_a_frame_of_such_objects_do_not_pile_up(nb_runner):
    nb_runner.create_notebook(
        [
            "import time\nimport pandas as pd",
            "class Fit:\n    def __init__(self, w):\n        self.w = w",
            f"def build():\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S * 2})\n"
            "    return pd.DataFrame({'m': [Fit([1]), Fit([2])], 'n': [1, 2]})",
            "df = build()",
            "df['m'].iloc[0].w.append(9)\nw = list(df['m'].iloc[0].w)",
        ]
    )
    nb_runner.start_kernel()
    for run in range(3):
        nb_runner.run_all()
        assert nb_runner.peek("w") == "[1, 9]", f"Run All {run + 1}"
