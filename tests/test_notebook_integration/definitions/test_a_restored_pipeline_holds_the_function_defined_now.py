"""A pipeline restored on a later Run All holds the notebook's functions as
the notebook defines them now, as a pipeline built again would.

The cell defining ``column_ratio`` and a transformer class is cheap and runs
again on every Run All, making new function and class objects under the same
names; the statement building the pipeline keeps its key and is restored. It
held the function from the first Run All, so pickling it (``joblib.dump``)
failed: "it's not the same object as __main__.column_ratio". When the
function's body is edited, the restored pipeline uses the new body.
"""

import pytest

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytest.importorskip("sklearn")

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

DATA = (
    "import time, pickle\nimport numpy as np\n"
    "X = np.random.RandomState(0).rand(4000, 4) + 1.0\n\n"
    "def pickles(value):\n"
    "    try:\n        return len(pickle.dumps(value)) > 0\n"
    "    except Exception as exc:\n        return repr(exc)"
)
RATIO = "def column_ratio(X):\n    return X[:, [0]] / X[:, [1]]\n\n"
DEFS = (
    "from sklearn.base import BaseEstimator, TransformerMixin\n"
    "from sklearn.cluster import KMeans\n"
    "{ratio}"
    "def ratio_name(function_transformer, feature_names_in):\n    return ['ratio']\n\n"
    "class ClusterSimilarity(BaseEstimator, TransformerMixin):\n"
    "    def __init__(self, n_clusters=5):\n        self.n_clusters = n_clusters\n"
    "    def fit(self, X, y=None):\n"
    "        self.kmeans_ = KMeans(self.n_clusters, n_init=2, random_state=0).fit(X)\n"
    "        return self\n"
    "    def transform(self, X):\n        return self.kmeans_.transform(X)\n"
)
BUILD = (
    "from sklearn.compose import ColumnTransformer\n"
    "from sklearn.pipeline import make_pipeline\n"
    "from sklearn.preprocessing import FunctionTransformer, StandardScaler\n"
    "preprocessing = ColumnTransformer([\n"
    "    ('ratio', make_pipeline(FunctionTransformer(column_ratio, feature_names_out=ratio_name), StandardScaler()), [0, 1]),\n"
    "    ('geo', ClusterSimilarity(), [2, 3]),\n"
    f"]).fit((time.sleep({ABOVE_PERSISTENCE_FLOOR_S * 2}), X)[1])"
)
CHECK = (
    "# @cash:no-cache\n"
    "held = preprocessing.named_transformers_['ratio'][0].func\n"
    "expected = StandardScaler().fit(column_ratio(X)).transform(column_ratio(X[:5]))[:, 0]\n"
    "ok = (held is column_ratio, pickles(preprocessing),\n"
    "      bool(np.allclose(preprocessing.transform(X[:5])[:, 0], expected)))"
)
OK = "(True, True, True)"


def _notebook():
    return [DATA, DEFS.format(ratio=RATIO), BUILD, CHECK]


def test_a_restored_pipeline_holds_the_function_defined_now(nb_runner):
    nb_runner.create_notebook(_notebook())
    nb_runner.start_kernel()
    for run in range(3):
        nb_runner.run_all()
        assert nb_runner.peek("ok") == OK, f"Run All {run + 1}"


def test_a_restored_pipeline_uses_the_edited_body(nb_runner):
    nb_runner.create_notebook(_notebook())
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    assert nb_runner.peek("ok") == OK
    nb_runner.set_cell_source(2, DEFS.format(ratio=RATIO.replace("X[:, [1]]", "X[:, [1]] ** 2")))
    nb_runner.run_all()
    assert nb_runner.peek("ok") == OK, "after the edit"
    nb_runner.run_all()
    assert nb_runner.peek("ok") == OK, "a Run All after the edit"
