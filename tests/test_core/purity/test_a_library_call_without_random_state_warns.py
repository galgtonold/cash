"""A library call that draws inside compiled code, left without a seed.

``train_test_split(X)``, ``KFold(shuffle=True)``, ``SGDClassifier()``,
``make_classification()`` and ``df.sample(3)`` in a cached body froze the
first split, fit or sample with no RANDOM-UNSEEDED: the source scan only
knows numpy / random / torch draws. The docs say an estimator with
``random_state=None`` counts as an unseeded draw; the notebook said so, the
decorator did not.
"""

from __future__ import annotations

import warnings

import pytest

from cash import Cash

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.model_selection import KFold, train_test_split

pytestmark = pytest.mark.core


def split(n):
    return train_test_split(list(range(10)), test_size=0.3)[1]


def fit(n):
    X = np.arange(60).reshape(20, 3).astype(float)
    return SGDClassifier().fit(X, np.arange(20) % 2).coef_.tolist()


def folds(n):
    return [t.tolist() for _, t in KFold(3, shuffle=True).split(np.arange(6))]


def dataset(n):
    return make_classification(n_samples=5)[0].sum()


def sample(n):
    df = pd.DataFrame({"a": range(10)})
    return df.sample(3).a.tolist()


def seeded(n):
    df = pd.DataFrame({"a": range(10)})
    return (
        train_test_split(list(range(10)), random_state=0)[1],
        [t.tolist() for _, t in KFold(3, shuffle=True, random_state=1).split(np.arange(6))],
        df.sample(3, random_state=0).a.tolist(),
        SGDClassifier(random_state=0),
    )


def unshuffled(n):
    return [t.tolist() for _, t in KFold(3).split(np.arange(6))], train_test_split([1, 2, 3, 4], shuffle=False)


def deterministic(n):
    return LinearRegression()


def generator_sample(n):
    import random

    r = random.Random(0)
    return r.sample(range(10), 3)


def _unseeded(tmp_path, fn):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        c.cache(fn)(1)  # the source is judged when it is decorated
    return [str(w.message) for w in rec if "RANDOM-UNSEEDED" in str(w.message)]


@pytest.mark.parametrize(
    ("fn", "said"),
    [
        (split, "train_test_split()"),
        (fit, "SGDClassifier()"),
        (folds, "KFold()"),
        (dataset, "make_classification()"),
        (sample, "df.sample()"),
    ],
    ids=lambda v: getattr(v, "__name__", None),
)
def test_an_unseeded_library_draw_warns(tmp_path, fn, said):
    found = _unseeded(tmp_path, fn)
    assert found and said in found[0] and "random_state" in found[0], found


@pytest.mark.parametrize("fn", [seeded, unshuffled, deterministic, generator_sample], ids=lambda f: f.__name__)
def test_a_seeded_or_deterministic_call_is_silent(tmp_path, fn):
    """Controls: a seed passed, shuffling off, no random_state at all, a
    seeded generator's own ``.sample``."""
    assert not _unseeded(tmp_path, fn)
