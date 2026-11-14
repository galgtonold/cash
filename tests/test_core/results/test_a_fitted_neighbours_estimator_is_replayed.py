"""A cached ``NearestNeighbors.fit``, or a resampler's ``fit_resample`` that
fits one (imbalanced-learn's ``SMOTE`` keeps it as ``nn_k_``), is stored
with the attributes it binds, and a hit leaves the estimator as a plain run
does.

The fitted estimator holds a scikit-learn ``KDTree`` or ``BallTree``, which
holds a ``DistanceMetric``: Cython classes with a ``__reduce__`` of their
own, whose pickle carries all they hold. They hold their arrays through
Cython memoryviews the garbage collector does not visit, so the share check
reads those arrays from their state instead. A tree built over the array
handed in is stored with a reference to it, and a hit builds it over the
array the caller hands in: a later write into that array reaches the tree
as in a plain run. A float32 tree on the Euclidean metric loses its
Euclidean flag in its pickle and still runs every time.
"""

from __future__ import annotations

import pickle
import textwrap
import warnings

import numpy as np
import pytest

from tests._scripts import run_python

sklearn_neighbors = pytest.importorskip("sklearn.neighbors")
NearestNeighbors = sklearn_neighbors.NearestNeighbors
KDTree = sklearn_neighbors.KDTree
BallTree = sklearn_neighbors.BallTree

#: Kept by the module: what a function binds that is also held here is held
#: by something outside the call.
KEPT: list = []


def _data(rows=200, seed=0):
    return np.random.default_rng(seed).random((rows, 3))


def _same_neighbours(a, b, X):
    for left, right in zip(a.kneighbors(X[:7]), b.kneighbors(X[:7])):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize(
    ("kind", "metric"),
    [(KDTree, "euclidean"), (KDTree, "chebyshev"), (BallTree, "minkowski"), (BallTree, "manhattan")],
)
def test_a_trees_pickle_carries_all_it_holds(kind, metric):
    X = _data(80)
    tree = kind(X, leaf_size=5, metric=metric)
    tree.query(X[:3], k=2)
    back = pickle.loads(pickle.dumps(tree))
    for held, restored in zip(tree.__getstate__(), back.__getstate__()):
        if isinstance(held, np.ndarray):
            np.testing.assert_array_equal(held, restored)
        elif type(held).__module__ == "sklearn.metrics._dist_metrics":
            assert type(held) is type(restored)
            for part, again in zip(held.__getstate__(), restored.__getstate__()):
                np.testing.assert_array_equal(part, again)
        else:
            assert held == restored
    for left, right in zip(tree.query(X[:9], k=4), back.query(X[:9], k=4)):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("algorithm", ["kd_tree", "ball_tree", "brute"])
def test_a_nearest_neighbors_fit_is_replayed(disk_cash, algorithm):
    runs = []

    @disk_cash.cache
    def fit(nn, X):
        runs.append(1)
        return nn.fit(X)

    X = _data()
    plain = NearestNeighbors(n_neighbors=4, algorithm=algorithm).fit(X)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            nn = NearestNeighbors(n_neighbors=4, algorithm=algorithm)
            assert fit(nn, X.copy()) is nn
            assert list(vars(nn)) == list(vars(plain))
            assert type(nn._tree) is type(plain._tree)
            _same_neighbours(nn, plain, X)
    assert len(runs) == 1
    assert fit.cache_info()["hits"] == 2
    assert not [w for w in caught if "in place" in str(w.message)], [str(w.message) for w in caught]


@pytest.mark.parametrize("algorithm", ["kd_tree", "ball_tree"])
def test_a_hits_tree_is_built_over_the_callers_array(disk_cash, algorithm):
    """After a hit the tree views the array handed in, as a plain fit's
    does: a write into it afterwards gives the plain run's neighbours."""

    @disk_cash.cache
    def fit(nn, X):
        return nn.fit(X)

    X = _data()
    fit(NearestNeighbors(n_neighbors=4, algorithm=algorithm), X.copy())
    plain_X, hit_X = X.copy(), X.copy()
    plain = NearestNeighbors(n_neighbors=4, algorithm=algorithm).fit(plain_X)
    nn = NearestNeighbors(n_neighbors=4, algorithm=algorithm)
    assert fit(nn, hit_X) is nn
    assert fit.cache_info()["hits"] == 1
    assert nn._fit_X is hit_X
    assert nn._tree.data.base is hit_X
    for written in (plain_X, hit_X):
        written[:20] += 5.0
    _same_neighbours(nn, plain, X)


def test_a_smote_fit_resample_is_replayed(disk_cash):
    over_sampling = pytest.importorskip("imblearn.over_sampling")
    runs = []

    @disk_cash.cache
    def resample(sampler, X, y):
        runs.append(1)
        return sampler.fit_resample(X, y)

    X = _data(400)
    y = np.array([0] * 320 + [1] * 80)
    plain = over_sampling.SMOTE(random_state=0)
    plain_X, plain_y = plain.fit_resample(X, y)
    for _ in range(3):
        sampler = over_sampling.SMOTE(random_state=0)
        got_X, got_y = resample(sampler, X.copy(), y.copy())
        np.testing.assert_array_equal(got_X, plain_X)
        np.testing.assert_array_equal(got_y, plain_y)
        assert list(vars(sampler)) == list(vars(plain))
        assert sampler.sampling_strategy_ == plain.sampling_strategy_
        assert type(sampler.nn_k_._tree) is type(plain.nn_k_._tree)
        _same_neighbours(sampler.nn_k_, plain.nn_k_, X)
    assert len(runs) == 1


class Holder:
    def __init__(self):
        self.k = 1


def test_a_tree_whose_array_something_else_holds_runs_every_time(disk_cash):
    """The array is held through the tree's memoryview, which the garbage
    collector does not visit: the module's hold on it is still seen."""
    runs = []

    @disk_cash.cache
    def build(holder, X):
        runs.append(1)
        data = X * 2.0
        KEPT.append(data)  # @cash:assume-safe
        holder.tree_ = KDTree(data)
        return 1

    try:
        for _ in range(2):
            build(Holder(), _data(30))
    finally:
        KEPT.clear()
    assert len(runs) == 2


def test_a_tree_over_a_view_of_the_argument_runs_every_time(disk_cash):
    """``KDTree(X[10:])`` views the argument through a memoryview the
    collector does not visit: a hit's tree would hold a copy, which a later
    write into ``X`` would not reach."""
    runs = []

    @disk_cash.cache
    def build(holder, X):
        runs.append(1)
        holder.tree_ = KDTree(X[10:])
        return 1

    for _ in range(2):
        holder, X = Holder(), _data(30)
        build(holder, X)
        assert np.shares_memory(holder.tree_.data.base, X)
    assert len(runs) == 2
    assert "shares memory with the argument 'X'" in build.explain(Holder(), _data(30)).details["why"]


def test_a_float32_tree_on_the_euclidean_metric_runs_every_time(disk_cash):
    kd_tree = pytest.importorskip("sklearn.neighbors._kd_tree")
    if not hasattr(kd_tree, "KDTree32"):
        pytest.skip("this scikit-learn has no float32 trees")
    runs = []

    @disk_cash.cache
    def build(holder, X):
        runs.append(1)
        holder.tree_ = kd_tree.KDTree32(X.astype(np.float32))
        return 1

    for _ in range(2):
        holder = Holder()
        build(holder, _data(30))
        assert holder.tree_.query(np.zeros((1, 3), np.float32))[1].shape == (1, 1)
    assert len(runs) == 2
    assert "without its Euclidean flag" in build.explain(Holder(), _data(30)).details["why"]


@pytest.mark.timeout(120)
def test_a_second_process_replays_the_fitted_estimator(tmp_path):
    script = tmp_path / "fit_nn.py"
    script.write_text(
        textwrap.dedent(
            """
            import os
            import sys
            import numpy as np
            import cash
            from sklearn.neighbors import NearestNeighbors

            c = cash.Cash(cache_dir=sys.argv[1], register_magic=False)
            LOG = sys.argv[2]

            @c.cache
            def fit(nn, X):
                fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe
                os.write(fd, b"ran\\n")
                os.close(fd)
                return nn.fit(X)

            X = np.random.default_rng(0).random((200, 3))
            plain = NearestNeighbors(n_neighbors=4).fit(X)
            for _ in range(2):
                nn = NearestNeighbors(n_neighbors=4)
                assert fit(nn, X.copy()) is nn
                assert list(vars(nn)) == list(vars(plain)), vars(nn)
                for a, b in zip(nn.kneighbors(X[:7]), plain.kneighbors(X[:7])):
                    assert np.array_equal(a, b)
            # A new array, equal to the first process's: the hit's tree views it.
            X = np.random.default_rng(0).random((200, 3))
            nn = NearestNeighbors(n_neighbors=4)
            assert fit(nn, X) is nn
            assert nn._fit_X is X and nn._tree.data.base is X
            X[:20] += 5.0
            plain_X = np.random.default_rng(0).random((200, 3))
            plain = NearestNeighbors(n_neighbors=4).fit(plain_X)
            plain_X[:20] += 5.0
            for a, b in zip(nn.kneighbors(X[:7]), plain.kneighbors(X[:7])):
                assert np.array_equal(a, b)
            print("ok")
            """
        ),
        encoding="utf-8",
    )
    log = tmp_path / "runs.log"
    for _ in range(2):
        out = run_python(script, tmp_path / ".cash", log, cwd=tmp_path)
        assert out.stdout.strip() == "ok", out.stderr
    assert log.read_text(encoding="utf-8").count("ran") == 1
