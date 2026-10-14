"""An unseeded draw a cached function makes through code it calls.

The source scan reads only the decorated function's own body. A helper that
builds a scikit-learn ``ColumnTransformer`` around ``KBinsDiscretizer`` left at
``random_state=None`` subsamples from numpy's global stream once the data
passes 200,000 rows: the cached transformer was one random draw, two plain
runs differed, and nothing was said. A computed call that moves a global
random stream no seed stands behind is now reported, whatever drew.

A generator the helper makes for itself (``np.random.default_rng()``) is no
global stream, so the stream check cannot see it; for a ``frozen=True``
function returning an array, recomputing the same key after a clear and
getting a different result is reported instead (CACHE-NONDETERMINISTIC).
"""

from __future__ import annotations

import random
import warnings

import pytest

from cash import Cash
from cash.tracking.randomness import lineage

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

pytestmark = pytest.mark.core

_X = np.arange(60, dtype=float).reshape(20, 3)
_Y = np.arange(20, dtype=float)


@pytest.fixture(autouse=True)
def no_seed_in_force(monkeypatch):
    """No seed left by another test: no notebook ledger, nothing seeded."""
    monkeypatch.setattr(lineage, "_ACTIVE_SEED_EPOCHS", None)
    monkeypatch.setattr(lineage, "_PROCESS_SEEDED", set())
    monkeypatch.setattr(lineage, "_TORCH_SEEDED", set())


def _forest(seed):
    return RandomForestRegressor(n_estimators=3, random_state=seed).fit(_X, _Y).predict(_X).sum()


def forest_unseeded(n):
    return _forest(None)


def forest_seeded(n):
    return _forest(0)


def _seed_then_draw():
    np.random.seed(0)
    return np.random.rand(3).tolist()


def seeds_itself(n):
    return _seed_then_draw()


def _global_draw():
    return np.random.rand(3).tolist()


def draws_global(n):
    return _global_draw()


def _python_draw():
    return random.random()


def draws_python_random(n):
    return _python_draw()


def _local_generator():
    return np.random.default_rng().normal(size=4)


def local_generator(n):
    return _local_generator()


def _mean(n):
    return float(np.arange(n).mean())


def deterministic(n):
    return _mean(n)


def allowed_in_source(n):
    # @cash:allow-random
    return _global_draw()


def _unseeded_forest():
    return RandomForestRegressor(n_estimators=3).fit(_X, _Y)


def returns_unseeded_forest(n):
    return _unseeded_forest()


def _make_preprocessor(n_bins):
    return ColumnTransformer(
        [
            ("veh", KBinsDiscretizer(n_bins=n_bins, encode="onehot-dense", strategy="quantile"), ["a"]),
            ("num", StandardScaler(), ["b"]),
        ],
        sparse_threshold=0.0,
    )


def fit_preprocessor(train, n_bins):
    pre = _make_preprocessor(n_bins)
    pre.fit(train)
    return pre


def _call(tmp_path, fn, *args, **options):
    """Call *fn* cached, once; its RANDOM-* and CACHE-NONDETERMINISTIC messages."""
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        c.cache(**options)(fn)(*args or (1,))
    return [str(w.message) for w in rec if "RANDOM-" in str(w.message) or "NONDETERMINISTIC" in str(w.message)]


@pytest.mark.parametrize(
    ("fn", "stream"),
    [
        (forest_unseeded, "numpy's global random stream"),
        (draws_global, "numpy's global random stream"),
        (draws_python_random, "Python's global random stream"),
    ],
    ids=lambda v: getattr(v, "__name__", None),
)
def test_a_helper_that_draws_without_a_seed_warns(tmp_path, fn, stream):
    found = _call(tmp_path, fn)
    assert len(found) == 1, found
    assert "RANDOM-UNSEEDED" in found[0]
    assert f"drew from {stream} without a seed" in found[0]
    assert fn.__name__ in found[0]


@pytest.mark.parametrize("fn", [forest_seeded, seeds_itself, deterministic], ids=lambda f: f.__name__)
def test_a_seeded_or_deterministic_helper_is_silent(tmp_path, fn):
    """Controls: ``random_state=0``; a helper that seeds the stream itself
    before drawing; no draw at all."""
    assert not _call(tmp_path, fn)


def test_a_deterministic_helper_leaves_the_stream_where_it_was(tmp_path):
    before = np.random.get_state()[1].copy()
    assert not _call(tmp_path, deterministic)
    assert (np.random.get_state()[1] == before).all()


def test_a_seed_the_caller_set_silences_it(tmp_path):
    np.random.seed(3)
    assert not _call(tmp_path, draws_global)


def test_a_seed_set_to_none_inside_the_call_does_not(tmp_path):
    """``seed(None)`` re-randomises: a draw after it is unseeded."""
    np.random.seed(3)

    def reseeds_with_entropy(n):
        np.random.seed(None)
        return _global_draw()

    found = _call(tmp_path, reseeds_with_entropy)
    assert found and "drew from numpy's global random stream without a seed" in found[0], found


def test_allow_random_silences_it(tmp_path):
    assert not _call(tmp_path, draws_global, allow_random=True)


def test_the_allow_random_comment_silences_it(tmp_path):
    assert not _call(tmp_path, allowed_in_source)


def test_an_unseeded_estimator_returned_is_named_once(tmp_path):
    """Its own warning names the estimator; the stream check does not repeat it."""
    found = _call(tmp_path, returns_unseeded_forest)
    assert len(found) == 1, found
    assert "returns a fitted estimator with random_state=None" in found[0]


def test_a_transformer_left_unseeded_inside_a_helper_built_column_transformer_warns(tmp_path):
    """The case that slipped through: quantile bins on more rows than
    ``KBinsDiscretizer`` keeps (200,000), so it subsamples from numpy's
    global stream; the transformer is built in a helper."""
    rs = np.random.RandomState(1)
    train = pd.DataFrame({"a": rs.gamma(2.0, size=200_500), "b": rs.normal(size=200_500)})
    found = _call(tmp_path, fit_preprocessor, train, 8, frozen=True)
    assert len(found) == 1, found
    assert "fit_preprocessor" in found[0]
    assert "drew from numpy's global random stream without a seed" in found[0]


def test_a_local_unseeded_generator_is_not_seen_by_the_stream_check(tmp_path):
    """No global stream moves: the stream check is silent (a documented
    limit). The recompute check below is what catches it."""
    assert not _call(tmp_path, local_generator)


def _clear_and_recompute(tmp_path, fn, **options):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    cached = c.cache(frozen=True, **options)(fn)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cached(1)
        cached.cache_clear()
        cached(1)
    return [str(w.message) for w in rec if "CACHE-NONDETERMINISTIC" in str(w.message)]


def test_a_result_that_differs_when_recomputed_after_a_clear_warns(tmp_path):
    found = _clear_and_recompute(tmp_path, local_generator)
    assert len(found) == 1, found
    assert "local_generator" in found[0] and "differs" in found[0]


def test_a_result_that_is_the_same_when_recomputed_is_silent(tmp_path):
    """Control: the same frozen array computed again."""

    def same_array(n):
        return np.arange(5.0)

    assert not _clear_and_recompute(tmp_path, same_array)


def test_allow_random_silences_the_recompute_check(tmp_path):
    assert not _clear_and_recompute(tmp_path, local_generator, allow_random=True)
