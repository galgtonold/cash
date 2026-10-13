"""``Ridge()`` without ``random_state`` is not an unseeded draw.

Ridge uses its ``random_state`` only with ``solver='sag'`` or ``'saga'``;
``LogisticRegression`` only with ``'sag'``, ``'saga'`` or ``'liblinear'``;
``Lasso`` only with ``selection='random'``; ``SVC`` only with
``probability=True``. With the defaults their fits are deterministic, yet
RANDOM-UNSEEDED said "Unseeded randomness detected" for each, at decoration
and again for the fitted estimator returned. The settings the call makes now
decide; where they cannot be read (a variable, ``PCA``'s data-dependent
``svd_solver='auto'``), the warning says what the draw depends on instead of
claiming one.
"""

from __future__ import annotations

import warnings

import pytest

from cash import Cash

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.svm import SVC

pytestmark = pytest.mark.core

X = np.arange(40, dtype=float).reshape(20, 2) % 7
y = np.arange(20, dtype=float) % 3
labels = (y > 0).astype(int)


def ridge_coef(n):
    return Ridge().fit(X, y).coef_.tolist()


def ridge_model(n):
    m = Ridge(alpha=2.0)
    m.fit(X, y)
    return m


def logistic(n):
    return LogisticRegression().fit(X, labels)


def lasso(n):
    return Lasso(alpha=0.1).fit(X, y)


def svc(n):
    return SVC().fit(X, labels)


def ridge_sag(n):
    return Ridge(solver="sag").fit(X, y)


def logistic_liblinear(n):
    return LogisticRegression(solver="liblinear").fit(X, labels)


def lasso_random(n):
    return Lasso(alpha=0.1, selection="random").fit(X, y).coef_.tolist()


def svc_probability(n):
    return SVC(probability=True).fit(X, labels).classes_.tolist()


def ridge_solver(solver):
    return Ridge(solver=solver).fit(X, y).coef_.tolist()


def pca(n):
    return PCA(1).fit(X).components_.tolist()


def _warnings(tmp_path, fn, arg=1):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        c.cache(fn)(arg)
    return [str(w.message) for w in rec if "RANDOM-UNSEEDED" in str(w.message)]


@pytest.mark.parametrize("fn", [ridge_coef, ridge_model, logistic, lasso, svc], ids=lambda f: f.__name__)
def test_a_setting_that_never_draws_is_silent(tmp_path, fn):
    assert not _warnings(tmp_path, fn)


@pytest.mark.parametrize(
    ("fn", "said"),
    [
        (ridge_sag, "Ridge()"),
        (logistic_liblinear, "LogisticRegression()"),
        (lasso_random, "Lasso()"),
        (svc_probability, "SVC()"),
    ],
    ids=lambda v: getattr(v, "__name__", None),
)
def test_a_setting_that_draws_still_warns(tmp_path, fn, said):
    found = _warnings(tmp_path, fn)
    assert found and "Unseeded randomness detected" in found[0] and said in found[0], found


def test_a_returned_estimator_that_draws_still_warns(tmp_path):
    found = _warnings(tmp_path, ridge_sag)
    assert any("returns a fitted estimator" in m for m in found), found


def test_a_setting_cash_cannot_read_says_what_the_draw_depends_on(tmp_path):
    """``solver`` is a variable here: whether this fit draws is not known
    when the function is decorated. The warning names the condition and
    claims no draw."""
    found = _warnings(tmp_path, ridge_solver, "cholesky")
    assert len(found) == 1, found
    assert "Unseeded randomness detected" not in found[0], found
    assert "Ridge uses random_state only with solver='sag' or 'saga'" in found[0], found


def test_a_data_dependent_setting_says_what_the_draw_depends_on(tmp_path):
    found = _warnings(tmp_path, pca)
    assert found and "Unseeded randomness detected" not in found[0], found
    assert "svd_solver='arpack' or 'randomized'" in found[0], found
