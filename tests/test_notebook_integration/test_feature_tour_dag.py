"""The feature tour's headline claim: change one thing, recompute one thing.

`examples/try_cash_binder.ipynb` is the front door -- the README, the docs
index and the quickstart all link to it. Its centrepiece is a lattice:

    panel  ->  cluster_feats (shared, expensive)
                     |
        +------------+------------+
        v            v            v
    lasso        boosted       neural net    (siblings, own hyperparameter each)
        +------------+------------+
                     v
                leaderboard

Three claims, one per test:

1. Editing ONE model's hyperparameter re-runs that model and the leaderboard,
   and leaves the other two models AND the shared upstream cached.
2. Editing `score_fit`, which all three models call, invalidates all three --
   while the k-means stage, which does not call it, stays cached.
3. Editing `N_CLUSTERS` cascades through everything.

None of this was covered before the tour was rebuilt around it; it had to be
verified by hand. The control assertion in test 1 -- `feats` stays CACHED --
is what gives the other two meaning: a bug that simply invalidated everything
on any edit would keep the rest of the suite green while making the tour's
whole point false.

Toy-sized on purpose. The real tour is ~19s cold and each arm needs three
runs; this replica exercises the same SHAPE in seconds. Because the toys are
tiny they fall below cash's "too cheap to cache" floor, so the thresholds are
pinned -- the real tour's 1.6-2.6s nodes clear the floor on their own.

Note for anyone tempted to "check" this file by disabling cash: you cannot.
`start_kernel()` execs `%cash_on` itself on both the fresh-boot and
warm-reuse paths, so the SETUP cell's copy is inert, and `with_cash=False`
records intent rather than outcome. What keeps these tests honest instead is
that they disagree on purpose: the cascade test expects `feats` EXECUTED and
the sibling test expects it CACHED. One reader, one node, two opposite
verdicts -- they cannot both pass unless cache state is really being read.

The helpers below are deliberately this file's OWN, with their own names and
their own tiny sizes -- it does not read or import anything from
``examples/try_cash_binder.ipynb``. What is under test is the dependency
shape, so the test stays green when the tour retunes a knob, and goes red
when cash stops tracking the shape.

Asserted on cache STATE per node, never on wall clock.
"""
import pytest
from conftest import CASH_TEST_PIN_THRESHOLDS, shows_cached, shows_executed

pytestmark = pytest.mark.timeout(180)

SETUP = (
    "import cash\n"
    "%cash_on\n"
    "%cash_badge print\n"
    + CASH_TEST_PIN_THRESHOLDS +
    "import numpy as np"
)

# Its own cell, ABOVE the models that call it -- both because that is the cell
# the tour's section 5 edits without running, and because cash does not see
# functions defined below a call site.
SCORER = '''
def score_fit(pred, actual):
    return float(np.sqrt(np.mean((pred - actual) ** 2)))
'''

HELPERS = '''
def kmeans_features(X, k, iters=6, seed=7):
    r = np.random.default_rng(seed)
    C = X[r.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        d = (X**2).sum(1)[:, None] - 2 * X @ C.T + (C**2).sum(1)[None, :]
        lab = d.argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = X[m].mean(0)
    return np.sqrt(np.maximum(d, 0))


def lasso_cd(X, y, alpha, sweeps=8):
    w = np.zeros(X.shape[1]); r = y - X @ w
    norms = (X**2).sum(0) + 1e-9
    for _ in range(sweeps):
        for j in range(X.shape[1]):
            r += X[:, j] * w[j]
            rho = X[:, j] @ r
            w[j] = np.sign(rho) * max(abs(rho) - alpha * len(X), 0) / norms[j]
            r -= X[:, j] * w[j]
    return score_fit(X @ w, y)


def boost_stumps(X, y, rounds, lr=0.1, cuts_n=4):
    resid = y.astype(float).copy()
    cuts = np.percentile(X, np.linspace(20, 80, cuts_n), axis=0)
    for _ in range(rounds):
        best = (np.inf, 0, 0.0, 0.0, 0.0)
        for j in range(X.shape[1]):
            col = X[:, j]
            for c in cuts[:, j]:
                m = col <= c
                if not m.any() or m.all():
                    continue
                lo, hi = resid[m].mean(), resid[~m].mean()
                sse = ((resid[m] - lo) ** 2).sum() + ((resid[~m] - hi) ** 2).sum()
                if sse < best[0]:
                    best = (sse, j, c, lo, hi)
        _, j, c, lo, hi = best
        m = X[:, j] <= c
        resid[m] -= lr * lo; resid[~m] -= lr * hi
    return score_fit(y - resid, y)


def mlp_fit(X, y, epochs, hidden=6, lr=0.05, seed=3):
    r = np.random.default_rng(seed)
    W1 = r.standard_normal((X.shape[1], hidden)) * 0.1; b1 = np.zeros(hidden)
    W2 = r.standard_normal(hidden) * 0.1; b2 = 0.0
    n = len(X)
    for _ in range(epochs):
        H = np.tanh(X @ W1 + b1); pred = H @ W2 + b2; err = pred - y
        dH = np.outer(err, W2) * (1 - H**2)
        W1 -= lr * (X.T @ dH / n); b1 -= lr * dH.mean(0)
        W2 -= lr * (H.T @ err / n); b2 -= lr * err.mean()
    return score_fit(np.tanh(X @ W1 + b1) @ W2 + b2, y)
'''

PANEL = """
rng = np.random.default_rng(0)
panel = rng.standard_normal((400, 6))
target = panel @ np.array([1.0, -2.0, 0.5, 0.0, 0.3, -0.7]) + 0.5 * rng.standard_normal(400)
print("panel", panel.shape)
"""

N_CLUSTERS = "N_CLUSTERS = 5"
FEATS = """
cluster_feats = kmeans_features(panel, N_CLUSTERS)
design = np.hstack([panel, cluster_feats])
print("design", design.shape)
"""

LASSO_ALPHA = "LASSO_ALPHA = 0.05"
LASSO = """
lasso_rmse = lasso_cd(design, target, LASSO_ALPHA)
print("lasso", round(lasso_rmse, 4))
"""

BOOST_ROUNDS = "BOOST_ROUNDS = 6"
BOOST = """
boost_rmse = boost_stumps(design, target, BOOST_ROUNDS)
print("boost", round(boost_rmse, 4))
"""

MLP_EPOCHS = "MLP_EPOCHS = 40"
MLP = """
mlp_rmse = mlp_fit(design, target, MLP_EPOCHS)
print("mlp", round(mlp_rmse, 4))
"""

BOARD = """
leaderboard = sorted([("lasso", lasso_rmse), ("boost", boost_rmse), ("mlp", mlp_rmse)],
                     key=lambda kv: kv[1])
print("winner", leaderboard[0][0])
"""

CELLS = [SETUP, SCORER, HELPERS, PANEL, N_CLUSTERS, FEATS,
         LASSO_ALPHA, LASSO, BOOST_ROUNDS, BOOST, MLP_EPOCHS, MLP, BOARD]

# 1-based cell numbers -- NotebookTestRunner is 1-based, not 0-based.
C_SCORER = 2
C_CLUSTERS, C_FEATS = 5, 6
C_ALPHA, C_LASSO = 7, 8
C_BOOST_N, C_BOOST = 9, 10
C_MLP_N, C_MLP = 11, 12
C_BOARD = 13


def _state(runner):
    """Cache state per node, read off the badge each cell printed."""
    out = {}
    for name, n in [("feats", C_FEATS), ("lasso", C_LASSO), ("boost", C_BOOST),
                    ("mlp", C_MLP), ("board", C_BOARD)]:
        raw = runner.get_raw_output(n)
        out[name] = ("CACHED" if shows_cached(raw)
                     else "EXECUTED" if shows_executed(raw) else "?")
    return out


def _warmed(nb_runner):
    """A notebook whose whole lattice is cached and verified cached."""
    r = nb_runner.create_notebook(CELLS)
    r.start_kernel()
    r.run_all()
    r.run_all()
    warm = _state(r)
    assert all(v == "CACHED" for v in warm.values()), (
        f"the lattice did not warm up; every node should be CACHED on the "
        f"second run: {warm}")
    return r


def test_editing_one_hyperparameter_leaves_siblings_cached(nb_runner):
    """Section 3, the tour's centrepiece."""
    r = _warmed(nb_runner)

    r.set_cell_source(C_ALPHA, "LASSO_ALPHA = 0.5")
    r.run_all()
    after = _state(r)

    assert after["lasso"] == "EXECUTED", f"lasso's own knob changed: {after}"
    assert after["board"] == "EXECUTED", f"leaderboard consumes lasso: {after}"
    # The control arm. Without these three, a bug that invalidated everything
    # on any edit would pass -- and the tour's whole point would be false.
    assert after["boost"] == "CACHED", f"boost is a SIBLING: {after}"
    assert after["mlp"] == "CACHED", f"mlp is a SIBLING: {after}"
    assert after["feats"] == "CACHED", f"shared upstream is untouched: {after}"


def test_editing_the_shared_scoring_helper_invalidates_every_model(nb_runner):
    """Section 5. `score_fit` is called by all three model helpers, so editing
    it invalidates three cached results at once -- without the reader touching
    any model cell. k-means does NOT call it, so it must stay cached: that is
    what makes this a call-graph demonstration rather than a blanket cascade.
    """
    r = _warmed(nb_runner)

    r.set_cell_source(
        C_SCORER,
        "\ndef score_fit(pred, actual):\n"
        "    return float(np.mean(np.abs(pred - actual)))\n",
    )
    r.run_all()
    after = _state(r)

    for node in ("lasso", "boost", "mlp"):
        assert after[node] == "EXECUTED", f"{node} calls score_fit: {after}"
    assert after["feats"] == "CACHED", (
        f"k-means does not call score_fit and must stay cached: {after}")


def test_editing_the_shared_upstream_cascades(nb_runner):
    """Section 4. Everything descends from N_CLUSTERS, so everything goes."""
    r = _warmed(nb_runner)

    r.set_cell_source(C_CLUSTERS, "N_CLUSTERS = 8")
    r.run_all()
    after = _state(r)

    for node in ("feats", "lasso", "boost", "mlp", "board"):
        assert after[node] == "EXECUTED", f"{node} descends from N_CLUSTERS: {after}"
