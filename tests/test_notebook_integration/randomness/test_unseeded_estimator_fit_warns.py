"""Warn when caching an UNSEEDED estimator fit as a frozen replay.

A bare ``estimator.fit(X, y)`` is cached. But cash's AST randomness detector cannot see the
randomness inside sklearn's compiled ``.fit()`` (bootstrap sampling, feature
subsampling, weight init). An estimator built WITHOUT a ``random_state`` therefore
gets cached and frozen on re-run with no warning of any kind -- the notebook-path
analogue of the inline unseeded-randomness hazard cash already warns about (and of the
object-cache hazard the unseeded-randomness check covers on the decorator path). Two genuine fits
differ; the cached fit is a replay.

The fix is advisory only -- it does NOT change cacheability. An unseeded
fit still caches, it just announces that the cached fit is frozen: at COMPUTE time
("Unseeded randomness detected ... frozen replay") and, on a cache hit, at RESTORE
time ("restored from cache ... a replay"), through the SAME
``CashRandomnessWarning`` path the inline warning uses so existing filters catch it.

The warning is about a FROZEN cached fit, so it is scoped to the path that
actually freezes one. A fit under ``# @cash:no-cache`` re-executes every run --
nothing is replayed, so there is nothing to warn about
(``test_an_uncached_fit_does_not_warn``).

Real-kernel end of the contract: the warning must reach the notebook's cell
output through a live kernel. The seeded / deterministic / allow-random controls
are what make the positive test discriminating rather than a detector that shouts
at every ``.fit()`` it sees.
"""

import pytest

from tests._nbharness.badge import shows_cached

pytest.importorskip("sklearn")

pytestmark = [pytest.mark.libraries, pytest.mark.timeout(240)]

WARNING_CLASS = "CashRandomnessWarning"
DETECTED_TEXT = "Unseeded randomness detected"
RESTORED_TEXT = "restored from cache"

SETUP = (
    "import numpy as np\n"
    "from sklearn.ensemble import RandomForestClassifier\n"
    "from sklearn.datasets import make_classification\n"
    "import cash\n"
    "%cash_on\n"
    "%cash_badge print"
)
DATA = "X, y = make_classification(n_samples=6000, n_features=20, random_state=0)"


def test_unseeded_estimator_fit_warns(nb_runner):
    """An opted-in ``clf.fit(X, y)`` on an UNSEEDED estimator warns at compute
    time, and the isolated re-run's restore announces the replay.

    Fails WITHOUT the fix (the fit is cached silently);
    passes with it. ``RandomForestClassifier(n_estimators=160)`` carries no
    ``random_state``, so ``get_params()['random_state']`` is ``None`` -> unseeded.
    """
    nb_runner.create_notebook(
        [
            SETUP,
            DATA,
            "clf = RandomForestClassifier(n_estimators=160)",  # UNSEEDED: no random_state
            "clf.fit(X, y)\nprint('fitted', clf.n_estimators)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()

    cold = nb_runner.get_raw_output(4)
    assert WARNING_CLASS in cold, f"unseeded fit did not warn on cold compute: {cold!r}"
    assert DETECTED_TEXT in cold, cold
    assert "frozen replay" in cold, cold
    # The escape hatch must be discoverable from the warning itself.
    assert "@cash:allow-random" in cold, cold
    # The statement still ran (the warning is advisory, not a block).
    assert "fitted 160" in nb_runner.get_output(4)

    # Isolated re-run: the fit restores, and the restore announces the replay.
    nb_runner.run_cell(4)
    warm = nb_runner.get_raw_output(4)
    assert shows_cached(nb_runner.get_output(4)), (
        f"unseeded fit did not restore on warm re-run: {nb_runner.get_output(4)!r}"
    )
    assert RESTORED_TEXT in warm, f"restore did not announce the replay: {warm!r}"
    assert "replay" in warm, warm


def test_seeded_estimator_fit_does_not_warn(nb_runner):
    """Control: ``RandomForestClassifier(random_state=42)`` is reproducible, so it
    must stay quiet. This is what makes the positive test discriminating."""
    nb_runner.create_notebook(
        [
            SETUP,
            DATA,
            "clf = RandomForestClassifier(n_estimators=40, random_state=42)",  # SEEDED
            "clf.fit(X, y)\nprint('fitted', clf.n_estimators)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()

    out = nb_runner.get_raw_output(4)
    assert WARNING_CLASS not in out, f"seeded fit should not warn: {out!r}"
    assert DETECTED_TEXT not in out, out
    assert "fitted 40" in nb_runner.get_output(4)


def test_deterministic_estimator_fit_does_not_warn(nb_runner):
    """Control: ``LinearRegression()`` has NO ``random_state`` param, so its fit is
    deterministic -- ``'random_state' not in get_params()`` -> no warning."""
    nb_runner.create_notebook(
        [
            "import numpy as np\n"
            "from sklearn.linear_model import LinearRegression\n"
            "from sklearn.datasets import make_regression\n"
            "import cash\n"
            "%cash_on\n"
            "%cash_badge print",
            "X, y = make_regression(n_samples=6000, n_features=20, random_state=0)",
            "reg = LinearRegression()",
            "reg.fit(X, y)\nprint('coef', reg.coef_.shape[0])",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()

    out = nb_runner.get_raw_output(4)
    assert WARNING_CLASS not in out, f"deterministic fit should not warn: {out!r}"
    assert DETECTED_TEXT not in out, out
    assert "coef 20" in nb_runner.get_output(4)


def test_allow_random_suppresses_unseeded_estimator_fit(nb_runner):
    """``# @cash:allow-random`` on a cached unseeded fit suppresses the warning
    -- the documented opt-out has to cover this detection too: the fit
    caches AND stays quiet.
    """
    nb_runner.create_notebook(
        [
            SETUP,
            DATA,
            "clf = RandomForestClassifier(n_estimators=40)",  # UNSEEDED
            "# @cash:allow-random\nclf.fit(X, y)\nprint('fitted', clf.n_estimators)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()

    out = nb_runner.get_raw_output(4)
    assert WARNING_CLASS not in out, f"allow-random should suppress the warning: {out!r}"
    assert DETECTED_TEXT not in out, out
    assert "fitted 40" in nb_runner.get_output(4)


def test_a_bare_fit_warns_without_the_directive(nb_runner):
    """The identical notebook to ``test_unseeded_estimator_fit_warns`` minus the
    old directive: the bare fit is cached all the same, so it warns all the same."""
    nb_runner.create_notebook(
        [
            SETUP,
            DATA,
            "clf = RandomForestClassifier(n_estimators=40)",  # UNSEEDED
            "clf.fit(X, y)\nprint('fitted', clf.n_estimators)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()

    out = nb_runner.get_raw_output(4)
    assert DETECTED_TEXT in out, f"a cached unseeded bare fit did not warn: {out!r}"
    assert "fitted 40" in nb_runner.get_output(4)


def test_an_uncached_fit_does_not_warn(nb_runner):
    """A fit under ``# @cash:no-cache`` stays quiet, on its re-run too.

    On the re-run the fit's receiver is pre-routed as a mutation and joins the
    statement's outputs, which the inline-fit check could read as a cached
    fitted estimator: "cash caches the fit ... frozen replay" and a
    ``[random: unseeded]`` badge for a fit that re-runs.
    """
    nb_runner.create_notebook(
        [
            SETUP,
            DATA,
            "clf = RandomForestClassifier(n_estimators=40)",  # UNSEEDED
            "# @cash:no-cache\nclf.fit(X, y)\nprint('fitted', clf.n_estimators)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()

    out = nb_runner.get_raw_output(4)
    assert "fitted 40" in nb_runner.get_output(4)
    assert not shows_cached(out), out
    assert DETECTED_TEXT not in out, f"a re-run of an uncached bare fit claimed a frozen replay: {out!r}"
    assert "frozen replay" not in out, out
    assert "random: unseeded" not in out, out
