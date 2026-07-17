"""CAS-167: warn when caching an UNSEEDED estimator fit as a frozen replay.

``# @cash:cache-fit`` makes a bare ``estimator.fit(X, y)`` cacheable (CAS-138,
opt-in since CAS-170). But cash's AST randomness detector cannot see the
randomness inside sklearn's compiled ``.fit()`` (bootstrap sampling, feature
subsampling, weight init). An estimator built WITHOUT a ``random_state`` therefore
gets cached and frozen on re-run with no warning of any kind -- the notebook-path
analogue of the inline unseeded-randomness hazard CAS-135 warns about (and of the
object-cache hazard CAS-158 covers on the decorator path). Two genuine fits
differ; the cached fit is a replay.

The fix is advisory only -- it does NOT change cacheability. An opted-in unseeded
fit still caches, it just announces that the cached fit is frozen: at COMPUTE time
("Unseeded randomness detected ... frozen replay") and, on a cache hit, at RESTORE
time ("restored from cache ... a replay"), through the SAME
``CashRandomnessWarning`` path CAS-135 uses so existing filters catch it.

Every notebook here carries ``# @cash:cache-fit``: the warning is about a FROZEN
cached fit, so it is scoped to the path that actually freezes one. A bare fit
without the directive re-executes every run (CAS-170) -- nothing is replayed, so
there is nothing to warn about (``test_default_bare_fit_does_not_warn``).

Real-kernel end of the contract: the warning must reach the notebook's cell
output through a live kernel. The seeded / deterministic / allow-random controls
are what make the positive test discriminating rather than a detector that shouts
at every ``.fit()`` it sees.
"""
import pytest

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

    Fails WITHOUT the CAS-167 fix (the cache-fit path caches the fit silently);
    passes with it. ``RandomForestClassifier(n_estimators=160)`` carries no
    ``random_state``, so ``get_params()['random_state']`` is ``None`` -> unseeded.
    """
    nb_runner.create_notebook([
        SETUP,
        DATA,
        "clf = RandomForestClassifier(n_estimators=160)",   # UNSEEDED: no random_state
        "# @cash:cache-fit\nclf.fit(X, y)\nprint('fitted', clf.n_estimators)",
    ])
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
    assert "RESTORED" in nb_runner.get_output(4), (
        f"unseeded fit did not restore on warm re-run: {nb_runner.get_output(4)!r}"
    )
    assert RESTORED_TEXT in warm, (
        f"restore did not announce the replay: {warm!r}"
    )
    assert "replay" in warm, warm


def test_seeded_estimator_fit_does_not_warn(nb_runner):
    """Control: ``RandomForestClassifier(random_state=42)`` is reproducible, so it
    must stay quiet. This is what makes the positive test discriminating."""
    nb_runner.create_notebook([
        SETUP,
        DATA,
        "clf = RandomForestClassifier(n_estimators=40, random_state=42)",   # SEEDED
        "# @cash:cache-fit\nclf.fit(X, y)\nprint('fitted', clf.n_estimators)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    out = nb_runner.get_raw_output(4)
    assert WARNING_CLASS not in out, f"seeded fit should not warn: {out!r}"
    assert DETECTED_TEXT not in out, out
    assert "fitted 40" in nb_runner.get_output(4)


def test_deterministic_estimator_fit_does_not_warn(nb_runner):
    """Control: ``LinearRegression()`` has NO ``random_state`` param, so its fit is
    deterministic -- ``'random_state' not in get_params()`` -> no warning."""
    nb_runner.create_notebook([
        "import numpy as np\n"
        "from sklearn.linear_model import LinearRegression\n"
        "from sklearn.datasets import make_regression\n"
        "import cash\n"
        "%cash_on\n"
        "%cash_badge print",
        "X, y = make_regression(n_samples=6000, n_features=20, random_state=0)",
        "reg = LinearRegression()",
        "# @cash:cache-fit\nreg.fit(X, y)\nprint('coef', reg.coef_.shape[0])",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    out = nb_runner.get_raw_output(4)
    assert WARNING_CLASS not in out, f"deterministic fit should not warn: {out!r}"
    assert DETECTED_TEXT not in out, out
    assert "coef 20" in nb_runner.get_output(4)


def test_allow_random_suppresses_unseeded_estimator_fit(nb_runner):
    """``# @cash:allow-random`` on an opted-in unseeded fit suppresses the warning
    -- the documented opt-out has to cover this detection too.

    Stacks with ``# @cash:cache-fit``: the two directives merge (both are sticky
    booleans), so the fit caches AND stays quiet.
    """
    nb_runner.create_notebook([
        SETUP,
        DATA,
        "clf = RandomForestClassifier(n_estimators=40)",   # UNSEEDED
        "# @cash:cache-fit\n# @cash:allow-random\nclf.fit(X, y)\nprint('fitted', clf.n_estimators)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    out = nb_runner.get_raw_output(4)
    assert WARNING_CLASS not in out, f"allow-random should suppress the warning: {out!r}"
    assert DETECTED_TEXT not in out, out
    assert "fitted 40" in nb_runner.get_output(4)


def test_default_bare_fit_does_not_warn(nb_runner):
    """Control (CAS-170): WITHOUT ``# @cash:cache-fit`` an unseeded bare fit is not
    cached, so there is no frozen replay to warn about.

    The identical notebook to ``test_unseeded_estimator_fit_warns`` minus the
    directive. The warning's whole claim is "the value you are seeing is a
    replay"; on the default path the fit re-executes every run and the claim would
    be FALSE. This is what keeps the warning honest rather than a blanket
    ``.fit()`` alarm.
    """
    nb_runner.create_notebook([
        SETUP,
        DATA,
        "clf = RandomForestClassifier(n_estimators=40)",   # UNSEEDED
        "clf.fit(X, y)\nprint('fitted', clf.n_estimators)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    out = nb_runner.get_raw_output(4)
    assert DETECTED_TEXT not in out, (
        f"uncached bare fit warned about a replay that cannot happen: {out!r}"
    )
    assert "frozen replay" not in out, out
    assert "fitted 40" in nb_runner.get_output(4)
