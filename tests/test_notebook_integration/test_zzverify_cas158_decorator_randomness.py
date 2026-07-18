"""CAS-158: the `@cash.cache` DECORATOR must warn about unseeded randomness,
the same way the notebook path already does.

Before the fix the decorator performed NO randomness detection at all: it would
silently freeze a non-deterministic result forever with nothing on screen. The
notebook path warned correctly, so the two paths disagreed about the same draw.

Both halves are still measured in one file so the *symmetry* is the assertion,
not an inference across two suites:

* notebook half  -> real kernel (nb_runner), `%cash_on`, unseeded draw.
* decorator half -> in-process `Cash(backend=FileBackend(tmp))`, same draw.

The fix reuses the notebook path's `RandomnessDetector` verbatim, at DECORATION
time, so "what counts as unseeded" cannot drift between the two paths.

NOTE ON FUNCTION DEFINITIONS: the decorator half defines its functions with real
`def` statements at module level rather than via `exec()`. Detection is
source-based (`inspect.getsourcelines`), and `exec`-defined functions have no
retrievable source -- the same blind spot the purity analyzer has. That boundary
is pinned explicitly by `test_decorator_silent_when_source_unavailable` rather
than being allowed to silently hide a regression.
"""
import tempfile
import warnings

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]

COLD_TEXT = "Unseeded randomness detected"


def _fresh_cash():
    from cash import Cash, FileBackend

    return Cash(backend=FileBackend(cache_dir=tempfile.mkdtemp()), register_magic=False)


def _randomness_warnings(records):
    from cash import CashRandomnessWarning

    return [
        str(r.message) for r in records
        if issubclass(r.category, CashRandomnessWarning)
    ]


# ---------------------------------------------------------------------------
# Notebook path: the protection CAS-135 shipped. The reference the decorator is
# compared against -- kept intact so a regression here is attributed correctly.
# ---------------------------------------------------------------------------

def test_notebook_path_warns_on_unseeded_randomness(nb_runner):
    nb_runner.create_notebook([
        "import numpy as np",
        "# @cash:persist\nx = np.random.rand(1000)\nprint('sum=', round(float(x.sum()), 6))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    raw = nb_runner.get_raw_output(2)
    assert COLD_TEXT in raw, f"notebook path no longer warns: {raw!r}"


# ---------------------------------------------------------------------------
# Decorator path: must now warn, and must warn at DECORATION time, once.
# ---------------------------------------------------------------------------

def _unseeded_numpy():
    return float(np.random.randn())


def _unseeded_stdlib():
    import random
    return random.random()


def _unseeded_generator():
    rng = np.random.default_rng()          # no seed
    return float(rng.normal())


@pytest.mark.parametrize("fn,label", [
    (_unseeded_numpy, "np.random.randn"),
    (_unseeded_stdlib, "random.random"),
    (_unseeded_generator, "np.random.default_rng (unseeded carrier)"),
])
def test_decorator_warns_on_unseeded_randomness(fn, label):
    """CAS-158: the decorator is no longer silent about an unseeded draw."""
    c = _fresh_cash()

    with warnings.catch_warnings(record=True) as at_decoration:
        warnings.simplefilter("always")
        cached = c.cache(fn)

    decoration_msgs = _randomness_warnings(at_decoration)
    assert decoration_msgs, f"{label}: decorator did not warn (CAS-158 regressed)"
    assert COLD_TEXT in decoration_msgs[0], (
        f"{label}: message diverged from the notebook path's wording: "
        f"{decoration_msgs[0]!r}"
    )

    # The freeze itself is unchanged -- the warning is advice, not a behaviour
    # change. The warm call still replays instead of drawing again.
    first = cached()
    second = cached()
    assert first == second, f"{label}: caching behaviour changed unexpectedly"


def test_decorator_warns_at_decoration_time_not_per_call():
    """The check is a pure function of the source: it must not touch the hot path.

    Pinning this is the point -- moving detection into the wrapper would make
    every cached call pay for an AST parse.
    """
    c = _fresh_cash()

    with warnings.catch_warnings(record=True) as at_decoration:
        warnings.simplefilter("always")
        cached = c.cache(_unseeded_numpy)
    assert len(_randomness_warnings(at_decoration)) == 1, "expected exactly one warning"

    with warnings.catch_warnings(record=True) as during_calls:
        warnings.simplefilter("always")
        for _ in range(5):
            cached()
    assert not _randomness_warnings(during_calls), (
        "randomness warning fired during calls -- detection leaked into the hot path"
    )


def test_decorator_warning_is_discoverable_via_cache_info():
    """The warning is also filed on the function, so it survives a missed stderr."""
    c = _fresh_cash()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        cached = c.cache(_unseeded_numpy)

    logged = [w["category"] for w in cached.cache_info()["warnings"]]
    assert "CashRandomnessWarning" in logged, logged


# ---------------------------------------------------------------------------
# The other direction: a SEEDED draw is reproducible and must stay silent.
# This is the whole point of the seed-tracking the notebook path already does
# (CAS-154/167) -- a detector that warns on everything is useless.
# ---------------------------------------------------------------------------

def _seeded_numpy():
    np.random.seed(0)
    return float(np.random.randn())


def _seeded_generator():
    rng = np.random.default_rng(42)
    return float(rng.normal())


def _seeded_stdlib():
    import random
    random.seed(7)
    return random.random()


def _no_randomness(a, b):
    return a + b


@pytest.mark.parametrize("fn,label", [
    (_seeded_numpy, "np.random.seed"),
    (_seeded_generator, "default_rng(42)"),
    (_seeded_stdlib, "random.seed"),
    (_no_randomness, "no randomness at all"),
])
def test_decorator_silent_when_seeded_or_deterministic(fn, label):
    c = _fresh_cash()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        c.cache(fn)
    assert not _randomness_warnings(rec), (
        f"{label}: false positive -- a seeded/deterministic function must not warn"
    )


# ---------------------------------------------------------------------------
# Opt-out, both spellings.
# ---------------------------------------------------------------------------

def test_allow_random_kwarg_suppresses_the_warning():
    """`@cash.cache(allow_random=True)` -- the decorator-path opt-out."""
    c = _fresh_cash()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        c.cache(allow_random=True)(_unseeded_numpy)
    assert not _randomness_warnings(rec), "allow_random=True did not suppress"


def _unseeded_with_comment_optout():
    # @cash:allow-random
    return float(np.random.randn())


def test_inline_allow_random_comment_suppresses_the_warning():
    """The notebook's `# @cash:allow-random` is honoured in decorated source too.

    Users arriving from `%cash_on` reach for the comment, and the decorator has
    the source in hand anyway -- so the same directive vocabulary works.
    """
    c = _fresh_cash()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        c.cache(_unseeded_with_comment_optout)
    assert not _randomness_warnings(rec), "# @cash:allow-random did not suppress"


def test_allow_random_does_not_disable_caching():
    """The opt-out silences the warning only -- it is not `no-cache`."""
    c = _fresh_cash()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        cached = c.cache(allow_random=True)(_unseeded_numpy)
    assert cached() == cached(), "allow_random must not change caching behaviour"


# ---------------------------------------------------------------------------
# Documented boundaries of a source-based check.
# ---------------------------------------------------------------------------

def test_decorator_silent_when_source_unavailable():
    """`exec`-defined functions have no retrievable source -> no claim is made.

    Not a bug being papered over: with no source there is nothing to scan, and
    inventing a warning would be a guess. Identical to the purity analyzer's
    treatment of source-less callables. Pinned so the silence stays *deliberate*.
    """
    c = _fresh_cash()
    ns = {"np": np}
    exec("def drawn():\n    return float(np.random.randn())\n", ns)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cached = c.cache(ns["drawn"])

    assert not _randomness_warnings(rec), "unexpected warning for source-less function"
    assert cached() == cached(), "source-less function should still cache"


def test_decorator_silent_on_unseeded_sklearn_fit():
    """OUT OF SCOPE for CAS-158's source-based check: randomness inside `.fit()`.

    An unseeded estimator is a real freeze hazard, but the randomness lives in
    sklearn's compiled `.fit()` -- bootstrap sampling, weight init -- not in any
    Python call an AST can see. The notebook path catches this via a separate
    RUNTIME channel (CAS-167) that inspects the live estimator
    (`get_params()['random_state'] is None`).

    That channel cannot be lifted to decoration time: it needs the estimator
    OBJECT, which only exists once the function runs. Porting it would mean a
    per-call check, which CAS-158 explicitly rules out. Recorded here as a known,
    deliberate gap so it is not mistaken for the bug CAS-158 fixed.
    """
    sk = pytest.importorskip("sklearn.ensemble")
    c = _fresh_cash()

    X = np.random.RandomState(0).rand(60, 4)
    y = (X[:, 0] > 0.5).astype(int)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")

        @c.cache
        def fit_model(n):
            m = sk.RandomForestClassifier(n_estimators=n)  # NO random_state
            m.fit(X, y)
            return m.feature_importances_.tolist()

    assert fit_model(5) == fit_model(5), "fit was not frozen"
    # Documents the gap; flip this if the runtime channel is ever ported.
    assert not _randomness_warnings(rec), (
        "decorator now warns on unseeded fit -- the CAS-167 runtime channel "
        "appears to have been ported; update this test to assert the new behaviour"
    )
