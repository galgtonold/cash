"""Real-kernel coverage for unseeded ``np.random.default_rng()`` detection.

CAS-135 hole 1. CAS-114 wired up ``CashRandomnessWarning``, but the detector is
rooted at RNG *module* names, so it only ever saw numpy's **legacy global** API
(``np.random.rand``). ``np.random.default_rng()`` — what numpy's own docs have
told everyone to use since 1.17, and what CAS-90 already replays state for — was
invisible. An unseeded Monte Carlo written against the modern API got cached and
replayed bit-identical forever with no warning of any kind.

The unit twin (``tests/test_notebook/test_randomness.py``) asserts on the
detector's call list. This file asserts the thing a user actually experiences:
that the warning reaches the notebook's cell output through a real kernel.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]

WARNING_TEXT = "Unseeded randomness detected"


def test_unseeded_default_rng_draw_warns(nb_runner):
    """The headline gap: a draw off an unseeded modern Generator must warn."""
    nb_runner.create_notebook([
        "import numpy as np",
        "rng = np.random.default_rng()",
        "x = rng.standard_normal(1000)",
        "print('len', len(x))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    out = nb_runner.get_raw_output(3)
    assert WARNING_TEXT in out
    assert "CashRandomnessWarning" in out
    # The draw, not the construction, is what is attributed.
    assert "standard_normal" in out
    # The escape hatch must be discoverable from the warning itself.
    assert "@cash:allow-random" in out
    # The statement still ran.
    assert "len 1000" in nb_runner.get_output(4)


def test_seeded_default_rng_draw_does_not_warn(nb_runner):
    """Control: a seeded Generator is reproducible, so it must stay quiet.

    This is what makes the test above discriminating rather than a detector that
    shouts at every method call it cannot resolve.
    """
    nb_runner.create_notebook([
        "import numpy as np",
        "rng = np.random.default_rng(42)",
        "x = rng.standard_normal(1000)",
        "print('len', len(x))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert WARNING_TEXT not in nb_runner.get_raw_output(3)
    assert "len 1000" in nb_runner.get_output(4)


def test_legacy_global_api_still_warns(nb_runner):
    """Control: do not regress CAS-114's original legacy-global detection."""
    nb_runner.create_notebook([
        "import numpy as np",
        "x = np.random.rand(1000)",
        "print('len', len(x))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    out = nb_runner.get_raw_output(2)
    assert WARNING_TEXT in out
    assert "numpy.random.rand" in out
    assert "len 1000" in nb_runner.get_output(3)


def test_generator_built_inside_function_body_warns(nb_runner):
    """The exact shape from the CAS-135 report: the RNG is a function *local*.

    ``g`` never reaches ``user_ns``, so no live-value classifier can see it — the
    detection has to come off the AST of the ``def`` statement itself.
    """
    nb_runner.create_notebook([
        "import numpy as np",
        "def draw_modern():\n"
        "    g = np.random.default_rng()\n"
        "    return float(g.standard_normal(1000).mean())",
        "print('ok', isinstance(draw_modern(), float))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert WARNING_TEXT in nb_runner.get_raw_output(2)
    assert "ok True" in nb_runner.get_output(3)


def test_np_random_seed_does_not_quiet_a_default_rng_draw(nb_runner):
    """``np.random.seed()`` seeds the legacy global singleton and nothing else.

    A ``default_rng()`` Generator is independent of it, so the module-level seed
    ledger must not be allowed to suppress the Generator's warning — that would
    trade a false negative for exactly the silence CAS-135 is about.
    """
    nb_runner.create_notebook([
        "import numpy as np\nnp.random.seed(42)",
        "rng = np.random.default_rng()",
        "x = rng.standard_normal(1000)",
        "print('len', len(x))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert WARNING_TEXT in nb_runner.get_raw_output(3)
    assert "len 1000" in nb_runner.get_output(4)


def test_allow_random_suppresses_default_rng_warning(nb_runner):
    """The documented opt-out has to cover the newly-detected API too."""
    nb_runner.create_notebook([
        "import numpy as np",
        "rng = np.random.default_rng()",
        "# @cash:allow-random\nx = rng.standard_normal(1000)",
        "print('len', len(x))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert WARNING_TEXT not in nb_runner.get_raw_output(3)
    assert "len 1000" in nb_runner.get_output(4)


def test_pure_function_param_does_not_warn_against_stale_global(nb_runner):
    """CAS-154 Symptom B: a def whose parameter shadows a stale same-named
    unseeded global must NOT warn — the parameter is a different variable.

    Before lexical scoping the def-only cell fired a spurious warning purely
    because ``rng`` was left in the session ledger by the cell above it.
    """
    nb_runner.create_notebook([
        "import numpy as np",             # cell 1
        "rng = np.random.default_rng()",  # cell 2: stale unseeded global, same name
        # cell 3 -- a pure function: the parameter shadows the global, no draw yet.
        "def draw(rng):\n    return float(rng.standard_normal(1000).mean())",
        "print('defined', callable(draw))",  # cell 4
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    # The def cell must be silent: the parameter is not the stale global.
    assert WARNING_TEXT not in nb_runner.get_raw_output(3)
    assert "defined True" in nb_runner.get_output(4)


def test_unseeded_rng_argument_at_call_site_warns(nb_runner):
    """CAS-154 Symptom A: handing an unseeded generator to a function that draws
    off that parameter must warn at the CALL site.

    This is the idiomatic Monte-Carlo shape that cached and froze in silence
    while the equivalent inline draw was correctly warned.
    """
    nb_runner.create_notebook([
        "import numpy as np",  # cell 1
        "def price(n, rng):\n    return float(rng.standard_normal(n).mean())",  # cell 2
        "m = price(1000, rng=np.random.default_rng())",  # cell 3: the call site
        "print('m', isinstance(m, float))",  # cell 4
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    out = nb_runner.get_raw_output(3)  # the call cell
    assert WARNING_TEXT in out
    assert "CashRandomnessWarning" in out
    assert "standard_normal" in out
    assert "@cash:allow-random" in out
    # The def cell itself said nothing — seededness is decided by the argument.
    assert WARNING_TEXT not in nb_runner.get_raw_output(2)
    assert "m True" in nb_runner.get_output(4)


def test_seeded_rng_argument_at_call_site_does_not_warn(nb_runner):
    """Control for the case above: a seeded argument is reproducible, so the
    call site must stay quiet."""
    nb_runner.create_notebook([
        "import numpy as np",  # cell 1
        "def price(n, rng):\n    return float(rng.standard_normal(n).mean())",  # cell 2
        "m = price(1000, rng=np.random.default_rng(42))",  # cell 3: seeded call site
        "print('m', isinstance(m, float))",  # cell 4
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert WARNING_TEXT not in nb_runner.get_raw_output(3)
    assert "m True" in nb_runner.get_output(4)
