"""Real-kernel coverage for announcing a *replayed* unseeded random value.

CAS-135 hole 2. CAS-114's warning fired on the COLD run — the run where the
value is freshly computed and correct — and went silent on every restore after
it, which is precisely when the number on screen is a frozen replay. The alarm
was quiet exactly when the house was on fire.

The two warnings are different claims and are tested as such:

* cold run  -> "your source is unseeded"        (advice about what you wrote)
* restore   -> "this value is a replay"         (a fact about what you are seeing)

Both are deduped once per statement per session, so a user who re-runs a cell
twenty times is told once, not twenty times.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]

COLD_TEXT = "Unseeded randomness detected"
STALE_TEXT = "Unseeded randomness restored from cache"


def _sum_line(output: str) -> str:
    """Pull just the ``sum=`` line out of a cell's output.

    Cell output carries the warning text too, so comparing whole outputs across
    runs would compare the *warnings* (which differ by design) rather than the
    value (which is the thing being asserted frozen).
    """
    lines = [ln for ln in output.splitlines() if ln.strip().startswith("sum=")]
    assert lines, f"no 'sum=' line in output: {output!r}"
    return lines[-1].strip()


def test_replayed_unseeded_value_is_announced_on_restore(nb_runner):
    """The headline: a frozen value must not be served in silence."""
    nb_runner.create_notebook([
        "import numpy as np",
        "# @cash:persist\nx = np.random.rand(1000)\nprint('sum=', round(float(x.sum()), 6))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    first, first_raw = nb_runner.get_output(2), nb_runner.get_raw_output(2)

    nb_runner.run_cells([2])
    second, second_raw = nb_runner.get_output(2), nb_runner.get_raw_output(2)

    # Premise: the value really is frozen. A recompute would advance the RNG.
    assert _sum_line(first) == _sum_line(second)
    # Cold run: source-level advice, and NOT the replay claim (it was computed).
    assert COLD_TEXT in first_raw
    assert STALE_TEXT not in first_raw
    # Restore: the user is told the number is a replay.
    assert STALE_TEXT in second_raw
    assert "@cash:no-cache" in second_raw


def test_replay_warning_does_not_repeat_on_further_reruns(nb_runner):
    """CAS-114 deliberately avoided per-run spam; that must survive this fix.

    The fact does not change between run 2 and run 20 — say it once.
    """
    nb_runner.create_notebook([
        "import numpy as np",
        "# @cash:persist\nx = np.random.rand(1000)\nprint('sum=', round(float(x.sum()), 6))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.run_cells([2])
    assert nb_runner.get_raw_output(2).count(STALE_TEXT) == 1

    nb_runner.run_cells([2])
    assert nb_runner.get_raw_output(2).count(STALE_TEXT) == 0


def test_unchanged_non_random_restore_says_nothing(nb_runner):
    """Control: the noise floor. A cached statement with no randomness in it
    must stay completely quiet — this is what stops the fix from becoming a
    'you hit the cache' banner on every restore in the notebook.
    """
    nb_runner.create_notebook([
        "import numpy as np",
        "# @cash:persist\ny = sum(i * i for i in range(200000))\nprint('y=', y)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.run_cells([2])
    second_raw = nb_runner.get_raw_output(2)
    assert "randomness" not in second_raw.lower()
    assert "y=" in nb_runner.get_output(2)


def test_seeded_restore_says_nothing(nb_runner):
    """Control: a seeded draw replays honestly — the cached value is exactly
    what a recompute would produce, so there is nothing to report."""
    nb_runner.create_notebook([
        "import numpy as np\nnp.random.seed(0)",
        "# @cash:persist\nx = np.random.rand(1000)\nprint('sum=', round(float(x.sum()), 6))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.run_cells([2])
    assert STALE_TEXT not in nb_runner.get_raw_output(2)


def test_allow_random_suppresses_the_replay_warning(nb_runner):
    """The directive means 'I know'. It has to mean that on both runs, or it
    stops being a suppression and becomes a half-suppression."""
    nb_runner.create_notebook([
        "import numpy as np",
        "# @cash:persist\n# @cash:allow-random\nx = np.random.rand(1000)\nprint('sum=', round(float(x.sum()), 6))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.run_cells([2])
    raw = nb_runner.get_raw_output(2)
    assert STALE_TEXT not in raw
    assert COLD_TEXT not in raw


def test_replayed_default_rng_value_is_announced(nb_runner):
    """H1 + H2 together: the modern API, frozen and served from cache.

    This is the CAS-135 report's actual shape — the combination that produced a
    wrong number with no warning anywhere.
    """
    nb_runner.create_notebook([
        "import numpy as np",
        "rng = np.random.default_rng()",
        "# @cash:persist\nx = rng.standard_normal(1000)\nprint('sum=', round(float(x.sum()), 6))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = nb_runner.get_output(3)

    nb_runner.run_cells([3])
    second_raw = nb_runner.get_raw_output(3)

    assert _sum_line(first) == _sum_line(nb_runner.get_output(3))
    assert STALE_TEXT in second_raw
    assert "standard_normal" in second_raw
