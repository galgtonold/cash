"""The full contract for an UNSEEDED draw across re-runs (CAS-230).

Re-running an unseeded draw reprints the SAME number. That is deliberate — the
documented "non-determinism is frozen, not blocked" rule — because unseeded
randomness is everywhere in the notebooks cash targets (``train_test_split``
with no ``random_state``, sklearn defaults, dropout, bootstrap). If every such
draw redrew on each run, every cached result downstream of it would
cascade-invalidate and cash would recompute the whole chain every time.

Worth being precise about the mechanism, because it is easy to misread as a
caching bug: no cache entry is involved. A cheap draw is under the persistence
floor and is never stored — it genuinely RE-EXECUTES, and lands on the same
value because the upstream checker rewinds the RNG to the position the cell
started from. Caching and freezing are separate levers.

Freezing by default is only defensible if the way out actually works, so all
three arms are pinned together here:

* plain            -> frozen, and the user is TOLD (``CashRandomnessWarning``,
  once per statement per session)
* ``no-cache``     -> genuinely fresh each run
* ``allow-random`` -> frozen, warning silenced

``no-cache`` was the gap. The warning names it as the way to "re-run it every
time", but it only switched off caching — which was not what froze the value —
so the statement re-executed, got rewound, and redrew the identical number.
Following cash's own advice changed nothing and produced no further signal.
"""
import pytest

pytestmark = pytest.mark.libraries

C_ON = "import cash\n%cash_on"
SETUP = "import random"


def _drawn(nb_runner, cell_num: int) -> str:
    return nb_runner.get_output(cell_num).split("r=")[-1].strip().splitlines()[0]


def _two_run_alls(nb_runner, draw_cell: str) -> tuple[str, str]:
    nb_runner.create_notebook([C_ON, SETUP, draw_cell])
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = _drawn(nb_runner, 3)
    nb_runner.run_all()
    return first, _drawn(nb_runner, 3)


@pytest.mark.timeout(180)
def test_plain_unseeded_draw_is_frozen_across_runs(nb_runner):
    first, second = _two_run_alls(nb_runner, "r = random.random()\nprint('r=', r)")
    assert first == second, (
        "an unseeded draw must reprint the same value on re-run (frozen, not redrawn)"
    )


@pytest.mark.timeout(180)
def test_frozen_draw_tells_the_user(nb_runner):
    """Freezing is only acceptable because it is announced, so pin the warning.

    It lands on the FIRST run and is then deduped — once per statement per
    session, deliberately, so a re-run is not flooded with a warning class users
    would learn to filter wholesale (CAS-114). Note a cheap draw never produces
    the *replay* warning, which is gated on an actual cache restore: it is under
    the persistence floor and re-executes instead. The compute-time warning is
    the only one it gets, which is why this pins that one.
    """
    nb_runner.create_notebook([C_ON, SETUP, "r = random.random()\nprint('r=', r)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    out = nb_runner.get_raw_output(3)
    assert "CashRandomnessWarning" in out, "a frozen unseeded draw must warn"
    assert "@cash:allow-random" in out, "the escape hatches must be discoverable"

    nb_runner.run_all()
    assert "CashRandomnessWarning" not in nb_runner.get_raw_output(3), (
        "the warning is deduped after the first run, by design"
    )


@pytest.mark.timeout(180)
def test_no_cache_redraws_every_run(nb_runner):
    """The escape hatch the warning names must actually produce a fresh draw."""
    first, second = _two_run_alls(
        nb_runner, "# @cash:no-cache\nr = random.random()\nprint('r=', r)",
    )
    assert first != second, (
        "# @cash:no-cache must redraw each run — it has to switch off the RNG "
        "rewind, not just caching, since the rewind is what freezes the value"
    )


@pytest.mark.timeout(180)
def test_allow_random_stays_frozen_and_silent(nb_runner):
    """allow-random suppresses the warning; it does not change the value."""
    first, second = _two_run_alls(
        nb_runner, "# @cash:allow-random\nr = random.random()\nprint('r=', r)",
    )
    assert first == second, "allow-random must not change the frozen behaviour"

    nb_runner.create_notebook(
        [C_ON, SETUP, "# @cash:allow-random\nr = random.random()\nprint('r=', r)"],
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    assert "CashRandomnessWarning" not in nb_runner.get_raw_output(3), (
        "allow-random must silence the randomness warning"
    )
