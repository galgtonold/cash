"""CAS-179 verification: is ``df['col'] = expensive(...)`` uncacheable while
``df = df.assign(col=expensive(...))`` caches?

Oracle = the real kernel + an EXTERNAL call counter read from outside it (the
badge is itself restored on a hit, so it cannot witness recomputation).

The A/B is run at three levels so the actual variable is isolated:

  * ``_warm_runalls`` -- 12 unchanged ``run_all``s, matching the ticket's
    "verified over 12 unchanged repetitions". This is the workflow the 20%
    runtime claim is about.
  * isolated single-cell re-run, SELF-REFERENTIAL (``df`` is both read and
    written) -- setitem vs ``df = df.assign(...)``.
  * isolated single-cell re-run, NON-self-referential (result bound to a NEW
    name) -- the only arrangement where the two forms can legitimately differ.

MEASURED VERDICT (2026-07-18, main): the headline claim does NOT hold. Over 12
unchanged ``run_all``s BOTH forms show cold=1 / warm=0 -- setitem caches exactly
as well as ``.assign()``, so "NEVER caches / recomputes forever" and the ~20%
runtime figure attributed to the idiom are not reproducible in the run_all
workflow the figure was measured over. Harness validated against a
``with_cash=False`` control on the same shape: cold=1 / warm=3.

What DOES survive is much narrower: on an ISOLATED single-cell re-run, setitem
onto a frame created in an upstream cell recomputes (warm=1) where the
``.assign()`` rebind restores (warm=0). But setitem onto a frame built in the
same cell also restores, so the trigger is in-place mutation of an upstream
object, not subscript-assignment.
"""
import pytest

pytestmark = [pytest.mark.timeout(300)]

SETUP = (
    "import cash\n"
    "%cash_on\n"
    "%cash_badge print\n"
    "import time\n"
    "import pandas as pd"
)

FRAME = "df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})\nN = 5"


def _expensive_def(counter):
    """RHS is independent of ``df`` so a self-reference cannot be blamed."""
    return (
        "def expensive(n):\n"
        f"    open(r'{counter}', 'a').write('X')\n"
        "    time.sleep(0.05)\n"
        "    return [i * 10 for i in range(n)]"
    )


def _warm_runalls(nb_runner, tmp_path, mutate_src, reader, n=12):
    """Cold run, then N unchanged run_alls. Returns (cold, warm, tail, badge)."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(
        [SETUP, _expensive_def(counter), FRAME, mutate_src, reader])
    nb_runner.start_kernel()
    nb_runner.run_all()
    cold = len(counter.read_bytes()) if counter.exists() else 0
    tail = nb_runner.get_output(5)

    for _ in range(n):
        nb_runner.run_all()
    warm = len(counter.read_bytes()) - cold
    return cold, warm, tail, nb_runner.get_output(4)


def _warm_isolated(nb_runner, tmp_path, mutate_src, reader):
    """Cold run, then re-run ONLY the mutating cell."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(
        [SETUP, _expensive_def(counter), FRAME, mutate_src, reader])
    nb_runner.start_kernel()
    nb_runner.run_all()
    cold = len(counter.read_bytes()) if counter.exists() else 0
    tail = nb_runner.get_output(5)

    nb_runner.run_cell(4)
    warm = len(counter.read_bytes()) - cold
    return cold, warm, tail, nb_runner.get_output(4)


READ_DF = "print(f\"b={list(df['b'])}\")"
READ_DF2 = "print(f\"b={list(df2['b'])}\")"
EXPECT = "b=[0, 10, 20, 30, 40]"


# ---------------------------------------------------------------------------
# The ticket's headline claim: 12 unchanged repetitions of the whole notebook.
# ---------------------------------------------------------------------------

def test_setitem_caches_across_12_unchanged_runalls(nb_runner, tmp_path):
    cold, warm, tail, badge = _warm_runalls(
        nb_runner, tmp_path, "df['b'] = expensive(N)", READ_DF)
    assert EXPECT in tail, tail
    assert cold == 1, f"cold={cold}"
    assert warm == 0, (
        f"CAS-179 REPRODUCES: df['b'] = ... recomputed {warm} times over 12 "
        f"unchanged run_alls (cold={cold})\nbadge: {badge!r}")


def test_assign_caches_across_12_unchanged_runalls(nb_runner, tmp_path):
    cold, warm, tail, badge = _warm_runalls(
        nb_runner, tmp_path, "df = df.assign(b=expensive(N))", READ_DF)
    assert EXPECT in tail, tail
    assert cold == 1, f"cold={cold}"
    assert warm == 0, (
        f".assign() recomputed {warm} times over 12 unchanged run_alls "
        f"(cold={cold})\nbadge: {badge!r}")


# ---------------------------------------------------------------------------
# Isolated single-cell re-run, SELF-REFERENTIAL. Both forms rebind/mutate the
# same ``df`` they read, so both are expected to run-from-start for idempotency.
# If the ticket is right, setitem recomputes here and .assign() does not.
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason="CAS-179, narrow surviving half: an in-place setitem onto a frame "
           "created in an UPSTREAM cell re-runs its RHS on an isolated "
           "single-cell re-run, where the .assign() rebind in the same position "
           "restores. Setitem onto a frame built in the SAME cell "
           "(test_setitem_on_a_copy_isolated_rerun) caches fine, so setitem is "
           "not itself uncacheable -- the trigger is mutating an upstream "
           "object. May be by-design (the in-place-mutation rebuild rule).",
    strict=False)
def test_setitem_isolated_rerun(nb_runner, tmp_path):
    cold, warm, tail, badge = _warm_isolated(
        nb_runner, tmp_path, "df['b'] = expensive(N)", READ_DF)
    assert EXPECT in tail, tail
    assert (cold, warm) == (1, 0), (
        f"setitem isolated re-run: cold={cold} warm={warm}\nbadge: {badge!r}")


def test_assign_isolated_rerun_self_referential(nb_runner, tmp_path):
    cold, warm, tail, badge = _warm_isolated(
        nb_runner, tmp_path, "df = df.assign(b=expensive(N))", READ_DF)
    assert EXPECT in tail, tail
    assert (cold, warm) == (1, 0), (
        f".assign() self-referential isolated re-run: cold={cold} warm={warm}\n"
        f"badge: {badge!r}")


# ---------------------------------------------------------------------------
# Isolated single-cell re-run, NON-self-referential (result -> a NEW name).
# This removes the run-from-start confound; it is the only fair A/B for
# "setitem is uncacheable".
# ---------------------------------------------------------------------------

def test_assign_isolated_rerun_new_name(nb_runner, tmp_path):
    cold, warm, tail, badge = _warm_isolated(
        nb_runner, tmp_path, "df2 = df.assign(b=expensive(N))", READ_DF2)
    assert EXPECT in tail, tail
    assert (cold, warm) == (1, 0), (
        f".assign() to a new name: cold={cold} warm={warm}\nbadge: {badge!r}")


def test_setitem_on_a_copy_isolated_rerun(nb_runner, tmp_path):
    """The nearest setitem equivalent of the new-name .assign(): build a fresh
    frame in the same cell, then setitem onto it."""
    cold, warm, tail, badge = _warm_isolated(
        nb_runner, tmp_path,
        "df2 = df.copy()\ndf2['b'] = expensive(N)", READ_DF2)
    assert EXPECT in tail, tail
    assert (cold, warm) == (1, 0), (
        f"setitem on a fresh copy: cold={cold} warm={warm}\nbadge: {badge!r}")
