"""An impure callee freezes the SAME WAY in both spellings (CAS-246).

This is NOT an endorsement of the freeze. Whether an unseeded, globally
mutating function like ``next_seq()`` *should* be frozen by caching at all is
an open question tracked as CAS-246 -- this test takes no position on it.
What it pins is narrower and not up for debate: whichever way that question
is eventually answered, a plain assignment (``a = next_seq()``, the ordinary
statement-cache path) and the same call made through a mutation
(``seen.append(next_seq())``, the call-unit path CAS-243 added) must agree.

That equivalence matters because the failure mode it guards against has
already happened twice on related tickets: someone observes the freeze
through one spelling, misdiagnoses it as specific to whichever caching
mechanism handled that spelling, and re-files it as a new bug against the
*other* mechanism. Before CAS-243, ``seen.append(next_seq())`` was
skip-cached outright (a mutation gets zero reuse), so the two spellings could
never have been compared this way -- interception is what makes the append
form cacheable at all, and hence what makes this comparison possible, and
necessary: a rule that freezes for one spelling and not the other would be a
genuinely new defect in the call-unit path, not a restatement of the
already-known, already-accepted statement-path behaviour.

Counted, never timed. `next_seq()` sleeps past the ~10ms cost floor so it is
actually eligible for caching -- a callee cheaper than that floor is never
stored at all, and "frozen" and "not cached" would be indistinguishable.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.loops]

SETUP = "import cash\n%cash_on\n"
DEFS = """\
import time
N = [0]
def next_seq():
    N[0] += 1
    time.sleep(0.3)
    return N[0]
"""


def test_both_spellings_agree(nb_runner):
    nb_runner.create_notebook([
        SETUP, DEFS,
        "a = next_seq()\nprint('A', a)",
        "seen = []",
        "seen.append(next_seq())\nprint('B', seen)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    # get_output is 1-BASED: the assignment cell is 3, the append cell is 5.
    # `N` is one shared counter for BOTH cells (not reset between them), so
    # cell 5's call is the counter's SECOND increment, not its first.
    first_a = nb_runner.get_output(3)
    first_b = nb_runner.get_output(5)
    assert "A 1" in first_a, first_a
    assert "B [2]" in first_b, first_b

    # Re-run both measurement cells, twice more, checking every intermediate
    # output -- not just the final one, so a mechanism that freezes on the
    # first rerun but drifts on the second could not slip past unnoticed.
    for _ in range(2):
        nb_runner.run_cells([3, 5])
        assert nb_runner.get_output(3) == first_a, "assignment spelling did not freeze"
        assert nb_runner.get_output(5) == first_b, (
            "append spelling must freeze the SAME WAY as the assignment "
            "spelling now that the call is cached -- see CAS-246"
        )


def test_both_spellings_agree_with_the_no_cash_oracle_showing_they_would_differ(nb_runner):
    """Positive control: without caching, both spellings keep advancing.

    Confirms the two tests above are actually observing a caching effect --
    ``N``'s counter genuinely increments every call -- rather than some
    accident of the notebook harness making repeated runs no-ops regardless
    of cash. Without this, a broken test fixture that always echoed a stale
    cell output would still pass the freeze assertions above.

    ``seen`` is only ever cleared once (cell 4, run once by ``run_all``), so
    with no caching each further ``.append()`` genuinely accumulates onto it
    -- ``[2]`` after the first run, ``[2, 4]`` after the next. That growth
    IS the "kept advancing" signal for the append spelling; the assignment
    spelling shows the same thing more simply, since ``a`` is overwritten
    rather than accumulated (``'A 3'`` after the second run).
    """
    nb_runner.create_notebook([
        "import cash\n", DEFS,
        "a = next_seq()\nprint('A', a)",
        "seen = []",
        "seen.append(next_seq())\nprint('B', seen)",
    ])
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()

    assert "A 1" in nb_runner.get_output(3)
    assert "B [2]" in nb_runner.get_output(5)

    nb_runner.run_cells([3, 5])
    assert "A 3" in nb_runner.get_output(3), (
        "the assignment spelling should keep advancing with cash off"
    )
    assert "B [2, 4]" in nb_runner.get_output(5), (
        "the append spelling should keep advancing with cash off"
    )
