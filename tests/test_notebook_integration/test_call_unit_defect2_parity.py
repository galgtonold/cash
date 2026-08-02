"""CAS-257 defect 2: pins that the call-unit spelling and the ordinary
statement spelling agree on a loop-var-mutated-before-use case.

The shape (see ``docs/known-limitations.md``, "A loop variable mutated
before it's read collides with an earlier iteration"): a loop whose target
is bound to an EQUAL value on two iterations, then made different by a
body mutation *before* the value that has to discriminate the iteration is
used. ``ForLoopHandler._process_one_iteration`` captures both the
statement path's ``__iteration_context__`` hash and the call unit's
``loop_var_digests`` at BINDING time, before any body statement (including
the mutating one) runs -- so neither channel can see a mutation that
happens later in the same iteration's body.

That is exactly the finding of task 10c's investigation (CAS-257 defect 2):
the call unit does not introduce a new class of defect here, it inherits an
existing property of ``build_iteration_context`` that the plain statement
path already has. Filing this again as a call-path-specific bug (it
already happened twice on the adjacent impure-callee question, CAS-246)
would be a re-diagnosis, not a new bug -- this test is the guard against
that.

**What this test does NOT claim**: that either spelling is CORRECT. Today
both are wrong relative to a cash-off oracle (both give ``[1, 1]`` where
the oracle gives ``[1, 2]``) -- that is the known, documented limitation.
This test pins that the two spellings stay in lock-step, whatever their
shared answer is. If a future fix changes ONE spelling's behaviour without
the other, this test fails and flags the newly introduced asymmetry --
which is the actual bug class this ticket cares about, not the specific
wrong value.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.loops]

_SLEEP = 0.15


def _defs(log):
    return (
        "import time, pathlib\n"
        f"MARK = pathlib.Path(r'{log}')\n"
        "state = {'k': 0}\n"
        "handle = 'conn-object'\n"
        "def pull(handle):\n"
        "    state['k'] += 1\n"
        "    with MARK.open('a') as fh:\n"
        "        fh.write('r\\n')\n"
        f"    time.sleep({_SLEEP})\n"
        "    return state['k']\n"
    )


def _runs(log):
    return len(log.read_text().splitlines()) if log.exists() else 0


# Call-unit spelling: `pull(handle)` is intercepted and cached as its own
# sub-unit, keyed (in part) on the frozen `loop_var_digests` entry for `q`.
_CALL_UNIT = (
    "accm = []\n"
    "# @cash:cache-calls\n"
    "for q in [[1], [1]]:\n"
    "    q.append(len(accm))\n"
    "    accm.append(pull(handle))\n"
    "print('OUT', accm)\n"
)

# Statement spelling: no `# @cash:cache-calls` at all. `v = pull(handle)` is
# an ordinary, non-mutating per-iteration cached statement, keyed (in part)
# on the frozen `__iteration_context__` hash for the same `q`.
# `accm.append(v)` mirrors `q.append(...)`: an always-executing mutation,
# structurally inert for this question (same role as `q.append` above).
_STATEMENT = (
    "accm = []\n"
    "for q in [[1], [1]]:\n"
    "    q.append(len(accm))\n"
    "    v = pull(handle)\n"
    "    accm.append(v)\n"
    "print('OUT', accm)\n"
)


@pytest.mark.parametrize("tag,code", [("call_unit", _CALL_UNIT), ("statement", _STATEMENT)])
def test_oracle_is_correct(tag, code, nb_runner, tmp_path):
    """cash OFF: both spellings must agree with each other AND be correct.

    Establishes there is nothing shape-specific about either spelling that
    would make the two arms an unfair comparison -- with caching off, both
    give the plain, correct answer.
    """
    log = tmp_path / "m.log"
    nb_runner.create_notebook([_defs(log), code])
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()
    out = nb_runner.get_output(2)
    assert "OUT [1, 2]" in out, f"{tag}: {out}"


@pytest.mark.parametrize("tag,code", [("call_unit", _CALL_UNIT), ("statement", _STATEMENT)])
def test_with_cash_both_spellings_collapse_identically(tag, code, nb_runner, tmp_path):
    """cash ON: pin that BOTH spellings collapse the two iterations onto one
    cached value, in the SAME way -- not that either is correct.

    A real-execution count (via the file-write log `pull()` makes on every
    genuine call) proves the collapse is an actual cache hit and not, say,
    an accidentally-uncached run that happened to print the same thing.
    """
    log = tmp_path / "m.log"
    nb_runner.create_notebook([_defs(log), code])
    nb_runner.start_kernel()
    nb_runner.run_all()
    out = nb_runner.get_output(2)
    runs = _runs(log)
    assert "OUT [1, 1]" in out, f"{tag}: {out}"
    assert runs == 1, f"{tag}: expected exactly 1 real pull() run (the collapse), got {runs}"
