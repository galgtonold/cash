"""`# @cash:persist` must reach a call entry, not just the statement (CAS-269).

`persist` forces disk persistence for entries that would otherwise stay in RAM
(below the ~0.1s persistence floor). It reached the STATEMENT entry only. When
the statement cannot cache -- the CAS-260 shape, where the callee writes a
global so the statement is skip-cached and re-executes -- the call entry is the
only thing cached, and the annotation acted on nothing.

Measured before the fix, real kernel, restart between runs, 20ms body (above
the ~10ms statement floor and the call unit's 3ms one, BELOW the ~0.1s
persistence floor, so nothing reaches disk unless `persist` says so):

    statement shape                     no annotation   `# @cash:persist`
    pure callee -- statement cacheable        1                 0   <- works
    callee writes a global -- skip-cached     1                 1   <- no-ops

**Both rows are load-bearing and the first one is why.** The pure-callee row is
the non-vacuity control: it proves `persist` is effective at this cost, so the
second row is measuring the statement/call boundary rather than the floor. Drop
it and a body that merely sat above the persistence floor would make this file
pass against a completely unwired annotation.

Not a correctness bug -- no wrong value is ever produced, and above the
persistence floor a call entry reaches disk anyway. The cost is a silent re-run
of cheap-ish work after a restart, which is why this is `cache-perf`.

`os.open`/`os.write`, never `builtins.open`, which `FileAccessTracker` patches
into a file dependency -- that changes the entry every run and disables the
caching under test.
"""
import pytest

pytestmark = [pytest.mark.integration]

SETUP = "import cash\n%cash_on\n"

# 20ms: above the statement cost floor (~10ms) and the call unit's (3ms),
# below the persistence floor (~0.1s). The whole measurement lives in that band.
BODY_S = 0.02


def _defs(ticks, *, writes_global):
    """Callee definitions. `writes_global` picks the row of the table.

    The global write is what makes the enclosing statement skip-cache (CAS-260),
    leaving the call entry as the only thing cached -- the shape where `persist`
    had nothing to act on.
    """
    body = (
        "    LOG.append(n)\n" if writes_global else ""
    )
    return (
        "import time, os\n"
        "LOG = []\n"
        "def _tick():\n"
        f"    fd = os.open(r'{ticks}', os.O_WRONLY | os.O_APPEND | os.O_CREAT)\n"
        "    os.write(fd, b'X')\n"
        "    os.close(fd)\n"
        "def work(n):\n"
        "    _tick()\n"
        f"{body}"
        f"    time.sleep({BODY_S})\n"
        "    return n * 2\n"
    )


def _n(path):
    return len(path.read_bytes()) if path.exists() else 0


def _reexecutions_after_restart(nb_runner, tmp_path, annotation, label, *, writes_global):
    """Cold run, kernel restart, re-run. Returns re-executions of `work`."""
    ticks = tmp_path / f"{label}.log"
    nb_runner.create_notebook([
        SETUP,
        _defs(ticks, writes_global=writes_global),
        f"{annotation}\nv = work(21)\n" if annotation else "v = work(21)\n",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    cold = _n(ticks)
    assert cold == 1, f"the cold run should execute work once, got {cold}"

    nb_runner.restart()
    nb_runner.run_all()
    assert nb_runner.peek("v") == "42", "the value changed, so this measures the wrong thing"
    return _n(ticks) - cold


def test_persist_reaches_a_call_entry_when_the_statement_is_skip_cached(nb_runner, tmp_path):
    """The bug: the callee writes a global, so only the call entry is cached."""
    n = _reexecutions_after_restart(
        nb_runner, tmp_path, "# @cash:persist", "impure_persist", writes_global=True,
    )
    assert n == 0, (
        "`# @cash:persist` did not reach the call entry: the statement is "
        "skip-cached (the callee writes a global), so the call entry is the only "
        "thing cached, and the annotation acted on nothing"
    )


def test_control_persist_works_when_the_statement_itself_caches(nb_runner, tmp_path):
    """NON-VACUITY. Proves `persist` is effective at this body cost.

    Without this, a body that happened to sit above the persistence floor would
    make the test above pass against an annotation that was never wired up.
    """
    n = _reexecutions_after_restart(
        nb_runner, tmp_path, "# @cash:persist", "pure_persist", writes_global=False,
    )
    assert n == 0, (
        "`# @cash:persist` did not survive a restart even for a plain cacheable "
        "statement -- the harness is measuring something other than persistence"
    )


@pytest.mark.parametrize("writes_global", [False, True], ids=["pure", "writes_global"])
def test_control_without_the_annotation_the_work_re_runs(nb_runner, tmp_path, writes_global):
    """NON-VACUITY. The body must be BELOW the persistence floor.

    If this ever reports 0, the body got expensive enough to persist on its own
    and both tests above are passing for free.
    """
    n = _reexecutions_after_restart(
        nb_runner, tmp_path, "", f"bare_{writes_global}", writes_global=writes_global,
    )
    assert n == 1, (
        f"expected the un-annotated {BODY_S}s body to be re-executed after a "
        f"restart (it is below the ~0.1s persistence floor), got {n}"
    )
