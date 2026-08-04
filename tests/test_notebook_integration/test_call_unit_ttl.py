"""A statement's `ttl` governs the calls inside it, not just the statement (CAS-268).

`call_unit.py` contained no reference to `ttl` at all, so call entries never
expired. Once call interception became the default (CAS-243) that quietly
hollowed out the annotation: the STATEMENT would expire and re-execute while
the expensive call inside it was still served from an entry with no expiry.
Measured on `# @cash:ttl=0` -- the spelling the docs give for data that must
never be served stale -- the work did not re-run at all until
`# @cash:no-cache-calls` was added too.

This is the same failure direction that made CAS-221 a P1 ("your fresh data is
not fresh, silently"), one layer down.

**The instrument is the hard part here, and it is easy to get backwards.** A
counter written INSIDE the callee measures whether the CALL ran; a counter
outside the statement measures whether the STATEMENT ran. During the
investigation, conflating them made TTL look entirely broken -- every arm
reported zero re-runs, including `ttl=2` well past its window -- because the
counter sat inside a callee whose call was being served. The controls below
exist so that reading cannot pass silently: if `ttl=3600` re-runs or `ttl=2`
does not, the harness is measuring something other than expiry.

`os.open`/`os.write`, never `builtins.open`, which `FileAccessTracker` patches
into a file dependency -- that changes the entry every run and disables the
caching under test.
"""
import time

import pytest

pytestmark = [pytest.mark.integration]

SETUP = "import cash\n%cash_on\n"


def _defs(ticks):
    return (
        "import time, os\n"
        "def _tick():\n"
        f"    fd = os.open(r'{ticks}', os.O_WRONLY | os.O_APPEND | os.O_CREAT)\n"
        "    os.write(fd, b'X')\n"
        "    os.close(fd)\n"
        "def fetch(n):\n"
        "    _tick()\n"
        "    time.sleep(0.15)\n"
        "    return n * 2\n"
    )


def _n(path):
    return len(path.read_bytes()) if path.exists() else 0


def _run(nb_runner, tmp_path, annotation, label, *, wait=0.0):
    """Cold run, optional wait, one re-run. Returns re-executions of `fetch`."""
    ticks = tmp_path / f"{label}.log"
    nb_runner.create_notebook([SETUP, _defs(ticks), f"{annotation}\nv = fetch(21)\n"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    cold = _n(ticks)
    assert cold == 1, f"the cold run should execute fetch once, got {cold}"

    if wait:
        time.sleep(wait)
    nb_runner.run_cell(3)
    assert nb_runner.peek("v") == "42", "the value changed, so this measures the wrong thing"
    return _n(ticks) - cold


def test_ttl_zero_re_executes_the_call(nb_runner, tmp_path):
    """The bug. With interception on -- the default -- `ttl=0` must still make
    the expensive call run again."""
    assert _run(nb_runner, tmp_path, "# @cash:ttl=0", "t0") == 1, (
        "`# @cash:ttl=0` did not re-execute the intercepted call: the statement "
        "expired but the call was served from an entry with no expiry"
    )


def test_a_long_ttl_still_serves_the_call(nb_runner, tmp_path):
    """CONTROL, and the one that matters most: an over-eager fix that simply
    stopped caching intercepted calls would satisfy every other test here."""
    assert _run(nb_runner, tmp_path, "# @cash:ttl=3600", "t3600") == 0, (
        "a long ttl re-executed the call -- call caching has been broken rather "
        "than made TTL-aware"
    )


def test_a_short_ttl_serves_inside_its_window_and_expires_after(nb_runner, tmp_path):
    """CONTROL: expiry tracks the clock rather than firing unconditionally."""
    assert _run(nb_runner, tmp_path, "# @cash:ttl=30", "t30_inside") == 0, (
        "a re-run well inside the window re-executed the call"
    )
    assert _run(nb_runner, tmp_path, "# @cash:ttl=2", "t2_after", wait=2.5) == 1, (
        "a re-run past the window did not re-execute the call"
    )


def test_an_unannotated_statement_is_unaffected(nb_runner, tmp_path):
    """No annotation means no TTL, which must remain exactly the old
    behaviour -- this is the path every existing notebook takes."""
    assert _run(nb_runner, tmp_path, "# an ordinary comment", "none") == 0


def test_the_statement_and_the_call_expire_together(nb_runner, tmp_path):
    """Both layers honour the same annotation, so adding `no-cache-calls` --
    the workaround this bug forced -- no longer changes the outcome."""
    with_optout = _run(
        nb_runner, tmp_path, "# @cash:no-cache-calls\n# @cash:ttl=0", "t0_optout")
    assert with_optout == 1, "the opt-out arm did not re-execute"
