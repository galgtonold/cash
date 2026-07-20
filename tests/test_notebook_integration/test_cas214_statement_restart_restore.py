"""CAS-214: a notebook STATEMENT's cache must survive a kernel restart.

Round-9 gate finding, hit independently by 4 of 5 testers. The value is written
to disk and the cache directory is found on the next session -- but the
cross-restart LOOKUP misses, so every expensive statement recomputes and is
written again under a new key. `.cash/` accumulates a fresh copy per restart
(one tester reached 12 GB for a 736 MB input) and the user gets none of the
headline benefit.

The `@cash.cache` DECORATOR path survives restarts correctly, which is exactly
why this hid: the wheel gate's S1/S3/S4 all exercise the decorator.

Ground truth is an EXTERNAL counter appended from inside the loop body and read
from outside the kernel. A print or a badge cannot witness this -- a cache hit
restores prints, so the notebook looks identical either way.
"""
import pytest

pytestmark = [pytest.mark.upstream, pytest.mark.timeout(300)]


def _calls(path):
    """Number of body invocations recorded so far, read from OUTSIDE the kernel."""
    if not path.exists():
        return 0
    return len(path.read_text().strip())


def test_statement_cache_survives_a_kernel_restart(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    cp = str(counter).replace("\\", "/")

    nb_runner.create_notebook([
        "import cash\n%cash_on",
        (
            "import time\n"
            "def slow(i):\n"
            "    time.sleep(0.01)\n"
            f"    with open(r'{cp}', 'a') as fh:\n"
            "        fh.write('X')\n"
            "    return i * 2\n"
            "result = [slow(i) for i in range(30)]\n"
            "print('sum', sum(result))"
        ),
    ])
    nb_runner.start_kernel()

    nb_runner.run_all()
    cold = _calls(counter)
    assert cold == 30, f"cold run should invoke the body 30x, got {cold}"

    # Warm, same session: the statement must restore.
    nb_runner.run_cell(2)
    warm = _calls(counter)
    assert warm == cold, (
        f"warm re-run in the SAME session recomputed: {cold} -> {warm}. "
        f"If this fails the bug is broader than CAS-214."
    )

    # The real test: a kernel restart. cash is disabled by a restart, so the
    # %cash_on cell is re-run first -- that is expected, documented behaviour.
    nb_runner.restart()
    nb_runner.run_cell(1)
    nb_runner.run_cell(2)

    after_restart = _calls(counter)
    assert after_restart == warm, (
        f"statement did NOT restore after a kernel restart: {warm} -> "
        f"{after_restart} body invocations. The value is on disk; the "
        f"cross-restart lookup missed and recomputed everything."
    )
