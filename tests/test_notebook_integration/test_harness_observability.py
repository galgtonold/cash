"""The test harness must report what the kernel IS, not what we asked for.

Three instrument failures, each of which has already cost real work:

1. ``start_kernel(with_cash=...)`` records INTENT. Warm-kernel reuse and the
   kernel pool can both hand back a kernel with cash already installed, so an
   arm that *asked* for cash-off may be running cash-ON. It then reports
   ``warm == 0`` — byte-identical to a genuine cache hit — so the comparison
   silently proves nothing. A verification sweep hit exactly this and its first
   control was worthless.
2. The runner always injected ``__vsc_ipynb_file__``, so cash's no-path branch
   was never exercised by the suite. It shipped an uncaught IndexError that
   disabled caching and printed an internal error on every cell under
   papermill/nbconvert (CAS-205) while thousands of integration tests stayed
   green.
3. There was no ``restart()``, so every restart test hand-rolled
   ``km._async_restart_kernel`` — 9 copies, each free to get the re-injection
   wrong. Restart behaviour is a whole bug class the suite was blind to
   (CAS-190).

These tests pin the instruments themselves. If they fail, every conclusion drawn
with the harness is suspect.
"""
import pytest

pytestmark = [pytest.mark.timeout(240)]


def test_probe_reports_cash_on_when_started_with_cash(nb_runner):
    """The positive arm: cash asked for, cash observed."""
    nb_runner.create_notebook(["x = 1", "print(f'x={x}')"])
    nb_runner.start_kernel(with_cash=True)
    assert nb_runner.probe_cash_active() is True
    nb_runner.assert_cash_active(True)


def test_probe_reports_cash_off_when_started_without(nb_runner):
    """The control arm: this is the reading that was previously unavailable.

    Without this, a control that silently ran cash-ON was indistinguishable
    from a correct one.
    """
    nb_runner.create_notebook(["x = 1", "print(f'x={x}')"])
    nb_runner.start_kernel(with_cash=False)
    assert nb_runner.probe_cash_active() is False
    nb_runner.assert_cash_active(False)


def test_assert_cash_active_fails_loudly_on_a_mismatched_control(nb_runner):
    """The guard must RAISE, not warn — a wrong control is worse than no control.

    This is the test that would have caught the sweep's blind control.
    """
    nb_runner.create_notebook(["x = 1"])
    nb_runner.start_kernel(with_cash=True)
    with pytest.raises(AssertionError, match="same warm-count as a genuine cache hit"):
        nb_runner.assert_cash_active(False)


def test_notebook_path_injection_can_be_disabled(nb_runner):
    """The papermill/nbconvert environment must be reachable from the suite.

    With injection off, ``__vsc_ipynb_file__`` must genuinely be absent — that
    absence is the whole point, and it is what CAS-205 needed to reproduce.
    """
    nb_runner.create_notebook([
        "print('HAS_PATH=' + str('__vsc_ipynb_file__' in dir()))",
    ])
    nb_runner.start_kernel(with_cash=True, inject_notebook_path=False)
    nb_runner.run_all()
    assert "HAS_PATH=False" in nb_runner.get_output(1)


def test_notebook_path_is_injected_by_default(nb_runner):
    """Control for the above: the default really does inject."""
    nb_runner.create_notebook([
        "print('HAS_PATH=' + str('__vsc_ipynb_file__' in dir()))",
    ])
    nb_runner.start_kernel(with_cash=True)
    nb_runner.run_all()
    assert "HAS_PATH=True" in nb_runner.get_output(1)


def test_cash_survives_restart_and_state_is_cleared(nb_runner):
    """``restart()`` really restarts: in-kernel state is gone afterwards."""
    nb_runner.create_notebook([
        "marker = 'before'",
        "print('MARKER=' + globals().get('marker', 'ABSENT'))",
    ])
    nb_runner.start_kernel(with_cash=True)
    nb_runner.run_all()
    assert "MARKER=before" in nb_runner.get_output(2)

    nb_runner.restart()

    nb_runner.run_cell(2)
    assert "MARKER=ABSENT" in nb_runner.get_output(2), "kernel state survived a restart"


def test_restart_preserves_the_no_injection_choice(nb_runner):
    """A no-path run must STAY a no-path run across a restart.

    Re-injecting on restart would silently repair the very environment the test
    is trying to reproduce — the failure would vanish mid-test.
    """
    nb_runner.create_notebook([
        "print('HAS_PATH=' + str('__vsc_ipynb_file__' in dir()))",
    ])
    nb_runner.start_kernel(with_cash=True, inject_notebook_path=False)
    nb_runner.run_all()
    assert "HAS_PATH=False" in nb_runner.get_output(1)

    nb_runner.restart()
    nb_runner.run_cell(1)
    assert "HAS_PATH=False" in nb_runner.get_output(1), "restart re-injected the path"
