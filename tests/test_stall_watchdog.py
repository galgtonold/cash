"""The stall watchdog must actually fire — and must not fire spuriously.

`timeout = 30` in pyproject is a PER-TEST backstop: it only covers time spent
inside a test, so it cannot see a stall in collection, in worker startup, or in
the xdist master waiting on workers. That is precisely where this suite has
hung — twice, for 15h and 11.5h, producing no output at all.

The watchdog closes that gap, but its failure branch never runs on a healthy
machine, which is exactly how a safety net rots into a no-op unnoticed. These
tests drive both branches directly. `_fire` is patched out throughout: the real
one calls `os._exit`, which would take the test runner with it.
"""
import time

import pytest

from tests.conftest import _StallWatchdog


def _watchdog(timeout):
    """A watchdog whose _fire is recorded instead of killing the process."""
    w = _StallWatchdog(timeout=timeout)
    fired = []
    w._fire = lambda idle, current: fired.append((idle, current))
    return w, fired


def test_fires_when_progress_stops():
    w, fired = _watchdog(0.3)
    w.start()
    time.sleep(1.5)

    assert fired, "watchdog did not fire after progress stopped"
    idle, current = fired[0]
    assert idle >= 0.3
    assert current == "<none yet>"


def test_does_not_fire_while_progress_continues():
    """The suite must not be killed just for being slow."""
    w, fired = _watchdog(0.5)
    w.start()
    for i in range(10):
        w.poke(f"test_{i}")
        time.sleep(0.1)

    assert not fired, f"watchdog fired despite steady progress: {fired}"


def test_poke_records_what_was_running():
    """The dump has to name the last thing seen, or it cannot be acted on."""
    w, fired = _watchdog(0.3)
    w.poke("started tests/test_x.py::test_hangs")
    w.start()
    time.sleep(1.5)

    assert fired
    assert fired[0][1] == "started tests/test_x.py::test_hangs"


def test_disabled_by_zero_timeout():
    w, fired = _watchdog(0)
    w.start()
    time.sleep(0.5)

    assert not fired
    assert not w._started, "a disabled watchdog must not start a thread"


def test_banner_names_the_stall_and_the_escape_hatch():
    w, _ = _watchdog(300)
    banner = w.banner(452.0, "started tests/test_x.py::test_hangs")

    assert "452s" in banner
    assert "300s" in banner
    assert "tests/test_x.py::test_hangs" in banner
    # Whoever hits this needs to know how to raise the limit without
    # hunting through conftest.
    assert "CASH_TEST_STALL_TIMEOUT" in banner


@pytest.mark.parametrize("timeout,expected", [
    (300.0, 5.0),    # long timeout -> capped poll, negligible overhead
    (0.4, 0.1),      # short timeout -> responsive enough to be testable
    (0.01, 0.05),    # floor, so the thread can never busy-spin
])
def test_poll_interval_scales_with_timeout(timeout, expected):
    assert _StallWatchdog(timeout=timeout).poll_interval == pytest.approx(expected)
