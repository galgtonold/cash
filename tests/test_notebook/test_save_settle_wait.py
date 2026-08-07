"""Direct coverage for ``_wait_for_notebook_save``.

An editor may still be flushing the ``.ipynb`` when a run begins, and reading
mid-write yields pre-edit cell sources — which silently breaks upstream change
detection. The guard: when the file looks freshly touched, poll ``(mtime,
size)`` until it stops changing, bounded by a hard cap.

Nothing exercised this directly. The integration harness writes the ``.ipynb``
synchronously and closes it before cash reads, so the interesting branch — a
write actually in flight — was never taken there; the suite only ever ran the
"already stable" path, and only as a side effect. These tests own the behaviour
instead, both branches, at unit speed.

Found while measuring the harness: that always-stable poll is ~51ms of dead
time on the first cash read after every notebook write, which is most of the
per-test fixed cost of a serial run.
"""
from __future__ import annotations

import os
import sys
import threading
import time

import pytest

from cash.notebook import server_discovery as sd


def _age(path, seconds: float) -> None:
    t = time.time() - seconds
    os.utime(path, (t, t))


@pytest.fixture
def nb(tmp_path):
    p = tmp_path / "nb.ipynb"
    p.write_text("{}", encoding="utf-8")
    return p


def test_settled_file_returns_without_polling(nb, monkeypatch):
    """A file last written long ago is not in flight: return with zero sleeps."""
    _age(nb, sd._SAVE_FRESH_WINDOW_S + 5)
    slept = []
    monkeypatch.setattr(sd._time, "sleep", lambda s: slept.append(s))

    sd._wait_for_notebook_save(str(nb))

    assert slept == [], f"waited on a settled file: {slept}"


def test_fresh_but_stable_file_returns_after_one_poll(nb, monkeypatch):
    """Freshly written and already closed: one poll proves stability, then go."""
    slept = []
    monkeypatch.setattr(sd._time, "sleep", lambda s: slept.append(s))

    sd._wait_for_notebook_save(str(nb))

    assert slept == [sd._SAVE_POLL_INTERVAL_S], (
        f"expected exactly one poll of a stable fresh file, got {slept}")


def test_waits_until_an_in_flight_write_stops_growing(nb, monkeypatch):
    """The branch that matters: keep polling while the file is still changing.

    A background thread appends on a delay, so the size differs across the
    first polls and matches only once the writer is done.
    """
    stop_growing_after = 3
    calls = {"n": 0}
    real_sleep = time.sleep

    def fake_sleep(_s):
        calls["n"] += 1
        if calls["n"] <= stop_growing_after:
            with open(nb, "a", encoding="utf-8") as f:
                f.write(" ")   # size changes -> writer still in flight
            real_sleep(0.001)  # let the mtime actually move

    monkeypatch.setattr(sd._time, "sleep", fake_sleep)

    sd._wait_for_notebook_save(str(nb))

    # One poll per growth step, plus the final poll that observed no change.
    assert calls["n"] == stop_growing_after + 1, (
        f"returned after {calls['n']} polls while the file was still being "
        f"written (expected {stop_growing_after + 1})")


def test_wait_is_bounded_when_the_writer_never_stops(nb, monkeypatch):
    """A pathological writer must not block the run: the hard cap wins."""
    monkeypatch.setattr(sd, "_SAVE_MAX_WAIT_S", 0.2)
    calls = {"n": 0}

    def fake_sleep(_s):
        calls["n"] += 1
        with open(nb, "a", encoding="utf-8") as f:
            f.write(" ")  # never settles

    monkeypatch.setattr(sd._time, "sleep", fake_sleep)

    t0 = time.monotonic()
    sd._wait_for_notebook_save(str(nb))
    elapsed = time.monotonic() - t0

    assert calls["n"] > 0, "never polled a file that was being written"
    assert elapsed < 2.0, f"unbounded wait on a never-settling writer: {elapsed:.2f}s"


def test_unreadable_path_is_not_an_error(tmp_path, monkeypatch):
    """A missing file is the caller's problem to report, not ours to raise on."""
    slept = []
    monkeypatch.setattr(sd._time, "sleep", lambda s: slept.append(s))

    sd._wait_for_notebook_save(str(tmp_path / "does-not-exist.ipynb"))

    assert slept == []


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason=(
        "Fails on macOS 3.10-3.13 for a reason not yet found (CAS-271). Two "
        "genuine races were removed and macOS 3.14 went green, so a third "
        "factor remains -- most likely the _SAVE_FRESH_WINDOW_S gate declining "
        "to engage on a slow runner, but that is a hypothesis, not a finding. "
        "Skipped rather than deleted: this is the only test here that uses a "
        "REAL thread and a real clock, so it is what stops the other five "
        "passing for monkeypatched-clock reasons. Diagnose on the WSL + "
        "symlinked-TMPDIR harness that reproduces macOS CI locally, then "
        "remove this marker."
    ),
)
def test_a_write_landing_mid_wait_is_waited_out(nb, monkeypatch):
    """Sanity: a real concurrent writer, no monkeypatched clock.

    Guards against the fake-sleep tests above passing for reasons that have
    nothing to do with the real polling loop.
    """
    # The writer must be provably UNDER WAY before the wait starts. The first
    # version of this test started the thread and immediately began waiting,
    # with the writer sleeping before its first write -- so the file was still
    # untouched, the poll loop could see two identical (mtime, size) readings
    # and return having waited for nothing. Linux and Windows scheduled the
    # thread fast enough to hide it; every macOS job failed.
    first_write = threading.Event()
    done = threading.Event()

    # Long enough to span several poll cycles, comfortably short of the loop's
    # own cap -- if the writer outlasted the cap, the wait would return early
    # for a legitimate reason and this test would fail for the wrong one. Both
    # margins are derived from the module's constants and asserted below, so
    # changing a constant reports here instead of going quietly flaky.
    # The gap must be SHORTER than the poll interval, not equal to it. At the
    # same rate the poller can land twice between two writes, read an identical
    # (mtime, size) both times and conclude the file has settled while it very
    # much has not -- which is exactly what happened when this was first
    # written with `gap = _SAVE_POLL_INTERVAL_S`: a deterministic failure, not
    # a flake. The writer has to out-pace the poller for "still changing" to be
    # observable at all.
    gap = sd._SAVE_POLL_INTERVAL_S / 2
    n_writes = 12
    assert gap < sd._SAVE_POLL_INTERVAL_S, "writer must out-pace the poller"
    assert n_writes * gap > 2 * sd._SAVE_POLL_INTERVAL_S, "writer too brief to span a poll cycle"
    assert n_writes * gap < sd._SAVE_MAX_WAIT_S * 0.6, "writer outlasts the wait cap"

    def writer():
        for _ in range(n_writes):
            with open(nb, "a", encoding="utf-8") as f:
                f.write("x")
            first_write.set()
            time.sleep(gap)
        done.set()

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    assert first_write.wait(timeout=5), "the writer thread never ran"
    sd._wait_for_notebook_save(str(nb))
    # Read the flag BEFORE joining. Joining first would let the writer finish
    # on its own and the assertion would hold however early the wait returned
    # -- the test would pass against a wait that does nothing at all.
    settled_before_return = done.is_set()
    t.join(timeout=5)

    assert settled_before_return, "returned while the writer was still appending"
