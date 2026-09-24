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

    assert slept == [sd._SAVE_POLL_INTERVAL_S], f"expected exactly one poll of a stable fresh file, got {slept}"


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
                f.write(" ")  # size changes -> writer still in flight
            real_sleep(0.001)  # let the mtime actually move

    monkeypatch.setattr(sd._time, "sleep", fake_sleep)

    sd._wait_for_notebook_save(str(nb))

    # One poll per growth step, plus the final poll that observed no change.
    assert calls["n"] == stop_growing_after + 1, (
        f"returned after {calls['n']} polls while the file was still being written (expected {stop_growing_after + 1})"
    )


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


def test_a_write_landing_mid_wait_is_waited_out(nb, monkeypatch):
    """Sanity: a real concurrent writer thread and real sleeps.

    Guards against the fake-sleep tests above passing for reasons that have
    nothing to do with the real polling loop: the writes come from another
    thread, land while the poller really sleeps, and move the file's real
    mtime and size.

    The writer runs in lockstep with the poller, one append per poll. A
    free-running writer on a real clock made this test fail under load: a
    writer thread starved for longer than one poll interval leaves the file
    unchanged across two polls, which is exactly what "the save has settled"
    looks like, so the wait returned early for a reason the test did not
    control. In lockstep every poll but the last finds the file changed, so
    the test fails only when the loop stops polling a changing file.
    """
    n_writes = 5
    real_sleep = time.sleep  # the hook below replaces time.sleep everywhere
    # Keep the two timing gates, which have their own tests above, out of
    # this one: a slow runner must not make the file look stale before the
    # wait starts, or run into the cap while the writer still has turns left.
    monkeypatch.setattr(sd, "_SAVE_FRESH_WINDOW_S", 60.0)
    monkeypatch.setattr(sd, "_SAVE_MAX_WAIT_S", 60.0)

    your_turn = threading.Semaphore(0)
    wrote = threading.Semaphore(0)
    done = threading.Event()

    def writer():
        for _ in range(n_writes):
            if not your_turn.acquire(timeout=30):
                return  # the poller stopped handing out turns; the asserts say so
            with open(nb, "a", encoding="utf-8") as f:
                f.write("x")
            wrote.release()
        done.set()

    polls = []

    def poll(interval):
        polls.append(interval)
        if len(polls) <= n_writes:
            your_turn.release()  # the write lands while this poll sleeps
            real_sleep(interval)
            # A starved writer delays the poll; it cannot pass for a settled file.
            assert wrote.acquire(timeout=30), "the writer thread never made its write"
        else:
            real_sleep(interval)

    monkeypatch.setattr(sd._time, "sleep", poll)
    t = threading.Thread(target=writer, daemon=True)
    t.start()
    sd._wait_for_notebook_save(str(nb))
    # Read the flag BEFORE joining. Joining first would let the writer finish
    # on its own and the assertion would hold however early the wait returned
    # -- the test would pass against a wait that does nothing at all.
    settled_before_return = done.is_set()
    for _ in range(n_writes):
        your_turn.release()  # let a writer left waiting finish, then join it
    t.join(timeout=5)

    assert settled_before_return, f"returned after {len(polls)} polls while the writer was still appending"
    # One poll per write, plus the one that found the file unchanged.
    assert polls == [sd._SAVE_POLL_INTERVAL_S] * (n_writes + 1), polls
    assert nb.read_text(encoding="utf-8") == "{}" + "x" * n_writes
