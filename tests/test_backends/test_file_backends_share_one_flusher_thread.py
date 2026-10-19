"""Every `FileBackend` in a process is flushed by one thread.

Each backend started its own thread to write access stamps back every few
seconds, and nothing ended it: a `Cash` keeps its backend for the exit drain.
A process that built hundreds of backends (a test worker held about 500 such
threads) had them all waking and competing with the caller's thread, and tests
that take a tenth of a second alone ran past their 30 s timeout.
"""

from __future__ import annotations

import threading
import time

from cash.backends.file_backend import FileBackend


def _flush_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "cash-flusher"]


def _used(path, interval=5) -> FileBackend:
    backend = FileBackend(str(path), flush_interval=interval)
    backend.set("k", "v")
    backend._writes.wait_all()
    return backend


def test_fifty_backends_run_one_flusher_thread(tmp_path):
    backends = [_used(tmp_path / str(i)) for i in range(50)]
    assert len(_flush_threads()) == 1
    for backend in backends:
        backend.shutdown()


def test_a_backend_s_reads_are_still_flushed_within_its_interval(tmp_path):
    backend = _used(tmp_path, interval=1)
    assert backend.get("k")[1] == "v"
    deadline = time.monotonic() + 10
    while "k" not in backend._access_flushed and time.monotonic() < deadline:
        time.sleep(0.05)
    assert "k" in backend._access_flushed, "the read's stamp was never flushed"
    backend.shutdown()


def test_shutdown_waits_for_a_flush_in_flight(tmp_path):
    backend = _used(tmp_path, interval=1)
    order = []
    backend._flush_lock.acquire()

    def end_the_flush():
        time.sleep(0.2)
        order.append("flush ended")
        backend._flush_lock.release()

    threading.Thread(target=end_the_flush).start()
    backend.shutdown()
    order.append("shutdown returned")
    assert order == ["flush ended", "shutdown returned"]


def test_a_flush_that_fails_stops_no_other_backend_s(tmp_path, monkeypatch):
    broken = _used(tmp_path / "broken", interval=1)
    healthy = _used(tmp_path / "healthy", interval=1)

    def fail(*args, **kwargs):
        raise OSError("disk gone")

    monkeypatch.setattr(broken, "_flush_metadata", fail)
    assert healthy.get("k")[1] == "v"
    deadline = time.monotonic() + 10
    while "k" not in healthy._access_flushed and time.monotonic() < deadline:
        time.sleep(0.05)
    assert "k" in healthy._access_flushed
    broken._stop_event.set()
    healthy.shutdown()
