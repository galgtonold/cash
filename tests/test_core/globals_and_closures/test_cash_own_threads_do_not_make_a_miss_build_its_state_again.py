"""Cash's own threads do not make a miss build its state segment again.

After a miss, the state the key read is built a second time when another
thread could have changed it while the body ran (`KeyBuilder.ledger_now`).
The check counted every live thread, cash's own among them: once a disk
cache had written anything, its writer and flusher threads were alive, and
every later miss of a single-threaded script paid a second key build.
Only a thread that runs code of the user's counts now; one does still.
"""

from __future__ import annotations

import threading

import pytest

from cash import Cash
from cash.decorator.runtime import KeyBuilder
from cash.tracking import trace_recorder

pytestmark = pytest.mark.core

RATE = 3


def _scaled(x):
    return x * RATE


@pytest.fixture(autouse=True)
def runner_threads_left_out():
    """The test runner's own threads (a plugin's server) are the host's,
    as a kernel's are in a notebook: left out, so only what a test starts
    counts."""
    host = [t for t in threading.enumerate() if t is not threading.current_thread() and not t.name.startswith("cash-")]
    trace_recorder.HOST_THREADS.update(host)
    yield
    for t in host:
        trace_recorder.HOST_THREADS.discard(t)


@pytest.fixture
def rebuilds(monkeypatch):
    calls = []
    real = KeyBuilder.ledger_now

    def counting(self, spec, args):
        calls.append(spec.name)
        return real(self, spec, args)

    monkeypatch.setattr(KeyBuilder, "ledger_now", counting)
    return calls


def test_a_miss_with_only_cash_threads_alive_builds_its_state_once(tmp_path, rebuilds):
    f = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False).cache(_scaled)
    assert f(1) == 3
    assert any(t.name.startswith("cash-") for t in threading.enumerate())  # the writers are up
    rebuilds.clear()
    for x in range(2, 6):
        assert f(x) == 3 * x
    assert rebuilds == []


def test_a_miss_with_a_user_thread_alive_still_builds_its_state_again(tmp_path, rebuilds):
    f = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False).cache(_scaled)
    f(1)
    stop = threading.Event()
    other = threading.Thread(target=stop.wait, args=(30,), daemon=True)
    other.start()
    try:
        rebuilds.clear()
        assert f(7) == 21
    finally:
        stop.set()
        other.join(30)
    assert len(rebuilds) == 1
