"""Under memory pressure, the RAM tier gives back its share -- not everything.

The pressure check reads the WHOLE MACHINE's memory. It used to drop entries
until the machine fell under the target, which never happens when the
pressure is someone else's: one check emptied the tier, and every later check
emptied it again.

Five people once shared one box, each with a cash kernel and an uncached
oracle kernel over its full dataset. One parameter sweep went 13.4 s ->
~100 s and stayed there through reruns, and through reverting the edit they
blamed for it, while the uncached kernel beside it barely moved; its
`cached=` count fell from 601 to ~170 and never recovered. A clean replay of
the same seven steps on a quiet machine stays at 9-12 s throughout.

Every reading here carries ``total``, as psutil's always does; a reading
without one takes the old path, and ``test_memory_eviction.py`` covers that.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cash.backends import memory_backend
from cash.backends.memory_backend import InMemoryBackend

MB = 1_000_000
GB = 1_000 * MB


def _put(backend, key, size, seconds):
    backend.set(key, bytes(size), {"execution_time": seconds})


def _held(backend, key):
    return backend.peek_metadata(key) is not None


@pytest.fixture
def machine(monkeypatch):
    """A 16 GB machine whose memory reading the test sets."""
    state = SimpleNamespace(percent=50.0, total=16 * GB)
    monkeypatch.setattr(
        memory_backend,
        "psutil",
        SimpleNamespace(virtual_memory=lambda: SimpleNamespace(percent=state.percent, total=state.total)),
    )
    return state


def _tier(n=100, size=MB):
    """n entries of *size*, checking pressure on every write."""
    b = InMemoryBackend(max_memory_percent=0.9, check_interval=1)
    return b


class TestPressureThatIsNotOurs:
    """The machine is full and this tier is a sliver of it."""

    def test_one_check_does_not_empty_the_tier(self, machine):
        b = _tier()
        for i in range(100):
            _put(b, "k%03d" % i, MB, 1.0)
        assert len(b._store) == 100

        machine.percent = 95.0  # 15.2 GB in use; this tier holds 100 MB
        b._check_and_evict()

        # Its share of a 2.24 GB overshoot is 2.24 GB * 100 MB / 15.2 GB ~ 15 MB.
        held = len(b._store)
        assert held >= 80, (
            "a tier holding 0.7% of the memory in use gave back %d of 100 "
            "entries to pressure it cannot relieve" % (100 - held)
        )
        assert held < 100, "it should still give back its share"

    def test_sustained_pressure_holds_the_tier_flat_instead_of_draining_it(self, machine):
        """The sweep's case: hundreds of writes, pressure the whole time."""
        b = _tier()
        for i in range(100):
            _put(b, "k%03d" % i, MB, 1.0)

        machine.percent = 95.0
        for i in range(400):  # a check on every one of these writes
            _put(b, "new%03d" % i, MB, 1.0)

        assert len(b._store) >= 70, (
            "400 writes under sustained external pressure drained the tier to "
            "%d entries; each check took its share again" % len(b._store)
        )

    def test_new_writes_displace_the_least_valuable_old_ones(self, machine):
        b = _tier()
        _put(b, "costly", MB, 30.0)
        for i in range(99):
            _put(b, "cheap%02d" % i, MB, 0.001)

        machine.percent = 95.0
        for i in range(50):
            _put(b, "new%02d" % i, MB, 0.001)

        assert _held(b, "costly"), "the 30 s result went before 1 ms ones"


class TestPressureThatIsOurs:
    """This tier IS most of the memory in use."""

    def test_it_still_gives_back_nearly_the_whole_overshoot(self, machine):
        machine.total = 1 * GB  # a small machine this tier fills
        b = _tier()
        for i in range(900):
            _put(b, "k%03d" % i, MB, 1.0)  # 900 MB of a 950 MB footprint

        machine.percent = 95.0  # 950 MB in use; target is 81%, 810 MB
        b._check_and_evict()

        freed = (900 - len(b._store)) * MB
        # Overshoot 140 MB; its share is 140 MB * 900/950 ~ 133 MB.
        assert freed >= 120 * MB, "a tier that is 95%% of the pressure freed only %d MB of a 140 MB overshoot" % (
            freed // MB
        )


class TestEpisodes:
    def test_pressure_that_keeps_climbing_takes_a_fresh_share(self, machine):
        """Holding flat is for STEADY pressure.

        If the machine keeps filling, something is still growing -- possibly
        this tier, if the memory it freed has not gone back to the OS -- and
        refusing to shed again would let the machine swap. The old loop erred
        towards emptying the cache; the new one must not err the other way.
        """
        b = _tier()
        for i in range(100):
            _put(b, "k%03d" % i, MB, 1.0)

        machine.percent = 91.0
        b._check_and_evict()
        after_first = len(b._store)

        machine.percent = 91.5  # steady: jitter, not worsening
        b._check_and_evict()
        assert len(b._store) == after_first, "steady pressure must hold the tier flat"

        machine.percent = 97.0  # worse by 6 points
        b._check_and_evict()
        assert len(b._store) < after_first, "pressure climbed from 91% to 97% and the tier gave back nothing more"

    def test_an_episode_ends_when_the_pressure_does(self, machine):
        b = _tier()
        for i in range(100):
            _put(b, "k%03d" % i, MB, 1.0)

        machine.percent = 95.0
        b._check_and_evict()
        floor = b._pressure_floor
        assert floor is not None

        machine.percent = 50.0
        b._check_and_evict()
        assert b._pressure_floor is None, "the next episode must take a fresh share"

    def test_clearing_the_tier_ends_the_episode(self, machine):
        b = _tier()
        _put(b, "k", MB, 1.0)
        machine.percent = 95.0
        b._check_and_evict()
        b.clear()
        assert b._pressure_floor is None
