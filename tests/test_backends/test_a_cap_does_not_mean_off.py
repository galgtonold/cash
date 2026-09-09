"""Setting `max_cache_size` must bound the cache, not switch it off.

Round-15 gate finding. `CASH_MAX_CACHE_SIZE=500MB` on a nightly job whose
working set is 263 MB cached **nothing**: three of four stages recomputed every
run, the cache directory held 29 KB, and the operator's reading of their own
setting -- "I capped it, so it is evicting" -- was the opposite of what
happened. 10/10 on the probe sweep, 5/5 on the real job.

The cause was a per-entry refusal threshold of HALF the cap (CAS-142's guard
against one big entry leaving less than half the cache for everything else).
The threshold is now the whole cap: an entry that fits is stored and LRU does
its job, and only an entry that cannot fit at all is refused. The treadmill
that motivated the old threshold is still caught when it actually happens, by
the evict-within-a-couple-of-writes warning in
``test_cache_cap_safety.TestEvictAfterWrite``.

This file asserts the property at the level the report was written at -- a
cache directory that a SECOND instance can read -- because that is what
"caching is on" means to the person who set the variable. The unit-level arms
live beside the old threshold in ``test_cache_cap_safety.py``.
"""
from __future__ import annotations

import time
import warnings

import pytest

from cash import Cash
from cash.backends.factory import build_backend_from_config
from cash.config import CashConfig


def _instance(tmp_path, cap):
    """A tiered RAM+disk stack over one directory, with an explicit cap."""
    backend = build_backend_from_config(
        CashConfig(cache_dir=str(tmp_path / "cache"), max_cache_size=cap)
    )
    return Cash(backend=backend, register_magic=False)


PAYLOAD_BYTES = 400_000


@pytest.mark.parametrize("cap,should_persist", [
    (PAYLOAD_BYTES * 4, True),      # comfortable: caching must work
    (PAYLOAD_BYTES * 3 // 2, True), # over half the cap -- the reported case
    (PAYLOAD_BYTES // 2, False),    # genuinely too big to fit: refused
], ids=["roomy", "over-half", "over-cap"])
def test_a_capped_cache_still_serves_a_second_instance(tmp_path, cap, should_persist):
    """A fresh instance over the same directory is the honest test of "cached".

    An in-process second call hits the RAM tier whatever the disk tier did, so
    it cannot tell "capped" from "switched off" -- which is exactly the
    distinction the report is about.
    """
    runs: list[int] = []
    writer = _instance(tmp_path, cap)

    @writer.cache(assume_safe=True)
    def build(n):
        runs.append(n)
        # Past the compute floor: cash only promotes a result past RAM when
        # computing it cost more than restoring it will. Without this the arms
        # below recompute for a reason that has nothing to do with the cap.
        time.sleep(0.2)
        return "x" * PAYLOAD_BYTES

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert len(build(1)) == PAYLOAD_BYTES
        writer.backend.shutdown()

        reader = _instance(tmp_path, cap)

        @reader.cache(assume_safe=True)
        def build(n):                                  # noqa: F811 - same body
            runs.append(n)
            time.sleep(0.2)
            return "x" * PAYLOAD_BYTES

        assert len(build(1)) == PAYLOAD_BYTES
        reader.backend.shutdown()

    if should_persist:
        assert len(runs) == 1, (
            f"a {cap}-byte cap did not store a {PAYLOAD_BYTES}-byte entry, so "
            f"capping the cache switched it off"
        )
    else:
        assert len(runs) == 2, "an entry larger than the whole cap cannot be stored"


def test_the_cap_is_compared_against_the_size_it_governs(tmp_path):
    """The other half of the report: two sizes, and the wrong one was used.

    The size gate compared the value's IN-MEMORY footprint against a DISK cap,
    while `cash inspect` reports serialized bytes -- so a tester saw a 160 MB
    entry refused by a 500 MB cap, with no number anywhere that explained it.
    A list of distinct short strings has the same shape at test scale: about
    3x larger in memory than pickled.

    The gate now measures the serialized size before refusing, so a cap set
    between the two numbers stores the value rather than dropping it.
    """
    import pickle

    from cash.backends.memory_backend import InMemoryBackend
    from cash.backends.tiered_backend import TieredBackend

    value = [f"string-number-{i}" for i in range(40_000)]
    in_memory = InMemoryBackend()._get_object_size(value)
    serialized = len(pickle.dumps(value))
    assert serialized < in_memory / 2, (
        "the fixture must actually reproduce the gap between the two sizes"
    )

    from cash.backends.file_backend import FileBackend
    cap = (in_memory + serialized) // 2      # fits on disk, "too big" in RAM
    disk = FileBackend(str(tmp_path / "c"), max_size_bytes=cap, flush_interval=0)
    tiered = TieredBackend(
        [InMemoryBackend(), disk], promotion_policy=lambda exec_t, size: True,
    )

    meta = {"execution_time": 2.0, "size": in_memory}
    tiered.set("k", value, meta)
    disk._writes.wait_all()

    assert "DISK" in meta.get("storage", []), (
        f"an entry of {serialized} serialized bytes was refused by a {cap}-byte "
        f"cap because it takes {in_memory} bytes in memory"
    )
    assert disk.get("k")[1] == value
    disk.shutdown()
