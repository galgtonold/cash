"""Bytes are only worth spending if they buy compute back.

Round 26's five caches held 58 GiB for 61-360 MB of input data. The
adjudication recorded the mechanism as "version retention across ordinary
edits"; reading the entries showed three different mechanisms, none of them
edits, and one thing in common -- every one of them is a population
``cash.backends.versions`` cannot ration:

* s5, 14.1 GB, all ``call:`` entries: seven ``build_features(lags=...)``
  frames, 1.3 GB each for 5.0 s of compute. Call entries carry no
  ``version_slot``, so pruning never applied. **263 MiB per second saved.**
* s4, 9.7 GB, 91% singleton slots: a loop body's source carries
  ``# __iteration_context__: <hash>`` and ``version_slot = f(source_hash)``,
  so every iteration is its own slot with one entry and nothing to supersede.
  72 entries of 48 MiB at ``execution_time 0.00s``. **188 MiB per second.**
* s3, 15.3 GB in slots pruning did see: ``superseded_to_drop`` keeps the
  newest superseded version unconditionally, so 113 slots each kept one spare
  ~1 GB frame against an 83 MiB budget.

So the 64 MiB-per-compute-second yardstick is right and was simply being
applied to the one population that was not the problem. It is now applied at
write time, to every entry.

The constants were measured, not chosen. The distribution separates hard --
p75 is 46 MiB/s, p90 is 430,000 -- so the ceiling's value is decided by its
bottom end: CAS-141's 45.8 MiB array at 0.6 s is 75 MiB/s and must stay
cached, so the ceiling is 128 MiB/s (72% of round 26's bytes refused for 1% of
its compute) rather than the 64 MiB/s that would refuse it. An 8 MiB floor
drops the entries the rule fires on from 739 to 277 while giving up 0.05% of
what it reclaims.

An explicit decision by the user -- ``@cash.cache`` or ``@cash:persist`` -- is
never second-guessed here, the same way it is not second-guessed by the
compute floor or the cost model.
"""
from __future__ import annotations

import pytest

from cash.backends import FileBackend, InMemoryBackend, TieredBackend
from cash.backends.value_policy import (
    WORTH_CEILING_BYTES_PER_SECOND,
    WORTH_FLOOR_BYTES,
    worth_its_bytes,
)

MIB = 1024 ** 2


class TestTheArithmetic:
    """The pure predicate, directly -- every branch without a backend."""

    def test_a_frame_that_rebuilds_in_five_seconds_is_not_worth_a_gigabyte(self):
        """r26s5's entry: 1.3 GB for 5.0 s is 263 MiB per second."""
        assert not worth_its_bytes(1300 * MIB, 5.0)

    def test_a_loop_iteration_that_computes_nothing_is_not_worth_48_mib(self):
        """r26s4's: 48 MiB at execution_time 0.00s."""
        assert not worth_its_bytes(48 * MIB, 0.0)

    def test_a_minute_of_work_is_worth_a_gigabyte(self):
        assert worth_its_bytes(1024 * MIB, 60.0)

    def test_the_ceiling_is_exactly_where_it_says_it_is(self):
        assert worth_its_bytes(WORTH_CEILING_BYTES_PER_SECOND * 10, 10.0)
        assert not worth_its_bytes(WORTH_CEILING_BYTES_PER_SECOND * 10 + 1, 10.0)

    def test_cas141_s_case_stays_cached(self):
        """45.8 MiB in 0.6 s = 75 MiB/s. The regression test for a policy that
        left big frames RAM-only must keep passing; the ceiling sits above it
        deliberately, not by luck."""
        assert worth_its_bytes(48_000_000, 0.6)

    def test_a_live_entry_is_worth_more_per_byte_than_a_spare_copy(self):
        """The two rates are related on purpose: `versions` rations superseded
        copies, which are speculative, at half what a live entry gets."""
        from cash.backends.value_policy import BYTES_PER_COMPUTE_SECOND
        assert WORTH_CEILING_BYTES_PER_SECOND == 2 * BYTES_PER_COMPUTE_SECOND

    @pytest.mark.parametrize("size", [0, 1, WORTH_FLOOR_BYTES - 1])
    def test_nothing_under_the_floor_is_ever_refused(self, size):
        """A small entry costs no disk worth reclaiming, whatever its ratio."""
        assert worth_its_bytes(size, 0.0)

    def test_a_negative_or_missing_cost_does_not_crash(self):
        assert not worth_its_bytes(500 * MIB, -1.0)


class TestThroughABackend:
    """The gate where it actually runs."""

    def _tiered(self, tmp_path):
        return TieredBackend([InMemoryBackend(), FileBackend(str(tmp_path),
                                                             flush_interval=0)])

    def _set(self, backend, key, nbytes, cost, **extra):
        meta = {"execution_time": cost, "size": nbytes, "key": key, **extra}
        backend.set(key, {"variables": {"v": b"x" * nbytes}}, meta)
        return meta

    def test_a_value_that_does_not_earn_its_disk_is_not_persisted(self, tmp_path):
        """Past the compute floor and refused anyway, which is the new part.

        200 MiB for 1.2 s is 167 MiB/s. The floor (1.0 s) and the cost model
        both say persist -- restoring 200 MiB beats recomputing it -- and
        neither of them asks what the answer costs.
        """
        b = self._tiered(tmp_path)
        meta = self._set(b, "spendy", 200 * MIB, 1.2)
        b.shutdown()
        assert "DISK" not in (meta.get("storage") or []), meta
        assert meta.get("persist_skipped") == "bytes", meta
        assert not list(tmp_path.glob("*.entry"))

    def test_the_refusal_says_so(self, tmp_path):
        """Round 26's unanimous complaint was silence, not size."""
        b = self._tiered(tmp_path)
        with pytest.warns(Warning, match="CACHE-NOT-WORTH-BYTES"):
            self._set(b, "spendy", 200 * MIB, 1.2)
        b.shutdown()

    def test_a_value_that_does_earn_it_is_persisted(self, tmp_path):
        """The control: the gate must not simply refuse everything large."""
        b = self._tiered(tmp_path)
        meta = self._set(b, "earned", 40 * MIB, 30.0)
        b.shutdown()
        assert "DISK" in (meta.get("storage") or []), meta
        assert list(tmp_path.glob("*.entry"))

    def test_a_decorated_result_is_still_stored_full_stop(self, tmp_path):
        """`@cash.cache` is the caller having decided; cash honours it."""
        b = self._tiered(tmp_path)
        meta = self._set(b, "dec", 40 * MIB, 0.0, decorator_entry=True)
        b.shutdown()
        assert "DISK" in (meta.get("storage") or []), meta

    def test_cash_persist_is_still_honoured(self, tmp_path):
        """`@cash:persist` likewise -- an explicit instruction, not a guess."""
        b = self._tiered(tmp_path)
        meta = self._set(b, "forced", 40 * MIB, 0.0, force_persist=True)
        b.shutdown()
        assert "DISK" in (meta.get("storage") or []), meta

    def test_a_small_value_is_never_refused_for_its_ratio(self, tmp_path):
        b = self._tiered(tmp_path)
        meta = self._set(b, "small", 64 * 1024, 0.0)
        b.shutdown()
        assert meta.get("persist_skipped") != "bytes", meta


class TestTheDecoratorIsNotReJudged:
    """Through the real decorator, because the exemption is wired in core.

    `TestThroughABackend` sets `decorator_entry` by hand, so it cannot see
    whether `Cash.cache` actually passes it. It does not for a `frozen=True`
    function: that flag carries a SECOND meaning -- "the stored value is what
    the next call hands back", which is what lets the RAM tier skip its
    defensive copy -- and `frozen=True` opts out of it because declaring a
    result frozen already promises exactly that. The rate ceiling then read the
    same flag as "not a decorated entry" and refused the write.

    Nothing about `frozen=` is a statement about persistence, so the two
    functions below must land in the same place.
    """

    def _cash(self, tmp_path, frozen):
        from cash import Cash, CashConfig

        backend = TieredBackend([InMemoryBackend(),
                                 FileBackend(str(tmp_path), flush_interval=0)])
        cash_obj = Cash(backend=backend, config=CashConfig(cache_dir=str(tmp_path)),
                        register_magic=False)

        @cash_obj.cache(frozen=frozen)
        def build(n):
            return b"x" * n

        return cash_obj, backend, build

    @pytest.mark.parametrize("frozen", [False, True])
    def test_a_decorated_result_reaches_disk_however_it_is_declared(
            self, tmp_path, frozen):
        """40 MiB for ~0 s is far over the ceiling -- and exempt either way."""
        cash_obj, backend, build = self._cash(tmp_path, frozen)
        build(40 * MIB)
        backend.shutdown()

        entries = list(tmp_path.rglob("*.entry"))
        assert entries, (
            "frozen=%s: a decorated result was not written to disk; the rate "
            "ceiling re-judged a decision the caller already took" % frozen
        )
