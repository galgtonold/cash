"""Machine-scaled cache caps (CAS-142).

A single 1 GiB ``max_cache_size`` used to cap *every* tier, so the disk tier
was pinned at one medium DataFrame and persist-heavy workloads thrashed
(write an entry, immediately LRU-evict it — slower than no cache). These
tests pin the pure clamp arithmetic, the psutil-absent fallback, that the
factory now gives the RAM and disk tiers *different* machine-scaled caps, and
that an explicit ``max_cache_size`` still wins.
"""
from __future__ import annotations

import tempfile
from types import SimpleNamespace

import pytest

from cash.backends import adaptive_caps as ac

_MIB = 1024 ** 2
_GIB = 1024 ** 3


# ---------------------------------------------------------------------------
# Pure disk-cap policy — every clamp branch, deterministic.
# ---------------------------------------------------------------------------

class TestAdaptiveDiskCap:
    def test_floor_branch_small_laptop(self):
        # 20 GiB free: 0.25·20 = 5 GiB < 8 GiB floor → floor wins.
        assert ac.adaptive_disk_cap(20 * _GIB) == 8 * _GIB

    def test_fraction_branch_midrange(self):
        # 60 GiB free: 0.25·60 = 15 GiB, above floor, below ceiling → fraction.
        assert ac.adaptive_disk_cap(60 * _GIB) == 15 * _GIB

    def test_ceiling_branch_workstation(self):
        # 2 TiB free: 0.25·2 TiB = 512 GiB → clamped to the 100 GiB ceiling.
        assert ac.adaptive_disk_cap(2048 * _GIB) == 100 * _GIB

    def test_safety_branch_tiny_disk(self):
        # 4 GiB free: floor (8 GiB) would exceed the disk, so SAFETY (0.8·4)
        # wins — the cap never claims more than the volume can give.
        cap = ac.adaptive_disk_cap(4 * _GIB)
        assert cap == int(0.8 * 4 * _GIB)
        assert cap < 4 * _GIB  # strictly under free space
        assert abs(cap / _GIB - 3.2) < 0.01

    def test_zero_free_returns_floor_not_zero(self):
        # Unmeasurable free space → floor (generous), never a 0 cap that would
        # evict on the first write.
        assert ac.adaptive_disk_cap(0) == 8 * _GIB
        assert ac.adaptive_disk_cap(-1) == 8 * _GIB


# ---------------------------------------------------------------------------
# Pure RAM-cap policy — clamps + psutil-absent fallback.
# ---------------------------------------------------------------------------

class TestAdaptiveRamCap:
    def test_fraction_branch(self):
        # 16 GiB RAM: 0.20·16 = 3.2 GiB, within [512 MiB, 4 GiB].
        assert ac.adaptive_ram_cap(16 * _GIB) == int(0.20 * 16 * _GIB)

    def test_floor_branch(self):
        # 2 GiB RAM: 0.20·2 = 0.4 GiB < 512 MiB floor → floor.
        assert ac.adaptive_ram_cap(2 * _GIB) == 512 * _MIB

    def test_ceiling_branch(self):
        # 64 GiB RAM: 0.20·64 = 12.8 GiB → clamped to the 4 GiB ceiling.
        assert ac.adaptive_ram_cap(64 * _GIB) == 4 * _GIB

    def test_psutil_absent_fallback(self):
        # None (psutil unavailable) → the fixed 1 GiB fallback, no import.
        assert ac.adaptive_ram_cap(None) == ac.RAM_FALLBACK == _GIB

    def test_zero_total_fallback(self):
        assert ac.adaptive_ram_cap(0) == ac.RAM_FALLBACK


# ---------------------------------------------------------------------------
# Resolvers — mocked disk_usage / psutil, never the real machine.
# ---------------------------------------------------------------------------

class TestResolvers:
    def test_resolve_disk_cap_uses_mocked_free_space(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            ac.shutil, "disk_usage",
            lambda p: SimpleNamespace(total=0, used=0, free=2048 * _GIB),
        )
        assert ac.resolve_disk_cap(str(tmp_path)) == 100 * _GIB  # ceiling

    def test_resolve_disk_cap_walks_up_to_existing_parent(self, monkeypatch, tmp_path):
        calls = []

        def fake_disk_usage(p):
            calls.append(p)
            if "does-not-exist" in str(p):
                raise FileNotFoundError(p)
            return SimpleNamespace(total=0, used=0, free=60 * _GIB)

        monkeypatch.setattr(ac.shutil, "disk_usage", fake_disk_usage)
        missing = tmp_path / "does-not-exist" / "cache"
        assert ac.resolve_disk_cap(str(missing)) == 15 * _GIB  # 0.25·60
        assert len(calls) >= 2  # walked up past the missing dir

    def test_resolve_ram_cap_with_mocked_psutil(self, monkeypatch):
        monkeypatch.setattr(ac, "_total_system_ram", lambda: 16 * _GIB)
        assert ac.resolve_ram_cap() == int(0.20 * 16 * _GIB)

    def test_resolve_ram_cap_psutil_import_absent(self, monkeypatch):
        # Simulate psutil being unimportable (bare install, CAS-129).
        import builtins
        real_import = builtins.__import__

        def blocked_import(name, *a, **k):
            if name == "psutil":
                raise ImportError("blocked")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", blocked_import)
        assert ac._total_system_ram() is None
        assert ac.resolve_ram_cap() == ac.RAM_FALLBACK


# ---------------------------------------------------------------------------
# Factory integration — RAM and disk tiers get DIFFERENT machine-scaled caps,
# and an explicit max_cache_size still pins the disk tier.
# ---------------------------------------------------------------------------

class TestFactoryCapWiring:
    def _build(self, config):
        from cash.backends.factory import build_backend_from_config
        return build_backend_from_config(config)

    def test_ram_and_disk_caps_differ_and_disk_exceeds_1gib(self, monkeypatch, tmp_path):
        from cash.backends import adaptive_caps
        # Big free disk → disk cap ≫ 1 GiB; modest RAM → its own smaller cap.
        monkeypatch.setattr(adaptive_caps, "_free_bytes_on_volume", lambda p: 500 * _GIB)
        monkeypatch.setattr(adaptive_caps, "_total_system_ram", lambda: 16 * _GIB)
        from cash.config import CashConfig
        backend = self._build(CashConfig(cache_dir=str(tmp_path / "c")))
        ram, disk = backend.backends[0], backend.backends[1]
        assert disk._max_size_bytes > _GIB, "the core fix: disk tier no longer 1 GiB"
        assert disk._max_size_bytes == 100 * _GIB  # 0.25·500 → ceiling
        assert ram._max_size_bytes == int(0.20 * 16 * _GIB)
        assert ram._max_size_bytes != disk._max_size_bytes
        backend.shutdown()

    def test_explicit_max_cache_size_pins_disk_not_ram(self, monkeypatch, tmp_path):
        from cash.backends import adaptive_caps
        monkeypatch.setattr(adaptive_caps, "_total_system_ram", lambda: 16 * _GIB)
        from cash.config import CashConfig
        backend = self._build(CashConfig(cache_dir=str(tmp_path / "c"), max_cache_size=777_000))
        ram, disk = backend.backends[0], backend.backends[1]
        assert disk._max_size_bytes == 777_000  # explicit value honored
        # RAM tier keeps its own modest auto cap regardless.
        assert ram._max_size_bytes == int(0.20 * 16 * _GIB)
        backend.shutdown()
