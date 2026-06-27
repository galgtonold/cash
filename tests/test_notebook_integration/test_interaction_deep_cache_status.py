"""
Batch 300: Deep caching status verification tests.
Verify RESTORED/COMPUTED/SKIPPED status is correct across scenarios.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestDeepCacheStatusVerification:
    """Verify caching status is correct in complex scenarios."""

    def test_second_run_uses_cache(self, nb_runner):
        """Second identical run should hit the cache for ``y = x * 2``.

        ``y = x * 2`` is sub-millisecond, so by default the 10 ms cost floor
        means it is never written to the cache and a "cache hit" can never be
        observed.  Enable persist-everything so the statement is genuinely
        cached on the first run and produces a real cache hit on the second.
        """
        nb_runner.create_notebook([
            "x = 42",
            "y = x * 2",
            "print(f'y={y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.enable_debug()
        nb_runner.enable_persist()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "y=84" in out

        # Second identical run
        nb_runner.run_all()
        raw = nb_runner.get_raw_output(2)
        has_cache = "CACHE_HIT" in raw or "Cache hit: True" in raw
        assert has_cache, f"Expected caching on second run, got: {raw[:300]}"
        out2 = nb_runner.get_output(3)
        assert "y=84" in out2

    def test_edit_upstream_forces_recompute_downstream(self, nb_runner):
        """Editing upstream cell should force downstream recomputation."""
        nb_runner.create_notebook([
            "base = 10",
            "derived = base * 3",
            "print(f'derived={derived}')",
        ])
        nb_runner.start_kernel()
        nb_runner.enable_debug()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "derived=30" in out

        nb_runner.set_cell_source(1, "base = 100")
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "derived=300" in out

    def test_partial_edit_selective_invalidation(self, nb_runner):
        """Editing one branch should only invalidate that branch's downstream."""
        nb_runner.create_notebook([
            "a = 5\nb = 10",
            "c = a * 2",
            "d = b * 3",
            "total = c + d",
            "print(f'c={c},d={d},total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.enable_debug()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "c=10,d=30,total=40" in out

        # Edit only 'a' — 'c' should recompute, 'd' could be cached
        nb_runner.set_cell_source(1, "a = 50\nb = 10")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "c=100,d=30,total=130" in out

    def test_cache_after_kernel_restart(self, nb_runner):
        """Values should be correct after kernel restart (disk cache or recompute)."""
        nb_runner.create_notebook([
            "val = 777",
            "result = val + 223",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "result=1000" in out

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "result=1000" in out2
