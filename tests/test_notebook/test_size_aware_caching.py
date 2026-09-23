"""Size-aware caching: a value whose restore would cost more than recomputing it is
not stored.

The StatementProcessor's store skips a large object when its predicted restore
time exceeds both the savings ratio budget and the fixed budget, unless the
compute was expensive enough or the statement is marked ``@cash:persist``.
"""

from unittest.mock import MagicMock


class TestSizeAwareCaching:
    """Tests for size-aware caching threshold in StatementProcessor."""

    def test_estimate_object_size_int(self):
        """Integer size estimation."""
        from cash.object_hashing import estimate_object_size

        assert estimate_object_size(42) > 0

    def test_estimate_object_size_str(self):
        """String size estimation."""
        from cash.object_hashing import estimate_object_size

        size = estimate_object_size("hello" * 1000)
        assert size > 5000  # At least 5KB for a 5000-char string

    def test_estimate_object_size_list(self):
        """List size estimation."""
        from cash.object_hashing import estimate_object_size

        size = estimate_object_size(list(range(10000)))
        assert size > 0

    def test_should_skip_large_object_basic(self):
        """A large object whose predicted restore exceeds both budgets is skipped.

        The numbers here moved when the cost model was refitted.
        This used to use a 100 MB ndarray with a 0.1 s compute, on the strength
        of a model that predicted over 160 ms to restore it; a fresh-process read
        of that array measures ~79 ms, so persisting it genuinely pays and the
        old assertion was pinning a mispricing. A 400 MB array keeps the case the
        test is about -- restore predicted well past the 80 ms ratio budget and
        the 50 ms fixed one -- on numbers the matrix supports.
        """
        import numpy as np

        sp = self._make_processor()
        mock_config = MagicMock()
        mock_config.min_cache_savings_pct = 0.20
        mock_config.min_cache_fixed_budget_seconds = 0.05
        # Disable the min-execution-time floor so the object-size policy is tested.
        mock_config.min_execution_time_to_cache_seconds = 0.0
        sp.cash_instance.config = mock_config

        large_var = np.zeros(50_000_000, dtype=np.float64)  # 400 MB
        result, reason, _ = sp._store.should_skip_large_object_caching(
            {"big_var": large_var},
            execution_time=0.1,  # ratio budget = 80 ms; restore predicted ~265 ms
        )
        assert result is True
        assert reason is not None
        assert "@cash:persist" in reason

    def test_should_not_skip_small_object(self):
        """Small objects fall under the 50 ms fixed budget and should be
        cached regardless of how fast the computation was."""
        sp = self._make_processor()
        mock_config = MagicMock()
        mock_config.min_cache_savings_pct = 0.20
        mock_config.min_cache_fixed_budget_seconds = 0.05
        # Disable the min-execution-time floor so the object-size policy is tested.
        mock_config.min_execution_time_to_cache_seconds = 0.0
        sp.cash_instance.config = mock_config

        result, reason, _ = sp._store.should_skip_large_object_caching(
            {"small_var": 42},
            execution_time=0.05,  # above the 10 ms floor; only size policy matters here
        )
        assert result is False
        assert reason is None

    def test_should_not_skip_when_compute_expensive(self):
        """When the computation itself is very expensive, even a slow
        restore is worth it. Restoring a 100 MB ndarray takes ~165 ms; the
        ratio budget at execution_time=100s is 80s, so the policy should
        cache."""
        import numpy as np

        sp = self._make_processor()
        mock_config = MagicMock()
        mock_config.min_cache_savings_pct = 0.20
        mock_config.min_cache_fixed_budget_seconds = 0.05
        sp.cash_instance.config = mock_config

        large_var = np.zeros(12_500_000, dtype=np.float64)  # 100 MB
        result, reason, _ = sp._store.should_skip_large_object_caching(
            {"big_var": large_var},
            execution_time=100.0,
        )
        assert result is False
        assert reason is None

    def test_should_not_skip_when_force_persist(self):
        """force_persist overrides the cost-model decision."""
        import numpy as np

        sp = self._make_processor()
        mock_config = MagicMock()
        mock_config.min_cache_savings_pct = 0.20
        mock_config.min_cache_fixed_budget_seconds = 0.05
        sp.cash_instance.config = mock_config

        large_var = np.zeros(12_500_000, dtype=np.float64)  # 100 MB
        result, reason, _ = sp._store.should_skip_large_object_caching(
            {"big_var": large_var},
            execution_time=0.1,
            force_persist=True,
        )
        assert result is False
        assert reason is None

    def test_no_config_uses_defaults(self):
        """If no config, defaults (20% ratio, 50 ms fixed budget) apply.
        A tiny object under the fixed budget should not be skipped."""
        sp = self._make_processor()
        sp.cash_instance.config = None

        result, reason, _ = sp._store.should_skip_large_object_caching(
            {"x": 42},
            execution_time=0.01,
        )
        assert result is False
        assert reason is None

    @staticmethod
    def _make_processor():
        """Create a minimal StatementProcessor for testing."""
        from cash.notebook.statement import StatementProcessor

        mock_shell = MagicMock()
        mock_shell.user_ns = {}
        mock_cash = MagicMock()
        mock_cash.config = None
        return StatementProcessor(mock_shell, mock_cash)


class TestSizeAwareConfig:
    """Tests for size-aware caching configuration."""

    def test_default_config_has_size_settings(self):
        """Default config includes size-aware caching settings."""
        from cash.config import CashConfig

        config = CashConfig()
        assert config.min_cache_savings_pct == 0.20

    def test_config_to_dict_has_size_settings(self):
        """to_dict() includes size-aware settings."""
        from cash.config import CashConfig

        config = CashConfig()
        d = config.to_dict()
        assert "min_cache_savings_pct" in d
