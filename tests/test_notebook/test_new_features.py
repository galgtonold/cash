"""
Tests for new features:
1. Size-aware caching threshold (P0)
2. Narrow file dependency propagation (P1)

Side-effect detection tests live in test_cacheability.py.
"""

from unittest.mock import MagicMock

from tests._cell_driver import run_cash_cell

# ===========================================================================
# Size-Aware Caching Tests
# ===========================================================================


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


# ===========================================================================
# Narrow File Dependency Propagation Tests
# ===========================================================================


class TestNarrowFileDependencyPropagation:
    """Tests for narrow file dependency propagation (only to data-bearing types)."""

    def test_scalar_does_not_inherit_file_deps(self, cash_magics, mock_shell, tmp_path):
        """Scalar output (int) should NOT inherit file deps from DataFrame input."""
        magics = cash_magics
        sp = magics._statement_processor

        # Simulate file-loaded DataFrame
        import pandas as pd

        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace("\\", "/")
        pd.DataFrame({"a": [1, 2, 3]}).to_csv(csv_path, index=False)

        # Cell 1: Load CSV
        run_cash_cell(magics, f"import pandas as pd; df = pd.read_csv('{csv_path_str}')")
        # Manually set file deps as if FileAccessTracker tracked it
        sp.tracking_state.executed_file_deps["df"] = {csv_path_str}

        # Cell 2: Compute scalar from DataFrame
        run_cash_cell(magics, "n_rows = len(df)")

        # n_rows (int) should NOT have file deps
        assert (
            "n_rows" not in sp.tracking_state.executed_file_deps
            or len(sp.tracking_state.executed_file_deps.get("n_rows", set())) == 0
        )

    def test_dataframe_does_inherit_file_deps(self, cash_magics, mock_shell, tmp_path):
        """DataFrame output should inherit file deps from DataFrame input."""
        magics = cash_magics
        sp = magics._statement_processor

        import pandas as pd

        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace("\\", "/")
        pd.DataFrame({"a": [1, 2, 3]}).to_csv(csv_path, index=False)

        run_cash_cell(magics, f"import pandas as pd; df = pd.read_csv('{csv_path_str}')")
        sp.tracking_state.executed_file_deps["df"] = {csv_path_str}

        # Cell 2: Transform DataFrame
        run_cash_cell(magics, "df2 = df[df['a'] > 1]")

        # df2 (DataFrame) SHOULD have file deps
        assert "df2" in sp.tracking_state.executed_file_deps
        assert csv_path_str in sp.tracking_state.executed_file_deps["df2"]

    def test_list_does_inherit_file_deps(self, cash_magics, mock_shell, tmp_path):
        """List output should inherit file deps (it could hold data)."""
        magics = cash_magics
        sp = magics._statement_processor

        import pandas as pd

        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace("\\", "/")
        pd.DataFrame({"a": [1, 2, 3]}).to_csv(csv_path, index=False)

        run_cash_cell(magics, f"import pandas as pd; df = pd.read_csv('{csv_path_str}')")
        sp.tracking_state.executed_file_deps["df"] = {csv_path_str}

        # Cell 2: Convert to list (data-bearing type)
        run_cash_cell(magics, "values = df['a'].tolist()")

        # values (list) SHOULD have file deps
        assert "values" in sp.tracking_state.executed_file_deps
        assert csv_path_str in sp.tracking_state.executed_file_deps["values"]

    def test_float_does_not_inherit_file_deps(self, cash_magics, mock_shell, tmp_path):
        """Float output should NOT inherit file deps."""
        magics = cash_magics
        sp = magics._statement_processor

        import pandas as pd

        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace("\\", "/")
        pd.DataFrame({"a": [1.0, 2.0, 3.0]}).to_csv(csv_path, index=False)

        run_cash_cell(magics, f"import pandas as pd; df = pd.read_csv('{csv_path_str}')")
        sp.tracking_state.executed_file_deps["df"] = {csv_path_str}

        # Cell 2: Compute float from DataFrame
        run_cash_cell(magics, "mean_val = df['a'].mean()")

        # mean_val (float) should NOT have file deps
        assert (
            "mean_val" not in sp.tracking_state.executed_file_deps
            or len(sp.tracking_state.executed_file_deps.get("mean_val", set())) == 0
        )

    def test_bool_does_not_inherit_file_deps(self, cash_magics, mock_shell, tmp_path):
        """Bool output should NOT inherit file deps."""
        magics = cash_magics
        sp = magics._statement_processor

        import pandas as pd

        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace("\\", "/")
        pd.DataFrame({"a": [1, 2, 3]}).to_csv(csv_path, index=False)

        run_cash_cell(magics, f"import pandas as pd; df = pd.read_csv('{csv_path_str}')")
        sp.tracking_state.executed_file_deps["df"] = {csv_path_str}

        # Cell 2: Compute bool from DataFrame
        run_cash_cell(magics, "has_data = len(df) > 0")

        # has_data (bool) should NOT have file deps
        assert (
            "has_data" not in sp.tracking_state.executed_file_deps
            or len(sp.tracking_state.executed_file_deps.get("has_data", set())) == 0
        )


# ===========================================================================
# Config Tests
# ===========================================================================


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
