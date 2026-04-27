"""
Tests for new features:
1. Size-aware caching threshold (P0)
2. Narrow file dependency propagation (P1)
3. Side-effect detection
4. Automatic purity analysis for user-defined functions
"""

import os
from unittest.mock import MagicMock

from cash.notebook.side_effects import SideEffectDetector
from cash.notebook.purity import (
    analyze_function_purity,
    clear_purity_cache,
    pure,
    stateful,
)


# ===========================================================================
# Side-Effect Detection Tests
# ===========================================================================

class TestSideEffectDetector:
    """Tests for SideEffectDetector that identifies I/O operations."""

    def test_no_side_effects_simple(self):
        """Simple computation has no side effects."""
        code = "x = 1 + 2"
        assert SideEffectDetector.has_side_effects(code) is False

    def test_no_side_effects_function_call(self):
        """Regular function calls are not side effects."""
        code = "result = sorted([3, 1, 2])"
        assert SideEffectDetector.has_side_effects(code) is False

    def test_no_side_effects_assignment(self):
        """Variable assignment is not a side effect."""
        code = "df = df.sort_values('col')"
        assert SideEffectDetector.has_side_effects(code) is False

    def test_open_write_mode(self):
        """open() with write mode is a side effect."""
        code = "f = open('output.txt', 'w')"
        effects = SideEffectDetector.detect_side_effects(code)
        assert len(effects) == 1
        assert effects[0].kind == 'file_write'

    def test_open_read_mode_no_side_effect(self):
        """open() with read mode is NOT a side effect."""
        code = "f = open('data.txt', 'r')"
        assert SideEffectDetector.has_side_effects(code) is False

    def test_open_default_mode_no_side_effect(self):
        """open() without mode argument defaults to read (not a side effect)."""
        code = "f = open('data.txt')"
        assert SideEffectDetector.has_side_effects(code) is False

    def test_open_append_mode(self):
        """open() with append mode is a side effect."""
        code = "f = open('log.txt', 'a')"
        effects = SideEffectDetector.detect_side_effects(code)
        assert len(effects) == 1
        assert effects[0].kind == 'file_write'

    def test_open_keyword_mode(self):
        """open() with mode= keyword argument."""
        code = "f = open('out.txt', mode='w')"
        effects = SideEffectDetector.detect_side_effects(code)
        assert len(effects) == 1
        assert effects[0].kind == 'file_write'

    def test_os_system(self):
        """os.system() is a system side effect."""
        code = "os.system('ls')"
        effects = SideEffectDetector.detect_side_effects(code)
        assert len(effects) == 1
        assert effects[0].kind == 'system'

    def test_subprocess_run(self):
        """subprocess.run() is a system side effect."""
        code = "subprocess.run(['echo', 'hello'])"
        effects = SideEffectDetector.detect_side_effects(code)
        assert len(effects) == 1
        assert effects[0].kind == 'system'

    def test_to_csv(self):
        """DataFrame.to_csv() is a file write side effect."""
        code = "df.to_csv('output.csv')"
        effects = SideEffectDetector.detect_side_effects(code)
        assert len(effects) == 1
        assert effects[0].kind == 'file_write'

    def test_to_parquet(self):
        """DataFrame.to_parquet() is a file write side effect."""
        code = "df.to_parquet('data.parquet')"
        effects = SideEffectDetector.detect_side_effects(code)
        assert len(effects) == 1
        assert effects[0].kind == 'file_write'

    def test_savefig(self):
        """matplotlib savefig() is a file write side effect."""
        code = "plt.savefig('chart.png')"
        effects = SideEffectDetector.detect_side_effects(code)
        assert len(effects) == 1
        assert effects[0].kind == 'file_write'

    def test_json_dump(self):
        """json.dump() is a file write side effect."""
        code = "json.dump(data, f)"
        effects = SideEffectDetector.detect_side_effects(code)
        assert len(effects) == 1
        assert effects[0].kind == 'file_write'

    def test_os_makedirs(self):
        """os.makedirs() is a file write side effect."""
        code = "os.makedirs('output_dir', exist_ok=True)"
        effects = SideEffectDetector.detect_side_effects(code)
        assert len(effects) == 1
        assert effects[0].kind == 'file_write'

    def test_requests_post(self):
        """requests.post() is a network side effect."""
        code = "requests.post('https://api.example.com', data=payload)"
        effects = SideEffectDetector.detect_side_effects(code)
        assert len(effects) == 1
        assert effects[0].kind == 'network'

    def test_multiple_side_effects(self):
        """Multiple side effects in one code block."""
        code = """df.to_csv('output.csv')
os.makedirs('backups', exist_ok=True)"""
        effects = SideEffectDetector.detect_side_effects(code)
        assert len(effects) == 2

    def test_syntax_error_returns_empty(self):
        """Syntax errors return empty list."""
        code = "def invalid syntax"
        effects = SideEffectDetector.detect_side_effects(code)
        assert effects == []

    def test_get_side_effect_reasons(self):
        """get_side_effect_reasons returns human-readable reasons."""
        code = "df.to_csv('out.csv')"
        reasons = SideEffectDetector.get_side_effect_reasons(code)
        assert len(reasons) == 1
        assert 'to_csv' in reasons[0]
        assert 'file_write' in reasons[0]

    def test_write_method(self):
        """file.write() is detected as a side effect."""
        code = "f.write('hello')"
        effects = SideEffectDetector.detect_side_effects(code)
        assert len(effects) == 1
        assert effects[0].kind == 'file_write'

    def test_numpy_save(self):
        """np.save() / arr.save() is a side effect."""
        code = "np.save('data.npy', arr)"
        effects = SideEffectDetector.detect_side_effects(code)
        assert len(effects) >= 1

    def test_shutil_rmtree(self):
        """shutil.rmtree() is a file write (destructive) side effect."""
        code = "shutil.rmtree('old_dir')"
        effects = SideEffectDetector.detect_side_effects(code)
        assert len(effects) == 1
        assert effects[0].kind == 'file_write'

    def test_read_csv_not_side_effect(self):
        """pd.read_csv() is NOT a side effect (read-only)."""
        code = "df = pd.read_csv('data.csv')"
        assert SideEffectDetector.has_side_effects(code) is False

    def test_display_not_side_effect(self):
        """display() is NOT treated as side effect (output is captured)."""
        code = "display(df)"
        assert SideEffectDetector.has_side_effects(code) is False


# ===========================================================================
# Automatic Purity Analysis Tests
# ===========================================================================

# ===========================================================================
# Helper functions for purity analysis tests
# Defined at module level so inspect.getsource() works under pytest
# (pytest's assertion rewriting breaks getsource for nested functions)
# ===========================================================================

def _pure_compute(x, y):
    return x + y

def _pure_transform(data):
    result = data * 2
    adjusted = result + 1
    return adjusted

def _pure_classify(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

def _impure_noisy(x):
    print(f"computing {x}")
    return x * 2

def _impure_load(path):
    f = open(path)  # noqa: SIM115
    return f.read()

def _impure_mutate_global():
    global counter
    counter += 1
    return counter

def _impure_outer():
    x = 0
    def inner():
        nonlocal x
        x += 1
        return x
    return inner

def _impure_modify(obj):
    obj.value = 42
    return obj

def _impure_update_dict(d, key, val):
    d[key] = val
    return d

def _impure_run_command(cmd):
    return os.system(cmd)

def _impure_add_item(lst, item):
    lst.append(item)
    return lst

def _impure_gen(n):
    for i in range(n):
        yield i

@pure
def _pure_side_effecty():
    print("hello")
    return 42

@stateful
def _stateful_pure_looking():
    return 42

def _pure_simple(x):
    return x + 1

def _impure_logged(x):
    import logging
    logging.info(f"processing {x}")
    return x

def _pure_sum_range(n):
    total = 0
    for i in range(n):
        total += i
    return total

def _pure_squares(data):
    return [x ** 2 for x in data]

def _impure_save_data(df):
    df.to_csv('output.csv')


class TestAutoPurityAnalysis:
    """Tests for automatic purity detection of user-defined functions."""

    def setup_method(self):
        """Clear purity cache before each test."""
        clear_purity_cache()

    def test_pure_math_function(self):
        """Simple math function should be detected as pure."""
        assert analyze_function_purity(_pure_compute) is True

    def test_pure_with_locals(self):
        """Function with local variables should be pure."""
        assert analyze_function_purity(_pure_transform) is True

    def test_pure_with_conditionals(self):
        """Function with if/else should be pure."""
        assert analyze_function_purity(_pure_classify) is True

    def test_impure_print(self):
        """Function with print() is impure."""
        assert analyze_function_purity(_impure_noisy) is False

    def test_impure_open(self):
        """Function with open() is impure."""
        assert analyze_function_purity(_impure_load) is False

    def test_impure_global(self):
        """Function with global statement is impure."""
        assert analyze_function_purity(_impure_mutate_global) is False

    def test_impure_nonlocal(self):
        """Function with nonlocal statement is impure."""
        assert analyze_function_purity(_impure_outer) is False

    def test_impure_attribute_assign(self):
        """Function that modifies object attributes is impure."""
        assert analyze_function_purity(_impure_modify) is False

    def test_impure_subscript_assign(self):
        """Function that modifies dict/list items is impure."""
        assert analyze_function_purity(_impure_update_dict) is False

    def test_impure_os_system(self):
        """Function calling os.system is impure."""
        assert analyze_function_purity(_impure_run_command) is False

    def test_impure_list_append(self):
        """Function calling list.append is impure (mutating method)."""
        assert analyze_function_purity(_impure_add_item) is False

    def test_impure_generator(self):
        """Generator function is not pure (in caching sense)."""
        assert analyze_function_purity(_impure_gen) is False

    def test_explicit_pure_decorator(self):
        """@pure decorated function is pure regardless of analysis."""
        assert analyze_function_purity(_pure_side_effecty) is True

    def test_explicit_stateful_decorator(self):
        """@stateful decorated function is impure."""
        assert analyze_function_purity(_stateful_pure_looking) is False

    def test_builtin_not_analyzable(self):
        """Built-in functions can't be analyzed (no source), return False."""
        assert analyze_function_purity(len) is False

    def test_lambda_not_analyzable(self):
        """Lambdas defined in test can't always get clean source, but if they can, analyze."""
        f = lambda x: x * 2  # noqa: E731
        result = analyze_function_purity(f)
        assert isinstance(result, bool)

    def test_caching_works(self):
        """Repeated calls should use cache."""
        result1 = analyze_function_purity(_pure_simple)
        result2 = analyze_function_purity(_pure_simple)
        assert result1 == result2

    def test_impure_logging(self):
        """Function calling logging methods is impure."""
        assert analyze_function_purity(_impure_logged) is False

    def test_pure_with_loop(self):
        """Function with loop but no side effects is pure."""
        assert analyze_function_purity(_pure_sum_range) is True

    def test_pure_with_list_comprehension(self):
        """Function with list comprehension is pure."""
        assert analyze_function_purity(_pure_squares) is True

    def test_impure_file_write(self):
        """Function that writes to file via to_csv is impure."""
        assert analyze_function_purity(_impure_save_data) is False
        assert analyze_function_purity(_impure_save_data) is False


# ===========================================================================
# Size-Aware Caching Tests
# ===========================================================================

class TestSizeAwareCaching:
    """Tests for size-aware caching threshold in StatementProcessor."""

    def test_estimate_object_size_int(self):
        """Integer size estimation."""
        sp = self._make_processor()
        assert sp._estimate_object_size(42) > 0

    def test_estimate_object_size_str(self):
        """String size estimation."""
        sp = self._make_processor()
        size = sp._estimate_object_size("hello" * 1000)
        assert size > 5000  # At least 5KB for a 5000-char string

    def test_estimate_object_size_list(self):
        """List size estimation."""
        sp = self._make_processor()
        size = sp._estimate_object_size(list(range(10000)))
        assert size > 0

    def test_should_skip_large_object_basic(self):
        """Large objects with fast computation should be skipped."""
        sp = self._make_processor()
        # Use a config with very slow serialization speed so restore is slow
        mock_config = MagicMock()
        mock_config.min_cache_savings_pct = 0.20
        mock_config.estimated_serialization_speed = 10  # 10 bytes/s → very slow
        sp.cash_instance.config = mock_config

        large_var = "x" * 200  # ~249 bytes, est_restore ≈ 24.9s >> 0.8*0.1 = 0.08
        result, reason = sp._should_skip_large_object_caching(
            {'big_var': large_var},
            execution_time=0.1,  # Fast computation
        )
        assert result is True
        assert reason is not None
        assert "@cash:persist" in reason

    def test_should_not_skip_small_object(self):
        """Small objects should never be skipped."""
        sp = self._make_processor()
        mock_config = MagicMock()
        mock_config.min_cache_savings_pct = 0.20
        mock_config.estimated_serialization_speed = 200 * 1024 * 1024  # 200 MB/s
        sp.cash_instance.config = mock_config

        result, reason = sp._should_skip_large_object_caching(
            {'small_var': 42},
            execution_time=0.01,
        )
        assert result is False
        assert reason is None

    def test_should_not_skip_when_compute_expensive(self):
        """Large objects with expensive computation should be cached."""
        sp = self._make_processor()
        mock_config = MagicMock()
        mock_config.min_cache_savings_pct = 0.20
        mock_config.estimated_serialization_speed = 10  # 10 bytes/s
        sp.cash_instance.config = mock_config

        large_var = "x" * 200
        # ~249 bytes, est_restore ≈ 24.9s.  exec_time=100 → threshold = 0.8*100=80
        # 24.9 < 80 → savings > 20% → worth caching
        result, reason = sp._should_skip_large_object_caching(
            {'big_var': large_var},
            execution_time=100.0,  # Very expensive → restore overhead is negligible
        )
        assert result is False
        assert reason is None

    def test_should_not_skip_when_force_persist(self):
        """force_persist overrides size-aware check."""
        sp = self._make_processor()
        mock_config = MagicMock()
        mock_config.min_cache_savings_pct = 0.20
        mock_config.estimated_serialization_speed = 10  # Would trigger skip without force_persist
        sp.cash_instance.config = mock_config

        large_var = "x" * 200
        result, reason = sp._should_skip_large_object_caching(
            {'big_var': large_var},
            execution_time=0.1,
            force_persist=True,
        )
        assert result is False
        assert reason is None

    def test_no_config_uses_defaults(self):
        """If no config, uses defaults (20% savings threshold, 200 MB/s disk speed)."""
        sp = self._make_processor()
        sp.cash_instance.config = None  # No config

        # Small object → should not skip
        result, reason = sp._should_skip_large_object_caching(
            {'x': 42},
            execution_time=0.01,
        )
        assert result is False
        assert reason is None

    @staticmethod
    def _make_processor():
        """Create a minimal StatementProcessor for testing."""
        from cash.notebook.statement_processor import StatementProcessor
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
        csv_path_str = str(csv_path).replace('\\', '/')
        pd.DataFrame({'a': [1, 2, 3]}).to_csv(csv_path, index=False)

        # Cell 1: Load CSV
        magics.cash("", f"import pandas as pd; df = pd.read_csv('{csv_path_str}')")
        # Manually set file deps as if FileAccessTracker tracked it
        sp.executed_file_deps['df'] = {csv_path_str}
        sp.executed_file_mtimes['df'] = {csv_path_str: os.path.getmtime(csv_path)}

        # Cell 2: Compute scalar from DataFrame
        magics.cash("", "n_rows = len(df)")

        # n_rows (int) should NOT have file deps
        assert 'n_rows' not in sp.executed_file_deps or len(sp.executed_file_deps.get('n_rows', set())) == 0

    def test_dataframe_does_inherit_file_deps(self, cash_magics, mock_shell, tmp_path):
        """DataFrame output should inherit file deps from DataFrame input."""
        magics = cash_magics
        sp = magics._statement_processor

        import pandas as pd
        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace('\\', '/')
        pd.DataFrame({'a': [1, 2, 3]}).to_csv(csv_path, index=False)

        magics.cash("", f"import pandas as pd; df = pd.read_csv('{csv_path_str}')")
        sp.executed_file_deps['df'] = {csv_path_str}
        sp.executed_file_mtimes['df'] = {csv_path_str: os.path.getmtime(csv_path)}

        # Cell 2: Transform DataFrame
        magics.cash("", "df2 = df[df['a'] > 1]")

        # df2 (DataFrame) SHOULD have file deps
        assert 'df2' in sp.executed_file_deps
        assert csv_path_str in sp.executed_file_deps['df2']

    def test_list_does_inherit_file_deps(self, cash_magics, mock_shell, tmp_path):
        """List output should inherit file deps (it could hold data)."""
        magics = cash_magics
        sp = magics._statement_processor

        import pandas as pd
        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace('\\', '/')
        pd.DataFrame({'a': [1, 2, 3]}).to_csv(csv_path, index=False)

        magics.cash("", f"import pandas as pd; df = pd.read_csv('{csv_path_str}')")
        sp.executed_file_deps['df'] = {csv_path_str}

        # Cell 2: Convert to list (data-bearing type)
        magics.cash("", "values = df['a'].tolist()")

        # values (list) SHOULD have file deps
        assert 'values' in sp.executed_file_deps
        assert csv_path_str in sp.executed_file_deps['values']

    def test_float_does_not_inherit_file_deps(self, cash_magics, mock_shell, tmp_path):
        """Float output should NOT inherit file deps."""
        magics = cash_magics
        sp = magics._statement_processor

        import pandas as pd
        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace('\\', '/')
        pd.DataFrame({'a': [1.0, 2.0, 3.0]}).to_csv(csv_path, index=False)

        magics.cash("", f"import pandas as pd; df = pd.read_csv('{csv_path_str}')")
        sp.executed_file_deps['df'] = {csv_path_str}

        # Cell 2: Compute float from DataFrame
        magics.cash("", "mean_val = df['a'].mean()")

        # mean_val (float) should NOT have file deps
        assert 'mean_val' not in sp.executed_file_deps or len(sp.executed_file_deps.get('mean_val', set())) == 0

    def test_bool_does_not_inherit_file_deps(self, cash_magics, mock_shell, tmp_path):
        """Bool output should NOT inherit file deps."""
        magics = cash_magics
        sp = magics._statement_processor

        import pandas as pd
        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace('\\', '/')
        pd.DataFrame({'a': [1, 2, 3]}).to_csv(csv_path, index=False)

        magics.cash("", f"import pandas as pd; df = pd.read_csv('{csv_path_str}')")
        sp.executed_file_deps['df'] = {csv_path_str}

        # Cell 2: Compute bool from DataFrame
        magics.cash("", "has_data = len(df) > 0")

        # has_data (bool) should NOT have file deps
        assert 'has_data' not in sp.executed_file_deps or len(sp.executed_file_deps.get('has_data', set())) == 0


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
        assert config.estimated_serialization_speed == 200 * 1024 * 1024

    def test_config_to_dict_has_size_settings(self):
        """to_dict() includes size-aware settings."""
        from cash.config import CashConfig
        config = CashConfig()
        d = config.to_dict()
        assert 'min_cache_savings_pct' in d
        assert 'estimated_serialization_speed' in d
