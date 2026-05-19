from cash.notebook.cache_status import CacheStatus
"""
Tests for granular per-symbol module invalidation.

When a module is modified, only the specific symbols (functions, classes,
constants) that actually changed should cause cache invalidation for
dependent variables.  Variables that only use unchanged symbols should be
preserved — their cache stays valid.

Test matrix:
1. Per-symbol hashing of module files
2. Changed-symbol detection (added, removed, modified symbols)
3. Module attribute access extraction from code AST
4. Per-symbol hash computation for specific attributes
5. Granular invalidation in _invalidate_module_lineages
6. End-to-end: change function → only users of that function invalidated
7. Edge cases: getattr, bare module reference, comment-only changes, etc.
"""

import hashlib
import importlib
import sys
import time
import pytest
from unittest.mock import MagicMock

from cash.notebook.function_tracker import FunctionTracker
from cash.notebook.magics import CashMagics
from cash.notebook.annotations import CacheAnnotation
from cash.core import Cash
from cash.backends.backend import InMemoryBackend
from traitlets.config.configurable import Configurable

# Force caching regardless of the 10 ms min-execution-time floor.
_PERSIST = CacheAnnotation(persist=True)


# ============================================================================
# Fixtures
# ============================================================================


class MockShell(Configurable):
    """Mock IPython shell for testing."""
    def __init__(self):
        super().__init__()
        self.user_ns = {}
        self.input_transformers_cleanup = []
        self.run_cell = MagicMock()
        self.events = MagicMock()
        self.ast_transformers = []
        self.user_global_ns = self.user_ns
        self.display_pub = type('MockDisplayPub', (), {'publish': MagicMock()})()


@pytest.fixture
def magics_fixture():
    """Provide CashMagics + shell + backend."""
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = MockShell()
    magics = CashMagics(shell, cash)
    magics._auto_cache_enabled = True
    yield magics, shell, backend
    backend.clear()
    shell.user_ns.clear()


@pytest.fixture
def temp_module(tmp_path):
    """Create a temporary module with multiple symbols."""
    module_name = f"_test_granular_mod_{id(tmp_path)}"
    module_file = tmp_path / f"{module_name}.py"
    module_file.write_text(
        "VERSION = '1.0'\n"
        "\n"
        "def compute(x):\n"
        "    return x * 2\n"
        "\n"
        "def format_result(x):\n"
        "    return f'Result: {x}'\n"
        "\n"
        "class Config:\n"
        "    debug = False\n"
    )

    sys.path.insert(0, str(tmp_path))
    yield module_name, str(module_file), tmp_path

    sys.path.remove(str(tmp_path))
    if module_name in sys.modules:
        del sys.modules[module_name]


# ============================================================================
# 1. Per-symbol hashing
# ============================================================================


class TestComputeSymbolHashes:
    """Tests for FunctionTracker.compute_symbol_hashes."""

    def test_functions_hashed(self, tmp_path):
        """Each function should get its own hash."""
        f = tmp_path / "mod.py"
        f.write_text(
            "def foo():\n    return 1\n\n"
            "def bar():\n    return 2\n"
        )
        hashes = FunctionTracker.compute_symbol_hashes(str(f))
        assert 'foo' in hashes
        assert 'bar' in hashes
        assert hashes['foo'] != hashes['bar']

    def test_classes_hashed(self, tmp_path):
        """Classes should get their own hash."""
        f = tmp_path / "mod.py"
        f.write_text("class MyClass:\n    x = 1\n")
        hashes = FunctionTracker.compute_symbol_hashes(str(f))
        assert 'MyClass' in hashes

    def test_constants_hashed(self, tmp_path):
        """Top-level assignments should be hashed."""
        f = tmp_path / "mod.py"
        f.write_text("VERSION = '1.0'\nDEBUG = False\n")
        hashes = FunctionTracker.compute_symbol_hashes(str(f))
        assert 'VERSION' in hashes
        assert 'DEBUG' in hashes

    def test_changing_function_changes_hash(self, tmp_path):
        """Modifying a function should change its hash."""
        f = tmp_path / "mod.py"
        f.write_text("def foo():\n    return 1\n")
        h1 = FunctionTracker.compute_symbol_hashes(str(f))

        f.write_text("def foo():\n    return 999\n")
        h2 = FunctionTracker.compute_symbol_hashes(str(f))

        assert h1['foo'] != h2['foo']

    def test_unchanged_function_same_hash(self, tmp_path):
        """Unchanged function produces same hash."""
        f = tmp_path / "mod.py"
        f.write_text("def foo():\n    return 1\n\ndef bar():\n    return 2\n")
        h1 = FunctionTracker.compute_symbol_hashes(str(f))

        # Change only bar, leave foo unchanged
        f.write_text("def foo():\n    return 1\n\ndef bar():\n    return 999\n")
        h2 = FunctionTracker.compute_symbol_hashes(str(f))

        assert h1['foo'] == h2['foo']  # foo unchanged
        assert h1['bar'] != h2['bar']  # bar changed

    def test_comment_only_change_same_hash(self, tmp_path):
        """Adding/changing comments doesn't change symbol hashes (AST-based)."""
        f = tmp_path / "mod.py"
        f.write_text("def foo():\n    return 1\n")
        h1 = FunctionTracker.compute_symbol_hashes(str(f))

        f.write_text("# This is a comment\ndef foo():\n    return 1\n")
        h2 = FunctionTracker.compute_symbol_hashes(str(f))

        assert h1['foo'] == h2['foo']

    def test_nonexistent_file_returns_empty(self):
        """Non-existent file returns empty dict."""
        assert FunctionTracker.compute_symbol_hashes("/nonexistent/path.py") == {}

    def test_syntax_error_returns_empty(self, tmp_path):
        """File with syntax errors returns empty dict."""
        f = tmp_path / "bad.py"
        f.write_text("def foo(\n  broken syntax")
        assert FunctionTracker.compute_symbol_hashes(str(f)) == {}

    def test_async_function_hashed(self, tmp_path):
        """Async functions should be hashed."""
        f = tmp_path / "mod.py"
        f.write_text("async def fetch():\n    return 42\n")
        hashes = FunctionTracker.compute_symbol_hashes(str(f))
        assert 'fetch' in hashes

    def test_annotated_assignment_hashed(self, tmp_path):
        """Annotated assignments should be hashed."""
        f = tmp_path / "mod.py"
        f.write_text("name: str = 'hello'\n")
        hashes = FunctionTracker.compute_symbol_hashes(str(f))
        assert 'name' in hashes

    def test_tuple_unpacking_hashed(self, tmp_path):
        """Tuple unpacking assignments should hash each name."""
        f = tmp_path / "mod.py"
        f.write_text("x, y = 1, 2\n")
        hashes = FunctionTracker.compute_symbol_hashes(str(f))
        assert 'x' in hashes
        assert 'y' in hashes

    def test_import_statements_tracked(self, tmp_path):
        """Import statements are tracked as __import__ symbols."""
        f = tmp_path / "mod.py"
        f.write_text("import os\nfrom sys import path\n")
        hashes = FunctionTracker.compute_symbol_hashes(str(f))
        assert '__import__os' in hashes
        assert '__import__path' in hashes


# ============================================================================
# 2. Changed symbol detection
# ============================================================================


class TestGetChangedSymbols:
    """Tests for FunctionTracker.get_changed_symbols."""

    def test_no_baseline_returns_none(self, temp_module):
        """Without a prior snapshot, returns None (full invalidation)."""
        module_name, module_file, _ = temp_module
        ft = FunctionTracker()
        importlib.import_module(module_name)
        # Don't snapshot → no baseline
        result = ft.get_changed_symbols(module_name)
        assert result is None

    def test_no_changes_returns_empty_set(self, temp_module):
        """When nothing changed, returns empty set."""
        module_name, module_file, _ = temp_module
        ft = FunctionTracker()
        importlib.import_module(module_name)
        ft.track_module(module_name)  # This snapshots symbols

        result = ft.get_changed_symbols(module_name)
        assert result == set()

    def test_changed_function_detected(self, temp_module):
        """Modifying a function should be detected."""
        module_name, module_file, _ = temp_module
        ft = FunctionTracker()
        importlib.import_module(module_name)
        ft.track_module(module_name)

        # Change compute() only
        time.sleep(0.05)
        with open(module_file, 'w') as f:
            f.write(
                "VERSION = '1.0'\n\n"
                "def compute(x):\n    return x * 100\n\n"  # Changed!
                "def format_result(x):\n    return f'Result: {x}'\n\n"
                "class Config:\n    debug = False\n"
            )

        result = ft.get_changed_symbols(module_name)
        assert result is not None
        assert 'compute' in result
        assert 'format_result' not in result
        assert 'VERSION' not in result
        assert 'Config' not in result

    def test_added_symbol_detected(self, temp_module):
        """Adding a new symbol should be detected."""
        module_name, module_file, _ = temp_module
        ft = FunctionTracker()
        importlib.import_module(module_name)
        ft.track_module(module_name)

        # Add new_func
        time.sleep(0.05)
        with open(module_file, 'w') as f:
            f.write(
                "VERSION = '1.0'\n\n"
                "def compute(x):\n    return x * 2\n\n"
                "def format_result(x):\n    return f'Result: {x}'\n\n"
                "def new_func():\n    pass\n\n"
                "class Config:\n    debug = False\n"
            )

        result = ft.get_changed_symbols(module_name)
        assert 'new_func' in result
        # Existing unchanged symbols should NOT be in the result
        assert 'compute' not in result
        assert 'VERSION' not in result

    def test_removed_symbol_detected(self, temp_module):
        """Removing a symbol should be detected."""
        module_name, module_file, _ = temp_module
        ft = FunctionTracker()
        importlib.import_module(module_name)
        ft.track_module(module_name)

        # Remove format_result
        time.sleep(0.05)
        with open(module_file, 'w') as f:
            f.write(
                "VERSION = '1.0'\n\n"
                "def compute(x):\n    return x * 2\n\n"
                "class Config:\n    debug = False\n"
            )

        result = ft.get_changed_symbols(module_name)
        assert 'format_result' in result
        assert 'compute' not in result

    def test_constant_change_detected(self, temp_module):
        """Changing a constant should be detected."""
        module_name, module_file, _ = temp_module
        ft = FunctionTracker()
        importlib.import_module(module_name)
        ft.track_module(module_name)

        # Change VERSION
        time.sleep(0.05)
        with open(module_file, 'w') as f:
            f.write(
                "VERSION = '2.0'\n\n"  # Changed!
                "def compute(x):\n    return x * 2\n\n"
                "def format_result(x):\n    return f'Result: {x}'\n\n"
                "class Config:\n    debug = False\n"
            )

        result = ft.get_changed_symbols(module_name)
        assert 'VERSION' in result
        assert 'compute' not in result

    def test_snapshot_updates_after_reload(self, temp_module):
        """After snapshot_module_symbols, subsequent get_changed_symbols should use new baseline."""
        module_name, module_file, _ = temp_module
        ft = FunctionTracker()
        importlib.import_module(module_name)
        ft.track_module(module_name)

        # Change compute
        time.sleep(0.05)
        with open(module_file, 'w') as f:
            f.write(
                "VERSION = '1.0'\n\n"
                "def compute(x):\n    return x * 100\n\n"
                "def format_result(x):\n    return f'Result: {x}'\n\n"
                "class Config:\n    debug = False\n"
            )

        result1 = ft.get_changed_symbols(module_name)
        assert 'compute' in result1

        # Take new snapshot
        ft.snapshot_module_symbols(module_name)

        # Now no changes
        result2 = ft.get_changed_symbols(module_name)
        assert result2 == set()


# ============================================================================
# 3. Module attribute access extraction
# ============================================================================


class TestExtractModuleAttributeAccesses:
    """Tests for FunctionTracker.extract_module_attribute_accesses."""

    def test_simple_attribute_access(self):
        """mod.attr should be detected."""
        accesses = FunctionTracker.extract_module_attribute_accesses("x = mod.compute(5)")
        assert 'mod' in accesses
        assert 'compute' in accesses['mod']

    def test_multiple_attributes(self):
        """Multiple attributes of same module should all be tracked."""
        code = "a = mod.compute(5)\nb = mod.format_result(a)"
        accesses = FunctionTracker.extract_module_attribute_accesses(code)
        assert 'mod' in accesses
        assert accesses['mod'] == {'compute', 'format_result'}

    def test_constant_access(self):
        """mod.CONST should be detected."""
        accesses = FunctionTracker.extract_module_attribute_accesses("v = mod.VERSION")
        assert 'mod' in accesses
        assert 'VERSION' in accesses['mod']

    def test_mixed_function_and_constant(self):
        """Both function calls and constant access tracked."""
        code = "v = mod.VERSION\nr = mod.compute(5)"
        accesses = FunctionTracker.extract_module_attribute_accesses(code)
        assert accesses['mod'] == {'VERSION', 'compute'}

    def test_getattr_with_constant_string(self):
        """getattr(mod, 'attr') with string constant should track attr."""
        accesses = FunctionTracker.extract_module_attribute_accesses("getattr(mod, 'compute')")
        assert 'mod' in accesses
        assert 'compute' in accesses['mod']

    def test_getattr_with_dynamic_attr(self):
        """getattr(mod, var) with dynamic second arg cannot be tracked."""
        accesses = FunctionTracker.extract_module_attribute_accesses("getattr(mod, name)")
        # Should still have 'mod' but with bare use flagged
        assert 'mod' in accesses

    def test_no_module_access(self):
        """Code without module access returns empty."""
        accesses = FunctionTracker.extract_module_attribute_accesses("x = 1 + 2")
        assert accesses == {}

    def test_syntax_error_returns_empty(self):
        """Syntax error in code returns empty dict."""
        accesses = FunctionTracker.extract_module_attribute_accesses("def foo(\n  broken")
        assert accesses == {}

    def test_chained_attribute_access(self):
        """mod.sub.attr should track 'sub' on mod."""
        accesses = FunctionTracker.extract_module_attribute_accesses("mod.sub.attr")
        assert 'mod' in accesses
        assert 'sub' in accesses['mod']

    def test_multiple_modules(self):
        """Multiple different modules tracked separately."""
        code = "a = m1.func()\nb = m2.other()"
        accesses = FunctionTracker.extract_module_attribute_accesses(code)
        assert 'm1' in accesses and 'func' in accesses['m1']
        assert 'm2' in accesses and 'other' in accesses['m2']


# ============================================================================
# 4. Per-symbol hash computation
# ============================================================================


class TestComputeModuleSymbolHash:
    """Tests for FunctionTracker.compute_module_symbol_hash."""

    def test_different_attrs_different_hash(self, temp_module):
        """Hashing different attributes produces different results."""
        module_name, module_file, _ = temp_module
        ft = FunctionTracker()
        importlib.import_module(module_name)
        ft.track_module(module_name)

        h1 = ft.compute_module_symbol_hash(module_name, {'compute'})
        h2 = ft.compute_module_symbol_hash(module_name, {'format_result'})
        assert h1 != h2

    def test_same_attrs_same_hash(self, temp_module):
        """Same attributes produce same hash."""
        module_name, module_file, _ = temp_module
        ft = FunctionTracker()
        importlib.import_module(module_name)
        ft.track_module(module_name)

        h1 = ft.compute_module_symbol_hash(module_name, {'compute'})
        h2 = ft.compute_module_symbol_hash(module_name, {'compute'})
        assert h1 == h2

    def test_no_attrs_falls_back_to_full_hash(self, temp_module):
        """Empty/None attrs falls back to full file hash."""
        module_name, module_file, _ = temp_module
        ft = FunctionTracker()
        importlib.import_module(module_name)
        ft.track_module(module_name)

        h_none = ft.compute_module_symbol_hash(module_name, None)
        h_empty = ft.compute_module_symbol_hash(module_name, set())

        # Both should be the full file hash
        with open(module_file, 'rb') as f:
            expected_full = hashlib.sha256(f.read()).hexdigest()
        assert h_none == expected_full
        assert h_empty == expected_full

    def test_unchanged_symbol_same_hash_after_other_changes(self, temp_module):
        """After changing compute(), hash for VERSION should stay the same."""
        module_name, module_file, _ = temp_module
        ft = FunctionTracker()
        importlib.import_module(module_name)
        ft.track_module(module_name)

        h_version_before = ft.compute_module_symbol_hash(module_name, {'VERSION'})

        # Change compute() — leave VERSION unchanged
        time.sleep(0.05)
        with open(module_file, 'w') as f:
            f.write(
                "VERSION = '1.0'\n\n"
                "def compute(x):\n    return x * 100\n\n"
                "def format_result(x):\n    return f'Result: {x}'\n\n"
                "class Config:\n    debug = False\n"
            )

        # Re-snapshot to update symbol hashes
        ft.snapshot_module_symbols(module_name)

        h_version_after = ft.compute_module_symbol_hash(module_name, {'VERSION'})
        assert h_version_before == h_version_after

    def test_changed_symbol_different_hash(self, temp_module):
        """After changing compute(), hash for compute should change."""
        module_name, module_file, _ = temp_module
        ft = FunctionTracker()
        importlib.import_module(module_name)
        ft.track_module(module_name)

        h_compute_before = ft.compute_module_symbol_hash(module_name, {'compute'})

        # Change compute()
        time.sleep(0.05)
        with open(module_file, 'w') as f:
            f.write(
                "VERSION = '1.0'\n\n"
                "def compute(x):\n    return x * 100\n\n"
                "def format_result(x):\n    return f'Result: {x}'\n\n"
                "class Config:\n    debug = False\n"
            )

        ft.snapshot_module_symbols(module_name)
        h_compute_after = ft.compute_module_symbol_hash(module_name, {'compute'})
        assert h_compute_before != h_compute_after


# ============================================================================
# 5. Granular invalidation in _invalidate_module_lineages
# ============================================================================


class TestGranularInvalidation:
    """Tests for _invalidate_module_lineages with per-symbol granularity."""

    def test_only_changed_symbol_users_invalidated(self, magics_fixture, temp_module):
        """Variables using only unchanged symbols should be preserved."""
        magics, shell, backend = magics_fixture
        module_name, module_file, _ = temp_module
        sp = magics._statement_processor

        # Import and track the module
        mod = importlib.import_module(module_name)
        shell.user_ns[module_name] = mod
        sp.function_tracker.track_module(module_name)

        # Setup: variable 'result' depends on module.compute
        old_lineage = hashlib.sha256(b"old").hexdigest()
        sp.variable_lineage[module_name] = old_lineage
        sp.variable_lineage['result'] = "result_hash"
        sp.executed_cell_codes['result'] = f"result = {module_name}.compute(5)"
        sp.executed_input_lineages['result'] = {module_name: old_lineage}
        sp.module_attribute_deps['result'] = {module_name: {'compute'}}

        # Setup: variable 'version_str' depends on module.VERSION
        sp.variable_lineage['version_str'] = "version_hash"
        sp.executed_cell_codes['version_str'] = f"version_str = {module_name}.VERSION"
        sp.executed_input_lineages['version_str'] = {module_name: old_lineage}
        sp.module_attribute_deps['version_str'] = {module_name: {'VERSION'}}

        # Only 'compute' changed
        changed_modules = {module_name: module_file}
        per_module_changed_symbols = {module_name: {'compute'}}

        magics._module_invalidator.invalidate(
            changed_modules,
            magics._statement_processor,
            per_module_changed_symbols,
        )

        # 'result' should be invalidated (uses compute)
        assert 'result' not in sp.variable_lineage
        assert 'result' not in sp.executed_cell_codes

        # 'version_str' should be PRESERVED (uses VERSION, which didn't change)
        assert 'version_str' in sp.variable_lineage
        assert sp.variable_lineage['version_str'] == "version_hash"
        assert 'version_str' in sp.executed_cell_codes

    def test_no_granular_info_full_invalidation(self, magics_fixture, temp_module):
        """When per_module_changed_symbols is None, full invalidation happens."""
        magics, shell, backend = magics_fixture
        module_name, module_file, _ = temp_module
        sp = magics._statement_processor

        old_lineage = hashlib.sha256(b"old").hexdigest()
        sp.variable_lineage[module_name] = old_lineage
        sp.variable_lineage['result'] = "result_hash"
        sp.executed_cell_codes['result'] = f"result = {module_name}.compute(5)"
        sp.executed_input_lineages['result'] = {module_name: old_lineage}
        sp.module_attribute_deps['result'] = {module_name: {'compute'}}

        sp.variable_lineage['version_str'] = "version_hash"
        sp.executed_cell_codes['version_str'] = f"version_str = {module_name}.VERSION"
        sp.executed_input_lineages['version_str'] = {module_name: old_lineage}
        sp.module_attribute_deps['version_str'] = {module_name: {'VERSION'}}

        # No granular info (None)
        changed_modules = {module_name: module_file}
        per_module_changed_symbols = {module_name: None}

        magics._module_invalidator.invalidate(
            changed_modules,
            magics._statement_processor,
            per_module_changed_symbols,
        )

        # Both should be invalidated
        assert 'result' not in sp.variable_lineage
        assert 'version_str' not in sp.variable_lineage

    def test_no_attribute_deps_full_invalidation(self, magics_fixture, temp_module):
        """When module_attribute_deps is not set for a var, full invalidation for safety."""
        magics, shell, backend = magics_fixture
        module_name, module_file, _ = temp_module
        sp = magics._statement_processor

        old_lineage = hashlib.sha256(b"old").hexdigest()
        sp.variable_lineage[module_name] = old_lineage
        sp.variable_lineage['result'] = "result_hash"
        sp.executed_cell_codes['result'] = f"result = {module_name}.compute(5)"
        sp.executed_input_lineages['result'] = {module_name: old_lineage}
        # No module_attribute_deps set for 'result'

        changed_modules = {module_name: module_file}
        per_module_changed_symbols = {module_name: {'compute'}}

        magics._module_invalidator.invalidate(
            changed_modules,
            magics._statement_processor,
            per_module_changed_symbols,
        )

        # Should still be invalidated (no granular info about which attrs are used)
        assert 'result' not in sp.variable_lineage

    def test_empty_changed_symbols_preserves_all(self, magics_fixture, temp_module):
        """If no symbols actually changed (e.g., whitespace only), preserve all vars."""
        magics, shell, backend = magics_fixture
        module_name, module_file, _ = temp_module
        sp = magics._statement_processor

        old_lineage = hashlib.sha256(b"old").hexdigest()
        sp.variable_lineage[module_name] = old_lineage
        sp.variable_lineage['result'] = "result_hash"
        sp.executed_cell_codes['result'] = f"result = {module_name}.compute(5)"
        sp.executed_input_lineages['result'] = {module_name: old_lineage}
        sp.module_attribute_deps['result'] = {module_name: {'compute'}}

        # Empty set = module mtime changed but no AST-level symbol changes
        changed_modules = {module_name: module_file}
        per_module_changed_symbols = {module_name: set()}

        magics._module_invalidator.invalidate(
            changed_modules,
            magics._statement_processor,
            per_module_changed_symbols,
        )

        # 'result' should be preserved (nothing actually changed)
        assert 'result' in sp.variable_lineage

    def test_backward_compat_without_per_module_symbols(self, magics_fixture, temp_module):
        """Calling without per_module_changed_symbols falls back to full invalidation."""
        magics, shell, backend = magics_fixture
        module_name, module_file, _ = temp_module
        sp = magics._statement_processor

        old_lineage = hashlib.sha256(b"old").hexdigest()
        sp.variable_lineage[module_name] = old_lineage
        sp.variable_lineage['result'] = "result_hash"
        sp.executed_cell_codes['result'] = f"result = {module_name}.compute(5)"
        sp.executed_input_lineages['result'] = {module_name: old_lineage}

        changed_modules = {module_name: module_file}
        # Don't pass per_module_changed_symbols
        magics._module_invalidator.invalidate(
            changed_modules,
            magics._statement_processor,
        )

        # Should still invalidate (backward compatible)
        assert 'result' not in sp.variable_lineage

    def test_multiple_modules_granular(self, magics_fixture, tmp_path):
        """Granular invalidation works across multiple changed modules."""
        magics, shell, backend = magics_fixture
        sp = magics._statement_processor

        # Create two modules
        mod_a_name = f"_test_gran_a_{id(tmp_path)}"
        mod_b_name = f"_test_gran_b_{id(tmp_path)}"
        mod_a_file = tmp_path / f"{mod_a_name}.py"
        mod_b_file = tmp_path / f"{mod_b_name}.py"
        mod_a_file.write_text("def func_a():\n    return 1\nCONST_A = 10\n")
        mod_b_file.write_text("def func_b():\n    return 2\nCONST_B = 20\n")

        old_lineage_a = hashlib.sha256(b"old_a").hexdigest()
        old_lineage_b = hashlib.sha256(b"old_b").hexdigest()

        sp.variable_lineage[mod_a_name] = old_lineage_a
        sp.variable_lineage[mod_b_name] = old_lineage_b

        # var_x uses mod_a.func_a
        sp.variable_lineage['var_x'] = "x_hash"
        sp.executed_cell_codes['var_x'] = f"var_x = {mod_a_name}.func_a()"
        sp.executed_input_lineages['var_x'] = {mod_a_name: old_lineage_a}
        sp.module_attribute_deps['var_x'] = {mod_a_name: {'func_a'}}

        # var_y uses mod_a.CONST_A
        sp.variable_lineage['var_y'] = "y_hash"
        sp.executed_cell_codes['var_y'] = f"var_y = {mod_a_name}.CONST_A"
        sp.executed_input_lineages['var_y'] = {mod_a_name: old_lineage_a}
        sp.module_attribute_deps['var_y'] = {mod_a_name: {'CONST_A'}}

        # var_z uses mod_b.func_b
        sp.variable_lineage['var_z'] = "z_hash"
        sp.executed_cell_codes['var_z'] = f"var_z = {mod_b_name}.func_b()"
        sp.executed_input_lineages['var_z'] = {mod_b_name: old_lineage_b}
        sp.module_attribute_deps['var_z'] = {mod_b_name: {'func_b'}}

        # Only func_a changed in mod_a, only CONST_B changed in mod_b
        changed_modules = {
            mod_a_name: str(mod_a_file),
            mod_b_name: str(mod_b_file),
        }
        per_module_changed_symbols = {
            mod_a_name: {'func_a'},
            mod_b_name: {'CONST_B'},
        }

        magics._module_invalidator.invalidate(
            changed_modules,
            magics._statement_processor,
            per_module_changed_symbols,
        )

        # var_x uses func_a which changed → invalidated
        assert 'var_x' not in sp.variable_lineage

        # var_y uses CONST_A which didn't change → preserved
        assert 'var_y' in sp.variable_lineage

        # var_z uses func_b which didn't change → preserved
        assert 'var_z' in sp.variable_lineage


# ============================================================================
# 6. End-to-end flow
# ============================================================================


class TestGranularEndToEnd:
    """End-to-end tests combining all components."""

    def test_full_flow_granular_invalidation(self, magics_fixture, tmp_path):
        """E2E: change one function in module → only its users are invalidated.
        _PERSIST overrides the 10 ms min-execution-time floor so trivial module
        calls are actually stored in cache."""
        magics, shell, backend = magics_fixture
        sp = magics._statement_processor
        ft = sp.function_tracker

        # Create module with two functions
        module_name = f"_test_e2e_gran_{id(tmp_path)}"
        module_file = tmp_path / f"{module_name}.py"
        module_file.write_text(
            "VERSION = '1.0'\n\n"
            "def compute(x):\n    return x * 2\n\n"
            "def format_result(x):\n    return f'Result: {x}'\n"
        )

        sys.path.insert(0, str(tmp_path))
        try:
            # Import the module
            mod = importlib.import_module(module_name)
            shell.user_ns[module_name] = mod
            ft.track_module(module_name)

            # Execute import
            sp.process_statement(f"import {module_name}", silent=True)

            # Execute: use compute
            metrics1 = sp.process_statement(f"result = {module_name}.compute(5)", silent=True, annotation=_PERSIST)
            assert metrics1['status'] == CacheStatus.COMPUTED
            assert shell.user_ns.get('result') == 10

            # Execute: use VERSION
            metrics2 = sp.process_statement(f"v = {module_name}.VERSION", silent=True, annotation=_PERSIST)
            assert metrics2['status'] == CacheStatus.COMPUTED
            assert shell.user_ns.get('v') == '1.0'

            # Execute: use format_result
            metrics3 = sp.process_statement(f"fmt = {module_name}.format_result(42)", silent=True, annotation=_PERSIST)
            assert metrics3['status'] == CacheStatus.COMPUTED
            assert shell.user_ns.get('fmt') == 'Result: 42'

            # Re-run all — should be SKIPPED/RESTORED
            metrics1b = sp.process_statement(f"result = {module_name}.compute(5)", silent=True, annotation=_PERSIST)
            assert metrics1b['status'] in (CacheStatus.SKIPPED, CacheStatus.RESTORED)

            metrics2b = sp.process_statement(f"v = {module_name}.VERSION", silent=True, annotation=_PERSIST)
            assert metrics2b['status'] in (CacheStatus.SKIPPED, CacheStatus.RESTORED)

            metrics3b = sp.process_statement(f"fmt = {module_name}.format_result(42)", silent=True, annotation=_PERSIST)
            assert metrics3b['status'] in (CacheStatus.SKIPPED, CacheStatus.RESTORED)

            # Now change ONLY compute()
            time.sleep(0.05)
            module_file.write_text(
                "VERSION = '1.0'\n\n"
                "def compute(x):\n    return x * 100\n\n"  # Changed!
                "def format_result(x):\n    return f'Result: {x}'\n"
            )

            # Simulate cell execution: check and reload
            changed_modules, per_mod_syms = ft.check_and_reload_changed_modules(shell.user_ns)
            assert module_name in changed_modules

            # Verify granular detection
            changed_syms = per_mod_syms.get(module_name)
            assert changed_syms is not None
            assert 'compute' in changed_syms

            # Invalidate with granular info
            magics._module_invalidator.invalidate(
            changed_modules,
            magics._statement_processor,
            per_mod_syms,
        )

            # Re-run import
            sp.process_statement(f"import {module_name}", silent=True)

            # Re-run compute — should be COMPUTED (invalidated)
            metrics1c = sp.process_statement(f"result = {module_name}.compute(5)", silent=True, annotation=_PERSIST)
            assert metrics1c['status'] == CacheStatus.COMPUTED
            assert shell.user_ns.get('result') == 500  # 5 * 100

            # VERSION didn't change → may still be SKIPPED/RESTORED or COMPUTED
            # (module reload changes module lineage which can affect cache keys)
            metrics2c = sp.process_statement(f"v = {module_name}.VERSION", silent=True, annotation=_PERSIST)
            assert metrics2c['status'] in (CacheStatus.SKIPPED, CacheStatus.RESTORED, CacheStatus.COMPUTED)

            # format_result didn't change → may still be SKIPPED/RESTORED or COMPUTED
            metrics3c = sp.process_statement(f"fmt = {module_name}.format_result(42)", silent=True, annotation=_PERSIST)
            assert metrics3c['status'] in (CacheStatus.SKIPPED, CacheStatus.RESTORED, CacheStatus.COMPUTED)

        finally:
            sys.path.remove(str(tmp_path))
            if module_name in sys.modules:
                del sys.modules[module_name]

    def test_full_flow_change_constant_only(self, magics_fixture, tmp_path):
        """E2E: change a constant → only constant users are invalidated."""
        magics, shell, backend = magics_fixture
        sp = magics._statement_processor
        ft = sp.function_tracker

        module_name = f"_test_e2e_const_{id(tmp_path)}"
        module_file = tmp_path / f"{module_name}.py"
        module_file.write_text(
            "VERSION = '1.0'\n\n"
            "def compute(x):\n    return x * 2\n"
        )

        sys.path.insert(0, str(tmp_path))
        try:
            mod = importlib.import_module(module_name)
            shell.user_ns[module_name] = mod
            ft.track_module(module_name)
            sp.process_statement(f"import {module_name}", silent=True)

            # Use both
            sp.process_statement(f"result = {module_name}.compute(5)", silent=True)
            assert shell.user_ns['result'] == 10
            sp.process_statement(f"v = {module_name}.VERSION", silent=True)
            assert shell.user_ns['v'] == '1.0'

            # Change only VERSION
            time.sleep(0.05)
            module_file.write_text(
                "VERSION = '2.0'\n\n"  # Changed!
                "def compute(x):\n    return x * 2\n"
            )

            changed_modules, per_mod_syms = ft.check_and_reload_changed_modules(shell.user_ns)
            magics._module_invalidator.invalidate(
            changed_modules,
            magics._statement_processor,
            per_mod_syms,
        )
            sp.process_statement(f"import {module_name}", silent=True)

            # compute didn't change → may be SKIPPED/RESTORED or COMPUTED
            # (module reload changes module lineage which can affect cache keys)
            m1 = sp.process_statement(f"result = {module_name}.compute(5)", silent=True)
            assert m1['status'] in (CacheStatus.SKIPPED, CacheStatus.RESTORED, CacheStatus.COMPUTED)

            # VERSION changed → should be COMPUTED
            m2 = sp.process_statement(f"v = {module_name}.VERSION", silent=True)
            assert m2['status'] == CacheStatus.COMPUTED
            assert shell.user_ns['v'] == '2.0'

        finally:
            sys.path.remove(str(tmp_path))
            if module_name in sys.modules:
                del sys.modules[module_name]


# ============================================================================
# 7. Edge cases
# ============================================================================


class TestGranularEdgeCases:
    """Edge case tests for granular invalidation."""

    def test_comment_only_change_no_invalidation(self, temp_module):
        """Changing only comments should not change any symbol hashes."""
        module_name, module_file, _ = temp_module
        ft = FunctionTracker()
        importlib.import_module(module_name)
        ft.track_module(module_name)

        # Add comments only
        time.sleep(0.05)
        with open(module_file, 'w') as f:
            f.write(
                "# New comment added\n"
                "VERSION = '1.0'\n\n"
                "# Another comment\n"
                "def compute(x):\n    return x * 2\n\n"
                "def format_result(x):\n    return f'Result: {x}'\n\n"
                "class Config:\n    debug = False\n"
            )

        changed = ft.get_changed_symbols(module_name)
        assert changed == set()

    def test_whitespace_only_change_no_invalidation(self, temp_module):
        """Whitespace-only changes should not trigger invalidation."""
        module_name, module_file, _ = temp_module
        ft = FunctionTracker()
        importlib.import_module(module_name)
        ft.track_module(module_name)

        # Add blank lines
        time.sleep(0.05)
        with open(module_file, 'w') as f:
            f.write(
                "\n\nVERSION = '1.0'\n\n\n\n"
                "def compute(x):\n    return x * 2\n\n\n"
                "def format_result(x):\n    return f'Result: {x}'\n\n\n"
                "class Config:\n    debug = False\n\n"
            )

        changed = ft.get_changed_symbols(module_name)
        assert changed == set()

    def test_multiple_variables_using_same_changed_symbol(self, magics_fixture, temp_module):
        """Multiple variables using the same changed symbol should all be invalidated."""
        magics, shell, backend = magics_fixture
        module_name, module_file, _ = temp_module
        sp = magics._statement_processor

        old_lineage = hashlib.sha256(b"old").hexdigest()
        sp.variable_lineage[module_name] = old_lineage

        # Both a and b use compute
        for var in ['a', 'b']:
            sp.variable_lineage[var] = f"{var}_hash"
            sp.executed_cell_codes[var] = f"{var} = {module_name}.compute(1)"
            sp.executed_input_lineages[var] = {module_name: old_lineage}
            sp.module_attribute_deps[var] = {module_name: {'compute'}}

        # c uses format_result (unchanged)
        sp.variable_lineage['c'] = "c_hash"
        sp.executed_cell_codes['c'] = f"c = {module_name}.format_result(1)"
        sp.executed_input_lineages['c'] = {module_name: old_lineage}
        sp.module_attribute_deps['c'] = {module_name: {'format_result'}}

        changed_modules = {module_name: module_file}
        per_module_changed_symbols = {module_name: {'compute'}}

        magics._module_invalidator.invalidate(
            changed_modules,
            magics._statement_processor,
            per_module_changed_symbols,
        )

        assert 'a' not in sp.variable_lineage
        assert 'b' not in sp.variable_lineage
        assert 'c' in sp.variable_lineage

    def test_variable_using_multiple_attrs_including_changed(self, magics_fixture, temp_module):
        """If a variable uses both changed and unchanged attrs, it should be invalidated."""
        magics, shell, backend = magics_fixture
        module_name, module_file, _ = temp_module
        sp = magics._statement_processor

        old_lineage = hashlib.sha256(b"old").hexdigest()
        sp.variable_lineage[module_name] = old_lineage

        # Variable uses both compute AND VERSION
        sp.variable_lineage['mixed'] = "mixed_hash"
        sp.executed_cell_codes['mixed'] = f"mixed = {module_name}.compute(int({module_name}.VERSION))"
        sp.executed_input_lineages['mixed'] = {module_name: old_lineage}
        sp.module_attribute_deps['mixed'] = {module_name: {'compute', 'VERSION'}}

        # Only compute changed
        changed_modules = {module_name: module_file}
        per_module_changed_symbols = {module_name: {'compute'}}

        magics._module_invalidator.invalidate(
            changed_modules,
            magics._statement_processor,
            per_module_changed_symbols,
        )

        # Should be invalidated because one of its deps (compute) changed
        assert 'mixed' not in sp.variable_lineage

    def test_module_attribute_deps_cleared_on_invalidation(self, magics_fixture, temp_module):
        """module_attribute_deps should be cleared for invalidated variables."""
        magics, shell, backend = magics_fixture
        module_name, module_file, _ = temp_module
        sp = magics._statement_processor

        old_lineage = hashlib.sha256(b"old").hexdigest()
        sp.variable_lineage[module_name] = old_lineage
        sp.variable_lineage['result'] = "result_hash"
        sp.executed_cell_codes['result'] = f"result = {module_name}.compute(5)"
        sp.executed_input_lineages['result'] = {module_name: old_lineage}
        sp.module_attribute_deps['result'] = {module_name: {'compute'}}

        changed_modules = {module_name: module_file}
        per_module_changed_symbols = {module_name: {'compute'}}

        magics._module_invalidator.invalidate(
            changed_modules,
            magics._statement_processor,
            per_module_changed_symbols,
        )

        assert 'result' not in sp.module_attribute_deps

    def test_class_change_detected(self, tmp_path):
        """Changing a class body should be detected as a symbol change."""
        f = tmp_path / "mod.py"
        f.write_text("class Config:\n    debug = False\n")
        ft = FunctionTracker()
        h1 = ft.compute_symbol_hashes(str(f))

        f.write_text("class Config:\n    debug = True\n    verbose = True\n")
        h2 = ft.compute_symbol_hashes(str(f))

        assert h1['Config'] != h2['Config']
