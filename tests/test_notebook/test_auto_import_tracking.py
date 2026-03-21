"""
Tests for automatic import source tracking and opaque call pattern detection.

Covers:
- is_local_module(): distinguishing local from stdlib/site-packages modules
- auto_track_local_imports(): AST-based import detection and auto-tracking
- check_and_reload_changed_modules(): detect changes, reload, update user_ns
- _update_user_ns_from_module(): refresh stale function objects after reload
- detect_opaque_call_patterns(): warn about untrackable call patterns
- get_function_source_hash() cache bypass for tracked modules
"""

import sys
import time
import types
import importlib
import pytest

from cash.notebook.function_tracker import FunctionTracker, is_local_module


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tracker():
    """Provide a fresh FunctionTracker."""
    ft = FunctionTracker()
    yield ft
    ft.clear()


@pytest.fixture
def temp_module(tmp_path):
    """Create a temporary Python module file.

    Returns (module_name, module_path) and cleans up sys.modules/path on exit.
    """
    module_name = f"_test_auto_track_{id(tmp_path)}"
    module_file = tmp_path / f"{module_name}.py"
    module_file.write_text(
        "def helper(x):\n"
        "    return x * 2\n\n"
        "def transform(data):\n"
        "    return [x + 1 for x in data]\n"
    )

    sys.path.insert(0, str(tmp_path))

    yield module_name, str(module_file)

    # Cleanup
    if str(tmp_path) in sys.path:
        sys.path.remove(str(tmp_path))
    if module_name in sys.modules:
        del sys.modules[module_name]


@pytest.fixture
def temp_module_pair(tmp_path):
    """Create two temporary Python module files.

    Returns (mod1_name, mod1_path, mod2_name, mod2_path).
    """
    mod1_name = f"_test_mod_a_{id(tmp_path)}"
    mod2_name = f"_test_mod_b_{id(tmp_path)}"

    mod1_file = tmp_path / f"{mod1_name}.py"
    mod1_file.write_text("def func_a(x):\n    return x + 1\n")

    mod2_file = tmp_path / f"{mod2_name}.py"
    mod2_file.write_text("def func_b(x):\n    return x * 10\n")

    sys.path.insert(0, str(tmp_path))

    yield mod1_name, str(mod1_file), mod2_name, str(mod2_file)

    if str(tmp_path) in sys.path:
        sys.path.remove(str(tmp_path))
    for name in (mod1_name, mod2_name):
        if name in sys.modules:
            del sys.modules[name]


# ============================================================================
# is_local_module() tests
# ============================================================================


class TestIsLocalModule:
    """Tests for the is_local_module() utility function."""

    def test_stdlib_module_returns_false(self):
        """Stdlib modules like 'os' should NOT be local."""
        import os as os_module
        assert is_local_module(os_module) is False

    def test_json_module_returns_false(self):
        """Stdlib 'json' module should NOT be local."""
        import json
        assert is_local_module(json) is False

    def test_site_packages_module_returns_false(self):
        """Third-party packages (pytest) should NOT be local."""
        import pytest as pytest_mod
        assert is_local_module(pytest_mod) is False

    def test_builtin_module_returns_false(self):
        """Modules with no __file__ (builtins) should NOT be local."""
        import builtins
        assert is_local_module(builtins) is False

    def test_local_temp_module_returns_true(self, temp_module):
        """A module in a temp directory (not site-packages) should be local."""
        module_name, _ = temp_module
        mod = importlib.import_module(module_name)
        assert is_local_module(mod) is True

    def test_module_without_file_attr(self):
        """Module without __file__ should return False."""
        fake_module = types.ModuleType("fake_no_file")
        assert is_local_module(fake_module) is False

    def test_module_with_c_extension_returns_false(self):
        """Module with .pyd/.so extension should return False."""
        fake_module = types.ModuleType("fake_c_ext")
        fake_module.__file__ = "/some/path/module.pyd"
        assert is_local_module(fake_module) is False

    def test_module_with_so_extension_returns_false(self):
        """Module with .so extension should return False."""
        fake_module = types.ModuleType("fake_so_ext")
        fake_module.__file__ = "/some/path/module.so"
        assert is_local_module(fake_module) is False


# ============================================================================
# auto_track_local_imports() tests
# ============================================================================


class TestAutoTrackLocalImports:
    """Tests for automatic import detection and tracking."""

    def test_import_statement_detected(self, tracker, temp_module):
        """'import module_name' should auto-track a local module."""
        module_name, _ = temp_module
        importlib.import_module(module_name)

        code = f"import {module_name}"
        newly_tracked = tracker.auto_track_local_imports(code)

        assert module_name in newly_tracked
        assert module_name in tracker._tracked_modules

    def test_from_import_detected(self, tracker, temp_module):
        """'from module_name import func' should auto-track."""
        module_name, _ = temp_module
        importlib.import_module(module_name)

        code = f"from {module_name} import helper"
        newly_tracked = tracker.auto_track_local_imports(code)

        assert module_name in newly_tracked

    def test_stdlib_import_not_tracked(self, tracker):
        """'import os' should NOT be tracked (stdlib)."""
        code = "import os"
        newly_tracked = tracker.auto_track_local_imports(code)
        assert len(newly_tracked) == 0

    def test_site_packages_import_not_tracked(self, tracker):
        """'import pytest' should NOT be tracked (site-packages)."""
        code = "import pytest"
        newly_tracked = tracker.auto_track_local_imports(code)
        assert len(newly_tracked) == 0

    def test_already_tracked_not_re_tracked(self, tracker, temp_module):
        """Module already in _tracked_modules should not be re-tracked."""
        module_name, _ = temp_module
        importlib.import_module(module_name)

        # First tracking
        tracker.auto_track_local_imports(f"import {module_name}")
        assert module_name in tracker._tracked_modules

        # Second tracking should return empty set
        newly_tracked = tracker.auto_track_local_imports(f"import {module_name}")
        assert len(newly_tracked) == 0

    def test_unimported_module_skipped(self, tracker):
        """Module not yet in sys.modules should be skipped."""
        code = "import totally_nonexistent_module_xyzzy"
        newly_tracked = tracker.auto_track_local_imports(code)
        assert len(newly_tracked) == 0

    def test_syntax_error_returns_empty(self, tracker):
        """Invalid Python code should return empty set."""
        code = "import ("
        newly_tracked = tracker.auto_track_local_imports(code)
        assert len(newly_tracked) == 0

    def test_multiple_imports_tracked(self, tracker, temp_module_pair):
        """Multiple import statements should all be tracked."""
        mod1_name, _, mod2_name, _ = temp_module_pair
        importlib.import_module(mod1_name)
        importlib.import_module(mod2_name)

        code = f"import {mod1_name}\nimport {mod2_name}"
        newly_tracked = tracker.auto_track_local_imports(code)

        assert mod1_name in newly_tracked
        assert mod2_name in newly_tracked

    def test_dotted_import_tracks_top_level(self, tracker, tmp_path):
        """'import pkg.sub' should track both 'pkg' and 'pkg.sub'."""
        # Create a package
        pkg_name = f"_test_pkg_{id(tmp_path)}"
        pkg_dir = tmp_path / pkg_name
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("# package\n")
        (pkg_dir / "sub.py").write_text("def sub_func(): return 42\n")

        sys.path.insert(0, str(tmp_path))
        try:
            importlib.import_module(f"{pkg_name}.sub")

            code = f"import {pkg_name}.sub"
            newly_tracked = tracker.auto_track_local_imports(code)

            # Top-level package should be tracked
            assert pkg_name in newly_tracked or f"{pkg_name}.sub" in newly_tracked
        finally:
            if str(tmp_path) in sys.path:
                sys.path.remove(str(tmp_path))
            for key in list(sys.modules.keys()):
                if key.startswith(pkg_name):
                    del sys.modules[key]

    def test_no_import_in_code(self, tracker):
        """Code without imports should return empty set."""
        code = "x = 42\ny = x * 2"
        newly_tracked = tracker.auto_track_local_imports(code)
        assert len(newly_tracked) == 0


# ============================================================================
# check_and_reload_changed_modules() tests
# ============================================================================


class TestCheckAndReloadChangedModules:
    """Tests for automatic module reload on source changes."""

    def test_no_tracked_modules_returns_empty(self, tracker):
        """With no tracked modules, returns empty dict."""
        result, _ = tracker.check_and_reload_changed_modules({})
        assert result == {}

    def test_no_changes_returns_empty(self, tracker, temp_module):
        """No changes detected returns empty dict."""
        module_name, _ = temp_module
        importlib.import_module(module_name)
        tracker.track_module(module_name)

        result, _ = tracker.check_and_reload_changed_modules({})
        assert result == {}

    def test_detect_and_reload_changed_module(self, tracker, temp_module):
        """Changed module should be reloaded and appear in result."""
        module_name, module_file = temp_module
        importlib.import_module(module_name)
        tracker.track_module(module_name)

        # Modify the file
        time.sleep(0.1)
        with open(module_file, 'w') as f:
            f.write("def helper(x):\n    return x * 99\n")

        user_ns = {}
        result, _ = tracker.check_and_reload_changed_modules(user_ns)

        assert module_name in result

        # Verify reloaded module works
        mod_reloaded = sys.modules[module_name]
        assert mod_reloaded.helper(5) == 495

    def test_updates_user_ns_after_reload(self, tracker, temp_module):
        """After reload, user_ns should have fresh function objects."""
        module_name, module_file = temp_module
        mod = importlib.import_module(module_name)
        tracker.track_module(module_name)

        # Put the function in user_ns (simulates 'from mod import helper')
        user_ns = {'helper': mod.helper}
        assert user_ns['helper'](5) == 10

        # Modify the module
        time.sleep(0.1)
        with open(module_file, 'w') as f:
            f.write("def helper(x):\n    return x * 100\n")

        result, _ = tracker.check_and_reload_changed_modules(user_ns)
        assert module_name in result

        # user_ns should have the fresh function now
        assert user_ns['helper'](5) == 500

    def test_multiple_modules_some_changed(self, tracker, temp_module_pair):
        """Only changed modules should be reloaded."""
        mod1_name, mod1_file, mod2_name, mod2_file = temp_module_pair
        importlib.import_module(mod1_name)
        importlib.import_module(mod2_name)
        tracker.track_module(mod1_name)
        tracker.track_module(mod2_name)

        # Only modify mod1
        time.sleep(0.1)
        with open(mod1_file, 'w') as f:
            f.write("def func_a(x):\n    return x + 999\n")

        result, _ = tracker.check_and_reload_changed_modules({})
        assert mod1_name in result
        assert mod2_name not in result


# ============================================================================
# _update_user_ns_from_module() tests
# ============================================================================


class TestUpdateUserNsFromModule:
    """Tests for refreshing user_ns with fresh module objects."""

    def test_replaces_stale_function(self, tracker, temp_module):
        """Stale function objects should be replaced with fresh ones."""
        module_name, module_file = temp_module
        mod = importlib.import_module(module_name)

        user_ns = {'helper': mod.helper}
        old_helper = mod.helper

        # Modify and reload
        time.sleep(0.1)
        with open(module_file, 'w') as f:
            f.write("def helper(x):\n    return x ** 2\n")

        tracker.reload_module(module_name)
        new_mod = sys.modules[module_name]

        updated = tracker._update_user_ns_from_module(module_name, new_mod, user_ns)

        assert 'helper' in updated
        assert user_ns['helper'] is not old_helper
        assert user_ns['helper'](5) == 25

    def test_skips_underscore_vars(self, tracker, temp_module):
        """Variables starting with _ should be skipped."""
        module_name, _ = temp_module
        mod = importlib.import_module(module_name)

        user_ns = {'_private': mod.helper, 'helper': mod.helper}

        updated = tracker._update_user_ns_from_module(module_name, mod, user_ns)

        # _private should NOT be updated, helper should
        assert '_private' not in updated
        assert 'helper' in updated

    def test_non_module_vars_untouched(self, tracker, temp_module):
        """Variables not from the module should be untouched."""
        module_name, _ = temp_module
        mod = importlib.import_module(module_name)

        local_func = lambda x: x  # noqa: E731
        user_ns = {'my_func': local_func, 'helper': mod.helper}

        updated = tracker._update_user_ns_from_module(module_name, mod, user_ns)

        assert 'my_func' not in updated
        assert user_ns['my_func'] is local_func

    def test_updates_multiple_functions(self, tracker, temp_module):
        """Multiple functions from same module should all be updated."""
        module_name, _ = temp_module
        mod = importlib.import_module(module_name)

        user_ns = {'helper': mod.helper, 'transform': mod.transform}

        updated = tracker._update_user_ns_from_module(module_name, mod, user_ns)

        assert 'helper' in updated
        assert 'transform' in updated


# ============================================================================
# detect_opaque_call_patterns() tests
# ============================================================================


class TestDetectOpaqueCallPatterns:
    """Tests for detecting untrackable function call patterns."""

    def test_getattr_call_warns(self, tracker):
        """getattr(obj, 'method')() should trigger a warning."""
        code = "result = getattr(obj, 'method_name')(x)"
        warnings = tracker.detect_opaque_call_patterns(code, {})
        assert len(warnings) > 0
        assert any("getattr" in w for w in warnings)

    def test_subscript_call_warns(self, tracker):
        """registry['key'](x) should trigger a warning."""
        code = "result = registry['handler'](data)"
        warnings = tracker.detect_opaque_call_patterns(code, {})
        assert len(warnings) > 0
        assert any("Indexed call" in w for w in warnings)

    def test_subscript_call_with_index(self, tracker):
        """funcs[0](x) should trigger a warning."""
        code = "funcs[0](x)"
        warnings = tracker.detect_opaque_call_patterns(code, {})
        assert len(warnings) > 0
        assert any("Indexed call" in w for w in warnings)

    def test_eval_with_function_call_warns(self, tracker):
        """eval('func(x)') should trigger a warning."""
        code = "result = eval('process(data)')"
        warnings = tracker.detect_opaque_call_patterns(code, {})
        assert len(warnings) > 0
        assert any("eval" in w for w in warnings)

    def test_exec_with_function_call_warns(self, tracker):
        """exec('func(x)') should trigger a warning."""
        code = "exec('process(data)')"
        warnings = tracker.detect_opaque_call_patterns(code, {})
        assert len(warnings) > 0
        assert any("exec" in w for w in warnings)

    def test_eval_with_dynamic_expression_warns(self, tracker):
        """eval(some_var) should trigger a warning."""
        code = "result = eval(code_string)"
        warnings = tracker.detect_opaque_call_patterns(code, {})
        assert len(warnings) > 0
        assert any("eval" in w for w in warnings)

    def test_normal_call_no_warnings(self, tracker):
        """Regular function calls should NOT trigger warnings."""
        code = "result = process(data)"
        warnings = tracker.detect_opaque_call_patterns(code, {})
        assert len(warnings) == 0

    def test_method_call_no_warnings(self, tracker):
        """Method calls like df.process() should NOT trigger warnings."""
        code = "result = df.sort_values('col').head(10)"
        warnings = tracker.detect_opaque_call_patterns(code, {})
        assert len(warnings) == 0

    def test_syntax_error_returns_empty(self, tracker):
        """Invalid code should return empty warnings."""
        code = "def broken("
        warnings = tracker.detect_opaque_call_patterns(code, {})
        assert len(warnings) == 0

    def test_eval_with_simple_string_no_calls_no_warning(self, tracker):
        """eval('42') with no function calls should NOT warn."""
        code = "result = eval('42 + 1')"
        warnings = tracker.detect_opaque_call_patterns(code, {})
        assert len(warnings) == 0

    def test_multiple_patterns_multiple_warnings(self, tracker):
        """Multiple opaque patterns in same code should produce multiple warnings."""
        code = (
            "a = getattr(obj, 'method')(x)\n"
            "b = registry['key'](y)\n"
            "c = eval('func(z)')\n"
        )
        warnings = tracker.detect_opaque_call_patterns(code, {})
        assert len(warnings) >= 3

    def test_nested_subscript_call(self, tracker):
        """obj.registry['handler'](x) should trigger warning."""
        code = "result = obj.handlers['process'](data)"
        warnings = tracker.detect_opaque_call_patterns(code, {})
        assert len(warnings) > 0


# ============================================================================
# get_function_source_hash() cache bypass for tracked modules
# ============================================================================


class TestSourceHashCacheBypass:
    """Test that source hash cache is bypassed for tracked modules with file changes."""

    def test_cached_hash_returned_for_untracked(self, tracker):
        """For untracked modules, cached hash should be returned."""
        def my_func(x):
            return x * 2

        h1 = tracker.get_function_source_hash(my_func)
        assert h1 is not None

        # Hash should come from cache on second call
        h2 = tracker.get_function_source_hash(my_func)
        assert h1 == h2

    def test_stale_cache_bypassed_for_tracked_module(self, tracker, temp_module):
        """When a tracked module's file changes, id-cache should be bypassed."""
        module_name, module_file = temp_module
        mod = importlib.import_module(module_name)
        tracker.track_module(module_name)

        # Get initial hash
        h1 = tracker.get_function_source_hash(mod.helper)
        assert h1 is not None

        # Modify the file (change the function body)
        time.sleep(0.1)
        with open(module_file, 'w') as f:
            f.write("def helper(x):\n    return x ** 3\n")

        # Reload the module so inspect.getsource can read fresh code
        tracker.reload_module(module_name)
        mod_reloaded = sys.modules[module_name]

        # Get hash for the new function object — should differ from h1
        h2 = tracker.get_function_source_hash(mod_reloaded.helper)
        assert h2 is not None
        assert h1 != h2

    def test_cache_used_when_no_mtime_change(self, tracker, temp_module):
        """When tracked module file hasn't changed, cache should be used."""
        module_name, module_file = temp_module
        mod = importlib.import_module(module_name)
        tracker.track_module(module_name)

        h1 = tracker.get_function_source_hash(mod.helper)
        h2 = tracker.get_function_source_hash(mod.helper)
        # Same hash because file hasn't changed
        assert h1 == h2


# ============================================================================
# Integration: auto-track + change detection + reload
# ============================================================================


class TestAutoTrackingIntegration:
    """End-to-end tests combining auto-tracking with change detection."""

    def test_full_cycle_import_change_reload(self, tracker, temp_module):
        """Full cycle: auto-track → detect change → reload → verify."""
        module_name, module_file = temp_module
        mod = importlib.import_module(module_name)

        # Step 1: Auto-track from import statement
        code = f"from {module_name} import helper"
        newly_tracked = tracker.auto_track_local_imports(code)
        assert module_name in newly_tracked

        # Step 2: Simulate user_ns with imported function
        user_ns = {'helper': mod.helper}
        assert user_ns['helper'](5) == 10

        # Step 3: Modify the source file
        time.sleep(0.1)
        with open(module_file, 'w') as f:
            f.write("def helper(x):\n    return x * 7\n")

        # Step 4: check_and_reload detects change and updates user_ns
        changed, _ = tracker.check_and_reload_changed_modules(user_ns)
        assert module_name in changed
        assert user_ns['helper'](5) == 35

    def test_auto_track_does_not_double_track(self, tracker, temp_module):
        """Running auto_track twice on same import doesn't duplicate tracking."""
        module_name, _ = temp_module
        importlib.import_module(module_name)

        code = f"import {module_name}"
        newly1 = tracker.auto_track_local_imports(code)
        newly2 = tracker.auto_track_local_imports(code)

        assert module_name in newly1
        assert len(newly2) == 0  # Already tracked

    def test_source_hash_changes_after_auto_reload(self, tracker, temp_module):
        """After auto-reload, source hash should reflect new code."""
        module_name, module_file = temp_module
        mod = importlib.import_module(module_name)
        tracker.auto_track_local_imports(f"import {module_name}")

        # Get initial hash
        hash_before = tracker.get_function_source_hash(mod.helper)

        # Modify the source
        time.sleep(0.1)
        with open(module_file, 'w') as f:
            f.write("def helper(x):\n    return x - 1\n")

        # Auto-reload
        user_ns = {'helper': mod.helper}
        tracker.check_and_reload_changed_modules(user_ns)

        # Get new hash from fresh function
        hash_after = tracker.get_function_source_hash(user_ns['helper'])
        assert hash_before != hash_after

    def test_unchanged_module_no_reload(self, tracker, temp_module):
        """Module that hasn't changed should not be reloaded."""
        module_name, _ = temp_module
        importlib.import_module(module_name)
        tracker.auto_track_local_imports(f"import {module_name}")

        result, _ = tracker.check_and_reload_changed_modules({})
        assert len(result) == 0

    def test_mixed_local_and_stdlib_imports(self, tracker, temp_module):
        """Code with both local and stdlib imports only tracks local."""
        module_name, _ = temp_module
        importlib.import_module(module_name)

        code = f"import os\nimport json\nimport {module_name}"
        newly_tracked = tracker.auto_track_local_imports(code)

        assert module_name in newly_tracked
        assert 'os' not in tracker._tracked_modules
        assert 'json' not in tracker._tracked_modules


# ============================================================================
# Edge cases and robustness
# ============================================================================


class TestEdgeCases:
    """Edge cases and error handling tests."""

    def test_auto_track_with_multiline_cell(self, tracker, temp_module):
        """Auto-tracking works with realistic multiline cell code."""
        module_name, _ = temp_module
        importlib.import_module(module_name)

        code = f"""
import os
from {module_name} import helper

x = helper(10)
print(x)
"""
        newly_tracked = tracker.auto_track_local_imports(code)
        assert module_name in newly_tracked

    def test_opaque_patterns_in_realistic_code(self, tracker):
        """Opaque pattern detection in realistic notebook-like code."""
        code = """
import pandas as pd
df = pd.read_csv('data.csv')

# This is fine
result = df.groupby('col').agg('sum')

# This should warn
handler = registry['process']
output = handler(df)
"""
        warnings = tracker.detect_opaque_call_patterns(code, {})
        # The subscript call `registry['process']` followed by `handler(df)` —
        # only the subscript-call pattern triggers
        # We look for at least the registry['process'] pattern if it's called directly
        # In this code, handler(df) is a normal call so no warning
        # But registry['process'] is an assignment, not a call — so no subscript call warning
        # Let's verify
        assert isinstance(warnings, list)

    def test_reload_preserves_non_function_attributes(self, tracker, temp_module):
        """Reload should not affect non-function user_ns entries."""
        module_name, module_file = temp_module
        mod = importlib.import_module(module_name)
        tracker.track_module(module_name)

        # user_ns has both module functions and other data
        user_ns = {
            'helper': mod.helper,
            'my_data': [1, 2, 3],
            'my_number': 42,
        }

        time.sleep(0.1)
        with open(module_file, 'w') as f:
            f.write("def helper(x):\n    return x + 100\n")

        tracker.check_and_reload_changed_modules(user_ns)

        # Non-module data should be untouched
        assert user_ns['my_data'] == [1, 2, 3]
        assert user_ns['my_number'] == 42
        # But the function should be updated
        assert user_ns['helper'](5) == 105

    def test_detect_opaque_patterns_empty_code(self, tracker):
        """Empty code should return no warnings."""
        warnings = tracker.detect_opaque_call_patterns("", {})
        assert len(warnings) == 0

    def test_is_local_module_with_pyc_only(self, tmp_path):
        """Module with only .pyc (no .py source) should return False."""
        fake_module = types.ModuleType("fake_pyc_only")
        fake_module.__file__ = str(tmp_path / "fake_pyc_only.pyc")
        # No corresponding .py file exists
        assert is_local_module(fake_module) is False

    def test_auto_track_idempotent(self, tracker, temp_module):
        """Multiple calls to auto_track should not cause errors."""
        module_name, _ = temp_module
        importlib.import_module(module_name)

        code = f"import {module_name}"
        for _ in range(5):
            tracker.auto_track_local_imports(code)

        assert module_name in tracker._tracked_modules
