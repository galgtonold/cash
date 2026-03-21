"""Tests for FunctionTracker - function source tracking for cache invalidation."""
from cash.notebook.function_tracker import FunctionTracker


class TestFunctionSourceHash:
    """Test function source hash computation."""

    def test_user_defined_function(self):
        """Source hash should be computed for user-defined functions."""
        tracker = FunctionTracker()
        
        def my_func(x):
            return x * 2
        
        result = tracker.get_function_source_hash(my_func)
        assert result is not None
        assert len(result) == 64  # SHA256 hex digest

    def test_builtin_function_returns_none(self):
        """Built-in functions should return None."""
        tracker = FunctionTracker()
        result = tracker.get_function_source_hash(len)
        assert result is None

    def test_lambda_returns_none(self):
        """Lambda functions should return None (unreliable source)."""
        tracker = FunctionTracker()
        f = lambda x: x * 2
        result = tracker.get_function_source_hash(f)
        assert result is None

    def test_non_callable_returns_none(self):
        """Non-callable objects should return None."""
        tracker = FunctionTracker()
        assert tracker.get_function_source_hash(42) is None
        assert tracker.get_function_source_hash("hello") is None
        assert tracker.get_function_source_hash([1, 2]) is None

    def test_same_function_same_hash(self):
        """Same function should produce same hash."""
        tracker = FunctionTracker()
        
        def f(x):
            return x + 1
        
        h1 = tracker.get_function_source_hash(f)
        h2 = tracker.get_function_source_hash(f)
        assert h1 == h2

    def test_different_functions_different_hash(self):
        """Different functions should produce different hashes."""
        tracker = FunctionTracker()
        
        def f1(x):
            return x * 2
        
        def f2(x):
            return x * 3
        
        h1 = tracker.get_function_source_hash(f1)
        h2 = tracker.get_function_source_hash(f2)
        assert h1 != h2

    def test_class_type_returns_none(self):
        """Class types (builtins) should return None."""
        tracker = FunctionTracker()
        assert tracker.get_function_source_hash(int) is None
        assert tracker.get_function_source_hash(str) is None

    def test_user_class_returns_hash(self):
        """User-defined classes should return a hash."""
        tracker = FunctionTracker()
        
        class MyClass:
            def method(self):
                return 42
        
        # Classes are callable, but isinstance(MyClass, type) is True for user classes too
        # Our implementation skips builtins module classes
        result = tracker.get_function_source_hash(MyClass)
        # User classes should get a hash since their source can change
        assert result is not None

    def test_caching_behavior(self):
        """Source hash should be cached for performance."""
        tracker = FunctionTracker()
        
        def f(x):
            return x
        
        h1 = tracker.get_function_source_hash(f)
        # Check that the cache is populated
        assert len(tracker._source_cache) == 1
        h2 = tracker.get_function_source_hash(f)
        assert h1 == h2

    def test_cache_eviction(self):
        """Cache should evict when it exceeds MAX_CACHE_SIZE."""
        tracker = FunctionTracker()
        tracker.MAX_CACHE_SIZE = 5  # Small for testing
        
        funcs = []
        for i in range(10):
            exec(f"def f_{i}(x): return x + {i}", globals())
            funcs.append(globals()[f"f_{i}"])
        
        for f in funcs:
            tracker.get_function_source_hash(f)
        
        # Cache should not exceed max size
        assert len(tracker._source_cache) <= 5

    def test_clear(self):
        """clear() should empty all caches."""
        tracker = FunctionTracker()
        
        def f(x):
            return x
        
        tracker.get_function_source_hash(f)
        tracker._function_hashes['f'] = 'hash'
        
        tracker.clear()
        assert len(tracker._source_cache) == 0
        assert len(tracker._function_hashes) == 0


class TestGetCallableSourceHashes:
    """Test getting source hashes for callable inputs."""

    def test_finds_callable_inputs(self):
        """Should return hashes for callable inputs in user_ns."""
        tracker = FunctionTracker()
        
        def process(x):
            return x * 2
        
        user_ns = {'process': process, 'data': [1, 2, 3]}
        result = tracker.get_callable_source_hashes({'process', 'data'}, user_ns)
        
        assert 'process' in result
        assert 'data' not in result

    def test_skips_missing_vars(self):
        """Should skip variables not in user_ns."""
        tracker = FunctionTracker()
        result = tracker.get_callable_source_hashes({'missing'}, {})
        assert result == {}

    def test_skips_builtins(self):
        """Should skip built-in functions."""
        tracker = FunctionTracker()
        user_ns = {'len': len, 'print': print}
        result = tracker.get_callable_source_hashes({'len', 'print'}, user_ns)
        assert result == {}


class TestDetectChangedFunctions:
    """Test function change detection."""

    def test_detect_changed_function(self):
        """Should detect when a function's source changes."""
        tracker = FunctionTracker()
        
        def process(x):
            return x * 2
        
        user_ns = {'process': process}
        
        # First call establishes baseline
        tracker.update_function_hash('process', process)
        
        # Create a new function with same name but different source
        def process_v2(x):
            return x * 3
        
        user_ns['process'] = process_v2
        
        changed = tracker.detect_changed_functions(user_ns, {'process'})
        assert 'process' in changed

    def test_no_change_detected_for_same_function(self):
        """Should not detect change when function hasn't changed."""
        tracker = FunctionTracker()
        
        def process(x):
            return x * 2
        
        user_ns = {'process': process}
        tracker.update_function_hash('process', process)
        
        changed = tracker.detect_changed_functions(user_ns, {'process'})
        assert len(changed) == 0

    def test_detect_deleted_function(self):
        """Should detect when a tracked function is deleted."""
        tracker = FunctionTracker()
        
        def process(x):
            return x * 2
        
        tracker.update_function_hash('process', process)
        
        # Function removed from namespace
        changed = tracker.detect_changed_functions({}, {'process'})
        assert 'process' in changed

    def test_detect_function_replaced_with_non_callable(self):
        """Should detect when a function is replaced with a non-callable."""
        tracker = FunctionTracker()
        
        def process(x):
            return x * 2
        
        tracker.update_function_hash('process', process)
        
        user_ns = {'process': 42}  # No longer callable
        changed = tracker.detect_changed_functions(user_ns, {'process'})
        assert 'process' in changed


class TestGetCalledFunctionNames:
    """Test extraction of called function names from code."""

    def test_simple_call(self):
        """Should extract function names from simple calls."""
        tracker = FunctionTracker()
        code = "result = process(10)"
        names = tracker.get_called_function_names(code)
        assert 'process' in names

    def test_multiple_calls(self):
        """Should extract all function names."""
        tracker = FunctionTracker()
        code = "a = f(x)\nb = g(y)\nc = h(a, b)"
        names = tracker.get_called_function_names(code)
        assert names == {'f', 'g', 'h'}

    def test_method_call_extracts_base(self):
        """Should extract base object for method calls."""
        tracker = FunctionTracker()
        code = "result = df.process(10)"
        names = tracker.get_called_function_names(code)
        assert 'df' in names

    def test_chained_method_calls(self):
        """Should extract base object for chained method calls."""
        tracker = FunctionTracker()
        code = "result = df.sort_values('col').head(10)"
        names = tracker.get_called_function_names(code)
        assert 'df' in names

    def test_no_calls(self):
        """Should return empty set when no function calls."""
        tracker = FunctionTracker()
        code = "x = 42\ny = x + 1"
        names = tracker.get_called_function_names(code)
        assert names == set()

    def test_syntax_error_returns_empty(self):
        """Should return empty set for invalid code."""
        tracker = FunctionTracker()
        code = "def incomplete("
        names = tracker.get_called_function_names(code)
        assert names == set()

    def test_nested_calls(self):
        """Should extract function names from nested calls."""
        tracker = FunctionTracker()
        code = "result = outer(inner(x))"
        names = tracker.get_called_function_names(code)
        assert 'outer' in names
        assert 'inner' in names


class TestUpdateFunctionHash:
    """Test updating stored function hashes."""

    def test_stores_hash(self):
        """update_function_hash should store the hash."""
        tracker = FunctionTracker()
        
        def f(x):
            return x
        
        result = tracker.update_function_hash('f', f)
        assert result is not None
        assert tracker._function_hashes['f'] == result

    def test_returns_none_for_builtin(self):
        """Should return None for built-in functions."""
        tracker = FunctionTracker()
        result = tracker.update_function_hash('len', len)
        assert result is None


class TestIntraModuleCallDeps:
    """Test get_intra_module_call_deps static method."""

    def test_simple_call_dependency(self, tmp_path):
        """Function calling another function should be detected."""
        mod_file = tmp_path / "mod.py"
        mod_file.write_text('''
def dep(a):
    return a + 1

def fun(a, b):
    return a + b + dep(a)
''')
        deps = FunctionTracker.get_intra_module_call_deps(str(mod_file))
        assert 'fun' in deps
        assert 'dep' in deps['fun']

    def test_no_self_dependency(self, tmp_path):
        """A function should not depend on itself."""
        mod_file = tmp_path / "mod.py"
        mod_file.write_text('''
def recursive(n):
    if n <= 0:
        return 0
    return recursive(n - 1) + 1
''')
        deps = FunctionTracker.get_intra_module_call_deps(str(mod_file))
        # recursive references itself, but should be excluded
        assert 'recursive' not in deps or 'recursive' not in deps.get('recursive', set())

    def test_transitive_chain(self, tmp_path):
        """A -> B -> C should be captured at each level."""
        mod_file = tmp_path / "mod.py"
        mod_file.write_text('''
def c():
    return 1

def b():
    return c() + 1

def a():
    return b() + 1
''')
        deps = FunctionTracker.get_intra_module_call_deps(str(mod_file))
        assert deps.get('b') == {'c'}
        assert deps.get('a') == {'b'}
        assert 'c' not in deps  # c doesn't call anything

    def test_class_method_deps(self, tmp_path):
        """Class referencing top-level functions should be detected."""
        mod_file = tmp_path / "mod.py"
        mod_file.write_text('''
def helper():
    return 42

class MyClass:
    def method(self):
        return helper()
''')
        deps = FunctionTracker.get_intra_module_call_deps(str(mod_file))
        assert 'MyClass' in deps
        assert 'helper' in deps['MyClass']

    def test_independent_functions(self, tmp_path):
        """Functions that don't reference each other should have no deps."""
        mod_file = tmp_path / "mod.py"
        mod_file.write_text('''
def foo(a):
    return a + 1

def bar(b):
    return b * 2
''')
        deps = FunctionTracker.get_intra_module_call_deps(str(mod_file))
        assert deps == {}

    def test_constant_reference(self, tmp_path):
        """Function referencing a module-level constant should be detected."""
        mod_file = tmp_path / "mod.py"
        mod_file.write_text('''
MULTIPLIER = 10

def scale(x):
    return x * MULTIPLIER
''')
        deps = FunctionTracker.get_intra_module_call_deps(str(mod_file))
        assert 'scale' in deps
        assert 'MULTIPLIER' in deps['scale']

    def test_nonexistent_file(self):
        """Should return empty dict for nonexistent file."""
        deps = FunctionTracker.get_intra_module_call_deps("/nonexistent/path.py")
        assert deps == {}

    def test_syntax_error_file(self, tmp_path):
        """Should return empty dict for file with syntax errors."""
        mod_file = tmp_path / "bad.py"
        mod_file.write_text("def broken(:\n    pass")
        deps = FunctionTracker.get_intra_module_call_deps(str(mod_file))
        assert deps == {}


class TestExpandChangedSymbolsTransitively:
    """Test expand_changed_symbols_transitively static method."""

    def test_direct_dependent(self):
        """Direct dependent of a changed symbol should be included."""
        call_deps = {'fun': {'dep'}}
        result = FunctionTracker.expand_changed_symbols_transitively({'dep'}, call_deps)
        assert result == {'dep', 'fun'}

    def test_transitive_chain(self):
        """A -> B -> C: changing C should expand to include B and A."""
        call_deps = {'a': {'b'}, 'b': {'c'}}
        result = FunctionTracker.expand_changed_symbols_transitively({'c'}, call_deps)
        assert result == {'a', 'b', 'c'}

    def test_no_dependents(self):
        """When nothing depends on the changed symbol, only it is returned."""
        call_deps = {'a': {'b'}}
        result = FunctionTracker.expand_changed_symbols_transitively({'c'}, call_deps)
        assert result == {'c'}

    def test_diamond_dependency(self):
        """Diamond: D depends on B and C, both depend on A."""
        call_deps = {'b': {'a'}, 'c': {'a'}, 'd': {'b', 'c'}}
        result = FunctionTracker.expand_changed_symbols_transitively({'a'}, call_deps)
        assert result == {'a', 'b', 'c', 'd'}

    def test_empty_changed_syms(self):
        """Empty changed set should return empty."""
        call_deps = {'a': {'b'}}
        result = FunctionTracker.expand_changed_symbols_transitively(set(), call_deps)
        assert result == set()

    def test_empty_call_deps(self):
        """No call deps means no expansion."""
        result = FunctionTracker.expand_changed_symbols_transitively({'x'}, {})
        assert result == {'x'}

    def test_multiple_changed(self):
        """Multiple initially changed symbols should all be expanded."""
        call_deps = {'c': {'a'}, 'd': {'b'}}
        result = FunctionTracker.expand_changed_symbols_transitively({'a', 'b'}, call_deps)
        assert result == {'a', 'b', 'c', 'd'}