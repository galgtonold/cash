import pytest
from cash import Cash
from ..dummy_lib import lib_func
from cash.notebook.analysis import CodeAnalyzer

# Module-level app and test functions for dependency testing
app = Cash(register_magic=False)
dep_runs = 0
main_runs = 0

@app.cache
def dep2(x):
    global dep_runs
    dep_runs += 1
    return x * 2

@app.cache
def main_func2(x):
    global main_runs
    main_runs += 1
    return dep2(x) + 1

# Test imported function - decorate it to register with our app instance
cached_lib_func = app.cache(lib_func)

@app.cache
def calls_import(x):
    return cached_lib_func(x)


@pytest.fixture(autouse=True)
def reset_module_state():
    """Reset module-level state before each test."""
    app.backend.clear()
    global dep_runs, main_runs
    dep_runs = 0
    main_runs = 0
    
    # Ensure source hashes are correct (use module-qualified keys)
    app.source_hashes[Cash._get_func_key(dep2)] = CodeAnalyzer.get_source_hash(dep2)
    app.source_hashes[Cash._get_func_key(main_func2)] = CodeAnalyzer.get_source_hash(main_func2)
    app.source_hashes[Cash._get_func_key(lib_func)] = CodeAnalyzer.get_source_hash(lib_func)
    
    yield
    
    # Cleanup after test
    app.backend.clear()


def test_basic_caching():
    """Test that basic function caching works correctly."""
    local_app = Cash()
    call_count = {'count': 0}
    
    @local_app.cache
    def add(a, b):
        call_count['count'] += 1
        return a + b

    # First call should execute the function
    result = add(1, 2)
    assert result == 3, "First call should return correct result"
    assert call_count['count'] == 1, "First call should execute the function"
    
    # Second call with same args should use cache
    result = add(1, 2)
    assert result == 3, "Cached call should return correct result"
    assert call_count['count'] == 1, "Cached call should not re-execute the function"
    
    # Third call with different args should execute again
    result = add(2, 3)
    assert result == 5, "Call with new args should return correct result"
    assert call_count['count'] == 2, "Call with new args should execute the function"


def test_dependency_invalidation():
    """Test that changing a dependency invalidates dependent cached functions."""
    global dep_runs, main_runs
    
    # Initial call should execute both functions
    res1 = main_func2(10)
    assert res1 == 21, "First call should return correct result (10*2 + 1)"
    assert dep_runs == 1, "dep2 should be called once"
    assert main_runs == 1, "main_func2 should be called once"
    
    # Second call with same args should use cache for both
    main_func2(10)
    assert dep_runs == 1, "dep2 should still be called only once (cached)"
    assert main_runs == 1, "main_func2 should still be called only once (cached)"
    
    # Simulate source code change in dep2
    app.source_hashes[Cash._get_func_key(dep2)] = "changed"
    
    # Call main_func2 again - should invalidate and re-execute both functions
    main_func2(10)
    assert dep_runs == 2, "dep2 should be called again after source change"
    assert main_runs == 2, "main_func2 should be re-executed due to dependency invalidation"


def test_imported_dependency():
    """Test that dependencies on imported functions are tracked correctly."""
    # calls_import -> cached_lib_func -> lib_func
    res = calls_import(5)
    assert res == 15, "Function should return correct result (5 * 3)"
    
    # Check that dependency is tracked in the graph
    deps = app.graph.get_dependencies(Cash._get_func_key(calls_import))
    expected_key = Cash._get_func_key(lib_func)
    assert expected_key in deps, f"lib_func ({expected_key}) should be tracked as a dependency of calls_import"
