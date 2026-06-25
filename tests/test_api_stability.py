"""Tests for cash.experimental namespace."""
import pytest


def test_experimental_import_cache_explorer():
    """Test that CacheExplorer can be imported from experimental."""
    from cash.experimental import CacheExplorer
    assert CacheExplorer is not None


def test_experimental_import_cache_debugger():
    """Test that CacheDebugger can be imported from experimental."""
    from cash.experimental import CacheDebugger
    assert CacheDebugger is not None


def test_experimental_import_analytics():
    """Test that AnalyticsManager can be imported from experimental."""
    from cash.experimental import AnalyticsManager
    assert AnalyticsManager is not None


def test_experimental_import_tiered_backend():
    """Test that TieredBackend can be imported from experimental."""
    from cash.experimental import TieredBackend
    assert TieredBackend is not None


def test_experimental_import_nonexistent():
    """Test that importing nonexistent attribute raises ImportError."""
    with pytest.raises(ImportError):
        from cash.experimental import NonExistentThing  # noqa


def test_core_exports_stable():
    """Test that core __init__ exports only stable APIs."""
    import cash
    expected_stable = {
        # Core API
        'Cash', 'CacheExplanation', 'cache', 'show_stats',
        'register_hasher', 'reset_session', 'configure', 'cleanup',
        # Purity declarations
        'pure', 'stateful', 'is_pure', 'is_stateful',
        'analyze_function_purity', 'mark_pure', 'mark_stateful',
        # Configuration
        'get_config', 'CashConfig', 'create_default_config',
        # Backends
        'InMemoryBackend', 'FileBackend', 'SQLiteBackend',
        'CascadingBackend',
        # Data sources
        'DataSource', 'FileDataSource',
        # Exception hierarchy
        'CashError', 'CacheBackendError', 'CacheExpiredError',
        'CacheSerializationError', 'DependencyNotFoundError',
        'AmbiguousCellError', 'UpstreamStateError',
        'CacheKeyComputationError', 'CashImpureFunctionError',
        # Warnings
        'CashWarning', 'CashCacheIneffectiveWarning',
        'CashCacheStoreFailedWarning', 'CashImpurityWarning',
    }
    actual = set(cash.__all__)
    assert actual == expected_stable
