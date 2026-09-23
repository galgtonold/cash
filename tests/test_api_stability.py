"""The public names ``cash`` exports."""


def test_core_exports_stable():
    """Test that core __init__ exports only stable APIs."""
    import cash

    expected_stable = {
        # Core API
        "Cash",
        "CacheExplanation",
        "cache",
        "show_stats",
        "register_hasher",
        "reset_session",
        "configure",
        "cleanup",
        "help",  # public since d30849a (orientation summary, aimed at coding agents)
        "disabled",  # a no-cache block that restores CASH_DISABLE
        # Purity declarations
        "pure",
        "stateful",
        "is_pure",
        "is_stateful",
        # Code-surface opt-out
        "opaque",  # public since 06c2bd6 (the code-surface escape hatch)
        # Configuration
        "get_config",
        "CashConfig",
        "create_default_config",
        # Backends
        "InMemoryBackend",
        "FileBackend",
        "SQLiteBackend",
        "TieredBackend",
        # Data sources
        "DataSource",
        "FileDataSource",
        "RemoteFileDataSource",  # track s3://, gs://, http(s):// objects
        # Exception hierarchy
        "CashError",
        "CacheBackendError",
        "CacheExpiredError",
        "CacheSerializationError",
        "DependencyNotFoundError",
        "AmbiguousCellError",
        "UpstreamStateError",
        # public since the forward-reference fix: a cell that
        # reads a name only a LATER cell binds now fails instead of
        # caching against a namespace an in-order run cannot rebuild.
        "ForwardReferenceError",
        "CacheKeyComputationError",
        "CashImpureFunctionError",
        # Warnings
        "CashWarning",
        "CashCacheIneffectiveWarning",
        "CashCacheStoreFailedWarning",
        "CashImpurityWarning",
        "CashRandomnessWarning",  # users filter on it
        "CashUpstreamSyntaxWarning",  # users filter on it
    }
    actual = set(cash.__all__)
    assert actual == expected_stable
