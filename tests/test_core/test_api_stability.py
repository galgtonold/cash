"""The public names ``cash`` exports."""


def test_core_exports_stable():
    """Test that core __init__ exports only stable APIs."""
    import cash

    expected_stable = {
        # Core API
        "Cash",
        "CacheExplanation",
        "reset_session",
        "configure",
        "cleanup",
        "disabled",  # a no-cache block that restores CASH_DISABLE
        # Purity declarations
        "pure",
        "stateful",
        "is_pure",
        "is_stateful",
        "assume_safe",  # waives purity findings for a block
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


def test_a_star_import_leaves_help_alone_and_builds_nothing():
    """``__all__`` listed ``help``, so ``from cash import *`` replaced the
    builtin, and ``cache`` / ``show_stats`` / ``register_hasher``, which the
    module ``__getattr__`` serves by building the global ``Cash``."""
    import subprocess
    import sys

    code = (
        "import builtins\n"
        "from cash import *\n"
        "import cash\n"
        "assert help is builtins.help, help\n"
        "assert cash._active.default_cash() is None\n"
        "assert {'cache', 'show_stats', 'register_hasher', 'help'} <= set(dir(cash))\n"
        "assert callable(cash.help) and callable(cash.cache)\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
