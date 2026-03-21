"""Experimental features — unstable API, subject to change."""

from __future__ import annotations

import warnings

from cash.exceptions import DependencyNotFoundError


def _warn_experimental(name):
    warnings.warn(
        f"'{name}' is experimental and may change in future versions. "
        f"Import from 'cash.experimental' to acknowledge this.",
        FutureWarning,
        stacklevel=3,
    )


def _load_simple(import_path: str, attr: str):
    """Import *attr* from *import_path* and return it."""
    import importlib
    mod = importlib.import_module(import_path, package='cash.experimental')
    return getattr(mod, attr)


def _load_redis_backend():
    try:
        from ..backends.redis_backend import RedisBackend
        return RedisBackend
    except ImportError:
        raise DependencyNotFoundError(
            "RedisBackend requires 'redis' package. Install with: pip install redis"
        ) from None


def _load_s3_backend():
    try:
        from ..backends.s3_backend import S3Backend
        return S3Backend
    except ImportError:
        raise DependencyNotFoundError(
            "S3Backend requires 'boto3' package. Install with: pip install boto3"
        ) from None


# Maps experimental attribute name -> loader callable (no args).
_LOADERS = {
    'CacheExplorer': lambda: _load_simple('..ui.explorer', 'CacheExplorer'),
    'CacheDebugger': lambda: _load_simple('..ui.debugger', 'CacheDebugger'),
    'visualize_notebook': lambda: _load_simple('..ui.visualizer', 'visualize_notebook'),
    'DependencyGraph': lambda: _load_simple('..graph', 'DependencyGraph'),
    'AnalyticsManager': lambda: _load_simple('..analytics', 'AnalyticsManager'),
    'TieredBackend': lambda: _load_simple('..backends.tiered_backend', 'TieredBackend'),
    'RedisBackend': _load_redis_backend,
    'S3Backend': _load_s3_backend,
}


def __getattr__(name):
    loader = _LOADERS.get(name)
    if loader is None:
        raise AttributeError(f"module 'cash.experimental' has no attribute '{name}'")
    _warn_experimental(name)
    return loader()


__all__ = [
    'CacheExplorer',
    'CacheDebugger',
    'visualize_notebook',
    'DependencyGraph',
    'AnalyticsManager',
    'TieredBackend',
    'RedisBackend',
    'S3Backend',
]
