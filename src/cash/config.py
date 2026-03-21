"""Configuration system: env vars > config file > defaults."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["CashConfig", "get_config", "create_default_config"]

# Environment variable mapping
ENV_VARS = {
    'CASH_CACHE_DIR': ('cache_dir', str),
    'CASH_DEBUG': ('debug', lambda v: v.lower() in ('1', 'true', 'yes')),
    'CASH_BACKEND': ('backend_type', str),
    'CASH_MAX_CACHE_SIZE': ('max_cache_size', int),
    'CASH_COMPRESS': ('compress', lambda v: v.lower() in ('1', 'true', 'yes')),
    'CASH_REDIS_URL': ('redis_url', str),
    'CASH_S3_BUCKET': ('s3_bucket', str),
    'CASH_MAX_MEMORY_ENTRIES': ('max_memory_entries', lambda v: int(v) if v else None),
    'CASH_MIN_CACHE_SAVINGS_PCT': ('min_cache_savings_pct', float),
    'CASH_SERIALIZATION_SPEED': ('estimated_serialization_speed', int),
}


@dataclass
class CashConfig:
    """Cash configuration settings."""
    cache_dir: str = '.cash'
    debug: bool = False
    backend_type: str = 'file'
    max_cache_size: int = 1024 ** 3
    compress: bool = False
    flush_interval: int = 5
    smart_persistence: bool = True
    smart_persistence_threshold: float = 1.0
    max_memory_entries: int | None = None
    min_cache_savings_pct: float = 0.20  # skip caching when expected savings < 20%
    estimated_serialization_speed: int = 200 * 1024 * 1024  # ~200 MB/s (disk; RAM uses deepcopy)
    redis_url: str = 'redis://localhost:6379'
    redis_prefix: str = 'cash:'
    s3_bucket: str = ''
    s3_prefix: str = 'cash/'

    # Source tracking
    _source: str = 'defaults'  # Where this config came from

    def to_dict(self) -> dict[str, Any]:
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if not f.name.startswith('_')
        }

def _load_toml_config(path: Path) -> dict[str, Any]:
    """Load configuration from a TOML file.

    Returns empty dict if file doesn't exist or can't be parsed.
    """
    if not path.exists():
        return {}

    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            logger.debug("Neither tomllib nor tomli available, skipping TOML config")
            return {}

    try:
        with open(path, 'rb') as f:
            data = tomllib.load(f)
        # Flatten nested structure: [cash] section
        return data.get('cash', data)
    except Exception as e:
        logger.warning("Error loading config from %s: %s", path, e)
        return {}


def _load_env_config() -> dict[str, Any]:
    """Load configuration from environment variables.

    Returns dict with only the values that were set.
    """
    result = {}
    for env_var, (config_key, converter) in ENV_VARS.items():
        value = os.environ.get(env_var)
        if value is not None:
            try:
                result[config_key] = converter(value)
            except (ValueError, TypeError) as e:
                logger.warning("Invalid value for %s=%r: %s", env_var, value, e)
    return result


def get_config(config_path: str | None = None) -> CashConfig:
    """Get the merged Cash configuration.

    Priority (highest to lowest):
    1. Environment variables
    2. Config file (specified path or ~/.cash/config.toml)
    3. Built-in defaults

    Args:
        config_path: Optional path to config file. If None, uses ~/.cash/config.toml

    Returns:
        CashConfig with merged settings
    """
    # Start with defaults from CashConfig dataclass
    defaults = CashConfig()
    config_dict = {
        f.name: getattr(defaults, f.name)
        for f in fields(CashConfig)
        if not f.name.startswith('_')
    }
    source = 'defaults'

    # Load from config file
    file_path = Path(config_path) if config_path else Path.home() / '.cash' / 'config.toml'

    file_config = _load_toml_config(file_path)
    if file_config:
        config_dict.update(file_config)
        source = f'file:{file_path}'

    # Load from environment (highest priority)
    env_config = _load_env_config()
    if env_config:
        config_dict.update(env_config)
        source = 'env' if not file_config else f'env+file:{file_path}'

    # Create config object
    config = CashConfig()
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config._source = source

    return config


def create_default_config(path: str | None = None) -> str:
    """Create a default config file with documented settings.

    Args:
        path: Where to create the file. Defaults to ~/.cash/config.toml

    Returns:
        Path to the created config file
    """
    if path is None:
        config_dir = Path.home() / '.cash'
        config_dir.mkdir(parents=True, exist_ok=True)
        path = str(config_dir / 'config.toml')

    content = '''# Cash Configuration
# See: https://github.com/your-repo/cash for documentation

[cash]
# Cache directory (relative to notebook or absolute path)
cache_dir = ".cash"

# Debug mode (show detailed caching information)
debug = false

# Backend type: "file", "memory", "redis", "s3"
backend_type = "file"

# Maximum cache size in bytes (default: 1GB)
max_cache_size = 1073741824

# Compress cache entries (reduces disk usage, increases CPU)
compress = false

# Smart persistence: only persist to disk if execution time > threshold
smart_persistence = true
smart_persistence_threshold = 1.0

# Maximum number of entries in memory cache (null = unlimited)
# max_memory_entries = 1000

# Redis backend settings
# redis_url = "redis://localhost:6379"
# redis_prefix = "cash:"

# S3 backend settings
# s3_bucket = "my-cache-bucket"
# s3_prefix = "cash/"
'''

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)

    return path
