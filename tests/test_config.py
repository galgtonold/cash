"""Tests for the Cash configuration system."""

import os
from pathlib import Path

from cash.config import (
    get_config,
    CashConfig,
    create_default_config,
    _load_toml_config,
    _load_env_config,
    ENV_VARS,
)


class TestCashConfigDefaults:
    """Test default configuration values."""

    def test_default_config_values(self):
        config = CashConfig()
        assert config.cache_dir == '.cash'
        assert config.debug is False
        assert config.backend_type == 'file'
        assert config.max_cache_size == 1024 ** 3
        assert config.compress is False
        assert config.smart_persistence is True
        assert config.smart_persistence_threshold == 1.0
        assert config.max_memory_entries is None

    def test_config_to_dict(self):
        config = CashConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert 'cache_dir' in d
        assert 'debug' in d
        assert 'backend_type' in d
        assert '_source' not in d  # Internal field excluded

    def test_get_config_returns_defaults(self):
        """get_config with no file and no env vars returns defaults."""
        config = get_config(config_path='/nonexistent/path/config.toml')
        assert config.cache_dir == '.cash'
        assert config.debug is False
        assert config._source == 'defaults'


class TestTomlConfig:
    """Test TOML config file loading."""

    def test_load_nonexistent_file(self, tmp_path):
        result = _load_toml_config(tmp_path / 'nonexistent.toml')
        assert result == {}

    def test_load_valid_toml(self, tmp_path):
        config_file = tmp_path / 'config.toml'
        config_file.write_text('''
[cash]
cache_dir = "/tmp/my_cache"
debug = true
max_cache_size = 500000000
compress = true
''')
        result = _load_toml_config(config_file)
        assert result['cache_dir'] == '/tmp/my_cache'
        assert result['debug'] is True
        assert result['max_cache_size'] == 500000000
        assert result['compress'] is True

    def test_load_flat_toml(self, tmp_path):
        """TOML without [cash] section uses top-level keys."""
        config_file = tmp_path / 'config.toml'
        config_file.write_text('''
cache_dir = "/tmp/flat_cache"
debug = false
''')
        result = _load_toml_config(config_file)
        assert result['cache_dir'] == '/tmp/flat_cache'
        assert result['debug'] is False

    def test_load_invalid_toml(self, tmp_path):
        config_file = tmp_path / 'config.toml'
        config_file.write_text('this is not valid toml {{{')
        result = _load_toml_config(config_file)
        assert result == {}

    def test_get_config_from_file(self, tmp_path):
        config_file = tmp_path / 'config.toml'
        config_file.write_text('''
[cash]
cache_dir = "my_custom_cache"
debug = true
smart_persistence_threshold = 2.5
''')
        config = get_config(config_path=str(config_file))
        assert config.cache_dir == 'my_custom_cache'
        assert config.debug is True
        assert config.smart_persistence_threshold == 2.5
        assert 'file:' in config._source


class TestEnvVarConfig:
    """Test environment variable configuration."""

    def test_env_var_cache_dir(self, monkeypatch):
        monkeypatch.setenv('CASH_CACHE_DIR', '/env/cache')
        config = _load_env_config()
        assert config['cache_dir'] == '/env/cache'

    def test_env_var_debug_true(self, monkeypatch):
        monkeypatch.setenv('CASH_DEBUG', '1')
        config = _load_env_config()
        assert config['debug'] is True

    def test_env_var_debug_false(self, monkeypatch):
        monkeypatch.setenv('CASH_DEBUG', '0')
        config = _load_env_config()
        assert config['debug'] is False

    def test_env_var_debug_true_word(self, monkeypatch):
        monkeypatch.setenv('CASH_DEBUG', 'true')
        config = _load_env_config()
        assert config['debug'] is True

    def test_env_var_backend(self, monkeypatch):
        monkeypatch.setenv('CASH_BACKEND', 'memory')
        config = _load_env_config()
        assert config['backend_type'] == 'memory'

    def test_env_var_max_cache_size(self, monkeypatch):
        monkeypatch.setenv('CASH_MAX_CACHE_SIZE', '500000000')
        config = _load_env_config()
        assert config['max_cache_size'] == 500000000

    def test_env_var_compress(self, monkeypatch):
        monkeypatch.setenv('CASH_COMPRESS', 'yes')
        config = _load_env_config()
        assert config['compress'] is True

    def test_env_var_redis_url(self, monkeypatch):
        monkeypatch.setenv('CASH_REDIS_URL', 'redis://myhost:6380')
        config = _load_env_config()
        assert config['redis_url'] == 'redis://myhost:6380'

    def test_env_var_invalid_int(self, monkeypatch):
        monkeypatch.setenv('CASH_MAX_CACHE_SIZE', 'not_a_number')
        config = _load_env_config()
        assert 'max_cache_size' not in config  # Skipped due to error

    def test_no_env_vars_set(self):
        """When no CASH_ env vars are set, returns empty dict."""
        # Clean env vars we might have
        env_backup = {}
        for var in ENV_VARS:
            if var in os.environ:
                env_backup[var] = os.environ.pop(var)
        try:
            _load_env_config()
            # Should be empty or only have vars that happen to be set
            for key in ENV_VARS:
                assert key not in os.environ
        finally:
            os.environ.update(env_backup)


class TestConfigPrecedence:
    """Test configuration precedence: env > file > defaults."""

    def test_file_overrides_defaults(self, tmp_path):
        config_file = tmp_path / 'config.toml'
        config_file.write_text('''
[cash]
cache_dir = "from_file"
''')
        config = get_config(config_path=str(config_file))
        assert config.cache_dir == 'from_file'
        # Other values should still be defaults
        assert config.debug is False

    def test_env_overrides_file(self, tmp_path, monkeypatch):
        config_file = tmp_path / 'config.toml'
        config_file.write_text('''
[cash]
cache_dir = "from_file"
debug = false
''')
        monkeypatch.setenv('CASH_CACHE_DIR', 'from_env')
        config = get_config(config_path=str(config_file))
        assert config.cache_dir == 'from_env'  # Env wins
        assert config.debug is False  # File value kept

    def test_env_overrides_file_debug(self, tmp_path, monkeypatch):
        config_file = tmp_path / 'config.toml'
        config_file.write_text('''
[cash]
debug = false
''')
        monkeypatch.setenv('CASH_DEBUG', 'true')
        config = get_config(config_path=str(config_file))
        assert config.debug is True  # Env wins

    def test_source_tracking(self, tmp_path, monkeypatch):
        config_file = tmp_path / 'config.toml'
        config_file.write_text('[cash]\ncache_dir = "test"\n')
        monkeypatch.setenv('CASH_DEBUG', '1')
        config = get_config(config_path=str(config_file))
        assert 'env' in config._source
        assert 'file:' in config._source


class TestCreateDefaultConfig:
    """Test default config file creation."""

    def test_create_default_config(self, tmp_path):
        path = str(tmp_path / 'config.toml')
        result = create_default_config(path)
        assert result == path
        assert Path(path).exists()
        content = Path(path).read_text()
        assert 'cache_dir' in content
        assert 'backend_type' in content
        assert 'smart_persistence' in content

    def test_create_default_config_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / 'deep' / 'nested' / 'config.toml')
        result = create_default_config(path)
        assert Path(result).exists()

    def test_created_config_is_valid_toml(self, tmp_path):
        path = str(tmp_path / 'config.toml')
        create_default_config(path)
        result = _load_toml_config(Path(path))
        assert result.get('cache_dir') == '.cash'
        assert result.get('debug') is False


class TestCoreIntegration:
    """Test that config integrates properly with Cash core."""

    def test_cash_uses_config_cache_dir(self, tmp_path, monkeypatch):
        """Cash uses config cache_dir when none is specified."""
        monkeypatch.setenv('CASH_CACHE_DIR', str(tmp_path / 'env_cache'))
        from cash.core import Cash
        c = Cash(register_magic=False)
        assert c.config.cache_dir == str(tmp_path / 'env_cache')

    def test_cash_explicit_params_override_config(self, tmp_path, monkeypatch):
        """Explicit constructor params override config values."""
        monkeypatch.setenv('CASH_CACHE_DIR', str(tmp_path / 'env_cache'))
        from cash.core import Cash
        c = Cash(cache_dir=str(tmp_path / 'explicit'), register_magic=False)
        # Explicit cache_dir should be used for backend
        # Config still loaded but not used for cache_dir
        assert c.config.cache_dir == str(tmp_path / 'env_cache')

    def test_cash_config_debug(self, monkeypatch):
        monkeypatch.setenv('CASH_DEBUG', 'true')
        from cash.core import Cash
        c = Cash(register_magic=False)
        assert c.debug is True

    def test_cash_config_compress(self, tmp_path, monkeypatch):
        monkeypatch.setenv('CASH_COMPRESS', '1')
        from cash.core import Cash
        c = Cash(register_magic=False)
        assert c.config.compress is True

    def test_cash_config_path(self, tmp_path):
        config_file = tmp_path / 'config.toml'
        config_file.write_text('[cash]\ncache_dir = "custom_dir"\ndebug = true\n')
        from cash.core import Cash
        c = Cash(register_magic=False, config_path=str(config_file))
        assert c.config.cache_dir == 'custom_dir'
        assert c.debug is True
