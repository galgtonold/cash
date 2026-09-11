"""A config value of the wrong type is refused or reported, never stored.

`Cash(max_cache_size="2GB")` was accepted and stored as the string. The first
disk write then compared it with an int and failed, and every write after it
did too: the cache kept nothing on disk for the rest of the process, reported
only as a count of discarded writes. The environment variable was checked
(and rejected "1GB" outright); the constructor, `cash.configure()` and the
TOML files were not checked at all.

Now every layer goes through one check. Explicit code raises -- that is a bug
at the call site -- while a file or the environment is reported and skipped, so
a stray setting cannot stop a program. Byte-size fields also take "2GB" /
"512MiB", since that is how people write them.
"""
from __future__ import annotations

import time

import pytest

import cash
from cash import Cash
from cash.backends._base import discarded_writes
from cash.config import get_config

pytestmark = pytest.mark.core


@pytest.mark.parametrize("raw, expected", [
    ("1024", 1024),
    ("2GB", 2_000_000_000),
    ("2gb", 2_000_000_000),
    ("512MiB", 512 * 2**20),
    ("1.5 GiB", int(1.5 * 2**30)),
    ("64KB", 64_000),
])
def test_sizes_parse(raw, expected):
    from cash.config import parse_size
    assert parse_size(raw) == expected


def test_a_size_string_in_the_constructor_caches_to_disk(tmp_path):
    """THE BUG: every disk write failed for the life of the process."""
    before = len(discarded_writes())
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False,
             max_cache_size="2GB")
    assert c.config.max_cache_size == 2_000_000_000

    @c.cache(assume_safe=True)
    def slow(x):
        time.sleep(0.2)                     # past the persistence floor
        return x

    slow(1)
    slow(1)
    c.shutdown()
    assert len(discarded_writes()) == before, discarded_writes()[before:]
    assert list((tmp_path / ".cash").glob("*.entry")), "nothing reached disk"


@pytest.mark.parametrize("kwargs, field", [
    ({"max_cache_size": "lots"}, "max_cache_size"),
    ({"compress": "maybe"}, "compress"),
    ({"flush_interval": 2.5}, "flush_interval"),
    ({"summary": 2}, "summary"),
])
def test_a_bad_constructor_value_raises_naming_the_field(tmp_path, kwargs, field):
    with pytest.raises(ValueError, match=field):
        Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False, **kwargs)


def test_configure_refuses_and_leaves_the_config_alone():
    before = cash._get_global_cash().config.flush_interval
    with pytest.raises(ValueError, match="flush_interval"):
        cash.configure(flush_interval="soon")
    assert cash._get_global_cash().config.flush_interval == before


def test_a_bad_toml_value_is_reported_and_skipped(tmp_path):
    """From a file: say so (CONFIG-INVALID) and carry on with the default."""
    project = tmp_path / "pyproject.toml"
    project.write_text('[tool.cash]\nmax_cache_size = "lots"\ncompress = true\n',
                       encoding="utf-8")
    pytest.importorskip("tomllib" if __import__("sys").version_info >= (3, 11) else "tomli")
    with pytest.warns(UserWarning, match=r"\[CONFIG-INVALID\].*max_cache_size"):
        cfg = get_config(project_config_path=str(project), user_config_path=None)
    assert cfg.max_cache_size is None, "the invalid value was stored"
    assert cfg.compress is True, "the valid setting beside it was lost"


def test_a_size_string_in_toml_is_read(tmp_path):
    pytest.importorskip("tomllib" if __import__("sys").version_info >= (3, 11) else "tomli")
    project = tmp_path / "pyproject.toml"
    project.write_text('[tool.cash]\nmax_cache_size = "512MiB"\n', encoding="utf-8")
    cfg = get_config(project_config_path=str(project), user_config_path=None)
    assert cfg.max_cache_size == 512 * 2**20


def test_the_env_var_takes_a_size_string(monkeypatch):
    monkeypatch.setenv("CASH_MAX_CACHE_SIZE", "500MB")
    assert get_config(user_config_path=None, project_config_path=None).max_cache_size == 500_000_000


def test_valid_values_of_every_kind_still_pass(tmp_path):
    """The control: the check must not reject what always worked."""
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False,
             max_cache_size=10**9, compress=True, flush_interval=3,
             min_cache_savings_pct=0, shutdown_write_timeout=5, persist_all=0)
    assert (c.config.max_cache_size, c.config.compress, c.config.flush_interval) == (10**9, True, 3)
    assert c.config.persist_all is False     # 0/1 are ordinary for a flag
    assert c.config.min_cache_savings_pct == 0.0
