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
from cash.backends._writes import discarded_writes
from cash.config import get_config

pytestmark = pytest.mark.core


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1024", 1024),
        ("2GB", 2_000_000_000),
        ("2gb", 2_000_000_000),
        ("512MiB", 512 * 2**20),
        ("1.5 GiB", int(1.5 * 2**30)),
        ("64KB", 64_000),
    ],
)
def test_sizes_parse(raw, expected):
    from cash.config import parse_size

    assert parse_size(raw) == expected


def test_a_size_string_in_the_constructor_caches_to_disk(tmp_path):
    """THE BUG: every disk write failed for the life of the process."""
    before = len(discarded_writes())
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False, max_cache_size="2GB")
    assert c.config.max_cache_size == 2_000_000_000

    @c.cache(assume_safe=True)
    def slow(x):
        time.sleep(0.2)  # past the persistence floor
        return x

    slow(1)
    slow(1)
    c.shutdown()
    assert len(discarded_writes()) == before, discarded_writes()[before:]
    assert list((tmp_path / ".cash").glob("*.entry")), "nothing reached disk"


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"max_cache_size": "lots"}, "max_cache_size"),
        ({"compress": "maybe"}, "compress"),
        ({"flush_interval": 2.5}, "flush_interval"),
        ({"summary": 2}, "summary"),
    ],
)
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
    project.write_text('[tool.cash]\nmax_cache_size = "lots"\ncompress = true\n', encoding="utf-8")
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
    c = Cash(
        cache_dir=str(tmp_path / ".cash"),
        register_magic=False,
        max_cache_size=10**9,
        compress=True,
        flush_interval=3,
        min_cache_savings_pct=0,
        shutdown_write_timeout=5,
        persist_all=0,
    )
    assert (c.config.max_cache_size, c.config.compress, c.config.flush_interval) == (10**9, True, 3)
    assert c.config.persist_all is False  # 0/1 are ordinary for a flag
    assert c.config.min_cache_savings_pct == 0.0


def test_a_tier_left_out_of_the_stack_is_named(monkeypatch):
    """A tier the environment describes only in part -- no file declares
    tier 1, so it has no type -- was dropped with a debug line, changing the
    stack without a word."""
    import warnings

    from cash import config as cash_config

    monkeypatch.setattr(cash_config, "_CONFIG_NOTICES", set())
    monkeypatch.setenv("CASH_TIER_0_TYPE", "memory")
    monkeypatch.setenv("CASH_TIER_1_HOST", "cache.prod")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cfg = get_config(user_config_path=None, project_config_path=None)
    assert [t.type for t in cfg.tiers] == ["memory"]
    said = [str(w.message) for w in rec if "[CONFIG-INVALID]" in str(w.message)]
    assert said and "tiers[1]" in said[0] and "CASH_TIER_<N>_TYPE" in said[0], [str(w.message) for w in rec]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"max_cache_szie": "1MB"}, r"`max_cache_szie` is not a cash setting\. Did you mean `max_cache_size`\?"),
        ({"ttl": 60}, r"`ttl` is not a cash setting\. ttl is set per function"),
        ({"tiers": [{"type": "file", "nonsense": 1}]}, r"`tiers\[0\]\.nonsense` is not a cash setting"),
        ({"tiers": "memory"}, r"expected a list of tier tables"),
    ],
)
def test_a_keyword_that_is_not_a_setting_raises_in_code(tmp_path, kwargs, match):
    """`Cash(ttl=60)` and typos were dropped without a word, while
    `cash.configure()` raised on the same keys: the user believed entries
    expired after a minute, and the same stale value was served forever.
    `tiers="memory"` was split into one CONFIG-INVALID per character."""
    with pytest.raises(ValueError, match=match):
        Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False, **kwargs)
    with pytest.raises(ValueError, match=match):
        cash.configure(**kwargs)


def test_use_locking_takes_only_a_flag(tmp_path):
    with pytest.raises(ValueError, match="use_locking"):
        Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False, use_locking="no")
    assert Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False, use_locking=1).use_locking is True


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"max_cache_size": -1}, "max_cache_size"),
        ({"max_cache_size": 0}, "max_cache_size"),
        ({"max_memory_entries": -10}, "max_memory_entries"),
        ({"flush_interval": -10}, "flush_interval"),
        ({"shutdown_write_timeout": -5}, "shutdown_write_timeout"),
        ({"shutdown_write_timeout": float("nan")}, "shutdown_write_timeout"),
        ({"min_cache_savings_pct": 1.5}, "min_cache_savings_pct"),
        ({"tiers": [{"type": "file", "max_size_bytes": -1}]}, "max_size_bytes"),
        ({"tiers": [{"type": "file", "default_ttl": -1}]}, "default_ttl"),
    ],
)
def test_a_value_out_of_range_raises_in_code(tmp_path, kwargs, field):
    """`Cash(max_cache_size=-1)` was accepted: nothing reached disk, and the
    messages said "up to -1 B" and "raise max_cache_size above 36 B"."""
    with pytest.raises(ValueError, match=field):
        Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False, **kwargs)


@pytest.mark.parametrize(
    "var, raw", [("CASH_MAX_MEMORY_ENTRIES", "-10"), ("CASH_FLUSH_INTERVAL", "-1"), ("CASH_TIER_0_MAX_SIZE_BYTES", "0")]
)
def test_a_variable_out_of_range_is_reported_and_skipped(monkeypatch, var, raw):
    monkeypatch.setenv(var, raw)
    if var.startswith("CASH_TIER_0"):
        monkeypatch.setenv("CASH_TIER_0_TYPE", "file")
    with pytest.warns(UserWarning, match=r"\[CONFIG-INVALID\]"):
        cfg = get_config(user_config_path=None, project_config_path=None)
    assert cfg.max_memory_entries is None and cfg.flush_interval == 5
    assert all(t.max_size_bytes is None for t in cfg.tiers)


def test_the_edges_of_each_range_are_accepted(tmp_path):
    c = Cash(
        cache_dir=str(tmp_path / ".cash"),
        register_magic=False,
        max_cache_size=1,
        max_memory_entries=1,
        flush_interval=0,
        shutdown_write_timeout=0,
        min_cache_savings_pct=1,
    )
    assert (c.config.max_cache_size, c.config.flush_interval, c.config.shutdown_write_timeout) == (1, 0, 0)


def test_an_empty_cache_dir_variable_counts_as_unset(monkeypatch, tmp_path):
    """`CASH_CACHE_DIR=${X:-}` in CI set it to "", and the cache was written
    into the current directory, among the user's source files; `cash info`
    showed a blank cache dir and `cash clear --all` found nothing."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CASH_CACHE_DIR", raising=False)
    default = get_config(user_config_path=None, project_config_path=None).cache_dir
    monkeypatch.setenv("CASH_CACHE_DIR", "")
    cfg = get_config(user_config_path=None, project_config_path=None)
    assert cfg.cache_dir == default != str(tmp_path)


@pytest.mark.parametrize("empty", ["", "  "])
def test_an_empty_cache_dir_in_code_raises(tmp_path, empty):
    with pytest.raises(ValueError, match="cache_dir"):
        Cash(cache_dir=empty, register_magic=False)
    with pytest.raises(ValueError, match="cache_dir"):
        Cash(register_magic=False, tiers=[{"type": "file", "cache_dir": empty}])
    with pytest.raises(ValueError, match="cache_dir"):
        cash.configure(cache_dir=empty)


def test_an_empty_cache_dir_in_a_file_is_reported_and_skipped(tmp_path):
    pytest.importorskip("tomllib" if __import__("sys").version_info >= (3, 11) else "tomli")
    project = tmp_path / "pyproject.toml"
    project.write_text('[tool.cash]\ncache_dir = ""\n', encoding="utf-8")
    with pytest.warns(UserWarning, match=r"\[CONFIG-INVALID\].*cache_dir"):
        cfg = get_config(project_config_path=str(project), user_config_path=None)
    assert cfg.cache_dir != str(tmp_path)
