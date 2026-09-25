"""A relative ``cache_dir`` given in code or the environment is resolved once,
when the setting is read, against the directory it was given in.

It was kept relative and resolved when used: ``configure(cache_dir="rel")``
then ``os.chdir`` before the first call built the cache in the new directory,
while pool workers (handed the path made absolute at configure time) used the
old one; and a ``configure(max_cache_size=...)`` after a chdir rebuilt the
backend in the new cwd. A tier's ``cache_dir`` behaved the same.
"""

from __future__ import annotations

import os

import pytest

from cash import Cash


@pytest.fixture
def two_dirs(tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.chdir(first)
    return first, second


def test_configure_then_chdir_keeps_the_directory(two_dirs, monkeypatch):
    first, second = two_dirs
    c = Cash(register_magic=False)
    c.reconfigure(cache_dir="rel")
    monkeypatch.chdir(second)
    assert c.backend.local_dir == str(first / "rel")


def test_a_rebuild_after_chdir_stays_where_the_setting_was_given(two_dirs, monkeypatch):
    first, second = two_dirs
    c = Cash(cache_dir="rel", register_magic=False)
    c.backend  # built in `first`
    monkeypatch.chdir(second)
    c.reconfigure(max_cache_size=10**9)  # rebuilds the tiers
    assert c.backend.local_dir == str(first / "rel")


def test_the_environment_variable_is_resolved_where_it_was_read(two_dirs, monkeypatch):
    first, second = two_dirs
    monkeypatch.setenv("CASH_CACHE_DIR", "rel")
    c = Cash(register_magic=False)
    monkeypatch.chdir(second)
    assert c.backend.local_dir == str(first / "rel")


@pytest.mark.parametrize("given", ["code", "env"])
def test_a_tier_s_cache_dir_is_resolved_the_same_way(two_dirs, monkeypatch, given):
    first, second = two_dirs
    if given == "code":
        c = Cash(register_magic=False, tiers=[{"type": "memory"}, {"type": "file", "cache_dir": "tier"}])
    else:
        monkeypatch.setenv("CASH_TIER_0_TYPE", "file")
        monkeypatch.setenv("CASH_TIER_0_CACHE_DIR", "tier")
        c = Cash(register_magic=False)
    monkeypatch.chdir(second)
    assert c.backend.local_dir == str(first / "tier")


def test_configure_resolves_a_tier_s_cache_dir(two_dirs, monkeypatch):
    first, second = two_dirs
    c = Cash(register_magic=False)
    c.reconfigure(tiers=[{"type": "file", "cache_dir": "tier"}])
    monkeypatch.chdir(second)
    assert c.backend.local_dir == str(first / "tier")
    assert os.path.isabs(c.config.tiers[0].cache_dir)
