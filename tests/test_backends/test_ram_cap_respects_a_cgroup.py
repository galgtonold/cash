"""The RAM tier's cap must be sized from memory this process may actually use.

Round-15 S2 watched a long-lived worker's RSS climb 141 MB -> 1502 MB over 300
keys and read it as a leak. It was not: the RAM tier is bounded by default (a
fifth of memory, clamped to [512 MiB, 4 GiB]) and was doing exactly what it
says. What survived the adjudication is narrower and sharper:

* the budget came from ``psutil.virtual_memory().total``, which reports the
  HOST's memory -- ``/proc/meminfo`` is not namespaced, so a 2 GiB container on
  a 64 GiB host was handed the full 4 GiB ceiling: a cache budget twice the
  memory the process is allowed, which is an OOM kill rather than an eviction;
* the resolved cap appeared nowhere a user could look, and the constructor
  docstring said "unbounded", which is true of the constructor and false of
  every backend the factory builds. That is what put "the in-memory tier is
  unbounded by default" into a ticket.

The cgroup files are read here from synthetic paths rather than a real
container -- there is none on this machine, and the parsing (v2's ``max``,
v1's huge sentinel) is the part that can be wrong. Labelled as such rather than
claimed as a container test.
"""
from __future__ import annotations

import pytest

from cash.backends import adaptive_caps


@pytest.fixture
def cgroup(tmp_path, monkeypatch):
    """Point the cgroup lookup at files this test writes."""
    v2 = tmp_path / "memory.max"
    v1 = tmp_path / "memory.limit_in_bytes"
    monkeypatch.setattr(adaptive_caps, "_CGROUP_LIMIT_PATHS", (str(v2), str(v1)))
    return v2, v1


def test_no_cgroup_files_means_no_limit(cgroup):
    """macOS, Windows, and any un-limited Linux process."""
    assert adaptive_caps._cgroup_memory_limit() is None


def test_a_v2_limit_is_read(cgroup):
    v2, _ = cgroup
    v2.write_text("2147483648\n", encoding="utf-8")
    assert adaptive_caps._cgroup_memory_limit() == 2 * 1024 ** 3


def test_v2_max_means_unlimited(cgroup):
    """cgroup v2 spells "no limit" as a word, not a number."""
    v2, _ = cgroup
    v2.write_text("max\n", encoding="utf-8")
    assert adaptive_caps._cgroup_memory_limit() is None


def test_a_v1_limit_is_read_when_v2_is_absent(cgroup):
    _, v1 = cgroup
    v1.write_text("1073741824\n", encoding="utf-8")
    assert adaptive_caps._cgroup_memory_limit() == 1024 ** 3


def test_the_v1_sentinel_is_not_a_limit(cgroup):
    """v1 spells "no limit" as a huge number; treating it as one would cap the
    RAM tier at eight exabytes and read as a limit that binds."""
    _, v1 = cgroup
    v1.write_text("9223372036854771712\n", encoding="utf-8")
    assert adaptive_caps._cgroup_memory_limit() is None


def test_garbage_is_ignored_rather_than_raising(cgroup):
    v2, _ = cgroup
    v2.write_text("not-a-number\n", encoding="utf-8")
    assert adaptive_caps._cgroup_memory_limit() is None


# --- what the budget does with it -------------------------------------------

def test_the_budget_is_the_smaller_of_the_two(cgroup, monkeypatch):
    """The container case: a big host, a small limit."""
    v2, _ = cgroup
    v2.write_text(str(2 * 1024 ** 3), encoding="utf-8")
    monkeypatch.setattr(adaptive_caps, "_total_system_ram", lambda: 64 * 1024 ** 3)

    assert adaptive_caps._memory_budget() == 2 * 1024 ** 3


def test_the_host_total_is_used_when_nothing_limits_the_process(cgroup, monkeypatch):
    monkeypatch.setattr(adaptive_caps, "_total_system_ram", lambda: 64 * 1024 ** 3)
    assert adaptive_caps._memory_budget() == 64 * 1024 ** 3


def test_a_limited_container_gets_a_cap_inside_its_limit(cgroup, monkeypatch):
    """The whole point: the cap must fit in the memory the process may use.

    2 GiB limit -> a fifth is 409.6 MiB -> the 512 MiB floor applies, which is
    a quarter of the limit. Before this, the same process was handed the 4 GiB
    ceiling: twice what it was allowed to allocate in total.
    """
    v2, _ = cgroup
    v2.write_text(str(2 * 1024 ** 3), encoding="utf-8")
    monkeypatch.setattr(adaptive_caps, "_total_system_ram", lambda: 64 * 1024 ** 3)

    cap = adaptive_caps.resolve_ram_cap()

    assert cap < 2 * 1024 ** 3, "a cache budget above the container's own limit"
    assert cap == adaptive_caps.RAM_FLOOR


def test_an_unlimited_process_is_unaffected(cgroup, monkeypatch):
    """The control: no cgroup limit, so the answer must not change."""
    monkeypatch.setattr(adaptive_caps, "_total_system_ram", lambda: 64 * 1024 ** 3)
    assert adaptive_caps.resolve_ram_cap() == adaptive_caps.RAM_CEILING
