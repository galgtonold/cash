"""A cached call's overhead is its own: the cash time of the cached calls its
body makes is theirs. Counted in the outer call's too, a study whose miss
paid ~565 ms of its own reported 8.2 s, the sum of its callees' keys and
stores, which those callees reported as well."""

from __future__ import annotations

import pytest

from cash import Cash, phase_times

_NOT_PHASES = ("total", "cash")


def _miss(info: dict) -> dict[str, dict]:
    return {k[5:]: v for k, v in info["phase_times"].items() if k.startswith("miss.")}


@pytest.mark.parametrize("depth", [1, 2])
def test_each_function_reports_only_its_own_overhead(tmp_path, monkeypatch, depth):
    monkeypatch.setattr(phase_times, "ON", True)
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    # The inner call's cash time is the hashing of a large argument and the
    # storing of a large result: no sleep, and far above the outer's own.
    rows = [(i, i * 2, f"x{i % 97}") for i in range(150_000)]

    @c.cache
    def inner(data):
        return list(data)

    @c.cache
    def middle(n):
        return len(inner(rows[:n]))

    @c.cache
    def outer(n):
        return middle(n) if depth == 2 else len(inner(rows[:n]))

    assert outer(len(rows)) == len(rows)
    infos = [
        {**f.cache_info(), **c._registry.cached[c.get_func_key(f.__wrapped__)].stats} for f in (outer, middle, inner)
    ]
    used = [i for i in infos if i["misses"]]
    assert len(used) == depth + 1
    inner_info = used[-1]
    assert inner_info["miss_overhead_seconds"] > 0.01
    whole = _miss(used[0])["total"]["total"]
    for info in used:
        miss = _miss(info)
        # cache_info's overhead is the phases' cash: the call less its body,
        # whose wall time holds the nested calls.
        assert info["miss_overhead_seconds"] == pytest.approx(
            miss["cash"]["total"], abs=0.25 * inner_info["miss_overhead_seconds"]
        )
    # Each function's own overhead adds up to at most the whole call.
    assert sum(i["miss_overhead_seconds"] for i in used) <= whole
