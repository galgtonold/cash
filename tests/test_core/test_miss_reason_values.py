"""Miss reasons are values, not strings to split: the kind, the detail, and
for a code change what changed, each in its own field."""

from __future__ import annotations

import pytest

from cash import Cash
from cash.decorator.explain import MissKind, MissReason

pytestmark = pytest.mark.core


def test_a_reason_reads_as_before():
    reason = MissReason(MissKind.CODE, "the code changed since the last call", "helper m._f changed")
    assert reason.text == "the code changed since the last call -- helper m._f changed"
    assert str(reason) == "code or state changed: the code changed since the last call -- helper m._f changed"
    assert str(MissReason(MissKind.FIRST)) == "no entry yet"


def test_a_kind_counts_as_its_words():
    assert {MissKind.ARGS: 1} == {"new arguments": 1}
    assert f"{MissKind.TTL}" == "ttl expired"


def test_cache_info_counts_under_plain_strings(tmp_path):
    c = Cash(cache_dir=str(tmp_path / "c"), register_magic=False)

    @c.cache
    def f(x):
        return x

    f(1)
    f(2)
    reasons = f.cache_info()["miss_reasons"]
    assert reasons == {"no entry yet": 1, "new arguments": 1}
    assert all(type(kind) is str for kind in reasons)
    c.shutdown()
