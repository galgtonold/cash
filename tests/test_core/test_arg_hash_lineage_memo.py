"""Session memo for argument content-hashing (speed A).

A repeated ``@cash.cache`` call with the SAME unmutated argument used to
re-hash the whole (possibly huge) value every time. cash now memoises the
reproducible content hash keyed on the value's ``_cash_lineage_hash`` — the same
mutation signal it already trusts to cache notebook statements — so the second
call reuses the hash instead of recomputing it. The stored cache key is
unchanged (still the content hash), so this is a pure within-session speedup.
"""
from __future__ import annotations

import pandas as pd

from cash import Cash, InMemoryBackend


def _cash() -> Cash:
    return Cash(backend=InMemoryBackend())


def _df(vals, lineage=None):
    d = pd.DataFrame({"a": vals})
    if lineage is not None:
        d._cash_lineage_hash = lineage
    return d


def _count_hashes(c: Cash):
    """Wrap the (expensive) builtin content hasher to count invocations."""
    calls = {"n": 0}
    orig = c._try_builtin_type_hash

    def spy(v):
        calls["n"] += 1
        return orig(v)

    c._try_builtin_type_hash = spy  # shadows the staticmethod for self. access
    return calls


def test_same_lineage_reuses_hash_without_recomputing():
    c = _cash()
    calls = _count_hashes(c)
    df = _df(range(100), lineage="L1")

    k1 = c._serialize_args("f", (df,), {})
    k2 = c._serialize_args("f", (df,), {})

    assert k1 == k2
    assert calls["n"] == 1, "second call must hit the memo, not re-hash"


def test_lineage_bump_recomputes_and_changes_key():
    c = _cash()
    calls = _count_hashes(c)
    df = _df(range(100), lineage="L1")
    k1 = c._serialize_args("f", (df,), {})

    # A real mutation changes the content AND bumps the lineage cash tracks.
    df["a"] = list(range(100, 200))
    df._cash_lineage_hash = "L2"
    k2 = c._serialize_args("f", (df,), {})

    assert calls["n"] == 2, "a changed lineage must force a fresh hash (memo miss)"
    assert k1 != k2, "changed content must change the cache key"


def test_memoised_key_equals_plain_content_hash():
    """The memo must NOT drift the cache key: a lineage-carrying value keys
    identically to the same value with no lineage (content hash), so persisted
    entries stay valid and survive a restart (where lineage is gone)."""
    c = _cash()
    with_lineage = c._serialize_args("f", (_df(range(100), lineage="L1"),), {})
    plain = _cash()._serialize_args("f", (_df(range(100)),), {})
    assert with_lineage == plain


def test_no_lineage_still_content_hashes_every_call():
    c = _cash()
    calls = _count_hashes(c)
    df = _df(range(100))                   # no lineage attribute
    c._serialize_args("f", (df,), {})
    c._serialize_args("f", (df,), {})
    assert calls["n"] == 2, "objects without lineage must not be memoised"


def test_two_objects_sharing_a_lineage_string_still_track_content():
    """The invariant (test_arg_hash_restart_stable): two DIFFERENT objects with
    an identical (possibly stale) lineage attr must key by CONTENT, not collide.
    The memo keys on id(), so distinct live objects never share a memo slot."""
    c = _cash()
    df_a = _df([1, 2, 3], lineage="same")
    df_b = _df([1, 2, 99], lineage="same")   # different content, same lineage str
    k_a = c._serialize_args("f", (df_a,), {})
    k_b = c._serialize_args("f", (df_b,), {})
    assert k_a != k_b


def test_memo_is_bounded():
    c = _cash()
    keep = []  # hold references so each df keeps a distinct id (forces growth)
    for i in range(c._ARG_HASH_MEMO_CAP + 50):
        df = _df(range(3), lineage=f"L{i}")
        keep.append(df)
        c._serialize_args("f", (df,), {})
    assert len(c._arg_hash_memo) <= c._ARG_HASH_MEMO_CAP
