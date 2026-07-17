"""CAS-202: the decorator arg-hash must key a content-bearing argument on its
CONTENT, not on the notebook's in-memory ``_cash_lineage_hash``.

Under ``%cash_on`` a notebook variable (e.g. ``X_train``) carries a
``_cash_lineage_hash`` attribute. That lineage hash is recomputed every kernel
session and is NOT reproducible across a restart -- three separate wheel-venv
sessions produced three different lineage hashes for a byte-identical
``X_train`` while its content hash was identical every time. The decorator's
``_hash_arg_payload`` used to read ``arg._cash_lineage_hash`` before any content
hash, so after a kernel restart ``train_model(X_train, ...)`` computed a
DIFFERENT cache key than the one it had stored (which DID persist to disk),
missed with ``no_entry``, and re-trained the model the docs promise survives a
restart.

The fix: content-authoritative builtin hashers (pandas / numpy / polars / ...)
win over ``_cash_lineage_hash``. These tests pin both directions -- a volatile
lineage attr must NOT move the key, and a real content change MUST move it --
and are proven fails-without / passes-with by stashing only ``src/cash/core.py``.
"""
from __future__ import annotations

import pytest

from cash import Cash, InMemoryBackend


def _cash() -> Cash:
    return Cash(backend=InMemoryBackend())


def _set_lineage(obj, value: str) -> None:
    """Attach a notebook-style lineage hash without pandas' set-attr warning."""
    object.__setattr__(obj, "_cash_lineage_hash", value)


# ---------------------------------------------------------------------------
# pandas DataFrame (the reported ML case)
# ---------------------------------------------------------------------------

def test_dataframe_arg_hash_ignores_volatile_lineage():
    """A byte-identical DataFrame keys the same even when its lineage attr
    changes (== a kernel restart re-deriving the lineage)."""
    pd = pytest.importorskip("pandas")
    c = _cash()

    df_session1 = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    df_session2 = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})  # identical content
    _set_lineage(df_session1, "lineage_from_cold_session")
    _set_lineage(df_session2, "lineage_from_warm_session_totally_different")

    h1 = c._hash_arg_payload((df_session1, 120), {})
    h2 = c._hash_arg_payload((df_session2, 120), {})
    assert h1 == h2, "content-identical DataFrame must key stably across restart"


def test_dataframe_arg_hash_still_tracks_content_change():
    """Guard against over-fixing: different content MUST change the key, even
    when the (stale) lineage attr is identical."""
    pd = pytest.importorskip("pandas")
    c = _cash()

    df_a = pd.DataFrame({"a": [1, 2, 3]})
    df_b = pd.DataFrame({"a": [1, 2, 99]})  # different content
    _set_lineage(df_a, "same_lineage")
    _set_lineage(df_b, "same_lineage")

    assert c._hash_arg_payload((df_a,), {}) != c._hash_arg_payload((df_b,), {})


# ---------------------------------------------------------------------------
# pandas Series (the pipeline's y_train arg)
# ---------------------------------------------------------------------------

def test_series_arg_hash_ignores_volatile_lineage():
    pd = pytest.importorskip("pandas")
    c = _cash()

    s1 = pd.Series([0, 1, 1, 0], name="churned")
    s2 = pd.Series([0, 1, 1, 0], name="churned")
    _set_lineage(s1, "cold")
    _set_lineage(s2, "warm-different")

    assert c._hash_arg_payload((s1,), {}) == c._hash_arg_payload((s2,), {})


def test_series_arg_hash_still_tracks_content_change():
    pd = pytest.importorskip("pandas")
    c = _cash()

    s1 = pd.Series([0, 1, 1, 0], name="churned")
    s2 = pd.Series([0, 1, 1, 1], name="churned")
    _set_lineage(s1, "x")
    _set_lineage(s2, "x")

    assert c._hash_arg_payload((s1,), {}) != c._hash_arg_payload((s2,), {})


# ---------------------------------------------------------------------------
# end-to-end: a persisted entry restores under a fresh Cash after the arg's
# lineage attr is re-derived (simulating a kernel restart).
# ---------------------------------------------------------------------------

def test_decorator_restart_survives_relineaged_dataframe(tmp_path):
    pd = pytest.importorskip("pandas")
    from cash import FileBackend

    cache_dir = str(tmp_path / "cache")
    calls = {"n": 0}

    def build_cash():
        c = Cash(backend=FileBackend(cache_dir=cache_dir))

        @c.cache
        def train(df, k):
            calls["n"] += 1
            return int(df["a"].sum()) + k

        return c, train

    # cold "session": DataFrame with one lineage hash
    df_cold = pd.DataFrame({"a": [1, 2, 3]})
    _set_lineage(df_cold, "cold_lineage_hash")
    _c1, train1 = build_cash()
    assert train1(df_cold, 10) == 16
    assert calls["n"] == 1

    # warm "session": fresh Cash (== restart), identical content, DIFFERENT
    # lineage hash. Must restore from disk rather than recompute.
    df_warm = pd.DataFrame({"a": [1, 2, 3]})
    _set_lineage(df_warm, "warm_lineage_hash_completely_different")
    _c2, train2 = build_cash()
    assert train2(df_warm, 10) == 16
    assert calls["n"] == 1, "restart re-ran the body: arg-hash was not restart-stable"


# ---------------------------------------------------------------------------
# regression guard: a lineage-only custom object (no content hasher) still
# short-circuits on lineage (test_lineage_custom_object.py contract).
# ---------------------------------------------------------------------------

def test_custom_object_without_content_hasher_still_uses_lineage():
    c = _cash()

    class Blob:
        def __init__(self, payload):
            self.payload = payload

    b1 = Blob([1, 2, 3])
    b2 = Blob([9, 9, 9])  # different payload, but forced-equal lineage
    _set_lineage(b1, "shared")
    _set_lineage(b2, "shared")

    # No builtin/registered hasher for Blob -> lineage is authoritative, so the
    # two share a key (the notebook lineage short-circuit is preserved).
    assert c._hash_arg_payload((b1,), {}) == c._hash_arg_payload((b2,), {})
