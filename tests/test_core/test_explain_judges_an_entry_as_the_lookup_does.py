"""explain() and a real lookup judge a found entry by one rule.

A chunked iterator entry whose chunk is gone is not served: the call
misses with "entry incomplete". explain() must say the call would miss,
not report a hit for the manifest it found.
"""

from __future__ import annotations

from cash.decorator.explain import EXPLAIN_HIT, EXPLAIN_NO_ENTRY


def test_explain_reports_a_miss_for_an_iterator_entry_missing_a_chunk(cash_instance):
    @cash_instance.cache
    def gen(n):
        yield from range(n)

    assert list(gen(3)) == [0, 1, 2]
    assert gen.explain(3).reason == EXPLAIN_HIT

    backend = cash_instance.backend
    chunk_keys = [e["key"] for e in backend.list_entries() if (e.get("key") or "").endswith(":chunk_0")]
    assert chunk_keys, "the iterator was not stored in chunks"
    for key in chunk_keys:
        backend.delete(key)

    explanation = gen.explain(3)
    assert not explanation.would_hit
    assert explanation.reason == EXPLAIN_NO_ENTRY
    assert "incomplete" in explanation.details["why"]

    assert list(gen(3)) == [0, 1, 2]
    assert gen.cache_info()["miss_reasons"].get("entry incomplete") == 1
