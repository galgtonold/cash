"""A value-keyed output past the hash budget is new each time.

A list of 8 million tuples drawn by ``t = next(G)`` was hashed in full on
every run (8 s): past the budget of a change check, the value is taken as
new, with a token of its own, and what reads it runs again.
"""

from __future__ import annotations


def test_a_value_past_the_hash_budget_is_new_each_time(monkeypatch):
    from cash.notebook import lineage_formula

    monkeypatch.setattr(lineage_formula, "budgeted_hash", lambda value: None)
    value = [(1, 2)] * 10
    first = lineage_formula.no_cache_value_digest(value)
    assert first.startswith("fresh:")
    assert lineage_formula.no_cache_value_digest(value) != first
