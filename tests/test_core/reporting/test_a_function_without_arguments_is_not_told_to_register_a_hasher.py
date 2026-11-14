"""A cached function that takes no arguments and costs more than it saves
is told its stored result is slow to load, not that an argument is hashed.

With no argument there is no costliest one to name, and the net-loss
warning fell through to "a large argument is being hashed in full" and
advice to register a hasher for "that argument's type": advice a
zero-argument function cannot follow. Its key costs next to nothing, so
the whole loss is the lookup -- reading the stored result back.
"""

from __future__ import annotations

import warnings

from cash.core import Cash
from cash.effectiveness import EffectivenessLedger
from cash.exceptions import CashCacheIneffectiveWarning


def test_the_ledger_blames_the_load_for_a_function_without_arguments():
    ledger = EffectivenessLedger(waste_threshold_seconds=0.01)
    verdict = None
    for hit in (False, True, True, True):
        verdict = verdict or ledger.record(
            "word_dict_trie", overhead_seconds=0.2, body_seconds=0.01, was_hit=hit, no_arguments=True
        )
    assert verdict is not None
    what, fix = verdict
    assert "loading the stored result" in what, what
    assert "takes no arguments" in what, what
    assert "hashed in full" not in what, what
    assert "register_hasher" not in fix, fix


def test_a_function_with_arguments_and_no_culprit_keeps_the_hasher_advice():
    """Control: one that takes arguments, none measured, is advised as before."""
    ledger = EffectivenessLedger(waste_threshold_seconds=0.01)
    verdict = None
    for hit in (False, True, True, True):
        verdict = verdict or ledger.record("f", overhead_seconds=0.2, body_seconds=0.01, was_hit=hit)
    assert verdict is not None
    assert "register_hasher" in verdict[1], verdict


class SlowToLoad:
    """A result whose every read out of the store does work its build does not."""

    def __init__(self) -> None:
        self.n = 1

    def __setstate__(self, state):
        acc = 0
        for i in range(400_000):
            acc = (acc * 31 + i) & 0xFFFFFFFF
        self.__dict__.update(state)


def test_the_decorator_says_the_load_of_a_function_without_arguments(tmp_path):
    cash = Cash(cache_dir=str(tmp_path), register_magic=False)
    cash._calls.effectiveness = EffectivenessLedger(waste_threshold_seconds=0.01)
    cash.reconfigure(warning_detail="full")

    @cash.cache
    def load():
        return SlowToLoad()

    kept = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(12):
            kept.append(load())
    found = [
        w for w in caught if issubclass(w.category, CashCacheIneffectiveWarning) and "CACHE-NET-LOSS" in str(w.message)
    ]
    assert len(found) == 1, [str(w.message) for w in caught]
    text = str(found[0].message)
    # Which part of the lookup it blames (the load, or the store when the
    # first miss's write outweighs the hits so far) is the ledger's rule.
    assert "the function takes no arguments" in text, text
    assert "hashed in full" not in text, text
    assert "register_hasher" not in text, text
