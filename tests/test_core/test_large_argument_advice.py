"""The two warnings a large or mutated argument produces point at the right fix.

Round 18:

* IMPURE-SIDE-EFFECTS for a cached step that changed its ARGUMENT in place
  called it a "subscript mutation" and offered `assume_safe` -- which hides the
  real consequence: a hit returns the stored result and does not repeat the
  change, so the caller's object differs between a hit and a miss (r18s5, 31
  of 64 values wrong downstream). It now says so, and leads with "return a
  copy".
* CACHE-NET-LOSS said "a large argument is being hashed in full" without saying
  which. It now names the costliest parameter, and when a cached function
  produced it, suggests `frozen=True` there.
"""
from __future__ import annotations

import warnings

import pytest

from cash import Cash
from cash.effectiveness import EffectivenessLedger

pytestmark = pytest.mark.core


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def test_changing_an_argument_in_place_says_what_a_hit_does(c):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")

        @c.cache
        def winsorize(feats):
            feats["x"] = 0
            return len(feats)

        winsorize({"x": 5})
    text = " ".join(str(w.message) for w in rec if "IMPURE-SIDE-EFFECTS" in str(w.message))
    assert "changes the argument 'feats' in place" in text, text
    assert "a cache hit would not make that change, so a call that makes it is not stored" in text
    assert "Fix: for a line that changes an argument in place, return a modified copy" in text


def test_a_global_mutation_keeps_the_old_wording(c):
    """Control: only a PARAMETER is described as the caller's object."""
    registry = {}

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")

        @c.cache
        def register(k):
            registry[k] = 1
            return k

        register(1)
    text = " ".join(str(w.message) for w in rec if "IMPURE-SIDE-EFFECTS" in str(w.message))
    assert "subscript mutation" in text
    assert "changes the argument" not in text


class Model:
    def __init__(self, n):
        self.weights = [float(i) for i in range(n)]


def test_net_loss_names_the_argument_and_suggests_frozen_on_its_producer(c):
    c._effectiveness = EffectivenessLedger(waste_threshold_seconds=0.0)

    @c.cache
    def train(n):
        return Model(n)

    @c.cache
    def first_weight(model):
        return model.weights[0]

    model = train(200_000)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for _ in range(5):
            first_weight(model)
    found = [str(w.message) for w in rec if "CACHE-NET-LOSS" in str(w.message)]
    assert found, [str(w.message) for w in rec]
    assert "The costliest argument is 'model' (Model)" in found[0]
    assert "declare @cash.cache(frozen=True) on" in found[0]
    assert "train" in found[0]


def test_net_loss_without_a_producer_keeps_the_hasher_advice():
    """A culprit nothing cached produced: the register_hasher advice stands."""
    ledger = EffectivenessLedger(waste_threshold_seconds=0.0)
    verdict = None
    for _ in range(5):
        verdict = verdict or ledger.record(
            "f", overhead_seconds=0.5, body_seconds=0.001, was_hit=True,
            culprit=("grid", "ndarray", 0.49, None, False))
    what, fix = verdict
    assert "'grid' (ndarray)" in what
    assert fix.startswith("register a cheaper hasher")
    assert "frozen" not in fix
