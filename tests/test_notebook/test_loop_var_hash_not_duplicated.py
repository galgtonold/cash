"""A loop target is fully hashed once per iteration, not twice.

`compute_hash_full` on a loop target is deliberate and load-bearing: a sampled
hash once collided two iterations onto one cache entry and produced a wrong
result on the first run, so `for_handler._process_one_iteration` pays the full
hash for every binding (`call_unit.py`'s module docstring records the incident
and the measured costs -- 19ms for a 200k-row DataFrame against a 3ms cost
floor).

Paying it TWICE is not deliberate. `build_iteration_context` independently
computed the same full hash for the same value, unaware that
`_process_one_iteration` had just computed and stored it in `loop_var_digests`
five lines earlier. Measured on the demo tour's bootstrap cell -- five ~200k-row
groups -- a cached re-run spent 328ms in `compute_hash_full`, of which 164ms was
this duplicate.

Counted, not timed: the defect is "the same value is hashed twice", which is
exact. A wall-clock threshold would be the flakiest possible way to assert it.
"""
from __future__ import annotations

import tempfile

import pytest

pytest.importorskip("IPython")
np = pytest.importorskip("numpy")

import cash.notebook.object_hashing as object_hashing
from cash import Cash
from cash.notebook.ipython.magics import CashMagics
from tests.conftest import MockShell

ITERATIONS = 4

CELL = (
    "seen = []\n"
    "for arr in arrays:\n"
    "    seen.append(float(arr.sum()))\n"
)


@pytest.fixture
def counting_magics(monkeypatch):
    """Magics whose full-content hashes are counted per object."""
    counts: dict[int, int] = {}
    real = object_hashing.compute_hash_full

    def counting_compute_hash_full(obj):
        counts[id(obj)] = counts.get(id(obj), 0) + 1
        return real(obj)

    # Both call sites import this function-locally, at call time, so patching
    # the module attribute reaches them.
    monkeypatch.setattr(
        object_hashing, "compute_hash_full", counting_compute_hash_full
    )

    shell = MockShell()
    cash = Cash(cache_dir=tempfile.mkdtemp(), register_magic=False)
    magics = CashMagics(shell, cash)
    magics.cash_on("")
    magics._badge_mode = "off"
    shell.user_ns["arrays"] = [
        np.arange(1000, dtype=float) + i for i in range(ITERATIONS)
    ]
    return magics, counts


def test_a_loop_target_is_not_hashed_twice_per_iteration(counting_magics):
    """The duplicate. `build_iteration_context` must reuse the digest
    `_process_one_iteration` already computed for the same value."""
    magics, counts = counting_magics
    magics.cash("", CELL)

    duplicated = {obj_id: n for obj_id, n in counts.items() if n > 1}
    assert not duplicated, (
        f"{len(duplicated)} loop-target value(s) were fully hashed more than "
        f"once in a single pass: counts={sorted(duplicated.values())}. "
        "`build_iteration_context` is recomputing a digest `loop_var_digests` "
        "already holds."
    )


def test_the_loop_target_is_still_hashed_at_all(counting_magics):
    """The control.

    Sharing the digest must not become skipping it. A full hash per loop
    target is what keeps two iterations over large values that agree in a
    sample from collapsing onto one cache entry -- the wrong-result bug the
    full hash exists to prevent. A fix that simply stopped hashing would pass
    the test above and reintroduce it.
    """
    magics, counts = counting_magics
    magics.cash("", CELL)

    assert len(counts) >= ITERATIONS, (
        f"expected at least one full hash per iteration ({ITERATIONS}), "
        f"got {len(counts)} distinct values hashed -- the discriminator is gone"
    )


def test_iterations_still_produce_distinct_results(counting_magics):
    """Behavioural backstop: whatever the hashing does, the loop's own answers
    must stay per-iteration correct."""
    magics, counts = counting_magics
    magics.cash("", CELL)

    expected = [float((np.arange(1000, dtype=float) + i).sum()) for i in range(ITERATIONS)]
    assert magics.shell.user_ns["seen"] == expected
