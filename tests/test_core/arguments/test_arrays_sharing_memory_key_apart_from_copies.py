"""Arrays and frames that share memory key apart from equal copies.

An array or frame is keyed by its content, so a view beside its base,
``f(base, base[2:])``, keyed like the base beside an equal copy of the
view, and ``f(df, df['a'])`` like the frame beside a copied column: the
call with the copies was served the shared call's result, though a write
through one shows through the other and ``np.shares_memory`` tells them
apart. Where such objects overlap, the key now holds where each sits in
the memory they share, from its lowest address, so the same layout keys
the same in another process. Nothing overlapping, the key is unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cash import Cash
from cash.decorator.arg_hashing import shared_memory_across

pytestmark = pytest.mark.filterwarnings("ignore::cash.exceptions.CashWarning")

BASE = np.arange(10.0)
FRAME = pd.DataFrame({"a": np.arange(3.0), "b": np.arange(3.0)})


def _shares(a, b):
    return bool(np.shares_memory(a, b))


CASES = {
    "a view and its base": (lambda: (BASE, BASE[2:]), lambda: (BASE, BASE[2:].copy()), _shares),
    "a strided view": (lambda: (BASE, BASE[::3]), lambda: (BASE, BASE[::3].copy()), _shares),
    "a frame and its column": (
        lambda: (FRAME, FRAME["a"]),
        lambda: (FRAME, FRAME["a"].copy()),
        lambda a, b: _shares(a["a"].to_numpy(), b.to_numpy()),
    ),
    "a frame and its column's array": (
        lambda: (FRAME, FRAME["a"].to_numpy()),
        lambda: (FRAME, FRAME["a"].to_numpy().copy()),
        lambda a, b: _shares(a["a"].to_numpy(), b),
    ),
    "a view in a list": (
        lambda: ([BASE[2:]], BASE),
        lambda: ([BASE[2:].copy()], BASE),
        lambda a, b: _shares(a[0], b),
    ),
    "both in one list": (
        lambda: ([BASE, BASE[2:]], 0),
        lambda: ([BASE, BASE[2:].copy()], 0),
        lambda a, b: _shares(a[0], a[1]),
    ),
}


def _cash(tmp_path, backend):
    if backend == "memory":
        return Cash(backend="memory", register_magic=False)
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


@pytest.mark.parametrize("backend", ["disk", "memory"])
@pytest.mark.parametrize("case", list(CASES.values()), ids=list(CASES))
def test_the_call_with_a_copy_is_not_served_the_shared_ones_result(tmp_path, backend, case):
    shared, apart, test = case
    runs = []

    def body(a, b):
        runs.append(1)
        return test(a, b)

    f = _cash(tmp_path, backend).cache(body)
    got = [f(*shared()) for _ in range(3)] + [f(*apart()), f(*shared())]
    assert got == [True, True, True, False, True]
    assert len(runs) == 2


def test_the_shared_call_is_reused_after_a_restart(tmp_path):
    runs = []

    def view_and_base(a, b):
        runs.append(1)
        return _shares(a, b)

    for _ in range(2):
        f = _cash(tmp_path, "disk").cache(view_and_base)
        assert f(BASE, BASE[2:]) is True
        assert f(BASE, BASE[2:].copy()) is False
    assert len(runs) == 2


@pytest.mark.parametrize(
    "values",
    [
        (BASE, np.arange(10.0)),
        (BASE, BASE),
        (FRAME, FRAME.copy()),
        (FRAME, FRAME["a"].copy()),
        (FRAME, pd.Index([1, 2, 3])),
    ],
    ids=[
        "two arrays",
        "one array twice",
        "a frame and its copy",
        "a frame and a copied column",
        "a frame and an index",
    ],
)
def test_nothing_is_added_to_the_key_when_no_memory_is_shared(values):
    """One object twice is marked as one object (`shared_across`); a copied
    column keeps its frame's index, which is never written in place."""
    keyed = [object() for _ in values]
    assert shared_memory_across(list(values), keyed, []) == ()
