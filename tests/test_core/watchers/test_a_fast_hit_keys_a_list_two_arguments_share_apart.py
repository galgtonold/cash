"""A fast hit tells a list two arguments share from two equal lists.

The memo key holds a small list by value, so ``same(a, a)`` and
``same(a, [1, 2, 3])`` had one key there, and after eight calls of the first
the second was served True on a fast hit; the ordinary key marks the shared
list and keys them apart. A list or dict met twice now makes no memo key:
the call takes the ordinary path.
"""

from __future__ import annotations

import pytest

from cash import Cash

pytestmark = pytest.mark.filterwarnings("ignore::cash.exceptions.CashWarning")


@pytest.mark.parametrize("backend", ["disk", "memory"])
@pytest.mark.parametrize(
    "shared, apart",
    [
        (lambda a: (a, a), lambda a: (a, [1, 2, 3])),
        (lambda a: (a, a), lambda a: ([1, 2, 3], a)),
        (lambda a: ([a, a],), lambda a: ([a, [1, 2, 3]],)),
        (lambda a: ({"k": a, "j": a},), lambda a: ({"k": a, "j": [1, 2, 3]},)),
    ],
    ids=["two arguments", "two arguments, the copy first", "inside one list", "inside one dict"],
)
def test_the_call_with_a_copy_is_not_served_the_shared_ones_result(tmp_path, backend, shared, apart):
    if backend == "memory":
        cash = Cash(backend="memory", register_magic=False)
    else:
        cash = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)

    def same(*args):
        flat = [
            x
            for arg in args
            for x in (arg.values() if type(arg) is dict else arg if type(arg) is list and len(args) == 1 else [arg])
        ]
        return flat[0] is flat[1]

    f = cash.cache(same)
    a = [1, 2, 3]
    for _ in range(8):
        assert f(*shared(a)) is True
    assert f(*apart(a)) is False
    for _ in range(8):
        assert f(*apart(a)) is False
    assert f(*shared(a)) is True
