"""A RAM-tier hit must hand back something the caller cannot use to poison it.

The backend copies on the way in and on the way out. `_safe_deep_copy` gained
a fast path -- a container of immutable scalars is copied shallowly, and a
tuple of them is shared outright -- because ``deepcopy`` pays a recursive call
per element and made restoring a large list slower than recomputing it.

That fast path is only sound while the ISOLATION property holds, so these
tests assert the property directly rather than the optimization. If the fast
path is ever widened to a container whose elements are mutable, these fail.
"""

import tempfile

import pytest

from cash import Cash
from cash.backends.memory_backend import InMemoryBackend


def test_returned_list_is_independent():
    stored = [1, 2, 3]
    handed_back = InMemoryBackend._safe_deep_copy(stored)
    handed_back.append(4)
    assert stored == [1, 2, 3], "mutating the result reached the stored entry"


def test_nested_container_is_still_deeply_isolated():
    """The fast path must NOT apply: the inner lists are mutable."""
    stored = [[1], [2]]
    handed_back = InMemoryBackend._safe_deep_copy(stored)
    handed_back[0].append(99)
    assert stored == [[1], [2]], "inner mutation reached the stored entry"


def test_dict_values_are_still_deeply_isolated():
    stored = {"a": [1]}
    handed_back = InMemoryBackend._safe_deep_copy(stored)
    handed_back["a"].append(2)
    assert stored == {"a": [1]}


def test_tuple_of_scalars_may_be_shared():
    """Sharing is safe only because neither the tuple nor its items can change."""
    stored = (1, "a", None)
    assert InMemoryBackend._safe_deep_copy(stored) == stored


def test_subclass_elements_are_not_shared():
    """A str SUBCLASS can carry a mutable ``__dict__``, so it is not a scalar."""

    class Tagged(str):
        pass

    item = Tagged("x")
    item.note = []  # the mutable state an exact-type check exists to protect
    stored = [item]
    handed_back = InMemoryBackend._safe_deep_copy(stored)
    handed_back[0].note.append(1)
    assert item.note == [], "a subclass element was shared, exposing its state"


def test_mutating_a_cached_result_does_not_poison_the_next_hit():
    """End to end: the property a user would actually notice."""
    cash = Cash(cache_dir=tempfile.mkdtemp(), register_magic=False)

    @cash.cache
    def build(n):
        return list(range(n))

    first = build(5)
    first.append(999)
    second = build(5)
    assert second == [0, 1, 2, 3, 4], f"cache was poisoned: {second}"


@pytest.mark.parametrize(
    "value",
    [
        [1, 2, 3],
        [1.5, 2.5],
        ["a", "b"],
        [True, False, None],
        [b"x", b"y"],
        [1, "a", None, 2.5],
        [],
        (1, 2, 3),
        (),
    ],
)
def test_scalar_containers_round_trip_equal(value):
    assert InMemoryBackend._safe_deep_copy(value) == value
