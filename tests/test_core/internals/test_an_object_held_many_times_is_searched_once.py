"""An object held many times in one value is searched for a set once.

A column holding one bound method in each of 845,000 rows
(``df['day'] = df['action_time'].dt.day_name``, the method, not its result)
searched the method's accessor and the frame behind it once per row: 224 s
to hash the frame for one statement. The walk now remembers, for its own
length, the objects it left to pickle whole and the objects it found holding
no set, and the key is the same as before.
"""

from __future__ import annotations

import pytest

from cash import canonical_form
from cash.content_hashers import BUILTIN_CONTENT

pytestmark = [pytest.mark.core]


class Holder:
    def __init__(self, items):
        self.items = items

    def read(self):
        return self.items


@pytest.fixture
def searches(monkeypatch):
    count = {"n": 0}
    real = canonical_form.contains_set

    def counting(*args, **kwargs):
        count["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(canonical_form, "contains_set", counting)
    return count


def test_one_object_in_every_item_is_searched_once(searches):
    method = Holder(list(range(1000))).read
    canonical_form.canonical_bytes([method] * 2000, BUILTIN_CONTENT)
    assert searches["n"] == 1


def test_the_key_is_the_same_for_the_same_objects():
    method = Holder([1, 2]).read
    once = canonical_form.canonical_bytes([method] * 50, BUILTIN_CONTENT)
    assert canonical_form.canonical_bytes([method] * 50, BUILTIN_CONTENT) == once
    assert canonical_form.canonical_bytes([Holder([1, 2]).read] * 50, BUILTIN_CONTENT) == once
    assert canonical_form.canonical_bytes([Holder([1, 3]).read] * 50, BUILTIN_CONTENT) != once


def test_many_bound_methods_of_one_object_search_it_once():
    holder = Holder(list(range(1000)))
    clean: dict = {}
    assert not canonical_form.contains_set(holder.read, None, clean)
    known = len(clean)
    for _ in range(100):
        assert not canonical_form.contains_set(holder.read, None, clean)
    # Each new method object is looked at, but not the holder behind it again.
    assert len(clean) - known <= 100 * 4


def test_an_object_reaching_a_set_still_holds_one_after_a_clean_search():
    shared = Holder([1, 2])
    with_set = Holder([shared, {3, 4}])
    clean: dict = {}
    assert not canonical_form.contains_set(Holder([shared]), None, clean)
    assert canonical_form.contains_set(with_set, None, clean)


def test_a_cycle_searched_from_inside_is_not_called_clean():
    a = Holder(None)
    b = Holder(a)
    a.items = [b, {1}]
    clean: dict = {}
    assert canonical_form.contains_set(b, None, clean)
    assert canonical_form.contains_set(a, None, clean)


def test_a_frame_with_a_column_of_one_method_hashes_like_before():
    pd = pytest.importorskip("pandas")
    from cash.value_hash import compute_hash

    frame = pd.DataFrame({"t": pd.to_datetime(range(3000), unit="s")})
    frame["day"] = frame["t"].dt.day_name
    other = frame.copy()
    other["day"] = other["t"].dt.day_name
    assert compute_hash(frame) == compute_hash(frame)
    assert compute_hash(frame) != compute_hash(frame.iloc[:-1])
