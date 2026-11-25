"""A pandas value that refuses duplicate labels keys apart from one that
allows them.

``df.set_flags(allows_duplicate_labels=False)`` makes pandas raise where an
operation would give the frame a duplicate label, and code reads the flag
(``df.flags.allows_duplicate_labels``). It was not in the key, so the
strict frame was served the answer computed for the plain one.
"""

from __future__ import annotations

import pytest

from cash.content_hashers import hash_pandas
from cash.value_hash import compute_hash

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

pytestmark = pytest.mark.core


@pytest.mark.parametrize(
    "make", [lambda: pd.DataFrame({"a": [1, 2]}), lambda: pd.Series([1, 2])], ids=["frame", "series"]
)
def test_the_flag_is_in_the_hash(make):
    plain, strict = make(), make().set_flags(allows_duplicate_labels=False)
    assert hash_pandas(plain) != hash_pandas(strict)
    assert compute_hash(plain) != compute_hash(strict)  # the notebook's value hash
    assert hash_pandas(plain) == hash_pandas(make())
    assert hash_pandas(strict) == hash_pandas(make().set_flags(allows_duplicate_labels=False))


def test_a_function_reading_the_flag_answers_for_its_argument(disk_cash):
    @disk_cash.cache
    def may_duplicate(df):
        return df.flags.allows_duplicate_labels

    assert may_duplicate(pd.DataFrame({"a": [1, 2]})) is True
    assert may_duplicate(pd.DataFrame({"a": [1, 2]}).set_flags(allows_duplicate_labels=False)) is False


def test_the_flag_set_in_place_on_a_big_frame_is_seen(disk_cash):
    """Big enough for the frame's digest to be remembered between calls."""

    @disk_cash.cache
    def may_duplicate(df):
        return df.flags.allows_duplicate_labels

    table = pd.DataFrame(np.zeros((70_000, 16)))
    assert [may_duplicate(table), may_duplicate(table)] == [True, True]
    table.flags.allows_duplicate_labels = False
    assert may_duplicate(table) is False
