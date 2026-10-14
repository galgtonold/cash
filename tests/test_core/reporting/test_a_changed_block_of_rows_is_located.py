"""A block fingerprint of a value made of rows places what changed since the
last one: rows appended, a block rewritten in the middle, everything new
(``cash.row_blocks``, behind ``CACHE-PER-ITEM-RERUN``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cash.row_blocks import (
    MAX_BLOCKS,
    HashBudgetSpent,
    RowBlocks,
    compare,
    describe_ranges,
    fingerprint,
    row_count,
)


def _change(before, after):
    old, _ = fingerprint(before)
    new, cut = fingerprint(after, like=old)
    return compare(old, new, cut)


def _images(n, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (n, 8, 8), dtype=np.uint8)


def test_appended_rows_are_the_only_ones_changed():
    imgs = _images(3000)
    grown = np.concatenate([imgs, _images(400, seed=1)])

    change = _change(imgs, grown)

    assert change.rows == 3400
    assert change.changed == ((3000, 3400),), "the old last, partial block still matches up to row 2999"
    assert change.changed_rows == 400
    assert describe_ranges(change.changed) == "rows 3000-3399"


def test_a_block_rewritten_in_the_middle_is_placed():
    imgs = _images(3400)
    redone = imgs.copy()
    redone[1000:1400] = _images(400, seed=2)

    change = _change(imgs, redone)

    assert change.changed_rows >= 400
    assert change.changed_rows < 3400 / 4, "only the blocks around the rewrite"
    ((start, stop),) = change.changed
    assert start <= 1000 and stop >= 1400
    assert change.unchanged_share > 0.8


def test_the_same_value_has_nothing_changed():
    imgs = _images(500)
    assert _change(imgs, imgs.copy()).changed == ()


def test_everything_changed_when_every_row_is_new():
    change = _change(_images(500), _images(500, seed=3))
    assert change.changed == ((0, 500),)
    assert change.unchanged_share == 0.0


def test_another_dtype_changes_every_block():
    imgs = _images(500)
    assert _change(imgs, imgs.astype(np.float32)).changed_rows == 500


def test_a_frame_that_grows_a_season_changes_only_that_season():
    df = pd.DataFrame({"season": np.repeat(np.arange(50), 300), "goals": np.arange(15000) % 7})
    grown = pd.concat([df, pd.DataFrame({"season": [50] * 300, "goals": np.arange(300) % 5})], ignore_index=True)

    change = _change(df, grown)

    assert change.changed == ((15000, 15300),)


def test_a_new_column_changes_every_row():
    df = pd.DataFrame({"a": np.arange(1000)})
    assert _change(df, df.assign(b=1)).changed_rows == 1000


def test_a_list_of_arrays_is_compared_item_by_item():
    items = list(_images(200))
    change = _change(items, items + list(_images(20, seed=4)))
    assert change.changed == ((200, 220),)


def test_what_is_not_made_of_rows_has_no_fingerprint():
    assert row_count(3) is None
    assert row_count(np.float64(1.0)) is None
    assert row_count(np.array(1.0)) is None
    assert row_count(np.array([object()] * 20)) is None, "object arrays are not hashed by block"
    assert row_count([1, 2, 3] * 10) is None, "a list counts only when it holds arrays"
    assert fingerprint(np.arange(5)) is None, "too few rows to cut"


def test_a_first_fingerprint_has_at_most_max_blocks():
    blocks, cut = fingerprint(_images(10_000))
    assert len(blocks.digests) <= MAX_BLOCKS and cut is None


def test_a_value_that_outgrew_the_grid_compares_nothing():
    old, _ = fingerprint(_images(100))
    new, cut = fingerprint(_images(100 * 10), like=old)
    assert compare(old, new, cut) is None


def test_the_fingerprint_survives_a_round_trip_through_a_dict():
    blocks, _ = fingerprint(_images(300))
    assert RowBlocks.from_dict(blocks.to_dict()) == blocks
    assert RowBlocks.from_dict({"kind": "ndarray"}) is None


def test_a_deadline_already_passed_stops_the_hashing():
    with pytest.raises(HashBudgetSpent):
        fingerprint(_images(1000), deadline=0.0)


def test_ranges_are_described_briefly():
    assert describe_ranges(((5, 6),)) == "row 5"
    assert describe_ranges(((0, 47), (2961, 3400))) == "rows 0-46, 2961-3399"
    assert describe_ranges(((0, 2), (4, 6), (8, 10), (12, 14))) == "rows 0-1, 4-5, 8-9, ..."
