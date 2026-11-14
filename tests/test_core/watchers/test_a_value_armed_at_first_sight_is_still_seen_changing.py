"""A value armed the first time it is hashed (`value_watch.arm_at_first_sight`),
or given a digest its caller took (`value_watch.adopt`), is handed back
unread only while it is unchanged.

A loop's candidate is fingerprinted before the loop and after it; armed at
its second hash, the first two hashes each read it whole. Armed at its
first, the second is handed back. A call's result digested by its store is
kept with that digest, so the next statement's first hash of it is handed
back too. Either way a change in place is seen, and the digest read again
is the fresh one.
"""

from __future__ import annotations

from cash import value_watch
from cash.value_hash import _compute_hash


def _rows(n: int = 200) -> list[dict]:
    return [{"name": f"a{i % 8}", "at": i, "tags": ["x", "y"]} for i in range(n)]


def test_a_value_is_handed_back_from_its_second_hash(watching, counted):
    rows = watching["rows"] = _rows()
    with value_watch.arm_at_first_sight(rows):
        first = value_watch.digest(rows, counted)
    assert value_watch.digest(rows, counted) == first
    assert counted.calls == 1


def test_without_the_block_it_is_armed_at_its_second_hash(watching, counted):
    rows = watching["rows"] = _rows()
    value_watch.digest(rows, counted)
    value_watch.digest(rows, counted)
    value_watch.digest(rows, counted)
    assert counted.calls == 2


def test_only_the_value_named_is_armed_early(watching, counted):
    rows = watching["rows"] = _rows()
    other = watching["other"] = _rows(150)
    with value_watch.arm_at_first_sight(rows):
        value_watch.digest(other, counted)
    value_watch.digest(other, counted)
    assert counted.calls == 2


def test_a_change_after_an_early_arm_is_read_again(watching, counted):
    rows = watching["rows"] = _rows()
    with value_watch.arm_at_first_sight(rows):
        before = value_watch.digest(rows, counted)
    rows[57]["tags"].append("z")
    after = value_watch.digest(rows, counted)
    assert after != before
    assert after == _compute_hash(rows)
    assert counted.calls == 2


def test_an_adopted_digest_is_handed_back_until_the_value_changes(watching, counted):
    rows = watching["rows"] = _rows()
    assert value_watch.adopt(rows, _compute_hash(rows))
    assert value_watch.digest(rows, counted) == _compute_hash(rows)
    assert counted.calls == 0
    rows[3]["at"] = -1
    assert value_watch.digest(rows, counted) == _compute_hash(rows)
    assert counted.calls == 1


def test_a_value_too_small_to_watch_is_not_adopted(watching, counted):
    value_watch._cw.configure(min_bytes=1 << 30)
    rows = watching["rows"] = _rows()
    assert not value_watch.adopt(rows, "not its digest")
    assert value_watch.digest(rows, counted) == _compute_hash(rows)
    assert counted.calls == 1
