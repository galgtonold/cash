"""The identity walk stops as soon as what it has queued for the next level
is more than its budget has left: each of those objects costs the budget
once when opened, so the walk is cut all the same. A list of tens of
thousands of small frames was opened frame by frame, every text column
read, only to pass the budget on their parts a level down.

Pinned by what the walk opens, not by time: the same verdict, and fewer
frames read."""

from __future__ import annotations

import types

import pytest

from cash.notebook.statement import identity_holders
from cash.notebook.statement.identity_holders import forget_flat_reads, identity_held_reason

pd = pytest.importorskip("pandas")


def _frames(count: int) -> list:
    return [
        pd.DataFrame({"name": [f"user{i}"] * 3, "note": pd.Series(["a", "b", "c"], dtype=object), "v": [1, 2, 3]})
        for i in range(count)
    ]


def _verdict_and_frames_read(monkeypatch, value: object, ahead: bool) -> tuple[str | None, int]:
    read: list = []
    parts = identity_holders.library_parts

    def counted(item):
        if isinstance(item, pd.DataFrame):
            read.append(item)
        return parts(item)

    monkeypatch.setattr(identity_holders, "library_parts", counted)
    if not ahead:
        monkeypatch.setattr(identity_holders._Walk, "_count_ahead", lambda self, into, start, queued: False)
    forget_flat_reads()
    try:
        return identity_held_reason("frames", value, {"frames": value}), len(read)
    finally:
        monkeypatch.undo()


def test_a_list_of_frames_past_the_budget_gives_up_before_reading_them_all(monkeypatch):
    monkeypatch.setattr(identity_holders, "_MAX_SEEN", 2_000)
    frames = _frames(800)

    walked, walked_read = _verdict_and_frames_read(monkeypatch, frames, ahead=False)
    monkeypatch.setattr(identity_holders, "_MAX_SEEN", 2_000)
    counted, counted_read = _verdict_and_frames_read(monkeypatch, frames, ahead=True)

    assert walked is not None and "more objects than cash looks through" in walked
    assert counted == walked
    assert walked_read == 800
    # Each frame queues its labels and columns: past the budget well
    # before the last frame is read.
    assert counted_read < 500


def test_a_list_of_frames_within_the_budget_is_read_whole(monkeypatch):
    monkeypatch.setattr(identity_holders, "_MAX_SEEN", 2_000)
    frames = _frames(100)
    verdict, read = _verdict_and_frames_read(monkeypatch, frames, ahead=True)
    assert verdict is None
    assert read == 100


def test_an_object_queued_and_opened_on_the_same_level_counts_once(monkeypatch):
    # ``[box, *items]`` with ``box = list(items)``: opening ``box`` queues
    # the items, which this level then opens itself. Each costs the budget
    # once, and the walk fits it exactly: not cut.
    items = [types.SimpleNamespace() for _ in range(50)]
    value = [list(items), *items]
    monkeypatch.setattr(identity_holders, "_MAX_SEEN", 1 + 1 + len(items))
    forget_flat_reads()
    assert identity_held_reason("value", value, {"value": value}) is None

    monkeypatch.setattr(identity_holders, "_MAX_SEEN", len(items))
    assert "more objects than cash looks through" in identity_held_reason("value", value, {"value": value})


def test_an_id_kept_in_a_frame_is_still_found_below_the_budget(monkeypatch):
    target = object()
    frames = _frames(30)
    frames[17]["v"] = [id(target)] * 3
    ns = {"frames": frames, "target": target}
    forget_flat_reads()
    found = identity_held_reason("frames", frames, ns)
    assert found is not None and found.kind == "an id()"
