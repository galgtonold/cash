"""A call's result that nothing but the caller holds, and that holds no
other object with an identity of its own (``pd.concat(frames)``: numbers and
text), cannot be held by any argument: the arguments are not read for it. A
list of tens of thousands of frames was walked frame by frame after every
concat. A result that is a view, or that something else holds, is still
looked for in them."""

from __future__ import annotations

import pytest

from cash.notebook import shared_objects
from cash.notebook.shared_objects import holds_part_of

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")


def _reads(monkeypatch) -> list:
    read: list = []
    identities = shared_objects._identities

    def counted(value, value_types):
        read.append(value)
        return identities(value, value_types)

    monkeypatch.setattr(shared_objects, "_identities", counted)
    return read


def test_a_fresh_frame_is_not_looked_for_in_the_frames_it_was_built_from(monkeypatch):
    frames = [pd.DataFrame({"t": ["a", "b"], "v": [i, i]}) for i in range(50)]
    result = pd.concat(frames)
    read = _reads(monkeypatch)
    assert holds_part_of(result, (frames,), value_held=False) is False
    assert read == [result]
    # The same answer, read the long way.
    read.clear()
    assert holds_part_of(result, (frames,)) is False
    assert len(read) == 2


def test_a_held_result_is_still_found_in_its_arguments():
    frame = pd.DataFrame({"v": [1, 2]})
    assert holds_part_of(frame, ([frame],)) is True


def test_a_view_is_looked_for_though_nothing_else_holds_it(monkeypatch):
    base = np.arange(10.0)
    view = base[2:5]
    read = _reads(monkeypatch)
    assert holds_part_of(view, (base,), value_held=False) is True
    assert len(read) == 2
