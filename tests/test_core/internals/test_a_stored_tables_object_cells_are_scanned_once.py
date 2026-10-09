"""Storing a table with an object column scans that column once.

The RAM tier asks whether an object column can hold a changeable cell
(``_holds_mutable_cells``, an ``infer_dtype`` pass over every cell) and then,
to share the table, whether its cells cannot change (`frame_sharing`): the
same question again, asked up to three times more. At a million rows of
``datetime.date`` that was 75 ms of a 100 ms store.
"""

from __future__ import annotations

import datetime

import pytest

pd = pytest.importorskip("pandas")

from cash.backends import frame_sharing, memory_backend  # noqa: E402 - imported after the importorskip above
from cash.backends.memory_backend import InMemoryBackend  # noqa: E402 - imported after the importorskip above

pytestmark = pytest.mark.skipif(not frame_sharing.enabled(), reason="needs pandas copy-on-write")


def test_the_store_scans_each_object_column_once(monkeypatch):
    scans: list = []
    real_frame, real_cells = memory_backend._holds_mutable_cells, frame_sharing._cells_unchangeable
    monkeypatch.setattr(memory_backend, "_holds_mutable_cells", lambda f: scans.append("frame") or real_frame(f))
    monkeypatch.setattr(frame_sharing, "_cells_unchangeable", lambda a: scans.append("cells") or real_cells(a))
    day = datetime.date(2024, 1, 1)
    df = pd.DataFrame({"day": [day] * 1000, "n": range(1000)})
    backend = InMemoryBackend()
    backend.set("k", df, {"execution_time": 1.0})
    assert scans == ["frame"]
    hit = backend.get("k")[1]
    assert scans == ["frame"]
    assert hit["day"].iloc[0] == day
