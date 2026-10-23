"""Frames whose object columns hold the records a list holds are one group
with the list, however many frames there are.

``df2 = df[mask]`` with ``recs`` holding ``df``'s records, and ``flat =
pd.json_normalize(recs)`` holding their ``tags`` lists while ``df`` holds the
records: each was refused as shared. A frame found holding the objects in its
cells was linked but did not join, and once two frames were read together the
lists that read keeps were one reference per record no count accounted for.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cash.notebook.shared_objects import share_group

N = 200


def _records():
    return [{"id": i, "tags": ["a", "b"][: i % 3], "meta": {"k": i % 7}} for i in range(N)]


def _namespace(kind):
    recs = _records()
    df = pd.DataFrame({"id": range(N), "rec": recs})
    ns = {"recs": recs, "df": df}
    if kind == "filtered":
        ns["out"] = df[df["id"] >= 0]
    else:
        ns["out"] = pd.json_normalize(recs)
    return ns


@pytest.mark.parametrize("kind", ["filtered", "normalized"])
def test_the_frames_and_the_records_join(kind):
    ns = _namespace(kind)
    holders, shared = share_group(["out"], {"out": ns["out"]}, ns)
    assert shared == set()
    assert set(holders) == {"df", "recs"}
