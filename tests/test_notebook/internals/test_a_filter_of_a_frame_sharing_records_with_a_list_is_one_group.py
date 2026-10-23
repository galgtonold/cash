"""A filter of a frame whose records a list holds too is one group with the
frame and the list.

``df2 = df[mask]`` with ``recs`` holding ``df``'s records was refused as
shared: once the two frames were read together, the lists that read keeps
were one reference per record no count accounted for, and ``recs`` joining
left each record looking held from outside.

``flat = pd.json_normalize(recs)``, whose ``tags`` lists ``recs``' records
hold while ``df`` holds the records, is still refused: ``df`` holds them only
in its cells, and a restore could not put them back there.
"""

from __future__ import annotations

import pandas as pd

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


def test_the_frames_and_the_records_join():
    ns = _namespace("filtered")
    holders, shared = share_group(["out"], {"out": ns["out"]}, ns)
    assert shared == set()
    assert set(holders) == {"df", "recs"}


def test_a_frame_holding_them_only_in_its_cells_still_refuses():
    ns = _namespace("normalized")
    holders, shared = share_group(["out"], {"out": ns["out"]}, ns)
    assert holders == {} and shared == {"out"}
