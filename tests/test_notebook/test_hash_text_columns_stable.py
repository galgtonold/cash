"""A DataFrame with a text column hashed by memory address.

``compute_hash`` sampled ``obj.head(5).values.tobytes()``. For a frame with a
text column ``.values`` is an object array, whose bytes are pointers: the hash
changed in every process and even between a frame and its copy. The notebook
folds that hash into the lineage of anything an ``if`` or ``for`` body may
reassign, so after every restart such a frame looked new, and nothing
downstream of an untaken ``if FLAG:`` ever restored (4/4). An integer
frame was always stable, which is why it went unnoticed.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from cash.object_hashing import compute_hash

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

PROBE = textwrap.dedent("""
    import numpy as np, pandas as pd
    from cash.object_hashing import compute_hash
    rs = np.random.RandomState(0)
    frame = pd.DataFrame({"id": np.arange(1000), "region": rs.choice(["NA", "OCE", "EU"], 1000),
                          "amount": rs.rand(1000)})
    arr = np.array(["a", 1, None, 2.5], dtype=object)
    print(compute_hash(frame), compute_hash(frame["region"]), compute_hash(arr))
""")


def _frame():
    rs = np.random.RandomState(0)
    return pd.DataFrame(
        {"id": np.arange(1000), "region": rs.choice(["NA", "OCE", "EU"], 1000), "amount": rs.rand(1000)}
    )


def test_a_text_frame_hashes_the_same_in_every_process():
    runs = {
        subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, check=True).stdout.strip()
        for _ in range(2)
    }
    assert len(runs) == 1, runs


def test_a_copy_of_a_text_frame_hashes_like_the_frame():
    frame = _frame()
    assert compute_hash(frame) == compute_hash(frame.copy())
    assert compute_hash(frame["region"]) == compute_hash(frame["region"].copy())
    arr = np.array(["a", 1, None, 2.5], dtype=object)
    assert compute_hash(arr) == compute_hash(arr.copy())


def test_different_text_still_hashes_differently():
    """Control: stability must not come from ignoring the text."""
    frame = _frame()
    other = frame.copy()
    other.loc[0, "region"] = "XX"
    assert compute_hash(frame) != compute_hash(other)
    arr = np.array(["a", 1, None, 2.5], dtype=object)
    assert compute_hash(arr) != compute_hash(np.array(["b", 1, None, 2.5], dtype=object))


def test_a_frame_hashes_by_its_whole_content():
    """A frame is hashed as the decorator keys it (`builtin_hash`), every row."""
    from cash.object_hashing import builtin_hash

    frame = pd.DataFrame({"a": np.arange(10), "b": np.arange(10) * 0.5})
    assert compute_hash(frame) == builtin_hash(frame)


def test_a_collection_of_frames_is_hashed_frame_by_frame():
    """A dict holding frames is hashed item by item, each frame as it is
    hashed on its own, not pickled whole with the dict."""
    from cash.object_hashing import builtin_hash

    big = pd.DataFrame({"a": np.arange(1000, dtype=float), "b": np.arange(1000, dtype=float)})
    blocks = {4: big, 8: big + 1}
    other = big + 1
    other.loc[500, "a"] = -1.0
    assert compute_hash(blocks) != compute_hash({4: big, 8: other})
    assert compute_hash({4: big}) != compute_hash({4: big + 1})
    assert builtin_hash(big) in {builtin_hash(v) for v in blocks.values()}


def test_a_plain_small_collection_hash_is_unchanged():
    """Keys of plain collections already on disk must not move."""
    import hashlib
    import pickle

    value = {"a": [1, 2, 3], "b": "text", "c": 2.5}
    assert compute_hash(value) == hashlib.sha256(pickle.dumps(value)).hexdigest()
