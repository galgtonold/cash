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

from cash.notebook.object_hashing import compute_hash

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

PROBE = textwrap.dedent("""
    import numpy as np, pandas as pd
    from cash.notebook.object_hashing import compute_hash
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


def test_a_numeric_frame_hash_is_unchanged():
    """Keys of numeric frames already on disk must not move."""
    import hashlib

    frame = pd.DataFrame({"a": np.arange(10), "b": np.arange(10) * 0.5})
    legacy = hashlib.sha256(
        f"{frame.shape}:{frame.dtypes.to_dict()}:{frame.head(5).values.tobytes()}".encode()
    ).hexdigest()
    assert compute_hash(frame) == legacy


def test_a_small_collection_of_large_frames_is_not_pickled_whole():
    """A dict holding a few large frames was hashed by
    pickling the WHOLE dict -- every frame byte for byte -- after each restore
    and after each loop iteration that changed it, while the same frame on its
    own is hashed by sampling. At 400 MiB a frame that is seconds per hit, and
    their "hits" cost more than the compute they saved."""
    import pickle
    import time

    big = pd.DataFrame({"a": np.arange(2_000_000, dtype=float), "b": np.arange(2_000_000, dtype=float)})
    blocks = {4: big, 8: big + 1}
    t0 = time.perf_counter()
    compute_hash(blocks)
    took = time.perf_counter() - t0
    t0 = time.perf_counter()
    pickle.dumps(blocks)
    full = time.perf_counter() - t0
    assert took < full / 4, (took, full)
    assert compute_hash(blocks) != compute_hash({4: big, 8: big + 2})


def test_a_plain_small_collection_hash_is_unchanged():
    """Keys of plain collections already on disk must not move."""
    import hashlib
    import pickle

    value = {"a": [1, 2, 3], "b": "text", "c": 2.5}
    assert compute_hash(value) == hashlib.sha256(pickle.dumps(value)).hexdigest()
