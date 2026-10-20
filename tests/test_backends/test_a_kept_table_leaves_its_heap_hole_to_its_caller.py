"""A big table the RAM tier keeps does not take glibc heap memory its
caller frees.

glibc serves a big allocation from a free hole of its main heap when one
fits. p04 rebinds a 47 MB signal in a loop; the tier kept each one (under
copy-on-write, the caller's own block), so the holes they would have left
stayed taken, and two temporaries of every later backtest were mapped and
page-faulted fresh: 48,924 faults a call under cash against 24,462 plain.
Now a block of at least 32 MB the tier would keep in the heap is copied
into a mapping of its own at its store (`heap_blocks`), and so is any big
copy the store makes.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from cash.backends import frame_sharing, heap_blocks
from cash.backends.memory_backend import InMemoryBackend

glibc_only = pytest.mark.skipif(not heap_blocks._glibc(), reason="glibc's heap only")

#: A 40 MB table: over `heap_blocks.MIN_BYTES`.
ROWS, COLS = 5000, 1000

_CHILD = textwrap.dedent(
    """
    import ctypes, json, resource, sys
    libc = ctypes.CDLL("libc.so.6")
    libc.mallopt(-3, 32 << 20)  # M_MMAP_THRESHOLD, fixed
    libc.mallopt(-1, 1 << 30)   # M_TRIM_THRESHOLD: freed heap stays heap
    import numpy as np
    import pandas as pd
    holes = [np.ones(2 << 20) for _ in range(12)]  # 192 MB of heap, then free
    del holes

    def signal(seed):
        return pd.DataFrame(np.full((ROWS, COLS), float(seed)))

    f = signal
    if sys.argv[1] == "cash":
        from cash import Cash
        from cash.backends.memory_backend import InMemoryBackend
        f = Cash(backend=InMemoryBackend(), register_magic=False).cache(signal)
    faults = []
    for seed in range(4):
        s = f(seed)  # the last one is freed, or kept by the tier
        for _ in range(3):
            before = resource.getrusage(resource.RUSAGE_THREAD).ru_minflt
            t = s * 2.0 + 1.0  # temporaries the size of the table
            del t
            faults.append(resource.getrusage(resource.RUSAGE_THREAD).ru_minflt - before)
    print(json.dumps(faults))
    """
).replace("ROWS, COLS", f"{ROWS}, {COLS}")


def _faults(mode: str, huge_pages: str) -> list[int]:
    env = dict(os.environ, NUMPY_MADVISE_HUGEPAGE=huge_pages)
    env.pop("CASH_DISABLE", None)
    out = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", _CHILD, mode],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


@glibc_only
@pytest.mark.parametrize("huge_pages", ["0", "1"], ids=["small pages", "numpy's huge pages"])
def test_later_work_faults_no_more_pages_than_without_cash(huge_pages):
    plain = _faults("plain", huge_pages)
    cached = _faults("cash", huge_pages)
    page_count = ROWS * COLS * 8 // 4096
    # Without the fix: 9,766 a call once the tier held three tables.
    assert sum(cached) <= sum(plain) + page_count // 10, (plain, cached)


def _heap_table() -> pd.DataFrame:
    """A table whose block glibc puts in the heap: built in a hole."""
    frame = pd.DataFrame(np.arange(float(ROWS * COLS)).reshape(ROWS, COLS))
    return frame[frame[0] >= 0]


@glibc_only
def test_a_table_in_the_heap_is_stored_in_a_mapping_of_its_own_and_served_as_returned():
    frame = _heap_table()
    expected = frame.copy(deep=True)
    backend = InMemoryBackend()
    with heap_blocks.per_store():
        in_heap = any(heap_blocks.in_heap(a) for a in frame_sharing.block_arrays(frame))
    backend.set("k", frame, {"execution_time": 1.0, "copy_required": True})
    kept = backend._store["k"][1]
    if in_heap:
        assert "k" not in backend._checked, "copied, not kept uncopied"
    assert all(not _in_heap_now(a) for a in frame_sharing.block_arrays(kept))
    frame.iloc[0, 0] = -1.0  # the caller writes its own table
    _meta, served = backend.get("k")
    pd.testing.assert_frame_equal(served, expected)
    assert pickle.dumps(served, protocol=5) == pickle.dumps(expected, protocol=5)


_WRITE_COPY_CHILD = textwrap.dedent(
    """
    import ctypes, json, pickle
    ctypes.CDLL("libc.so.6").mallopt(-3, 64 << 20)  # M_MMAP_THRESHOLD: a table this size is heap
    import numpy as np
    import pandas as pd
    from cash.backends import frame_sharing, heap_blocks
    from cash.backends.memory_backend import InMemoryBackend
    ROWS, COLS = SHAPE
    frame = pd.DataFrame(np.full((ROWS, COLS), 1.5), columns=[f"c{i}" for i in range(COLS)])
    copied = InMemoryBackend().write_copy("k", frame, {"execution_time": 1.0})
    with heap_blocks.per_store():
        in_heap = [heap_blocks.in_heap(a) for a in frame_sharing.block_arrays(copied)]
        usual = heap_blocks.in_heap(frame_sharing.block_arrays(frame.copy())[0])
    same = pickle.dumps(copied, protocol=5) == pickle.dumps(frame, protocol=5)
    print(json.dumps({"in_heap": in_heap, "usual_copy_in_heap": usual, "same": same}))
    """
).replace("SHAPE", f"{ROWS}, {COLS}")


@glibc_only
def test_the_copy_a_disk_write_waits_on_is_in_a_mapping_of_its_own():
    """With a disk tier, a store also copies the value for the background
    write (`InMemoryBackend.write_copy`), alive until the writer has it on
    disk. Made in the heap, it took the hole the caller's freed table had
    left: p04's notebook, whose 47 MB signal is written late each time,
    faulted about 1.8 GB of fresh huge pages more than plain (2,465 against
    1,541 huge-page faults), and its backtests ran 1.16x slower."""
    env = dict(os.environ)
    env.pop("CASH_DISABLE", None)
    out = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", _WRITE_COPY_CHILD],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=True,
    )
    found = json.loads(out.stdout.strip().splitlines()[-1])
    assert found["usual_copy_in_heap"], "the setup puts a copy of this size in the heap"
    assert found["in_heap"] == [False], found
    assert found["same"]


def _in_heap_now(array) -> bool:
    with heap_blocks.per_store():
        return heap_blocks.in_heap(array)


def test_a_table_kept_uncopied_under_copy_on_write_is_still_served_as_returned():
    """Below `heap_blocks.MIN_BYTES`, or out of the heap: kept uncopied as
    before, and served as returned after the caller writes its own."""
    if not frame_sharing._checked_works():
        pytest.skip("needs pandas copy-on-write")
    n = frame_sharing.CHECKED_MIN_BYTES // 8 + 1
    frame = pd.DataFrame({"x": np.arange(float(n))})
    expected = frame.copy(deep=True)
    backend = InMemoryBackend()
    backend.set("k", frame, {"execution_time": 1.0, "copy_required": True})
    frame.loc[0, "x"] = -1.0
    _meta, served = backend.get("k")
    pd.testing.assert_frame_equal(served, expected)


@glibc_only
@pytest.mark.parametrize(
    "make",
    [
        lambda: np.arange(float(ROWS * COLS)).reshape(ROWS, COLS),
        lambda: np.asfortranarray(np.arange(float(ROWS * COLS)).reshape(ROWS, COLS)),
        lambda: np.zeros(ROWS * COLS, dtype=[("a", "u1"), ("b", "f8")]),
    ],
    ids=["C order", "F order", "a record dtype"],
)
def test_a_big_copy_at_a_store_is_the_same_bytes_in_a_mapping(make):
    array = make()
    with heap_blocks.per_store():
        copied = heap_blocks.fresh_copy(array)
    assert copied is not None and copied.base is not None
    assert copied.flags.c_contiguous == array.flags.c_contiguous
    assert copied.flags.f_contiguous == array.flags.f_contiguous
    assert pickle.dumps(copied, protocol=5) == pickle.dumps(_usual_copy(array), protocol=5)


def _usual_copy(array):
    from cash.backends import memory_backend

    return memory_backend._copy_values(array)  # outside a store: the usual copy


def test_nothing_changes_outside_a_store_or_without_glibc(monkeypatch):
    array = np.zeros(heap_blocks.MIN_BYTES // 8)
    assert heap_blocks.fresh_copy(array) is None, "a hit's copy is the caller's"
    monkeypatch.setattr(heap_blocks, "_GLIBC", [False])
    with heap_blocks.per_store():
        assert heap_blocks.fresh_copy(array) is None
        assert heap_blocks.in_heap(array) is False


@glibc_only
def test_the_heap_is_read_once_a_store_and_only_for_a_big_block(monkeypatch):
    reads = []
    real = heap_blocks._read_heap_range
    monkeypatch.setattr(heap_blocks, "_read_heap_range", lambda: reads.append(1) or real())
    with heap_blocks.per_store():
        heap_blocks.in_heap(np.zeros(1000))
        assert reads == []
        big = np.zeros(heap_blocks.MIN_BYTES // 8)
        heap_blocks.in_heap(big)
        heap_blocks.in_heap(big[1:])
    assert reads == [1]


@glibc_only
def test_an_unreadable_heap_keeps_todays_path(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise OSError("no /proc")

    monkeypatch.setattr("builtins.open", refuse)
    with heap_blocks.per_store():
        assert heap_blocks.in_heap(np.zeros(heap_blocks.MIN_BYTES // 8)) is False
