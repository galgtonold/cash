"""Work counts of the decorator: what a hit and a miss hash, pickle and copy.

A hit should hash its arguments once and copy its value in a few C-level
steps; a miss should pickle its value about once. Each test counts that work
and asserts it under a generous bound, so growth (a second hash of an
argument, a pickle per hit, a deepcopy walk per item) fails here without a
timer. Today's counts are in the comments. Counters: ``tests/_work_counts.py``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cash.backends._writes import all_pending_writes
from cash.decorator import arg_hashing
from tests._scripts import CASH_SRC, run_python
from tests._work_counts import deepcopy_steps, hashed_bytes, pickled_bytes, tree_walks

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())

pytestmark = pytest.mark.core

KB = 1024


def small_dict(n):
    return {f"key{i}": i * 0.5 + n for i in range(10)}


def nested_records(n):
    return [{"id": i + n, "user": f"u{i % 50}", "tags": [i, i + 1], "meta": {"ok": i % 2 == 0}} for i in range(2000)]


def numpy_array(n):
    return np.arange(200_000, dtype=np.float64) + n


def pandas_frame(n):
    return pd.DataFrame({"id": np.arange(20_000) + n, "user": [f"u{i % 100}" for i in range(20_000)]})


@dataclasses.dataclass
class Point:
    x: float
    label: str


def dataclass_list(n):
    return [Point(i * 0.5 + n, f"p{i % 7}") for i in range(2000)]


def keyed_args(arr, frame, config, names):
    return float(arr[:10].sum()) + len(frame) * config["lr"] + len(names)


VALUES = [small_dict, nested_records, numpy_array, pandas_frame, dataclass_list]


def _wait_for_writes() -> None:
    for queue in all_pending_writes():
        queue.wait_all()


def _pickle_size(value) -> int:
    return len(pickle.dumps(value, protocol=5))


def _written_size(value) -> int:
    """The bytes a disk entry holds *value* in: fewer than pickle's own for
    a column of Python strings (`cash.backends.text_columns`)."""
    from cash.backends.serialization import PickleSerializer

    stream, buffers = PickleSerializer().serialize_split(value)
    return len(stream) + sum(memoryview(b).nbytes for b in buffers)


@pytest.mark.parametrize("fn", VALUES, ids=lambda fn: fn.__name__)
def test_a_miss_pickles_its_value_about_once(disk_cash, fn):
    f = disk_cash.cache(fn)
    with pickled_bytes() as pickled:
        f(1)
        # The disk write serializes the RAM tier's copy on the writer
        # thread after the call returns: count it too.
        _wait_for_writes()
    size = _pickle_size(fn(1))
    # 1.0x today, 2.0x for the dataclass list (stored, and copied into RAM by
    # a pickle round trip); the bound catches one more pickle of the value.
    factor = 2.5 if fn is dataclass_list else 1.5
    assert pickled.bytes >= min(size, _written_size(fn(1))), "the value was never pickled: is it stored?"
    assert pickled.bytes <= factor * size + 16 * KB, (pickled.bytes, size)


@pytest.mark.parametrize("fn", VALUES, ids=lambda fn: fn.__name__)
def test_a_hit_hashes_a_small_argument_little_and_copies_in_c(disk_cash, fn):
    f = disk_cash.cache(fn)
    f(1)
    # The miss's disk write digests the value on the writer thread; let it
    # finish so the counters see only the hit.
    _wait_for_writes()
    with hashed_bytes() as hashed, pickled_bytes() as pickled, deepcopy_steps() as copies:
        f(1)
    assert f.cache_info()["hits"] == 1
    # ~0.4 KB hashed today (2.4 KB for the dataclass list's class).
    assert hashed.bytes <= 16 * KB
    # No deepcopy walk: JSON-like data, arrays and frames are copied a level
    # or a buffer at a time in C.
    assert copies.calls <= 5
    if fn is dataclass_list:
        # Objects are copied by a pickle round trip (as a disk hit reads
        # them): one pickle of the value, ~1.0x today.
        assert pickled.bytes <= 1.25 * _pickle_size(fn(1)) + 4 * KB
    else:
        assert pickled.bytes <= 4 * KB


def test_a_hit_hashes_a_big_argument_once(disk_cash):
    arr = np.random.default_rng(0).random(1_000_000)
    frame = pd.DataFrame({"a": np.arange(10_000)})
    args = (arr, frame, {"lr": 0.1}, ["a", "b"])
    f = disk_cash.cache(keyed_args)
    f(*args)
    with hashed_bytes() as hashed:
        f(*args)
    assert f.cache_info()["hits"] == 1
    # The argument's content is checked on every hit (it could have changed
    # in place), once: 8.0 MB of an 8 MB array plus a 80 KB frame today.
    assert arr.nbytes <= hashed.bytes <= 1.25 * (arr.nbytes + frame.memory_usage().sum()) + 16 * KB


def column_total(frame):
    return float(frame["a"].sum())


@pytest.fixture
def copy_on_write(monkeypatch, request):
    """The frame memo needs copy-on-write: the only mode from pandas 3 on,
    opt-in before it, so turned on here under pandas 2. Without it every
    frame is read before and after each call, by design."""
    if int(pd.__version__.split(".", 1)[0]) < 3:
        before = pd.get_option("mode.copy_on_write")
        pd.set_option("mode.copy_on_write", True)
        request.addfinalizer(lambda: pd.set_option("mode.copy_on_write", before))
        monkeypatch.setattr(arg_hashing, "_COW_PANDAS", None)


@pytest.mark.usefixtures("copy_on_write")
def test_a_frame_argument_is_read_once_on_a_miss_and_not_on_a_hit(disk_cash):
    frame = pd.DataFrame({"a": np.arange(500_000, dtype=np.float64), "b": np.arange(500_000)})
    size = int(frame.memory_usage(index=False).sum())
    f = disk_cash.cache(column_total)
    with hashed_bytes() as miss:
        f(frame)
    with hashed_bytes() as hit:
        f(frame)
    assert f.cache_info()["hits"] == 1
    # 1.01x the frame's 8 MB on the miss today. The hit is the same frame
    # object, unchanged: its digest is kept, 0.6 KB hashed today.
    assert size <= miss.bytes <= 1.25 * size + 64 * KB
    assert hit.bytes <= 16 * KB


@dataclasses.dataclass
class Settings:
    weights: np.ndarray
    name: str
    steps: int


def weighted(settings):
    return float(settings.weights.sum()) * settings.steps


def test_an_object_argument_is_read_at_most_twice_on_a_miss_and_once_on_a_hit(disk_cash):
    settings = Settings(np.arange(250_000, dtype=np.float64), "base", 3)
    size = settings.weights.nbytes
    f = disk_cash.cache(weighted)
    with hashed_bytes() as miss, deepcopy_steps() as miss_copies:
        f(settings)
    with hashed_bytes() as hit, deepcopy_steps() as hit_copies, pickled_bytes() as pickled:
        f(settings)
    assert f.cache_info()["hits"] == 1
    # An object can change in place, so each call reads it: 2.08x its 2 MB
    # array on the miss today (before and after the body), 1.0x on the hit.
    assert size <= miss.bytes <= 2.5 * size + 64 * KB
    assert size <= hit.bytes <= 1.25 * size + 64 * KB
    # The object is never copied or pickled to key it (1.6 KB pickled today,
    # the small result).
    assert miss_copies.calls == 0 and hit_copies.calls == 0
    assert pickled.bytes <= 16 * KB


def _scale(x):
    return x * 3


def _shift(x):
    return _scale(x) + 1


def _clean(x):
    return _shift(x) - 2


def through_helpers(x):
    return _clean(x)


def test_a_helper_chain_is_analysed_on_the_first_call_only(disk_cash):
    f = disk_cash.cache(through_helpers)
    with tree_walks() as first:
        f(1)
    with tree_walks() as second_miss, hashed_bytes() as miss:
        f(2)
    with tree_walks() as on_hit, hashed_bytes() as hit:
        f(1)
    assert f.cache_info()["hits"] == 1
    # 113 walks to read the function and its three helpers the first time;
    # the bound catches an analysis per helper per call.
    assert 0 < first.calls <= 400
    assert second_miss.calls == 0 and on_hit.calls == 0
    # 4.9 KB hashed by a later miss, 1.3 KB by a hit, today: the helpers'
    # code is checked from what the first call kept, not hashed again.
    assert miss.bytes <= 32 * KB
    assert hit.bytes <= 16 * KB


_FIRST_CALL = """
import json

import numpy as np
import pandas as pd

import cash
from tests._work_counts import hashed_bytes, pickled_bytes, tree_walks

if int(pd.__version__.split(".", 1)[0]) < 3:
    pd.set_option("mode.copy_on_write", True)  # the frame memo needs it


def _scale(x):
    return x * 3


def _shift(x):
    return _scale(x) + 1


@cash.cache
def total(frame, k):
    return _shift(float(frame["a"].sum())) + k


frame = pd.DataFrame({"a": np.arange(500_000, dtype=np.float64)})
with hashed_bytes() as hashed, pickled_bytes() as pickled, tree_walks() as walks:
    total(frame, 2)
print(
    json.dumps(
        {
            "hits": total.cache_info()["hits"],
            "hashed": hashed.bytes,
            "pickled": pickled.bytes,
            "walks": walks.calls,
            "frame": int(frame.memory_usage(index=False).sum()),
        }
    )
)
"""


def test_the_first_call_in_a_new_process_reads_its_frame_once_and_analyses_less(tmp_path):
    (tmp_path / "main.py").write_text(_FIRST_CALL, encoding="utf-8")
    # The child imports the counters from this checkout's tests package.
    path = {"PYTHONPATH": os.pathsep.join((CASH_SRC, str(REPO_ROOT)))}

    def run() -> dict:
        return json.loads(run_python("main.py", cwd=tmp_path, env=path).stdout.strip().splitlines()[-1])

    cold, warm = run(), run()
    assert (cold["hits"], warm["hits"]) == (0, 1)
    # A new process holds no digest of the frame: the hit reads it once,
    # 1.005x its 4 MB today, as the miss did.
    for counts in (cold, warm):
        assert counts["frame"] <= counts["hashed"] <= 1.25 * counts["frame"] + 64 * KB, counts
    # The stored result is a float: 0.4 KB unpickled on the hit today.
    assert warm["pickled"] <= 16 * KB
    # 71 walks in the process that stored the entry, 22 in the next one: a
    # hit checks the code it was stored under and does not analyse the
    # body again for what it may store.
    assert warm["walks"] < cold["walks"], (cold, warm)
