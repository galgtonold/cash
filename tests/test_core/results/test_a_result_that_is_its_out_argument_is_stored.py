"""A cached call that fills an ``out=`` array in place and returns it is
stored, and a hit fills the caller's array and returns that same object.

``librosa.griffinlim`` calls a cached ``stft(y, out=rebuilt)``, which writes
into ``rebuilt`` and returns it. The result being one of the arguments
refused the store (a hit would have handed back a separate copy), so every
call ran. The entry now keeps the state the call left ``out`` in, once, and
a hit writes it into the ``out`` handed in and returns that object: ``result
is out`` holds as in a plain run, and a view's write lands in its base.

Only arguments a hit can write back in place (NumPy arrays, pandas frames
and Series, plain lists, dicts and sets) are handled so; any other object
the result is still refuses the store.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

pytestmark = pytest.mark.core


def _filler(cache):
    runs = []

    @cache.cache
    def fill(x, out=None):
        runs.append(1)
        if out is None:
            out = np.empty_like(x)
        np.multiply(x, 2.0, out=out)  # @cash:assume-safe
        return out

    return fill, runs


_SCRIPT = textwrap.dedent(
    """
    import json
    import numpy as np
    from cash import Cash

    c = Cash(cache_dir=".c", register_magic=False)
    runs = []

    @c.cache
    def fill(x, out=None):
        runs.append(1)
        if out is None:
            out = np.empty_like(x)
        np.multiply(x, 2.0, out=out)  # @cash:assume-safe
        return out

    out = np.zeros(6)
    result = fill(np.arange(6.0), out=out)
    print(json.dumps({"runs": len(runs), "same": result is out, "out": out.tolist()}))
    """
)


def _run(tmp_path):
    import json

    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    p = subprocess.run(
        [sys.executable, "-c", _SCRIPT], cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=120
    )
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_a_hit_in_another_process_fills_out_and_returns_it(tmp_path):
    first = _run(tmp_path)
    second = _run(tmp_path)
    expected = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    assert first == {"runs": 1, "same": True, "out": expected}
    assert second == {"runs": 0, "same": True, "out": expected}, "the second process ran the call again"


def test_a_hit_returns_the_out_handed_in(disk_cash):
    fill, runs = _filler(disk_cash)
    x = np.arange(6.0)
    fill(x, out=np.zeros(6))
    out = np.zeros(6)
    alias = out
    result = fill(x, out=out)
    assert len(runs) == 1, "the call that fills and returns out ran again"
    assert result is out and alias is out
    assert out.tolist() == (x * 2).tolist()
    # The caller's array, not a copy: a write through one shows in the other.
    result[0] = 7.0
    assert out[0] == 7.0


def test_an_out_of_another_shape_or_dtype_misses(disk_cash):
    fill, runs = _filler(disk_cash)
    x = np.arange(6.0)
    fill(x, out=np.zeros(6))
    out32 = np.zeros(6, dtype=np.float32)
    result = fill(x, out=out32)
    assert len(runs) == 2
    assert result is out32 and out32.dtype == np.float32
    with pytest.raises(ValueError):
        fill(x, out=np.zeros(5))
    assert len(runs) == 3


def test_out_holding_other_values_before_the_call_misses(disk_cash):
    """The key holds what ``out`` held before the call, as for any argument."""
    fill, runs = _filler(disk_cash)
    x = np.arange(6.0)
    fill(x, out=np.zeros(6))
    fill(x, out=np.ones(6))
    assert len(runs) == 2


def test_a_hit_into_a_view_writes_into_its_base(disk_cash):
    fill, runs = _filler(disk_cash)
    x = np.arange(6.0)
    fill(x, out=np.zeros(10)[2:8])
    big = np.zeros(10)
    view = big[2:8]
    result = fill(x, out=view)
    assert len(runs) == 1
    assert result is view
    assert big.tolist() == [0.0, 0.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 0.0, 0.0]


class Buffer:
    """An object a hit cannot write a state back into: the call changes the
    list it holds in place, which no replay of its attributes puts back."""

    def __init__(self):
        self.values = []


def test_any_other_out_object_still_runs_every_time(disk_cash):
    runs = []

    @disk_cash.cache
    def fill(n, out):
        runs.append(1)
        out.values.extend(range(n))
        return out

    for _ in range(3):
        out = Buffer()
        assert fill(3, out) is out
        assert out.values == [0, 1, 2]
    assert len(runs) == 3
