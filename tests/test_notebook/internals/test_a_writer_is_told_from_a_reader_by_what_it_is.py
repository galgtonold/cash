"""What a function holds is a writer by what it is, not by its name alone
(`call_key._a_writer`, read by `holds_a_stream` and `changes_its_closure`).

A builtin method is a writer unless its object cannot change or the
method is on the exact list of those that only read; a library's Python
method is one when named as a method that changes its object; a partial
or a ``methodcaller`` by what it calls and hands on. Pinned on the walk
itself: which holder is refused is how the work is done.
"""

import collections
import functools
import operator
import struct

import numpy as np
import pandas as pd

from cash.notebook.call_key import changes_its_closure, holds_a_stream


def _holding(held):
    def g(x):
        return held(x)

    return g


def test_writers_are_refused():
    calls, table, arr = [], {}, np.zeros(2)
    for writer in (
        calls.append,
        table.__setitem__,
        arr.fill,
        functools.partial(setattr, object()),
        {"n": calls.append},
        collections.Counter().update,
        collections.UserList().append,
        pd.Series([1]).update,
        functools.partial(np.add, arr, 1, out=arr),
        functools.partial(np.copyto, arr),
        operator.methodcaller("append", 1),
        collections.defaultdict(int).__getitem__,
        arr.sum,  # takes ``out=``: the call it is given is not read
        functools.partial(np.add, arr, 1),  # named ``add``, handed an array
    ):
        assert holds_a_stream(_holding(writer)) and changes_its_closure(_holding(writer)), writer


def test_readers_are_not():
    calls, table, arr = [], {}, np.arange(4.0)
    for reader in (
        calls.count,
        table.get,
        table.__getitem__,
        len,
        str.join,
        pd.Series([1]).get,
        arr.__getitem__,
        arr.astype,
        arr.tolist,
        arr.searchsorted,
        np.float64(2.0).__mul__,
        struct.Struct("<i").unpack,
        operator.methodcaller("get", 1),
    ):
        assert not holds_a_stream(_holding(reader)) and not changes_its_closure(_holding(reader)), reader
