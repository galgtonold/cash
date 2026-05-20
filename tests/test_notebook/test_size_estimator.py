"""Unit tests for the extended `_estimate_object_size` dispatch table."""
from __future__ import annotations

import sys

import numpy as np
import pytest
import scipy.sparse as sp
from unittest.mock import MagicMock
from traitlets.config.configurable import Configurable

from cash.core import Cash
from cash.backends.backend import InMemoryBackend
from cash.notebook.statement_processor import StatementProcessor


class MockShell(Configurable):
    def __init__(self):
        super().__init__()
        self.user_ns = {}
        self.input_transformers_cleanup = []
        self.ast_transformers = []
        self.user_global_ns = self.user_ns
        self.display_pub = type('MockDisplayPub', (), {'publish': MagicMock()})()


@pytest.fixture
def processor():
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = MockShell()
    proc = StatementProcessor(shell=shell, cash_instance=cash, debug=False)
    yield proc
    backend.clear()


def test_csr_matrix_size_equals_data_plus_indices_plus_indptr(processor):
    m = sp.csr_matrix(([1.0] * 1000, ([0] * 1000, list(range(1000)))),
                      shape=(1, 1000))
    expected = m.data.nbytes + m.indices.nbytes + m.indptr.nbytes
    assert processor._estimate_object_size(m) == expected


def test_csc_matrix_size_equals_data_plus_indices_plus_indptr(processor):
    m = sp.csc_matrix(([1.0] * 1000, ([0] * 1000, list(range(1000)))),
                      shape=(1, 1000))
    expected = m.data.nbytes + m.indices.nbytes + m.indptr.nbytes
    assert processor._estimate_object_size(m) == expected


def test_coo_matrix_size_equals_data_plus_row_plus_col(processor):
    m = sp.coo_matrix(([1.0] * 1000, ([0] * 1000, list(range(1000)))),
                      shape=(1, 1000))
    expected = m.data.nbytes + m.row.nbytes + m.col.nbytes
    assert processor._estimate_object_size(m) == expected


from collections import namedtuple
from dataclasses import dataclass


def test_dataclass_with_mixed_payload(processor):
    @dataclass
    class Bundle:
        arr: np.ndarray
        df_size: int

    arr = np.zeros(1_000_000, dtype=np.float64)  # 8_000_000 bytes
    bundle = Bundle(arr=arr, df_size=42)
    size = processor._estimate_object_size(bundle)
    # arr is 8_000_000 bytes; the int is tiny; total should be just above that
    assert size > 8_000_000
    assert size < 9_000_000  # not catastrophically over


def test_namedtuple_of_arrays(processor):
    Point = namedtuple('Point', 'x y')
    pt = Point(np.zeros(1_000_000), np.zeros(1_000_000))  # 2 x 8_000_000 bytes
    size = processor._estimate_object_size(pt)
    assert size > 16_000_000
    assert size < 17_000_000


def test_dataclass_class_object_is_not_treated_as_instance(processor):
    """``is_dataclass()`` returns True for both classes and instances;
    we must only recurse for instances."""
    @dataclass
    class Empty:
        x: int = 0

    # Passing the class itself, not an instance — should fall back to
    # sys.getsizeof(Empty) not iterate over a class's "fields".
    size = processor._estimate_object_size(Empty)
    assert size == sys.getsizeof(Empty)


def test_homogeneous_list_of_ndarrays(processor):
    arrs = [np.zeros(1_000_000, dtype=np.float64) for _ in range(10)]  # 10 x 8 MB SI
    size = processor._estimate_object_size(arrs)
    assert size > 80_000_000
    assert size < 82_000_000


def test_heterogeneous_monotonic_list_uses_first_middle_last(processor):
    # Sizes grow monotonically: arr[i] = np.zeros(i * 100_000), so
    # first=0, middle=500k floats (4 MB), last=900k floats (7.2 MB).
    # K=3 average ~ (0 + 4MB + 7.2MB) / 3 = 3.7 MB; x 10 elements
    # ~ 37 MB. Truth is sum of arange(0, 10) * 100_000 * 8 = 36 MB.
    arrs = [np.zeros(i * 100_000, dtype=np.float64) for i in range(10)]
    size = processor._estimate_object_size(arrs)
    # Within 30% of truth (the K=3 average is necessarily approximate)
    assert 25_000_000 < size < 50_000_000


def test_dict_of_dataframes_uses_first_and_last(processor):
    import pandas as pd
    # First and last are 5MB DataFrames; middle ones are tiny.
    big = pd.DataFrame({'x': np.zeros(625_000, dtype=np.float64)})  # ~5 MB
    small = pd.DataFrame({'x': [0.0]})
    d = {0: big, 1: small, 2: small, 3: small, 4: big}
    size = processor._estimate_object_size(d)
    # K=2 average is (5MB + 5MB) / 2 = 5 MB; x 5 entries = 25 MB
    assert 20_000_000 < size < 30_000_000


def test_set_of_bytes(processor):
    chunk = b"x" * 1000
    s = {chunk + bytes([i]) for i in range(100)}  # 100 distinct 1001-byte values
    size = processor._estimate_object_size(s)
    # K=1: sizeof(set wrapper) + 100 x ~1001 bytes ~ 100 KB
    assert 90_000 < size < 200_000


def test_bytes_and_bytearray_are_their_length(processor):
    b = b"x" * 12345
    ba = bytearray(b"y" * 67890)
    assert processor._estimate_object_size(b) == 12345
    assert processor._estimate_object_size(ba) == 67890


def test_depth_cap_terminates_deep_nesting(processor):
    # Build a 6-deep nested dict — deeper than _MAX_ESTIMATE_DEPTH = 4
    payload = np.zeros(1_000_000, dtype=np.float64)  # 8 MB
    d = {'leaf': payload}
    for _ in range(6):
        d = {'level': d}
    # Without depth cap this would recurse and find the 8 MB payload.
    # With the cap, it returns sys.getsizeof at depth 4 - small number.
    size = processor._estimate_object_size(d)
    assert size < 1_000_000  # well under 1 MB; the cap fired


def test_empty_containers_are_their_wrapper_size(processor):
    assert processor._estimate_object_size([]) == sys.getsizeof([])
    assert processor._estimate_object_size({}) == sys.getsizeof({})
    assert processor._estimate_object_size(set()) == sys.getsizeof(set())
    assert processor._estimate_object_size(()) == sys.getsizeof(())


def test_unknown_type_falls_back_to_sys_getsizeof(processor):
    class Custom:
        pass
    obj = Custom()
    assert processor._estimate_object_size(obj) == sys.getsizeof(obj)
