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
