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
