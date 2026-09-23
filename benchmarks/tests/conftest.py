"""Tests for the benchmark tooling itself (drivers, result files, fits).

They are not part of the main suite; run them with

    pytest benchmarks/tests

The drivers call ``InteractiveShell.instance()``, which registers a
process-global singleton that outlives the test. Clearing it after each test
keeps a later test in the same worker from seeing a live IPython session.
"""

import sys
from pathlib import Path

import pytest
from IPython.core.interactiveshell import InteractiveShell

# The tests import the tooling as ``benchmarks.<module>``.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pytest.fixture(autouse=True)
def _clear_ipython_singleton():
    yield
    InteractiveShell.clear_instance()
