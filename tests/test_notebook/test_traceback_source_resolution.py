"""CAS-201: a function DEFINED in a cash cell must resolve its source.

Every statement used to compile under one reused literal ``<cash>`` filename that
was never registered in :mod:`linecache`, so a traceback frame inside a
cell-defined function printed ``File "<cash>", line N`` with NO source line, and
``inspect.getsource`` raised "could not get source code" — losing the failing
line in every debug loop.

Each statement now compiles under a stable, per-statement ``<cash-{digest}>``
name whose source is registered in linecache.
"""
import inspect
import linecache
import traceback
from unittest.mock import MagicMock

import pytest
from traitlets.config import Configurable

from cash import Cash
from cash.backends import InMemoryBackend
from cash.notebook.compiled_source import (
    is_cash_filename,
    register_cell_source,
)
from cash.notebook.ipython.magics import CashMagics


class MockShell(Configurable):
    def __init__(self):
        super().__init__()
        self.user_ns = {}
        self.input_transformers_cleanup = []
        self.run_cell = MagicMock()
        self.events = MagicMock()
        self.ast_transformers = []
        self.user_global_ns = self.user_ns
        self.display_pub = type('MockDisplayPub', (), {'publish': MagicMock()})()


@pytest.fixture
def magics_fixture():
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = MockShell()
    magics = CashMagics(shell, cash)
    magics._auto_cache_enabled = True
    yield magics, shell
    backend.clear()
    shell.user_ns.clear()


# --------------------------------------------------------------------------
# the helper itself
# --------------------------------------------------------------------------

def test_register_cell_source_is_stable_and_resolvable():
    """Same source -> same name (one linecache entry, not unbounded growth)."""
    src = "def f():\n    return 1\n"
    a = register_cell_source(src)
    b = register_cell_source(src)
    assert a == b
    assert linecache.getline(a, 2).strip() == "return 1"


def test_register_cell_source_distinguishes_statements():
    """Different source -> different name, so line numbers can't collide."""
    a = register_cell_source("x = 1")
    b = register_cell_source("y = 2")
    assert a != b


@pytest.mark.parametrize("name,expected", [
    ("<cash-6f20c37e324f>", True),
    ("<cash>", True),            # historical bare form still recognised
    ("/home/u/real_module.py", False),
    ("<string>", False),
    ("", False),
    (None, False),
])
def test_is_cash_filename(name, expected):
    assert is_cash_filename(name) is expected


# --------------------------------------------------------------------------
# the actual CAS-201 symptom
# --------------------------------------------------------------------------

def test_traceback_shows_source_of_cell_defined_function(magics_fixture):
    """The failing line appears in the traceback, not a bare '<cash>' frame."""
    magics, shell = magics_fixture
    magics.cash("", "def compute_ratio(a, b):\n    scaled = a * 100\n    return scaled / b")

    fn = shell.user_ns["compute_ratio"]
    assert is_cash_filename(fn.__code__.co_filename)

    try:
        fn(5, 0)
    except ZeroDivisionError:
        tb = traceback.format_exc()
    else:
        pytest.fail("expected ZeroDivisionError")

    # The whole point of CAS-201: the failing SOURCE LINE is present.
    assert "return scaled / b" in tb, tb
    assert "compute_ratio" in tb


def test_inspect_getsource_works_on_cell_defined_function(magics_fixture):
    """``inspect.getsource`` no longer raises "could not get source code"."""
    magics, shell = magics_fixture
    magics.cash("", "def greet(name):\n    return f'hi {name}'")

    src = inspect.getsource(shell.user_ns["greet"])
    assert "def greet(name):" in src
    assert "return f'hi {name}'" in src
