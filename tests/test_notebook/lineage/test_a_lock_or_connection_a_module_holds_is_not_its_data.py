"""A lock or a database connection a local module holds is not data a statement is keyed on.

``x = mylib.slow(3)`` with ``slow`` using a module-level ``threading.Lock()``
or ``sqlite3.connect(...)``: the module data a statement reaches is keyed by
value, and these hash by identity, so every new kernel made a new key and the
cell never hit after a restart. No result is computed from a lock, and what
a database answers is read through its connection, not held in it.
"""

import os
import sys

import pytest

from cash.notebook.callee_reach import reached_user_code

LIB = (
    "import sqlite3, threading, logging\n"
    "K = 2\n_LOCK = threading.Lock()\n_DB = sqlite3.connect(':memory:')\n_LOG = logging.getLogger('x')\n"
    "def slow(n):\n    with _LOCK:\n        _DB.execute('select 1')\n        _LOG.debug('n')\n        return K * n\n"
)


@pytest.fixture
def lib(tmp_path, monkeypatch):
    name = f"_handlelib_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(LIB, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield name
    sys.modules.pop(name, None)


def test_only_the_data_is_reached(lib):
    module = __import__(lib)
    assert [label for label, _ in reached_user_code("x = lib.slow(3)", {"lib": module}).data] == [f"{lib}.K"]
