"""A class in a module of the user's named like a library's is not keyed
as one of that library's.

The content hashers, and the frame checks beside them, picked a class by
its module *prefix*: a ``DataFrame`` in the user's ``pandas_tools.py`` was
claimed by pandas' hasher, which reads a frame's ``index`` and ``_mgr``.
A class belongs to a library only when it is defined in that package or
one of its modules.
"""

from __future__ import annotations

import sys

import pytest

from cash.content_hashers import builtin_hash_family

SOURCE = """
CALLS = []


class DataFrame:
    def __init__(self, rows):
        self.rows = rows

    def __getattr__(self, name):
        if not name.startswith("__"):
            CALLS.append(name)
        raise AttributeError(name)

    @property
    def index(self):
        CALLS.append("index")
        return None

    @property
    def _mgr(self):
        CALLS.append("_mgr")
        return None
"""


@pytest.fixture
def pandas_tools(tmp_path, monkeypatch):
    (tmp_path / "pandas_tools.py").write_text(SOURCE)
    monkeypatch.syspath_prepend(str(tmp_path))
    import pandas_tools

    yield pandas_tools
    sys.modules.pop("pandas_tools", None)


def test_its_class_is_not_claimed_by_a_librarys_hasher(pandas_tools):
    assert builtin_hash_family(pandas_tools.DataFrame) is None
