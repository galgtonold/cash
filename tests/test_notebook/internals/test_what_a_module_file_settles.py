"""What a module's file, read without importing it, tells a restore: its
state (`shadow._file_state`), whether importing it runs code of the
user's (`shadow._inert_imports`), whether what it imports is a library's
(`shadow._is_library`), and whether a loaded module's file is unchanged
since the load (`shadow._unchanged_since_loaded`)."""

from __future__ import annotations

import ast
import os
import sys
import time
import types

import pytest

from cash.notebook import shadow


def test_a_file_swapped_keeping_its_mtime_and_size_is_read_again(tmp_path):
    """``cp -p`` or ``touch -r``: the same mtime and size, other text. The
    file's state was kept by mtime and size and the old one served."""
    path = tmp_path / "swapped.py"
    path.write_text("def f(x):\n    return x * 2\n", encoding="utf-8")
    old = path.stat()
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns - 10_000_000_000))
    old = path.stat()
    first = shadow._file_state(str(path))
    path.write_text("def f(x):\n    return x * 9\n", encoding="utf-8")
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))

    assert shadow._file_state(str(path)).digest != first.digest


def test_a_docstring_edit_changes_the_files_digest(tmp_path):
    """A docstring can be read as the code runs (``int(f.__doc__)``,
    ``description=__doc__``): an edit of one is an edit of the file."""
    path = tmp_path / "doc.py"
    path.write_text('def f(x):\n    "2"\n    return x\n', encoding="utf-8")
    first = shadow._file_state(str(path)).digest
    path.write_text('def f(x):\n    "9"\n    return x\n# a comment\n', encoding="utf-8")

    assert shadow._file_state(str(path)).digest != first


INERT = {
    "defs_and_literals": "'''Doc.'''\nimport os\nK = 2\nNAMES = ('a', 'b')\ndef f(x, k=2):\n    return x * K\n",
    "library_decorators": (
        "import functools\nfrom dataclasses import dataclass, field\n"
        "@dataclass(frozen=True)\nclass P:\n    x: int = 0\n    tags: list[str] = field(default_factory=list)\n"
        "@functools.lru_cache(maxsize=None)\ndef f(x):\n    return x\n"
        "class Q:\n    @property\n    def y(self):\n        return 1\n    __radd__ = y\n"
    ),
    "library_and_own_bases": (
        "import enum\nclass Color(enum.Enum):\n    RED = enum.auto()\n"
        "class Err(ValueError):\n    pass\nclass Sub(Err):\n    CODE = 3\n"
    ),
    "allowed_calls": (
        "import logging, re\nfrom pathlib import Path\nlog = logging.getLogger(__name__)\n"
        "PAT = re.compile('x+')\nHERE = Path(__file__).parent\n"
    ),
    "guards": (
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import helpers\n"
        "def main():\n    pass\nif __name__ == '__main__':\n    main()\n"
    ),
}

RUNS_CODE = {
    # Round 6's seven shapes: each binds module state from outside the file.
    "class_body_call": "import os\nclass C:\n    K = os.getenv('HK', '2')\n",
    "own_decorator": "import os\ndef scale(fn):\n    return fn\n@scale\ndef f(x):\n    return x\n",
    "own_class_decorator": "def mark(cls):\n    cls.K = 9\n    return cls\n@mark\nclass C:\n    K = 2\n",
    "init_subclass": (
        "import os\nclass Base:\n    def __init_subclass__(cls):\n        cls.K = 9\nclass C(Base):\n    K = 2\n"
    ),
    "module_getattr": "import os\nclass _C:\n    E = 2\ndef __getattr__(name):\n    return _C.E\n",
    "import_module": "import importlib\nCFG = 2\nimportlib.import_module('overrides')\n",
    "import_module_assigned": "import importlib\nM = importlib.import_module('overrides')\n",
    # Others.
    "own_metaclass": "class Meta(type):\n    pass\nclass C(metaclass=Meta):\n    pass\n",
    "library_call_reading_the_environment": "import os\nK = os.getenv('HK')\n",
    "attribute_store": "import os\nos.environ['X'] = '1'\n",
    "branch": "import os\nif os.environ.get('X'):\n    K = 9\n",
    "star": "from os.path import *\n",
    "default_from_a_call": "import os\ndef f(x, k=os.getenv('K')):\n    return x\n",
}


@pytest.mark.parametrize("case", list(INERT))
def test_a_body_that_runs_no_code_of_the_users_settles_its_names(case):
    assert shadow._inert_imports(ast.parse(INERT[case])) is not None


@pytest.mark.parametrize("case", list(RUNS_CODE))
def test_a_body_that_runs_code_as_it_is_imported_settles_nothing(case):
    assert shadow._inert_imports(ast.parse(RUNS_CODE[case])) is None


def test_a_library_is_told_by_where_its_file_is_without_importing_it(tmp_path, monkeypatch):
    """After a restart a helper's ``import numpy`` has not run when a
    restore is checked: told by its spec, it is a library's still."""
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "users_overrides_mod.py").write_text("K = 9\n", encoding="utf-8")
    for name in ("colorsys", "json.decoder"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(shadow, "_LIBRARIES", {})
    assert shadow._is_library("colorsys")
    assert shadow._is_library("json.decoder")
    assert shadow._is_library("sys")
    assert "colorsys" not in sys.modules
    assert not shadow._is_library("users_overrides_mod")
    assert not shadow._is_library("no_such_module_anywhere")


def _loaded(path, cached=None):
    module = types.ModuleType("m")
    module.__file__ = str(path)
    module.__cached__ = cached
    return module


def test_a_file_older_than_the_process_needs_no_compiled_file(tmp_path, monkeypatch):
    """``PYTHONDONTWRITEBYTECODE=1``, ``python -B``: no compiled file, and
    nothing vouched although nothing changed."""
    path = tmp_path / "m.py"
    path.write_text("K = 2\n", encoding="utf-8")
    monkeypatch.setattr("cash.process_start.process_start_time", lambda: path.stat().st_ctime + 60)
    assert shadow._unchanged_since_loaded(_loaded(path), str(path))
    monkeypatch.setattr("cash.process_start.process_start_time", lambda: path.stat().st_ctime - 60)
    assert not shadow._unchanged_since_loaded(_loaded(path), str(path))


def test_an_edit_in_the_second_the_compiled_file_was_written_is_seen(tmp_path, monkeypatch):
    """The compiled file records the source's mtime in whole seconds: an
    edit keeping the size within that second passed."""
    import importlib.util
    import py_compile

    monkeypatch.setattr("cash.process_start.process_start_time", lambda: 0.0)
    path = tmp_path / "m.py"
    path.write_text("K = 2\n", encoding="utf-8")
    cached = importlib.util.cache_from_source(str(path))
    st = path.stat()
    py_compile.compile(str(path), cfile=cached)
    os.utime(cached, ns=(st.st_atime_ns, st.st_ctime_ns))
    assert shadow._unchanged_since_loaded(_loaded(path, cached), str(path))

    time.sleep(0.05)  # a later change time, within the same second as likely as not
    path.write_text("K = 9\n", encoding="utf-8")
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))  # same mtime and size
    assert not shadow._unchanged_since_loaded(_loaded(path, cached), str(path))
