"""After ``importlib.reload``, a caller that still holds the old cached function
is keyed by the old function, the one it runs.

The caller's key read the cached callee's state from the registry slot, which
the reload had filled with the NEW function. So the old code's result was
stored under the new code's key, and every later process -- running the new
code -- was served it.
"""

from __future__ import annotations

import pytest

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S
from tests.test_core._edited_project import edited_runs

pytestmark = pytest.mark.core

INNER = """
import cash


@cash.cache
def load(n):
    return n * 2
"""

OUTER = f"""
import time

import cash
from inner import load


@cash.cache
def top(n):
    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})
    return load(n) + 1
"""

RELOAD = """
import importlib
import pathlib

import inner
import outer

path = pathlib.Path("inner.py")
path.write_text(path.read_text().replace("n * 2", "n * 30"))
importlib.reload(inner)  # outer.load is still the old function
print(outer.top(2))
"""


def test_a_result_of_the_old_code_is_not_served_to_the_new_code(tmp_path):
    files = {"inner.py": INNER, "outer.py": OUTER, "main.py": RELOAD}
    first, _, _ = edited_runs(tmp_path, files)
    assert first == "5", "the old function still runs in the process that reloaded"
    # A later process runs the file as it is now.
    (tmp_path / "proj" / "main.py").write_text("import outer\nprint(outer.top(2))\n", encoding="utf-8")
    _, after, uncached = edited_runs(tmp_path, {}, [])
    assert after == uncached == "61"
