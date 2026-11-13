"""A plain function that something else in the process wraps with
``cash.cache`` keys its callers the same way whether or not that wrapping
has happened yet.

Libraries that hand caching to a factory (``cache(func)(args)`` inside a
body, the joblib ``Memory.cache`` shape) wrap a helper only when that line
runs. The caller names the bare helper, which runs uncached and is keyed by
its code like any helper. When the helper also counted as a cached-function
dependency once some earlier call had wrapped it, the caller's key depended
on which calls ran their bodies first: a warm run, where earlier calls hit
and wrapped nothing, missed every call the cold run had stored.
"""

from __future__ import annotations

import textwrap

from tests._scripts import run_python

MODULE = """
import cash

def cache(func):
    return cash.cache(func)

def smooth(x):
    return x * {factor}

@cash.cache
def outer(x):
    return cache(smooth)(x) + 1
"""

SCRIPT = """
import os, sys
sys.path.insert(0, os.getcwd())
import mod
if os.environ.get("WRAP_FIRST"):
    mod.cache(mod.smooth)
print(mod.outer.explain(3).reason)
print(mod.outer(3))
"""


def _run(tmp_path, wrap_first: bool) -> list[str]:
    env = {"WRAP_FIRST": "1" if wrap_first else None}
    return run_python("main.py", cwd=tmp_path, env=env).stdout.split()


def test_a_caller_hits_whether_or_not_its_helper_was_wrapped_first(tmp_path):
    (tmp_path / "mod.py").write_text(textwrap.dedent(MODULE.format(factor=2)), encoding="utf-8")
    (tmp_path / "main.py").write_text(SCRIPT, encoding="utf-8")

    assert _run(tmp_path, wrap_first=False) == ["no_entry", "7"]
    assert _run(tmp_path, wrap_first=True) == ["hit", "7"]
    assert _run(tmp_path, wrap_first=False) == ["hit", "7"]


def test_an_edit_to_the_wrapped_helper_still_reaches_the_caller(tmp_path):
    """The positive control: keyed as a helper, the bare function's code is
    still in the caller's key."""
    (tmp_path / "mod.py").write_text(textwrap.dedent(MODULE.format(factor=2)), encoding="utf-8")
    (tmp_path / "main.py").write_text(SCRIPT, encoding="utf-8")
    assert _run(tmp_path, wrap_first=True) == ["no_entry", "7"]

    (tmp_path / "mod.py").write_text(textwrap.dedent(MODULE.format(factor=5)), encoding="utf-8")
    reason, value = _run(tmp_path, wrap_first=True)
    assert reason != "hit"
    assert value == "16"
