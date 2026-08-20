"""A helper whose source cannot be read must still invalidate its callers.

Such a helper used to be dropped as an "opaque leaf": it contributed
NOTHING to the caller's state hash, so any edit to it went unnoticed and
the caller served a stale result. Not merely a changed constant --
replacing the entire body was invisible too.

The fixture builds the helper with ``exec`` under a filename absent from
``linecache``, which is what genuinely breaks ``inspect.getsource``. Each
test asserts that premise, because a fixture that quietly regains its
source would turn every assertion below into a tautology.
"""

import importlib
import importlib.util
import linecache
import shutil
import sys

import pytest

from cash import Cash

DYNMOD = '''
_SRC = {src!r}
exec(compile(_SRC, "<dynamic-absent-from-linecache>", "exec"), globals())
'''

USERMOD = """
import dynmod

@cash_instance.cache
def compute(n):
    return dynmod.helper(n)
"""

ALPHA = 'def helper(x):\n    return "alpha"\n'


@pytest.fixture()
def env(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.syspath_prepend(str(work))
    cash = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    return cash, work


def _load(cash, work, helper_body):
    """Rewrite the helper module and import both modules fresh.

    ``__pycache__`` is removed every time: two helper bodies of the SAME
    LENGTH written within one mtime second let Python reuse a stale
    ``.pyc``, so the edit never reaches the interpreter. That artefact
    silently faked a cache hit while this test was being written.
    """
    (work / "dynmod.py").write_text(DYNMOD.format(src=helper_body))
    (work / "usermod.py").write_text(USERMOD)
    shutil.rmtree(work / "__pycache__", ignore_errors=True)
    for name in ("dynmod", "usermod"):
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    linecache.clearcache()

    dynmod = importlib.import_module("dynmod")
    spec = importlib.util.spec_from_file_location("usermod", str(work / "usermod.py"))
    usermod = importlib.util.module_from_spec(spec)
    sys.modules["usermod"] = usermod
    usermod.__dict__["cash_instance"] = cash
    spec.loader.exec_module(usermod)
    return usermod, dynmod


def _call(cash, work, helper_body):
    """Load, assert the premise, call once; return (was_hit, value)."""
    import inspect

    usermod, dynmod = _load(cash, work, helper_body)

    with pytest.raises(Exception):
        inspect.getsource(dynmod.helper)  # premise: source really is gone

    before = usermod.compute.cache_info()["hits"]
    value = usermod.compute(3)
    return usermod.compute.cache_info()["hits"] > before, value


def test_identical_helper_still_hits(env):
    """Null control. Without it the tests below prove nothing."""
    cash, work = env
    hit, _ = _call(cash, work, ALPHA)
    assert hit is False, "first call must compute"
    hit, value = _call(cash, work, ALPHA)
    assert hit is True and value == "alpha"


def test_changed_constant_recomputes(env):
    cash, work = env
    _call(cash, work, ALPHA)
    hit, value = _call(cash, work, 'def helper(x):\n    return "omega"\n')
    assert hit is False, "an edited helper must not serve the old result"
    assert value == "omega"


def test_replaced_body_recomputes(env):
    """The decisive case: this is not about constants at all."""
    cash, work = env
    _call(cash, work, ALPHA)
    body = "def helper(x):\n    t = 0\n    for i in range(x):\n        t += i * 7\n    return t\n"
    hit, value = _call(cash, work, body)
    assert hit is False
    assert value == 21


def test_helper_is_recorded_rather_than_dropped(env):
    """Names the mechanism, so a regression says WHY it broke."""
    cash, work = env
    usermod, _ = _load(cash, work, ALPHA)
    usermod.compute(3)
    report = cash._purity_reports[cash._get_func_key(usermod.compute)]

    assert "dynmod.helper" in report.opaque_callees, (
        "still opaque for PURITY -- we cannot read what it does"
    )
    assert "dynmod.helper" in report.helper_source_hashes, (
        "but it must contribute to the CACHE KEY, which is the bug this pins"
    )
    assert "dynmod.helper" in report.helper_resolution_paths, (
        "and be re-resolvable, or the per-call rehash falls back to the snapshot"
    )
