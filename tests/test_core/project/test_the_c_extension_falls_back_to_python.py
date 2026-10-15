"""cash runs without its C extension, and says why the fast path is off.

``cash._cwatch`` is optional: the pure-Python wheel, a source build without
a compiler, Python before 3.12, ``CASH_PURE_PYTHON=1``, a module that does not
load, one from another build, or one that reports itself unusable all leave
cash on its pure-Python path. ``%cash_status`` carries one line saying which.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

import pytest

from cash import _native
from tests._scripts import run_python

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE = "cash._cwatch"

needs_312 = pytest.mark.skipif(sys.version_info < (3, 12), reason="the extension is only tried on Python 3.12+")


@pytest.fixture
def no_loaded_module(monkeypatch):
    """Forget any ``cash._cwatch`` this process loaded, for the test's length."""
    import cash

    monkeypatch.delitem(sys.modules, MODULE, raising=False)
    monkeypatch.delattr(cash, "_cwatch", raising=False)
    monkeypatch.delenv("CASH_PURE_PYTHON", raising=False)
    return monkeypatch


def _fake(**attrs) -> types.ModuleType:
    module = types.ModuleType(MODULE)
    module.__dict__.update(attrs)
    return module


@needs_312
def test_a_matching_module_turns_the_fast_path_on(no_loaded_module):
    no_loaded_module.setitem(sys.modules, MODULE, _fake(API_VERSION=_native.EXPECTED_API))
    module, reason = _native._load()
    assert module is sys.modules[MODULE] and reason is None


@needs_312
def test_a_missing_module_means_an_install_without_it(no_loaded_module):
    no_loaded_module.setitem(sys.modules, MODULE, None)  # import raises ModuleNotFoundError
    module, reason = _native._load()
    assert module is None
    assert "no C extension" in reason


@needs_312
def test_a_module_that_fails_to_load_is_reported_with_the_error(no_loaded_module):
    class Broken(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == MODULE:
                raise ImportError("undefined symbol: PyDict_AddWatcher")
            return None

    no_loaded_module.setattr(sys, "meta_path", [Broken(), *sys.meta_path])
    module, reason = _native._load()
    assert module is None
    assert "does not load" in reason and "undefined symbol" in reason


@needs_312
def test_a_module_from_another_build_is_not_used(no_loaded_module):
    no_loaded_module.setitem(sys.modules, MODULE, _fake(API_VERSION=_native.EXPECTED_API + 1))
    module, reason = _native._load()
    assert module is None
    assert f"API {_native.EXPECTED_API + 1}" in reason


@needs_312
def test_a_module_that_reports_itself_unusable_is_not_used(no_loaded_module):
    fake = _fake(API_VERSION=_native.EXPECTED_API, USABLE=False, UNUSABLE_REASON="free-threaded build")
    no_loaded_module.setitem(sys.modules, MODULE, fake)
    assert _native._load() == (None, "free-threaded build")


def test_the_switch_keeps_the_module_unloaded(no_loaded_module):
    no_loaded_module.setitem(sys.modules, MODULE, _fake(API_VERSION=_native.EXPECTED_API))
    no_loaded_module.setenv("CASH_PURE_PYTHON", "1")
    assert _native._load() == (None, "CASH_PURE_PYTHON is set")


@pytest.mark.skipif(sys.version_info >= (3, 12), reason="Python 3.12+ tries the extension")
def test_before_python_3_12_the_extension_is_not_tried(no_loaded_module):
    no_loaded_module.setitem(sys.modules, MODULE, _fake(API_VERSION=_native.EXPECTED_API))
    module, reason = _native._load()
    assert module is None and "3.12" in reason


def test_cash_status_shows_the_fast_path_line(cash_magics):
    line = cash_magics.cash_status("dict")["fast_path"]
    assert line == _native.status_line()
    assert line.startswith("on (") if _native.AVAILABLE else line == f"off: {_native.REASON}"


def test_a_fresh_interpreter_with_the_switch_imports_cash_on_the_pure_path(tmp_path):
    done = run_python(
        "-c",
        "import cash, cash._native as n, sys; print(n.status_line()); print('cash._cwatch' in sys.modules)",
        cwd=tmp_path,
        env={"CASH_PURE_PYTHON": "1"},
    )
    assert done.stdout.split() == ["off:", "CASH_PURE_PYTHON", "is", "set", "False"]


@needs_312
@pytest.mark.slow
@pytest.mark.timeout(120)
def test_the_compiled_extension_loads_and_turns_the_fast_path_on(no_loaded_module, tmp_path):
    """Positive control for the tests above: the real module, compiled from
    src/cash/_c/ by the build hook, loads and passes every check."""
    pytest.importorskip("setuptools")
    spec = importlib.util.spec_from_file_location("hatch_build", REPO_ROOT / "hatch_build.py")
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    if hook.skip_reason():
        pytest.skip(hook.skip_reason())
    built = hook.compile_extension(tmp_path / "lib", tmp_path / "temp")

    loader = importlib.machinery.ExtensionFileLoader(MODULE, str(built))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(MODULE, loader))
    loader.exec_module(module)
    no_loaded_module.setitem(sys.modules, MODULE, module)
    loaded, reason = _native._load()
    assert reason is None or reason == getattr(module, "UNUSABLE_REASON", None)
    assert loaded is module or getattr(module, "USABLE", True) is False


def test_cash_info_shows_the_fast_path_line(tmp_path):
    done = run_python("-m", "cash", "info", cwd=tmp_path, env={"CASH_PURE_PYTHON": "1"})
    assert "  Fast path:  off: CASH_PURE_PYTHON is set" in done.stdout.splitlines(), done.stdout
