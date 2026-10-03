"""An edited module whose reload raises is reported, and stays owed.

``importlib.reload`` of a file that no longer compiles raises and leaves the
module holding its old code. The tracker must say so on every check until
the file loads again, not take the edit as seen after the first try, and
must not reload the modules that import the broken one over its old code.
"""

import importlib
import os
import sys

import pytest

from cash.tracking.function_tracker import FunctionTracker

GOOD = "def weekly(n):\n    return n - 7\n"
BROKEN = "def weekly(n):\n    return n - (7\n"
FIXED = "def weekly(n):\n    return n - 9\n"


def _write(path, text, bump):
    """Write *text* and move the mtime on, so the edit is seen whatever the
    filesystem's timestamp resolution."""
    path.write_text(text, encoding="utf-8")
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + bump * 1_000_000_000))


@pytest.fixture
def tracker():
    ft = FunctionTracker()
    yield ft
    ft.clear()


@pytest.fixture
def module_dir(tmp_path):
    sys.path.insert(0, str(tmp_path))
    names = []
    yield tmp_path, names
    sys.path.remove(str(tmp_path))
    for name in names:
        sys.modules.pop(name, None)


def test_a_syntax_error_is_reported_on_every_check_until_the_file_loads(tracker, module_dir):
    tmp_path, names = module_dir
    name = f"_replen_{id(tmp_path)}"
    names.append(name)
    path = tmp_path / f"{name}.py"
    _write(path, GOOD, 0)
    mod = importlib.import_module(name)
    tracker.track_module(name)

    _write(path, BROKEN, 2)
    reloaded, _ = tracker.check_and_reload_changed_modules({})
    assert reloaded == {}
    errors = tracker.reload_errors()
    assert list(errors) == [name]
    assert isinstance(errors[name], SyntaxError)
    assert errors[name].lineno == 2
    assert mod.weekly(10) == 3  # the kernel still holds the old code

    # Nothing changed on disk since, but the module still does not load.
    reloaded, _ = tracker.check_and_reload_changed_modules({})
    assert reloaded == {}
    assert isinstance(tracker.reload_errors()[name], SyntaxError)

    _write(path, FIXED, 4)
    reloaded, _ = tracker.check_and_reload_changed_modules({})
    assert name in reloaded
    assert tracker.reload_errors() == {}
    assert sys.modules[name].weekly(10) == 1

    # Loaded: nothing is owed any more.
    reloaded, _ = tracker.check_and_reload_changed_modules({})
    assert reloaded == {}
    assert tracker.reload_errors() == {}


def test_a_top_level_error_is_reported_like_a_syntax_error(tracker, module_dir):
    tmp_path, names = module_dir
    name = f"_replen_top_{id(tmp_path)}"
    names.append(name)
    path = tmp_path / f"{name}.py"
    _write(path, GOOD, 0)
    importlib.import_module(name)
    tracker.track_module(name)

    _write(path, "import _no_such_module_for_cash_tests\n" + GOOD, 2)
    tracker.check_and_reload_changed_modules({})
    assert isinstance(tracker.reload_errors()[name], ImportError)


def test_a_module_importing_the_broken_one_is_not_reloaded_over_its_old_code(tracker, module_dir):
    tmp_path, names = module_dir
    dep = f"_dep_{id(tmp_path)}"
    top = f"_top_{id(tmp_path)}"
    names.extend([dep, top])
    dep_path = tmp_path / f"{dep}.py"
    top_path = tmp_path / f"{top}.py"
    _write(dep_path, GOOD, 0)
    _write(top_path, f"from {dep} import weekly\n\ndef run(n):\n    return weekly(n)\n", 0)
    importlib.import_module(top)
    tracker.track_module(top)
    tracker.track_module(dep)

    _write(dep_path, BROKEN, 2)
    reloaded, _ = tracker.check_and_reload_changed_modules({})
    assert top not in reloaded
    assert list(tracker.reload_errors()) == [dep]

    _write(dep_path, FIXED, 4)
    reloaded, _ = tracker.check_and_reload_changed_modules({})
    assert {dep, top} <= set(reloaded)
    assert sys.modules[top].run(10) == 1
