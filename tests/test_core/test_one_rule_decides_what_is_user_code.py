"""One rule decides what is the user's code, for every part of cash.

`cash.install_paths` holds the verdicts; the decorator's folds, the helper
walk and the notebook's writer walk all ask it. These pin where they used
to answer differently.
"""

from __future__ import annotations

import linecache
import sys
import types

from cash.analysis.namespace_effects import user_callee_writing_files
from cash.analysis.purity_analyzer import own_code_is_user
from cash.decorator.user_code import is_user_class
from cash.install_paths import clear_caches

# The verdicts are imported inside each test: `scripts/fails_first.py` runs
# these against code that does not have them yet.


def _function_in(path: str, body: str, name: str, module: str = "vendored_writer"):
    namespace: dict = {"__name__": module}
    exec(compile(body, path, "exec"), namespace)
    return namespace[name]


def test_a_package_directory_outside_the_known_roots_is_not_user_code(tmp_path):
    site = tmp_path / "opt" / "lib" / "site-packages" / "vendored"
    site.mkdir(parents=True)
    path = site / "writer.py"
    body = "def save(fig, name):\n    fig.savefig(name)\n"
    path.write_text(body, encoding="utf-8")
    clear_caches()
    save = _function_in(str(path), body, "save")
    from cash.install_paths import is_user_code_file

    assert not is_user_code_file(str(path))
    assert user_callee_writing_files(save) is None, "the writer walk read an installed package's code"


def test_a_notebook_cell_function_is_user_code():
    body = "def export(fig, path):\n    fig.savefig(path)\n"
    cell = "<ipython-input-3-abc>"
    from cash.install_paths import is_user_code_file

    # IPython puts a cell's source in linecache, where getsource finds it.
    linecache.cache[cell] = (len(body), None, body.splitlines(True), cell)
    export = _function_in(cell, body, "export", module="__main__")
    assert is_user_code_file(cell)
    assert user_callee_writing_files(export) == "export"


def test_cash_itself_is_never_the_users_code():
    import cash.tracking.reader_patches as shim
    from cash.install_paths import is_user_module

    assert not is_user_module(shim)
    assert not own_code_is_user(shim._is_user_module, "__main__")


def test_the_own_package_shortcut_has_one_spelling():
    from cash.install_paths import in_own_package

    assert in_own_package("pkg.sub", "pkg")
    assert in_own_package("pkg", "pkg")
    assert not in_own_package("pkgx", "pkg")
    assert in_own_package("__main__", "__main__")
    assert not in_own_package("other", "__main__")


def test_a_class_in_a_module_with_no_file_is_a_user_class():
    cell = types.ModuleType("cell_module_without_file")
    exec("class Cfg:\n    LIMIT = 3\n", cell.__dict__)
    sys.modules[cell.__name__] = cell
    from cash.install_paths import is_user_code_module, is_user_module

    try:
        assert is_user_code_module(cell)
        assert not is_user_module(cell)
        assert is_user_class(cell.Cfg)
    finally:
        del sys.modules[cell.__name__]
