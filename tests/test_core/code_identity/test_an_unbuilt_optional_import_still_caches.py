"""A function that imports a module no finder knows, in a branch that does
not run, still caches, and so do its callers.

pvlib's ``spa_c`` reads ``from pvlib.spa_c_files.spa_py import ...`` inside
``try``/``except ImportError``; only ``spa_py.pyx`` ships, unbuilt. The
module had no file to read, so its digest was None and the function and
every caller ran uncached in every process, reported as code picked by a
run-time name. A module no finder knows is now keyed by the files that
could become it (an unbuilt ``.pyx``, a build's output), and a package
already loaded on the way to it by its file alone, not by what its top
level binds. Creating the module, editing the ``.pyx`` a finder installed
during the call would load, or editing the package's ``__init__`` all move
the key.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

SPA = """
    from solpkg.tools import helper

    def spa_c(x):
        try:
            from solpkg.cfiles.spa_py import fast
        except ImportError:
            raise
        return fast(x)

    def solar(x, how="python"):
        if how == "c":
            return spa_c(x)
        return helper(x)

    def outer(x):
        return solar(x) + 1
"""

JOB = """
    import os, sys, warnings
    sys.path.insert(0, os.getcwd())
    import cash
    import solpkg.spa as spa

    how = sys.argv[1] if len(sys.argv) > 1 else "python"
    solar, outer = cash.cache(spa.solar), cash.cache(spa.outer)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        print(solar(3, how), outer(3), solar.cache_info()["hits"], outer.cache_info()["hits"])
    print(sorted({str(w.message).split("]")[0].lstrip("[") for w in caught if "KEY-" in str(w.message)}))
"""


def _write(proj, name, text):
    path = proj / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")


def _project(proj, init_pkg=True):
    if init_pkg:
        # The package's top level reads the environment: not cash's to vouch
        # for, and loaded before the import below it.
        _write(
            proj,
            "solpkg/__init__.py",
            "import os\nMODE = os.environ.get('SOLPKG_MODE', 'a')\nfrom solpkg import tools\n",
        )
    _write(proj, "solpkg/tools.py", "def helper(x):\n    return x * 2\n")
    _write(proj, "solpkg/cfiles/__init__.py", "")
    _write(proj, "solpkg/cfiles/spa_py.pyx", "def fast(x):\n    return x * 1000\n")
    _write(proj, "solpkg/spa.py", SPA)
    _write(proj, "job.py", JOB)


def _run(proj, *args):
    return run_python("job.py", *args, cwd=proj).stdout.splitlines()


def test_the_next_process_hits_and_so_does_a_caller(tmp_path):
    _project(tmp_path)
    assert _run(tmp_path) == ["6 7 0 0", "[]"]
    assert _run(tmp_path) == ["6 7 1 1", "[]"], "the unbuilt module's import kept the call uncached"


def test_creating_the_missing_module_recomputes(tmp_path):
    _project(tmp_path)
    _run(tmp_path)
    _write(tmp_path, "solpkg/cfiles/spa_py.py", "def fast(x):\n    return x * 1000\n")
    assert _run(tmp_path, "c")[0].split()[0] == "3000"
    assert _run(tmp_path)[0] == "6 7 0 0", "served the entry stored while the module was absent"
    assert _run(tmp_path)[0] == "6 7 1 1"


def test_a_build_beside_the_pyx_or_an_edit_to_it_recomputes(tmp_path):
    _project(tmp_path)
    _run(tmp_path)
    assert _run(tmp_path)[0] == "6 7 1 1"
    # A finder installed during a call (pyximport) would load the .pyx: its
    # bytes stand for the module until then.
    _write(tmp_path, "solpkg/cfiles/spa_py.pyx", "def fast(x):\n    return x * 7\n")
    assert _run(tmp_path)[0] == "6 7 0 0"
    assert _run(tmp_path)[0] == "6 7 1 1"
    _write(tmp_path, "solpkg/cfiles/spa_py.c", "/* generated */\n")
    assert _run(tmp_path)[0] == "6 7 0 0"


def test_editing_the_loaded_package_init_recomputes(tmp_path):
    _project(tmp_path)
    _run(tmp_path)
    assert _run(tmp_path)[0] == "6 7 1 1"
    _write(tmp_path, "solpkg/__init__.py", "import os\nMODE = 'b'\nfrom solpkg import tools\n")
    assert _run(tmp_path)[0] == "6 7 0 0"
    assert _run(tmp_path)[0] == "6 7 1 1"


def test_under_a_namespace_package_it_caches_and_creating_it_recomputes(tmp_path):
    _project(tmp_path, init_pkg=False)
    assert _run(tmp_path) == ["6 7 0 0", "[]"]
    assert _run(tmp_path)[0] == "6 7 1 1"
    _write(tmp_path, "solpkg/cfiles/spa_py.py", "def fast(x):\n    return x * 1000\n")
    assert _run(tmp_path)[0] == "6 7 0 0"


def test_a_finder_installed_during_the_call_does_not_serve_a_stale_result(tmp_path):
    """The body installs a finder that loads ``.pyx`` files as Python, as
    ``pyximport.install()`` does, and then imports the module."""
    _project(tmp_path)
    _write(
        tmp_path,
        "solpkg/spa.py",
        SPA.replace(
            "        try:\n",
            "        import pyxfinder\n        pyxfinder.install()\n        try:\n",
        ),
    )
    _write(
        tmp_path,
        "pyxfinder.py",
        """
        import importlib.machinery, sys

        class _Pyx(importlib.machinery.PathFinder):
            @classmethod
            def find_spec(cls, name, path=None, target=None):
                for entry in path or ():
                    import os
                    file = os.path.join(entry, name.rpartition(".")[2] + ".pyx")
                    if os.path.exists(file):
                        loader = importlib.machinery.SourceFileLoader(name, file)
                        return importlib.machinery.ModuleSpec(name, loader, origin=file)
                return None

        def install():
            if _Pyx not in sys.meta_path:
                sys.meta_path.append(_Pyx)
        """,
    )
    assert _run(tmp_path, "c")[0].split()[:1] == ["3000"]
    assert _run(tmp_path, "c")[0].split()[:1] == ["3000"]
    _write(tmp_path, "solpkg/cfiles/spa_py.pyx", "def fast(x):\n    return x * 7\n")
    assert _run(tmp_path, "c")[0].split()[:1] == ["21"], "served the result of the .pyx before the edit"


def test_a_module_cash_cannot_read_is_named_as_such_in_explain(tmp_path):
    _write(tmp_path, "envmod.py", "import os\nK = int(os.environ.get('K', '1'))\n")
    _write(
        tmp_path,
        "job.py",
        """
        import os, sys, warnings
        sys.path.insert(0, os.getcwd())
        warnings.simplefilter("ignore")
        import cash

        @cash.cache
        def f(x, use):
            if use:
                from envmod import K
                return x * K
            return x

        f(1, False)
        print(f.explain(1, False))
        """,
    )
    out = run_python("job.py", cwd=tmp_path).stdout
    assert "imports a module cash cannot read before it is loaded" in out
    assert "run-time name" not in out
