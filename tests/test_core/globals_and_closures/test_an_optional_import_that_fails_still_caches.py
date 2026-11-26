"""A function whose optional import fails still caches, and installing the
module recomputes.

librosa reads ``resampy = lazy_loader.load("resampy")``: with resampy not
installed, the name holds a module that raises ``ModuleNotFoundError`` on any
attribute read. The helper walk and the key read ``resampy.resample`` to
find the code a call reaches, the error escaped them, and ``resample``,
``load``, ``pitch_shift`` and ``cqt`` ran uncached in every run
(KEY-HELPERS-UNWALKABLE, then KEY-BUILD-FAILED). An attribute of a module
whose import failed is not there, to cash as to the code; which module the
name holds is keyed (``module_ref``), and so is the library's version once
it is installed, so installing it moves the key.

The run after installing it stores again, and the run after that hits: a
body that imports a module of the user's not imported yet (``import
optmod_inner``) never hit in a fresh process, since its trace named code
of a module the next process had not loaded.
"""

from __future__ import annotations

import textwrap

from tests._scripts import run_python

MYLIB = textwrap.dedent("""
    import importlib.util
    import sys
    import types


    class _Missing(types.ModuleType):
        # What lazy_loader.load gives for a module that is not installed.
        def __init__(self, name):
            super().__init__("DelayedImportErrorModule")
            self.__missing = name

        def __getattr__(self, attr):
            raise ModuleNotFoundError(f"No module named {self.__missing!r}")


    def lazy_load(name):
        spec = importlib.util.find_spec(name)
        if spec is None:
            return _Missing(name)
        loader = importlib.util.LazyLoader(spec.loader)
        spec.loader = loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        loader.exec_module(module)
        return module


    try:
        import optmod_eager
    except ImportError:
        optmod_eager = None

    optmod_lazy = lazy_load("optmod_lazy")


    def guarded_local_import(x):
        try:
            import optmod_inner
        except ImportError:
            return x + 1
        return optmod_inner.go(x)


    def lazy_stub(x):
        try:
            return optmod_lazy.go(x)
        except ImportError:
            return x + 2


    def eager_none(x):
        if optmod_eager is None:
            return x + 3
        return optmod_eager.go(x)
""")

JOB = textwrap.dedent("""
    import os
    import sys
    import warnings

    if os.environ.get("INSTALL"):
        sys.path.insert(0, os.path.join(os.getcwd(), "site"))

    import cash
    import mylib

    names = ["guarded_local_import", "lazy_stub", "eager_none"]
    fns = [cash.cache(getattr(mylib, n)) for n in names]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        print(" ".join(str(f(1)) for f in fns))
    print(" ".join(str(f.cache_info()["hits"]) for f in fns))
    print(sorted({str(w.message).split("]")[0].lstrip("[") for w in caught if "KEY-" in str(w.message)}))
""")


def _write(tmp_path):
    (tmp_path / "mylib.py").write_text(MYLIB, encoding="utf-8")
    (tmp_path / "job.py").write_text(JOB, encoding="utf-8")
    site = tmp_path / "site"
    site.mkdir()
    for name in ("optmod_inner", "optmod_lazy", "optmod_eager"):
        (site / f"{name}.py").write_text("def go(x):\n    return x * 100\n", encoding="utf-8")


def _run(tmp_path, installed):
    lines = run_python("job.py", cwd=tmp_path, env={"INSTALL": "1" if installed else None}).stdout.splitlines()
    return lines[0].split(), lines[1].split(), lines[2]


def test_a_failed_optional_import_caches_and_hits_in_the_next_run(tmp_path):
    _write(tmp_path)
    results, hits, warned = _run(tmp_path, installed=False)
    assert results == ["2", "3", "4"]
    assert warned == "[]", "nothing about the missing module stops the key"
    results, hits, _ = _run(tmp_path, installed=False)
    assert results == ["2", "3", "4"]
    assert hits == ["1", "1", "1"]


def test_installing_the_module_recomputes_then_hits_again(tmp_path):
    _write(tmp_path)
    _run(tmp_path, installed=False)
    results, hits, _ = _run(tmp_path, installed=True)
    assert results == ["100", "100", "100"], "served the result computed without the module"
    assert hits == ["0", "0", "0"]
    results, hits, _ = _run(tmp_path, installed=True)
    assert results == ["100", "100", "100"]
    assert hits == ["1", "1", "1"]


LOCAL_IMPORT = textwrap.dedent("""
    import cash

    @cash.cache
    def g(x):
        import helpmod
        return helpmod.go(x)

    print(g(1), g.cache_info()["hits"])
""")


def test_a_body_importing_a_module_not_loaded_yet_hits_in_a_fresh_process(tmp_path):
    (tmp_path / "helpmod.py").write_text("def go(x):\n    return x * 100\n", encoding="utf-8")
    (tmp_path / "job.py").write_text(LOCAL_IMPORT, encoding="utf-8")
    assert run_python("job.py", cwd=tmp_path).stdout.split() == ["100", "0"]
    assert run_python("job.py", cwd=tmp_path).stdout.split() == ["100", "1"]
    (tmp_path / "helpmod.py").write_text("def go(x):\n    return x * 7\n", encoding="utf-8")
    assert run_python("job.py", cwd=tmp_path).stdout.split() == ["7", "0"]
