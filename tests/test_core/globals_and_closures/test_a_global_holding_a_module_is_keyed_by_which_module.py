"""A name that holds a module is keyed by which module it holds.

librosa's ``stft`` calls ``get_fftlib()``, which returns the module global
``__FFTLIB``; a commit switched it from ``numpy.fft`` to ``scipy.fft`` with
neither function's code changing, and the next run served the result
computed with the old module. A module was left out of the key wherever a
name could hold one: a module global read by name or through a helper, an
attribute of the user's module or of a library module, an import in the
body, an attribute stored on a function, a helper's closure, a table.
Each is keyed by the module's name now, the same in every process.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import textwrap
import types

from cash.decorator.module_identity import module_ref
from tests._scripts import run_python

# The same source in both runs: only which library module ``LIB`` holds
# differs, as ``numpy.fft`` and ``scipy.fft`` did.
BACKENDS = textwrap.dedent("""
    import cmath
    import math
    import os

    LIB = cmath if os.environ.get("USE_CMATH") else math

    def get_lib():
        return LIB
""")

JOB = textwrap.dedent("""
    import xml
    import cash
    import backends
    from backends import LIB, get_lib

    xml.LIB = backends.LIB

    def holder():
        pass

    holder.lib = backends.LIB

    def make(mod):
        def encode(x):
            return mod.sqrt(x)
        return encode

    encode = make(backends.LIB)
    TABLE = {"enc": backends.LIB}

    @cash.cache
    def via_helper(x):
        return get_lib().sqrt(x)

    @cash.cache
    def direct(x):
        return LIB.sqrt(x)

    @cash.cache
    def user_module_attr(x):
        return backends.LIB.sqrt(x)

    @cash.cache
    def user_module_helper(x):
        return backends.get_lib().sqrt(x)

    @cash.cache
    def local_import(x):
        from backends import LIB as lib
        return lib.sqrt(x)

    @cash.cache
    def local_module_attr(x):
        import backends as b
        return b.LIB.sqrt(x)

    @cash.cache
    def library_attr(x):
        return xml.LIB.sqrt(x)

    @cash.cache
    def function_attr(x):
        return holder.lib.sqrt(x)

    @cash.cache
    def helper_closure(x):
        return encode(x)

    @cash.cache
    def through_value(x):
        return TABLE["enc"].sqrt(x)

    fns = [via_helper, direct, user_module_attr, user_module_helper, local_import, local_module_attr,
           library_attr, function_attr, helper_closure, through_value]
    print("|".join(str(f(4)) for f in fns))
    print(sum(f.cache_info()["hits"] for f in fns))
""")


def _run(tmp_path, plain):
    out = run_python("job.py", cwd=tmp_path, env={"USE_CMATH": "1" if plain else None}).stdout.split()
    return out[0].split("|"), int(out[1])


def _write(tmp_path):
    (tmp_path / "job.py").write_text(JOB, encoding="utf-8")
    (tmp_path / "backends.py").write_text(BACKENDS, encoding="utf-8")


def test_switching_the_module_a_global_holds_recomputes_in_the_next_run(tmp_path):
    _write(tmp_path)
    first, _ = _run(tmp_path, False)
    assert first == ["2.0"] * 10
    second, hits = _run(tmp_path, True)
    assert second == ["(2+0j)"] * 10
    assert hits == 0


def test_unchanged_code_hits_in_a_fresh_process(tmp_path):
    _write(tmp_path)
    _run(tmp_path, False)
    out, hits = _run(tmp_path, False)
    assert out == ["2.0"] * 10
    assert hits == 10


REBOUND = textwrap.dedent("""
    import cmath
    import math
    import cash
    import backends
    from backends import get_lib

    @cash.cache
    def via_helper(x):
        return get_lib().sqrt(x)

    print(via_helper(4))
    backends.LIB = cmath
    print(via_helper(4), via_helper.cache_info()["hits"])
    backends.LIB = math
    print(via_helper(4), via_helper.cache_info()["hits"])
    via_helper(9)
    backends.LIB = cmath
    print(via_helper.explain(9).details["changed"])
""")


def test_rebinding_the_module_in_this_process_recomputes_and_names_the_global(tmp_path):
    _write(tmp_path)
    (tmp_path / "job.py").write_text(REBOUND, encoding="utf-8")
    lines = run_python("job.py", cwd=tmp_path).stdout.splitlines()
    assert lines[:3] == ["2.0", "(2+0j) 0", "2.0 1"]
    assert lines[3] == "global LIB (read by get_lib) changed"


def test_a_lazy_module_is_keyed_by_its_name_without_loading_it(tmp_path, monkeypatch):
    (tmp_path / "lazy_target.py").write_text("LOADED = True\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = importlib.util.find_spec("lazy_target")
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "lazy_target", module)
    loader.exec_module(module)
    before = module_ref(module)
    assert type(module) is not types.ModuleType, "keying it must not load it"
    assert module.LOADED
    assert type(module) is types.ModuleType
    assert before == module_ref(module) == "lazy_target"


def test_a_module_that_renames_itself_keeps_its_import_name(monkeypatch):
    module = types.ModuleType("pkg_renaming")
    module.__spec__ = importlib.machinery.ModuleSpec("pkg_renaming", None)
    monkeypatch.setitem(sys.modules, "pkg_renaming", module)
    before = module_ref(module)
    module.__name__ = "pkg_renaming._impl"
    assert before == module_ref(module) == "pkg_renaming"


def test_a_module_not_importable_by_name_is_keyed_by_name_and_file(tmp_path):
    for folder in ("a", "b"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "conf.py").write_text("X = 1\n", encoding="utf-8")
    refs = []
    for folder in ("a", "b"):
        spec = importlib.util.spec_from_file_location("conf", tmp_path / folder / "conf.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        refs.append(module_ref(module))
    assert refs == [f"conf@{tmp_path / 'a' / 'conf.py'}", f"conf@{tmp_path / 'b' / 'conf.py'}"]
    # Made in memory: its name, never something of this process such as its id.
    assert module_ref(types.ModuleType("made_here")) == "made_here"
