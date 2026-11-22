"""A value that holds a module's ``globals()`` is keyed as that module.

MetPy marks each function of ``metpy.calc`` with ``@exporter.export``,
where ``exporter = Exporter(globals())`` keeps the module's namespace.
Every function read ``exporter``; walked as data, the dict led to every
other function of the module, each read ``exporter`` again, one function
deeper each time: caching ``parcel_profile`` spent 8-22 s per call on the
key and then failed it with RecursionError (KEY-BUILD-FAILED). The dict is
the module held another way, keyed as a module held by a name is.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.core]

EXPORTER = """
    import functools


    class Exporter:
        def __init__(self, globls):
            self.globls = globls
            self.exports = globls.setdefault("__all__", [])

        def export(self, defn):
            self.exports.append(defn.__name__)
            return defn


    def wrap(func):
        # MetPy's `preprocess_and_wrap`, as far as the key is concerned.
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper
"""

#: As many functions as `metpy.calc.thermo` exports, each calling the one before.
FUNCTIONS = 120


def _thermo(scale: str) -> str:
    lines = ["from exporter import Exporter, wrap", "", "exporter = Exporter(globals())", f"SCALE = {scale}", ""]
    lines += ["@exporter.export", "@wrap", "def f0(x):", "    return x * SCALE", ""]
    for i in range(1, FUNCTIONS):
        lines += ["@exporter.export", "@wrap", f"def f{i}(x):", f"    return f{i - 1}(x) + 1", ""]
    return "\n".join(lines)


MAIN = """
    import sys
    import warnings

    import cash
    import thermo

    warnings.simplefilter("always")


    @cash.cache
    def profile(x):
        return thermo.f119(x)


    with warnings.catch_warnings(record=True) as caught:
        print(profile(float(sys.argv[1])), thermo.f119(float(sys.argv[1])))
    info = profile.cache_info()
    # Below Python 3.12 the first call a process stores says this once.
    unseen = "KEY-LIBRARY-STATE-UNSEEN" if sys.version_info < (3, 12) else ""
    codes = sorted({str(getattr(w.message, "code", "")) for w in caught} - {"", unseen})
    print(info["hits"], info["misses"], codes)
"""


def _project(folder: Path, scale: str) -> None:
    folder.mkdir(exist_ok=True)
    (folder / "exporter.py").write_text(textwrap.dedent(EXPORTER), encoding="utf-8")
    (folder / "thermo.py").write_text(_thermo(scale), encoding="utf-8")
    (folder / "main.py").write_text(textwrap.dedent(MAIN), encoding="utf-8")


def _run(folder: Path, x: str) -> list[str]:
    return run_python("main.py", x, cwd=folder).stdout.strip().splitlines()[-2:]


@pytest.mark.timeout(240)
def test_a_module_exporting_its_functions_keys_and_hits_in_a_new_process(tmp_path):
    project = tmp_path / "proj"
    _project(project, "2.0")
    # The result is plain Python's, and the key builds: no KEY-BUILD-FAILED.
    assert _run(project, "1.0") == ["121.0 121.0", "0 1 []"]
    assert _run(project, "1.0") == ["121.0 121.0", "1 0 []"]
    # Another argument is another call.
    assert _run(project, "3.0") == ["125.0 125.0", "0 1 []"]
    # The functions the call runs are keyed: editing what they read misses.
    _project(project, "5.0")
    assert _run(project, "1.0") == ["124.0 124.0", "0 1 []"]
    assert _run(project, "1.0") == ["124.0 124.0", "1 0 []"]


def test_a_module_namespace_held_in_a_value_is_named_by_its_module():
    from cash.decorator.key_values import stabilize_for_global_hash
    from cash.decorator.module_identity import namespace_module

    module = pytest.importorskip("json")
    assert namespace_module(vars(module)) is module
    # A copy of the namespace, or any other dict, is data.
    assert namespace_module(dict(vars(module))) is None
    assert namespace_module({"__name__": "json"}) is None

    class Holder:
        def __init__(self, globls):
            self.globls = globls

    form = stabilize_for_global_hash(Holder(vars(module)), repr)
    assert form[2] == {"globls": ("__cash_module__", "json")}
