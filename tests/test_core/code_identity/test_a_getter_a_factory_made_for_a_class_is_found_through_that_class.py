"""A getter a factory made for a class is found through that class.

Reported against scanpy checked out as the user's code: its settings class
gets its ``cached_property`` attributes from a factory
(``preset_property(...)`` returns ``cached_property(get)``, ``get`` a
closure over the factory's table), and the class is renamed to the package
that exports it (``@set_module("scanpy")``). A traced call that ran the
getter was refused: no name holds a closure, and the class holding it was
skipped for naming another module. The class is the getter's holder when
its own name finds it again; its digest keys the getter by code and closure,
so an edit to the table is seen.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [
    pytest.mark.core,
    pytest.mark.timeout(300),
]

CONF = textwrap.dedent("""
    from functools import cached_property


    def preset_property(table):
        def get(self):
            return table[self.name]

        return cached_property(get)


    class Preset:
        flavor = preset_property({{"a": {a}, "b": 7}})

        def __init__(self, name):
            self.name = name


    Preset.__module__ = "cfgpkg"
""")

OPS = textwrap.dedent("""
    import operator

    from cfgpkg.conf import Preset


    def run(xs, how):
        return [getattr(operator, how)(x, Preset("a").flavor) for x in xs]
""")

MAIN = textwrap.dedent("""
    import sys
    import cash
    from cfgpkg import ops


    @cash.cache
    def scaled(xs, how):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        return ops.run(xs, how)


    print(scaled([1, 2], "mul"))
""")


def _run(tmp_path, a):
    pkg = tmp_path / "cfgpkg"
    pkg.mkdir(exist_ok=True)
    files = {"__init__.py": "from .conf import Preset\n", "conf.py": CONF.format(a=a), "ops.py": OPS}
    for name, text in files.items():
        (pkg / name).write_text(text, encoding="utf-8")
    (tmp_path / "main.py").write_text(MAIN, encoding="utf-8")
    proc = run_python("main.py", cwd=tmp_path)
    return proc.stdout.strip(), proc.stderr


def test_the_call_is_stored_and_an_edit_to_the_factory_s_table_is_seen(tmp_path):
    out, err = _run(tmp_path, 3)
    assert out == "[3, 6]" and "[RUN]" in err
    out, err = _run(tmp_path, 3)
    assert out == "[3, 6]" and "[RUN]" not in err, f"an unchanged second process must hit:\n{err}"
    out, err = _run(tmp_path, 5)
    assert out == "[5, 10]" and "[RUN]" in err, f"served the result from before the edit:\n{err}"
