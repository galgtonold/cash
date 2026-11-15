"""A traced call's read through an object's data is checked by that value.

Reported against scanpy checked out as the user's code: ``leiden``'s
``_validate_flavor`` does ``from scanpy import settings`` and reads
``settings.preset.leiden.flavor``. A call that picks code by name is served
on a trace of what it ran, and the trace checked such a read by the whole
object past the module, which holds a logger and so could not be checked:
the call was never stored. The attributes the code read, where the object
holds them as data (its ``__dict__``, a filled ``cached_property``, a
``NamedTuple`` field), are what the read took, and are what is checked.
"""

from __future__ import annotations

import sys
import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [
    pytest.mark.core,
    pytest.mark.timeout(300),
    pytest.mark.skipif(sys.version_info < (3, 12), reason="a trace of what a call ran needs sys.monitoring"),
]

CONF = textwrap.dedent("""
    import threading
    from functools import cached_property
    from typing import NamedTuple


    class LeidenPreset(NamedTuple):
        flavor: str


    class Preset:
        def __init__(self, flavor):
            self.flavor = flavor

        @cached_property
        def leiden(self):
            return LeidenPreset(self.flavor)


    class Settings:
        def __init__(self):
            self.preset = Preset("igraph")
            self._lock = threading.Lock()  # the object as a whole cannot be checked


    settings = Settings()
""")

OPS = textwrap.dedent("""
    import operator


    def flavor_of():
        from cfgpkg.conf import settings

        return settings.preset.leiden.flavor


    def run(xs, how):
        return [getattr(operator, how)(x, 2) for x in xs], flavor_of()
""")

MAIN = textwrap.dedent("""
    import sys
    import cash
    from cfgpkg import conf, ops

    conf.settings.preset = conf.Preset(sys.argv[1])
    conf.settings.preset.leiden


    @cash.cache
    def cluster(xs, how):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        return ops.run(xs, how)


    print(cluster([1, 2], "mul"))
""")


def _run(tmp_path, flavor):
    pkg = tmp_path / "cfgpkg"
    pkg.mkdir(exist_ok=True)
    for name, text in {"__init__.py": "", "conf.py": CONF, "ops.py": OPS}.items():
        (pkg / name).write_text(text, encoding="utf-8")
    (tmp_path / "main.py").write_text(MAIN, encoding="utf-8")
    proc = run_python("main.py", flavor, cwd=tmp_path)
    return proc.stdout.strip(), proc.stderr


def test_the_call_is_stored_and_another_value_recomputes(tmp_path):
    out, err = _run(tmp_path, "igraph")
    assert out == "([2, 4], 'igraph')" and "[RUN]" in err
    out, err = _run(tmp_path, "igraph")
    assert "[RUN]" not in err, f"an unchanged second process must hit:\n{err}"
    out, err = _run(tmp_path, "leidenalg")
    assert out == "([2, 4], 'leidenalg')", f"served the result from before the change:\n{err}"
    assert "[RUN]" in err
