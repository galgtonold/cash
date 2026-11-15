"""Stacked wrappers of a lazily imported function are checked each by its own cells.

Reported against scanpy checked out as the user's code: a traced call did
``from ._pca import pca`` inside a function, and ``pca`` is under several
``@deprecated_arg(...)`` wrappers, each a closure of one decorator's code
holding an ``inspect.Signature`` whose defaults include a
``mappingproxy({})``. Two things kept the call from being served: a
signature with a read-only mapping in it could not be digested, so the
value was refused; and the cells each wrapper started with were looked up
by code, so every wrapper of that code was digested by the cells of the
first one that ran, a digest no later process reproduced. Each wrapper now
goes by its own cells, and a change to what one holds is still seen.
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

DECO = textwrap.dedent("""
    import functools
    import inspect


    class renamed:
        def __init__(self, old, scale):
            self.old = old
            self.scale = scale

        def __call__(self, func):
            sig = inspect.signature(func)

            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                if self.old in kwargs:
                    sig.bind(*args, **kwargs)
                return func(*args, **kwargs) * self.scale

            return wrapped
""")

LIB = textwrap.dedent("""
    from types import MappingProxyType

    from .deco import renamed


    @renamed("a", {outer})
    @renamed("b", 1)
    @renamed("c", 1)
    def project(x, *, opts=MappingProxyType({{}}), a=None, b=None, c=None):
        return x + len(opts)
""")

OPS = textwrap.dedent("""
    import operator


    def run(xs, how):
        from .lib import project

        return [getattr(operator, how)(project(x), 1) for x in xs]
""")

MAIN = textwrap.dedent("""
    import sys
    import cash
    import wrappkg.lib  # loaded already, as a package's other entry points load it
    from wrappkg import ops


    @cash.cache
    def embed(xs, how):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        return ops.run(xs, how)


    print(embed([1, 2], "mul"))
""")


def _run(tmp_path, outer):
    pkg = tmp_path / "wrappkg"
    pkg.mkdir(exist_ok=True)
    files = {"__init__.py": "", "deco.py": DECO, "lib.py": LIB.format(outer=outer), "ops.py": OPS}
    for name, text in files.items():
        (pkg / name).write_text(text, encoding="utf-8")
    (tmp_path / "main.py").write_text(MAIN, encoding="utf-8")
    proc = run_python("main.py", cwd=tmp_path)
    return proc.stdout.strip(), proc.stderr


def test_an_unchanged_stack_hits_and_a_changed_wrapper_recomputes(tmp_path):
    out, err = _run(tmp_path, 1)
    assert out == "[1, 2]" and "[RUN]" in err
    out, err = _run(tmp_path, 1)
    assert out == "[1, 2]" and "[RUN]" not in err, f"an unchanged second process must hit:\n{err}"
    out, err = _run(tmp_path, 3)
    assert out == "[3, 6]" and "[RUN]" in err, f"served the result from before the change:\n{err}"
