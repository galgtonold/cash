"""The modules a function's body imports (`library_versions._imported_modules`,
which decides the libraries whose versions a key folds) are read from
``co_code`` in one loop: the same absolute names, in the same order, as
reading ``dis.get_instructions`` gives, for nested code, relative imports
and a body long enough that its arguments take an ``EXTENDED_ARG``.
"""

from __future__ import annotations

import dis
import json
import sys
import textwrap
import types

from cash.decorator.library_versions import _imported_modules


def _with_dis(code: types.CodeType) -> tuple[str, ...]:
    found: dict[str, None] = {}
    stack = [code]
    while stack:
        c = stack.pop()
        stack.extend(k for k in c.co_consts if isinstance(k, types.CodeType))
        ins = list(dis.get_instructions(c))
        for k, i in enumerate(ins):
            if i.opname == "IMPORT_NAME" and isinstance(i.argval, str) and k >= 2 and ins[k - 2].argval == 0:
                found[i.argval] = None
    return tuple(found)


_SOURCE = textwrap.dedent(
    """
    def f():
        import json
        from os import path
        from . import sibling
        from .pkg import thing
        import xml.dom.minidom as md

        def inner():
            import csv
            from collections import abc
            return csv, abc

        g = lambda: __import__("ignored")
        {names}
        import email.utils
        return json, path, sibling, thing, md, inner, g
    """
)


def _code() -> types.CodeType:
    # Enough distinct names that the late import's name index passes 255.
    names = "\n    ".join(f"v{i} = {i}" for i in range(300)) + "\n    " + "\n    ".join(f"w{i} = q{i}" for i in range(300))
    module = compile(_SOURCE.format(names=names), "<imports>", "exec")
    return next(c for c in module.co_consts if isinstance(c, types.CodeType))


def test_the_imported_modules_match_dis():
    code = _code()
    expected = _with_dis(code)
    assert {"json", "os", "xml.dom.minidom", "csv", "collections"} <= set(expected)
    assert "pkg" not in expected and "sibling" not in expected
    assert _imported_modules(code) == expected


def test_the_standard_library_s_functions_match_dis():
    seen = 0
    for module in (json, textwrap, dis, sys.modules["collections"]):
        for value in vars(module).values():
            if isinstance(value, types.FunctionType):
                seen += 1
                assert _imported_modules(value.__code__) == _with_dis(value.__code__), value
    assert seen > 20
