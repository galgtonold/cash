"""pandas' private names are spelled in ``cash/_pandas_compat.py`` only.

pandas is not a dependency of cash: the user's ``pip install -U pandas``
decides which pandas cash meets, and a private attribute or module can go
with any release. With every such name in one module, that module is what
has to be read against a new pandas, and its start-up check
(``structure_problems``) says when one of them is gone.

Held to it here: the attributes and modules whose names are pandas' alone.
A name cash's own classes also use (``_data``, ``_values``, ``.blocks``)
cannot be told apart by its spelling; those go through the module by
convention.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "src" / "cash"
HOME = "_pandas_compat.py"

#: Attributes only pandas has.
PRIVATE_ATTRIBUTES = frozenset(
    {
        "_mgr",
        "_pa_array",
        "_ndarray",
        "_references",
        "_item_cache",
        "_constructor_from_mgr",
        "_from_backing_data",
        "referenced_blocks",
        "has_reference",
        "blknos",
        "blklocs",
    }
)
#: Modules a release of pandas may move or drop.
PRIVATE_MODULES = ("pandas.core", "pandas._libs", "pandas.io.formats", "pandas.util._")


def _docstrings(tree: ast.AST) -> set[int]:
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                found.add(id(body[0].value))
    return found


def private_touches(source: str) -> list[tuple[int, str]]:
    """``(line, name)`` for each private pandas attribute read or written,
    each one named as a string (``getattr(x, "_mgr")``) and each private
    pandas module imported or named in *source*. Docstrings and comments
    may say what they like."""
    tree = ast.parse(source)
    prose = _docstrings(tree)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in PRIVATE_ATTRIBUTES:
            found.append((node.lineno, node.attr))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in prose:
            text = node.value
            if text in PRIVATE_ATTRIBUTES or text.startswith(PRIVATE_MODULES):
                found.append((node.lineno, text))
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith(PRIVATE_MODULES):
            found.append((node.lineno, node.module))
        elif isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names if alias.name.startswith(PRIVATE_MODULES)]
    return sorted(found)


def _by_file() -> dict[str, list[tuple[int, str]]]:
    out = {}
    for path in sorted(ROOT.rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        if relative == HOME:
            continue
        touches = private_touches(path.read_text(encoding="utf-8"))
        if touches:
            out[relative] = touches
    return out


def test_no_module_but_the_one_names_a_private_part_of_pandas():
    found = _by_file()
    lines = [f"  {name}:{line}: {what}" for name, touches in found.items() for line, what in touches]
    assert not found, (
        "pandas' private names belong in cash/_pandas_compat.py (add an accessor there and call it):\n"
        + "\n".join(lines)
    )


def test_the_one_module_is_where_they_are():
    """The scan sees what it is meant to: the home module is full of them."""
    touches = private_touches((ROOT / HOME).read_text(encoding="utf-8"))
    names = {what for _line, what in touches}
    assert {"_mgr", "_pa_array", "_ndarray", "referenced_blocks"} <= names
    assert any(what.startswith("pandas.core") for what in names)


@pytest.mark.parametrize(
    "source, name",
    [
        ("x = df._mgr.blocks\n", "_mgr"),
        ("x = getattr(values, '_pa_array', None)\n", "_pa_array"),
        ("values._ndarray = other\n", "_ndarray"),
        ("from pandas.core.generic import NDFrame\n", "pandas.core.generic"),
        ("import pandas._libs.index\n", "pandas._libs.index"),
        ("m = loaded_module('pandas.core.arrays.arrow.array')\n", "pandas.core.arrays.arrow.array"),
        ("if refs.has_reference():\n    pass\n", "has_reference"),
    ],
)
def test_each_shape_of_touch_is_found(source, name):
    assert [what for _line, what in private_touches(source)] == [name]


def test_a_docstring_may_name_them():
    assert private_touches('def f():\n    """Reads ``df._mgr`` and "_mgr"."""\n    return 1\n') == []
