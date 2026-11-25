"""cash looks for a library by name only through ``loaded_module``.

``sys.modules.get("numpy")`` answers with a module whose import another
thread has only begun: no ``ndarray`` in it yet, and a cached call on this
thread failed inside cash. ``cash.loaded_modules.loaded_module`` leaves
such a module out. One such look fixed at a time leaves the next; this
holds the whole package to the one helper.

Exempt: ``__main__``, ``sys`` and ``builtins`` (nothing imports them
late) and cash's own modules.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "src" / "cash"
HELPER = "loaded_modules.py"

#: Files with looks not yet moved to the helper, each owned by other work
#: in flight when the rule came in. Move a file's looks over, then drop it
#: here: the test fails for a file listed that has none left.
PENDING = frozenset(
    {
        "canonical_form.py",
        "decorator/hit_memo.py",
        "notebook/call_key.py",
        "notebook/shared_objects.py",
    }
)


def _exempt(name: str) -> bool:
    return name in ("__main__", "sys", "builtins") or name == "cash" or name.startswith("cash.")


def _is_sys_modules(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "modules"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    )


def _library(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and not _exempt(node.value):
        return node.value
    return None


def bare_looks(source: str) -> list[tuple[int, str]]:
    """``(line, library)`` for each ``sys.modules.get("lib")`` and
    ``"lib" [not] in sys.modules`` in *source*."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and _is_sys_modules(node.func.value)
            and node.args
            and _library(node.args[0])
        ):
            found.append((node.lineno, _library(node.args[0])))
        elif (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], (ast.In, ast.NotIn))
            and _is_sys_modules(node.comparators[0])
            and _library(node.left)
        ):
            found.append((node.lineno, _library(node.left)))
    return found


def _by_file() -> dict[str, list[tuple[int, str]]]:
    out = {}
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if rel != HELPER:
            looks = bare_looks(path.read_text(encoding="utf-8"))
            if looks:
                out[rel] = looks
    return out


def test_no_library_is_looked_up_in_sys_modules_directly():
    found = [
        f"src/cash/{rel}:{line}: {library}"
        for rel, looks in _by_file().items()
        if rel not in PENDING
        for line, library in looks
    ]
    assert not found, (
        "use loaded_module(name) from cash.loaded_modules: a module another thread is still importing "
        "is in sys.modules without its attributes:\n" + "\n".join(found)
    )


def test_every_pending_file_still_has_a_look_to_move():
    by_file = _by_file()
    done = sorted(rel for rel in PENDING if rel not in by_file)
    assert not done, f"no direct look left in {done}: drop them from PENDING"


@pytest.mark.parametrize(
    "line",
    [
        'np = sys.modules.get("numpy")',
        'if "pandas" in sys.modules: pass',
        'if "pandas" not in sys.modules: pass',
        'x = getattr(sys.modules.get("matplotlib.pyplot"), "gcf", None)',
    ],
)
def test_the_scan_finds_each_shape_of_a_direct_look(line):
    assert bare_looks("import sys\n" + line + "\n")


@pytest.mark.parametrize(
    "line",
    [
        'main = sys.modules.get("__main__")',
        'own = sys.modules.get("cash.decorator.arg_hashing")',
        "module = sys.modules.get(name)",
        'np = loaded_module("numpy")',
        '# sys.modules.get("numpy")',
        'np = sys.modules["numpy"]',
    ],
)
def test_the_scan_leaves_other_lines_alone(line):
    assert not bare_looks("import sys\n" + line + "\n")
