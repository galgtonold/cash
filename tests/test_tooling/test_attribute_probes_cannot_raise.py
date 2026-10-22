"""cash's notebook check asks the notebook's objects about attributes
through ``attribute_or`` and ``has_attribute``, never ``getattr(obj, name,
default)`` or ``hasattr``: those take AttributeError as "missing", and a
class whose ``__getattr__`` raises KeyError for a name it does not hold (a
record reading its fields from a dict) failed the cell where plain Python
runs it. Fixed one call at a time, the next one asked another name.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src" / "cash"
#: The package that asks the notebook's objects questions. Not ``analysis``:
#: the decorator's code walk raises on purpose when it cannot look into a
#: callable (UnwalkableLayers), so the function runs uncached and warns.
CHECKED = ("notebook",)


def _probes(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            if (node.func.id, len(node.args)) in (("getattr", 3), ("hasattr", 2)):
                found.append(f"{path.relative_to(ROOT.parent).as_posix()}:{node.lineno}: {node.func.id}")
    return found


def test_attribute_probes_cannot_raise():
    found = [probe for package in CHECKED for path in sorted((ROOT / package).rglob("*.py")) for probe in _probes(path)]
    assert not found, "use attribute_or / has_attribute (cash._user_attrs):\n" + "\n".join(found)
