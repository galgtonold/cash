"""No module under src/cash imports another cash module's private name.

ruff's PLC2701 is on for src/, but it skips imports from the same top-level
package, so it cannot see ``from cash.config import _x`` inside cash. This
walks the source instead. A name another module needs is part of its owner's
API and goes without the underscore.

Private *modules* (``cash.backends._base``, ``cash._plain_data``) are fine to
import: the underscore there marks the module as package-internal, not a
name as module-internal.
"""

from __future__ import annotations

import ast
from pathlib import Path

import cash

SRC = Path(cash.__file__).resolve().parent

# `cash._global_cash` is the default Cash, created on first use. A few modules
# peek at it without creating it; it lives on the package namespace, where any
# public name becomes user-facing API, so it keeps its underscore.
ALLOWED = {("cash", "_global_cash")}


def _is_private(name: str) -> bool:
    return name.startswith("_") and not name.startswith("__")


def _module_name(path: Path) -> tuple[str, bool]:
    parts = list(path.relative_to(SRC.parent).with_suffix("").parts)
    is_pkg = parts[-1] == "__init__"
    if is_pkg:
        parts = parts[:-1]
    return ".".join(parts), is_pkg


def _is_module(dotted: str) -> bool:
    rel = Path(*dotted.split("."))
    return (SRC.parent / rel).with_suffix(".py").is_file() or (SRC.parent / rel / "__init__.py").is_file()


def _private_uses(path: Path) -> list[str]:
    name, is_pkg = _module_name(path)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    module_aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = name.split(".") if is_pkg else name.split(".")[:-1]
                base = base[: len(base) - (node.level - 1)]
                source = ".".join(base + ([node.module] if node.module else []))
            else:
                source = node.module or ""
            if source != "cash" and not source.startswith("cash."):
                continue
            for alias in node.names:
                target = f"{source}.{alias.name}"
                if _is_module(target):
                    module_aliases[alias.asname or alias.name] = target
                elif _is_private(alias.name) and (source, alias.name) not in ALLOWED:
                    found.append(f"{name}:{node.lineno} imports {alias.name} from {source}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "cash" or alias.name.startswith("cash."):
                    if alias.asname:
                        module_aliases[alias.asname] = alias.name
                    else:
                        module_aliases[alias.name.split(".")[0]] = alias.name.split(".")[0]
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in module_aliases
            and _is_private(node.attr)
        ):
            owner = module_aliases[node.value.id]
            if owner != name and not _is_module(f"{owner}.{node.attr}") and (owner, node.attr) not in ALLOWED:
                found.append(f"{name}:{node.lineno} reads {owner}.{node.attr}")
    return found


def test_no_module_imports_another_modules_private_name():
    offenders = [use for path in sorted(SRC.rglob("*.py")) for use in _private_uses(path)]
    assert not offenders, "\n".join(offenders)


def test_the_check_sees_a_private_import(tmp_path, monkeypatch):
    # The walk above must actually catch both spellings, or the test passes
    # for the wrong reason.
    fake_pkg = tmp_path / "cash"
    fake_pkg.mkdir()
    (fake_pkg / "__init__.py").write_text("")
    (fake_pkg / "owner.py").write_text("_secret = 1\n")
    user = fake_pkg / "user.py"
    user.write_text("from . import owner\nfrom .owner import _secret\nprint(owner._secret, _secret)\n")
    monkeypatch.setitem(globals(), "SRC", fake_pkg)
    uses = _private_uses(user)
    assert any("imports _secret" in u for u in uses), uses
    assert any("reads cash.owner._secret" in u for u in uses), uses
