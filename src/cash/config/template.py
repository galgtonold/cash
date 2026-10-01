"""The documented config file `create_default_config` writes.

Built from `CashConfig` and `TierConfig` themselves -- each field's docstring
and default -- so the template cannot fall behind the settings.
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from dataclasses import fields
from pathlib import Path
from typing import Any

from .._location import default_user_config_path
from . import resolve
from .schema import CashConfig, TierConfig

__all__ = ["create_default_config"]


_CONFIG_DOCS_URL = "https://cash-lib.readthedocs.io/en/latest/getting-started/configuration/"

#: The ``[[cash.tiers]]`` example in the template: a RAM tier in front of Redis.
_TIER_EXAMPLE = """\
[[cash.tiers]]
type = "memory"
max_entries = 10000

[[cash.tiers]]
type = "redis"
host = "redis.internal"
port = 6379"""


def _field_docs(cls: type) -> dict[str, str]:
    """The docstring under each field of dataclass *cls*, read from its source.

    Python drops attribute docstrings at compile time, so the only copy is
    the source. Empty when the source is not available (a frozen app).
    """

    try:
        body = ast.parse(textwrap.dedent(inspect.getsource(cls))).body[0].body
    except (OSError, TypeError):
        return {}
    docs = {}
    for node, nxt in zip(body, body[1:]):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(nxt, ast.Expr)
            and isinstance(nxt.value, ast.Constant)
            and isinstance(nxt.value.value, str)
        ):
            docs[node.target.id] = inspect.cleandoc(nxt.value.value)
    return docs


def _tier_key_docs() -> dict[str, str]:
    """Each ``TierConfig`` key's entry in its docstring's Attributes section.

    Empty when docstrings are stripped (``python -OO``).
    """
    doc = inspect.cleandoc(TierConfig.__doc__ or "")
    section = doc.partition("\nAttributes:\n")[2]
    docs: dict[str, str] = {}
    name = None
    for line in section.splitlines():
        entry = re.match(r"    (\w+): (.*)", line)
        if entry:
            name = entry[1]
            docs[name] = entry[2]
        elif name and line.startswith("        "):
            docs[name] += " " + line.strip()
        else:
            break
    return docs


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return repr(value)


def _comment(text: str) -> list[str]:
    return [f"# {line}".rstrip() for line in text.splitlines()]


def _default_config_text() -> str:
    """The template `create_default_config` writes, built from `CashConfig`.

    Every setting is commented out at its default, under its field docstring,
    so the file changes nothing until a line is uncommented and a new default
    in cash still reaches a user who never touched that line.
    """
    # The layer list in `resolve`'s docstring, so the two cannot disagree.
    precedence = (resolve.__doc__ or "").partition("highest priority wins):")[2].strip("\n").split("\n\n")[0]
    lines = [
        f"# Cash configuration -- see {_CONFIG_DOCS_URL}",
        "#",
        "# Resolution order (highest wins):",
        *_comment(textwrap.dedent(precedence).strip("\n")),
        "#",
        "# Every setting below is commented out at its default. Uncomment a line",
        "# to change it. A setting marked (unset) has no default value.",
        "",
        "[cash]",
    ]
    docs = _field_docs(CashConfig)
    defaults = CashConfig()
    for f in fields(CashConfig):
        if f.name.startswith("_") or f.name == "tiers":
            continue
        value = getattr(defaults, f.name)
        lines += ["", *_comment(docs.get(f.name, ""))]
        lines.append(f"# {f.name} = " + ("(unset)" if value is None else _toml_value(value)))

    tier_docs = _tier_key_docs()
    lines += ["", *_comment(docs.get("tiers", "")), "#", "# Keys of a [[cash.tiers]] table:"]
    for f in fields(TierConfig):
        first = tier_docs.get(f.name, "")
        lines += _comment(textwrap.fill(f"{f.name}: {first}", 72, initial_indent="  ", subsequent_indent="      "))
    lines += ["#", *_comment(_TIER_EXAMPLE)]
    return "\n".join(lines) + "\n"


def create_default_config(path: str | None = None, *, force: bool = False) -> str:
    """Write a documented config template to *path* and return the path.

    *path* defaults to the user config file (``~/.config/cash/config.toml``,
    or ``%APPDATA%\\cash\\config.toml`` on Windows). Every setting is listed
    commented out at its default, with its documentation above it.

    Raises ``FileExistsError`` if the file exists, unless ``force=True``.
    """
    out_path = Path(path) if path is not None else default_user_config_path()
    if out_path.exists() and not force:
        raise FileExistsError(f"{out_path} already exists; pass force=True to overwrite it")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_default_config_text(), encoding="utf-8")
    return str(out_path)
