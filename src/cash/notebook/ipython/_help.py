"""The ``%cash_help`` card, built from the registered magics themselves.

Every line comes from what IPython actually registered and from each magic's
own docstring, so the card cannot name a magic that does not exist or miss one
that does.
"""

from __future__ import annotations

import inspect
from typing import Any

from ...analysis.annotations import KNOWN_DIRECTIVES, parse_annotation_line

__all__ = ["help_text"]

DOCS_URL = "https://cash-lib.readthedocs.io/"
ISSUES_URL = "https://github.com/galgtonold/cash/issues"
DISCUSSIONS_URL = "https://github.com/galgtonold/cash/discussions"

#: Reading order for the card: switching caching on and off first, then
#: display, then inspection. Only an ordering -- a registered magic missing
#: from it is still listed, after these.
_ORDER = (
    "cash_on",
    "cash_off",
    "cash_persist",
    "cash_badge",
    "cash_debug",
    "cash_status",
    "cash_stats",
    "cash_provenance",
    "cash_help",
)


def _registered(magics: Any) -> dict[str, Any]:
    """Name -> bound method for every line magic *magics* registers, in card order."""
    line = dict(magics.magics["line"])
    ordered = [name for name in _ORDER if name in line]
    ordered += sorted(name for name in line if name not in _ORDER)
    return {name: line[name] for name in ordered}


def _summary(func: Any) -> str:
    doc = inspect.getdoc(func) or ""
    return doc.strip().splitlines()[0] if doc.strip() else ""


def _annotation_examples() -> list[str]:
    """One working example per notebook ``# @cash:`` directive.

    Each is checked against the real parser, so the card never shows a line
    that would be silently ignored.
    """
    examples = []
    for directive in KNOWN_DIRECTIVES:
        example = f"# @cash:{directive}=300" if directive == "ttl" else f"# @cash:{directive}"
        if parse_annotation_line(example) is not None:
            examples.append(example)
    return examples


def _footer() -> list[str]:
    return [
        f"Docs: {DOCS_URL}",
        f"Bug reports & feature requests: {ISSUES_URL}",
        f"Questions & discussion: {DISCUSSIONS_URL}",
    ]


def help_text(magics: Any, topic: str = "") -> str:
    """The ``%cash_help`` card for *magics*, or one magic's usage for *topic*.

    *topic* names a magic with or without its ``%`` and ``cash_`` prefix
    (``badge``, ``cash_badge`` and ``%cash_badge`` all work). An unknown topic
    says so and prints the full card.
    """
    registered = _registered(magics)
    topic = topic.strip().lower().lstrip("%")
    lines: list[str] = []
    if topic:
        name = topic if topic.startswith("cash_") else f"cash_{topic}"
        if name in registered:
            lines.append(f"%{name}")
            lines.append("")
            lines.append(inspect.getdoc(registered[name]) or "(no description)")
            lines.append("")
            lines.extend(_footer())
            return "\n".join(lines)
        lines.append(f"No magic named %{name}.")
        lines.append("")

    width = max(len(name) for name in registered) + 3
    lines.append("Cash: smart caching for Jupyter notebooks")
    lines.append("")
    lines.append("Magics:")
    for name, func in registered.items():
        lines.append(f"  {('%' + name).ljust(width)}{_summary(func)}")
    lines.append("")
    lines.append("Run %cash_help <name> (e.g. %cash_help badge) for a magic's full usage.")
    lines.append("")
    lines.append("Annotations (a comment on or just above a statement):")
    lines.extend(f"  {example}" for example in _annotation_examples())
    lines.append("")
    lines.append("In a terminal, `cash inspect` lists every cache entry: its size,")
    lines.append("the time it saves, and how often it was used.")
    lines.append("")
    lines.extend(_footer())
    return "\n".join(lines)
