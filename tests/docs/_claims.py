"""Claim anchors: link a prose claim to the source that decides it.

An anchor is an HTML comment immediately before the claim it grounds::

    <!-- claim: cash/core.py:Cash.cache @7a77d1c5 -->
    Cash keys a call on the function source plus its arguments.

See ``docs/superpowers/specs/2026-07-27-docs-claim-grounding-design.md``.

This module is pure — no pytest import — so ``scripts/claims.py`` can use it.
"""
from __future__ import annotations

import ast
import hashlib
import io
import re
import textwrap
import tokenize
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs"
SRC_ROOT = REPO_ROOT / "src"

# ``superpowers/`` is internal planning (gitignored, not built).
# ``architecture_decisions.md`` is in mkdocs.yml ``exclude_docs`` — never built.
_BLACKLIST_DIRS = {"superpowers"}
_EXCLUDED_FILES = {"architecture_decisions.md"}


class AnchorError(Exception):
    """A claim anchor is malformed, unresolvable, or not comparable."""


def published_pages() -> list[Path]:
    """Every Markdown page mkdocs actually builds."""
    return sorted(
        p
        for p in DOCS_ROOT.rglob("*.md")
        if not any(part in _BLACKLIST_DIRS for part in p.relative_to(DOCS_ROOT).parts)
        and p.relative_to(DOCS_ROOT).as_posix() not in _EXCLUDED_FILES
    )


@dataclass(frozen=True)
class Target:
    path: str                 # relative to src/, e.g. "cash/core.py"
    symbol: str | None = None  # dotted, e.g. "Cash.cache"; None means the module
    pin: str | None = None     # 8 hex chars, or "?" for an unfilled placeholder
    value: str | None = None   # raw literal text from `== ...`


@dataclass(frozen=True)
class Anchor:
    page: Path
    line: int                 # 1-based, the line the comment starts on
    claim: str                # first non-blank line after the comment
    targets: tuple[Target, ...]
    broad: str | None = None


_CLAIM_RE = re.compile(r"<!--\s*claim:\s*(?P<body>.*?)-->", re.DOTALL)
_BROAD_RE = re.compile(r'broad\s*=\s*"(?P<reason>[^"]*)"')
_TARGET_RE = re.compile(
    r"""^\s*
        (?P<path>[\w./-]+\.py)
        (?:\s*:\s*(?P<symbol>[\w.]+))?
        (?:\s*(?:@\s*(?P<pin>[0-9a-f]{8}|\?)|==\s*(?P<value>\S.*?)))?
        \s*$""",
    re.VERBOSE,
)


def _claim_text(lines: list[str], end_line_idx: int) -> str:
    """The first non-blank line after the comment — used in error messages."""
    for line in lines[end_line_idx + 1:]:
        if line.strip():
            return line.strip()[:120]
    return ""


def parse_anchors(text: str, page: Path) -> list[Anchor]:
    """Every claim anchor in *text*, in source order.

    Raises AnchorError on a malformed anchor. Skipping one silently would hide
    the claim it was meant to ground, which is the whole failure this exists to
    prevent.
    """
    lines = text.splitlines()
    out: list[Anchor] = []
    for m in _CLAIM_RE.finditer(text):
        line_no = text.count("\n", 0, m.start()) + 1
        end_idx = text.count("\n", 0, m.end())
        body = m.group("body")

        broad_m = _BROAD_RE.search(body)
        broad = broad_m.group("reason") if broad_m else None
        if broad_m:
            body = _BROAD_RE.sub("", body)

        targets: list[Target] = []
        for chunk in body.split(","):
            if not chunk.strip():
                continue
            tm = _TARGET_RE.match(chunk)
            if tm is None:
                raise AnchorError(
                    f"{page}:{line_no}: cannot parse claim target {chunk.strip()!r}"
                )
            raw_value = tm.group("value")
            targets.append(
                Target(
                    path=tm.group("path"),
                    symbol=tm.group("symbol"),
                    pin=tm.group("pin"),
                    value=raw_value.strip() if raw_value else None,
                )
            )
        if not targets:
            raise AnchorError(f"{page}:{line_no}: claim comment names no target")

        out.append(
            Anchor(
                page=page,
                line=line_no,
                claim=_claim_text(lines, end_idx),
                targets=tuple(targets),
                broad=broad,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Resolution                                                                  #
# --------------------------------------------------------------------------- #

_DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _children_named(node: ast.AST, name: str) -> list[ast.AST]:
    """EVERY child binding *name*, in source order — not just the first.

    One name can have several definitions: ``@overload`` stubs before the real
    implementation, or a conditional redefinition. ``Cash.cache`` is three
    ``def cache``s, the first a one-line stub. Returning only that stub would
    pin something that never changes, leaving the anchor green forever while
    the implementation drifted -- the precise false negative this exists to
    prevent.
    """
    found: list[ast.AST] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _DEF_NODES) and child.name == name:
            found.append(child)
        elif isinstance(child, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in child.targets
        ):
            found.append(child)
        elif (
            isinstance(child, ast.AnnAssign)
            and isinstance(child.target, ast.Name)
            and child.target.id == name
        ):
            found.append(child)
    return found


def resolve(target: Target, src_root: Path = SRC_ROOT) -> tuple[list[ast.AST], str]:
    """Resolve *target* to its AST node(s) and the source of its module.

    Returns a LIST: see ``_children_named``. Callers wanting a single node
    (a value anchor) take ``nodes[-1]``, the effective definition.
    """
    file = src_root / target.path
    if not file.is_file():
        raise AnchorError(f"no such source file: src/{target.path}")
    source = file.read_text(encoding="utf-8")
    tree = ast.parse(source)
    if not target.symbol:
        return [tree], source

    nodes: list[ast.AST] = [tree]
    seen: list[str] = []
    for part in target.symbol.split("."):
        # Descend through the LAST binding of each intermediate name.
        found = _children_named(nodes[-1], part)
        if not found:
            where = ".".join(seen) if seen else "module scope"
            raise AnchorError(
                f"src/{target.path}: no symbol {part!r} in {where}"
            )
        nodes = found
        seen.append(part)
    return nodes, source


# --------------------------------------------------------------------------- #
# Fingerprinting                                                              #
# --------------------------------------------------------------------------- #


def normalize(node: ast.AST, source: str) -> str:
    """The node's source, stripped of everything that is not code.

    Comments, blank lines and trailing whitespace go; the result is dedented.
    Deterministic across Python versions *by construction*, unlike ``ast.dump``
    (whose node fields change between releases) — which matters because CI runs
    3.12 and development runs 3.14.
    """
    if isinstance(node, ast.Module):
        segment = source
    else:
        start = node.lineno
        # A decorator is part of what the node does, so include it.
        for deco in getattr(node, "decorator_list", []) or []:
            start = min(start, deco.lineno)
        lines = source.splitlines()
        segment = "\n".join(lines[start - 1: node.end_lineno])

    segment = textwrap.dedent(segment)

    # Column at which a comment starts, per line. tokenize (not a regex) so a
    # '#' inside a string literal is not mistaken for one.
    cuts: dict[int, int] = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(segment).readline):
            if tok.type == tokenize.COMMENT:
                line, col = tok.start
                cuts[line] = min(cuts.get(line, col), col)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # An unparseable slice keeps its comments rather than losing content.
        pass

    out: list[str] = []
    for i, line in enumerate(segment.splitlines(), 1):
        if i in cuts:
            line = line[: cuts[i]]
        line = line.rstrip()
        if line:
            out.append(line)
    return "\n".join(out)


def fingerprint(nodes: list[ast.AST], source: str) -> str:
    """8-hex-char digest over the normalized source of every node.

    Takes a list so that all of a name's definitions are covered — a change to
    any ``@overload`` signature moves the digest, not only a change to the
    implementation.
    """
    blob = "\n".join(normalize(n, source) for n in nodes)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]
