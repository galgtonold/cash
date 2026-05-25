"""Harness for running documentation code fences as tests.

PR1 scope: plain Python fences only. nb-cell handling deferred to PR2/PR3.
"""

from __future__ import annotations

import asyncio
import re
from ast import PyCF_ALLOW_TOP_LEVEL_AWAIT
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tests.docs._annotations import find_skip_for_fence


_FENCE_RE = re.compile(
    r"^```python(?P<attrs>(?:\s+\{[^}]*\})?)\s*$"
    r"(?P<body>.*?)^```\s*$",
    re.MULTILINE | re.DOTALL,
)


@dataclass
class Fence:
    """One python code fence extracted from a markdown file."""

    code: str
    line_start: int  # 1-based, the line containing ```python
    line_end: int    # 1-based, the line containing the closing ```
    attrs: str = ""  # raw attrs string, e.g. "{ .nb-cell }"
    skip: bool = False
    skip_reason: str | None = None

    @property
    def is_nb_cell(self) -> bool:
        return ".nb-cell" in self.attrs


def extract_fences(md_path: Path) -> list[Fence]:
    """Extract every ```python ... ``` fence from a markdown file in source order."""
    text = md_path.read_text(encoding="utf-8")
    fences: list[Fence] = []
    # Walk line-by-line so we get accurate line numbers.
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^```python(?P<attrs>(?:\s+\{[^}]*\})?)\s*$", line)
        if m:
            attrs = m.group("attrs").strip()
            start_line = i + 1  # 1-based
            body_lines: list[str] = []
            j = i + 1
            while j < len(lines) and lines[j].rstrip() != "```":
                body_lines.append(lines[j])
                j += 1
            end_line = j + 1  # 1-based line of closing ```
            skip_ann = find_skip_for_fence(lines, start_line)
            fences.append(
                Fence(
                    code="\n".join(body_lines),
                    line_start=start_line,
                    line_end=end_line,
                    attrs=attrs,
                    skip=skip_ann is not None,
                    skip_reason=skip_ann.reason if skip_ann else None,
                )
            )
            i = j + 1
        else:
            i += 1
    return fences


class PageExecutionError(RuntimeError):
    """Raised when a docs-parity page fails to exec."""


@dataclass
class ClaimResult:
    claim: "CacheClaim"
    actual_hits: int
    actual_misses: int
    matched: bool


class ClaimMismatchError(AssertionError):
    """Raised when a documented hit/miss claim does not match actual cache state."""


@dataclass
class PageResult:
    page: Path
    total_fences: int
    tested_fences: int
    skipped_fences: list[tuple[int, str]] = field(default_factory=list)
    namespace: dict[str, Any] = field(default_factory=dict)
    claim_results: list[ClaimResult] = field(default_factory=list)


def _apply_inject_comments(script: str) -> str:
    """Replace ``# test:inject: <code>`` comment lines with the code.

    Lines matching ``^(\\s*)# test:inject:\\s*(.+)$`` are replaced with
    ``<indent><code>``, preserving indentation so injections inside function
    bodies remain syntactically valid.  All other lines pass through unchanged.
    """
    out: list[str] = []
    for raw_line in script.splitlines(keepends=True):
        m = re.match(r"^(\s*)# test:inject:\s*(.+)$", raw_line.rstrip("\n"))
        if m:
            indent, code = m.group(1), m.group(2)
            out.append(indent + code + "\n")
        else:
            out.append(raw_line)
    return "".join(out)


def run_page(
    md_path: Path,
    namespace_overrides: dict[str, Any] | None = None,
    strict_claims: bool = True,
) -> PageResult:
    """Concatenate non-skipped fences, exec, then verify cache claims."""
    fences = extract_fences(md_path)

    result = PageResult(
        page=md_path,
        total_fences=len(fences),
        tested_fences=0,
    )

    pieces: list[str] = []
    for f in fences:
        if f.skip:
            result.skipped_fences.append((f.line_start, f.skip_reason or "<no reason>"))
            continue
        if f.is_nb_cell:
            result.skipped_fences.append(
                (f.line_start, "nb-cell: requires IPython kernel (PR2+ scope)")
            )
            continue
        pad = max(0, f.line_start - sum(p.count("\n") + 2 for p in pieces) - 1)
        pieces.append("\n" * pad + f.code)
        result.tested_fences += 1

    if not pieces:
        return result

    script = "\n\n".join(pieces)
    script = _apply_inject_comments(script)
    namespace: dict[str, Any] = {"__name__": "__cash_docs_test__"}
    if namespace_overrides:
        namespace.update(namespace_overrides)

    try:
        code_obj = compile(
            script,
            str(md_path),
            "exec",
            flags=PyCF_ALLOW_TOP_LEVEL_AWAIT,
        )
        # If the compiled code is a coroutine (top-level await), run it.
        maybe_coro = eval(code_obj, namespace)
        if asyncio.iscoroutine(maybe_coro):
            asyncio.run(maybe_coro)
    except Exception as e:
        raise PageExecutionError(
            f"{md_path}: exec failed with {type(e).__name__}: {e}"
        ) from e

    result.namespace = namespace

    # Now verify cache claims by introspecting the decorated functions left
    # in `namespace`. For each `@cash.cache` function the harness inferred,
    # call .cache_info() and compare with the claim.
    claims = infer_claims(script)
    for claim in claims:
        fn = namespace.get(claim.function)
        if fn is None or not hasattr(fn, "cache_info"):
            # Function not in the post-exec namespace (e.g. stateful funcs
            # whose decorator returns the original). Skip the assertion;
            # mark as matched only if claim expected 0 hits/0 misses.
            actual_hits, actual_misses = 0, 0
            matched = claim.expected_hits == 0 and claim.expected_misses == 0
        else:
            info = fn.cache_info()
            actual_hits = info.get("hits", 0)
            actual_misses = info.get("misses", 0)
            matched = (
                actual_hits == claim.expected_hits
                and actual_misses == claim.expected_misses
            )
        result.claim_results.append(
            ClaimResult(
                claim=claim,
                actual_hits=actual_hits,
                actual_misses=actual_misses,
                matched=matched,
            )
        )

    if strict_claims:
        mismatches = [r for r in result.claim_results if not r.matched]
        if mismatches:
            lines = [f"{md_path}: cache claim mismatch:"]
            for r in mismatches:
                lines.append(
                    f"  {r.claim.function}: "
                    f"expected hits={r.claim.expected_hits} misses={r.claim.expected_misses}, "
                    f"got hits={r.actual_hits} misses={r.actual_misses}"
                )
            raise ClaimMismatchError("\n".join(lines))

    return result


import ast


@dataclass
class CacheClaim:
    function: str
    expected_hits: int
    expected_misses: int


# Patterns that downgrade a call to "expected to miss" via comment proximity.
_MISS_HINT = re.compile(
    r"#.*?(cache\s*miss|first\s+call|fresh|computed|EXECUTED|recompute|API\s+call|invalidat)",
    re.IGNORECASE,
)
# Patterns that mark a call as "expected to hit" via comment proximity.
_HIT_HINT = re.compile(
    r"#.*?(cache\s*hit|second\s+call|instant|RESTORED|cached)",
    re.IGNORECASE,
)


def _is_cache_decorator(deco: ast.expr) -> bool:
    """Match @cash.cache, @cache, @c.cache, @app.cache (any 'cache' attribute name)."""
    if isinstance(deco, ast.Attribute) and deco.attr == "cache":
        return True
    if isinstance(deco, ast.Name) and deco.id == "cache":
        return True
    if isinstance(deco, ast.Call):
        return _is_cache_decorator(deco.func)
    return False


def _is_stateful_decorator(deco: ast.expr) -> bool:
    if isinstance(deco, ast.Attribute) and deco.attr == "stateful":
        return True
    if isinstance(deco, ast.Name) and deco.id == "stateful":
        return True
    return False


def infer_claims(source: str) -> list[CacheClaim]:
    """Parse the source, find @cash.cache-decorated functions, and infer the
    expected (hits, misses) per function based on call counts, unique argument
    tuples, and any inline-comment overrides.
    """
    tree = ast.parse(source)

    cached_funcs: set[str] = set()
    stateful_funcs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for deco in node.decorator_list:
                if _is_cache_decorator(deco):
                    cached_funcs.add(node.name)
                elif _is_stateful_decorator(deco):
                    stateful_funcs.add(node.name)

    # Map each call to its function name and arg tuple (text representation).
    lines = source.splitlines()
    calls: dict[str, list[tuple[str, int]]] = {}  # func -> [(args_repr, lineno)]
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target_name = None
            if isinstance(node.func, ast.Name):
                target_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                target_name = node.func.attr
            if target_name is None:
                continue
            if target_name not in cached_funcs and target_name not in stateful_funcs:
                continue
            args_repr = ",".join(
                ast.unparse(a) if hasattr(ast, "unparse") else repr(a) for a in node.args
            )
            calls.setdefault(target_name, []).append((args_repr, node.lineno))

    claims: list[CacheClaim] = []
    for func, sites in calls.items():
        if func in stateful_funcs:
            claims.append(
                CacheClaim(function=func, expected_hits=0, expected_misses=len(sites))
            )
            continue
        # Count comment-tagged misses/hits and infer the rest from arg uniqueness.
        explicit_miss = 0
        explicit_hit = 0
        ambiguous: list[tuple[str, int]] = []
        for args_repr, lineno in sites:
            line_text = lines[lineno - 1] if 1 <= lineno <= len(lines) else ""
            if _MISS_HINT.search(line_text):
                explicit_miss += 1
            elif _HIT_HINT.search(line_text):
                explicit_hit += 1
            else:
                ambiguous.append((args_repr, lineno))
        # For ambiguous calls, infer: first call per unique args = miss, rest = hits.
        seen: dict[str, int] = {}
        ambig_hits = 0
        ambig_misses = 0
        for args_repr, _ in ambiguous:
            n = seen.get(args_repr, 0)
            if n == 0:
                ambig_misses += 1
            else:
                ambig_hits += 1
            seen[args_repr] = n + 1
        claims.append(
            CacheClaim(
                function=func,
                expected_hits=explicit_hit + ambig_hits,
                expected_misses=explicit_miss + ambig_misses,
            )
        )
    return claims
