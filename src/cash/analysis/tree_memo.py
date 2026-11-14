"""What the helper walk works out from a function's source text alone, kept
for the process.

Every ``@cash.cache`` function's first call walks each helper it reaches
(`helper_walk.HelperWalk`), and a helper many cached functions share -- a
library's parameter checks, its logger -- is walked once per cached function.
The parts of that work that follow from the source TEXT alone are the same
every time: the parse, and the facts read off the function's tree (its
locals, the fresh-value flow given the fresh library calls). Those are kept
here, keyed by the text and by the tree, so a second walk reuses them; what
depends on what the names are bound to now (the namespace, the purity
visitor's resolution of each call) is worked out again on every walk.

A tree handed out here is shared, so nothing may change it: `HelperWalk`
strips the decorators off a copy of the function node (`stripped_def`).

``CASH_ANALYSIS_MEMO=0`` turns it off; ``CASH_ANALYSIS_MEMO_VERIFY=1`` works
each reused answer out again and raises `TreeMemoMismatch` when it differs, the
shared tree included (a change to it would show as a different dump).
"""

from __future__ import annotations

import ast
import copy
import os
from collections.abc import Callable
from typing import Any, TypeVar

from .. import _weak
from .._memo import PARSED_SOURCES, LruMemo

__all__ = ["ENABLED", "VERIFY", "TreeMemoMismatch", "fact", "parse", "status", "stripped_def"]

T = TypeVar("T")


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in ("", "0", "false", "no", "off")


#: Is the memo on?
ENABLED = _flag("CASH_ANALYSIS_MEMO", "1")
#: Work every reused answer out again and compare (the on-disk module scans
#: of `scan_store` too).
VERIFY = _flag("CASH_ANALYSIS_MEMO_VERIFY", "0")


class TreeMemoMismatch(AssertionError):
    """``CASH_ANALYSIS_MEMO_VERIFY=1``: a reused tree or fact is not what working
    it out again gives."""


#: Source text -> its tree (and, for the verify mode, the tree's dump when parsed).
_TREES: LruMemo[str, tuple[ast.Module, str | None]] = LruMemo(PARSED_SOURCES)
#: A shared function node -> {fact name: answer}. Weak: an entry goes with
#: its tree when `_TREES` drops it.
_FACTS: _weak.WeakKeyDictionary = _weak.WeakKeyDictionary()
#: A stripped copy -> the shared node it was made from.
_ORIGIN: _weak.WeakKeyDictionary = _weak.WeakKeyDictionary()

STATS = {"parsed": 0, "parse_reused": 0, "facts": 0, "facts_reused": 0, "verified": 0}


def status() -> dict:
    """What the memo did in this process."""
    return {"enabled": ENABLED, "verify": VERIFY, "trees": len(_TREES), **STATS}


def parse(src: str, path: str | None = None) -> ast.Module:
    """``ast.parse(src)``, shared with every earlier caller that parsed the
    same text. Raises `SyntaxError` as the parse does (not remembered).

    *path*, the file *src* was read from, lets the facts of its functions
    be stored across processes too (`fact_store`)."""
    if not ENABLED:
        return ast.parse(src)
    hit = _TREES.get(src)
    if hit is not None:
        tree, dumped = hit
        STATS["parse_reused"] += 1
        if VERIFY:
            _verify("tree", dumped, ast.dump(ast.parse(src), include_attributes=True))
            _verify("tree", dumped, ast.dump(tree, include_attributes=True))
    else:
        tree = ast.parse(src)
        STATS["parsed"] += 1
        _TREES[src] = (tree, ast.dump(tree, include_attributes=True) if VERIFY else None)
    if path:
        from . import fact_store

        fact_store.register(tree, src, path)
    return tree


def stripped_def(func_def: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """A copy of *func_def* with no decorators, sharing everything else:
    the body the purity rules read, without changing the shared tree."""
    stripped = copy.copy(func_def)
    stripped.decorator_list = []
    if ENABLED:
        _ORIGIN[stripped] = _ORIGIN.get(func_def, func_def)
    return stripped


def fact(
    node: ast.AST,
    name: Any,
    compute: Callable[[ast.AST], T],
    stored: tuple[str, Callable[[ast.AST, T], Any], Callable[[ast.AST, Any], T]] | None = None,
) -> T:
    """``compute(node)``, remembered under *name* for the shared tree *node*
    comes from. *compute* must read nothing but *node* (and what *name*
    spells out): the answer is reused for every walk that meets the tree.

    *stored*, ``(key, encode, decode)``, keeps it across processes as well
    (`fact_store.kept`): *key* spells out what *name* does, without ids."""
    if not ENABLED:
        return compute(node)
    origin = _ORIGIN.get(node)
    if origin is None:
        # Not a copy made by `stripped_def`: keyed by the node itself.
        origin = node
    try:
        facts = _FACTS.get(origin)
        if facts is None:
            facts = _FACTS.setdefault(origin, {})
    except TypeError:  # not weak-referenceable
        return compute(node)
    # Keyed by the node handed in too: the stripped copy and its source
    # differ in their decorators, which some facts read.
    key = (name, node is origin)
    if key in facts:
        answer = facts[key]
        STATS["facts_reused"] += 1
        if VERIFY:
            _verify(f"fact {name!r}", answer, compute(node))
        return answer
    if stored is None:
        answer = compute(node)
    else:
        from . import fact_store

        answer = fact_store.kept(origin, node, stored[0], compute, stored[1], stored[2])
    STATS["facts"] += 1
    facts[key] = answer
    return answer


def _verify(what: str, kept: Any, fresh: Any) -> None:
    STATS["verified"] += 1
    if kept != fresh:
        raise TreeMemoMismatch(f"tree memo: the reused {what} differs from working it out again")
