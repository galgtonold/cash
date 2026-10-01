"""Module globals a cached function reads that its module changes.

A global that some function in the module reassigns or mutates can differ
between two calls with the same arguments, while the cached result stays the
first one. `mutable_global_reads` reports each such read; a global nothing
writes (a constant, a dispatch table) is not reported.
"""

from __future__ import annotations

import ast
import functools
import inspect
import logging
import textwrap
from collections.abc import Callable
from typing import Any

from .._memo import MODULE_ANALYSES, LruMemo
from ..exceptions import SOURCE_RETRIEVAL_ERRORS
from ..source_norm import settled_source_version, source_version_unchanged
from ..value_types import BUILTIN_NAMES
from .callee_effects import module_function_global_changes, scope_locals
from .purity_report import ISSUE_MUTABLE_GLOBAL, PurityIssue

logger = logging.getLogger(__name__)


def _imported_module_names(tree: ast.AST) -> frozenset[str]:
    """Names bound by a plain ``import x`` / ``import x as y`` in *tree*.

    ``import os.path`` binds ``os``, so the top-level segment is what counts.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return frozenset(names)


def module_modified_globals(module: Any) -> frozenset[str]:
    """Module-global names that are reassigned/mutated somewhere in *module*.

    Empty when the source can't be read, so nothing is flagged on incomplete
    information. The scan is memoised on the source text, so a module edited
    under a running process is scanned again; and, once its file has settled,
    on the file's version, so each helper the walk meets in a big module does
    not read and hash the whole file again to find the scan it already did.
    """
    version = settled_source_version(module)
    if version is not None:
        hit = _MODULE_MUTATIONS.get(version)
        if hit is not None:
            return hit
    try:
        source = inspect.getsource(module)
    except SOURCE_RETRIEVAL_ERRORS:
        return frozenset()
    modified = modified_globals_in_source(source)
    if version is not None and source_version_unchanged(version):
        _MODULE_MUTATIONS[version] = modified
    return modified


#: `module_modified_globals` per module file version: (path, mtime_ns, size).
_MODULE_MUTATIONS: LruMemo[tuple[str, int, int], frozenset[str]] = LruMemo(MODULE_ANALYSES)


@functools.lru_cache(maxsize=256)
def modified_globals_in_source(source: str) -> frozenset[str]:
    try:
        tree = ast.parse(textwrap.dedent(source))
    except (SyntaxError, ValueError):
        return frozenset()
    # Changes made by code at module level run once, at import, before any
    # cached function is called, so a registry filled at import reads as
    # constant; only function bodies count. A function that only ever runs
    # at import (a decorator body) still counts: a false flag is a warning, a
    # missed one a stale cache. A method call on a plainly imported module
    # (``requests.post``) calls a function and does not change the module;
    # ``from config import SETTINGS`` binds an object, which still counts.
    try:
        return module_function_global_changes(tree, _imported_module_names(tree))
    except RecursionError:
        logger.debug("global-mutation scan gave up on a deeply nested module")
        return frozenset()


def mutable_global_reads(
    func: Callable[..., Any],
    func_def: ast.AST,
    qualname: str,
    read_names: dict[str, None],
) -> list[PurityIssue]:
    """An issue for each module global *func* reads that is reassigned or
    mutated elsewhere in its module - the result would go stale when that
    global changes. Reads of never-written globals (constants, dispatch
    tables) are not flagged."""
    if not read_names:
        return []
    module = inspect.getmodule(func)
    if module is None:
        return []
    modified = module_modified_globals(module)
    if not modified:
        return []
    module_ns = getattr(func, "__globals__", None) or {}
    locals_ = scope_locals(func_def)
    freevars = set(getattr(getattr(func, "__code__", None), "co_freevars", ()) or ())
    own_name = getattr(func, "__name__", None)
    candidates = (read_names.keys() & modified) - locals_ - freevars - BUILTIN_NAMES
    return [
        PurityIssue(
            kind=ISSUE_MUTABLE_GLOBAL,
            description=(
                f"reads module global {name!r} that is reassigned or mutated "
                f"elsewhere - cached results won't reflect changes to it; pass "
                f"it as an argument or declare it via depends_on"
            ),
            where=qualname,
            line=0,
            subject=name,
        )
        for name in sorted(candidates)
        if name != own_name and name in module_ns
    ]
