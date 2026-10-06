"""Types and pure AST helpers every control-structure module shares.

A leaf: it imports nothing from this package, so the orchestrator and the
handlers can all import it at module level.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ...value_hash import compute_hash
from ...value_types import IMMUTABLE_PRIMS

if TYPE_CHECKING:
    from ..statement import ProcessResult


@dataclass
class ControlStructureResult:
    """Result from executing a control structure."""

    success: bool
    metrics: list[ProcessResult]  # Metrics for each processed statement
    error: Exception | None = None
    total_iterations: int = 0
    cached_iterations: int = 0
    computed_iterations: int = 0


def is_control_structure(node: ast.AST) -> bool:
    """Check if an AST node is a control structure that should be processed.

    A structure whose header binds a name with ``:=`` is not: the handlers
    evaluate an ``if``/``elif`` test or a ``for`` iterable themselves, outside
    any statement, so the name it binds would get no lineage and an edit
    above it would never re-run it. Run as one plain statement, its header
    is an output like any other assignment's, at the runtime and in the
    upstream simulation alike, which both ask here.
    """
    if not isinstance(node, (ast.For, ast.While, ast.If, ast.With, ast.Try)):
        return False
    return not _header_binds(node)


def _header_binds(node: ast.AST) -> bool:
    """Whether an ``if``/``elif`` test or a ``for`` iterable of *node* holds a ``:=``."""
    headers: list[ast.AST] = []
    if isinstance(node, (ast.For, ast.AsyncFor)):
        headers.append(node.iter)
    current = node
    while isinstance(current, ast.If):
        headers.append(current.test)
        current = current.orelse[0] if len(current.orelse) == 1 else None
    return any(isinstance(sub, ast.NamedExpr) for header in headers for sub in ast.walk(header))


def get_control_structure_type(node: ast.AST) -> str | None:
    """Get the type of control structure for an AST node."""
    if isinstance(node, ast.For):
        return "for"
    if isinstance(node, ast.While):
        return "while"
    if isinstance(node, ast.If):
        return "if"
    if isinstance(node, ast.With):
        return "with"
    if isinstance(node, ast.Try):
        return "try"
    return None


def contains_break_or_continue(nodes: list[ast.AST]) -> bool:
    """
    Check if any of the given AST nodes (or their children) contain break or
    continue statements.

    These statements cannot be executed outside of a loop context, so loops
    containing them must be executed as a single unit.
    """
    for node in nodes:
        for child in ast.walk(node):
            if isinstance(child, (ast.Break, ast.Continue)):
                return True
    return False


def contains_top_level_await(node: ast.AST) -> bool:
    """True if *node* holds an ``await`` / ``async for`` / ``async with`` that
    would need ``PyCF_ALLOW_TOP_LEVEL_AWAIT`` to compile — i.e. one that is NOT
    nested inside a ``def`` / ``async def`` / ``lambda`` (those own their own
    coroutine scope and compile fine without the flag).

    ``ast.walk`` is scope-blind, so this recurses manually and prunes function
    bodies. Used to detect a control structure whose body awaits (``for x in xs:
    r = await fetch(x)``) — the one shape the per-iteration / sync single-unit
    path cannot compile, which must instead run as ONE awaited unit.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.Await, ast.AsyncFor, ast.AsyncWith)):
            return True
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue  # separate scope — its awaits are not top-level here
        if contains_top_level_await(child):
            return True
    return False


# --- Helper Functions (Module Level) ---


def extract_target_names(target: ast.AST) -> list[str]:
    """Extract variable names from a for loop target."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names = []
        for elt in target.elts:
            names.extend(extract_target_names(elt))
        return names
    return []


def bind_target_values(target: ast.AST, value, user_ns: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively bind loop target variables to their values,
    respecting nested tuple/list unpacking structure.

    For ``for a, (b, c) in data``, the AST target is
    Tuple(Name('a'), Tuple(Name('b'), Name('c'))) and the
    iteration value is e.g. (1, (2, 3)).  This function
    correctly maps a->1, b->2, c->3 and assigns into user_ns.

    Returns dict of {name: value} for all bound variables.
    """
    bindings: dict[str, Any] = {}
    if isinstance(target, ast.Name):
        user_ns[target.id] = value
        bindings[target.id] = value
    elif isinstance(target, (ast.Tuple, ast.List)):
        if any(isinstance(elt, ast.Starred) for elt in target.elts):
            return _bind_by_assignment(target, value, user_ns)
        vals = list(value)
        if len(vals) != len(target.elts):
            raise _unpack_error(len(target.elts), len(vals))
        for elt, val in zip(target.elts, vals, strict=True):
            bindings.update(bind_target_values(elt, val, user_ns))
    elif isinstance(target, ast.Starred):
        if isinstance(target.value, ast.Name):
            user_ns[target.value.id] = value
            bindings[target.value.id] = value
    return bindings


def _unpack_error(expected: int, got: int) -> ValueError:
    """The error Python raises when ``a, b = value`` finds the wrong count."""
    if got > expected:
        return ValueError(f"too many values to unpack (expected {expected})")
    return ValueError(f"not enough values to unpack (expected {expected}, got {got})")


def _bind_by_assignment(target: ast.AST, value: Any, user_ns: dict[str, Any]) -> dict[str, Any]:
    """Bind a target with a starred part by running the assignment itself,
    so ``for a, *rest, z in rows`` unpacks, and fails, as Python does."""
    scratch = "__cash_loop_value__"
    assign = ast.Assign(targets=[target], value=ast.Name(id=scratch, ctx=ast.Load()))
    module = ast.fix_missing_locations(ast.Module(body=[assign], type_ignores=[]))
    user_ns[scratch] = value
    try:
        exec(compile(module, "<cash: loop target>", "exec", dont_inherit=True), user_ns)
    finally:
        user_ns.pop(scratch, None)
    return {name: user_ns[name] for name in extract_target_names(target) if name in user_ns}


#: Context entries under this prefix hold the content digest of a hashable,
#: non-primitive loop value kept by value under its own name. The dunder
#: prefix keeps them out of ``loop_vars``, like ``__iterable_lineage__``.
_DIGEST_PREFIX = "__digest__:"


def _is_primitive(value: Any) -> bool:
    # A primitive's ``repr`` is its full value; any other loop value goes into
    # the context hash by content digest instead.
    if type(value) in IMMUTABLE_PRIMS:
        return True
    return type(value) is tuple and all(_is_primitive(v) for v in value)


def build_iteration_context(
    target_names: list[str],
    user_ns: dict[str, Any],
    parent_context: dict[str, Any] | None,
    loop_var_digests: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Build a context dict containing current iteration variable values.

    Used to differentiate cache keys across loop iterations.

    *loop_var_digests* carries full-content digests the caller has ALREADY
    computed for these same bindings (``for_handler._process_one_iteration``
    fills it), so a large loop value is not hashed a second time per
    iteration.

    Primitive values go in by value. An unhashable value is replaced by its
    digest. A hashable, non-primitive value (a user object, a numpy scalar)
    stays by value, because ``loop_vars`` hands the live object to call
    caching, and its digest rides alongside under ``__digest__:<name>`` for
    :func:`compute_context_hash`. Its ``str()`` is no key: a default
    ``repr`` carries the memory address, which changes every run and never
    changes on mutation.
    """
    context = dict(parent_context) if parent_context else {}
    digests = loop_var_digests or {}

    for name in target_names:
        if name not in user_ns:
            continue
        value = user_ns[name]
        # A nested loop may rebind an outer loop's name; drop the outer digest.
        context.pop(_DIGEST_PREFIX + name, None)
        if _is_primitive(value):
            context[name] = value
            continue
        # repr() TRUNCATES large numpy/pandas objects and embeds addresses for
        # plain objects, so hash the full content. Reuse the caller's digest
        # when it has one: it is the same `compute_hash` of the same object.
        digest = digests.get(name)
        if digest is None:
            digest = compute_hash(value)
        try:
            hash(value)
        except TypeError:
            context[name] = digest
        else:
            context[name] = value
            context[_DIGEST_PREFIX + name] = digest

    return context


def compute_context_hash(context: dict[str, Any]) -> str:
    """Compute a hash of the iteration context.

    Primitive values are hashed by ``str()``; any other value by its content
    digest, taken from the ``__digest__:`` entry
    :func:`build_iteration_context` stored, or computed here if absent.
    """
    items = []
    for key, value in context.items():
        if key.startswith(_DIGEST_PREFIX):
            continue
        if not _is_primitive(value):
            digest = context.get(_DIGEST_PREFIX + key)
            if digest is None:
                digest = compute_hash(value)
            value = digest
        items.append((key, value))
    context_str = str(sorted(items))
    return hashlib.sha256(context_str.encode("utf-8")).hexdigest()[:16]
