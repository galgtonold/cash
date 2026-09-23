"""Names a statement binds to an object another variable already holds.

A cache hit rebinds such a name to a deserialised copy, so the identity the
alias relied on breaks; these are refused. Pure AST: which names are bound to
modules, the one fact about the namespace :func:`reference_alias_targets`
needs, is passed in by the caller.
"""

from __future__ import annotations

import ast

from .mutations import _module_level_stmts

__all__ = ["aliased_sources", "bare_alias_targets", "reference_alias_targets"]


def _cell_alias_map(tree: ast.Module) -> dict[str, str]:
    """Map each alias name in the cell to its direct source name (shared object).

     Recognises every binding form that makes the target share the RHS object,
     not just ``y = x``:

     * simple / chained ``Name`` assignment — ``y = x``, ``a = b = x`` (each
       target aliases x);
     * 1:1 tuple / list unpack of a literal — ``(y,) = (x,)``, ``a, b = c, d``
       (element-wise, only ``Name``-to-``Name`` pairs);
     * walrus binding — ``(y := x).append(..)``.

     Bindings inside control-flow bodies (if / for / while / with / try) are
     scanned too — an alias formed in a loop body still shares the object
    . Deferred scopes (def / class) are not descended into. Only a bare
     ``Name`` RHS counts as aliasing; ``y = x.copy()`` / ``y = x[:]`` are copies
     and excluded. Self-binds (``x = x``) are skipped. A ternary
     (``y = x if c else z``) is intentionally not handled here (flow-sensitive,
     two possible sources) — tracked separately.
    """
    alias_map: dict[str, str] = {}

    def _bind(target: ast.AST, value: ast.AST) -> None:
        if isinstance(target, ast.Name) and isinstance(value, ast.Name) and target.id != value.id:
            alias_map[target.id] = value.id
        elif (
            isinstance(target, (ast.Tuple, ast.List))
            and isinstance(value, (ast.Tuple, ast.List))
            and len(target.elts) == len(value.elts)
        ):
            # NESTED 1:1 literal unpack -- ``(p, (q,)) = (x, (y,))``. Nesting
            # changes the shape of the unpack, not the aliasing: every leaf still
            # shares its partner's object. Binding only the outer level left
            # ``q`` unmapped, so a mutation through it (``q.append(9)``) was not
            # attributed to ``y`` and an idempotent re-run appended twice.
            for elt_target, elt_value in zip(target.elts, value.elts):
                _bind(elt_target, elt_value)

    for node in _module_level_stmts(tree.body):
        # Walrus binding in this statement's own expressions. (Nested control
        # bodies are visited as their own yielded statements, so restrict the
        # walk to NamedExprs that are not themselves inside a deferred scope.)
        for sub in ast.walk(node):
            if isinstance(sub, ast.NamedExpr):
                _bind(sub.target, sub.value)
        if not isinstance(node, ast.Assign):
            continue
        # Simple or chained ``Name`` assignment: ``y = x`` / ``a = b = x``.
        if isinstance(node.value, ast.Name):
            for tgt in node.targets:
                _bind(tgt, node.value)
        # 1:1 literal unpack: ``(y,) = (x,)`` / ``a, b = c, d`` (Name pairs only).
        for tgt in node.targets:
            if (
                isinstance(tgt, (ast.Tuple, ast.List))
                and isinstance(node.value, (ast.Tuple, ast.List))
                and len(tgt.elts) == len(node.value.elts)
            ):
                for te, ve in zip(tgt.elts, node.value.elts):
                    _bind(te, ve)
    return alias_map


def _resolve_alias_root(name: str, alias_map: dict[str, str]) -> str:
    """Follow the (transitive) alias chain from *name* to its root source."""
    cur, seen = name, {name}
    while cur in alias_map and alias_map[cur] not in seen:
        cur = alias_map[cur]
        seen.add(cur)
    return cur


def aliased_sources(tree: ast.Module | None, names) -> frozenset[str]:
    """Root source names that *names* alias via a top-level ``Name = Name`` bind.

    For each name that resolves through the alias map to a different root, return
    that root. Used to extend an existing mutation set (selfref writes, method
    receivers) from an in-cell alias back to the upstream holder it shares an
    object with, so the holder resets on an isolated re-run. Names that
    are not aliases contribute nothing.
    """
    if tree is None or not names:
        return frozenset()
    alias_map = _cell_alias_map(tree)
    if not alias_map:
        return frozenset()
    out: set[str] = set()
    for name in names:
        root = _resolve_alias_root(name, alias_map)
        if root != name:
            out.add(root)
    return frozenset(out)


def bare_alias_targets(tree: ast.Module | None) -> frozenset[str]:
    """Names bound by a top-level statement that does NOTHING but pointer-copy a
    bare ``Name`` — ``b = a``, ``b = c = a``, ``b, c = a, d``.

    Such a statement must never be cached. Two independent reasons, either alone
    sufficient:

    * **Correctness.** ``b = a`` binds *the same object* to a second name; Python
      guarantees ``b is a``. A cache hit rebinds ``b`` to a DESERIALISED COPY, so
      the two names silently stop being the same object and a later mutation
      through ``a`` (``a.fit(..)``, ``a.append(..)``) is invisible through ``b``.
      The window opens on the SECOND warm re-run — the first re-run still
      re-executes — which is why a one-repetition test reports it as working.
    * **Cost.** A pointer copy is free. Caching it serialises and deserialises a
      whole object to avoid a nanosecond of work, so refusing is also a strictly
      cheaper default. There is no configuration in which caching this shape wins.

    Deliberately NARROW — only bindings that are provably pure pointer copies:

    * every target of a ``Name``-valued assign must itself be a plain ``Name``, so
      ``b, c = a`` (an unpack that INDEXES ``a``, binding ``a[0]``/``a[1]``, not
      ``a``) is excluded;
    * the tuple form is restricted to a 1:1 literal unpack of bare ``Name``\\ s, so
      ``b, c = a, f()`` (``f()`` is real work worth caching) is excluded;
    * anything computed from a name — ``b = a.attr``, ``b = a[0]``, ``b = f(a)``,
      ``b = a.copy()`` — is excluded. Those CAN alias a live mutable object too,
      but they can equally be expensive, so the cost half of the argument does not
      transfer and they keep their cache.

    Self-binds (``x = x``) are skipped, matching :func:`_cell_alias_map`.

    The statement still executes and still participates in lineage: the caller
    only refuses to store/restore the value (``capture_and_track_variables`` runs
    unconditionally), so downstream cache keys and the upstream simulation are
    unaffected.
    """
    if tree is None:
        return frozenset()
    out: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        # ``b = a`` / ``b = c = a``: each target names the very same object.
        if isinstance(value, ast.Name):
            if all(isinstance(t, ast.Name) for t in node.targets):
                out.update(t.id for t in node.targets if isinstance(t, ast.Name) and t.id != value.id)
            continue
        # ``b, c = a, d``: the RHS tuple is built and unpacked element-wise, so
        # every binding is its own pointer copy. Requires equal arity and bare
        # ``Name``\\ s on both sides (no ``*rest``, no computed element).
        if isinstance(value, (ast.Tuple, ast.List)) and len(node.targets) == 1:
            aliases = _literal_unpack_aliases(node.targets[0], value)
            if aliases:
                out.update(aliases)
    return frozenset(out)


def _literal_unpack_aliases(target: ast.expr, value: ast.expr) -> set[str] | None:
    """Names pointer-copied by a 1:1 literal unpack, recursing through nesting.

    ``b, c = a, d`` binds every name to the very same object the matching RHS
    name holds — and so does ``(p, (q,)) = (x, (y,))``. Nesting changes the shape
    of the unpack, not the fact that each leaf is a pointer copy.

    Handling only the FLAT form left the nested one looking like ordinary work,
    so it kept its cache. A later mutation through the nested alias
    (``q.append(9)``) was then recorded against a name cash did not know aliased
    ``y``, and an idempotent re-run appended a second time — ``[3, 4, 9, 9]``.

    Returns ``None`` when the shape is not a pure 1:1 literal unpack, which keeps
    the all-or-nothing rule this generalises: one ``*rest`` or computed element
    anywhere opts the whole statement out, exactly as before, so a genuinely
    expensive element (``b, c = a, f()``) keeps its cache rather than being
    refused because a sibling happened to be an alias.
    """
    if isinstance(target, ast.Name) and isinstance(value, ast.Name):
        return {target.id} if target.id != value.id else set()
    if (
        isinstance(target, (ast.Tuple, ast.List))
        and isinstance(value, (ast.Tuple, ast.List))
        and len(target.elts) == len(value.elts)
    ):
        found: set[str] = set()
        for elt_target, elt_value in zip(target.elts, value.elts):
            nested = _literal_unpack_aliases(elt_target, elt_value)
            if nested is None:
                return None
            found |= nested
        return found
    return None


def _is_free_reference_expr(node: ast.expr) -> bool:
    """True for an expression that only DEREFERENCES existing state.

    These are the shapes that share BOTH halves of the argument, which is
    what makes them safe to refuse:

    * they alias — the result can be a live sub-object of a tracked variable, so
      restoring a deserialised copy silently breaks the identity Python
      guarantees; and
    * they are FREE to re-run — an attribute lookup or a constant-key subscript
      is a pointer dereference, so refusing to cache is also strictly cheaper.

    ``bare_alias_targets``' docstring lumps these in with ``b = f(a)`` and
    excludes them all on the grounds that they "can equally be expensive". That
    is true of a CALL, but not of a deref: whatever built ``obj.inner`` is cached
    at its own statement; re-reading the attribute costs nothing.

    Accepted, rooted in a bare ``Name`` so the base is a live tracked variable:

    * ``a.attr``, ``a.b.c``        — attribute chains
    * ``a[0]``, ``a['k']``         — subscript with a LITERAL key
    * ``a if cond else b``         — ternary whose branches are themselves free

    Deliberately NOT accepted:

    * any ``Call`` (``f(a)``, ``list(a)``, ``a.copy()``) — may do real work, so
      the cost half does not transfer. ``b = list(a)`` additionally aliases only
      one level down (``b[0] is a[0]`` while ``b is not a``), which refusing to
      cache the binding would not fix anyway.
    * a subscript with a NON-literal key (``a[i]``, ``a[mask]``, ``a[1:]``) —
      ``df[mask]`` is a filter that does real work and must keep its cache.
    """
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.Attribute):
        return _is_free_reference_expr(node.value)
    if isinstance(node, ast.Subscript):
        # Literal key only: a computed key can be an expensive filter.
        key = node.slice
        if not isinstance(key, ast.Constant):
            return False
        return _is_free_reference_expr(node.value)
    if isinstance(node, ast.IfExp):
        return _is_free_reference_expr(node.body) and _is_free_reference_expr(node.orelse)
    if isinstance(node, ast.Constant):
        return True  # the ``else None`` arm of a ternary
    return False


def _root_name(node: ast.expr) -> str | None:
    """The base variable a dereference chain is rooted in, or None."""
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def reference_alias_targets(
    tree: ast.Module | None,
    module_names: frozenset[str] | set[str] = frozenset(),
) -> frozenset[str]:
    """Names bound by a pure DEREFERENCE of live state — the dereference half.

    ``b = obj.inner`` / ``b = holder['k']`` / ``b = lst[0]`` / ``b = obj if c
    else None`` each bind the *same object* that is already reachable through a
    tracked variable. A cache hit rebinds the target to a deserialised copy, so
    ``b is obj.inner`` stops holding and a later mutation through ``obj`` is
    invisible through ``b`` — measured divergence from a ``%cash_off`` kernel on
    the FIRST warm re-run, surviving a kernel restart.

    Same enforcement as :func:`bare_alias_targets`: the statement still executes
    and still participates in lineage; only store/restore is refused.

    *module_names* are the names bound to modules right now; a dereference of
    one is exempt (below). Left empty, every dereference is refused.
    """
    if tree is None:
        return frozenset()
    out: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        # The RHS must itself be a DEREFERENCE. Gating on the top-level node type
        # matters: ``_is_free_reference_expr`` accepts a bare ``Name`` (a valid
        # base) and a ``Constant`` (a valid ternary arm), but neither can alias
        # as a whole RHS -- ``b = a`` is bare_alias_targets' job, and ``x = 21``
        # binds a fresh immutable. Accepting Constant here refused caching for
        # every plain literal assignment in the suite.
        if not isinstance(node.value, (ast.Attribute, ast.Subscript, ast.IfExp)):
            continue
        if not _is_free_reference_expr(node.value):
            continue
        # A MODULE attribute (``v = mod.VERSION``) is not an alias hazard worth
        # refusing: module-level names are overwhelmingly immutable constants and
        # functions, so there is no live object whose identity a restore could
        # break -- and granular module-dependency invalidation relies on these
        # bindings being cached. Only skip when the caller SAW it is a module.
        if _root_name(node.value) in module_names:
            continue
        if all(isinstance(t, ast.Name) for t in node.targets):
            out.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return frozenset(out)
