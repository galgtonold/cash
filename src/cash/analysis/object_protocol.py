"""Mutations hidden behind the object protocol.

``obj.x = v`` running a ``__setattr__``, ``obj[k]`` a ``__getitem__``, ``a += b``
an ``__iadd__``: what a user class's dunder methods and properties change, read
from the class source. Pure AST.
"""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass

from .callee_effects import all_param_names, factory_body_scope, free_vars_mutated_in_function, resolve_function_def
from .mutations import (
    DEFERRED_SCOPES,
    MUTATING_METHODS,
    PANDAS_INPLACE_METHODS,
    extract_base_name,
    iter_store_targets,
)

__all__ = ["object_protocol_mutations"]


# ---------------------------------------------------------------------------
# Object-protocol hidden state
#
# A ``with`` statement, a custom-dunder operation (``s[k]=v`` / ``del s[k]`` /
# ``v=s[k]`` / ``a(x)``), a constructor, a decorated call, or an instance /
# class method invokes a user-defined method whose body mutates hidden state.
# The mutation is invisible to the cell text, so on an isolated re-run it
# accumulates. This generalises the earlier case (hidden state via a called function) to
# the object protocol: resolve the receiver's class (or the wrapper / context
# manager), analyse the invoked method body, and attribute the mutation to one
# of three reset channels:
#
#   * **free_vars**   — a module/free variable the method mutates (``log``);
# reset like the A path (add to the cell's mutated set + inputs so the
#     producer's cell-entry base is restored).
#   * **receivers**   — the receiver INSTANCE whose ``self`` attribute the
#     method mutates in place (``cm.n += 1``); reset like a method receiver.
#   * **class_defs**  — the CLASS whose class variable the method mutates
#     (``Reg.registry.append`` / ``cls.log.append``); reset by re-running the
#     class ``def`` so the class-level container is recreated fresh.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectProtocolResets:
    """Reset targets attributed to hidden object-protocol mutations, one set per
    reset channel (see the module-section comment above)."""

    free_vars: frozenset[str]
    receivers: frozenset[str]
    class_defs: frozenset[str]
    # Free vars mutated by a base ``__init_subclass__`` hook fired during class
    # creation. Kept apart from ``free_vars`` because — unlike an
    # ordinary free-var mutation, whose upstream occurrences the simulator
    # sees as top-level statements — this mutation is hidden behind class creation
    # in EVERY subclass cell, so the simulator's content-base cross-cell guard is
    # blind to it. The caller applies the cross-cell suppression before
    # routing these to the (otherwise self-protecting) free-var reset.
    init_subclass_free_vars: frozenset[str] = frozenset()


def _first_param_name(func: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """The name of *func*'s first positional parameter (``self`` / ``cls``), or
    ``None`` for a zero-arg function."""
    positional = list(func.args.posonlyargs) + list(func.args.args)
    return positional[0].arg if positional else None


def _has_named_decorator(func: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> bool:
    """True if *func* carries a ``@name`` decorator (bare or called)."""
    for dec in func.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == name:
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == name:
            return True
    return False


def _resolve_class_def(name, resolve_class_source) -> ast.ClassDef | None:
    """Parse *name*'s source via *resolve_class_source* into a ClassDef, or None."""
    if not name or resolve_class_source is None:
        return None
    source = resolve_class_source(name)
    if not source:
        return None
    try:
        parsed = ast.parse(textwrap.dedent(source))
    except (SyntaxError, ValueError):
        return None
    if parsed.body and isinstance(parsed.body[0], ast.ClassDef):
        return parsed.body[0]
    return None


def _class_bases(classdef: ast.ClassDef) -> list[str]:
    """The bare-``Name`` base-class names of *classdef* (in declaration order).
    Non-``Name`` bases (``Generic[T]``, ``a.B``) are skipped — only notebook
    classes resolvable by name are followed."""
    return [b.id for b in classdef.bases if isinstance(b, ast.Name)]


def _iter_class_hierarchy(classdef: ast.ClassDef | None, resolve_class_source, _seen: set[str] | None = None):
    """Yield *classdef* and its resolvable base classes, depth-first.

    Follows each ``Name`` base via *resolve_class_source*, so an inherited method
    / class variable is seen. Cycle-guarded by class name. When
    *resolve_class_source* is ``None`` (unit-test callers), only *classdef* is
    yielded — the original own-class-only behaviour."""
    if classdef is None:
        return
    if _seen is None:
        _seen = set()
    if classdef.name in _seen:
        return
    _seen.add(classdef.name)
    yield classdef
    if resolve_class_source is None:
        return
    for base_name in _class_bases(classdef):
        base = _resolve_class_def(base_name, resolve_class_source)
        if base is not None:
            yield from _iter_class_hierarchy(base, resolve_class_source, _seen)


def _class_method(
    classdef: ast.ClassDef, method_name: str, resolve_class_source=None
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """The method named *method_name* on *classdef* or an inherited base (the
    first match walking the hierarchy)."""
    for cls in _iter_class_hierarchy(classdef, resolve_class_source):
        for node in cls.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
                return node
    return None


def _own_class_level_attr_names(classdef: ast.ClassDef) -> frozenset[str]:
    """Class variables assigned directly in *classdef*'s body (not inherited)."""
    out: set[str] = set()
    for node in classdef.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                for leaf in iter_store_targets(tgt):
                    if isinstance(leaf, ast.Name):
                        out.add(leaf.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.add(node.target.id)
    return frozenset(out)


def _class_level_attr_names(classdef: ast.ClassDef, resolve_class_source=None) -> frozenset[str]:
    """Class variables of *classdef* including those inherited from base classes
    (``registry = []``, ``count = 0``, ``ClassVar[...] = []``). These live on the
    owning class object and are shared by every instance, so mutating one
    accumulates until that class ``def`` re-runs."""
    out: set[str] = set()
    for cls in _iter_class_hierarchy(classdef, resolve_class_source):
        out |= _own_class_level_attr_names(cls)
    return frozenset(out)


def _instance_attr_names(classdef: ast.ClassDef, resolve_class_source=None) -> frozenset[str]:
    """Attribute names assigned as ``self.<attr> = ...`` in any method of
    *classdef* or an inherited base — the per-instance attributes. A method
    mutating one of these mutates the receiver instance (reset the receiver), not
    the shared class."""
    out: set[str] = set()
    for cls in _iter_class_hierarchy(classdef, resolve_class_source):
        for meth in cls.body:
            if not isinstance(meth, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            recv = _first_param_name(meth)
            if recv is None:
                continue
            for node in ast.walk(meth):
                if not isinstance(node, ast.Assign):
                    continue
                for tgt in node.targets:
                    for leaf in iter_store_targets(tgt):
                        if (
                            isinstance(leaf, ast.Attribute)
                            and isinstance(leaf.value, ast.Name)
                            and leaf.value.id == recv
                        ):
                            out.add(leaf.attr)
    return frozenset(out)


def _class_var_owner(attr: str, classdef: ast.ClassDef, resolve_class_source) -> str | None:
    """The name of the class in *classdef*'s hierarchy that DEFINES class variable
    *attr* at its own body level — the class whose ``def`` must re-run to reset it
    (``Base.registry`` is owned by ``Base`` even when mutated via ``Sub``).
    The nearest defining class wins; ``None`` if no class defines it."""
    for cls in _iter_class_hierarchy(classdef, resolve_class_source):
        if attr in _own_class_level_attr_names(cls):
            return cls.name
    return None


def _property_accessor(
    classdef: ast.ClassDef, attr: str, kind: str, resolve_class_source=None
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """The ``@property`` getter (``kind='getter'``) or ``@<attr>.setter`` setter
    (``kind='setter'``) named *attr* in *classdef*'s hierarchy, or None.

    A property defines two methods both named *attr*: the getter carries
    ``@property`` and the setter ``@<attr>.setter``. ``_class_method`` would return
    whichever comes first, so the accessors are matched by their decorator."""
    for cls in _iter_class_hierarchy(classdef, resolve_class_source):
        for node in cls.body:
            if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == attr):
                continue
            if kind == "getter" and _has_named_decorator(node, "property"):
                return node
            if kind == "setter":
                for dec in node.decorator_list:
                    if (
                        isinstance(dec, ast.Attribute)
                        and dec.attr == "setter"
                        and isinstance(dec.value, ast.Name)
                        and dec.value.id == attr
                    ):
                        return node
    return None


def _descriptor_class(classdef: ast.ClassDef, attr: str, resolve_class_source) -> ast.ClassDef | None:
    """The ClassDef of the data descriptor bound to class attribute *attr*
    (``field = Tracked()`` → ``Tracked``'s ClassDef), or None. Accessing
    ``obj.field`` / assigning ``obj.field = v`` dispatches to that class's
    ``__get__`` / ``__set__``."""
    for cls in _iter_class_hierarchy(classdef, resolve_class_source):
        for node in cls.body:
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == attr for t in node.targets)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
            ):
                return _resolve_class_def(node.value.func.id, resolve_class_source)
    return None


def _walk_executable(node: ast.AST):
    """Yield *node* and every descendant that executes when it runs, WITHOUT
    descending into deferred scopes (def/async def/class). Used to scan a cell or
    a method body for the operations that actually run, skipping nested
    definitions."""
    yield node
    for child in ast.iter_child_nodes(node):
        if isinstance(child, DEFERRED_SCOPES):
            continue
        yield from _walk_executable(child)


def _iter_method_body_nodes(method: ast.FunctionDef | ast.AsyncFunctionDef):
    """Yield every executable node in *method*'s body (skipping nested scopes)."""
    for stmt in method.body:
        yield from _walk_executable(stmt)


def _first_attr_after_root(chain: ast.expr) -> str | None:
    """For a receiver chain like ``self.data`` / ``cls.log`` / ``self.cache[k]``
    return the attribute name attached directly to the root ``Name``
    (``data`` / ``log`` / ``cache``), or ``None`` if the root is bare."""
    node: ast.AST = chain
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        if isinstance(node.value, ast.Name):
            return node.attr if isinstance(node, ast.Attribute) else None
        node = node.value
    return None


def _iter_inplace_mutation_chains(method: ast.FunctionDef | ast.AsyncFunctionDef):
    """Yield the receiver-chain expr of every IN-PLACE mutation in *method*'s
    body: a known-mutating method call, an augmented assignment, a subscript
    assignment / delete, or a pandas ``inplace=True`` call. Plain attribute
    rebinds (``self.x = value`` — construction / idempotent re-set) are excluded
    so a constructor is not flagged."""
    for node in _iter_method_body_nodes(method):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr in MUTATING_METHODS:
                yield node.func.value
            elif attr in PANDAS_INPLACE_METHODS:
                for kw in node.keywords:
                    if kw.arg == "inplace" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        yield node.func.value
                        break
        elif isinstance(node, ast.AugAssign):
            yield node.target
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                for leaf in iter_store_targets(tgt):
                    if isinstance(leaf, ast.Subscript):
                        yield leaf.value
        elif isinstance(node, ast.Delete):
            for tgt in node.targets:
                if isinstance(tgt, ast.Subscript):
                    yield tgt.value


def _super_called_methods(method: ast.FunctionDef | ast.AsyncFunctionDef) -> frozenset[str]:
    """Method names invoked via ``super().<name>(...)`` in *method*'s body — the
    inherited implementations whose hidden mutations must also be attributed
    (``super.__init__`` running ``Base.__init__``)."""
    out: set[str] = set()
    for node in _iter_method_body_nodes(method):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id == "super"
        ):
            out.add(node.func.attr)
    return frozenset(out)


def _classify_method_mutations(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    recv_class_name: str,
    cdef: ast.ClassDef,
    resolve_class_source=None,
    _seen: set[tuple[str, str]] | None = None,
    _current: ast.ClassDef | None = None,
) -> tuple[bool, frozenset[str], frozenset[str]]:
    """Classify the hidden mutations *method* performs into the three reset
    channels: ``(mutates_self_instance, class_reset_targets, free_vars)``.

    * ``self.<attr>`` where ``<attr>`` is a per-instance attribute → instance
      mutation (reset the receiver);
    * ``cls.<attr>`` (classmethod), ``BaseOrOwnClass.<attr>``, or ``self.<attr>``
      where ``<attr>`` is a class-level variable → class-variable mutation. The
      reset target is the class that OWNS the variable (a base class when the var
      is inherited), so ``class_reset_targets`` is a SET of class names;
    * a module/free variable (neither a parameter nor a local) → free-var
      mutation (reset via the A path).

    Inheritance is followed: *cdef* + *resolve_class_source* give hierarchy-aware
    attribute sets and owner lookup, and ``super().<m>()`` calls recurse into the
    base implementation. *cdef* is fixed across the recursion (attr/owner context
    is the receiver's full hierarchy); *_current* is the class whose method is
    being classified, used to resolve ``super()`` against ITS bases. The recursion
    is guarded by ``(current_class, method)`` — a stable key, since
    :func:`_resolve_class_def` re-parses source and yields fresh node ids each
    call. A ``@staticmethod`` has no receiver binding.
    """
    if _current is None:
        _current = cdef
    if _seen is None:
        _seen = set()
    key = (_current.name, method.name)
    if key in _seen:
        return False, frozenset(), frozenset()
    _seen.add(key)

    class_attrs = _class_level_attr_names(cdef, resolve_class_source)
    instance_attrs = _instance_attr_names(cdef, resolve_class_source)
    recv = _first_param_name(method)
    is_classmethod = _has_named_decorator(method, "classmethod")
    is_staticmethod = _has_named_decorator(method, "staticmethod")
    params = all_param_names(method)
    global_decls: set[str] = set()
    local_assigned: set[str] = set()
    for node in _iter_method_body_nodes(method):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            global_decls.update(node.names)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                for leaf in iter_store_targets(tgt):
                    if isinstance(leaf, ast.Name):
                        local_assigned.add(leaf.id)
    local_assigned -= global_decls

    mutates_self = False
    class_targets: set[str] = set()
    free: set[str] = set()

    def _add_class_var(attr):
        owner = _class_var_owner(attr, cdef, resolve_class_source) if attr else None
        class_targets.add(owner or recv_class_name)

    for chain in _iter_inplace_mutation_chains(method):
        root = extract_base_name(chain)
        if root is None:
            continue
        if not is_staticmethod and recv is not None and root == recv:
            first_attr = _first_attr_after_root(chain)
            if is_classmethod:
                _add_class_var(first_attr)
            elif first_attr is not None and first_attr in instance_attrs:
                mutates_self = True
            elif first_attr is not None and first_attr in class_attrs:
                _add_class_var(first_attr)
            else:
                # A bare ``self`` mutation or an attribute assigned nowhere we can
                # see — attribute it to the instance (the conservative reset).
                mutates_self = True
        elif _resolve_class_def(root, resolve_class_source) is not None:
            # ``ClassName.<attr>`` — the class (own or a base) owns the var.
            class_targets.add(root)
        elif root in params or root in local_assigned:
            # A non-receiver parameter (arg-mutation path) or a local.
            continue
        else:
            free.add(root)

    # ``super().<m>()`` runs the inherited implementation — attribute its hidden
    # mutations too (the first resolvable base of the CURRENT class defining <m>
    # wins, MRO-ish).
    for m_name in _super_called_methods(method):
        for base_name in _class_bases(_current):
            base_cdef = _resolve_class_def(base_name, resolve_class_source)
            base_m = _class_method(base_cdef, m_name, resolve_class_source) if base_cdef is not None else None
            if base_m is not None:
                s2, c2, f2 = _classify_method_mutations(
                    base_m,
                    recv_class_name,
                    cdef,
                    resolve_class_source,
                    _seen,
                    base_cdef,
                )
                mutates_self = mutates_self or s2
                class_targets |= c2
                free |= f2
                break

    return mutates_self, frozenset(class_targets), frozenset(free)


def _decorator_free_var_mutations(
    decorator_def: ast.FunctionDef | ast.AsyncFunctionDef,
) -> frozenset[str]:
    """Module/free variables mutated by the wrapper a decorator returns
    . ``def logged(f): def wrap(*a): calls.append('x'); ...; return
     wrap`` — calling a ``@logged``-decorated function runs ``wrap``, which
     appends to the module list ``calls``. Collect the free vars each inner
     function mutates that are NOT local to the decorator (those are the
     closure case)."""
    scope = factory_body_scope(decorator_def)
    out: set[str] = set()
    for node in ast.walk(decorator_def):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not decorator_def:
            out |= free_vars_mutated_in_function(node) - scope
    return frozenset(out)


def _decorator_names(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """The decorator NAMES applied to *func* (``@logged`` → ``logged``,
    ``@app.route(...)`` → ``route``-less, skipped). Bare-name and simple
    ``name(...)`` decorators are resolved by name."""
    names: list[str] = []
    for dec in func.decorator_list:
        if isinstance(dec, ast.Name):
            names.append(dec.id)
        elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
            names.append(dec.func.id)
    return names


# ``obj <op>= x`` dispatches to the in-place operator dunder, falling back to the
# binary form when the in-place one is absent (``obj = obj.__add__(x)``). Maps the
# AST op node name to ``(inplace_dunder, fallback_dunder)``.
_AUGOP_DUNDERS: dict[str, tuple[str, str]] = {
    "Add": ("__iadd__", "__add__"),
    "Sub": ("__isub__", "__sub__"),
    "Mult": ("__imul__", "__mul__"),
    "Div": ("__itruediv__", "__truediv__"),
    "FloorDiv": ("__ifloordiv__", "__floordiv__"),
    "Mod": ("__imod__", "__mod__"),
    "Pow": ("__ipow__", "__pow__"),
    "MatMult": ("__imatmul__", "__matmul__"),
    "BitOr": ("__ior__", "__or__"),
    "BitAnd": ("__iand__", "__and__"),
    "BitXor": ("__ixor__", "__xor__"),
    "LShift": ("__ilshift__", "__lshift__"),
    "RShift": ("__irshift__", "__rshift__"),
}


def object_protocol_mutations(
    tree: ast.Module | None,
    resolve_class_source,
    instance_class,
    resolve_source,
    resolve_var_factory,
    decorated_class=None,
) -> ObjectProtocolResets:
    """Hidden mutations reached through the object protocol.

    Walks the cell's executable nodes and, for each object-protocol invocation —
    a ``with`` statement, a subscript op (``s[k]=v`` / ``del s[k]`` / ``v=s[k]``),
    a call (constructor / instance ``__call__`` / decorated function), or a
    method call — resolves the receiver's class (via *resolve_class_source* +
    *instance_class*), a ``@contextmanager`` generator or decorator (via
    *resolve_source*), or a reassignment-decorator factory (via
    *resolve_var_factory*), analyses the invoked method body, and returns the
    reset targets grouped by channel (see :class:`ObjectProtocolResets`).

    * *resolve_class_source* — ``class_name -> source`` (or ``None``).
    * *instance_class* — ``var -> class_name`` for a ``var = ClassName(...)``
      instance whose class is resolvable (or ``None``).
    * *resolve_source* — ``func_name -> source`` (or ``None``), for
      ``@contextmanager`` generators and decorator functions.
    * *resolve_var_factory* — ``var -> factory FunctionDef`` (or ``None``), for a
      reassignment decorator ``g = counting(g)``.
    * *decorated_class* — ``var -> class_name`` for a class-based decorator
      binding ``@Counter def task`` (or ``None``).
    """
    free_vars: set[str] = set()
    receivers: set[str] = set()
    class_defs: set[str] = set()
    if decorated_class is None:

        def decorated_class(_var):
            return None

    _class_cache: dict[str, ast.ClassDef | None] = {}

    def _classdef(name):
        if name not in _class_cache:
            _class_cache[name] = _resolve_class_def(name, resolve_class_source)
        return _class_cache[name]

    def _apply_method(cdef, class_name, method, recv_var, *, allow_self):
        si, class_targets, fv = _classify_method_mutations(
            method,
            class_name,
            cdef,
            resolve_class_source,
        )
        if fv:
            free_vars.update(fv)
        if class_targets:
            # The class-var reset target is the OWNING class (a base when the var
            # is inherited), which must re-run to recreate the class-level
            # container. When the var is inherited (owner != receiver's class),
            # also reset the receiver's class so its instances re-derive against
            # the fresh base — a subclass method mutating an inherited class var
            # via ``self`` needs both, since the reset cascade is one level deep
            # . For a non-inherited var the owner IS the receiver's class,
            # so this adds nothing.
            class_defs.update(class_targets)
            class_defs.add(class_name)
        if si and allow_self and recv_var is not None:
            receivers.add(recv_var)

    def _apply_ctor(cdef, class_name):
        """A construction ``X()`` — the fresh instance's self-init is discarded, so
        only class-var / free-var mutations in ``__init__`` (or a dataclass
        ``__post_init__``) persist."""
        for ctor in ("__init__", "__post_init__"):
            method = _class_method(cdef, ctor, resolve_class_source)
            if method is not None:
                _apply_method(cdef, class_name, method, None, allow_self=False)

    def _dispatch_dunder(recv_var, dunder):
        cls = instance_class(recv_var)
        cdef = _classdef(cls) if cls else None
        if cdef is None:
            return
        method = _class_method(cdef, dunder, resolve_class_source)
        if method is not None:
            _apply_method(cdef, cls, method, recv_var, allow_self=True)

    def _dispatch_context(recv_var):
        """A context-managed instance ``recv_var`` (``with cm:`` or
        ``stack.enter_context(cm)``): analyse its ``__enter__`` / ``__exit__``."""
        cls = instance_class(recv_var)
        cdef = _classdef(cls) if cls else None
        if cdef is None:
            return
        for dunder in ("__enter__", "__exit__"):
            method = _class_method(cdef, dunder, resolve_class_source)
            if method is not None:
                _apply_method(cdef, cls, method, recv_var, allow_self=True)

    def _return_class(fdef):
        """The ``(name, ClassDef)`` of a notebook class a factory function
         RETURNS (``def cm(): return Mgr()`` → ``Mgr``), or ``(None, None)``
        . Used for ``with cm() as x:`` where ``cm`` is a plain factory."""
        for sub in ast.walk(fdef):
            if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Call) and isinstance(sub.value.func, ast.Name):
                rcdef = _classdef(sub.value.func.id)
                if rcdef is not None:
                    return sub.value.func.id, rcdef
        return None, None

    def _dispatch_descriptor(cdef, attr, dunder):
        """A data descriptor's ``__set__`` / ``__get__`` receives ``self`` (the
        descriptor, a shared class attribute) and ``obj`` (the instance) — both
        parameters — so only its FREE-var side effects are attributed."""
        ddef = _descriptor_class(cdef, attr, resolve_class_source)
        if ddef is None:
            return
        method = _class_method(ddef, dunder, resolve_class_source)
        if method is not None:
            free_vars.update(free_vars_mutated_in_function(method))

    def _dispatch_attr_set(recv_var, attr):
        """``recv.attr = v`` dispatching to a ``@property`` setter or a data
        descriptor's ``__set__``. A plain attribute assign resolves to
        neither and is a no-op."""
        cls = instance_class(recv_var)
        cdef = _classdef(cls) if cls else None
        if cdef is None:
            return
        setter = _property_accessor(cdef, attr, "setter", resolve_class_source)
        if setter is not None:
            _apply_method(cdef, cls, setter, recv_var, allow_self=True)
        else:
            _dispatch_descriptor(cdef, attr, "__set__")

    def _dispatch_attr_get(recv_var, attr):
        """``recv.attr`` (load) dispatching to a ``@property`` getter or a data
        descriptor's ``__get__`` with a side effect."""
        cls = instance_class(recv_var)
        cdef = _classdef(cls) if cls else None
        if cdef is None:
            return
        getter = _property_accessor(cdef, attr, "getter", resolve_class_source)
        if getter is not None:
            _apply_method(cdef, cls, getter, recv_var, allow_self=True)
        else:
            _dispatch_descriptor(cdef, attr, "__get__")

    if tree is None:
        return ObjectProtocolResets(frozenset(), frozenset(), frozenset())

    nodes = list(_walk_executable(tree))

    for node in nodes:
        # --- with statements: __enter__ / __exit__ or a @contextmanager gen ----
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                ctx = item.context_expr
                if isinstance(ctx, ast.Name):
                    _dispatch_context(ctx.id)
                elif isinstance(ctx, ast.Call) and isinstance(ctx.func, ast.Name):
                    nm = ctx.func.id
                    cdef = _classdef(nm)
                    if cdef is not None:
                        # ``with SomeCM():`` — anonymous instance, no receiver to
                        # reset; only class-var / free-var mutations persist.
                        for dunder in ("__enter__", "__exit__"):
                            method = _class_method(cdef, dunder, resolve_class_source)
                            if method is not None:
                                _apply_method(cdef, nm, method, None, allow_self=False)
                    else:
                        fdef = resolve_function_def(nm, resolve_source)
                        if fdef is not None:
                            # A ``@contextmanager`` generator's free-var mutations,
                            free_vars.update(free_vars_mutated_in_function(fdef))
                            # or a plain factory ``def cm(): return Mgr()`` — the
                            # returned instance's __enter__/__exit__ run anonymously
                            # (``with cm() as x:``), so only class-var / free-var
                            # mutations persist.
                            ret_name, ret_cdef = _return_class(fdef)
                            if ret_cdef is not None:
                                for dunder in ("__enter__", "__exit__"):
                                    method = _class_method(ret_cdef, dunder, resolve_class_source)
                                    if method is not None:
                                        _apply_method(ret_cdef, ret_name, method, None, allow_self=False)
        # --- subscript operations dispatching to custom dunders ----------------
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                for leaf in iter_store_targets(tgt):
                    if isinstance(leaf, ast.Subscript) and isinstance(leaf.value, ast.Name):
                        _dispatch_dunder(leaf.value.id, "__setitem__")
                    elif isinstance(leaf, ast.Attribute) and isinstance(leaf.value, ast.Name):
                        _dispatch_attr_set(leaf.value.id, leaf.attr)
        elif isinstance(node, ast.Delete):
            for tgt in node.targets:
                if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name):
                    _dispatch_dunder(tgt.value.id, "__delitem__")
        # --- ``obj <op>= x`` dispatching to an in-place operator dunder ---------
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            cls = instance_class(node.target.id)
            cdef = _classdef(cls) if cls else None
            dunders = _AUGOP_DUNDERS.get(type(node.op).__name__)
            if cdef is not None and dunders is not None:
                inplace_dunder, fallback_dunder = dunders
                method = _class_method(cdef, inplace_dunder, resolve_class_source)
                if method is not None:
                    # ``__iadd__`` mutates the receiver in place (returns self).
                    _apply_method(cdef, cls, method, node.target.id, allow_self=True)
                else:
                    # No in-place form: ``obj += x`` REASSIGNS obj to a fresh
                    # ``obj.__add__(x)`` (idempotent), so only its free-var /
                    # class-var side effects persist.
                    method = _class_method(cdef, fallback_dunder, resolve_class_source)
                    if method is not None:
                        _apply_method(cdef, cls, method, node.target.id, allow_self=False)
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load) and isinstance(node.value, ast.Name):
            _dispatch_dunder(node.value.id, "__getitem__")
        # --- ``recv.attr`` load dispatching to a property getter / __get__ -----
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) and isinstance(node.value, ast.Name):
            _dispatch_attr_get(node.value.id, node.attr)
        # --- calls: constructor / instance __call__ / decorated fn / method ----
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                nm = func.id
                # ``next(it)`` advances the iterator via ``It.__next__``.
                if nm == "next" and node.args and isinstance(node.args[0], ast.Name):
                    _dispatch_dunder(node.args[0].id, "__next__")
                    continue
                cdef = _classdef(nm)
                if cdef is not None:
                    _apply_ctor(cdef, nm)
                    continue
                # A class-based decorator ``@Counter def task`` — ``task`` is a
                # Counter INSTANCE holding the wrapped function, so it is a
                # stateful callable: calling it runs ``Counter.__call__`` (which
                # mutates ``self.n``). Re-run its producer (the decorated def) to
                # reset — the receiver's content is unhashable (holds a function),
                # so route a self-mutation to the class-def channel.
                deco_cls = decorated_class(nm)
                dcdef = _classdef(deco_cls) if deco_cls else None
                if dcdef is not None:
                    call_m = _class_method(dcdef, "__call__", resolve_class_source)
                    if call_m is not None:
                        si, ct, fv = _classify_method_mutations(
                            call_m,
                            deco_cls,
                            dcdef,
                            resolve_class_source,
                        )
                        free_vars.update(fv)
                        class_defs.update(ct)
                        if si:
                            class_defs.add(nm)
                    continue
                # An instance called via __call__ (``a('z')``).
                cls = instance_class(nm)
                icdef = _classdef(cls) if cls else None
                if icdef is not None:
                    call_m = _class_method(icdef, "__call__", resolve_class_source)
                    if call_m is not None:
                        _apply_method(icdef, cls, call_m, nm, allow_self=True)
                # A decorated function whose wrapper mutates a free var, or a
                # reassignment decorator ``g = counting(g)``.
                fdef = resolve_function_def(nm, resolve_source)
                if fdef is not None:
                    for dname in _decorator_names(fdef):
                        ddef = resolve_function_def(dname, resolve_source)
                        if ddef is not None:
                            free_vars.update(_decorator_free_var_mutations(ddef))
                factory = resolve_var_factory(nm) if resolve_var_factory else None
                if factory is not None:
                    free_vars.update(_decorator_free_var_mutations(factory))
            elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                recv = func.value.id
                method_name = func.attr
                # ``stack.enter_context(cm)`` runs ``cm.__enter__`` / ``__exit__``
                # regardless of what ``stack`` is (an ExitStack), so dispatch to
                # the ARGUMENT's context manager.
                if method_name == "enter_context" and node.args and isinstance(node.args[0], ast.Name):
                    _dispatch_context(node.args[0].id)
                    continue
                own_class = _classdef(recv)
                if own_class is not None:
                    # A class-level call ``Registry.record()`` — no instance.
                    method = _class_method(own_class, method_name, resolve_class_source)
                    if method is not None:
                        _apply_method(own_class, recv, method, None, allow_self=False)
                else:
                    cls = instance_class(recv)
                    cdef = _classdef(cls) if cls else None
                    if cdef is not None:
                        method = _class_method(cdef, method_name, resolve_class_source)
                        if method is not None:
                            _apply_method(cdef, cls, method, recv, allow_self=True)

    # --- top-level ``class Sub(Base):`` triggering a base __init_subclass__ -----
    # A subclass def runs the nearest base's ``__init_subclass__(cls, ...)`` hook
    # during CLASS CREATION (before any node in this cell's body executes), so it
    # is invisible to the executable-node walk above. The hook receives ``cls``
    # (the fresh subclass, discarded on re-derivation) — analyse its body exactly
    # like a constructor (``allow_self=False``) so only its class-var / free-var
    # mutations persist. A class-var accumulator (``Base.registry``) routes to the
    # class-def channel (already under the cross-cell guard); a module/free
    # var (``registry.append``) is collected SEPARATELY so the caller can apply the
    # same guard — the simulator cannot see this hidden cross-cell mutation.
    # Only ``__init_subclass__`` is handled here (metaclass hooks / __set_name__
    # are separate follow-ups). ClassDefs are deferred scopes so the walk above
    # never yields them — scan ``tree.body`` directly.
    init_subclass_free: set[str] = set()
    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and node.bases):
            continue
        for base_name in _class_bases(node):
            base_cdef = _classdef(base_name)
            if base_cdef is None:
                continue
            hook = _class_method(base_cdef, "__init_subclass__", resolve_class_source)
            if hook is not None:
                _, class_targets, fv = _classify_method_mutations(
                    hook,
                    base_name,
                    base_cdef,
                    resolve_class_source,
                )
                class_defs.update(class_targets)
                init_subclass_free.update(fv)
                break

    return ObjectProtocolResets(
        frozenset(free_vars),
        frozenset(receivers),
        frozenset(class_defs),
        frozenset(init_subclass_free),
    )
