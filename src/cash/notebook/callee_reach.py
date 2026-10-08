"""The user's own code a statement reaches when it runs: the functions it
calls, the local modules whose code and data those functions read, and that
data itself.

A statement's inputs name what it mentions. ``m = mylib.mode()`` mentions
``mylib``; what decides its value is ``mode``'s body, the environment that
body reads and the data ``mylib`` holds. Both engines call this with the same
statement text and the live namespace, so they reach the same answer.
"""

from __future__ import annotations

import ast
import dis
import functools
import importlib
import inspect
import operator
import sqlite3
import sys
import textwrap
import types
from collections.abc import Iterable, Mapping
from typing import Any, NamedTuple

from ..analysis.ast_util import parse_cached
from ..analysis.callee_effects import source_global_mutations
from ..analysis.mutations import MUTATING_METHODS
from ..exceptions import SOURCE_RETRIEVAL_ERRORS
from ..analysis.helper_code import own_code_is_user
from ..tracking.function_tracker import is_local_module
from ..value_types import IMMUTABLE_PRIMS, INTERPRETER_MANAGED_GLOBALS, is_runtime_machinery

__all__ = [
    "Reach",
    "module_globals",
    "module_holders",
    "module_state_names",
    "module_state_writes",
    "reached_user_code",
    "rebound_modules",
    "state_holders",
]


class Reach(NamedTuple):
    """What :func:`reached_user_code` found."""

    #: User functions the statement can call, in the order first met.
    functions: tuple[types.FunctionType, ...]
    #: Names of the local modules whose code or data those functions use.
    modules: frozenset[str]
    #: ``(label, value)`` of each piece of a local module's data the code
    #: reads, sorted by label: ``mylib.K`` read by ``mylib.from_k``,
    #: ``mylib.Settings.scale``. What the module's source says is not the
    #: whole story: ``mylib.K = 7`` in a cell, or ``mylib.set_k(7)``, changes
    #: what ``from_k`` returns and leaves the file as it was. A global one of
    #: the reached functions changes itself (``_counter += n``) is left out:
    #: keyed on its value before the call, the statement would key anew on
    #: every run, as a notebook global a called function changes is not.
    data: tuple[tuple[str, Any], ...] = ()


_EMPTY = Reach((), frozenset())


def reached_user_code(code: str, namespace: Mapping[str, Any] | None, *, close: bool = True) -> Reach:
    """The user functions *code* can call and the local modules it reaches.

    Followed: a name bound to a function, to a class (its ``__init__``), or
    to a local module; ``module.attr`` chains through local modules; the
    globals a notebook function reads when called, and ``module.attr`` in its
    body, transitively; and from each local module reached, the local modules
    its own names come from. A function belongs to the user when it was defined in *namespace*
    (a cell) or in their own code (`own_code_is_user`).

    Without *close*, the local modules the reached ones take their names
    from are left out: what calling the reached functions can change, not
    all the data they may lead to (`module_globals`).
    """
    if not code or not namespace:
        return _EMPTY
    steps = names_read(code)
    if not steps or all(_leads_nowhere(namespace, root, attrs) for root, attrs in steps):
        return _EMPTY
    found = _Found(namespace)
    for root, attrs in steps:
        if attrs:
            found.chain(root, attrs)
        else:
            found.value(namespace.get(root))
    if close:
        found.close_modules()
    data = tuple(sorted(item for item in found.data.items() if item[0] not in found.written))
    return Reach(tuple(found.functions), frozenset(found.modules), data)


def _leads_nowhere(namespace: Mapping[str, Any], root: str, attrs: tuple[str, ...]) -> bool:
    """Whether `_Found` would take nothing from the step ``root.attrs``:
    every value it meets is ``None``, a library module, a C routine
    (``print``, ``np.arange``), a C type or an instance of one
    (`_FOREIGN_C_TYPES`). Asked before a walk is set up: the upstream scan
    asks for every cell above on every cell, and most steps are ``x1``,
    ``np.arange`` or ``print``. Decides each value as `_Found.value` and
    `_Found.chain` would, in their order; anything else is walked."""
    obj = namespace.get(root)
    if not attrs:
        return _inert(obj)
    for attr in reversed(attrs):
        if not isinstance(obj, types.ModuleType):
            return True  # `_Found.chain` stops here
        if _is_local(obj):
            return False
        obj = vars(obj).get(attr)
        if not _inert(obj):
            return False
    return True


def _inert(value: Any) -> bool:
    """`_Found.value` of *value*, found with no label, adds nothing."""
    if isinstance(value, types.ModuleType):
        return not _is_local(value)
    if isinstance(value, (types.MethodType, types.FunctionType)):
        return False
    if isinstance(value, type):
        # A C type of no local module: its ``__init__`` is a slot, if any.
        return value in _FOREIGN_C_TYPES
    return value is None or type(value) in _FOREIGN_C_TYPES or inspect.isroutine(value)


@functools.lru_cache(maxsize=4096)
def names_read(code: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """What :func:`reached_user_code` looks up for *code*, in the order its
    walk meets it: ``(name, ())`` for a name read, ``(root, attrs)`` for
    each ``root.a.b`` chain (``attrs`` innermost first); empty when *code*
    does not parse.

    The text alone decides it, so it is found once per text. The upstream
    scan asks for every cell above on every cell: walking each cell's tree
    again was 0.14 s of a cell at the end of a 400-cell notebook.
    """
    tree = parse_cached(code)
    if tree is None:
        return ()
    steps: list[tuple[str, tuple[str, ...]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            steps.append((node.id, ()))
        elif isinstance(node, ast.Attribute):
            attrs: list[str] = []
            root: ast.expr = node
            while isinstance(root, ast.Attribute):
                attrs.append(root.attr)
                root = root.value
            if isinstance(root, ast.Name):
                steps.append((root.id, tuple(attrs)))
    return tuple(steps)


def module_state_writes(code: str, namespace: Mapping[str, Any] | None) -> frozenset[str]:
    """The local modules whose state *code* sets, by name.

    ``mylib.K = 7``, ``mylib.CONFIG["k"] = 7``, ``mylib.K += 1``,
    ``del mylib.K``, ``setattr(mylib, "K", 7)``, ``mylib.REGISTRY.update(...)``
    and a call of a module function that changes the module's globals
    (``mylib.set_k(7)``, or ``set_k(7)`` imported from it), or of a notebook
    function whose body does any of these, and ``importlib.reload(mylib)``,
    which sets every global anew. A reload runs the module's top
    level again and drops all of these; the notebook's cells that made them
    are what puts them back. A ``def`` sets nothing: its body runs when the
    function is called.
    """
    if not code or not namespace:
        return frozenset()
    tree = parse_cached(code)
    if tree is None:
        return frozenset()
    found: set[str] = set()
    _state_writes(tree.body, namespace, found, set())
    return frozenset(found)


def _state_writes(
    statements: Iterable[ast.AST], namespace: Mapping[str, Any], found: set[str], followed: set[int]
) -> None:
    """Add to *found* the local modules *statements* set state on, run
    with *namespace* as their globals; *followed* are the notebook functions
    already walked."""

    def rooted(node: ast.expr) -> None:
        """Add the local module the store target *node* sets something on."""
        if not isinstance(node, (ast.Attribute, ast.Subscript)):
            return
        root = node.value
        while isinstance(root, (ast.Attribute, ast.Subscript)):
            root = root.value
        if isinstance(root, ast.Name):
            module = namespace.get(root.id)
            if isinstance(module, types.ModuleType) and _is_local(module):
                found.add(module.__name__)

    pending = list(statements)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            # What runs at definition: the decorators and the defaults.
            pending.extend(getattr(node, "decorator_list", ()))
            pending.extend(d for d in (*node.args.defaults, *node.args.kw_defaults) if d is not None)
            continue
        pending.extend(ast.iter_child_nodes(node))
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
            for target in targets:
                for part in ast.walk(target):
                    if isinstance(part, ast.expr) and isinstance(getattr(part, "ctx", None), (ast.Store, ast.Del)):
                        rooted(part)
        elif isinstance(node, ast.Call):
            func = node.func
            if node.args and _called(func, namespace) is importlib.reload:
                # A reload runs the module's top level again: every global it
                # has is set anew, whatever the cells above set on it.
                target = _called(node.args[0], namespace)
                if isinstance(target, types.ModuleType) and _is_local(target):
                    found.add(target.__name__)
            if isinstance(func, ast.Name) and func.id in ("setattr", "delattr") and node.args:
                target = namespace.get(node.args[0].id) if isinstance(node.args[0], ast.Name) else None
                if isinstance(target, types.ModuleType) and _is_local(target):
                    found.add(target.__name__)
            elif isinstance(func, ast.Attribute) and func.attr in MUTATING_METHODS:
                rooted(func.value)
            callee = _called(func, namespace)
            if isinstance(callee, types.FunctionType):
                found |= _modules_changed_by(callee, namespace)
                if callee.__globals__ is namespace and id(callee) not in followed:
                    followed.add(id(callee))
                    _state_writes(_function_body(callee), namespace, found, followed)


def _function_body(fn: types.FunctionType) -> list[ast.stmt]:
    """The statements of *fn*'s body, from its source; none when it has none."""
    try:
        tree = parse_cached(textwrap.dedent(inspect.getsource(fn)))
    except SOURCE_RETRIEVAL_ERRORS:
        return []
    if tree is None:
        return []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return list(node.body)
    return []


def module_state_names(code: str, namespace: Mapping[str, Any] | None, *, structure: bool = False) -> frozenset[str]:
    """The names that see the state of a local module *code* sets state on
    (`module_state_writes`, `module_holders`): ``mylib`` for ``mylib.K =
    k``, ``mylib.cfg["a"] = v``, ``mylib.CACHE.append(x)`` or
    ``mylib.set_k(k)``, under every name the namespace holds it by, and
    ``from_k`` taken from it by ``from mylib import from_k``.

    The module is a value of the notebook that ``import mylib`` made and
    these statements change in place, so each of them is one of its
    producers: its lineage moves with them, as a list's moves with
    ``items.append(x)``. A reload of the edited module puts back the
    import's state only, and the upstream check rebuilds the rest from these
    producers, the way it rebuilds any variable.

    A loop or branch is answered for as a whole only with *structure*: it
    changes the module as it changes a list it appends to, through the
    names its body changes (``control_structure_mutations``), so both
    engines ask from there, never as of a plain statement.
    """
    if not code or not namespace:
        return frozenset()
    if not structure:
        from .control_structures.common import is_control_structure  # noqa: PLC0415 - imports this module

        tree = parse_cached(code)
        if tree is not None and len(tree.body) == 1 and is_control_structure(tree.body[0]):
            return frozenset()
    mentioned = {root for root, _ in names_read(code)}
    # Only a module, or a function that may call into one, can lead there.
    if not any(isinstance(namespace.get(root), (types.ModuleType, types.FunctionType, type)) for root in mentioned):
        return frozenset()
    modules = module_state_writes(code, namespace)
    if not modules:
        return frozenset()
    return state_holders(modules, code, namespace)


def state_holders(modules: Iterable[str], code: str, namespace: Mapping[str, Any]) -> frozenset[str]:
    """The names of *namespace* through which a statement *code* that sets
    state on the local *modules* changes what the notebook sees
    (`module_holders`). *code* is kept for the callers' symmetry."""
    del code
    names: set[str] = set()
    for module in modules:
        value = sys.modules.get(module)
        if isinstance(value, types.ModuleType):
            names |= module_holders(value, namespace)
    return frozenset(names)


def module_holders(module: types.ModuleType, namespace: Mapping[str, Any]) -> frozenset[str]:
    """The names of *namespace* that see the state of *module*: the module
    itself under any name, and what ``from mylib import ...`` took from it
    that reads or is that state -- a function or class defined in it
    (``from_k`` reads ``K``) and a mutable object it holds (``CFG``).

    A statement that sets state on the module changes each of them, as
    ``items.append(x)`` changes every name holding ``items``: a cell reading
    ``from_k(2)`` depends on ``set_k(5)`` above it as surely as one reading
    ``mylib.from_k(2)``. ``from mylib import K`` took a copy: setting
    ``mylib.K`` later leaves it as it was, so it is none of them.
    """
    own = vars(module)
    data_ids = {
        id(value)
        for value in list(own.values())
        if not isinstance(value, (types.ModuleType, type, *IMMUTABLE_PRIMS, tuple, frozenset))
        and not inspect.isroutine(value)
    }
    names: set[str] = set()
    for name, held in list(namespace.items()):
        if held is module:
            if name == module.__name__ or not name.startswith("_"):
                names.add(name)
        elif name.startswith("_"):
            continue
        elif isinstance(held, types.FunctionType):
            if held.__globals__ is own:
                names.add(name)
        elif isinstance(held, type):
            if getattr(held, "__module__", None) == module.__name__:
                names.add(name)
        elif id(held) in data_ids:
            names.add(name)
    return frozenset(names)


def module_globals(code: str, namespace: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """What the globals of each local module *code* reaches hold before it
    runs, to tell after it which of them it rebound (`rebound_modules`).

    ``module_state_writes`` reads a write in the text: ``global K`` in a
    function, ``CFG["k"] = v``. One it cannot read -- ``globals()[name] =
    v``, ``setattr(sys.modules[__name__], ...)``, ``global K`` in a method
    of an instance's class -- is seen by running it. Empty for an import or
    a reload, which put the module's own top level in place, and for a
    statement that reaches no local module.
    """
    if not code or not namespace:
        return {}
    tree = parse_cached(code)
    if tree is None or "get_ipython()" in code:
        return {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return {}
        if isinstance(node, ast.Call) and _call_name(node.func) == "reload":
            return {}
    try:
        modules = reached_user_code(code, namespace, close=False).modules
    except Exception:  # noqa: BLE001 - an analysis of arbitrary code
        return {}
    found: dict[str, dict[str, Any]] = {}
    for name in modules:
        module = sys.modules.get(name)
        if isinstance(module, types.ModuleType) and vars(module) is not namespace:
            found[name] = dict(vars(module))
    return found


_ABSENT = object()


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def rebound_modules(before: Mapping[str, Mapping[str, Any]]) -> frozenset[str]:
    """The modules of *before* (`module_globals`) whose globals now hold
    another object than they did, or one more or fewer."""
    changed = set()
    for name, held in before.items():
        module = sys.modules.get(name)
        if not isinstance(module, types.ModuleType):
            continue
        now = vars(module)
        # By identity, at C speed: *held* keeps every object it saw alive,
        # so an id cannot be reused for another one meanwhile.
        if (now.keys() != held.keys() or not all(map(operator.is_, now.values(), held.values()))) and any(
            key not in INTERPRETER_MANAGED_GLOBALS and now.get(key, _ABSENT) is not held.get(key, _ABSENT)
            for key in now.keys() | held.keys()
        ):
            changed.add(name)
    return frozenset(changed)


def _modules_changed_by(fn: types.FunctionType, namespace: Mapping[str, Any]) -> set[str]:
    """The local modules whose globals calling *fn* changes: its own, or
    those of the helpers it calls, at any depth. ``mylib.add(5)`` calling
    ``_bump(5)``, which does ``global COUNT; COUNT += n``, changes ``mylib``
    as surely as a body that does it itself. A notebook function is followed
    to the module functions it calls; its own writes are the notebook's."""
    found: set[str] = set()
    seen: set[int] = set()
    pending = [fn]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        home_ns = current.__globals__
        if home_ns is not namespace:
            home = _loaded(home_ns.get("__name__"))
            if home is None or not _is_local(home):
                continue
            if _changed_globals(current.__code__):
                found.add(home.__name__)
        for co in _code_objects(current.__code__):
            for name in _loaded_globals(co):
                value = home_ns.get(name)
                if isinstance(value, types.FunctionType):
                    pending.append(value)
                elif isinstance(value, types.ModuleType) and _is_local(value):
                    attrs = vars(value)
                    pending.extend(attrs[a] for a in co.co_names if isinstance(attrs.get(a), types.FunctionType))
    return found


def _called(func: ast.expr, namespace: Mapping[str, Any]) -> Any:
    """What the call target *func* (``f`` or ``mod.sub.f``) names, or None."""
    attrs: list[str] = []
    while isinstance(func, ast.Attribute):
        attrs.append(func.attr)
        func = func.value
    if not isinstance(func, ast.Name):
        return None
    obj = namespace.get(func.id)
    for attr in reversed(attrs):
        if not isinstance(obj, types.ModuleType):
            return None
        obj = vars(obj).get(attr)
    return obj


class _Found:
    """What the walk has found so far, deduplicated."""

    def __init__(self, namespace: Mapping[str, Any]) -> None:
        self.namespace = namespace
        self.functions: list[types.FunctionType] = []
        self._function_ids: set[int] = set()
        self.modules: set[str] = set()
        self.data: dict[str, Any] = {}
        #: ``module.name`` of each global a reached module function changes.
        self.written: set[str] = set()

    def value(self, value: Any, label: str | None = None) -> None:
        """Take in *value*, found at *label* (``module.name``) when it is a
        name of a local module."""
        if isinstance(value, types.ModuleType):
            if _is_local(value):
                self.modules.add(value.__name__)
        elif isinstance(value, types.MethodType):
            self.value(value.__func__)
        elif isinstance(value, type):
            self._class(value)
        elif isinstance(value, types.FunctionType) and self._is_user(value):
            if id(value) in self._function_ids:
                return
            self._function_ids.add(id(value))
            self.functions.append(value)
            self._owner(value.__globals__.get("__name__"))
            if value.__globals__ is self.namespace:
                self._body_reads(value.__code__)
            else:
                self._module_body_reads(value)
            # A decorator's wrapper reaches the function it wraps through
            # its closure (``def w(*a): return f(*a)``).
            for cell in value.__closure__ or ():
                try:
                    self.value(cell.cell_contents)
                except ValueError:  # an empty cell
                    pass
            self._wrapped(value)
        elif isinstance(value, types.FunctionType):
            # A wrapper that is not the user's code (``@cash.cache``'s): what
            # calling it runs is the function it wraps.
            self._wrapped(value)
        else:
            if label is not None and _is_data(value):
                self.data.setdefault(label, value)
            cls = type(value)
            if value is not None and cls not in _FOREIGN_C_TYPES and not inspect.isroutine(value):
                # An instance: its methods run when the statement calls them.
                self._class(cls)

    def _wrapped(self, value: Any) -> None:
        """The function *value* wraps (``functools.wraps``' ``__wrapped__``)."""
        try:
            wrapped = inspect.getattr_static(value, "__wrapped__", None)
        except Exception:  # noqa: BLE001 - an object's attribute lookup
            return
        if isinstance(wrapped, types.FunctionType):
            self.value(wrapped)

    def _class(self, cls: type) -> None:
        """A class: its ``__init__``; when it is the user's, its methods and
        those it inherits from the user's classes (``model.predict(2)`` runs
        ``Model.predict``, and what that reads), and when it is a local
        module's, the data it holds (``Settings.scale``), which its methods
        read through ``self``."""
        if not self._users_class(cls):
            self.value(vars(cls).get("__init__"))
            return
        if id(cls) in self._function_ids:
            return
        self._function_ids.add(id(cls))
        home = _loaded(getattr(cls, "__module__", None))
        if home is not None and _is_local(home):
            self.modules.add(home.__name__)
            for attr, attr_value in list(vars(cls).items()):
                if not attr.startswith("__") and _is_data(attr_value):
                    self.data.setdefault(f"{home.__name__}.{cls.__qualname__}.{attr}", attr_value)
        for klass in cls.__mro__:
            if not self._users_class(klass):
                continue
            for attr_value in list(vars(klass).values()):
                for fn in _class_member_functions(attr_value):
                    self.value(fn)

    def _users_class(self, cls: type) -> bool:
        """Defined in a local module, or in a cell: its methods' globals are the namespace."""
        home = _loaded(getattr(cls, "__module__", None))
        if home is not None and _is_local(home):
            return True
        if _is_c_type(cls):
            # No ``class`` statement made it, so no cell's function is its
            # member. Asked of every instance a statement reads (an int, an
            # array): walking ``vars(np.ndarray)`` for each, for every cell
            # above, grew a cell at the end of a 400-cell notebook by 60 ms.
            _FOREIGN_C_TYPES.add(cls)
            return False
        return any(
            fn.__globals__ is self.namespace
            for member in list(vars(cls).values())
            for fn in _class_member_functions(member)
            if isinstance(fn, types.FunctionType)
        )

    def chain(self, root: str, attrs: tuple[str, ...]) -> None:
        """``mod.sub.attr`` (*root* ``mod``, *attrs* innermost first): each
        local module on the way, and what it ends at."""
        obj = self.namespace.get(root)
        for attr in reversed(attrs):
            if not isinstance(obj, types.ModuleType):
                return
            label = f"{obj.__name__}.{attr}" if _is_local(obj) else None
            obj = vars(obj).get(attr)
            self.value(obj, label)

    def _body_reads(self, code: types.CodeType) -> None:
        """What a notebook function reads when it is called: the globals its
        body names, and ``mod.f`` for a local module it names. Its globals
        are resolved when it runs, so a function defined below the caller
        counts as well."""
        stack = [code]
        while stack:
            co = stack.pop()
            stack.extend(c for c in co.co_consts if isinstance(c, types.CodeType))
            for name in co.co_names:
                value = self.namespace.get(name)
                self.value(value)
                if isinstance(value, types.ModuleType) and _is_local(value):
                    self._module_attributes(value, co.co_names)

    def _module_body_reads(self, fn: types.FunctionType) -> None:
        """What a function of a local module reads when it is called: the
        module's globals its code loads, and ``mod.attr`` of a local module it
        names. Its helpers are followed through :meth:`value`."""
        home = fn.__globals__
        home_name = home.get("__name__")
        self.written.update(f"{home_name}.{name}" for name in _changed_globals(fn.__code__))
        for co in _code_objects(fn.__code__):
            for name in _loaded_globals(co):
                if name.startswith("__") or name not in home:
                    continue
                value = home[name]
                self.value(value, f"{home_name}.{name}")
                if isinstance(value, types.ModuleType) and _is_local(value):
                    self._module_attributes(value, co.co_names)

    def _module_attributes(self, module: types.ModuleType, names: Iterable[str]) -> None:
        """``module.attr`` for each of *names* the local *module* has."""
        namespace = vars(module)
        for attr in names:
            if attr in namespace and not attr.startswith("__"):
                self.value(namespace[attr], f"{module.__name__}.{attr}")

    def close_modules(self) -> None:
        """Add the local modules the reached ones take their names from."""
        pending = list(self.modules)
        while pending:
            module = _loaded(pending.pop())
            if module is None:
                continue
            for value in list(vars(module).values()):
                for name in _local_homes(value):
                    if name not in self.modules:
                        self.modules.add(name)
                        pending.append(name)

    def _owner(self, module_name: Any) -> None:
        module = _loaded(module_name)
        if module is not None and _is_local(module):
            self.modules.add(module.__name__)

    def _is_user(self, fn: types.FunctionType) -> bool:
        # No root package: a library function is never the user's because
        # of the package it is in, only because it lives in their files.
        return fn.__globals__ is self.namespace or own_code_is_user(fn, None)


#: C types of no local module: an instance of one leads nowhere (`_Found._class`
#: takes only its ``__init__``, a slot wrapper), so it is not asked again.
_FOREIGN_C_TYPES: set[type] = set()

_HEAPTYPE = 1 << 9
_IMMUTABLETYPE = 1 << 8


def _is_c_type(cls: type) -> bool:
    """Was *cls* made by C code rather than a ``class`` statement? A static
    type, or a heap type an extension created immutable; Python cannot make
    either, nor give one a method."""
    flags = getattr(cls, "__flags__", 0)
    return not flags & _HEAPTYPE or bool(flags & _IMMUTABLETYPE)


def _class_member_functions(member: Any) -> Iterable[Any]:
    """The functions a class attribute runs: a method, a static or class
    method, a property's accessors."""
    if isinstance(member, (staticmethod, classmethod)):
        yield member.__func__
    elif isinstance(member, property):
        yield from (f for f in (member.fget, member.fset, member.fdel) if f is not None)
    elif isinstance(member, types.FunctionType):
        yield member


def _is_data(value: Any) -> bool:
    """Whether *value* is data a function reads, rather than code or a
    module, which other channels key, or a handle no result is computed
    from: a lock, a logger, a stream (`is_runtime_machinery`) or a database
    connection, whose answers are read through it, not held in it. Hashed
    by identity, such a handle gave the statement a new key in every
    process."""
    return not (
        value is None
        or isinstance(value, (types.ModuleType, type, sqlite3.Connection, sqlite3.Cursor))
        or inspect.isroutine(value)
        or is_runtime_machinery(value)
        or getattr(value, "_is_file_tracker_patch", False) is True
    )


def _code_objects(code: types.CodeType) -> Iterable[types.CodeType]:
    """*code* and the code nested in it: comprehensions, inner functions."""
    stack = [code]
    while stack:
        co = stack.pop()
        yield co
        stack.extend(c for c in co.co_consts if isinstance(c, types.CodeType))


@functools.lru_cache(maxsize=4096)
def _loaded_globals(code: types.CodeType) -> frozenset[str]:
    """The global names *code* loads (not its attribute names)."""
    try:
        return frozenset(
            ins.argval
            for ins in dis.get_instructions(code)
            if ins.opname in ("LOAD_GLOBAL", "LOAD_NAME") and isinstance(ins.argval, str)
        )
    except (TypeError, ValueError):
        return frozenset(code.co_names)


@functools.lru_cache(maxsize=4096)
def _changed_globals(code: types.CodeType) -> frozenset[str]:
    """The module globals the function whose code is *code* changes:
    ``global K; K = v``, ``CONFIG["k"] = v``."""
    try:
        return source_global_mutations(inspect.getsource(code))
    except SOURCE_RETRIEVAL_ERRORS:
        return frozenset()


def _loaded(name: Any) -> types.ModuleType | None:
    return sys.modules.get(name) if isinstance(name, str) else None


def _is_local(module: types.ModuleType) -> bool:
    try:
        return is_local_module(module)
    except (TypeError, AttributeError):
        return False


def _local_homes(value: Any) -> Iterable[str]:
    """The local module *value* is, or was defined in."""
    if isinstance(value, types.ModuleType):
        if _is_local(value):
            yield value.__name__
        return
    if isinstance(value, (types.FunctionType, type)):
        home = value.__globals__.get("__name__") if isinstance(value, types.FunctionType) else value.__module__
        module = _loaded(home)
        if module is not None and _is_local(module):
            yield module.__name__
