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
import inspect
import sys
import types
from collections.abc import Iterable, Mapping
from typing import Any, NamedTuple

from ..analysis.ast_util import parse_cached
from ..analysis.callee_effects import source_global_mutations
from ..analysis.mutations import MUTATING_METHODS
from ..exceptions import SOURCE_RETRIEVAL_ERRORS
from ..analysis.helper_code import own_code_is_user
from ..tracking.function_tracker import is_local_module

__all__ = ["Reach", "module_state_writes", "reached_user_code"]


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


def reached_user_code(code: str, namespace: Mapping[str, Any] | None) -> Reach:
    """The user functions *code* can call and the local modules it reaches.

    Followed: a name bound to a function, to a class (its ``__init__``), or
    to a local module; ``module.attr`` chains through local modules; the
    globals a notebook function reads when called, and ``module.attr`` in its
    body, transitively; and from each local module reached, the local modules
    its own names come from. A function belongs to the user when it was defined in *namespace*
    (a cell) or in their own code (`own_code_is_user`).
    """
    if not code or not namespace:
        return _EMPTY
    tree = parse_cached(code)
    if tree is None:
        return _EMPTY
    found = _Found(namespace)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            found.value(namespace.get(node.id))
        elif isinstance(node, ast.Attribute):
            found.chain(node)
    found.close_modules()
    data = tuple(sorted(item for item in found.data.items() if item[0] not in found.written))
    return Reach(tuple(found.functions), frozenset(found.modules), data)


def module_state_writes(code: str, namespace: Mapping[str, Any] | None) -> frozenset[str]:
    """The local modules whose state *code* sets, by name.

    ``mylib.K = 7``, ``mylib.CONFIG["k"] = 7``, ``mylib.K += 1``,
    ``del mylib.K``, ``setattr(mylib, "K", 7)``, ``mylib.REGISTRY.update(...)``
    and a call of a module function that changes the module's globals
    (``mylib.set_k(7)``, or ``set_k(7)`` imported from it). A reload runs the
    module's top level again and drops all of these; the notebook's cells
    that made them are what puts them back.
    """
    if not code or not namespace:
        return frozenset()
    tree = parse_cached(code)
    if tree is None:
        return frozenset()
    found: set[str] = set()

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

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
            for target in targets:
                for part in ast.walk(target):
                    if isinstance(part, ast.expr) and isinstance(getattr(part, "ctx", None), (ast.Store, ast.Del)):
                        rooted(part)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in ("setattr", "delattr") and node.args:
                target = namespace.get(node.args[0].id) if isinstance(node.args[0], ast.Name) else None
                if isinstance(target, types.ModuleType) and _is_local(target):
                    found.add(target.__name__)
            elif isinstance(func, ast.Attribute) and func.attr in MUTATING_METHODS:
                rooted(func.value)
            callee = _called(func, namespace)
            if isinstance(callee, types.FunctionType):
                found |= _modules_changed_by(callee, namespace)
    return frozenset(found)


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
        elif label is not None and _is_data(value):
            self.data.setdefault(label, value)

    def _wrapped(self, value: Any) -> None:
        """The function *value* wraps (``functools.wraps``' ``__wrapped__``)."""
        try:
            wrapped = inspect.getattr_static(value, "__wrapped__", None)
        except Exception:  # noqa: BLE001 - an object's attribute lookup
            return
        if isinstance(wrapped, types.FunctionType):
            self.value(wrapped)

    def _class(self, cls: type) -> None:
        """A class: its ``__init__``, and when it is the user's, the data it
        holds (``Settings.scale``), which its methods read through ``self``."""
        home = _loaded(getattr(cls, "__module__", None))
        local = home is not None and _is_local(home)
        if local:
            if id(cls) in self._function_ids:
                return
            self._function_ids.add(id(cls))
            self.modules.add(home.__name__)
            for attr, attr_value in list(vars(cls).items()):
                if not attr.startswith("__") and _is_data(attr_value):
                    self.data.setdefault(f"{home.__name__}.{cls.__qualname__}.{attr}", attr_value)
        self.value(vars(cls).get("__init__"))

    def chain(self, node: ast.Attribute) -> None:
        """``mod.sub.attr``: each local module on the way, and what it ends at."""
        attrs: list[str] = []
        root: ast.expr = node
        while isinstance(root, ast.Attribute):
            attrs.append(root.attr)
            root = root.value
        if not isinstance(root, ast.Name):
            return
        obj = self.namespace.get(root.id)
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


def _is_data(value: Any) -> bool:
    """Whether *value* is data a function reads, rather than code or a
    module, which other channels key."""
    return not (
        value is None
        or isinstance(value, (types.ModuleType, type))
        or inspect.isroutine(value)
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
