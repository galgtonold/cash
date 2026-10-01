"""The class-level code a cached method reaches through ``self``: the
members it reads (``self.helper()``, ``self.RATE``) and the bases
``super()`` reaches, resolved against the real class at call time."""

from __future__ import annotations

import ast
import hashlib
import inspect
import pickle
import textwrap
import types
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .._memo import CODE_OBJECTS, LruMemo
from ..exceptions import SOURCE_RETRIEVAL_ERRORS
from ..source_norm import getsource
from .function_identity import hash_callable_source

if TYPE_CHECKING:
    from .arg_hashing import ArgHasher


class MethodClassDeps:
    """Folds into a cached method's key the class members it reaches
    through its first parameter, and its base classes when it calls
    ``super()``."""

    def __init__(self, args: ArgHasher) -> None:
        self._args = args
        # (first_param, self_attrs, uses_super) per code object; see
        # `_analyze_method_self_deps`.
        self._method_self_dep_cache: LruMemo[Any, tuple[str | None, tuple[str, ...], bool]] = LruMemo(CODE_OBJECTS)

    def _analyze_method_self_deps(self, func: Callable) -> tuple[str | None, tuple[str, ...], bool]:
        """Attributes a method reads on its first parameter, and whether it calls super().

        A ``@cash.cache`` method reaching class-level code -- ``self.helper()``,
        ``self.RATE``, ``super().m()`` -- had none of that in its key, because at
        decoration time the class does not exist yet and the analyzer sees only
        an attribute access on a parameter. Recorded here (source-derived,
        cached per code object) and resolved against the real class at call time
        by :meth:`MethodClassDeps.fold_method_class_deps`.

        Returns ``(first_param_name, attr_names_accessed_on_it, uses_super)``.
        ``first_param_name`` is ``None`` when there is no source / no parameters.
        """
        code = getattr(func, "__code__", None)
        if code is not None:
            cached = self._method_self_dep_cache.get(code)
            if cached is not None:
                return cached
        result: tuple[str | None, tuple[str, ...], bool] = (None, (), False)
        try:
            src = textwrap.dedent(getsource(func))
            tree = ast.parse(src)
        except SOURCE_RETRIEVAL_ERRORS + (SyntaxError,):
            if code is not None:
                self._method_self_dep_cache[code] = result
            return result
        func_def = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_def = node
                break
        if func_def is None or not func_def.args.args:
            if code is not None:
                self._method_self_dep_cache[code] = result
            return result
        self_name = func_def.args.args[0].arg
        attrs: set[str] = set()
        uses_super = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.Attribute):
                v = node.value
                # ``self.<attr>`` in any position (read or call receiver).
                if isinstance(v, ast.Name) and v.id == self_name:
                    attrs.add(node.attr)
                # ``type(self).<attr>`` -- resolves to the same class member as
                # self.<attr> for class-level attributes; treat it the same.
                elif (
                    isinstance(v, ast.Call)
                    and isinstance(v.func, ast.Name)
                    and v.func.id == "type"
                    and len(v.args) == 1
                    and isinstance(v.args[0], ast.Name)
                    and v.args[0].id == self_name
                ):
                    attrs.add(node.attr)
            # ``super()`` / ``super(...)`` anywhere.
            if isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Name) and f.id == "super":
                    uses_super = True
                elif (
                    isinstance(f, ast.Attribute)
                    and isinstance(f.value, ast.Call)
                    and isinstance(f.value.func, ast.Name)
                    and f.value.func.id == "super"
                ):
                    uses_super = True
        result = (self_name, tuple(sorted(attrs)), uses_super)
        if code is not None:
            self._method_self_dep_cache[code] = result
        return result

    def fold_method_class_deps(self, func: Callable, args: tuple, state_hash: str) -> str:
        """Fold class-level code a cached method reaches into its key.

        At call time the real class IS known (``args[0]`` is the instance, or the
        class for a ``classmethod``), so ``self.helper`` resolves to
        ``type(self).helper`` and its source can be folded; ``self.RATE`` folds
        the class constant's value; ``super()`` folds the user base classes.

        Only CLASS-level members are folded. An instance attribute (in
        ``self.__dict__``) is already covered by hashing ``self`` itself, so it
        is skipped here -- ``getattr(class, attr)`` simply misses it.
        """
        self_name, attrs, uses_super = self._analyze_method_self_deps(func)
        if self_name is None or (not attrs and not uses_super):
            return state_hash
        if not args:
            return state_hash
        owner = args[0]
        # Resolve the class this method was called against, and confirm ``owner``
        # really is its ``self``/``cls`` (guard against a plain function whose
        # first parameter merely happens to be named ``self``). ``owner`` is the
        # class itself for a classmethod, else an instance.
        owner_class = owner if isinstance(owner, type) else type(owner)
        try:
            raw = inspect.getattr_static(owner_class, getattr(func, "__name__", ""))
        except (AttributeError, Exception):  # noqa: BLE001 - never break a call
            return state_hash
        target = raw.__func__ if isinstance(raw, (classmethod, staticmethod)) else raw
        target = getattr(target, "__wrapped__", target)
        if target is not func:
            # Not this class's method (unbound call, or a look-alike param).
            return state_hash

        parts: list[str] = []
        # Transitive, not one-hop: a method reached via self may itself read a
        # class constant or call another method, and editing THAT must also
        # invalidate. Walk the reachable self-members, folding each once. Keyed
        # by attribute name -- within one class hierarchy ``self.X`` always
        # resolves to the same member -- so a ``seen`` set both dedups and stops
        # a mutually-recursive method pair from looping. Every member reached
        # is folded: the names come from the class's source, so they end.
        seen: set[str] = set()
        worklist: list[str] = list(attrs)
        while worklist:
            attr = worklist.pop()
            if attr in seen:
                continue
            seen.add(attr)
            try:
                member = inspect.getattr_static(owner_class, attr)
            except (AttributeError, Exception):  # noqa: BLE001 - never break a call
                continue  # instance-only attr (already in self's hash) or unresolved
            if isinstance(member, property):
                # Fold the getter's source, and follow what the getter reads.
                getter = member.fget
                if getter is not None:
                    try:
                        parts.append(f"p:{attr}:{hash_callable_source(getter)}")
                    except (OSError, TypeError, ValueError):
                        pass
                    _, sub_attrs, _ = self._analyze_method_self_deps(getter)
                    worklist.extend(a for a in sub_attrs if a not in seen)
                continue
            if isinstance(member, (staticmethod, classmethod)):
                member = member.__func__
            if inspect.isfunction(member) or inspect.ismethod(member):
                try:
                    parts.append(f"m:{attr}:{hash_callable_source(member)}")
                except (OSError, TypeError, ValueError):
                    continue
                # Recurse into what this method itself reaches through self.
                _, sub_attrs, _ = self._analyze_method_self_deps(member)
                worklist.extend(a for a in sub_attrs if a not in seen)
            elif not isinstance(member, (types.ModuleType, type)) and not callable(member):
                # A class-level DATA attribute (a constant). Fold its value.
                try:
                    parts.append(f"c:{attr}:{self._args.hash_payload((member,), {})}")
                except (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError):
                    continue
        if uses_super:
            for base in owner_class.__mro__[1:]:
                if base is object:
                    continue
                try:
                    parts.append(f"b:{base.__qualname__}:{hash_callable_source(base)}")
                except (OSError, TypeError, ValueError):
                    continue
        if not parts:
            return state_hash
        payload = ":".join(sorted(parts))
        return hashlib.sha256(f"{state_hash}:selfdeps:{payload}".encode("utf-8")).hexdigest()
