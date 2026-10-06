"""Module data a function reaches through a name other than a plain data
global: ``module.ATTR`` chains, attributes stored on a function, imports
written in its body and modules a closure holds, and the docstrings its
code reads."""

from __future__ import annotations

import hashlib
import inspect
import pickle
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..analysis.helper_bindings import resolve_local_import
from ..install_paths import is_user_module
from ..value_types import CODELESS_PRIMS
from .closure_fold import iter_code_scopes
from .key_values import iter_contained, stabilize_for_global_hash
from .user_code import cash_wrapped, is_cash_wrapper, is_user_class, is_user_code_object, own_package, wraps_code

if TYPE_CHECKING:
    from .arg_hashing import ArgHasher
    from .class_data import ClassDataFold
    from .global_reads import GlobalReads
    from .global_values import GlobalValues


def _resolve_dotted(g: dict, path: str) -> Any:
    """The object ``pkg.conf`` names in globals *g*: the global, then each
    attribute through modules only. None when a link is missing."""
    head, _, rest = path.partition(".")
    value = g.get(head)
    for attr in rest.split(".") if rest else ():
        if not isinstance(value, types.ModuleType):
            return None
        value = vars(value).get(attr)
    return value


@dataclass(frozen=True)
class _AttrReader:
    """The function whose ``module.ATTR`` reads are folded, and the drift
    guard's and the dedup's state for that fold (`ModuleAttrFold.module_attr_parts`)."""

    func: Callable
    func_name: str
    own_pkg: str | None
    learned: frozenset | set
    watch: dict | None
    owner_code: Any
    seen: set | None


class ModuleAttrFold:
    """Key parts for module data a function reaches other than through a
    plain data global it names."""

    def __init__(self, args: ArgHasher, reads: GlobalReads, values: GlobalValues, classes: ClassDataFold) -> None:
        self._args = args
        self._reads = reads
        self._values = values
        self._classes = classes

    def docstring_parts(self, code: Any, g: dict, own_pkg: str | None) -> list[tuple[str, str]]:
        """Key parts for the docstrings code that reads docstrings can reach.

        A docstring is not part of the key: it documents the code. Unless the
        code reads it -- a tool description, a prompt, help text built from
        ``__doc__`` -- and then it is an input like any string constant.
        Every user function, class and
        module the code names (and ``module.attr`` of those it reads), and the
        module's own docstring when it reads ``__doc__``.
        """
        parts: list[tuple[str, str]] = []
        names: dict[str, None] = {}
        for scope in iter_code_scopes(code):
            names.update(dict.fromkeys(scope.co_names or ()))
        attr_reads: dict[str, set[str]] = {}
        for mod_name, attr in self._reads.known_module_attr_pairs(code):
            attr_reads.setdefault(mod_name, set()).add(attr)

        def fold(label: str, value: Any) -> None:
            if isinstance(value, types.ModuleType):
                if not is_user_module(value, own_pkg):
                    return
            elif is_cash_wrapper(value):
                pass
            elif not isinstance(value, (types.FunctionType, type)) or not is_user_code_object(value):
                return
            doc = getattr(value, "__doc__", None)
            if isinstance(doc, str):
                parts.append((f"{label}.__doc__", hashlib.sha256(doc.encode("utf-8")).hexdigest()))

        for name in names:
            if name not in g:
                continue
            value = g[name]
            if name == "__doc__":
                if isinstance(value, str):
                    parts.append(("__doc__", hashlib.sha256(value.encode("utf-8")).hexdigest()))
                continue
            fold(name, value)
            if isinstance(value, types.ModuleType) and is_user_module(value, own_pkg):
                for attr in sorted(attr_reads.get(name, ())):
                    fold(f"{name}.{attr}", getattr(value, attr, None))
        return parts

    def local_binding_parts(self, func: Callable) -> list[tuple[str, str]]:
        """Key parts for data reached through names the module's globals never see.

        Two shapes:

        * an import written INSIDE the body -- ``from .settings import
          ROUNDING``, or ``from . import settings`` then ``settings.ROUNDING``
          -- binds a local, which the globals channels never see (the helper
          walk follows only the FUNCTIONS such an import binds);
        * a module held in a closure: ``from . import settings`` inside a
          decorator factory, read by the wrapper as ``settings.ROUNDING``.

        Data values are folded, and a module's ``ATTR`` reads, the same way the
        ``module.ATTR`` channel folds a global module's. A user module the
        body has not imported yet is imported here -- the import the body is
        about to make; a library module only if it is already loaded.
        """
        plan = self._reads.local_binding_plan(func)
        if not plan:
            return []

        imports, attr_reads, bare_reads = plan
        own_pkg = own_package(func)
        root_module = getattr(func, "__module__", None)
        code = func.__code__
        cells = dict(zip(code.co_freevars or (), getattr(func, "__closure__", None) or ()))
        parts: list[tuple[str, str]] = []

        def resolve(name: str) -> Any:
            if name in imports:
                module_name, prefix = imports[name]
                return resolve_local_import(module_name, prefix, root_module)
            cell = cells.get(name)
            if cell is None:
                return None
            try:
                return cell.cell_contents
            except ValueError:
                return None

        def fold(label: str, value: Any) -> None:
            if isinstance(value, (types.ModuleType, type)):
                return
            if callable(value) and not isinstance(value, (dict, list, tuple, set)):
                return  # code: the helper walk follows it
            try:
                stabilized = stabilize_for_global_hash(value, self._values.data_callable_identity)
                parts.append((label, self._args.hash_payload((stabilized,), {})))
            except (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError):
                pass

        for name, attrs in attr_reads.items():
            obj = resolve(name)
            if not isinstance(obj, types.ModuleType) or not is_user_module(obj, own_pkg):
                continue
            for attr in sorted(attrs):
                try:
                    value = getattr(obj, attr)
                except AttributeError:
                    continue
                fold(f"local:{name}.{attr}", value)
        for name in sorted(bare_reads):
            obj = resolve(name)
            if obj is not None and not isinstance(obj, types.ModuleType):
                fold(f"local:{name}", obj)
        return parts

    def module_attr_parts(
        self,
        func: Callable,
        func_name: str,
        g: dict,
        *,
        learned: frozenset | set = frozenset(),
        watch: dict | None = None,
        owner_code: Any = None,
        seen: set | None = None,
    ) -> list[tuple[str, str]]:
        """Key parts for ``module.ATTR`` data reads, one level of recursion deep.

        Two shapes are covered:

        * ``conf.RATE`` - fold the attribute's content.
        * ``conf.get_rate()`` - the callable itself is already tracked by the
          helper-source channel, but that only sees its *source*. A helper whose
          source never changes while the constant it returns does was stale, so
          fold the data globals the callee reads from its own module too.

        Callables, classes and nested modules are skipped as data (the first is
        handled by the helper channel, the others carry no editable value) --
        except what a library-made callable was built with (``conf.SMOOTH =
        partial(gaussian_filter, sigma=...)``), which is folded when *watch*
        is given, so the drift guard can see it too (`GlobalValues.carried_global_hash`).
        *learned* is the drift guard's verdict: labels not to fold. A user
        class read as ``module.Class`` (and an instance's class) is folded by
        `ClassDataFold.class_parts`, *owner_code* and *seen* as there.
        """
        reader = _AttrReader(func, func_name, own_package(func), learned, watch, owner_code, seen)
        parts: list[tuple[str, str]] = []
        for mod_name, attr in self._reads.module_attr_pairs(func):
            parts.extend(self._pair_parts(reader, g, mod_name, attr))
        return parts

    def _pair_parts(self, reader: _AttrReader, g: dict, mod_name: str, attr: str) -> list[tuple[str, str]]:
        """Key parts for one ``mod_name.attr`` read (`GlobalReads.module_attr_pairs`)."""
        func, func_name, own_pkg = reader.func, reader.func_name, reader.own_pkg
        obj = _resolve_dotted(g, mod_name)
        is_mod = isinstance(obj, types.ModuleType) and is_user_module(obj, own_pkg)
        # ``Cfg.LIMIT`` -- a class constant read through the class NAME -- is
        # the same bytecode shape (LOAD_GLOBAL Cfg; LOAD_ATTR LIMIT), with a
        # class in place of the module. Fold user-class attributes too.
        is_cls = isinstance(obj, type) and is_user_class(obj, own_pkg)
        if not (is_mod or is_cls):
            # `scale.k` with `scale.k = 1` set on a function of the
            # user's: an attribute stored on the function object, which
            # its source does not show.
            if isinstance(obj, types.FunctionType):
                return self._function_attr_parts(obj, attr, mod_name, func_name)
            return []
        try:
            value = inspect.getattr_static(obj, attr) if is_cls else getattr(obj, attr)
        except (AttributeError, Exception):  # noqa: BLE001 - never break a call
            return []
        label = f"{mod_name}.{attr}"
        if isinstance(value, type):
            # `cfg.Cfg.RATE`, `cfg.Color.RED.value`: the pair is (cfg, Cfg)
            # and the constant is one attribute further in.
            if is_mod and is_user_class(value, own_pkg):
                return self._classes.class_parts(
                    value, func_name, owner_code=reader.owner_code, seen=reader.seen, reader=func
                )
            return []
        if isinstance(value, types.ModuleType):
            return []
        parts: list[tuple[str, str]] = []
        if is_mod:
            parts.extend(self._held_class_parts(reader, label, value))
        if is_cls and wraps_code(value):
            # Read statically, a classmethod, property or cached_property is
            # its descriptor, which is neither callable nor data: hashing it
            # warned KEY-UNHASHABLE-GLOBAL for `A.make(v)`, whose code is
            # followed like any method's.
            return parts
        if callable(value) and not isinstance(value, (dict, list, tuple, set)):
            # A class method/staticmethod/classmethod is handled by the
            # helper-source / self-dep channels; only recurse into a
            # module-level helper's own constants here.
            if is_mod:
                parts.extend(self._module_callable_parts(reader, obj, attr, label, value))
            return parts
        h = self._values.safe_global_hash(value, func_name, label)
        if h is not None:
            parts.append((label, h))
        return parts

    def _held_class_parts(self, reader: _AttrReader, label: str, value: Any) -> list[tuple[str, str]]:
        """Key parts for the user classes of the instances *value* holds: their
        data (`ClassDataFold.class_parts`) and their code. An instance read as
        `lib.SVC`: its pickle is its own attributes, not what its class holds
        (`helper = CC(10)`), nor the methods that run on it without being
        named -- a property, an operator, ``len()``, ``__getattr__``."""
        parts: list[tuple[str, str]] = []
        item_types = {type(item) for item in iter_contained(value) if type(item) not in CODELESS_PRIMS}
        for item_type in sorted(item_types, key=lambda t: f"{t.__module__}.{t.__qualname__}"):
            if item_type is not type and is_user_class(item_type, reader.own_pkg):
                parts.extend(
                    self._classes.class_parts(
                        item_type, reader.func_name, owner_code=reader.owner_code, seen=reader.seen, reader=reader.func
                    )
                )
                parts.append((f"{label}#cls:{item_type.__qualname__}", self._values.data_callable_identity(item_type)))
        return parts

    def _module_callable_parts(
        self, reader: _AttrReader, module: Any, attr: str, label: str, value: Any
    ) -> list[tuple[str, str]]:
        """Key parts for a callable a user module holds: what a library-made
        callable carries, when the drift guard watches (*reader.watch*),
        or else the data globals the helper itself reads."""
        if reader.watch is not None and label not in reader.learned:
            carried = self._values.carried_global_hash(value, getattr(reader.func, "__module__", None))
            if carried is not None:
                reader.watch[label] = (carried, "carrier", (vars(module), attr), None)
                return [(f"{label}#carried", carried)]
        return self._helper_constant_parts(value, label, reader.func_name)

    def _helper_constant_parts(self, value: Any, label: str, func_name: str) -> list[tuple[str, str]]:
        """Key parts for the data globals a module-level helper reads.

        One level only: the constants the helper itself reads. Deeper
        recursion would drag in whole transitive namespaces for a
        diminishing chance of catching a real edit. A cached helper's
        globals are those of the function it wraps, not of cash's wrapper.
        """
        value = cash_wrapped(value)
        helper_globals = getattr(value, "__globals__", None)
        if not isinstance(helper_globals, dict):
            return []
        parts: list[tuple[str, str]] = []
        for inner in self._reads.read_global_data_names(value):
            if inner not in helper_globals:
                continue
            iv = helper_globals[inner]
            if isinstance(iv, types.ModuleType) or isinstance(iv, type):
                continue
            if callable(iv) and not isinstance(iv, (dict, list, tuple, set)):
                continue
            h = self._values.safe_global_hash(iv, func_name, f"{label}.{inner}")
            if h is not None:
                parts.append((f"{label}.{inner}", h))
        return parts

    def _function_attr_parts(self, fn: Any, attr: str, name: str, func_name: str) -> list[tuple[str, str]]:
        """The key part for data stored as an attribute of the user's function *fn*."""
        stored = getattr(fn, "__dict__", None)
        if not isinstance(stored, dict) or attr not in stored or not is_user_code_object(fn):
            return []
        value = stored[attr]
        if isinstance(value, (types.ModuleType, type)) or (
            callable(value) and not isinstance(value, (dict, list, tuple, set))
        ):
            return []
        h = self._values.safe_global_hash(value, func_name, f"{name}.{attr}")
        return [(f"{name}.{attr}", h)] if h is not None else []
