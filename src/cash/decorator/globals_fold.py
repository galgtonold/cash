"""Folding the globals a function reads, and what they carry, into its key."""

from __future__ import annotations

import contextvars
import enum
import hashlib
import inspect
import pickle
import sys
import types
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .._memo import CODE_OBJECTS, LruMemo
from ..analysis.purity_analyzer import (
    PurityReport,
    get_analyzer,
    resolve_binding,
    resolve_local_import,
)
from ..dependency_state import ledger_note
from ..effects import environment_component
from ..exceptions import CashImpurityWarning
from ..install_paths import is_user_module
from ..value_types import CODELESS_PRIMS
from .arg_hashing import is_opaque
from .call_state import CAPTURE_WATCH, KeyBuildFailed
from .closure_fold import iter_code_scopes
from .global_reads import DOCSTRING_READS, MACHINERY_DUNDERS, bytecode_written_attrs
from .global_values import UNHASHABLE_GLOBAL_FIX
from .key_values import (
    SYNC_TYPES,
    carried_payload,
    is_immutable_capture,
    iter_contained,
    plain_data_kind,
    stabilize_for_global_hash,
)
from .user_code import cash_wrapped, is_cash_wrapper, is_user_class, is_user_code_object, own_package, wraps_code

if TYPE_CHECKING:
    from .arg_hashing import ArgHasher
    from .code_args import CodeArgs
    from .code_surface import CodeSurface
    from .global_reads import GlobalReads
    from .global_values import GlobalValues
    from .purity_checks import LearnedMutations
    from .registry import FunctionRegistry
    from .reporting import Notices

#: The classes `GlobalsFold.class_parts` has folded during one key build, by
#: id. A class's methods can read a global instance of that same class, and
#: the instance leads back to the class: without this the walk never ended.
CLASSES_FOLDED: contextvars.ContextVar[set[int] | None] = contextvars.ContextVar("_cash_classes_folded", default=None)

#: ``(id(function), id(owner code), extra names) -> (function, the *seen*
#: entries its fold skipped)`` for each function whose globals
#: `GlobalsFold.fold_read_globals` folded during one key build. A class's
#: method is reached both by the helper walk and by its class's
#: `GlobalsFold.class_parts`: folded again, every global it reads was
#: already in *seen* and skipped, and only the parts *seen* does not govern
#: were hashed once more, as duplicates -- 200us of a hit on a function that
#: builds two small classes. None outside a key build.
READS_FOLDED: contextvars.ContextVar[dict | None] = contextvars.ContextVar("_cash_reads_folded", default=None)


def _user_bases(cls: type) -> list[type]:
    """*cls*'s own user classes in method-resolution order, and its metaclass's."""
    found = [b for b in cls.__mro__ if b is not object and not is_opaque(b) and is_user_code_object(b)]
    meta = type(cls)
    if meta is not type:
        found += [b for b in meta.__mro__ if b not in (type, object) and not is_opaque(b) and is_user_code_object(b)]
    return found


def _function_layers(fn: Any) -> list[types.FunctionType]:
    """*fn* and the functions it wraps (``__wrapped__``), each a plain function.

    sklearn wraps every ``transform`` a ``TransformerMixin`` subclass defines,
    and cash wraps a cached method: the class attribute is the wrapper, whose
    globals are the library's, and the user's function is inside it.
    """
    layers: list[types.FunctionType] = []
    walked: set[int] = set()
    while fn is not None and id(fn) not in walked:  # every layer; a cycle ends
        walked.add(id(fn))
        if isinstance(fn, types.FunctionType) and not is_cash_wrapper(fn):
            layers.append(fn)
        fn = getattr(fn, "__wrapped__", None)
    return layers


def _member_functions(member: Any) -> list[types.FunctionType]:
    """The functions a class attribute runs: a method, the function inside a
    staticmethod, classmethod, property, cached_property or partialmethod."""
    if isinstance(member, (staticmethod, classmethod)):
        member = member.__func__
    if isinstance(member, property):
        candidates = [member.fget, member.fset, member.fdel]
    elif isinstance(member, types.FunctionType):
        candidates = [member]
    else:
        candidates = [getattr(member, attr, None) for attr in ("__func__", "fget", "func")]
    return [layer for c in candidates for layer in _function_layers(c)]


def class_surface_functions(cls: type) -> list[types.FunctionType]:
    """Every function an instance of *cls* can run: its own methods and every
    user base's, inherited ``__init__`` and ``__init_subclass__`` included,
    property and ``cached_property`` accessors, its metaclass's methods, and
    the methods of user objects it holds as class attributes (a descriptor's
    ``__get__``, a callable instance's ``__call__``) -- and of the user
    objects THEIR classes hold, however deep."""
    found: list[types.FunctionType] = []
    classes = [cls]
    walked: set[type] = {cls}
    while classes:
        current = classes.pop(0)
        for base in _user_bases(current):
            for member in list(vars(base).values()):
                functions = _member_functions(member)
                found.extend(functions)
                if (
                    not functions
                    and not isinstance(member, (type, types.ModuleType))
                    and not wraps_code(member)
                    and type(member) not in walked
                    and is_user_code_object(type(member))
                ):
                    walked.add(type(member))
                    classes.append(type(member))
    seen: set[int] = set()
    return [f for f in found if not (id(f) in seen or seen.add(id(f)))]


def _is_class_machinery(name: str) -> bool:
    """A class-body name the class machinery owns (``__module__``,
    ``__slots__``, ``_abc_impl``...), never data the user reads."""
    return (name.startswith("__") and name.endswith("__")) or name.startswith("_abc_")


def class_data_items(
    cls: type, skip: frozenset[str] = frozenset(), *, bases: list[type] | None = None
) -> list[tuple[str, Any]]:
    """``(label, value)`` for the class-level DATA of *cls* and its user bases.

    A constant (``RATE = 1``), a table, a callable instance held as a class
    attribute, a namedtuple's ``_fields`` and ``_field_defaults``: what its
    methods read through ``self`` or the class. Code (functions and the
    descriptors around them), classes and modules are left to the code
    channels. An Enum is its members' names and values: a member pickles by
    name, so ``RED = 1`` -> ``RED = 5`` changed nothing a pickle sees.
    Names in *skip* -- what the class's own code writes -- are left out.
    *bases* limits the walk to those classes.
    """
    items: list[tuple[str, Any]] = []
    for base in _user_bases(cls) if bases is None else bases:
        prefix = base.__qualname__
        if isinstance(base, enum.EnumMeta):
            members = [(name, member._value_) for name, member in base.__members__.items()]
            items.append((f"{prefix}.__members__", members))
            continue
        for name, value in list(vars(base).items()):
            if _is_class_machinery(name) or name in skip:
                continue
            if isinstance(value, (types.FunctionType, type, types.ModuleType)) or wraps_code(value):
                continue
            if is_cash_wrapper(value):
                continue
            items.append((f"{prefix}.{name}", value))
    return items


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


class GlobalsFold:
    """The module data a function and its helpers read, folded into the state
    segment: globals, ``module.ATTR`` reads, data reached through local
    bindings, what a library-made callable carries, and the environment."""

    def __init__(
        self,
        args: ArgHasher,
        reads: GlobalReads,
        values: GlobalValues,
        code: CodeSurface,
        registry: FunctionRegistry,
        mutations: LearnedMutations,
        notices: Notices,
    ) -> None:
        self._args = args
        self._reads = reads
        self._values = values
        self._code = code
        self._registry = registry
        self._mutations = mutations
        self._notices = notices
        # class -> (its surface functions, the names their code reads); see
        # `class_parts`. A redefined class is a new key.
        self._class_code_cache: LruMemo[type, tuple[tuple, frozenset]] = LruMemo(CODE_OBJECTS)
        # class -> (the immutable data values last hashed, their digest); see
        # `_class_data_digest`.
        self._class_data_memo: LruMemo[type, tuple[tuple, str]] = LruMemo(CODE_OBJECTS)
        # class -> its user bases and their data names; see `_class_layout`.
        self._class_layout_cache: LruMemo[type, tuple] = LruMemo(CODE_OBJECTS)
        #: The argument walk, which a data global's code goes through too;
        #: set by `CodeArgs`, which is built after this.
        self.code_args: CodeArgs | None = None

    def fold_environment(self, func: Callable, func_name: str, state_hash: str) -> str:
        """Fold the current value of every environment read into the key.

        ``os.environ["TENANT"]`` in a cached body served the first tenant's
        answer to every other tenant: the value is an input that never reached
        the key. The analyzer lists the reads whose name is written out
        (`PurityReport.environment_reads`), in the function, its helpers and
        the cached functions it calls -- a dependency's own key moves with
        the variable, but this function's stored result would not. Each is
        read again on every call, and a new value is a new entry.

        Nothing is added when there are none, so such a key is unchanged.
        """
        entries = self._environment_reads(func_name, set(), self._registry.report_for(func, func_name))
        if not entries:
            return state_hash
        component = environment_component(entries, note=lambda label, digest: ledger_note(("env", label), digest))
        return hashlib.sha256(f"{state_hash}{component}".encode("utf-8")).hexdigest()

    def _environment_reads(
        self, func_name: str, visited: set[str], report: PurityReport | None = None
    ) -> set[tuple[str, str]]:
        """The environment reads of *func_name* and every cached function it
        (transitively) depends on (cycle-guarded). *report* is *func_name*'s
        own, when the caller has it (`FunctionRegistry.report_for`)."""
        if func_name in visited:
            return set()
        visited.add(func_name)
        if report is None:
            report = self._registry.purity_reports.get(func_name)
        found = set(getattr(report, "environment_reads", ()) or ())
        for dep in self._registry.graph.get_dependencies(func_name):
            found |= self._environment_reads(dep, visited)
        return found

    def fold_read_globals(
        self,
        func: Callable,
        func_name: str,
        state_hash: str,
        owner_code: Any = None,
        seen: set | None = None,
        extra_names: tuple[str, ...] = (),
    ) -> str:
        """Fold module-level DATA globals the function reads into the key.

        ``owner_code`` is the code object the DRIFT GUARD is recorded under.
        It defaults to *func*'s own, which is right when *func* is the cached
        function. When folding a HELPER's globals it must be the CACHED
        function's code instead: `PurityChecks.learn_mutating_captures` records drift
        against the function whose call was observed, so looking it up under
        the helper's code would never find the entry, fold a drifting
        accumulator anyway, and miss forever.

        A cached function reading a mutable module global (a config constant, a
        dispatch dict of callables) returned stale results when that global
        changed, with no warning. Fold the content of read *data* globals so a
        change invalidates. Modules, plain callables (helpers - tracked via the
        purity analyzer / dependency graph), and classes are excluded; unhashable
        data globals warn once and are skipped.
        """
        if (
            seen is not None
            and not extra_names
            and isinstance(func, types.FunctionType)
            and not self._reads.may_read_data(func)
        ):
            # A helper that reads no data (most methods, every one a dataclass
            # generates): the fold would find nothing.
            return state_hash
        folded = READS_FOLDED.get() if seen is not None else None
        if folded is None:
            return self._fold_read_globals(func, func_name, state_hash, owner_code, seen, extra_names)
        done_key = (id(func), id(owner_code), extra_names)
        done = folded.get(done_key)
        pairs = self._read_pairs(func, extra_names)
        if done is not None and done[0] is func and done[1] <= seen:
            # Folded earlier in this key, for the same cached function. Each
            # global and default that fold skipped as seen is skipped here too;
            # everything else it hashed and put into this key, the same values.
            # Folding again would add each of them once more, nothing new.
            seen.update(pairs)
            return state_hash
        skipped = frozenset(p for p in pairs if p in seen)
        state_hash = self._fold_read_globals(func, func_name, state_hash, owner_code, seen, extra_names)
        folded[done_key] = (func, skipped)
        return state_hash

    def _read_pairs(self, func: Callable, extra_names: tuple[str, ...]) -> list[tuple]:
        """The *seen* entries `GlobalsFold.fold_read_globals` records for
        *func*: each global it reads by name, and each function default."""
        g = getattr(func, "__globals__", None)
        if not isinstance(g, dict):
            return []
        gid = id(g)
        pairs: list[tuple] = [(gid, n) for n in (*self._reads.read_global_data_names(func), *extra_names) if n in g]
        pairs.extend(("default", id(default)) for default in self._reads.function_defaults(func))
        return pairs

    def _fold_read_globals(
        self,
        func: Callable,
        func_name: str,
        state_hash: str,
        owner_code: Any,
        seen: set | None,
        extra_names: tuple[str, ...],
    ) -> str:
        names = self._reads.read_global_data_names(func)
        if extra_names:
            names = tuple(dict.fromkeys(names + extra_names))
        g = getattr(func, "__globals__", None)
        if not isinstance(g, dict):
            return state_hash
        # NOTE: no early return on an empty ``names``. A body whose only global
        # reads are module attributes (``return conf.RATE``) has NO plain data
        # globals, so bailing here skipped the module-attribute channel in
        # exactly the case it exists for.
        parts: list[tuple[str, str]] = []
        own_pkg = own_package(func)
        root_module = getattr(func, "__module__", None)
        code = getattr(func, "__code__", None)
        # A missing provisional entry means "unknown", not "none" -- watch every
        # folded name rather than fold one blind (see `GlobalReads.read_global_data_names`).
        provisional = self._reads.provisional_names(code)
        learned_mutating = self._mutations.of(owner_code if owner_code is not None else code, "global")
        watch: dict[str, str] = {}
        classes: list[type] = []
        for name in names:
            if name not in g:
                continue
            if seen is not None:
                pair = (id(g), name)
                if pair in seen:
                    continue
                seen.add(pair)
            if name in learned_mutating:
                # Observed to drift as a result of calling this function. Folding
                # it would key the entry on the function's own output and miss
                # forever, and the decorator has no perpetual-miss guard to
                # catch it.
                continue
            v = g[name]
            # Skip modules, classes, and plain callables (helpers/deps handled
            # elsewhere). Containers of callables (dispatch dicts) ARE folded.
            if isinstance(v, type):
                # A class's code is keyed as code; what it holds and what its
                # methods read is data (`GlobalsFold.class_parts`).
                if is_user_class(v, own_pkg):
                    classes.append(v)
                continue
            if isinstance(v, types.ModuleType):
                continue
            if callable(v) and not isinstance(v, (dict, list, tuple, set)):
                carried = self._values.carried_global_hash(v, root_module)
                if carried is not None:
                    parts.append((f"{name}#carried", carried))
                    watch[name] = (carried, "carrier", (g, name), None)
                    continue
                # An instance of the user's own callable class is data as well
                # as code: `x * SCALE.k` reads its attributes without calling
                # it, so no helper binding keys them. Folded like any data
                # global; plain callables are the helper walk's.
                payload = carried_payload(v)
                if payload is None or payload[0] != "instance":
                    continue
            plain = plain_data_kind(v)
            try:
                h = self._values.global_value_digest(v, plain)
                parts.append((name, h))
                # Free: this is the hash the key already needed. Keeping it is
                # what makes the post-call check cost one hash instead of two.
                if callable(v) and not isinstance(v, (dict, list, tuple, set)):
                    # Calling it may move what it holds (a memo in `self`):
                    # always watched, and dropped quietly, like a carrier.
                    watch[name] = (h, "instance", g, func)
                elif provisional is None or name in provisional:
                    # `g`, not the decorated function's globals: this may be a
                    # helper's module (see `GlobalsFold.fold_helper_read_globals`).
                    watch[name] = (h, "global", g, func)
            except (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError):
                self._notices.warn_once(
                    CashImpurityWarning,
                    func_name,
                    name,
                    f"@cash.cache on {func_name}: reads module global '{name}' whose "
                    f"value could not be hashed, so changes to it will NOT "
                    f"invalidate the cache.",
                    code="KEY-UNHASHABLE-GLOBAL",
                    fix=UNHASHABLE_GLOBAL_FIX,
                )
                continue
            if plain is not None:
                # Numbers, strings, dates in builtin containers: no class, no
                # instance and no code anywhere inside for the walks below.
                continue
            # A pre-built user-class INSTANCE (or a container of them) is only
            # value-hashed above -- its class's method SOURCE is invisible to the
            # pickle. Fold the class-graph source too (memoized per class; see
            # instance_class_source_parts).
            for item in iter_contained(v):
                if is_user_class(type(item), own_pkg):
                    for cname, chash in self._code.instance_class_source_parts(item, own_pkg=own_pkg):
                        parts.append((f"{name}#cls:{cname}", chash))
                elif isinstance(item, type) and is_user_class(item, own_pkg):
                    # The CLASS itself, not an instance of it: `TABLE = {"fast":
                    # impl.Fast}` pickles by reference, so editing `Fast.run`
                    # moved nothing while the same dict holding a FUNCTION was
                    # followed.
                    surface = self._code.code_surface_hash(item)
                    if surface is not None:
                        parts.append((f"{name}#cls:{item.__qualname__}", surface))
            # Code deeper in: an instance held in a tuple in a list, a
            # function an instance holds (`Runner(scale)`), a user transformer
            # inside a library pipeline. The pickle above has them by name
            # only, and the one-level look above does not reach them; the
            # argument walk does, so a global goes through it too.
            if self.code_args is not None:
                code_parts = self.code_args.carrier_parts(
                    v, func_name, owner_code=owner_code if owner_code is not None else code
                )
                if code_parts:
                    digest = hashlib.sha256(":".join(sorted(set(code_parts))).encode("utf-8")).hexdigest()
                    parts.append((f"{name}#code", digest))
        parts.extend(
            self.module_attr_parts(
                func,
                func_name,
                g,
                learned=learned_mutating,
                watch=watch,
                owner_code=owner_code if owner_code is not None else code,
                seen=seen,
            )
        )
        pending = CAPTURE_WATCH.get()
        if pending is not None:
            pending.update(watch)
        parts.extend(self._local_binding_parts(func))
        if code is not None and self._reads.reads_docstrings(code):
            parts.extend(self._docstring_parts(code, g, own_pkg))
        # A function default is evaluated where the `def` stands, so what a
        # default LAMBDA reads (`def g(x, fn=lambda v: v + K)`) is in no scope
        # of *func*'s, so it is folded here or editing K would keep the key.
        for default in self._reads.function_defaults(func):
            if seen is not None:
                if ("default", id(default)) in seen:
                    continue
                seen.add(("default", id(default)))
            h = self.fold_read_globals(
                default, func_name, "", owner_code=owner_code if owner_code is not None else code, seen=seen
            )
            if h:
                parts.append((f"#default:{default.__qualname__}", h))
        for cls in classes:
            parts.extend(
                self.class_parts(
                    cls, func_name, owner_code=owner_code if owner_code is not None else code, seen=seen, reader=func
                )
            )
        # By name, for a miss that has to say which global moved. A helper's
        # or a called function's reads are labelled with the reader.
        ledger_note(("globals", None if owner_code is None else getattr(func, "__qualname__", None)), parts)
        if not parts:
            return state_hash
        payload = ":".join(f"{n}={h}" for n, h in sorted(parts))
        return hashlib.sha256(f"{state_hash}:globals:{payload}".encode("utf-8")).hexdigest()

    def _docstring_parts(self, code: Any, g: dict, own_pkg: str | None) -> list[tuple[str, str]]:
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

    def fold_helper_read_globals(self, func: Callable, func_name: str, state_hash: str) -> str:
        """Fold globals the transitive HELPERS read, not just *func*'s own.

        A helper reading module state made its caller stale with no warning::

            THRESHOLD = 5
            def helper(): return THRESHOLD

            @cash.cache
            def via_helper(n): return helper()   # THRESHOLD=6 -> HIT -> 5

        The helper's SOURCE was folded, so editing its body invalidated; the
        DATA it read was invisible. Same fold as the one-level case, applied
        to the helpers the purity analyzer already walks, so it inherits the
        written-global exclusion and the drift guard rather than re-deriving
        them.

        Helpers are re-resolved from ``sys.modules`` per call, matching
        ``SysModulesHelperResolver``: a redefined helper reads the redefined
        module's globals.
        """
        report = self._registry.report_for(func, func_name)
        if report is None or not (report.helper_resolution_paths or report.helper_objects):
            return state_hash
        owner_code = getattr(func, "__code__", None)
        # Pre-seed with what the cached function itself already folded, so a
        # global it reads directly is not folded a second time on behalf of a
        # helper that also reads it.
        seen: set = set()
        own_globals = getattr(func, "__globals__", None)
        if isinstance(own_globals, dict):
            for name in self._reads.read_global_data_names(func):
                seen.add((id(own_globals), name))
        return self._fold_paths_read_globals(
            report,
            func,
            func_name,
            state_hash,
            owner_code=owner_code,
            seen=seen,
        )

    def fold_passed_function_reads(self, fn: types.FunctionType, func_name: str, owner_code: Any) -> str:
        """What a function that reaches the call as data reads, as one digest
        ("" when it reads nothing): an argument, or a function a data global
        holds.

        The folds a cached function's own reads go through: its globals
        (`GlobalsFold.fold_read_globals`), then what its helpers read
        (`GlobalsFold.fold_passed_helper_reads`). *owner_code* is the cached
        function's code, which the drift guard records under.

        Raises `KeyBuildFailed` when the helpers cannot be found.
        """
        seen: set = set()
        digest = self.fold_read_globals(fn, func_name, "", owner_code=owner_code, seen=seen)
        return self.fold_passed_helper_reads(fn, func_name, digest, owner_code=owner_code, seen=seen)

    def fold_passed_helper_reads(
        self, fn: types.FunctionType, func_name: str, state_hash: str, *, owner_code: Any, seen: set
    ) -> str:
        """Fold what the helpers of *fn*, a function that reaches the call as
        data, read: their globals and the data their bindings carry (a global
        ``partial(scale, k=2)``), from *fn*'s own purity report, as
        `GlobalsFold.fold_helper_read_globals` does for the cached function.

        Raises `KeyBuildFailed` when the helpers cannot be found.
        """
        try:
            report = get_analyzer().analyze(fn)
        except Exception as e:  # noqa: BLE001 - no report means no key, not a partial one
            report = PurityReport(unwalkable=f"cash could not find the helpers it calls ({type(e).__name__}: {e})")
        if report.unwalkable:
            raise KeyBuildFailed(
                "KEY-HELPERS-UNWALKABLE",
                f"@cash.cache on {func_name}: {getattr(fn, '__qualname__', '?')} reaches the call as data, "
                f"and {report.unwalkable}, so the call ran uncached.",
                "If the function itself runs fine, this is a bug in cash: report it with the error.",
            )
        return self._fold_paths_read_globals(report, fn, func_name, state_hash, owner_code=owner_code, seen=seen)

    def _fold_paths_read_globals(
        self,
        report: PurityReport,
        func: Callable,
        func_name: str,
        state_hash: str,
        *,
        owner_code: Any,
        seen: set,
    ) -> str:
        """Fold the read globals of every helper *report* resolved.

        Shared by the two callers that need it: a cached function's own helpers
        and the helpers of the cached functions it calls.
        """
        for qual in sorted(report.helper_resolution_paths):
            module_name, attr_chain = report.helper_resolution_paths[qual]
            target: Any = sys.modules.get(module_name)
            if target is None:
                continue
            for attr in attr_chain:
                target = getattr(target, attr, None)
                if target is None:
                    break
            if target is None or target is func or not callable(target):
                continue
            if isinstance(target, type):
                # A class the code constructs or calls: what it holds and what
                # its methods -- inherited, properties, `__init__` -- read.
                state_hash = self._fold_class_parts(target, func, func_name, state_hash, owner_code, seen)
                continue
            if getattr(target, "__globals__", None) is None:
                continue
            state_hash = self.fold_read_globals(target, func_name, state_hash, owner_code=owner_code, seen=seen)
        # A callable bound at a call site carries DATA besides its code: a
        # partial's arguments, a bound method's instance, a callable
        # instance's attributes. Its code is followed as a helper; this is the
        # rest (`F = partial(base, k=2)` against `k=3`, `F = S(2).f`).
        carried: list[str] = []
        # A callable that changes what it carries when called -- an instance
        # memoising into its own dict -- would key each call on the last one's
        # output: watched like a library carrier, and dropped once it moves.
        learned = self._mutations.of(owner_code, "global")
        watch: dict[str, tuple] = {}
        for module_name, chain, _ref in report.helper_bindings:
            if (module_name, chain) in report.waived_bindings:
                continue
            label = f"{module_name}.{'.'.join(chain)}"
            if label in learned:
                continue
            live = resolve_binding(module_name, chain)
            if live is func:
                continue
            digest = self._values.carried_state_digest(live)
            if digest is not None:
                carried.append(f"{label}={digest}")
                watch[label] = (digest, "binding", (module_name, chain), None)
        pending = CAPTURE_WATCH.get()
        if pending is not None and watch:
            pending.update(watch)
        if carried:
            state_hash = hashlib.sha256(f"{state_hash}:carried:{':'.join(sorted(carried))}".encode("utf-8")).hexdigest()
        # Helpers with no path of their own -- the function inside a decorator,
        # a closure from a factory -- are held by reference. What THEY read
        # counts as much: `@add1 def h(x): return x * K` computes with K, and
        # the wrapper bound to the name `h` never mentions it.
        for qual in sorted(report.helper_objects):
            if qual in report.helper_resolution_paths:
                continue
            target = report.helper_objects[qual]()
            if isinstance(target, type):
                state_hash = self._fold_class_parts(target, func, func_name, state_hash, owner_code, seen)
                continue
            if target is None or target is func or getattr(target, "__globals__", None) is None:
                continue
            state_hash = self.fold_read_globals(
                target,
                func_name,
                state_hash,
                owner_code=owner_code,
                seen=seen,
                extra_names=self._reads.decorator_global_names(target),
            )
        return state_hash

    def class_parts(
        self,
        cls: type,
        func_name: str,
        *,
        owner_code: Any = None,
        seen: set | None = None,
        reader: Any = None,
    ) -> list[tuple[str, str]]:
        """Key parts for the DATA a user class brings: what it holds, and what
        its code reads.

        A class's code reaches the key through its source and its bases';
        the data behind it is folded here:

        * a module global read by an inherited method, a property, a mixin,
          ``__init__`` or ``cached_property`` (``x * RATE`` in ``Base.scale``,
          called as ``Model().scale(x)``);
        * a class attribute set at run time (``Settings.RATE =
          int(sys.argv[1])``) and read through ``self``;
        * ``cfg.Cfg.RATE`` and ``cfg.Color.RED.value`` through ``import cfg``;
        * a class with no source to read: ``namedtuple(...)`` (its fields and
          defaults), ``make_dataclass``, ``type(...)``;
        * a callable instance held as a class attribute.

        Once per class per key build (`CLASSES_FOLDED`). The class data is one
        hash, re-read on every call and watched: a class attribute the call
        itself moves (a counter, a registry) is dropped from the key after the
        first miss that moves it, member by member. The globals the class's
        functions read go through `GlobalsFold.fold_read_globals`, with the
        same drift guard (*owner_code*, the cached function's code) and the
        same dedup (*seen*); a global any of them writes is state, not input,
        and is left out. *reader* is the function that reached the class, for
        naming a member that cannot be hashed.
        """
        folded = CLASSES_FOLDED.get()
        token = None
        if folded is None:
            folded = set()
            token = CLASSES_FOLDED.set(folded)
        try:
            if id(cls) in folded:
                return []
            folded.add(id(cls))
            return self._class_parts(cls, func_name, owner_code, seen, reader)
        finally:
            if token is not None:
                CLASSES_FOLDED.reset(token)

    def _class_parts(
        self, cls: type, func_name: str, owner_code: Any, seen: set | None, reader: Any
    ) -> list[tuple[str, str]]:
        if owner_code is None:
            owner = self._registry.functions.get(func_name)
            owner_code = getattr(owner, "__code__", None)
        learned = self._mutations.of(owner_code, "global") if owner_code is not None else frozenset()
        label = f"class:{cls.__module__}.{cls.__qualname__}"
        parts: list[tuple[str, str]] = []
        watch: dict[str, tuple] = {}
        if label not in learned:
            digest, unhashable = self.class_data_digest(cls)
            if digest is not None:
                parts.append((label, digest))
                watch[label] = (digest, "classdata", (cls, None), None)
        else:
            # The call moved this class's data: key each member alone, so the
            # one that moves is found and left out, and the rest stay keyed.
            unhashable = []
            for item_label, _ in self._class_data_items(cls):
                member_label = f"{label}:{item_label}"
                if member_label in learned:
                    continue
                digest, bad = self.class_data_digest(cls, item_label)
                unhashable.extend(bad)
                if digest is not None:
                    parts.append((member_label, digest))
                    watch[member_label] = (digest, "classdata", (cls, item_label), None)
        pending = CAPTURE_WATCH.get()
        if pending is not None and watch:
            pending.update(watch)
        functions, excluded, read_names, _ = self._class_code(cls)
        if DOCSTRING_READS & read_names:
            # `self.__doc__` / `inspect.getdoc(type(self))` in its own code.
            for base, _, _ in self._class_layout(cls):
                doc = vars(base).get("__doc__")
                if isinstance(doc, str):
                    parts.append(
                        (f"{label}:{base.__qualname__}.__doc__", hashlib.sha256(doc.encode("utf-8")).hexdigest())
                    )
        if unhashable:
            self._warn_unhashable_class_data(func_name, unhashable, read_names, reader)
        class_seen = set(seen) if seen is not None else set()
        class_seen |= excluded
        for fn in functions:
            if not self._reads.may_read_data(fn):
                continue
            h = self.fold_read_globals(fn, func_name, "", owner_code=owner_code, seen=class_seen)
            if h:
                parts.append((f"{label}#reads:{fn.__qualname__}", h))
        if seen is not None:
            seen |= class_seen - excluded
        return parts

    def _class_code(self, cls: type) -> tuple[tuple, frozenset, frozenset, frozenset]:
        """``(functions, written globals, names read, attributes written)``
        for *cls*, per class.

        The functions are `class_surface_functions`. The written globals are
        ``(id(module globals), name)`` for every global one of them assigns or
        mutates in place -- state the class keeps, excluded for all of them.
        The names are what their code looks up, for naming a class attribute
        that is read but cannot be hashed. The attributes written are those
        its code stores or mutates in place (`bytecode_written_attrs`): a
        class-level counter or registry is state, not an input.
        """
        cached = self._class_code_cache.get(cls)
        if cached is not None:
            return cached
        functions = tuple(class_surface_functions(cls))
        excluded: set[tuple[int, str]] = set()
        names: set[str] = set()
        written: set[str] = set()
        for fn in functions:
            g = getattr(fn, "__globals__", None)
            if not isinstance(g, dict):
                continue
            data = set(self._reads.read_global_data_names(fn))
            for scope in iter_code_scopes(fn.__code__):
                written |= bytecode_written_attrs(scope)
                names.update(scope.co_names or ())
                names.update(c for c in scope.co_consts or () if isinstance(c, str) and c.isidentifier())
                for n in scope.co_names or ():
                    if n in g and n not in data and n not in MACHINERY_DUNDERS:
                        excluded.add((id(g), n))
        result = (functions, frozenset(excluded), frozenset(names), frozenset(written))
        self._class_code_cache[cls] = result
        return result

    def _class_layout(self, cls: type) -> tuple[tuple[type, int, tuple[str, ...] | None], ...]:
        """``(base, size of its namespace, its data names)`` per user base of
        *cls* (`class_data_items`; None for an Enum), per class.

        Finding the bases and telling data from code costs far more than
        reading the values, and changes only when a name is added to or
        removed from a class, which the namespace sizes stand guard for.
        """
        cached = self._class_layout_cache.get(cls)
        if cached is not None and all(len(vars(base)) == size for base, size, _ in cached):
            return cached
        skip = self._class_code(cls)[3]
        layout = []
        for base in _user_bases(cls):
            if isinstance(base, enum.EnumMeta):
                layout.append((base, len(vars(base)), None))
                continue
            names = tuple(label.split(".")[-1] for label, _ in class_data_items(base, skip, bases=[base]))
            layout.append((base, len(vars(base)), names))
        result = tuple(layout)
        self._class_layout_cache[cls] = result
        return result

    def _class_data_items(self, cls: type) -> list[tuple[str, Any]]:
        """`class_data_items` of *cls*, read through its memoized layout."""
        items: list[tuple[str, Any]] = []
        for base, _, names in self._class_layout(cls):
            prefix = base.__qualname__
            if names is None:
                members = [(name, member._value_) for name, member in base.__members__.items()]
                items.append((f"{prefix}.__members__", members))
                continue
            namespace = vars(base)
            for name in names:
                if name in namespace:
                    items.append((f"{prefix}.{name}", namespace[name]))
        return items

    def class_data_digest(self, cls: type, only: str | None = None) -> tuple[str | None, list[str]]:
        """``(digest, unhashable labels)`` of *cls*'s class-level data
        (`class_data_items`), or of the one member *only*.

        A digest of immutable values is reused while every value is the same
        object, so an unchanged class of constants costs a scan, not a hash.
        A member that cannot be hashed is left out and named.
        """
        items = self._class_data_items(cls)
        if only is not None:
            items = [(label, value) for label, value in items if label == only]
        if not items:
            return None, []
        if only is None:
            entry = self._class_data_memo.get(cls)
            if (
                entry is not None
                and len(entry[0]) == len(items)
                and all(a[0] == b[0] and a[1] is b[1] for a, b in zip(entry[0], items))
            ):
                return entry[1], []
        unhashable: list[str] = []
        try:
            stabilized = {
                label: stabilize_for_global_hash(value, self._values.data_callable_identity) for label, value in items
            }
            digest = self._args.hash_payload((stabilized,), {})
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError):
            kept: dict[str, str] = {}
            for label, value in items:
                try:
                    kept[label] = self._args.hash_payload(
                        (stabilize_for_global_hash(value, self._values.data_callable_identity),), {}
                    )
                except (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError):
                    if not isinstance(value, SYNC_TYPES):
                        unhashable.append(label)
            if not kept:
                return None, unhashable
            digest = hashlib.sha256(repr(sorted(kept.items())).encode("utf-8")).hexdigest()
        if only is None and not unhashable and all(is_immutable_capture(value) for _, value in items):
            self._class_data_memo[cls] = (tuple(items), digest)
        return digest, unhashable

    def _warn_unhashable_class_data(
        self, func_name: str, labels: list[str], read_names: frozenset, reader: Any
    ) -> None:
        """Warn once per member that a class attribute the code reads could not be hashed."""
        reader_code = getattr(reader, "__code__", None)
        if reader_code is not None:
            read_names = read_names | {n for scope in iter_code_scopes(reader_code) for n in scope.co_names or ()}
        for label in labels:
            if label.rsplit(".", 1)[-1] not in read_names:
                continue
            self._notices.warn_once(
                CashImpurityWarning,
                func_name,
                label,
                f"@cash.cache on {func_name}: reads the class attribute '{label}' whose "
                f"value could not be hashed, so changes to it will NOT invalidate the cache.",
                code="KEY-UNHASHABLE-GLOBAL",
                fix=UNHASHABLE_GLOBAL_FIX,
            )

    def _fold_class_parts(
        self, cls: type, func: Callable, func_name: str, state_hash: str, owner_code: Any, seen: set
    ) -> str:
        """`GlobalsFold.class_parts` for a class the helper walk reached, into *state_hash*."""
        if not is_user_class(cls, own_package(func)):
            return state_hash
        parts = self.class_parts(cls, func_name, owner_code=owner_code, seen=seen)
        if not parts:
            return state_hash
        payload = ":".join(f"{n}={h}" for n, h in sorted(parts))
        return hashlib.sha256(f"{state_hash}:classes:{payload}".encode("utf-8")).hexdigest()

    def fold_dependency_read_globals(self, func: Callable, func_name: str, state_hash: str) -> str:
        """Fold the globals the CACHED functions this one calls read.

        The third of the three channels a global can reach a key through, and
        the one that was missing. A cached function's own globals are folded by
        ``GlobalsFold.fold_read_globals``; its plain helpers' by
        ``GlobalsFold.fold_helper_read_globals``; a cached CALLEE's were folded into that
        callee's key and nowhere else::

            # rules.py
            THRESHOLD = 20
            def keep(x): return (x % 100) < THRESHOLD

            # calc.py
            @cash.cache
            def inner(n): return sum(i for i in range(n) if keep(i))

            @cash.cache
            def outer(n): return inner(n)

        Edit THRESHOLD and ``inner`` recomputes -- its key moved -- while
        ``outer`` returns the answer computed under the old value, with zero
        executions and no warning. One process disagreeing with itself, which
        is what a user reported after building exactly this shape as a
        library (config module, io module, build module).

        The SOURCE side of the same edge already worked: editing ``inner``'s
        body, or ``keep``'s, invalidates ``outer`` through the graph and the
        transitive helper hashes. Only the DATA those functions read was
        invisible.

        Transitive, because the chain is: ``outer`` -> ``inner`` -> a helper in
        a third module reading a global in a fourth. Each dependency's own
        helpers go through the same fold as if they were this function's.
        """
        owner_code = getattr(func, "__code__", None)
        # Pre-seed with what this function already folded for itself, so a
        # shared global is hashed once rather than once per reader.
        seen: set = set()
        own_globals = getattr(func, "__globals__", None)
        if isinstance(own_globals, dict):
            for name in self._reads.read_global_data_names(func):
                seen.add((id(own_globals), name))
        visited = {func_name}
        stack = sorted(self._registry.graph.get_dependencies(func_name))
        while stack:
            dep = stack.pop()
            if dep in visited:
                continue
            visited.add(dep)
            stack.extend(sorted(self._registry.graph.get_dependencies(dep)))
            dep_func = self._registry.functions.get(dep)
            if dep_func is None or getattr(dep_func, "__globals__", None) is None:
                continue
            state_hash = self.fold_read_globals(
                dep_func,
                func_name,
                state_hash,
                owner_code=owner_code,
                seen=seen,
            )
            dep_report = self._registry.purity_reports.get(dep)
            if dep_report is not None and (dep_report.helper_resolution_paths or dep_report.helper_objects):
                state_hash = self._fold_paths_read_globals(
                    dep_report,
                    func,
                    func_name,
                    state_hash,
                    owner_code=owner_code,
                    seen=seen,
                )
        return state_hash

    def _local_binding_parts(self, func: Callable) -> list[tuple[str, str]]:
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
        `GlobalsFold.class_parts`, *owner_code* and *seen* as there.
        """
        parts: list[tuple[str, str]] = []
        own_pkg = own_package(func)
        for mod_name, attr in self._reads.module_attr_pairs(func):
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
                    parts.extend(self._function_attr_parts(obj, attr, mod_name, func_name))
                continue
            try:
                value = inspect.getattr_static(obj, attr) if is_cls else getattr(obj, attr)
            except (AttributeError, Exception):  # noqa: BLE001 - never break a call
                continue
            label = f"{mod_name}.{attr}"
            if isinstance(value, type):
                # `cfg.Cfg.RATE`, `cfg.Color.RED.value`: the pair is (cfg, Cfg)
                # and the constant is one attribute further in.
                if is_mod and is_user_class(value, own_pkg):
                    parts.extend(self.class_parts(value, func_name, owner_code=owner_code, seen=seen, reader=func))
                continue
            if isinstance(value, types.ModuleType):
                continue
            if is_mod:
                # An instance read as `lib.SVC`: its pickle is its own
                # attributes, not what its class holds (`helper = CC(10)`).
                item_types = {type(item) for item in iter_contained(value) if type(item) not in CODELESS_PRIMS}
                for item_type in sorted(item_types, key=lambda t: f"{t.__module__}.{t.__qualname__}"):
                    if item_type is not type and is_user_class(item_type, own_pkg):
                        parts.extend(
                            self.class_parts(item_type, func_name, owner_code=owner_code, seen=seen, reader=func)
                        )
            if is_cls and wraps_code(value):
                # Read statically, a classmethod, property or cached_property is
                # its descriptor, which is neither callable nor data: hashing it
                # warned KEY-UNHASHABLE-GLOBAL for `A.make(v)`, whose code is
                # followed like any method's.
                continue
            if callable(value) and not isinstance(value, (dict, list, tuple, set)):
                # A class method/staticmethod/classmethod is handled by the
                # helper-source / self-dep channels; only recurse into a
                # module-level helper's own constants here.
                if not is_mod:
                    continue
                if watch is not None and label not in learned:
                    carried = self._values.carried_global_hash(value, getattr(func, "__module__", None))
                    if carried is not None:
                        parts.append((f"{label}#carried", carried))
                        watch[label] = (carried, "carrier", (vars(obj), attr), None)
                        continue
                # One level only: fold the constants the helper itself reads.
                # Deeper recursion would drag in whole transitive namespaces for
                # a diminishing chance of catching a real edit.
                # A cached helper's globals are those of the function it
                # wraps, not of cash's wrapper.
                value = cash_wrapped(value)
                helper_globals = getattr(value, "__globals__", None)
                if not isinstance(helper_globals, dict):
                    continue
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
                continue
            h = self._values.safe_global_hash(value, func_name, label)
            if h is not None:
                parts.append((label, h))
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
