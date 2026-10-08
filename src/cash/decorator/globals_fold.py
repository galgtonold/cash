"""Folding the globals a function reads, and what they carry, into its key."""

from __future__ import annotations

import contextvars
import hashlib
import pickle
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..analysis.helper_bindings import resolve_binding
from ..analysis.purity_analyzer import get_analyzer
from ..analysis.purity_report import PurityReport
from ..dependency_state import ledger_note
from ..effects import environment_component
from ..exceptions import CashImpurityWarning
from .call_state import CAPTURE_WATCH, KeyBuildFailed
from .global_values import UNHASHABLE_GLOBAL_FIX
from .key_values import (
    carried_payload,
    iter_contained,
    plain_data_kind,
)
from .user_code import is_user_class, own_package

if TYPE_CHECKING:
    from .class_data import ClassDataFold
    from .code_args import CodeArgs
    from .code_surface import CodeSurface
    from .global_reads import GlobalReads
    from .global_values import GlobalValues
    from .module_attrs import ModuleAttrFold
    from .purity_checks import LearnedMutations
    from .registry import FunctionRegistry
    from .reporting import Notices

#: ``(id(function), id(owner code), extra names) -> (function, the *seen*
#: entries its fold skipped)`` for each function whose globals
#: `GlobalsFold.fold_read_globals` folded during one key build. A class's
#: method is reached both by the helper walk and by its class's
#: `ClassDataFold.class_parts`: folded again, every global it reads was
#: already in *seen* and skipped, and only the parts *seen* does not govern
#: were hashed once more, as duplicates -- 200us of a hit on a function that
#: builds two small classes. None outside a key build.
READS_FOLDED: contextvars.ContextVar[dict | None] = contextvars.ContextVar("_cash_reads_folded", default=None)


@dataclass
class _ReadsFold:
    """One function's globals fold in progress (`GlobalsFold._fold_read_globals`):
    whose globals, the drift guard's state for it, and what it found."""

    func: Callable
    func_name: str
    g: dict
    own_pkg: str | None
    #: The code the drift guard records under: the cached function's.
    drift_owner: Any
    provisional: frozenset | None
    learned: frozenset
    seen: set | None
    parts: list[tuple[str, str]] = field(default_factory=list)
    watch: dict[str, tuple] = field(default_factory=dict)
    classes: list[type] = field(default_factory=list)


class GlobalsFold:
    """The module data a function reads, and the functions it reaches read,
    folded into the state segment: the cached function's own globals, its
    helpers', its cached callees', and those of functions passed as data.

    Each global's value is hashed by `GlobalValues`; a class's data by
    `ClassDataFold`; ``module.ATTR`` reads and local bindings by
    `ModuleAttrFold`. This class walks the functions and their names."""

    def __init__(
        self,
        reads: GlobalReads,
        values: GlobalValues,
        classes: ClassDataFold,
        attrs: ModuleAttrFold,
        code: CodeSurface,
        registry: FunctionRegistry,
        mutations: LearnedMutations,
        notices: Notices,
    ) -> None:
        self._reads = reads
        self._values = values
        self._classes = classes
        self._attrs = attrs
        classes.bind_reads_fold(self.fold_read_globals)
        classes.bind_held_code(self.held_value_code_parts)
        attrs.bind_held_code(self.held_value_code_parts)
        self._code = code
        self._registry = registry
        self._mutations = mutations
        self._notices = notices
        #: The argument walk, which a data global's code goes through too;
        #: set by `CodeArgs`, which is built after this.
        self.code_args: CodeArgs | None = None

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
        code = getattr(func, "__code__", None)
        drift_owner = owner_code if owner_code is not None else code
        fold = _ReadsFold(
            func,
            func_name,
            g,
            own_package(func),
            drift_owner,
            # A missing provisional entry means "unknown", not "none" -- watch every
            # folded name rather than fold one blind (see `GlobalReads.read_global_data_names`).
            self._reads.provisional_names(code),
            self._mutations.of(drift_owner, "global"),
            seen,
        )
        for name in names:
            self._fold_global(fold, name)
        parts = fold.parts
        parts.extend(
            self._attrs.module_attr_parts(
                func, func_name, g, learned=fold.learned, watch=fold.watch, owner_code=drift_owner, seen=seen
            )
        )
        pending = CAPTURE_WATCH.get()
        if pending is not None:
            pending.update(fold.watch)
        parts.extend(self._attrs.local_binding_parts(func))
        if code is not None and self._reads.reads_docstrings(code):
            parts.extend(self._attrs.docstring_parts(code, g, fold.own_pkg))
        parts.extend(self._default_parts(fold))
        for cls in fold.classes:
            parts.extend(self._classes.class_parts(cls, func_name, owner_code=drift_owner, seen=seen, reader=func))
        # By name, for a miss that has to say which global moved. A helper's
        # or a called function's reads are labelled with the reader.
        ledger_note(("globals", None if owner_code is None else getattr(func, "__qualname__", None)), parts)
        if not parts:
            return state_hash
        payload = ":".join(f"{n}={h}" for n, h in sorted(parts))
        return hashlib.sha256(f"{state_hash}:globals:{payload}".encode("utf-8")).hexdigest()

    def _fold_global(self, fold: _ReadsFold, name: str) -> None:
        """Add to *fold* what the global *name* holds: its value's digest, the
        data a library-made callable carries, the code an object holds; a
        user class is kept for `ClassDataFold.class_parts`."""
        g = fold.g
        if name not in g:
            return
        if fold.seen is not None:
            pair = (id(g), name)
            if pair in fold.seen:
                return
            fold.seen.add(pair)
        if name in fold.learned:
            # Observed to drift as a result of calling this function. Folding
            # it would key the entry on the function's own output and miss
            # forever, and the decorator has no perpetual-miss guard to
            # catch it.
            return
        v = g[name]
        # Skip modules, classes, and plain callables (helpers/deps handled
        # elsewhere). Containers of callables (dispatch dicts) ARE folded.
        if isinstance(v, type):
            # A class's code is keyed as code; what it holds and what its
            # methods read is data (`ClassDataFold.class_parts`).
            if is_user_class(v, fold.own_pkg):
                fold.classes.append(v)
            return
        if isinstance(v, types.ModuleType):
            return
        if callable(v) and not isinstance(v, (dict, list, tuple, set)):
            carried = self._values.carried_global_hash(v, getattr(fold.func, "__module__", None))
            if carried is not None:
                fold.parts.append((f"{name}#carried", carried))
                fold.watch[name] = (carried, "carrier", (g, name), None)
                return
            # An instance of the user's own callable class is data as well
            # as code: `x * SCALE.k` reads its attributes without calling
            # it, so no helper binding keys them. Folded like any data
            # global; plain callables are the helper walk's.
            payload = carried_payload(v)
            if payload is None or payload[0] != "instance":
                return
        plain = plain_data_kind(v)
        if not self._fold_value(fold, name, v, plain):
            return
        if plain is not None:
            # Numbers, strings, dates in builtin containers: no class, no
            # instance and no code anywhere inside for the walks below.
            return
        fold.parts.extend(self._held_code_parts(fold, name, v))

    def _fold_value(self, fold: _ReadsFold, name: str, v: Any, plain: str | None) -> bool:
        """Add the digest of global *name*'s value *v* to *fold*, watched by
        the drift guard; False, after warning once, when it cannot be hashed."""
        try:
            h = self._values.global_value_digest(v, plain)
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError):
            self._notices.warn_once(
                CashImpurityWarning,
                fold.func_name,
                name,
                f"@cash.cache on {fold.func_name}: reads module global '{name}' whose "
                f"value could not be hashed, so changes to it will NOT "
                f"invalidate the cache.",
                code="KEY-UNHASHABLE-GLOBAL",
                fix=UNHASHABLE_GLOBAL_FIX,
            )
            return False
        fold.parts.append((name, h))
        # Free: this is the hash the key already needed. Keeping it is
        # what makes the post-call check cost one hash instead of two.
        if callable(v) and not isinstance(v, (dict, list, tuple, set)):
            # Calling it may move what it holds (a memo in `self`):
            # always watched, and dropped quietly, like a carrier.
            fold.watch[name] = (h, "instance", fold.g, fold.func)
        elif fold.provisional is None or name in fold.provisional:
            # `g`, not the decorated function's globals: this may be a
            # helper's module (see `GlobalsFold.fold_helper_read_globals`).
            fold.watch[name] = (h, "global", fold.g, fold.func)
        return True

    def _held_code_parts(self, fold: _ReadsFold, name: str, v: Any) -> list[tuple[str, str]]:
        """Key parts for the user code the value *v* of global *name* holds,
        which its pickle names only by reference (`held_value_code_parts`)."""
        return self.held_value_code_parts(name, v, fold.func_name, fold.drift_owner, fold.own_pkg)

    def held_value_code_parts(
        self, label: str, v: Any, func_name: str, owner_code: Any, own_pkg: str | None
    ) -> list[tuple[str, str]]:
        """Key parts for the user code the data value *v* holds -- a global,
        a ``module.ATTR``, a class attribute -- which its pickle names only
        by reference: the classes of the instances in it, the functions and
        classes in it and what that code reads, its environment reads at
        their current values. *owner_code* is the cached function's code,
        for the drift guard."""
        parts: list[tuple[str, str]] = []
        # A pre-built user-class INSTANCE (or a container of them) is only
        # value-hashed -- its class's method SOURCE is invisible to the
        # pickle. Fold the class-graph source too (memoized per class; see
        # instance_class_source_parts).
        for item in iter_contained(v):
            if is_user_class(type(item), own_pkg):
                for cname, chash in self._code.instance_class_source_parts(item, own_pkg=own_pkg):
                    parts.append((f"{label}#cls:{cname}", chash))
            elif isinstance(item, type) and is_user_class(item, own_pkg):
                # The CLASS itself, not an instance of it: `TABLE = {"fast":
                # impl.Fast}` pickles by reference, so editing `Fast.run`
                # moved nothing while the same dict holding a FUNCTION was
                # followed.
                surface = self._code.code_surface_hash(item)
                if surface is not None:
                    parts.append((f"{label}#cls:{item.__qualname__}", surface))
        # Code deeper in: an instance held in a tuple in a list, a
        # function an instance holds (`Runner(scale)`), a user transformer
        # inside a library pipeline. The pickle has them by name
        # only, and the one-level look above does not reach them; the
        # argument walk does, so a global goes through it too.
        if self.code_args is not None:
            env: set = set()
            code_parts = self.code_args.carrier_parts(v, func_name, owner_code=owner_code, env_entries=env)
            if code_parts:
                digest = hashlib.sha256(":".join(sorted(set(code_parts))).encode("utf-8")).hexdigest()
                parts.append((f"{label}#code", digest))
            if env:
                component = environment_component(env, note=lambda what, digest: ledger_note(("env", what), digest))
                parts.append((f"{label}#env", hashlib.sha256(component.encode("utf-8")).hexdigest()))
        return parts

    def _default_parts(self, fold: _ReadsFold) -> list[tuple[str, str]]:
        """Key parts for what *fold*'s function's default functions read.

        A function default is evaluated where the `def` stands, so what a
        default LAMBDA reads (`def g(x, fn=lambda v: v + K)`) is in no scope
        of the function's, so it is folded here or editing K would keep the key.
        """
        parts: list[tuple[str, str]] = []
        seen = fold.seen
        for default in self._reads.function_defaults(fold.func):
            if seen is not None:
                if ("default", id(default)) in seen:
                    continue
                seen.add(("default", id(default)))
            h = self.fold_read_globals(default, fold.func_name, "", owner_code=fold.drift_owner, seen=seen)
            if h:
                parts.append((f"#default:{default.__qualname__}", h))
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
            report = get_analyzer().analyze_reached(fn)
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
        state_hash = self._fold_named_helpers(report, func, func_name, state_hash, owner_code, seen)
        state_hash = self._fold_binding_data(report, func, state_hash, owner_code)
        return self._fold_held_helpers(report, func, func_name, state_hash, owner_code, seen)

    def _fold_named_helpers(
        self, report: PurityReport, func: Callable, func_name: str, state_hash: str, owner_code: Any, seen: set
    ) -> str:
        """Fold what each helper *report* resolves by module path reads,
        re-resolved from ``sys.modules`` now."""
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
                state_hash = self._classes.fold_class_parts(target, func, func_name, state_hash, owner_code, seen)
                continue
            if getattr(target, "__globals__", None) is None:
                continue
            state_hash = self.fold_read_globals(target, func_name, state_hash, owner_code=owner_code, seen=seen)
        return state_hash

    def _fold_binding_data(self, report: PurityReport, func: Callable, state_hash: str, owner_code: Any) -> str:
        """Fold the data the callables bound at *report*'s call sites carry.

        A callable bound at a call site carries DATA besides its code: a
        partial's arguments, a bound method's instance, a callable
        instance's attributes. Its code is followed as a helper; this is the
        rest (`F = partial(base, k=2)` against `k=3`, `F = S(2).f`).
        """
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
        return state_hash

    def _fold_held_helpers(
        self, report: PurityReport, func: Callable, func_name: str, state_hash: str, owner_code: Any, seen: set
    ) -> str:
        """Fold what the helpers *report* holds by reference read.

        Helpers with no path of their own -- the function inside a decorator,
        a closure from a factory -- are held by reference. What THEY read
        counts as much: `@add1 def h(x): return x * K` computes with K, and
        the wrapper bound to the name `h` never mentions it.
        """
        for qual in sorted(report.helper_objects):
            if qual in report.helper_resolution_paths:
                continue
            target = report.helper_objects[qual]()
            if isinstance(target, type):
                state_hash = self._classes.fold_class_parts(target, func, func_name, state_hash, owner_code, seen)
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
