"""Folding the globals a function reads, and what they carry, into its key."""

from __future__ import annotations

import ast
import dis
import functools
import hashlib
import inspect
import pickle
import sys
import textwrap
import types
from collections.abc import Callable
from typing import Any

from ..dependency_state import ledger_note
from ..effects import environment_component
from ..exceptions import SOURCE_RETRIEVAL_ERRORS, CashImpurityWarning
from ..purity_analyzer import (
    REPORTED_METHODS,
    PurityReport,
    callable_layers,
    get_analyzer,
    is_mock,
    local_import_map,
    own_code_is_user,
    resolve_binding,
    resolve_local_import,
)
from ..source_norm import own_source
from .call_state import CAPTURE_WATCH
from .closure_fold import iter_code_scopes, unsafe_uses_of, waived_use_filter
from .code_identity import (
    func_key,
    hash_callable_source,
    is_user_class,
    is_user_module,
    iter_contained,
    own_package,
    wraps_code,
)

# Two fix lines are shared by more than one emit site, because more than one
# site tells the same story: a global whose value cannot be hashed is one
# problem reached through two channels (a function's own globals and a
# helper's), and a refused write is one problem whether the value went whole or
# as a chunked manifest. Sharing the text is what keeps the two halves of each
# pair from drifting into two different pieces of advice for one doc section.
UNHASHABLE_GLOBAL_FIX = (
    "register a hasher for its type with cash.register_hasher, or read the "
    "part the result actually depends on -- a URL, a connection string -- "
    "instead of the live object."
)


def reduced_state(value: Any) -> Any:
    """What ``__reduce_ex__`` says *value* was built with, or None.

    For a C callable with no ``__dict__`` -- ``operator.itemgetter("n")``
    reduces to ``(itemgetter, ("n",))`` -- that is the only place its data
    lives. None when the reduction is just a global name (``np.add``, ``len``:
    nothing carried) or the object refuses to be reduced.
    """
    try:
        reduced = value.__reduce_ex__(4)
    except Exception:  # noqa: BLE001 - not reducible: nothing to fold
        return None
    if isinstance(reduced, str) or not isinstance(reduced, tuple) or len(reduced) < 2:
        return None
    return reduced[:3]


def held_partials(value: Any) -> list[tuple[tuple, dict]]:
    """The arguments of the ``functools.partial`` objects a wrapper instance
    holds as attributes (``np.vectorize.pyfunc``), for a wrapper that is not
    itself a partial."""
    if isinstance(value, functools.partial):
        return []
    state = getattr(value, "__dict__", None)
    if not isinstance(state, dict):
        return []
    return [(p.args, dict(p.keywords)) for p in state.values() if isinstance(p, functools.partial)]


LOG_METHOD_NAMES = frozenset(
    {
        "debug",
        "info",
        "warning",
        "warn",
        "error",
        "exception",
        "critical",
        "log",
    }
)


#: Dunder globals that are machine or import machinery, never user data.
#:
#: These are skipped: ``__file__`` and ``__name__`` differ per checkout and
#: per invocation, so folding them would make a cache key un-shareable
#: between two machines and between ``python job.py`` and ``python -m
#: job``. Every other dunder is folded like any global, because a library
#: declares its data that way too: a bumped ``__version__`` must invalidate
#: what a report stamped with it.
MACHINERY_DUNDERS = frozenset(
    {
        "__name__",
        "__file__",
        "__doc__",
        "__package__",
        "__loader__",
        "__spec__",
        "__builtins__",
        "__path__",
        "__cached__",
        "__debug__",
        "__annotations__",
        "__dict__",
        "__module__",
        "__qualname__",
    }
)


def stabilize_for_global_hash(v: Any, hash_callable, _depth: int = 0) -> Any:
    """Rewrite *v* so callables (incl. lambdas held in containers) are
    replaced by their source hash, making a container of callables hashable
    and content-sensitive (dict-dispatch channel)."""
    if _depth > 8:
        return v
    if callable(v) and not isinstance(v, type):
        try:
            return ("__cash_callable__", hash_callable(v))
        except (OSError, TypeError, ValueError):
            return ("__cash_callable__", getattr(v, "__qualname__", repr(v)))
    if isinstance(v, dict):
        return {k: stabilize_for_global_hash(val, hash_callable, _depth + 1) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(stabilize_for_global_hash(x, hash_callable, _depth + 1) for x in v)
    return v


class GlobalsFoldMixin:
    """The globals a function and its helpers read, folded into the state segment."""

    def _fold_environment(self, func_name: str, state_hash: str) -> str:
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
        entries = self._environment_reads(func_name, set())
        if not entries:
            return state_hash
        component = environment_component(entries, note=lambda label, digest: ledger_note(("env", label), digest))
        return hashlib.sha256(f"{state_hash}{component}".encode("utf-8")).hexdigest()

    def _environment_reads(self, func_name: str, visited: set[str]) -> set[tuple[str, str]]:
        """The environment reads of *func_name* and every cached function it
        (transitively) depends on (cycle-guarded)."""
        if func_name in visited:
            return set()
        visited.add(func_name)
        report = self._purity_reports.get(func_name)
        found = set(getattr(report, "environment_reads", ()) or ())
        for dep in self.graph.get_dependencies(func_name):
            found |= self._environment_reads(dep, visited)
        return found

    def _read_global_data_names(self, func: Callable) -> tuple[str, ...]:
        """Global names *func* references that are candidates for data-folding.

        ``co_names`` intersected with the function's globals, minus the import
        machinery dunders (``_MACHINERY_DUNDERS``) and minus any global the
        function WRITES (``STORE_GLOBAL`` /
        ``DELETE_GLOBAL``). A written global is a side-effect accumulator (a
        ``global counter; counter += 1``) whose value drifts every call - folding
        it would make every call miss (the lesson, applied to globals).
        Modules / callables / classes are filtered per-call at fold time (a
        name's bound value can change). Cached per code object.

        Both bytecode-derived channels walk the NESTED scopes too:
        a global read only inside a genexp/lambda otherwise never invalidated
        (silent stale results), and — the reason the two must move together —
        the ``STORE_GLOBAL`` of a walrus accumulator inside a genexp lives in
        the genexp's own code object, so collecting nested reads without
        collecting nested writes would fold a drifting counter and miss
        forever. The in-place-mutation exclusion below needs no such change:
        it is AST-based, and ``ast.walk`` over the function's source already
        descends into comprehension and lambda bodies.
        """
        code = getattr(func, "__code__", None)
        if code is None:
            return ()
        cached = self._global_read_cache.get(code)
        if cached is not None:
            return cached
        g = getattr(func, "__globals__", {}) or {}

        scopes = tuple(iter_code_scopes(code))
        written = {
            instr.argval
            for scope in scopes
            for instr in dis.get_instructions(scope)
            if instr.opname in ("STORE_GLOBAL", "DELETE_GLOBAL")
        }
        candidates = {
            n
            for scope in scopes
            for n in (scope.co_names or ())
            if n in g and n not in MACHINERY_DUNDERS and n not in written
        }
        # A name spelled as a string reads the same global: `globals()["K"]`
        # is a LOAD_CONST, so `co_names` never had it and editing K served the
        # old answer -- 20 where an uncached run gives 500. The code channel already resolves string
        # constants this way (`_referenced_user_code`); this is its data twin.
        # A string that merely happens to match a global costs a fold, never a
        # stale value.
        candidates |= {
            c
            for scope in scopes
            for c in (scope.co_consts or ())
            if isinstance(c, str) and c.isidentifier() and c in g and c not in MACHINERY_DUNDERS and c not in written
        }
        # Also exclude globals the body mutates IN PLACE (``g['k'] += 1``,
        # ``g.append(...)``) - a STORE_GLOBAL-free accumulator that would
        # otherwise drift every call and cause a permanent miss.
        #
        # `hard` is that set: mutations visible in this function's own source.
        # `provisional` is the weaker case - a name merely PASSED to a call. Those are folded (so a change
        # invalidates) and confirmed at runtime by
        # `_learn_mutating_captures`, which demotes any that the call is actually
        # observed to mutate.
        provisional: frozenset = frozenset()
        if candidates:
            try:
                tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
                hard = unsafe_uses_of(
                    tree,
                    candidates,
                    bare_args=False,
                    mutating_methods_only=True,
                )
                suspected = unsafe_uses_of(tree, candidates) - hard
                provisional = unsafe_uses_of(tree, suspected, waived=waived_use_filter(func))
                # Suspected only on waived lines (`LEDGER.record(r)  #
                # @cash:assume-safe`): the audited effect moves it on every
                # call, so keying on it made every hit impossible and demoting
                # it warned about the very line that was audited.
                hard |= suspected - provisional
                candidates -= hard
            except SOURCE_RETRIEVAL_ERRORS:
                # No source: the conservative answer. Without an AST
                # there is no way to tell a read from a mutation, and folding
                # blind would risk the permanent-miss trap with nothing to
                # learn from.
                candidates, provisional = set(), frozenset()
        names = tuple(sorted(candidates))
        if len(self._global_read_cache) < 4096:
            self._global_read_cache[code] = names
            # Kept in lockstep with the names cache so the two can never
            # disagree about a code object. A MISSING entry is not "nothing is
            # provisional" -- `_fold_read_globals` reads that as "watch every
            # folded name", which costs an extra hash per miss and is the safe
            # direction.
            self._provisional_global_cache[code] = provisional
        return names

    def _data_callable_identity(self, fn: Any) -> str:
        """A callable found INSIDE a data global, identified by what calling it runs.

        A registry -- ``STEPS = {"load": load_step}`` read by a cached
        ``run(name)`` that calls ``STEPS[name](x)`` -- was keyed by each
        function's own source, so an edit to a helper the step calls was a HIT
        with the old result; and a cached function stored there was keyed by
        cash's own wrapper, so not even an edit to its body moved the key. A
        cached function counts as its dependency state, the same
        as a call to it would; a plain function of the user's as its source
        plus its helpers, re-resolved live like any helper's.
        """
        return self._data_callable_identity_of(fn)

    def _data_callable_identity_of(self, fn: Any) -> str:
        if getattr(fn, "_cash_cached", False):
            inner = getattr(fn, "__wrapped__", None)
            if inner is not None:
                name = func_key(inner)
                if name in self.functions:
                    if name not in self._populated:
                        self._ensure_closure_analyzed(inner)
                    return "cached:" + self._state_hasher.compute(name, own_source_override=self._pin_own_source(inner))
                fn = inner
        if not isinstance(fn, types.FunctionType):
            return hash_callable_source(fn)
        own = self._hash_helper_identity(fn)

        if not own_code_is_user(fn, getattr(fn, "__module__", None)):
            return own
        try:
            report = get_analyzer().analyze(fn)
        except (OSError, TypeError, SyntaxError, RecursionError):
            return own
        if not report.helper_source_hashes:
            return own
        live = self._data_helper_resolver.current_hashes(report)
        helpers = ",".join(f"{q}={live.get(q, h)}" for q, h in sorted(report.helper_source_hashes.items()))
        return hashlib.sha256(f"{own}|{helpers}".encode("utf-8")).hexdigest()

    def _fold_read_globals(
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
        function's code instead: `_learn_mutating_captures` records drift
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
        names = self._read_global_data_names(func)
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
        # folded name rather than fold one blind (see `_read_global_data_names`).
        provisional = self._provisional_global_cache.get(code)
        learned_mutating = self._mutating_globals.get(
            (owner_code if owner_code is not None else code, "global"), frozenset()
        )
        watch: dict[str, str] = {}
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
            if isinstance(v, types.ModuleType) or isinstance(v, type):
                continue
            if callable(v) and not isinstance(v, (dict, list, tuple, set)):
                carried = self._carried_global_hash(v, root_module)
                if carried is not None:
                    parts.append((f"{name}#carried", carried))
                    watch[name] = (carried, "carrier", (g, name), None)
                continue
            try:
                stabilized = stabilize_for_global_hash(v, self._data_callable_identity)
                h = self._hash_arg_payload((stabilized,), {})
                parts.append((name, h))
                # Free: this is the hash the key already needed. Keeping it is
                # what makes the post-call check cost one hash instead of two.
                if provisional is None or name in provisional:
                    # `g`, not the decorated function's globals: this may be a
                    # helper's module (see `_fold_helper_read_globals`).
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
            # A pre-built user-class INSTANCE (or a container of them) is only
            # value-hashed above -- its class's method SOURCE is invisible to the
            # pickle, so editing a method served stale. Fold the class-graph
            # source too (memoized per class; see _instance_class_source_parts).
            for item in iter_contained(v):
                if is_user_class(type(item), own_pkg):
                    for cname, chash in self._instance_class_source_parts(item, own_pkg=own_pkg):
                        parts.append((f"{name}#cls:{cname}", chash))
                elif isinstance(item, type) and is_user_class(item, own_pkg):
                    # The CLASS itself, not an instance of it: `TABLE = {"fast":
                    # impl.Fast}` pickles by reference, so editing `Fast.run`
                    # moved nothing while the same dict holding a FUNCTION was
                    # followed.
                    surface = self._code_surface_hash(item)
                    if surface is not None:
                        parts.append((f"{name}#cls:{item.__qualname__}", surface))
        parts.extend(self._module_attr_parts(func, func_name, g, learned=learned_mutating, watch=watch))
        pending = CAPTURE_WATCH.get()
        if pending is not None:
            pending.update(watch)
        parts.extend(self._local_binding_parts(func))
        # By name, for a miss that has to say which global moved. A helper's
        # or a called function's reads are labelled with the reader.
        ledger_note(("globals", None if owner_code is None else getattr(func, "__qualname__", None)), parts)
        if not parts:
            return state_hash
        payload = ":".join(f"{n}={h}" for n, h in sorted(parts))
        return hashlib.sha256(f"{state_hash}:globals:{payload}".encode("utf-8")).hexdigest()

    def _fold_helper_read_globals(self, func: Callable, func_name: str, state_hash: str) -> str:
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
        report = self._purity_reports.get(func_name)
        if report is None or not (report.helper_resolution_paths or report.helper_objects):
            return state_hash
        owner_code = getattr(func, "__code__", None)
        # Pre-seed with what the cached function itself already folded, so a
        # global it reads directly is not folded a second time on behalf of a
        # helper that also reads it.
        seen: set = set()
        own_globals = getattr(func, "__globals__", None)
        if isinstance(own_globals, dict):
            for name in self._read_global_data_names(func):
                seen.add((id(own_globals), name))
        return self._fold_paths_read_globals(
            report,
            func,
            func_name,
            state_hash,
            owner_code=owner_code,
            seen=seen,
        )

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
            if getattr(target, "__globals__", None) is None:
                continue
            state_hash = self._fold_read_globals(target, func_name, state_hash, owner_code=owner_code, seen=seen)
        # A callable bound at a call site carries DATA besides its code: a
        # partial's arguments, a bound method's instance, a callable
        # instance's attributes. Its code is followed as a helper; this is the
        # rest (`F = partial(base, k=2)` -> `k=3`, and `F = S(2).f`, were both
        # served stale).
        carried: list[str] = []
        for module_name, chain, _ref in report.helper_bindings:
            if (module_name, chain) in report.waived_bindings:
                continue
            live = resolve_binding(module_name, chain)
            if live is func:
                continue
            digest = self._carried_state_digest(live)
            if digest is not None:
                carried.append(f"{module_name}.{'.'.join(chain)}={digest}")
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
            if target is None or target is func or getattr(target, "__globals__", None) is None:
                continue
            state_hash = self._fold_read_globals(
                target,
                func_name,
                state_hash,
                owner_code=owner_code,
                seen=seen,
                extra_names=self._decorator_global_names(target),
            )
        return state_hash

    def _carried_state_digest(self, value: Any) -> str | None:
        """Digest of the data a callable carries besides its code, or None.

        A partial's arguments; a bound method's instance (any class -- the
        same as that instance read as a global); a callable instance's own
        state, for USER classes only: a library's callable instance
        (`np.vectorize`) keeps lazy caches that change after its first call,
        which would make every call miss. Silent on failure: the code is
        still keyed, and a warning here would fire on every class-based
        decorator whose state is just the function it wraps.
        """
        if isinstance(value, functools.partial):
            payload: Any = (value.args, dict(value.keywords))
        elif isinstance(value, types.MethodType):
            owner = value.__self__
            if isinstance(owner, (type, types.ModuleType)):
                return None
            payload = owner
        elif (
            callable(value)
            and not isinstance(value, (types.FunctionType, types.BuiltinFunctionType, type, types.ModuleType))
            and not is_mock(value)
            and is_user_class(type(value), own_package(type(value)))
        ):
            payload = value
        elif callable(value) and not is_mock(value) and held_partials(value):
            # A LIBRARY wrapper around the user's code keeps its own caches,
            # but the partials it holds are data the user built it with:
            # `np.vectorize(partial(scale, k=K))` ran with the old K.
            payload = ("wrapped partials", held_partials(value))
        else:
            return None
        try:
            stabilized = stabilize_for_global_hash(payload, self._data_callable_identity)
            return self._hash_arg_payload((stabilized,), {})
        except Exception:  # noqa: BLE001 - never break a call over this
            return None

    def _carried_global_hash(self, value: Any, root_module: str | None) -> str | None:
        """Hash of the data a LIBRARY-made callable carries, or None.

        ``SMOOTH = partial(ndimage.gaussian_filter, sigma=SIGMA)``, ``POLY =
        np.poly1d(COEFFS)``, ``CAL = interp1d(X, Y)``, ``LOOKUP = RATES.get``:
        the code is a library's, so the helper walk does not follow it, and
        what it was built with reached no channel -- editing SIGMA served the
        old result. The same partial passed as an argument was
        keyed all along.

        None for what another channel keys or what carries no data: a
        function, a class, a module, a mock, a cached function, a method of a
        class or module, a C object without a ``__dict__`` (``np.add``), and
        any callable that runs USER code, whose binding the helper walk notes
        and `_carried_state_digest` keys.

        Some of these change when called -- a bound ``rng.normal`` advances
        its generator, ``np.vectorize`` fills a cache -- so every one is
        watched by `_learn_mutating_captures`, which stops folding it after
        the first call that moved it (one extra miss, no warning: the user
        did not write the mutation).
        """
        if isinstance(value, (type, types.ModuleType, types.FunctionType)) or is_mock(value):
            return None
        # Whether it runs user code, and whether it could be hashed at all,
        # are decided once per object: the user-code test resolves file paths,
        # which cost more than the hash (a logger's `.info` hit went 45 -> 250
        # microseconds without this).
        verdict = self._carrier_verdicts.get(id(value))
        if verdict is not None and verdict[0] is value and not verdict[1]:
            return None
        try:
            if getattr(value, "_cash_cached", False):
                return None
            if isinstance(value, functools.partial):
                payload: Any = ("partial", value.func, value.args, dict(value.keywords))
            elif isinstance(value, (types.MethodType, types.BuiltinMethodType)):
                owner = getattr(value, "__self__", None)
                if owner is None or isinstance(owner, (type, types.ModuleType)):
                    return None
                method = getattr(value, "__name__", "")

                if method in REPORTED_METHODS or method in LOG_METHOD_NAMES:
                    # `record = RESULTS.append`, `log = logger.info`: what the
                    # owner holds is the call's OUTPUT, not an input.
                    return None
                payload = ("method", method, owner)
            else:
                state = getattr(value, "__dict__", None)
                cls = type(value)
                if isinstance(state, dict) and state:
                    payload = ("instance", cls.__module__, cls.__qualname__, state)
                else:
                    # A C callable keeps what it was built with where only
                    # `__reduce__` reaches it: `operator.itemgetter("n")`,
                    # `attrgetter`, `methodcaller` -- changing the sort key
                    # served the mis-sorted report. A reduce that
                    # is just a global name (`np.add`, `len`) carries no data.
                    reduced = reduced_state(value)
                    if reduced is None:
                        return None
                    payload = ("reduce", cls.__module__, cls.__qualname__, reduced)
            if verdict is None:
                runs_user_code = any(own_code_is_user(layer, root_module) for layer in callable_layers(value))
                if runs_user_code:
                    # Its code is the helper walk's. What a LIBRARY wrapper
                    # around that code holds besides is still data the user
                    # built it with: `np.vectorize(partial(scale, k=K))` ran
                    # with the old K. Only the partials: the
                    # wrapper's own caches move when it is called.
                    held = held_partials(value)
                    self._note_carrier_verdict(value, "partials" if held else False)
                    if not held:
                        return None
                    payload = ("wrapped partials", held)
                else:
                    self._note_carrier_verdict(value, True)
            elif verdict[1] == "partials":
                payload = ("wrapped partials", held_partials(value))
            stabilized = stabilize_for_global_hash(payload, self._data_callable_identity)
            return self._hash_arg_payload((stabilized,), {})
        except Exception:  # noqa: BLE001 - unkeyable before, never break a call over it
            self._note_carrier_verdict(value, False)
            return None

    def _note_carrier_verdict(self, value: Any, keyable: bool | str) -> None:
        # Holds the object, so its id cannot be reused while the entry stands.
        if len(self._carrier_verdicts) >= 4096:
            self._carrier_verdicts.clear()
        self._carrier_verdicts[id(value)] = (value, keyable)

    def _decorator_global_names(self, fn: Callable) -> tuple[str, ...]:
        """Names the decorator expressions on *fn*'s ``def`` read from its module.

        ``@np.vectorize(otypes=OT)`` is evaluated once, at import, from the
        module's ``OT`` -- a name the function's body never mentions, so
        changing it changed nothing the key could see. Only the function a
        decorator wraps has these lines (its source starts at the first
        decorator); the values are folded like any other read global, so
        modules and callables among them are skipped there. Cached per code.
        """
        code = getattr(fn, "__code__", None)
        if code is None:
            return ()
        cached = self._decorator_names_cache.get(code)
        if cached is not None:
            return cached
        names: tuple[str, ...] = ()
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(code)))
            node = next((n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
            if node is not None and node.decorator_list:
                names = tuple(
                    dict.fromkeys(
                        n.id
                        for deco in node.decorator_list
                        for n in ast.walk(deco)
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                    )
                )
        except SOURCE_RETRIEVAL_ERRORS + (SyntaxError, ValueError):
            names = ()
        if len(self._decorator_names_cache) >= 4096:
            self._decorator_names_cache.clear()
        self._decorator_names_cache[code] = names
        return names

    def _fold_dependency_read_globals(self, func: Callable, func_name: str, state_hash: str) -> str:
        """Fold the globals the CACHED functions this one calls read.

        The third of the three channels a global can reach a key through, and
        the one that was missing. A cached function's own globals are folded by
        ``_fold_read_globals``; its plain helpers' by
        ``_fold_helper_read_globals``; a cached CALLEE's were folded into that
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
            for name in self._read_global_data_names(func):
                seen.add((id(own_globals), name))
        visited = {func_name}
        stack = sorted(self.graph.get_dependencies(func_name))
        while stack:
            dep = stack.pop()
            if dep in visited:
                continue
            visited.add(dep)
            stack.extend(sorted(self.graph.get_dependencies(dep)))
            dep_func = self.functions.get(dep)
            if dep_func is None or getattr(dep_func, "__globals__", None) is None:
                continue
            state_hash = self._fold_read_globals(
                dep_func,
                func_name,
                state_hash,
                owner_code=owner_code,
                seen=seen,
            )
            dep_report = self._purity_reports.get(dep)
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

    def _read_module_attr_pairs(self, func: Callable) -> tuple[tuple[str, str], ...]:
        """``(module_global, attribute)`` pairs the body reads, from bytecode.

         ``import conf; conf.RATE`` compiles to ``LOAD_GLOBAL conf`` followed by
         ``LOAD_ATTR RATE``. Only the *module* reaches ``_read_global_data_names``,
         and modules are filtered out at fold time, so the attribute was never
         keyed on: ``conf.RATE`` went permanently stale while the equivalent
         ``from conf import RATE`` invalidated correctly. Two spellings of one
         dependency, one of them silently wrong.

         Walks nested scopes for the same reason the sibling channel does
        : a read that happens only inside a genexp still counts.
        """
        code = getattr(func, "__code__", None)
        if code is None:
            return ()
        cached = self._module_attr_cache.get(code)
        if cached is not None:
            return cached

        pairs: set[tuple[str, str]] = set()
        # `vars(conf)["K"]` / `getattr(conf, "K")`: the attribute is a string
        # constant rather than a LOAD_ATTR, so the pair below never formed and
        # the constant was not keyed on. Every module read in this scope is
        # paired with every identifier-shaped constant in it; a pair that does
        # not exist is dropped at fold time by the getattr below.
        g = getattr(func, "__globals__", None) or {}
        for scope in iter_code_scopes(code):
            modules = [n for n in (scope.co_names or ()) if isinstance(g.get(n), types.ModuleType)]
            if modules:
                for const in scope.co_consts or ():
                    if isinstance(const, str) and const.isidentifier():
                        pairs.update((m, const) for m in modules)
        for scope in iter_code_scopes(code):
            instrs = list(dis.get_instructions(scope))
            for prev, nxt in zip(instrs, instrs[1:]):
                if prev.opname != "LOAD_GLOBAL":
                    continue
                if nxt.opname not in ("LOAD_ATTR", "LOAD_METHOD"):
                    continue
                name, attr = prev.argval, nxt.argval
                if not isinstance(name, str) or not isinstance(attr, str):
                    continue
                if attr.startswith("__"):
                    continue
                pairs.add((name, attr))
        result = tuple(sorted(pairs))
        if len(self._module_attr_cache) < 4096:
            self._module_attr_cache[code] = result
        return result

    def _local_binding_plan(self, func: Callable) -> tuple | None:
        """What `_local_binding_parts` needs from *func*'s source, per code object.

        ``(imports, attr_reads, bare_reads)``: the names an import written in
        the body binds (``name -> (module, prefix)``), the ``name.ATTR`` reads
        of any local or closure name, and the bare reads of local names.
        """
        code = getattr(func, "__code__", None)
        if code is None:
            return None
        if code in self._local_binding_cache:
            return self._local_binding_cache[code]
        plan = None
        try:
            tree = ast.parse(textwrap.dedent(own_source(func)))
            func_def = next(
                (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))), None
            )
            if func_def is not None:
                imports = local_import_map(func_def, func)
                watched = set(imports) | set(code.co_freevars or ())
                attr_reads: dict[str, set[str]] = {}
                bare_reads: set[str] = set()
                for node in ast.walk(func_def):
                    if (
                        isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Name)
                        and node.value.id in watched
                        and not node.attr.startswith("__")
                    ):
                        attr_reads.setdefault(node.value.id, set()).add(node.attr)
                    elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in imports:
                        bare_reads.add(node.id)
                if attr_reads or bare_reads:
                    plan = (imports, attr_reads, bare_reads)
        except (*SOURCE_RETRIEVAL_ERRORS, SyntaxError, ValueError):
            plan = None
        if len(self._local_binding_cache) < 4096:
            self._local_binding_cache[code] = plan
        return plan

    def _local_binding_parts(self, func: Callable) -> list[tuple[str, str]]:
        """Key parts for data reached through names the module's globals never see.

        Two shapes, both served stale (a constant 2 -> 0 and the old
        report back):

        * an import written INSIDE the body -- ``from .settings import
          ROUNDING``, or ``from . import settings`` then ``settings.ROUNDING``
          -- binds a local, so the globals channels never saw it (#132 followed
          only the FUNCTIONS such an import binds);
        * a module held in a closure: ``from . import settings`` inside a
          decorator factory, read by the wrapper as ``settings.ROUNDING``.

        Data values are folded, and a module's ``ATTR`` reads, the same way the
        ``module.ATTR`` channel folds a global module's. A user module the
        body has not imported yet is imported here -- the import the body is
        about to make; a library module only if it is already loaded.
        """
        plan = self._local_binding_plan(func)
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
                stabilized = stabilize_for_global_hash(value, self._data_callable_identity)
                parts.append((label, self._hash_arg_payload((stabilized,), {})))
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

    def _module_attr_parts(
        self,
        func: Callable,
        func_name: str,
        g: dict,
        *,
        learned: frozenset | set = frozenset(),
        watch: dict | None = None,
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
        is given, so the drift guard can see it too (`_carried_global_hash`).
        *learned* is the drift guard's verdict: labels not to fold.
        """
        parts: list[tuple[str, str]] = []
        own_pkg = own_package(func)
        for mod_name, attr in self._read_module_attr_pairs(func):
            obj = g.get(mod_name)
            is_mod = isinstance(obj, types.ModuleType) and is_user_module(obj, own_pkg)
            # ``Cfg.LIMIT`` -- a class constant read through the class NAME -- is
            # the same bytecode shape (LOAD_GLOBAL Cfg; LOAD_ATTR LIMIT) but was
            # skipped because ``Cfg`` is a class, not a module, so editing the
            # constant served stale. Fold user-class attributes too.
            is_cls = isinstance(obj, type) and is_user_class(obj, own_pkg)
            if not (is_mod or is_cls):
                continue
            try:
                value = inspect.getattr_static(obj, attr) if is_cls else getattr(obj, attr)
            except (AttributeError, Exception):  # noqa: BLE001 - never break a call
                continue
            label = f"{mod_name}.{attr}"
            if isinstance(value, types.ModuleType) or isinstance(value, type):
                continue
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
                    carried = self._carried_global_hash(value, getattr(func, "__module__", None))
                    if carried is not None:
                        parts.append((f"{label}#carried", carried))
                        watch[label] = (carried, "carrier", (vars(obj), attr), None)
                        continue
                # One level only: fold the constants the helper itself reads.
                # Deeper recursion would drag in whole transitive namespaces for
                # a diminishing chance of catching a real edit.
                if getattr(value, "_cash_cached", False):
                    # A cached helper is cash's wrapper, whose globals are
                    # cash's own: it warned KEY-UNHASHABLE-GLOBAL for
                    # 'rates.fetch.ACTIVE_CONFIG' on every run (the class-method
                    # twin is handled in source_norm).
                    value = getattr(value, "__wrapped__", value)
                helper_globals = getattr(value, "__globals__", None)
                if not isinstance(helper_globals, dict):
                    continue
                for inner in self._read_global_data_names(value):
                    if inner not in helper_globals:
                        continue
                    iv = helper_globals[inner]
                    if isinstance(iv, types.ModuleType) or isinstance(iv, type):
                        continue
                    if callable(iv) and not isinstance(iv, (dict, list, tuple, set)):
                        continue
                    h = self._safe_global_hash(iv, func_name, f"{label}.{inner}")
                    if h is not None:
                        parts.append((f"{label}.{inner}", h))
                continue
            h = self._safe_global_hash(value, func_name, label)
            if h is not None:
                parts.append((label, h))
        return parts

    def _safe_global_hash(self, value: Any, func_name: str, label: str) -> str | None:
        """Hash *value* for the key, warning once and skipping if it cannot be."""
        try:
            stabilized = stabilize_for_global_hash(value, self._data_callable_identity)
            return self._hash_arg_payload((stabilized,), {})
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError):
            self._notices.warn_once(
                CashImpurityWarning,
                func_name,
                label,
                f"@cash.cache on {func_name}: reads '{label}' whose value could not "
                f"be hashed, so changes to it will NOT invalidate the cache.",
                code="KEY-UNHASHABLE-GLOBAL",
                fix=UNHASHABLE_GLOBAL_FIX,
            )
            return None
