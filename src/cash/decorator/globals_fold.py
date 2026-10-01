"""Folding the globals a function reads, and what they carry, into its key."""

from __future__ import annotations

import ast
import contextvars
import dis
import enum
import functools
import hashlib
import inspect
import pickle
import sys
import textwrap
import types
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .._memo import CODE_OBJECTS, LruMemo
from ..analysis.purity_analyzer import (
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
from ..dependency_state import SysModulesHelperResolver, ledger_note
from ..effects import environment_component
from ..exceptions import SOURCE_RETRIEVAL_ERRORS, CashImpurityWarning
from ..install_paths import is_user_module
from ..source_norm import getsource, own_source
from ..value_types import CODELESS_PRIMS
from .arg_hashing import is_opaque
from .call_state import CAPTURE_WATCH, KeyBuildFailed
from .closure_fold import iter_code_scopes, unsafe_uses_of, waived_use_filter
from .function_identity import hash_callable_source
from .key_values import (
    SYNC_TYPES,
    carried_payload,
    held_partials,
    is_immutable_capture,
    iter_contained,
    plain_data_kind,
    reduced_state,
    stabilize_for_global_hash,
)
from .user_code import cash_wrapped, is_cash_wrapper, is_user_class, is_user_code_object, own_package, wraps_code

if TYPE_CHECKING:
    from ..dependency_state import DependencyStateHasher
    from .arg_hashing import ArgHasher
    from .closure_fold import HelperIdentity
    from .code_args import CodeArgs
    from .code_identity import CodeIdentity
    from .purity_checks import LearnedMutations
    from .registry import FunctionRegistry
    from .reporting import Notices

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


#: Names whose load means the code reads a docstring at run time.
DOCSTRING_READS = frozenset({"__doc__", "getdoc", "cleandoc"})


#: A plain operand load: the key between a global's load and a subscript store.
#: 3.12 adds LOAD_FAST_CHECK (a local that may be unbound); 3.14 loads most
#: locals with LOAD_FAST_BORROW and small int constants with LOAD_SMALL_INT, so
#: without them `G[k] = v` and `del G[0]` read as plain reads there.
_OPERAND_LOADS = frozenset(
    {"LOAD_FAST", "LOAD_FAST_CHECK", "LOAD_FAST_BORROW", "LOAD_CONST", "LOAD_SMALL_INT", "LOAD_DEREF", "LOAD_NAME"}
)


def _bytecode_written_attrs(code: types.CodeType) -> set[str]:
    """Attribute names *code* stores, deletes or mutates in place:
    ``C.count += 1``, ``self.n = 0``, ``cls.registry[k] = v``,
    ``type(self).seen.append(x)``. Exact shapes, as for
    `_bytecode_mutated_globals`."""
    found: set[str] = set()
    instrs = list(dis.get_instructions(code))
    for i, ins in enumerate(instrs):
        if ins.opname in ("STORE_ATTR", "DELETE_ATTR"):
            found.add(ins.argval)
            continue
        if ins.opname not in ("LOAD_ATTR", "LOAD_METHOD"):
            continue
        after = instrs[i + 1 : i + 3]
        if not after:
            continue
        first = after[0]
        if first.opname in ("LOAD_ATTR", "LOAD_METHOD") and first.argval in REPORTED_METHODS:
            found.add(ins.argval)
        elif (
            len(after) == 2 and first.opname in _OPERAND_LOADS and after[1].opname in ("STORE_SUBSCR", "DELETE_SUBSCR")
        ):
            found.add(ins.argval)
    return {n for n in found if isinstance(n, str)}


def _bytecode_mutated_globals(scopes: tuple, names: set[str]) -> set[str]:
    """The *names* compiled code plainly writes into, for code without source.

    A load of the global followed by a writing method (`REPORTED_METHODS`:
    ``append``, ``update``, ...) or an attribute store (``obj.x = v``), or
    by one operand and a subscript store (``table[k] = v``). Exact shapes
    only: a global read as the VALUE stored (``d[k] = G``) must stay
    folded. What this misses is caught at run time by the provisional
    watch, one miss later.
    """
    found: set[str] = set()
    for scope in scopes:
        instrs = list(dis.get_instructions(scope))
        for i, ins in enumerate(instrs):
            if ins.opname != "LOAD_GLOBAL" or ins.argval not in names:
                continue
            after = instrs[i + 1 : i + 3]
            if not after:
                continue
            first = after[0]
            if (first.opname in ("LOAD_ATTR", "LOAD_METHOD") and first.argval in REPORTED_METHODS) or first.opname in (
                "STORE_ATTR",
                "DELETE_ATTR",
            ):
                found.add(ins.argval)
            elif (
                len(after) == 2
                and first.opname in _OPERAND_LOADS
                and after[1].opname in ("STORE_SUBSCR", "DELETE_SUBSCR")
            ):
                found.add(ins.argval)
    return found


#: A miss in `GlobalsFold._local_binding_cache`, whose entries may be None.
_NO_PLAN = object()


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


def _is_cash_decorator(deco: ast.expr, module_globals: dict[str, Any]) -> bool:
    """Whether the decorator expression *deco* is cash's own ``@app.cache``.

    Its names configure the caching, not the result, and the ``Cash``
    instance they reach cannot be hashed: ``@app.cache`` over a
    ``functools.wraps`` decorator warned that the function read the
    unhashable global ``app``.
    """
    from ..core import Cash  # deferred: core builds this module

    root = deco
    while isinstance(root, (ast.Call, ast.Attribute)):
        root = root.func if isinstance(root, ast.Call) else root.value
    if not isinstance(root, ast.Name):
        return False
    value = module_globals.get(root.id)
    return isinstance(value, Cash) or value is sys.modules.get("cash")


#: The opcodes that read a module global by name (``LOAD_NAME`` in a class
#: body or at module level; ``LOAD_FROM_DICT_OR_GLOBALS`` in 3.12+ class bodies).
_GLOBAL_LOADS = frozenset({"LOAD_GLOBAL", "LOAD_NAME", "LOAD_FROM_DICT_OR_GLOBALS"})
#: The opcodes that read an attribute (``LOAD_METHOD`` before 3.12).
_ATTR_OPS = frozenset({"LOAD_ATTR", "LOAD_METHOD"})


class GlobalsFold:
    """The module data a function and its helpers read, folded into the state
    segment: globals, ``module.ATTR`` reads, data reached through local
    bindings, what a library-made callable carries, and the environment."""

    def __init__(
        self,
        args: ArgHasher,
        code: CodeIdentity,
        helpers: HelperIdentity,
        registry: FunctionRegistry,
        state_hasher: DependencyStateHasher,
        mutations: LearnedMutations,
        notices: Notices,
    ) -> None:
        self._args = args
        self._code = code
        self._helpers = helpers
        self._registry = registry
        self._state_hasher = state_hasher
        self._mutations = mutations
        self._notices = notices
        # The same live re-resolution the state hasher does, for functions
        # found inside data globals (`data_callable_identity`).
        self._data_helper_resolver = SysModulesHelperResolver(helpers.identity)
        # code object -> global names its decorator expressions read
        self._decorator_names_cache: LruMemo[Any, tuple[str, ...]] = LruMemo(CODE_OBJECTS)
        # code object -> (tuple of global names it reads, the names among them
        # folded only provisionally). One entry, so the two never disagree.
        # See `read_global_data_names`; a missing entry means "unknown", which
        # `fold_read_globals` treats as "watch everything".
        self._global_read_cache: LruMemo[Any, tuple[tuple[str, ...], frozenset]] = LruMemo(CODE_OBJECTS)
        # (module_global, attribute) read pairs per code object; see
        # `_read_module_attr_pairs`.
        self._module_attr_cache: LruMemo[Any, tuple[tuple[str, str], ...]] = LruMemo(CODE_OBJECTS)
        self._local_binding_cache: LruMemo[Any, tuple | None] = LruMemo(CODE_OBJECTS)
        self._carrier_verdicts: LruMemo[int, tuple[Any, bool | str]] = LruMemo(CODE_OBJECTS)
        # code object -> whether it reads a docstring; see `_reads_docstrings`.
        self._docstring_reads: LruMemo[Any, bool] = LruMemo(CODE_OBJECTS)
        # class -> (its surface functions, the names their code reads); see
        # `class_parts`. A redefined class is a new key.
        self._class_code_cache: LruMemo[type, tuple[tuple, frozenset]] = LruMemo(CODE_OBJECTS)
        # class -> (the immutable data values last hashed, their digest); see
        # `_class_data_digest`.
        self._class_data_memo: LruMemo[type, tuple[tuple, str]] = LruMemo(CODE_OBJECTS)
        # class -> its user bases and their data names; see `_class_layout`.
        self._class_layout_cache: LruMemo[type, tuple] = LruMemo(CODE_OBJECTS)
        # code object -> whether folding its globals can find anything; see
        # `_may_read_data`.
        self._reads_anything: LruMemo[Any, bool] = LruMemo(CODE_OBJECTS)
        # id(value) -> (value, digest) for immutable plain data globals; see
        # `global_value_digest`. The value is held, so its id is not reused.
        self._immutable_digests: LruMemo[int, tuple[Any, str]] = LruMemo(CODE_OBJECTS)
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

    def read_global_data_names(self, func: Callable) -> tuple[str, ...]:
        """Global names *func* references that are candidates for data-folding.

        ``co_names`` intersected with the function's globals, minus the import
        machinery dunders (``MACHINERY_DUNDERS``) and minus any global the
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
            return cached[0]
        g = getattr(func, "__globals__", {}) or {}

        scopes = tuple(iter_code_scopes(code))
        written = {
            instr.argval
            for scope in scopes
            for instr in dis.get_instructions(scope)
            if instr.opname in ("STORE_GLOBAL", "DELETE_GLOBAL")
        }
        # Names LOADED as globals, not every name in ``co_names``: that also
        # holds attribute names, so `b.lock` read the module's unrelated `lock`
        # and warned KEY-UNHASHABLE-GLOBAL about a global never read.
        candidates = {
            instr.argval
            for scope in scopes
            for instr in dis.get_instructions(scope)
            if instr.opname in _GLOBAL_LOADS
            and instr.argval in g
            and instr.argval not in MACHINERY_DUNDERS
            and instr.argval not in written
        }
        # A name spelled as a string reads the same global: `globals()["K"]`
        # is a LOAD_CONST, so `co_names` never had it and editing K served the
        # old answer -- 20 where an uncached run gives 500. The code channel already resolves string
        # constants this way (`CodeRefs.targets`); this is its data twin.
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
        # `PurityChecks.learn_mutating_captures`, which demotes any that the call is actually
        # observed to mutate.
        provisional: frozenset = frozenset()
        if candidates:
            try:
                tree = ast.parse(textwrap.dedent(getsource(func)))
                hard = unsafe_uses_of(
                    tree,
                    candidates,
                    bare_args=False,
                    mutating_methods_only=True,
                )
                suspected = unsafe_uses_of(tree, candidates) - hard
                provisional = unsafe_uses_of(tree, suspected, waived=waived_use_filter(func, tree))
                # Suspected only on waived lines (`LEDGER.record(r)  #
                # @cash:assume-safe`, or inside `with cash.assume_safe():`):
                # the audited effect moves it on every call, so keying on it
                # made every hit impossible and demoting it warned about the
                # very line that was audited.
                hard |= suspected - provisional
                candidates -= hard
            except SOURCE_RETRIEVAL_ERRORS:
                # No source (`python - <<EOF`, `python -c`, `exec`): the
                # bytecode stands in for the AST. What it plainly writes
                # (`calls.append(x)`, `table[k] = v`) stays out; the rest is
                # folded PROVISIONALLY -- watched after a miss, and dropped
                # once a call is seen to move it. Folding none of them
                # served a stale result whenever a global it reads changed.
                candidates -= _bytecode_mutated_globals(scopes, candidates)
                provisional = frozenset(candidates)
        names = tuple(sorted(candidates))
        # A MISSING entry is not "nothing is provisional" --
        # `GlobalsFold.fold_read_globals` reads that as "watch every folded
        # name", which costs an extra hash per miss and is the safe direction.
        self._global_read_cache[code] = (names, provisional)
        return names

    def data_callable_identity(self, fn: Any) -> str:
        """A callable found INSIDE a data global, identified by what calling it runs.

        A registry -- ``STEPS = {"load": load_step}`` read by a cached
        ``run(name)`` that calls ``STEPS[name](x)`` -- runs each step's helpers
        too, so each function's own source is not enough, and a cached
        function stored there is not cash's wrapper code. A
        cached function counts as its dependency state, the same
        as a call to it would; a plain function of the user's as its source
        plus its helpers, re-resolved live like any helper's.
        """
        if is_cash_wrapper(fn) and not is_mock(fn):
            state = getattr(fn, "_cash_state", None)
            if state is not None:
                # Its whole state, globals and environment included, built by
                # the instance that owns it: the dependency state alone left
                # out the globals it reads, and one on another instance was
                # not in this registry at all.
                return "cached:" + state()
            fn = getattr(fn, "__wrapped__", fn)
        if not isinstance(fn, types.FunctionType):
            return hash_callable_source(fn)
        own = self._helpers.identity(fn)

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
            and not self._may_read_data(func)
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
        pairs: list[tuple] = [(gid, n) for n in (*self.read_global_data_names(func), *extra_names) if n in g]
        pairs.extend(("default", id(default)) for default in self._function_defaults(func))
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
        names = self.read_global_data_names(func)
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
        # folded name rather than fold one blind (see `GlobalsFold.read_global_data_names`).
        cached = self._global_read_cache.get(code)
        provisional = cached[1] if cached is not None else None
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
                carried = self.carried_global_hash(v, root_module)
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
                h = self.global_value_digest(v, plain)
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
        if code is not None and self._reads_docstrings(code):
            parts.extend(self._docstring_parts(code, g, own_pkg))
        # A function default is evaluated where the `def` stands, so what a
        # default LAMBDA reads (`def g(x, fn=lambda v: v + K)`) is in no scope
        # of *func*'s, so it is folded here or editing K would keep the key.
        for default in self._function_defaults(func):
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

    def global_value_digest(self, value: Any, plain: str | None = None) -> str:
        """The digest a data global's *value* is keyed by, and checked against
        after the call (`PurityChecks`). *plain* is `plain_data_kind` of it.

        Plain data is hashed as it is: it holds no callable for
        `stabilize_for_global_hash` to replace. Immutable plain data -- a
        number, a string, a tuple of them -- cannot change, so its digest is
        kept while the global holds that same object: a module constant read
        on every hit cost a full hash each time.
        """
        if plain is None:
            return self._args.hash_payload((stabilize_for_global_hash(value, self.data_callable_identity),), {})
        args = self._args
        memo = plain == "immutable" and not (args.override_hashers or args.type_hashers)
        if memo:
            entry = self._immutable_digests.get(id(value))
            if entry is not None and entry[0] is value:
                return entry[1]
        digest = args.plain_value_digest(value)
        if digest is None:
            digest = args.hash_payload((value,), {})
        if memo:
            self._immutable_digests[id(value)] = (value, digest)
        return digest

    def _reads_docstrings(self, code: Any) -> bool:
        """Does *code* read a docstring at run time (``f.__doc__``,
        ``inspect.getdoc(tool)``, ``getattr(C, "__doc__")``)? Cached per code."""
        cached = self._docstring_reads.get(code)
        if cached is None:
            cached = any(
                DOCSTRING_READS & set(scope.co_names or ()) or "__doc__" in (scope.co_consts or ())
                for scope in iter_code_scopes(code)
            )
            self._docstring_reads[code] = cached
        return cached

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
        for mod_name, attr in self._module_attr_cache.get(code) or ():
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

    @staticmethod
    def _function_defaults(func: Callable) -> list[types.FunctionType]:
        """The user functions among *func*'s parameter defaults."""
        found: list[types.FunctionType] = []
        for container in (
            getattr(func, "__defaults__", None) or (),
            (getattr(func, "__kwdefaults__", None) or {}).values(),
        ):
            for value in container:
                if (
                    isinstance(value, types.FunctionType)
                    and value is not func
                    and own_code_is_user(value, getattr(func, "__module__", None))
                ):
                    found.append(value)
        return found

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
            for name in self.read_global_data_names(func):
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
            digest = self.carried_state_digest(live)
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
                extra_names=self._decorator_global_names(target),
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
            if not self._may_read_data(fn):
                continue
            h = self.fold_read_globals(fn, func_name, "", owner_code=owner_code, seen=class_seen)
            if h:
                parts.append((f"{label}#reads:{fn.__qualname__}", h))
        if seen is not None:
            seen |= class_seen - excluded
        return parts

    def _may_read_data(self, fn: Any) -> bool:
        """Could `GlobalsFold.fold_read_globals` find anything in *fn*? False for
        a method that reads no global, no ``module.attr``, no import in its
        body, no docstring and has no function default -- most of a class's
        methods, and every one a dataclass generates -- so a class costs a
        lookup per such method instead of a fold."""
        code = fn.__code__
        cached = self._reads_anything.get(code)
        if cached is None:
            cached = bool(
                self.read_global_data_names(fn)
                or self._read_module_attr_pairs(fn)
                or self._reads_docstrings(code)
                or self._local_binding_plan(fn)
            )
            self._reads_anything[code] = cached
        return cached or bool(self._function_defaults(fn))

    def _class_code(self, cls: type) -> tuple[tuple, frozenset, frozenset, frozenset]:
        """``(functions, written globals, names read, attributes written)``
        for *cls*, per class.

        The functions are `class_surface_functions`. The written globals are
        ``(id(module globals), name)`` for every global one of them assigns or
        mutates in place -- state the class keeps, excluded for all of them.
        The names are what their code looks up, for naming a class attribute
        that is read but cannot be hashed. The attributes written are those
        its code stores or mutates in place (`_bytecode_written_attrs`): a
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
            data = set(self.read_global_data_names(fn))
            for scope in iter_code_scopes(fn.__code__):
                written |= _bytecode_written_attrs(scope)
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
                label: stabilize_for_global_hash(value, self.data_callable_identity) for label, value in items
            }
            digest = self._args.hash_payload((stabilized,), {})
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError):
            kept: dict[str, str] = {}
            for label, value in items:
                try:
                    kept[label] = self._args.hash_payload(
                        (stabilize_for_global_hash(value, self.data_callable_identity),), {}
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

    def carried_state_digest(self, value: Any) -> str | None:
        """Digest of the data a callable carries besides its code, or None.

        See `carried_payload`. Silent on failure: the code is still keyed,
        and a warning here would fire on every class-based decorator whose
        state is just the function it wraps.
        """
        payload = carried_payload(value)
        if payload is None:
            return None
        try:
            stabilized = stabilize_for_global_hash(payload, self.data_callable_identity)
            return self._args.hash_payload((stabilized,), {})
        except Exception:  # noqa: BLE001 - never break a call over this
            return None

    def carried_global_hash(self, value: Any, root_module: str | None) -> str | None:
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
        and `carried_state_digest` keys.

        Some of these change when called -- a bound ``rng.normal`` advances
        its generator, ``np.vectorize`` fills a cache -- so every one is
        watched by `PurityChecks.learn_mutating_captures`, which stops folding it after
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
            if is_cash_wrapper(value):
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
            stabilized = stabilize_for_global_hash(payload, self.data_callable_identity)
            return self._args.hash_payload((stabilized,), {})
        except Exception:  # noqa: BLE001 - unkeyable before, never break a call over it
            self._note_carrier_verdict(value, False)
            return None

    def _note_carrier_verdict(self, value: Any, keyable: bool | str) -> None:
        # Holds the object, so its id cannot be reused while the entry stands.
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
            tree = ast.parse(textwrap.dedent(getsource(code)))
            node = next((n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
            if node is not None and node.decorator_list:
                g = getattr(fn, "__globals__", None) or {}
                names = tuple(
                    dict.fromkeys(
                        n.id
                        for deco in node.decorator_list
                        if not _is_cash_decorator(deco, g)
                        for n in ast.walk(deco)
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                    )
                )
        except SOURCE_RETRIEVAL_ERRORS + (SyntaxError, ValueError):
            names = ()
        self._decorator_names_cache[code] = names
        return names

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
            for name in self.read_global_data_names(func):
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

    def _read_module_attr_pairs(self, func: Callable) -> tuple[tuple[str, str], ...]:
        """``(module_path, attribute)`` pairs the body reads, from bytecode.

        ``conf.RATE``, ``pkg.conf.RATE`` and ``m = pkg.conf; m.RATE`` are
        three spellings of one dependency, and the global the body names is
        a module in each: none of them reaches
        `GlobalsFold.read_global_data_names`, so the attribute is keyed here.

        * A global followed by a chain of attribute loads
          (``LOAD_GLOBAL pkg; LOAD_ATTR conf; LOAD_ATTR RATE``) gives one
          pair per link: ``("pkg", "conf")`` and ``("pkg.conf", "RATE")``.
          `GlobalsFold.module_attr_parts` resolves the dotted path and folds
          the pairs whose path is a user module.
        * A module the code takes whole (bound to a local, passed on, or read
          with ``vars(conf)["K"]`` / ``getattr(conf, "K")``) is paired with
          every attribute name and identifier-shaped string constant in the
          code. A pair that names nothing is dropped at fold time; one that
          names an attribute the code does not read only adds a part.

        Nested scopes count: a read inside a genexp is a read of the body.
        """
        code = getattr(func, "__code__", None)
        if code is None:
            return ()
        cached = self._module_attr_cache.get(code)
        if cached is not None:
            return cached

        g = getattr(func, "__globals__", None) or {}
        pairs: set[tuple[str, str]] = set()
        whole: set[str] = set()
        names: set[str] = set()
        scopes = list(iter_code_scopes(code))
        for scope in scopes:
            names.update(c for c in scope.co_consts or () if isinstance(c, str) and c.isidentifier())
            instrs = [i for i in dis.get_instructions(scope) if i.opname != "EXTENDED_ARG"]
            for i, ins in enumerate(instrs):
                if ins.opname in _ATTR_OPS and isinstance(ins.argval, str):
                    names.add(ins.argval)
                if ins.opname != "LOAD_GLOBAL" or not isinstance(ins.argval, str):
                    continue
                path = ins.argval
                value = g.get(path)
                j = i + 1
                while j < len(instrs) and instrs[j].opname in _ATTR_OPS and isinstance(instrs[j].argval, str):
                    attr = instrs[j].argval
                    if attr.startswith("__"):
                        break
                    pairs.add((path, attr))
                    path = f"{path}.{attr}"
                    value = getattr(value, attr, None) if isinstance(value, types.ModuleType) else None
                    j += 1
                else:
                    if isinstance(value, types.ModuleType):
                        whole.add(path)
        for path in whole:
            pairs.update((path, n) for n in names if not n.startswith("__"))
        result = tuple(sorted(pairs))
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
        cached = self._local_binding_cache.get(code, _NO_PLAN)
        if cached is not _NO_PLAN:
            return cached
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
        self._local_binding_cache[code] = plan
        return plan

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
                stabilized = stabilize_for_global_hash(value, self.data_callable_identity)
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
        is given, so the drift guard can see it too (`GlobalsFold.carried_global_hash`).
        *learned* is the drift guard's verdict: labels not to fold. A user
        class read as ``module.Class`` (and an instance's class) is folded by
        `GlobalsFold.class_parts`, *owner_code* and *seen* as there.
        """
        parts: list[tuple[str, str]] = []
        own_pkg = own_package(func)
        for mod_name, attr in self._read_module_attr_pairs(func):
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
                    carried = self.carried_global_hash(value, getattr(func, "__module__", None))
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
                for inner in self.read_global_data_names(value):
                    if inner not in helper_globals:
                        continue
                    iv = helper_globals[inner]
                    if isinstance(iv, types.ModuleType) or isinstance(iv, type):
                        continue
                    if callable(iv) and not isinstance(iv, (dict, list, tuple, set)):
                        continue
                    h = self.safe_global_hash(iv, func_name, f"{label}.{inner}")
                    if h is not None:
                        parts.append((f"{label}.{inner}", h))
                continue
            h = self.safe_global_hash(value, func_name, label)
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
        h = self.safe_global_hash(value, func_name, f"{name}.{attr}")
        return [(f"{name}.{attr}", h)] if h is not None else []

    def safe_global_hash(self, value: Any, func_name: str, label: str) -> str | None:
        """Hash *value* for the key, warning once and skipping if it cannot be."""
        try:
            stabilized = stabilize_for_global_hash(value, self.data_callable_identity)
            return self._args.hash_payload((stabilized,), {})
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
