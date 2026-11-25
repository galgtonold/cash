"""Folding what a function captures -- closure cells, parameter defaults, the
instance it is bound to, its helpers' captures -- into its key."""

from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import operator
import pickle
import textwrap
import types
import weakref
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from .. import _weak
from .._memo import CODE_OBJECTS, LruMemo
from .._plain_data import identity_changed, identity_snapshot
from .._type_memo import class_mro, class_name, class_qualname
from .._user_attrs import plain_getattr
from ..analysis.annotations import assume_safe_block_lines
from ..analysis.bytecode import DEREF_WRITES, argvals, instructions
from ..analysis.helper_code import is_mock
from ..analysis.purity_policy import REPORTED_METHODS
from ..effect_observer import line_waived
from ..exceptions import SOURCE_RETRIEVAL_ERRORS, CashCacheIneffectiveWarning
from ..jit_functions import jit_identity, jit_python_function
from ..source_reading import getsource, getsourcelines
from ..tracking.randomness import rng_carrier_kind
from ..value_types import IMMUTABLE_VALUE_TYPES
from .arg_hashing import CODE_VALUE_TYPES, is_opaque
from .call_state import CAPTURE_WATCH, KEY_BUILD_DIGESTS, KeyBuildFailed
from .function_identity import code_fingerprint, hash_callable_source
from .key_values import (
    SYNC_TYPES,
    carried_payload,
    code_ref,
    is_immutable_capture,
    plain_data_kind,
    reduced_state,
    rewrite_held_code,
    stabilize_for_global_hash,
)
from .module_identity import module_ref
from .user_code import is_cash_wrapper, is_user_class, is_user_code_object, own_package

if TYPE_CHECKING:
    from .arg_hashing import ArgHasher
    from .class_data import ClassDataFold
    from .code_args import CodeArgs
    from .global_values import GlobalValues
    from .purity_checks import LearnedMutations
    from .reporting import Notices


#: Callables that are code, not an object holding data: followed as helpers.
_CODE_CALLABLES = (
    types.FunctionType,
    types.MethodType,
    types.BuiltinFunctionType,
    type,
    functools.partial,
)


def is_user_callable_instance(value: Any) -> bool:
    """Is *value* an instance of the user's own class with a ``__call__``,
    rather than a function, method, class or partial?"""
    if isinstance(value, _CODE_CALLABLES) or is_cash_wrapper(value):
        return False
    cls = type(value)
    return is_user_code_object(cls) and not is_opaque(value)


def bound_owner(func: Any) -> tuple[bool, Any]:
    """``(True, the object)`` *func* is bound to, ``(False, None)`` when it
    is bound to none: the ``__self__`` of a bound method (a class for a
    classmethod), of a C method (``d.get``) or a method-wrapper
    (``a.__len__``), through any ``functools.partial`` around one.

    The body's ``self`` that the call's arguments do not carry: the key
    folds it (`ClosureFold.fold_bound_self`) and the in-place-change check
    looks at it (`PurityChecks.bound_state`). A builtin function's
    ``__self__`` is its module and binds nothing.
    """
    while isinstance(func, functools.partial):
        func = func.func
    if inspect.ismethod(func):
        return True, func.__self__
    if isinstance(func, (types.BuiltinMethodType, types.MethodWrapperType)):
        owner = plain_getattr(func, "__self__", None)
        if owner is not None and not isinstance(owner, types.ModuleType):
            return True, owner
    return False, None


def bound_values(func: Any) -> list[tuple[str, Any]]:
    """``[(name, value)]`` of what *func* carries bound into every call: the
    instance it is bound to (`bound_owner`), as ``'self'`` (``'cls'`` for a
    class), or *func* itself when it is the user's callable instance, then
    the arguments of any ``functools.partial`` around it, named by the
    parameter they fill where the signature says."""
    if is_user_callable_instance(func):
        return [("self", func)]
    found: list[tuple[str, Any]] = []
    bound, owner = bound_owner(func)
    if bound:
        found.append(("cls" if isinstance(owner, type) else "self", owner))
    partials = []
    while isinstance(func, functools.partial):
        partials.append(func)
        func = func.func
    if not partials:
        return found
    # The outermost partial holds every positional binding in order:
    # ``partial(partial(f, a), b)`` is flattened to ``partial(f, a, b)``
    # by functools; one that is not still names them by position.
    try:
        params = [
            p.name
            for p in inspect.signature(func).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):
        params = []
    seen = set()
    for part in partials:
        for i, value in enumerate(part.args):
            name = params[i] if i < len(params) else f"partial args[{i}]"
            if name not in seen:
                seen.add(name)
                found.append((name, value))
        for name, value in (part.keywords or {}).items():
            if name not in seen:
                seen.add(name)
                found.append((name, value))
    return found


def unsafe_uses_of(
    tree: ast.AST,
    names: set[str],
    *,
    bare_args: bool = True,
    mutating_methods_only: bool = False,
    waived: Callable[[ast.AST], bool] | None = None,
) -> frozenset:
    """Return the subset of *names* the AST body *may mutate*.

    Disqualifying uses of a name ``n``: method calls on it
    (``n.append(...)``), passing it as a bare argument (the callee may
    mutate), subscript/attribute stores or aug-assigns rooted at it, and
    ``del``. Iteration, subscript reads, and arithmetic stay safe. Shared by
    the closure-capture and module-global folds.

    Two knobs, and both exist to move a *suspicion* out of the HARD set and
    into the provisional one, where it is folded and then confirmed at
    runtime by ``PurityChecks.learn_mutating_captures``.

    ``bare_args=False`` drops the "passed as an argument" rule. That refusal
    was over-broad and chose the worse failure: `sum(G)`, `len(G)`,
    `helper(G)` and `model.predict(G)` all put `G` beyond it, so a later
    `G = ...` never reached the key and the function served a stale value
    for ever, silently.

    ``mutating_methods_only=True`` narrows the method-call rule to methods
    that actually write -- ``append``, ``update``, ``sort`` and their
    relatives, the same table the purity analyzer uses. "Any method, since
    we cannot prove purity" made the same over-broad choice one level down,
    and it cost: a lookup table read as
    ``ALIASES.get(v, v)`` never reached the key, so editing the table
    published stale labels with nothing to see. `ALIASES[v]`, `v in
    ALIASES`, `d = ALIASES; d.get(v)` and a bare read all tracked
    correctly, which is what made it so hard to believe.

    ``waived`` skips uses on a ``# @cash:assume-safe`` line or inside a
    ``with cash.assume_safe():`` block: the effect there was audited as one
    a hit may lose (see `waived_use_filter`).
    """
    unsafe: set[str] = set()
    write_methods: frozenset[str] = frozenset()
    if mutating_methods_only:
        write_methods = REPORTED_METHODS
    for node in ast.walk(tree):
        if waived is not None and isinstance(node, (ast.Call, ast.stmt)) and waived(node):
            continue
        if isinstance(node, ast.Call):
            f = node.func
            # The receiver's root: `cfg.calls.append(x)` writes to what `cfg`
            # holds as surely as `cfg.append(x)` does.
            receiver = f.value if isinstance(f, ast.Attribute) else None
            while isinstance(receiver, (ast.Attribute, ast.Subscript)):
                receiver = receiver.value
            if (
                isinstance(receiver, ast.Name)
                and receiver.id in names
                and (not mutating_methods_only or f.attr in write_methods)
            ):
                unsafe.add(receiver.id)
            if bare_args:
                for a in list(node.args) + [kw.value for kw in node.keywords]:
                    if isinstance(a, ast.Starred):
                        a = a.value
                    if isinstance(a, ast.Name) and a.id in names:
                        unsafe.add(a.id)
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.Delete)):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AugAssign):
                targets = [node.target]
            else:
                targets = node.targets
            for t in targets:
                root = t
                while isinstance(root, (ast.Subscript, ast.Attribute, ast.Starred)):
                    root = root.value
                if isinstance(root, ast.Name) and root.id in names:
                    unsafe.add(root.id)
    return frozenset(unsafe)


def waived_use_filter(func: Callable, tree: ast.AST) -> Callable[[ast.AST], bool] | None:
    """A predicate: is a node of *tree*, *func*'s dedented source, on a waived line?

    Line numbers in that tree count from the ``def``. A comment waiver is
    read from the file, a ``with cash.assume_safe():`` block from *tree*.
    ``None`` when neither can waive anything.
    """
    blocks = assume_safe_block_lines(tree, func)
    try:
        filename = inspect.getsourcefile(func) or inspect.getfile(func)
        first = getsourcelines(func)[1]
    except SOURCE_RETRIEVAL_ERRORS:
        filename, first = None, 1
    if not filename and not blocks:
        return None

    offset = max(first, 1) - 1

    def waived(node: ast.AST) -> bool:
        start = getattr(node, "lineno", None)
        if start is None:
            return False
        end = getattr(node, "end_lineno", None) or start
        return any(n in blocks or (filename and line_waived(filename, offset + n)) for n in range(start, end + 1))

    return waived


#: `defaults_of`'s *leave_out* for a ``key=`` function: what it returns
#: stands for every argument, defaults included.
EVERY_PARAMETER: frozenset[str] = frozenset({"*"})


def _left_out(name: str, leave_out: frozenset[str]) -> bool:
    return leave_out is EVERY_PARAMETER or name in leave_out


def defaults_of(func: Callable, leave_out: frozenset[str] = frozenset()) -> tuple[tuple, dict]:
    """The parameter defaults that decide what *func* computes.

    ``__defaults__`` (positional/keyword params) and ``__kwdefaults__``
    (keyword-only params) are separate containers; both are collected.

    Wrapped callees are walked too. ``func.__defaults__`` is what the call
    literally binds, but when *func* is a ``functools.wraps`` wrapper its own
    defaults are typically empty (a ``*args, **kwargs`` passthrough) while the
    values that actually decide the result sit on ``__wrapped__`` — which is
    also what ``inspect.signature`` reports and therefore what
    ``ArgHasher.normalize_call_args`` binds. Folding every level is the conservative
    choice: folding a default that turns out not to bind costs at most a
    one-time miss, whereas missing one that does bind is a silent wrong
    answer.

    The defaults of the parameters in *leave_out* are not collected: a
    parameter left out of the key (``ignore=``, ``cash.Ignore``, or every
    one under ``key=``, whose return value stands for them) is left out
    with its default, which may be a connection or a lock that cannot be
    hashed.
    """
    pos: list[Any] = []
    kwd: dict[str, Any] = {}
    seen: set[int] = set()
    fn: Any = func
    depth = 0
    while fn is not None and id(fn) not in seen:  # every layer; a cycle ends
        seen.add(id(fn))
        level_pos = getattr(fn, "__defaults__", None) or ()
        code = plain_getattr(fn, "__code__", None)
        if leave_out and level_pos and code is not None:
            named = code.co_varnames[: code.co_argcount][-len(level_pos) :]
            level_pos = tuple(v for name, v in zip(named, level_pos) if not _left_out(name, leave_out))
        pos.extend(level_pos)
        # Qualify by depth so a wrapper and its wrappee can't collide on a
        # shared kwonly name; sort so dict order never leaks into the key.
        level_kwd = getattr(fn, "__kwdefaults__", None) or {}
        for name in sorted(level_kwd):
            if not _left_out(name, leave_out):
                kwd[f"{depth}:{name}"] = level_kwd[name]
        fn = plain_getattr(fn, "__wrapped__", None)
        depth += 1
    return tuple(pos), kwd


#: `ArgHasher` -> the `HelperIdentity` built over it: `capture_digest` is
#: asked with the hasher alone (the drift guard hashes a capture again after
#: the body ran) and must answer as the fold did.
_HELPERS_BY_HASHER: weakref.WeakKeyDictionary = _weak.WeakKeyDictionary()


def capture_digest(args: ArgHasher, value: Any) -> str:
    """The digest a captured data value is keyed by, and checked against
    after the call. What it holds that pickles by name -- a class in a
    registry ``{"d": Double}``, a function in a list of steps -- is keyed by
    its code and what it was built with (`HelperIdentity.value_digest`). Any
    other object is hashed as it is, by its ``__cash_key__`` or registered
    hasher if it has one. Raises what hashing raises."""
    helpers = _HELPERS_BY_HASHER.get(args)
    if helpers is not None:
        return helpers.value_digest(value)
    if type(value) in (dict, list, tuple) and plain_data_kind(value) is None:
        value = stabilize_for_global_hash(value, hash_callable_source)
    return args.hash_payload((value,), {})


def _owner_is_output(method: Any, owner: Any) -> bool:
    """Is what *owner* holds no input of a call of its bound *method*: a C
    method that writes to it (``results.append``), a logger's logging
    method, or a method of a random generator, whose stream the
    random-number checks own?"""
    name = plain_getattr(method, "__name__", "")
    if type(method) is types.MethodType:
        import logging

        if name in _LOG_METHODS and isinstance(owner, (logging.Logger, logging.LoggerAdapter)):
            return True
    elif name in REPORTED_METHODS:
        return True
    return rng_carrier_kind(owner) is not None


_LOG_METHODS = frozenset({"debug", "info", "warning", "warn", "error", "exception", "critical", "log"})


def carrier_label(value: Any) -> str:
    """A name for the callable *value* in a message: its qualified name, or
    its class's."""
    name = plain_getattr(value, "__qualname__", None)
    if isinstance(name, str) and name:
        return name
    inner = plain_getattr(value, "func", None)
    inner_name = plain_getattr(inner, "__qualname__", None)
    if isinstance(inner_name, str) and inner_name:
        return f"a {class_qualname(type(value))} of {inner_name}"
    return f"a {class_qualname(type(value))}"


def module_ref_of(owner: Any) -> str:
    """The name a class or module a method is bound to is keyed by."""
    if issubclass(type(owner), types.ModuleType):
        return f"module:{module_ref(owner)}"
    return code_ref(owner) or class_qualname(owner)


def iter_code_scopes(code: types.CodeType) -> Iterator[types.CodeType]:
    """Yield *code* and every code object nested inside it, recursively.

    A generator expression, comprehension, or ``lambda`` compiles to its
    OWN code object hung off the enclosing ``co_consts``, so anything it
    references is invisible in the outer ``co_names`` / instruction stream.
    Walking the const tree is what `walk_nested_code` does, for exactly
    this reason.

    Comprehensions nest, so this recurses. Note that CPython 3.12+ inlines
    list/set/dict comprehensions into the enclosing scope (PEP 709) — those
    already land in the outer ``co_names``; generator expressions and
    lambdas still get their own scope on every version.
    """
    yield code
    for const in code.co_consts or ():
        if isinstance(const, types.CodeType):
            yield from iter_code_scopes(const)


def module_capture_identity(module: types.ModuleType, attrs: frozenset[str]) -> tuple:
    """What a captured *module* stands for in a closure's key.

    Its name, which is the same in every process (never its ``id``), and the
    code of each function or class in *attrs* the module holds, so editing
    ``cap.helper`` moves the key while editing something the closure never
    reads does not. The data among *attrs* is folded by the local-binding
    channel of the globals fold, and code outside the user's own files is
    left out, as a module global's is.
    """
    namespace = plain_getattr(module, "__dict__", {})
    code = []
    for attr in sorted(attrs):
        value = namespace.get(attr)
        if issubclass(type(value), (types.FunctionType, type)) and is_user_code_object(value):
            code.append((attr, hash_callable_source(value)))
    return ("module", module_ref(module), tuple(code))


_ATTR_WRITES = frozenset({"STORE_ATTR", "DELETE_ATTR"})


class CaptureAnalysis:
    """Which of a closure's captured variables the function body may change,
    read from its code and source once per code object (closures from one
    factory share it)."""

    def __init__(self) -> None:
        # code object -> frozenset of reassigned freevars
        self._deref_writes: LruMemo[Any, frozenset] = LruMemo(CODE_OBJECTS)
        # code object -> (free vars with capture-unsafe uses, free vars folded
        # only provisionally: passed to a call, so folded and then confirmed
        # by observation). One entry, so the two never disagree.
        self._use_cache: LruMemo[Any, tuple[frozenset, frozenset]] = LruMemo(CODE_OBJECTS)
        # code object -> {free var: attributes read from it}; see `attr_reads`.
        self._attr_reads: LruMemo[Any, dict[str, frozenset[str]]] = LruMemo(CODE_OBJECTS)
        # code object -> the attribute names it assigns; see `assigned_attributes`.
        self._attr_writes: LruMemo[Any, frozenset[str]] = LruMemo(CODE_OBJECTS)

    def written_freevars(self, code: Any) -> frozenset:
        """Free-variable names the function reassigns (``STORE_DEREF`` /
        ``DELETE_DEREF``) - i.e. ``nonlocal`` counters that drift between calls.
        Cached per code object (closures share a code object per factory)."""
        cache = self._deref_writes
        hit = cache.get(code)
        if hit is not None:
            return hit

        written = frozenset(argvals(code, DEREF_WRITES))
        cache[code] = written
        return written

    def assigned_attributes(self, code: Any) -> frozenset[str]:
        """The attribute names *code*, or code nested in it, assigns or
        deletes on any object (``wrapper.calls += 1``, ``self.n = 0``).
        Cached per code object."""
        cached = self._attr_writes.get(code)
        if cached is not None:
            return cached
        found: set[str] = set()
        for scope in iter_code_scopes(code):
            found.update(n for n in argvals(scope, _ATTR_WRITES) if isinstance(n, str))
        result = frozenset(found)
        self._attr_writes[code] = result
        return result

    def attr_reads(self, code: Any) -> dict[str, frozenset[str]]:
        """The attributes the function reads from each captured variable.

        ``cap.helper(x)`` compiles to ``LOAD_DEREF cap`` followed by
        ``LOAD_ATTR helper`` (``LOAD_METHOD`` before 3.12), in the function
        or in a genexp or lambda nested in it. ``getattr(cap, "helper")``
        names the attribute as a string constant instead, so every
        identifier-shaped constant of a scope that loads the variable counts
        too, as the ``module.ATTR`` channel of the globals fold does. Cached
        per code object (closures from one factory share it).
        """
        cached = self._attr_reads.get(code)
        if cached is not None:
            return cached
        reads: dict[str, set[str]] = {}
        for scope in iter_code_scopes(code):
            instrs = instructions(scope)
            loaded = {argval for opname, argval in instrs if opname == "LOAD_DEREF" and isinstance(argval, str)}
            if not loaded:
                continue
            consts = [c for c in scope.co_consts or () if isinstance(c, str) and c.isidentifier()]
            for name in loaded:
                reads.setdefault(name, set()).update(consts)
            for (prev, prev_val), (nxt, nxt_val) in zip(instrs, instrs[1:]):
                if prev == "LOAD_DEREF" and nxt in ("LOAD_ATTR", "LOAD_METHOD"):
                    if isinstance(nxt_val, str) and not nxt_val.startswith("__"):
                        reads.setdefault(prev_val, set()).add(nxt_val)
        result = {name: frozenset(attrs) for name, attrs in reads.items()}
        self._attr_reads[code] = result
        return result

    def unsafe_uses(self, func: Callable) -> frozenset:
        """Free-variable names whose captured object *may be mutated* by the
        function body.

        A capture is only content-foldable into the cache key when the body
        provably just READS it. Disqualifying uses of a free variable ``n``:
        method calls on it (``n.append(...)`` — any method, since we can't
        prove purity), passing it as a bare argument (the callee may mutate),
        subscript/attribute stores or aug-assigns rooted at it, and ``del``.
        Iteration, subscript reads, and arithmetic stay safe.

        Cached per code object (closures from one factory share it). When the
        source is unavailable, every free var is reported unsafe.
        """
        code = plain_getattr(func, "__code__", None)
        if code is None:
            return frozenset()
        cached = self._use_cache.get(code)
        if cached is not None:
            return cached[0]
        freevars = set(code.co_freevars or ())
        provisional: frozenset = frozenset()
        try:
            tree = ast.parse(textwrap.dedent(getsource(func)))
        except SOURCE_RETRIEVAL_ERRORS:
            # No source: the conservative answer. Nothing can be told
            # apart, so nothing is folded and nothing is provisional.
            result = frozenset(freevars)
        else:
            # Only mutations visible in this function's own source disqualify a
            # capture outright. "Passed to a call" is provisional: folded, then
            # confirmed by observation, so `sum(data)` does not put `data`
            # beyond the fold, where the closure would serve stale forever.
            result = unsafe_uses_of(
                tree,
                freevars,
                bare_args=False,
                mutating_methods_only=True,
            )
            suspected = unsafe_uses_of(tree, freevars) - result
            provisional = unsafe_uses_of(tree, suspected, waived=waived_use_filter(func, tree))
            # Only on waived lines: as for globals (`GlobalReads.read_global_data_names`).
            result = result | (suspected - provisional)
        self._use_cache[code] = (result, provisional)
        return result

    def provisional(self, code: Any) -> frozenset | None:
        """The free vars of *code* folded only provisionally, or None if unknown."""
        cached = self._use_cache.get(code)
        return cached[1] if cached is not None else None


_UNHASHABLE_CAPTURE_ERRORS = (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError)


def unhashable_capture(fn: Any, name: str, value: Any, error: Exception) -> KeyBuildFailed:
    """KEY-UNHASHABLE-CAPTURE for *fn*'s captured *name*, whose *value* the
    key needs and cannot hash. Left out, a change to it would not reach
    the key; the call runs uncached instead."""
    owner = plain_getattr(fn, "__qualname__", repr(fn))
    kind = class_qualname(type(value))
    return KeyBuildFailed(
        "KEY-UNHASHABLE-CAPTURE",
        f"@cash.cache: {owner} reads the captured variable '{name}', a {kind} that could "
        f"not be hashed ({class_name(type(error))}), so the call ran uncached rather than "
        f"risk serving a result computed under a value that changed.",
        f"pass what {owner} needs from '{name}' as an argument, capture only the plain "
        f"values it reads, or register a hasher with cash.register_hasher({kind}, ...).",
    )


# Cells the compiler adds for a class body, never ones the user wrote: on
# Python 3.14 an annotated class's ``__annotate__`` closes over
# ``__classdict__``, the class's own namespace, whose methods are keyed as
# helpers already. Hashing it raised KEY-UNHASHABLE-CAPTURE (it holds
# descriptors), so every function reaching an annotated class -- any
# dataclass -- ran uncached.
CLASS_BODY_CELLS = frozenset({"__class__", "__classdict__", "__classdictcell__"})

#: Stands for a cell that holds nothing yet, in `HelperIdentity._read_cells`.
_EMPTY_CELL = object()

#: How many containers `_snapshot_of` follows inside one value before it
#: gives up (the value is then hashed on every call instead).
_SNAPSHOT_CONTAINERS = 10_000


def _snapshot_of(value: Any, functions_held: bool = False) -> tuple | None:
    """What could change in *value* without changing its identity, as object
    identities; None when that cannot be watched this way.

    A list or tuple of plain data is `_plain_data.identity_snapshot` (one
    pass of ``is`` at C speed, however long). A dict or a set is watched by
    the identities of its keys and values, and each value that can change
    is watched in turn. Anything else -- an instance, a ``bytearray``, an
    array -- gives None, and so does a function that carries state of its
    own (captured values, attributes, a default that can change), keyed
    with that state (`HelperIdentity.callable_state`); *functions_held*
    says the caller checks the functions the dict holds itself.
    """
    budget = [_SNAPSHOT_CONTAINERS]

    def snap(v: Any) -> tuple | None:
        budget[0] -= 1
        if budget[0] < 0:
            return None
        t = type(v)
        if t is list or t is tuple:
            levels = identity_snapshot(v)
            return None if levels is None else ("seq", v, levels)
        if t is dict:
            keys = list(v)
            values = list(v.values())
            inner = []
            for item in values:
                if type(item) is types.FunctionType:
                    if functions_held or _stateless_function(item):
                        continue
                    return None
                if _cannot_change(item):
                    continue
                sub = snap(item)
                if sub is None:
                    return None
                inner.append(sub)
            return ("dict", v, keys, values, inner)
        if t is set or t is frozenset:
            items = list(v)
            if not all(map(_cannot_change, items)):
                return None
            if not all(_stateless_function(item) for item in items if type(item) is types.FunctionType):
                return None
            return ("set", v, items)
        return None

    found = snap(value)
    if found is None:
        # A dispatch table or a step list of plain functions: what it holds
        # is keyed by their code, which `CodeTable` watches by identity.
        from .code_tables import CodeTable

        table = CodeTable.of(value)
        if table is not None:
            return ("table", value, table)
    return found


def _stateless_function(fn: types.FunctionType) -> bool:
    """Is the function *fn* its code alone: no captured value, no attribute
    of its own, no default that can change in place?"""
    if fn.__closure__ is not None or fn.__dict__:
        return False
    return all(is_immutable_capture(d) for d in (*(fn.__defaults__ or ()), *(fn.__kwdefaults__ or {}).values()))


def _snapshot_moved(snap: tuple) -> bool:
    """Has the value `_snapshot_of` took *snap* of changed since?"""
    kind, value = snap[0], snap[1]
    if kind == "table":
        return not snap[2].holds(value)
    if kind == "seq":
        return identity_changed(value, snap[2])
    if kind == "dict":
        keys, values = snap[2], snap[3]
        if len(value) != len(keys):
            return True
        if not all(map(operator.is_, value, keys)) or not all(map(operator.is_, value.values(), values)):
            return True
        return any(map(_snapshot_moved, snap[4]))
    items = snap[2]
    return len(value) != len(items) or not all(map(operator.is_, value, items))


def _cannot_change(value: Any) -> bool:
    """Is *value* keyed the same for as long as it is the same object? An
    immutable value, or one keyed by its code or name (a function, a class,
    a module), or one never keyed (a lock)."""
    return (
        is_immutable_capture(value)
        or isinstance(value, IMMUTABLE_VALUE_TYPES)
        or isinstance(value, SYNC_TYPES)
        or issubclass(type(value), (types.FunctionType, types.BuiltinFunctionType, types.ModuleType, type))
    )


class HelperIdentity:
    """What a callable is for the key: its code and everything it was built
    with (`callable_state`), wherever the callable is held.

    One answer for a helper the cached function calls, a function passed as
    an argument, a callable the cached function or a helper captured, one a
    data global holds and a parameter default: each of those folds asks
    here. A function is its code, its parameter defaults, the values its
    closure captured and the attributes stored on it (`identity`); a bound
    method adds the instance it is bound to, a ``functools.partial`` what it
    binds, a wrapper (``functools.wraps``, ``functools.lru_cache``) what it
    wraps; and every callable met on the way -- captured, bound, held in a
    captured list or dict, stored as an attribute -- is keyed the same way,
    however deep, a callable that holds itself ending at ``cycle``. A value
    that cannot be hashed raises `KeyBuildFailed`, so the call runs uncached
    with a warning; nothing is left out of the key for it.
    """

    def __init__(self, args: ArgHasher, captures: CaptureAnalysis, mutations: LearnedMutations) -> None:
        self._args = args
        self._captures = captures
        self._mutations = mutations
        _HELPERS_BY_HASHER[args] = self
        # id(helper) -> (helper, __defaults__, __kwdefaults__, what its read
        # cells held, identity snapshots of the mutable values it keyed,
        # identity, the drift guard's entries, the code's digest, the
        # functions it holds with their own identities); see `identity`.
        # Holding the helper keeps its id from being recycled while the
        # entry lives. A helper that keyed a value no snapshot can watch (an
        # instance, a bound method) has no entry and is hashed per call.
        self._defaults_memo: LruMemo[int, tuple[Any, Any, Any, tuple, list, str, dict, str, list]] = LruMemo(
            CODE_OBJECTS
        )
        #: Counts every ``cycle`` a walk ended at: a digest built around one
        #: depends on where the walk started, so it is never remembered.
        self._cycles = 0

    def _read_cells(self, fn: Callable) -> tuple:
        """What each closure cell *fn* reads (not writes) holds now, in order:
        a ``nonlocal`` setter in a sibling closure re-sets one without
        touching *fn* itself."""
        closure = getattr(fn, "__closure__", None)
        code = plain_getattr(fn, "__code__", None)
        if not closure or code is None:
            return ()
        written = self._captures.written_freevars(code)
        held = []
        for name, cell in zip(code.co_freevars, closure):
            if name in written or name in CLASS_BODY_CELLS:
                continue
            try:
                held.append(cell.cell_contents)
            except ValueError:
                held.append(_EMPTY_CELL)
        return tuple(held)

    # -- any callable ------------------------------------------------------

    def callable_state(self, value: Any, seen: frozenset[int] = frozenset()) -> str:
        """The key part of the callable *value*: its code and what it was
        built with (see the class). *seen* holds the callables this walk is
        inside of.

        * a function: `identity` (code, defaults, captures, attributes);
        * a bound method: its function, and the instance it is bound to by
          value (a class or a module by name);
        * a ``functools.partial``: the function it calls and what it binds;
        * a callable object of the user's class: its class's code and what
          the object holds;
        * a wrapper with ``__wrapped__`` (``functools.lru_cache``): its
          name and what it wraps, never the results it remembers;
        * a cached function: its whole state, as a call of it is keyed;
        * a class: its code, its bases' and its metaclass's;
        * anything else (a builtin, a library's callable object): its name,
          and for a C callable built from arguments
          (``operator.itemgetter("n")``) those arguments.

        A method that writes to its instance by design (``results.append``,
        ``logger.info``) and a method of a random generator are their name
        alone: what the instance holds is the call's output, or a stream
        the random-number checks own.

        Raises `KeyBuildFailed` when a value it must key cannot be hashed.
        """
        if id(value) in seen:
            self._cycles += 1
            return "cycle"
        if is_mock(value):
            return f"mock:{class_qualname(type(value))}"
        inner = seen | {id(value)}
        if is_cash_wrapper(value):
            state = getattr(value, "_cash_state", None)
            if state is not None:
                return "cached:" + state()
            wrapped = plain_getattr(value, "__wrapped__", None)
            if wrapped is not None and wrapped is not value:
                return self.callable_state(wrapped, inner)
            return hash_callable_source(value)
        if isinstance(value, type):
            return self.identity(value)
        compiled = jit_python_function(value)
        if compiled is not None:
            return jit_identity(self.callable_state(compiled, inner), value)
        t = type(value)
        if t is types.FunctionType:
            return self._function_state(value, seen)
        if t is types.MethodType:
            func_state = self.callable_state(value.__func__, inner)
            owner = value.__self__
            if issubclass(type(owner), (type, types.ModuleType)):
                return f"method:{func_state}:of:{code_ref(owner) or module_ref_of(owner)}"
            if _owner_is_output(value, owner):
                return f"method:{func_state}"
            return f"method:{func_state}:self:{self._bound_digest(value, owner, inner)}"
        if isinstance(value, functools.partial):
            func_state = self.callable_state(value.func, inner)
            bound: Any = (value.args, dict(value.keywords))
            attrs = plain_getattr(value, "__dict__", None)
            if attrs:
                bound = (*bound, dict(attrs))
            return f"partial:{func_state}:{self._bound_digest(value, bound, inner)}"
        if t is types.BuiltinFunctionType or t is types.MethodWrapperType:
            ref = code_ref(value) or hash_callable_source(value)
            owner = plain_getattr(value, "__self__", None)
            if owner is None or issubclass(type(owner), (type, types.ModuleType)) or _owner_is_output(value, owner):
                return ref
            return f"{ref}:self:{self._bound_digest(value, owner, inner)}"
        if is_user_callable_instance(value):
            return f"instance:{hash_callable_source(value)}:{self._bound_digest(value, value, seen)}"
        ref = code_ref(value) or hash_callable_source(value)
        wrapped = plain_getattr(value, "__wrapped__", None)
        if wrapped is not None and callable(wrapped):
            return f"{ref}:wraps:{self.callable_state(wrapped, inner)}"
        carried = carried_payload(value)
        if carried is None and not plain_getattr(value, "__dict__", None):
            reduced = reduced_state(value)
            carried = ("reduce", reduced) if reduced is not None else None
        if carried is None:
            return ref
        return f"{ref}:carries:{self._bound_digest(value, carried, inner)}"

    def value_digest(self, value: Any, seen: frozenset[int] = frozenset()) -> str:
        """The digest of a data value a callable holds -- captured, bound,
        a default, an attribute: hashed as an argument is, with every
        callable in it, however deep, by `callable_state`. A value that
        holds no code is hashed as it is. Raises what hashing raises."""
        if callable(value) and not isinstance(value, type):
            return "callable:" + self.callable_state(value, seen)
        if type(value) in (dict, list, tuple, set, frozenset) and plain_data_kind(value) is not None:
            return self._args.hash_payload((value,), {})
        if self._args.keys_by_registration(value):
            # The user said what identifies it.
            return self._args.hash_payload((value,), {})
        rewritten, holds_code = rewrite_held_code(value, lambda held: self._named_state(held, seen), carried=False)
        return self._args.hash_payload((rewritten if holds_code else value,), {})

    def _named_state(self, value: Any, seen: frozenset[int]) -> str:
        """A callable inside a held container, by its name and its
        `callable_state`: a table whose entry is replaced by another
        function running the same code is another table."""
        return self.held_callable_part(value, seen) or self.callable_state(value, seen)

    def _bound_digest(self, holder: Any, value: Any, seen: frozenset[int]) -> str:
        """`value_digest` of what the callable *holder* is bound to or
        carries; KEY-UNHASHABLE-CAPTURE when it cannot be hashed."""
        try:
            if value is holder:
                # A callable object: what it holds, not itself again.
                if self._args.keys_by_registration(value):
                    return self._args.hash_payload((value,), {})
                state = carried_payload(value)
                rewritten, holds_code = rewrite_held_code(
                    state, lambda held: self._named_state(held, seen | {id(value)}), carried=False
                )
                return self._args.hash_payload((rewritten,) if holds_code else (value,), {})
            return self.value_digest(value, seen)
        except _UNHASHABLE_CAPTURE_ERRORS as e:
            name = carrier_label(holder)
            kind = class_qualname(type(value if not isinstance(value, tuple) else holder))
            raise KeyBuildFailed(
                "KEY-UNHASHABLE-CAPTURE",
                f"@cash.cache: the call reaches {name}, which carries a {kind} that could not be "
                f"hashed ({class_name(type(e))}), so the call ran uncached rather than risk serving "
                f"a result computed with another one.",
                f"pass what {name} needs as plain arguments of the cached function, or register a "
                f"hasher with cash.register_hasher({kind}, ...).",
            ) from e

    # -- a function --------------------------------------------------------

    def _capture_part(self, fn: Callable) -> str:
        return self._captures_of(fn, {}, [], frozenset())[0]

    def _captures_of(self, fn: Callable, watch: dict, held: list, seen: frozenset[int]) -> tuple[str, list]:
        """Digest of the values a function's closure captured and only reads, or "".

        A decorator's arguments live there: ``@scale(10)`` builds a wrapper
        whose closure holds ``k=10``, so ``@scale(100)`` -- or ``@scale(K)``
        after ``K`` changed -- runs different code under an identical source.
        So does a factory's config: ``make_scorer(cfg)``
        with ``cfg`` a dataclass, an ``argparse.Namespace`` or any instance.
        Never a variable the function reassigns (``nonlocal calls; calls +=
        1``), nor a mutable value it may write to: decorators often keep
        caches, counters and registries in their closures, and folding state
        that drifts on every call would make every call miss. A value that
        cannot be hashed raises KEY-UNHASHABLE-CAPTURE
        (`unhashable_capture`), rather than being left out of the key.

        A captured callable is keyed by `callable_state`: the function a
        two-level factory's ``pipeline`` calls by what THAT was built with,
        a captured ``model.predict`` by the model, a captured partial by
        what it binds. Its code is the helper walk's to follow as well;
        what it names is keyed here (``make(np.float32)`` and
        ``make(np.float64)``, ``make(sum)`` and ``make(max)``). A captured
        list or dict holding callables keys each of them the same way.

        A value the function calls a method on or passes to a call
        (``weights.sum()``, ``model.predict(x)``, ``np.sum(a)``) is keyed as
        the cached function's own capture is: provisionally, with its digest
        put in *watch* for the drift guard
        (`PurityChecks.learn_mutating_captures`), which stops keying it once
        a call is seen to change it.

        Returns the digest and the values it keyed that can change in place
        (a config dict the function only reads), which `identity` watches;
        the functions among the captures are added to *held*.
        """
        closure = getattr(fn, "__closure__", None)
        code = plain_getattr(fn, "__code__", None)
        if not closure or code is None:
            return "", []
        inner = seen | {id(fn)}
        mutable: list = []
        written = self._captures.written_freevars(code)
        learned = self._mutations.of(code, "closure")
        unsafe: frozenset | None = None
        captures = []
        for name, cell in zip(code.co_freevars, closure):
            if name in written or name in CLASS_BODY_CELLS or name in learned:
                continue
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if isinstance(value, SYNC_TYPES):
                continue
            if issubclass(type(value), types.ModuleType):
                # `make(numpy.fft)` and `make(scipy.fft)`: which module.
                captures.append((name, f"module:{module_ref(value)}"))
                continue
            if isinstance(value, type):
                # A class: its code is the helper walk's; which class, here.
                ref = code_ref(value)
                if ref is not None:
                    captures.append((name, f"ref:{ref}"))
                continue
            if callable(value):
                part = self._held_callable_part(value, inner, held, mutable)
                if part is not None:
                    captures.append((name, part))
                continue
            watched = False
            if not (is_immutable_capture(value) or isinstance(value, IMMUTABLE_VALUE_TYPES)):
                # A value the helper only READS is data like any other:
                # `lambda: when` with `when` a list, a dict or a datetime (a
                # frozen-clock fixture), `x * cfg.weight` with `cfg` a frozen
                # dataclass. What the body mutates (a decorator's cache
                # dict, a counter list) stays out: folding it would make
                # every call miss. A value it calls a method on or passes on
                # (`weights.sum()`, `model.predict(x)`) is keyed and watched,
                # as the cached function's own capture is: left out, a
                # factory helper over a new array or model served the old
                # result, in this process and the next.
                if unsafe is None:
                    unsafe = self._captures.unsafe_uses(fn)
                if name in unsafe:
                    continue
                provisional = self._captures.provisional(code)
                watched = provisional is None or name in provisional
            try:
                digest = capture_digest(self._args, value)
            except _UNHASHABLE_CAPTURE_ERRORS as e:
                raise unhashable_capture(fn, name, value, e) from e
            captures.append((name, digest))
            if watched:
                watch[f"{name}@helper:{id(fn)}"] = (digest, "helper", (fn, name), fn)
            if not _cannot_change(value):
                # A config dict the helper only reads: `weights["a"] = 3`
                # changes it under the same object.
                mutable.append(value)
        if not captures:
            return "", mutable
        return hashlib.sha256(repr(captures).encode("utf-8")).hexdigest(), mutable

    def held_callable_part(self, value: Any, seen: frozenset[int]) -> str | None:
        """The key part of a callable *value* the cached function itself
        captured (`ClosureFold.fold_closure`), as one a helper captured."""
        return self._held_callable_part(value, seen, [], [])

    def _held_callable_part(self, value: Any, seen: frozenset[int], held: list, mutable: list) -> str | None:
        """The key part of a callable *value* a function captured, has as a
        default or stores as an attribute: its name and its
        `callable_state`. A function among them joins *held* (`identity`
        checks its own identity again before serving a remembered one);
        any other callable that carries state joins *mutable*, which keeps
        the function's identity from being remembered at all."""
        if is_mock(value):
            return None
        if is_cash_wrapper(value):
            # What it computes can change with any global it reads.
            mutable.append(value)
            return self.callable_state(value, seen)
        t = type(value)
        if t is types.FunctionType:
            ref = code_ref(value)
            if not is_user_code_object(value):
                # A library's function: its name; its version is
                # `library_versions`'.
                return f"ref:{ref}"
            held.append(value)
            return f"ref:{ref}:{code_fingerprint(value.__code__)}:{self.callable_state(value, seen)}"
        if t is types.BuiltinFunctionType and issubclass(
            type(plain_getattr(value, "__self__", None)), (type(None), types.ModuleType)
        ):
            return f"ref:{code_ref(value)}"  # `sum`, `math.floor`
        ref = code_ref(value)
        state = self.callable_state(value, seen)
        if state != ref:
            mutable.append(value)
        return f"ref:{ref}:{state}" if state != ref else f"ref:{ref}"

    def _attribute_part(self, fn: Callable, held: list, mutable: list, seen: frozenset[int]) -> str:
        """Digest of the attributes stored on the function *fn*, or "".

        ``def scale(x): return x * scale.k`` with ``scale.k = k`` set by the
        factory reads its factor from its own ``__dict__``: two functions
        one factory made share their code and differ only there, and
        ``fn.k = 5`` changes what the next call computes. Every attribute
        counts but: a dunder (``__wrapped__`` is followed by
        `callable_state`), a lock, one the function's own code assigns
        (``wrapper.calls += 1``), and one holding the very object of a
        captured variable the function changes (``wrapper.cache = cache``):
        state that drifts with every call, as in `_captures_of`. One that
        cannot be hashed raises KEY-UNHASHABLE-CAPTURE.
        """
        attrs = plain_getattr(fn, "__dict__", None)
        if not attrs or type(attrs) is not dict:
            return ""
        inner = seen | {id(fn)}
        code = plain_getattr(fn, "__code__", None)
        assigned = self._captures.assigned_attributes(code) if code is not None else frozenset()
        changed = self._changed_captures(fn) if code is not None else ()
        parts = []
        for name in sorted(attrs, key=str):
            value = attrs[name]
            if not isinstance(name, str):
                continue
            if name == "__wrapped__":
                if callable(value) and value is not fn:
                    part = self._held_callable_part(value, inner, held, mutable)
                    if part is not None:
                        parts.append((name, part))
                continue
            if name.startswith("__") or name in assigned or isinstance(value, SYNC_TYPES):
                continue
            if any(value is drifting for drifting in changed):
                continue
            if isinstance(value, type):
                ref = code_ref(value)
                if ref is not None:
                    parts.append((name, f"ref:{ref}"))
                continue
            if callable(value):
                part = self._held_callable_part(value, inner, held, mutable)
                if part is not None:
                    parts.append((name, part))
                continue
            if type(value) is types.MappingProxyType:
                value = dict(value)
            try:
                digest = self.value_digest(value, inner)
            except _UNHASHABLE_CAPTURE_ERRORS as e:
                owner = plain_getattr(fn, "__qualname__", repr(fn))
                kind = class_qualname(type(value))
                raise KeyBuildFailed(
                    "KEY-UNHASHABLE-CAPTURE",
                    f"@cash.cache: the call reaches {owner}, whose attribute '{name}' holds a {kind} "
                    f"that could not be hashed ({class_name(type(e))}), so the call ran uncached rather "
                    f"than risk serving a result computed with another one.",
                    f"keep '{name}' off the function (a module global, an argument), or register a "
                    f"hasher with cash.register_hasher({kind}, ...).",
                ) from e
            parts.append((name, digest))
            if not _cannot_change(value):
                mutable.append(value)
        if not parts:
            return ""
        return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()

    def _changed_captures(self, fn: Callable) -> list:
        """The values of the captured variables *fn* may change: the ones
        `_captures_of` leaves out of the key."""
        closure = getattr(fn, "__closure__", None)
        code = fn.__code__
        if not closure:
            return []
        drifting = (
            self._captures.written_freevars(code) | self._mutations.of(code, "closure") | self._captures.unsafe_uses(fn)
        )
        found = []
        for name, cell in zip(code.co_freevars, closure):
            if name in drifting:
                try:
                    found.append(cell.cell_contents)
                except ValueError:
                    continue
        return found

    def defaults_digest(self, pos: tuple, kwd: dict, seen: frozenset[int] = frozenset()) -> str:
        """The digest of parameter defaults (`defaults_of`), by value as
        arguments are, a callable among them -- or inside a list or dict
        default -- by `callable_state`: ``def run(xs, fn=make(3))`` is told
        from ``fn=make(1)``, a ``functools.partial`` default by what it
        binds. Raises what hashing raises."""
        if not pos and not kwd:
            return self._args.hash_payload((), {})
        if all(plain_data_kind(v) is not None for v in (*pos, *kwd.values())):
            return self._args.hash_payload(pos, kwd)

        def keyed(value: Any) -> Any:
            if callable(value) and not isinstance(value, type):
                return f"__cash_callable__:{self.callable_state(value, seen)}"
            if self._args.keys_by_registration(value):
                return value
            rewritten, holds_code = rewrite_held_code(
                value, lambda held: self.callable_state(held, seen), carried=False
            )
            return rewritten if holds_code else value

        return self._args.hash_payload(tuple(keyed(v) for v in pos), {k: keyed(v) for k, v in kwd.items()})

    def identity(self, fn: Callable) -> str:
        """A function's identity for the key: its code, its parameter
        defaults, the values its closure captured and the attributes stored
        on it (a class: its code, its bases' and its metaclass's).

        A default is evaluated once, at ``def`` time, and lives on the function
        object -- so ``def shrink(v, alpha=ALPHA)`` reads the same after
        ``ALPHA`` changes, the source digest does not move, and global folding
        never sees the name (it is not read in the body): a service whose ridge
        penalty was a helper's default served 8 wrong answers in 8. The cached function's own defaults were already
        folded (``ClosureFold.fold_defaults``); now every followed helper's are, by value,
        through the same payload hasher and the same callable fallback.

        Not inside ``hash_callable_source``'s memo: that is keyed per CODE
        object, and two closures from one factory share a code object while
        holding different defaults.

        A JIT-compiled helper (numba's ``@njit``) is the identity of the
        function it compiles, defaults and captures included, with its
        compile options.
        """
        return self._function_state(fn, frozenset())

    def _function_state(self, fn: Callable, seen: frozenset[int]) -> str:
        """`identity`, inside a walk that is already inside *seen*. Worked
        out once per key build (`KEY_BUILD_DIGESTS`): a function another
        one holds is asked about by each holder and once more for itself,
        and each time hashed what it captured (an array of a million
        numbers, three times a call). Not one whose walk met a callable
        that holds itself: what stands for the loop depends on where the
        walk came from."""
        build = KEY_BUILD_DIGESTS.get()
        if build is None:
            return self._function_state_now(fn, seen)
        item = ("callable state", id(fn))
        pending = CAPTURE_WATCH.get()
        entry = build.get(item)
        if entry is not None and entry[0] is fn and entry[2] is pending:
            return entry[1]
        cycles = self._cycles
        state = self._function_state_now(fn, seen)
        if self._cycles == cycles:
            # *pending* is where the walk left what it watches for drift:
            # a walk for another call's watch list leaves it there itself.
            build[item] = (fn, state, pending)
        return state

    def _function_state_now(self, fn: Callable, seen: frozenset[int]) -> str:
        compiled = jit_python_function(fn)
        if compiled is not None:
            return jit_identity(self._function_state(compiled, seen), fn)
        source = hash_callable_source(fn)
        if isinstance(fn, type):
            # A class's own source says nothing about what it inherits, and
            # this channel is what the key folds: ``Worker(Base)`` calling an
            # inherited ``run`` kept serving the old answer after ``Base.run``
            # was rewritten -- 20 where an uncached run gives 500, in one file.
            # An OPAQUE base still contributes nothing, as for a class passed as an argument.
            bases = [
                hash_callable_source(base)
                for base in class_mro(fn)[1:]
                if base is not object and not is_opaque(base) and is_user_code_object(base)
            ]
            if bases:
                source = f"{source}:bases:{','.join(bases)}"
            # Its metaclass's code runs too: `Model()` calls the metaclass's
            # `__call__`, `Model.factor` can be a property on it.
            meta = type(fn)
            if meta is not type and not is_opaque(meta) and is_user_code_object(meta):
                source = f"{source}:metaclass:{self.identity(meta)}"
        # A class has no defaults and no closure. Not asked for them either: a
        # metaclass with a catch-all `__getattr__` answers both.
        is_class = isinstance(fn, type)
        defaults = None if is_class else getattr(fn, "__defaults__", None)
        kwdefaults = None if is_class else getattr(fn, "__kwdefaults__", None)
        memo_key = id(fn)
        code_source = source
        cells = () if is_class else self._read_cells(fn)
        # A function's own attributes. An instance's ``__dict__`` is its
        # state, a value: keyed where the instance is held (a global's
        # fold, `callable_state`), with what watches it for drift.
        attrs = plain_getattr(fn, "__dict__", None) if type(fn) is types.FunctionType else None
        cached = self._defaults_memo.get(memo_key)
        if (
            cached is not None
            and cached[0] is fn
            and cached[1] is defaults
            and cached[2] is kwdefaults
            and len(cached[3]) == len(cells)
            and all(map(operator.is_, cached[3], cells))
            and not any(map(_snapshot_moved, cached[4]))
            # The code itself: IPython's ``%autoreload`` swaps a function's
            # ``__code__`` in place, keeping the object, its defaults and its
            # cells, so only the source digest says the helper changed.
            and cached[7] == code_source
            # The functions it holds -- captured, a default, an attribute --
            # are part of its identity with all THEY hold: each still its own.
            and all(self._function_state(held_fn, frozenset()) == was for held_fn, was in cached[8])
        ):
            if cached[6]:
                pending = CAPTURE_WATCH.get()
                if pending is not None:
                    pending.update(cached[6])
            return cached[5]
        # After the memo: its entry already holds the captures, and hashing
        # a captured object on every call only to throw the digest away cost
        # as much as the object is large. What can change in place under the
        # same objects -- a dict the helper reads, a list default -- is
        # watched by identity snapshot (`_snapshot_of`); one that cannot be
        # watched leaves no entry, and is hashed on every call, as a data
        # global is.
        cycles = self._cycles
        watch: dict = {}
        held: list = []
        captured, mutable = ("", []) if is_class else self._captures_of(fn, watch, held, seen)
        if captured:
            source = f"{source}:captures:{captured}"
        stored = "" if is_class or not attrs else self._attribute_part(fn, held, mutable, seen)
        if stored:
            source = f"{source}:attributes:{stored}"
        pos, kwd = ((), {}) if is_class else defaults_of(fn)
        for value in (*pos, *kwd.values()):
            if type(value) is types.FunctionType and not is_cash_wrapper(value):
                if is_user_code_object(value):
                    held.append(value)
            elif not _cannot_change(value):
                mutable.append(value)
        snapshots: list | None = []
        if attrs and type(attrs) is dict:
            # The attributes: which objects, each function among them
            # checked as one it holds.
            snap = _snapshot_of(attrs, functions_held=True)
            if snap is None:
                snapshots = None
            else:
                snapshots.append(snap)
        for value in mutable:
            if snapshots is None:
                break
            snap = _snapshot_of(value)
            if snap is None:
                snapshots = None
                break
            snapshots.append(snap)
        try:
            digest = self.defaults_digest(pos, kwd, seen | {id(fn)})
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError) as e:
            bad_type = self._args.first_unhashable_arg_type(pos, kwd)
            name = plain_getattr(fn, "__qualname__", repr(fn))
            raise KeyBuildFailed(
                "KEY-UNHASHABLE-DEFAULT",
                f"@cash.cache: a parameter default of type {bad_type} on the helper "
                f"{name} could not be hashed ({class_name(type(e))}), so the call ran "
                f"uncached rather than risk serving a result computed under a "
                f"default that changed.",
                f"get the value out of {name}'s signature -- build it in the body or "
                f"pass it at the call site -- or register a hasher with "
                f"cash.register_hasher({bad_type}, ...).",
            ) from e
        identity = f"{source}:defaults:{digest}"
        if watch:
            pending = CAPTURE_WATCH.get()
            if pending is not None:
                pending.update(watch)
        if snapshots is not None and self._cycles == cycles:
            # Each held function by its identity from a walk of its own,
            # which is what a later check compares with.
            own = [(held_fn, self._function_state(held_fn, frozenset())) for held_fn in held]
            if self._cycles == cycles:
                self._defaults_memo[memo_key] = (
                    fn,
                    defaults,
                    kwdefaults,
                    cells,
                    snapshots,
                    identity,
                    watch,
                    code_source,
                    own,
                )
                return identity
        self._defaults_memo.pop(memo_key, None)
        return identity


class ClosureFold:
    """The closure, default and bound-instance folds of the state segment."""

    def __init__(
        self,
        args: ArgHasher,
        captures: CaptureAnalysis,
        helpers: HelperIdentity,
        values: GlobalValues,
        classes: ClassDataFold,
        mutations: LearnedMutations,
        notices: Notices,
    ) -> None:
        self._args = args
        self._captures = captures
        self._helpers = helpers
        self._values = values
        self._classes = classes
        self._mutations = mutations
        self._notices = notices
        # function object -> digest of its parameter defaults, for defaults that
        # are immutable and therefore cannot drift between calls.
        # Weak so the memo dies with the function instead of pinning it (and so
        # a later function object can never inherit a dead one's entry by
        # id-reuse). Mutable defaults are deliberately absent: they must be
        # re-hashed per call to stay correct.
        self._defaults_pins: weakref.WeakKeyDictionary = _weak.WeakKeyDictionary()
        # Set by `CodeArgs`: the walk of the code a value holds, for a captured
        # container of functions.
        self.code_args: CodeArgs | None = None
        #: `GlobalsFold.captured_code_digest`, bound by `Cash`: the digest of
        #: the code a captured container holds and of what it reads.
        self.held_code: Callable[[str, Any, str, Any], str | None] | None = None

    def fold_closure(
        self, func: Callable, func_name: str, state_hash: str, _walked: frozenset[int] = frozenset()
    ) -> str:
        """Mix a fingerprint of *func*'s captured free variables into the
        state hash.

        Two closures produced by the same factory share a source AND a qualname
        (``factory.<locals>.f``) but capture different values - so without this
        they collide on the same cache key and return each other's results (a
        silent wrong answer, e.g. ``make(2)`` vs ``make(5)``).

        Immutable captures are folded by value. Mutable captures (lists,
        dicts, arrays — e.g. a weights vector) are folded by CONTENT HASH,
        but only when the body provably just reads them: captures
        the function reassigns (``nonlocal`` counters) or may mutate in place
        (accumulators) are skipped, so their keys don't drift call-to-call.
        A side effect of per-call content hashing: externally mutating a
        folded capture correctly invalidates the closure's entries.

        A captured MODULE is folded by its name and the code of the functions
        and classes the body reads from it (`module_capture_identity`).
        """
        closure = getattr(func, "__closure__", None)
        code = plain_getattr(func, "__code__", None)
        if not closure or code is None:
            return state_hash
        freevars = getattr(code, "co_freevars", ()) or ()
        # Free vars the function REASSIGNS (nonlocal counters) drift across calls
        # even when their value type is immutable - exclude them.
        written = self._captures.written_freevars(code)
        unsafe = self._captures.unsafe_uses(func)
        # Captures excluded ONLY because they were passed to a call. Folded, then
        # confirmed at runtime -- same treatment as module globals. A
        # missing entry means "unknown": watch it rather than fold it blind.
        provisional = self._captures.provisional(code)
        learned_mutating = self._mutations.of(code, "closure")
        captures = []
        for name, cell in zip(freevars, closure):
            if name in written or name in learned_mutating:
                continue
            try:
                v = cell.cell_contents
            except ValueError:
                continue
            if issubclass(type(v), types.ModuleType):
                # A module pickles as nothing, so it fell to the content-hash
                # path below, failed, and was skipped: `make(ha)` and
                # `make(hb)`, each reading `cap.helper`, shared one key and
                # returned each other's results.
                attrs = self._captures.attr_reads(code).get(name, frozenset())
                captures.append((name, module_capture_identity(v, attrs)))
                continue
            if is_cash_wrapper(v):
                # A captured CACHED function is what it computes: its
                # dependency state, as a registry holding one counts it
                # (`GlobalValues.data_callable_identity`). Not cash's wrapper
                # around it, whose closure holds this Cash instance and the
                # function's spec, backend and all: none of that is an
                # input, and the backend's dicts change under the write
                # thread while a key is built.
                captures.append((name, self._values.data_callable_identity(v)))
                continue
            # A captured CALLABLE is its code and what it was built with
            # (`HelperIdentity.callable_state`). Reaching this before the
            # `unsafe` check is the point: a capture the body PASSES TO A
            # CALL is marked unsafe and skipped (watch it, don't fold it
            # blind), and calling is exactly what you do with a captured
            # function. So the strategy-factory shape
            #
            #     def make(weight_fn):
            #         @cash.cache
            #         def score(px, n): return f(px, weight_fn(n))
            #         return score
            #
            # needs its captured function keyed: two scorers built with
            # different weightings share a source and a qualname
            # (`make.<locals>.score`), and only `weight_fn` tells them apart
            # -- by its code, by the model a captured `model.predict` is
            # bound to, by what a captured `functools.partial` binds, by
            # what the function a captured closure calls was built with.
            #
            # A call cannot mutate a function, so the reason `unsafe` exists
            # does not apply. A class and a callable INSTANCE of the user's
            # take the paths below: a class is keyed with its data, an
            # instance is hashed and watched like any captured object.
            if (
                callable(v)
                and not isinstance(v, type)
                and not isinstance(v, SYNC_TYPES)
                and not is_user_callable_instance(v)
            ):
                part = self._helpers.held_callable_part(v, _walked | {id(func)})
                if part is not None:
                    captures.append((name, part))
                continue
            part = self._captured_value_part(func, func_name, name, v, name not in unsafe, provisional)
            if part is not None:
                captures.append(part)
        if not captures:
            return state_hash
        clo = self._args.serialize_args(func_name, tuple(captures), {})
        if not clo:
            return state_hash
        return hashlib.sha256(f"{state_hash}:closure:{clo}".encode()).hexdigest()

    def _captured_value_part(
        self, func: Callable, func_name: str, name: str, v: Any, read_only: bool, provisional: Any
    ) -> tuple[str, Any] | None:
        """The key part of a captured value *v* that is not code or a module,
        or ``None`` when it is not part of the key. *read_only*: the body
        provably only reads it."""
        if is_immutable_capture(v):
            return (name, v)
        if isinstance(v, SYNC_TYPES):
            # A lock holds no data.
            return None
        if callable(v) and not is_user_callable_instance(v):
            # A class is code, the user's followed by the helper walk; keyed
            # here by what it names, for the code that walk does not follow
            # (`make(np.float32)`). Any other callable was keyed by
            # `HelperIdentity.callable_state` before this is reached.
            ref = code_ref(v)
            if ref is None:
                return None
            if isinstance(v, type) and is_user_class(v, own_package(func)):
                # What a class of the user's holds, as for one read as a
                # module global (`ClassDataFold.class_parts`): its data,
                # what its code reads, its annotations when they are read.
                parts = self._classes.class_parts(
                    v, func_name, owner_code=plain_getattr(func, "__code__", None), reader=func
                )
                if parts:
                    payload = ":".join(f"{n}={h}" for n, h in sorted(parts))
                    return (name, f"ref:{ref}:" + hashlib.sha256(payload.encode("utf-8")).hexdigest())
            return (name, f"ref:{ref}")
        if read_only:
            # Read-only mutable capture: fold its content hash. One that
            # cannot be hashed runs the call uncached rather than being
            # left out of the key.
            try:
                h = self._values.table_digest("capture", v, lambda: capture_digest(self._args, v))
            except _UNHASHABLE_CAPTURE_ERRORS as e:
                raise unhashable_capture(func, name, v, e) from e
            pending = CAPTURE_WATCH.get()
            if pending is not None and (provisional is None or name in provisional):
                pending[name] = (h, "closure", None, func)
            # The code it holds (`steps = [lambda x: price(x)]`) and what
            # that code reads: its digest has the code alone, and `price`
            # reading RATE moved nothing. As for a module global holding it
            # (`GlobalsFold._held_code_parts`).
            # Kept per version of a table of plain functions
            # (`GlobalsFold.captured_code_digest`).
            if self.held_code is not None:
                code_digest = self.held_code(name, v, func_name, plain_getattr(func, "__code__", None))
                if code_digest is not None:
                    h = f"{h}:{code_digest}"
            return (name, h)
        return None

    def fold_defaults(
        self,
        func: Callable,
        func_name: str,
        state_hash: str,
        leave_out: frozenset[str] = frozenset(),
    ) -> str | None:
        """Mix the callee's parameter defaults into the state hash.

        A default is an input to the result exactly like a passed argument, but
        it lives on the FUNCTION OBJECT, not in the code object — so the bytecode
        fingerprint the state hash falls back to when source is unavailable
        (functions defined in an IPython cell) cannot see it: editing
        ``n_estimators=300`` to ``400`` changes no code.

        Defaults are hashed by VALUE through the same payload hasher arguments
        use, so ``register_hasher`` and the pandas/numpy-aware hashers apply
        identically. Returns ``None`` when a default cannot be hashed; the caller
        must then refuse to cache, because silently ignoring it would resurrect
        exactly the silent staleness this fold exists to prevent.

        The defaults of the parameters in *leave_out* -- left out of the key
        with ``ignore=``, ``cash.Ignore`` or ``key=`` -- are not folded
        (`defaults_of`).
        """
        # Memo first: this runs on EVERY decorated call, so the hot path must be
        # one lookup plus one hash, with no re-walk of the function.
        # Not every callable can be weak-referenced (numpy's dispatcher can't),
        # and WeakKeyDictionary raises on LOOKUP too, not just on store.
        try:
            entry = self._defaults_pins.get(func)
            pinnable = True
        except TypeError:
            entry = None
            pinnable = False
        if entry is not None:
            # Validate against the live containers rather than trusting the
            # function object's identity: `f.__defaults__ = (400,)` rebinds them
            # on the SAME object, and a memo keyed on identity alone would pin
            # the old digest and hand back a stale result — the very failure
            # this fold exists to prevent. Only immutable defaults are pinned,
            # so comparing the containers by value is sound (and is a cheap
            # C-level compare of a tiny tuple/dict).
            pin_pos, pin_kwd, pin_leave_out, digest = entry
            if (
                pin_leave_out == leave_out
                and pin_pos == getattr(func, "__defaults__", None)
                and pin_kwd == (getattr(func, "__kwdefaults__", None) or {})
            ):
                return hashlib.sha256(f"{state_hash}:defaults:{digest}".encode("utf-8")).hexdigest()
        pos, kwd = defaults_of(func, leave_out)
        try:
            # A callback default (`def f(x, key=lambda v: v)`) is not what
            # pickle would write of it (a name, or nothing): it is its code
            # and what it was built with, as wherever else a callable is
            # held (`HelperIdentity.defaults_digest`).
            digest = self._helpers.defaults_digest(pos, kwd, frozenset({id(func)}))
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError) as e:
            return self._defaults_unhashable(func_name, pos, kwd, e)
        return self._finish_defaults_fold(func, state_hash, digest, pos, kwd, pinnable, leave_out)

    def _defaults_unhashable(
        self,
        func_name: str,
        pos: tuple,
        kwd: dict,
        e: Exception,
    ) -> None:
        """Warn (once) that a default is unhashable; ``None`` = refuse to cache."""
        bad_type = self._args.first_unhashable_arg_type(pos, kwd)
        self._notices.warn_once(
            CashCacheIneffectiveWarning,
            func_name,
            bad_type,
            f"@cash.cache on {func_name}: a parameter default of type "
            f"{bad_type} could not be hashed ({class_name(type(e))}), so the "
            f"function does not cache for any caller rather than risk "
            f"serving a result computed under a default that changed.",
            code="KEY-UNHASHABLE-DEFAULT",
            fix="get the value out of the signature -- build it in the body "
            "or require it at the call site"
            + (
                ""
                if isinstance(self._args.first_unhashable_arg(pos, kwd), CODE_VALUE_TYPES)
                else f" -- or register a hasher with cash.register_hasher({bad_type}, ...)"
            )
            + "; a parameter that does not change the result (a connection, a lock) can "
            "instead be left out of the key, default and all, with ignore=[...] or cash.Ignore.",
        )
        return None

    def _finish_defaults_fold(
        self,
        func: Callable,
        state_hash: str,
        digest: str,
        pos: tuple,
        kwd: dict,
        pinnable: bool,
        leave_out: frozenset[str],
    ) -> str:
        """Memoize *digest* when it cannot drift, then mix it into *state_hash*."""
        # Two conditions gate the memo, and both are load-bearing:
        #
        # 1. Every default must be immutable. A mutable default is shared across
        #    calls and can be mutated in place (`def f(xs=[])`), so its content
        #    must be re-read every call; a memo would pin the first call's value
        #    and hand that result back forever.
        # 2. The callee must not wrap another function. The memo is validated
        #    against `func`'s OWN containers, which say nothing about defaults
        #    reached through `__wrapped__`; re-hashing those per call keeps the
        #    validation honest rather than merely cheap.
        if (
            pinnable
            and plain_getattr(func, "__wrapped__", None) is None
            and all(is_immutable_capture(v) for v in pos)
            and all(is_immutable_capture(v) for v in kwd.values())
        ):
            try:
                self._defaults_pins[func] = (
                    getattr(func, "__defaults__", None),
                    dict(getattr(func, "__kwdefaults__", None) or {}),
                    leave_out,
                    digest,
                )
            except TypeError:
                pass  # not weak-referenceable; recompute per call
        return hashlib.sha256(f"{state_hash}:defaults:{digest}".encode("utf-8")).hexdigest()

    def fold_bound_partial(self, func: Callable, func_name: str, state_hash: str) -> str:
        """Mix what a cached ``functools.partial`` binds into the key.

        ``c.cache(partial(total, arr))``: the bound values never appear in
        the call's arguments. A ``repr`` taken at decoration is not enough: it
        elides the middle of an array, so two arrays differing there would
        share an entry, and it holds an ordinary object's address, which
        never hits in another process and does not see a change to the
        object. They are hashed per call, as arguments are.
        One that cannot be hashed leaves the call uncached: keying it on
        anything less would serve one binding's result to another.
        """
        if not isinstance(func, functools.partial):
            return state_hash
        try:
            bound = self._args.hash_payload(func.args, func.keywords)
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError) as e:
            suspect = self._args.first_unhashable_arg_type(func.args, func.keywords)
            what = "a value" if suspect == "<unknown>" else f"a {suspect}"
            raise KeyBuildFailed(
                "KEY-UNHASHABLE-ARG",
                f"@cash.cache on {func_name}: {what} the functools.partial binds could "
                f"not be hashed ({class_name(type(e))}), so this call does not cache.",
                "pass the value as an argument of the call instead of binding it, or "
                "register a hasher for its type with cash.register_hasher(...).",
            ) from e
        return hashlib.sha256(f"{state_hash}:partial:{bound}".encode("utf-8")).hexdigest()

    def key_function_part(self, key_fn: Callable) -> str:
        """What a ``key=`` function holds besides its code, for the key.

        A function's captured values and defaults (`HelperIdentity.identity`):
        ``key=by("id")`` and ``key=by("sku")`` share their code and differ
        only in the cell. A callable object's state, hashed as an argument
        is: ``key=Pick("id")``. A bound method's instance, the same. Keyed by
        code alone, editing ``by("id")`` to ``by("sku")`` kept the entries
        and served one record's result for another.

        Raises `KeyBuildFailed` when that state cannot be hashed.
        """
        if type(key_fn) is types.FunctionType:
            return self._helpers.identity(key_fn)
        if is_user_callable_instance(key_fn):
            owner = key_fn
        elif inspect.ismethod(key_fn):
            owner = key_fn.__self__
        else:
            return ""
        try:
            return "state:" + self._args.hash_payload((owner,), {})
        except _UNHASHABLE_CAPTURE_ERRORS as e:
            kind = class_qualname(type(owner))
            raise KeyBuildFailed(
                "KEY-UNHASHABLE-CAPTURE",
                f"@cash.cache: the key= function is a {kind} whose state could not be hashed "
                f"({class_name(type(e))}), so the call ran uncached rather than risk keying it with "
                f"a key function that changed.",
                f"make the key function a plain function of the arguments, or register a hasher "
                f"with cash.register_hasher({kind}, ...).",
            ) from e

    def fold_bound_self(
        self,
        func: Callable,
        func_name: str,
        state_hash: str,
    ) -> str:
        """Mix a bound method's instance state into the key.

        ``c.cache(obj.method)`` wraps an already-bound method: ``self`` never
        appears in ``args``, so two instances with different state shared one
        cache key and silently returned each other's results. Fold
        ``func.__self__`` through the same machinery as an ordinary argument
        (so ``register_hasher`` applies exactly as it does for in-class
        decoration, where ``self`` arrives via ``args``). Hashed per call,
        not at decoration: instance state may change between calls.

        A callable instance passed to ``cash.cache`` is its own ``self``.

        Unhashable instances fall back to ``id(self)`` — correct (distinct
        instances stay distinct) but process-local; a one-shot warning points
        at ``register_hasher``.
        """
        if is_user_callable_instance(func):
            # `cash.cache(Scaler(2))`: the instance is its own `self`, and
            # what its code reads besides is the class's
            # (`ClassDataFold.class_parts`). `sc.k = 5` served the result for 2.
            owner = func
            parts = self._classes.class_parts(type(func), func_name)
            if parts:
                payload = ":".join(f"{n}={h}" for n, h in sorted(parts))
                state_hash = hashlib.sha256(f"{state_hash}:classes:{payload}".encode("utf-8")).hexdigest()
        else:
            # A bound method, a C method (``{1: "a"}.get``) or a
            # method-wrapper, also under a ``functools.partial``: keyed on
            # its function alone, ``{1: "b"}.get`` was served ``"a"``.
            bound, owner = bound_owner(func)
            if not bound:
                return state_hash
        try:
            self_hash = self._args.hash_payload((owner,), {})
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError) as e:
            owner_type = class_name(type(owner))
            self._notices.warn_once(
                CashCacheIneffectiveWarning,
                func_name,
                owner_type,
                f"@cash.cache on {'bound method ' if owner is not func else ''}{func_name}: the instance's state "
                f"could not be hashed ({class_name(type(e))}), so cash fell back "
                f"to the instance's identity - entries are not shared across "
                f"equal instances and do not survive the process.",
                code="KEY-INSTANCE-STATE",
                fix=f"register a hasher with "
                f"cash.register_hasher({owner_type}, ...), returning "
                f"something derived from the state that affects the "
                f"result.",
            )
            self_hash = f"selfid:{id(owner)}"
        return hashlib.sha256(f"{state_hash}:boundself:{self_hash}".encode("utf-8")).hexdigest()
