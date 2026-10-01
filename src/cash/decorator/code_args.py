"""Functions and classes passed as arguments, or carried inside one: their code
is part of the key, not their pickled bytes."""

from __future__ import annotations

import functools
import hashlib
import logging
import sys
import types
from typing import TYPE_CHECKING, Any

from .. import _plain_data
from .._active import EXPLAINING as _EXPLAINING
from .._memo import CODE_OBJECTS, LruMemo
from ..analysis.purity_analyzer import ISSUE_UNTRACKABLE_DEP, get_analyzer
from ..diagnostics import log_diagnostic, warn_diagnostic
from ..exceptions import CashImpurityWarning
from ..object_hashing import held_objects
from ..source_norm import class_functions
from ..value_types import BUILTIN_CONTAINERS, CODELESS_PRIMS, is_runtime_machinery
from .arg_hashing import is_opaque, plain_census
from .cash_key import cash_key_method
from ..install_paths import is_user_code_module
from .code_identity import cached_function_in, is_user_code_object
from .globals_fold import class_surface_functions

if TYPE_CHECKING:
    from .arg_hashing import ArgHasher
    from .code_identity import CodeIdentity
    from .frozen import FrozenResults
    from .globals_fold import GlobalsFold

logger = logging.getLogger(__name__)

#: How many containers deep one recursive stretch of the search for user code
#: goes. Not a limit on what is searched: what lies deeper is set aside and
#: searched from there once the stretch is done (`CodeArgs.iter_code_carriers`),
#: so the search reaches everything, however deep, without running out of stack.
CODE_SEARCH_DEPTH = 100


class _Deeper:
    """A value the search reached past `CODE_SEARCH_DEPTH`, to search on from.

    *find* is True for values met while SEARCHING a library object
    (`CodeArgs._find_user_code`), False for values walked as an argument is.
    """

    __slots__ = ("find", "values")

    def __init__(self, values: Any, find: bool) -> None:
        self.values = values
        self.find = find


def carrier_name(carrier: Any) -> str:
    """A stable, address-free name for a code carrier.

    ``__qualname__`` for anything that has one -- a class, a function.
    Otherwise the carrier's TYPE, deliberately not ``repr()``: a
    ``functools.partial`` reprs as ``functools.partial(<function f at
    0x...>, 3)``, and that address is unique per object, so a
    ``repr()``-derived key would make ``_warned_unhashable_code`` dedup
    nothing (one warning per partial ever constructed, plus a global set
    that grows without bound) and would make a folded part label differ
    between two processes holding equal arguments.
    """
    name = getattr(carrier, "__qualname__", None) or getattr(carrier, "__name__", None)
    if isinstance(name, str) and name:
        return name
    t = type(carrier)
    return getattr(t, "__qualname__", None) or getattr(t, "__name__", None) or "?"


def is_user_code_carrier(carrier: Any) -> bool:
    """``is_user_code_object`` for the ADVISORY rather than for hashing.

    ``is_user_code_object`` answers "could not confirm reachability ->
    treat as user code". That is the safe direction when deciding whether
    to HASH something and the wrong one when deciding whether to WARN about
    it: an object with no ``__qualname__`` of its own -- a
    ``functools.partial``, a ``weakref.ref`` -- can never be confirmed, so
    every single one was reported as un-hashable user code.

    Judge such an object by what it WRAPS (``.func``, the same attribute
    ``CodeIdentity.class_surface_parts`` already follows for ``singledispatchmethod``
    and ``cached_property``), else by its TYPE. Measured:
    ``functools.partial(json.dumps)`` and ``weakref.ref(x)`` stop warning,
    while ``functools.partial(<a user function>)`` still warns -- and it
    must, because the wrapped body genuinely is absent from the key.
    """
    if getattr(carrier, "__qualname__", None) or getattr(carrier, "__name__", None):
        return is_user_code_object(carrier)
    wrapped = getattr(carrier, "func", None)
    if wrapped is not None:
        return is_user_code_object(wrapped)
    return is_user_code_object(type(carrier))


def defined_in_user_code(obj: Any) -> bool:
    """`is_user_code_carrier`, for a function or class found INSIDE a library
    object: judged by the module it names when that module has a file.

    `is_user_code_object` takes an object it cannot find again in its module
    for the user's (the safe side when keying): a function defined in another
    function (``WeakSet.__init__.<locals>._remove``), one a decorator replaced
    (traitlets' observers). Inside a library object that is the library's
    own code, and taking it for the user's walked on through its closure into
    the rest of the process. A module without a file (a notebook's
    ``__main__``, exec'd code) keeps the lenient verdict.
    """
    mod = sys.modules.get(getattr(obj, "__module__", None) or "")
    if mod is not None and getattr(mod, "__file__", None) is not None:
        return is_user_code_module(mod)
    return is_user_code_carrier(obj)


class CodeArgs:
    """The user code an argument carries -- a class, a function, an instance
    of the user's own class -- folded into the state segment."""

    def __init__(self, code: CodeIdentity, globals_fold: GlobalsFold, frozen: FrozenResults, args: ArgHasher) -> None:
        self._code = code
        # A value a registered hasher keys (`ArgHasher.keys_by_registration`)
        # is not searched: the user has said what identifies it. The two
        # registries are held too, to skip the question per element while
        # they are empty; `ArgHasher.register_hasher` fills them in place.
        self._args = args
        self._registries = (args.override_hashers, args.type_hashers)
        self._globals = globals_fold
        self._frozen = frozen
        # A data global carries code the same way an argument does
        # (`GlobalsFold.fold_read_globals`), and folds it through this walk.
        globals_fold.code_args = self
        # ``(class, is user code)`` per class id, for `_iter_attribute_carriers`:
        # a list of 50k instances must not pay the verdict per element. The
        # class is kept so a recycled id is never trusted.
        self._attribute_walk_verdicts: LruMemo[tuple[str, int], tuple[type, bool]] = LruMemo(CODE_OBJECTS)
        # Code carriers already reported (`_warn_unhashable_code_once`,
        # `_warn_untrackable_in_carrier_once`): once per carrier and function.
        self._warned_unhashable_code: set[tuple] = set()
        self._warned_untrackable_carrier: set[tuple] = set()

    def _warn_untrackable_in_carrier_once(self, carrier: Any, func_name: str = "?", param: str | None = None) -> None:
        """Say once that code reached through an argument resolves a dependency
        at runtime, so an edit behind it will NOT invalidate.

        The cached function's own body refuses such a line
        (``untrackable_dep``); a method of an argument's class was never
        analysed, and ``getattr(MOD, name)()`` there served a stale result in
        silence. ``# @cash:assume-safe`` on the line, or a ``with
        cash.assume_safe():`` block around it, waives it, as in the function
        itself.
        """
        if _EXPLAINING.get():
            return

        functions = class_functions(carrier) if isinstance(carrier, type) else [getattr(carrier, "__func__", carrier)]
        for fn in functions:
            if not isinstance(fn, types.FunctionType):
                continue
            mark = (id(fn.__code__), func_name)
            if mark in self._warned_untrackable_carrier:
                continue
            self._warned_untrackable_carrier.add(mark)
            try:
                issues = [i for i in get_analyzer().analyze(fn).issues if i.kind == ISSUE_UNTRACKABLE_DEP]
            except Exception:  # noqa: BLE001 - never break a call
                continue
            if not issues:
                continue
            first = issues[0]
            where = f"the argument `{param}` of {func_name}" if param else f"a call of {func_name}"
            what = (
                f"{fn.__qualname__}, reached through {where}, resolves a dependency "
                f"from a runtime value (line {first.line}: {first.description}), so an "
                f"edit to what it reaches will NOT invalidate the cache."
            )
            fix = (
                "call what it needs by name, or name it with "
                "@cash.cache(depends_on=[...]); put `# @cash:assume-safe` on that "
                "line, or `with cash.assume_safe():` around it, once you have "
                "checked a stale result cannot matter."
            )
            log_diagnostic(logger, "KEY-DYNAMIC-DEPENDENCY", what, fix)
            warn_diagnostic(CashImpurityWarning, "KEY-DYNAMIC-DEPENDENCY", what, fix)

    def _warn_unhashable_code_once(self, carrier: Any, func_name: str = "?", param: str | None = None) -> None:
        """Tell the user once that a reached type's code is NOT in the key.

        "reached", not "passed": the code channel keys off the BOUND arguments,
        so a carrier arriving as a parameter default the caller never typed
        gets here too.

        Names the parameter, the cached function and what the object wraps.
        With only the object's type named, an object the user had already
        covered with depends_on= and a new one nobody had covered printed the
        same line -- a new hole looked like a handled one.
        Once per (object, function, parameter) for the same reason.
        """
        if _EXPLAINING.get():
            return
        name = carrier_name(carrier)
        inner = getattr(carrier, "func", None) or getattr(carrier, "__wrapped__", None)
        inner_name = (
            getattr(inner, "__qualname__", None) or getattr(inner, "__name__", None) if inner is not None else None
        )
        mark = (f"{name}:{inner_name}", func_name, param)
        if mark in self._warned_unhashable_code:
            return
        self._warned_unhashable_code.add(mark)
        where = f"the argument `{param}` of {func_name}" if param else f"a call of {func_name}"
        wrapping = f", wrapping {inner_name}," if inner_name else ""
        what = (
            f"{name}{wrapping} reached {where}, but its code could not be "
            f"hashed, so editing it will NOT invalidate the cache."
        )
        kind = carrier if isinstance(carrier, type) else type(carrier)
        kind_name = getattr(kind, "__qualname__", None) or getattr(kind, "__name__", "?")
        fix = (
            f"name what it runs with @cash.cache(depends_on=[...]) if the result "
            f"depends on its implementation, or pass it in a form cash can read "
            f"(a plain function and keyword arguments). cash.opaque({kind_name}) "
            f"records that the code does not matter -- for every {kind_name} in "
            f"the process, including ones added later."
        )
        # The log carries the same rendered text as the warning, code and all,
        # so a log-only reader is not the one person without a handle to search.
        log_diagnostic(logger, "KEY-OPAQUE-CALLABLE", what, fix)
        warn_diagnostic(
            CashImpurityWarning,
            "KEY-OPAQUE-CALLABLE",
            what,
            fix,
        )

    def iter_code_carriers(self, value: Any, _seen: set | None = None):
        """Yield objects in *value* that carry user code, however deep.

        ``_seen`` guards self-referential containers, and doubles as a
        once-per-argument dedup for the classes yielded on behalf of
        instances: a list of 50k objects of one class must evaluate the
        user-code gate once, not 50k times. Dedup by identity is safe in both
        roles -- a class already yielded does not need yielding again, and
        the fold takes a ``set`` of the parts anyway.

        The walk recurses `CODE_SEARCH_DEPTH` containers at a time; what lies
        deeper is set aside and walked from there after. It stopped there
        before, and a function held deeper was keyed by its name only, so
        editing it served the old result.
        """
        if _seen is None:
            _seen = set()
        pending = [_Deeper(value, False)]
        while pending:
            deeper = pending.pop()
            if deeper.find:
                walk = self._find_user_code(deeper.values, 0, _seen)
            else:
                walk = self._walk_carriers(deeper.values, 0, _seen)
            for carrier in walk:
                if type(carrier) is _Deeper:
                    pending.append(carrier)
                else:
                    yield carrier

    def _walk_carriers(self, value: Any, _depth: int, _seen: set):
        """`iter_code_carriers` for one stretch of `CODE_SEARCH_DEPTH` containers."""
        if _depth > CODE_SEARCH_DEPTH:
            if type(value) not in CODELESS_PRIMS:
                yield _Deeper(value, False)
            return
        # Primitives (and numpy numbers) carry no user code, and in a large argument they ARE the
        # argument. Returning before ``_seen`` is touched keeps a list of a
        # million numbers allocation-free; otherwise the id-set below would grow
        # to the container's length on every cached call. Mirrors
        # ``iter_contained``'s first line. See `CODELESS_PRIMS` for why this
        # is an exact-type test against a tuple rather than an isinstance.
        if type(value) in CODELESS_PRIMS or type(value) in _plain_data.numpy_scalar_set():
            return
        # A lock, a thread, a logger, a stream: no code a result depends on,
        # and the whole process is reachable through them (`is_runtime_machinery`).
        if is_runtime_machinery(value):
            return
        # A frozen function's list/tuple/dict result is keyed by the call that
        # produced it (`FrozenResults.remember_container`), code inside it included:
        # walking two million rows for functions was most of a hit's cost.
        # Checked BEFORE the plain-data census below, which is itself a walk:
        # in that order frozen=True on a list still cost a linear pass per call.
        if (
            self._frozen.containers
            and id(value) in self._frozen.containers
            and self._frozen.containers[id(value)][0] is value
        ):
            return
        # Plain and JSON-like data carries no code (`plain_census`); walking
        # two million rows to find that out was 14% of a warm hit.
        if _depth == 0 and type(value) in _plain_data.TREE_NODES and plain_census(value) is not None:
            return
        # ``_seen`` is recorded on the paths that need it -- containers, for
        # cycle safety, and yielded carriers, to yield each once -- and NOT for
        # a leaf instance. A leaf cannot contain itself, and its class is
        # deduped by `_instance_class_carrier` anyway, so an entry per element
        # bought nothing and cost a set insert per element: measured 2000
        # ns/element at 200k against 470 ns/element at 10k, i.e. the set itself
        # had become the superlinear term.
        if isinstance(value, type):
            if id(value) not in _seen:
                _seen.add(id(value))
                yield value
            return
        if callable(value):
            # Distinguish a callable INSTANCE (whose class defines ``__call__``
            # in Python) from a function/method/C-callable. An instance has no
            # ``__code__`` of its own -- its code lives on its class -- so
            # yielding the instance would fold NOTHING, while the identical
            # object WITHOUT ``__call__`` takes the instance branch below and
            # folds its class: adding ``__call__`` to a class must not remove
            # that class's code from the key. ``is_opaque`` returns the same verdict for a class as
            # for one of its instances, so routing the class here rather than
            # the instance leaves opacity unchanged.
            call = getattr(type(value), "__call__", None)
            if getattr(call, "__code__", None) is not None:
                cls = self._instance_class_carrier(value, _seen)
                if cls is not None:
                    yield cls
                if not self._keyed_by_registration(value):
                    yield from self._iter_attribute_carriers(value, _depth, _seen)
                return
            if id(value) not in _seen:
                _seen.add(id(value))
                yield value
            return
        # The primitive test is repeated INLINE in each loop below rather than
        # left to the recursive call's own first line. It is the same test and
        # the same result, but it skips building a generator frame per element,
        # and a container of primitives is the overwhelmingly common argument:
        # measured 25.8ms -> 7.2ms for a 200k-int list.
        if isinstance(value, dict):
            if id(value) in _seen:
                return
            _seen.add(id(value))
            # A dict SUBCLASS is both a container to walk and a user object
            # whose class carries code. Handling only the first is the same
            # "invisible because of what it inherits from" hole that the
            # exact-type test above closes for str/int subclasses.
            if type(value) is not dict:
                cls = self._instance_class_carrier(value, _seen)
                if cls is not None:
                    yield cls
            for k, v in value.items():
                if type(k) not in CODELESS_PRIMS:
                    yield from self._walk_carriers(k, _depth + 1, _seen)
                if type(v) not in CODELESS_PRIMS:
                    yield from self._walk_carriers(v, _depth + 1, _seen)
        elif isinstance(value, (list, tuple, set, frozenset)):
            if id(value) in _seen:
                return
            _seen.add(id(value))
            if type(value) not in BUILTIN_CONTAINERS:
                cls = self._instance_class_carrier(value, _seen)
                if cls is not None:
                    yield cls
            for v in value:
                if type(v) not in CODELESS_PRIMS:
                    yield from self._walk_carriers(v, _depth + 1, _seen)
        else:
            # An instance contributes its class's code. Deliberately NOT gated
            # on ``hasattr(value, "__dict__")``: a class using ``__slots__``
            # gives its instances no ``__dict__``, and that has nothing to do
            # with whether the user edits the class. Same family as the two
            # holes above -- a user class invisible because of how it is
            # declared rather than because of what it is.
            cls = self._instance_class_carrier(value, _seen)
            if cls is not None:
                yield cls
            # A value a registered hasher keys contributes its class's code
            # and nothing it holds. `register_hasher(logging.Logger, ...)`
            # still walked every logger, handler and stream in the process
            # (a logger holds its manager), and a handler holding a bound
            # builtin warned that its code was not in the key.
            # So does one its class's ``__cash_key__`` keys.
            # Inline, not `_keyed_by_registration`: this runs per element.
            if cash_key_method(value) is None and not (
                (self._registries[0] or self._registries[1]) and self._args.keys_by_registration(value)
            ):
                yield from self._iter_attribute_carriers(value, _depth, _seen)

    def _keyed_by_registration(self, value: Any) -> bool:
        """`ArgHasher.keys_by_registration`, skipped while nothing is registered
        and *value* has no ``__cash_key__``."""
        if cash_key_method(value) is not None:
            return True
        override, typed = self._registries
        return bool(override or typed) and self._args.keys_by_registration(value)

    def _iter_attribute_carriers(self, value: Any, _depth: int, _seen: set):
        """Code carried by what an instance of the user's own class HOLDS.

        ``f(A(1, B()))`` keyed ``A``'s code, and ``A.f`` calling ``self.b.f()``
        reached ``B`` -- whose code, and everything it calls, never entered the
        key: editing ``B.f`` or a function it called served the old result.
        Only an instance whose class is user code is looked into (a library
        object's attributes are its own business), each once per walk, bounded
        by the same depth; attributes that are plain values cost a type test.
        """
        if id(value) in _seen:
            return
        cls = type(value)
        key = ("user-class", id(cls))
        verdict = self._attribute_walk_verdicts.get(key)
        if verdict is None or verdict[0] is not cls:
            verdict = (cls, is_user_code_object(cls))
            self._attribute_walk_verdicts[key] = verdict
        if not verdict[1]:
            yield from self._iter_library_held(value, _depth, _seen)
            return
        attrs = getattr(value, "__dict__", None)
        values: list[Any] = list(attrs.values()) if isinstance(attrs, dict) else []
        for klass in type(value).__mro__:
            slots = klass.__dict__.get("__slots__", ())
            for slot in (slots,) if isinstance(slots, str) else slots:
                if slot in ("__dict__", "__weakref__"):
                    continue
                try:
                    values.append(getattr(value, slot))
                except AttributeError:
                    continue
        held = [v for v in values if type(v) not in CODELESS_PRIMS]
        if not held:
            return
        _seen.add(id(value))
        for v in held:
            yield from self._walk_carriers(v, _depth + 1, _seen)

    def _iter_library_held(self, value: Any, _depth: int, _seen: set):
        """User code a LIBRARY object holds: looked for, not keyed on the way.

        ``make_pipeline(Scale(), FunctionTransformer(double))`` is sklearn's,
        so the walk stopped at it, and an edit to ``Scale.transform`` or
        ``double`` was served the old result -- as an argument and as a global.
        The library's own attributes are only searched: what is found and is
        user code (a function, a class, an instance of one) is walked like an
        argument, and nothing of the library's own reaches the key, so its
        caches and fitted state cannot churn it. Every value is looked at:
        a search that gave up after 2000 missed the user function in a
        pipeline whose fitted step held a large vocabulary.
        """
        if id(value) in _seen:
            return
        # What an object array or column holds: every element is looked at,
        # as a list's are (`held_objects`).
        held = held_objects(value)
        if held:
            _seen.add(id(value))
            for v in held:
                yield from self._walk_carriers(v, _depth + 1, _seen)
        attrs = getattr(value, "__dict__", None)
        if not isinstance(attrs, dict) or not attrs:
            return
        _seen.add(id(value))
        yield from self._find_user_code(attrs.values(), _depth + 1, _seen)

    def _find_user_code(self, values: Any, _depth: int, _seen: set):
        """The user code among *values*, searched through library objects."""
        if _depth > CODE_SEARCH_DEPTH:
            yield _Deeper(list(values), True)
            return
        for v in values:
            if type(v) in CODELESS_PRIMS or id(v) in _seen or isinstance(v, types.ModuleType):
                continue
            if is_runtime_machinery(v):
                continue
            if type(v) in BUILTIN_CONTAINERS:
                _seen.add(id(v))
                yield from self._find_user_code(v.values() if isinstance(v, dict) else v, _depth + 1, _seen)
            elif isinstance(v, (types.FunctionType, type)):
                if defined_in_user_code(v):
                    yield from self._walk_carriers(v, _depth, _seen)
            elif isinstance(v, types.MethodType):
                _seen.add(id(v))
                yield from self._find_user_code((v.__func__, v.__self__), _depth + 1, _seen)
            elif isinstance(v, functools.partial):
                _seen.add(id(v))
                yield from self._find_user_code((v.func, *v.args, *v.keywords.values()), _depth + 1, _seen)
            elif self._is_user_instance(v) and defined_in_user_code(type(v)):
                yield from self._walk_carriers(v, _depth, _seen)
            elif self._keyed_by_registration(v):
                continue
            else:
                attrs = getattr(v, "__dict__", None)
                if isinstance(attrs, dict) and attrs:
                    _seen.add(id(v))
                    yield from self._find_user_code(attrs.values(), _depth + 1, _seen)

    def _is_user_instance(self, value: Any) -> bool:
        """Is *value*'s class user code? Memoized per class, as for the attribute walk."""
        cls = type(value)
        key = ("user-class", id(cls))
        verdict = self._attribute_walk_verdicts.get(key)
        if verdict is None or verdict[0] is not cls:
            verdict = (cls, is_user_code_object(cls))
            self._attribute_walk_verdicts[key] = verdict
        return verdict[1]

    def _instance_class_carrier(self, value: Any, _seen: set) -> type | None:
        """``type(value)`` if it is user code and not already seen this walk.

        Split out because three branches need it, and because the dedup is the
        difference between one user-code gate evaluation per ARGUMENT and one
        per ELEMENT -- ``is_user_code_object`` is a ``sys.modules`` lookup plus
        a ``__qualname__`` walk, and a list of 50k instances of one class was
        paying it 50k times (measured: 50000 calls -> 1).

        A plain function rather than a generator on purpose: the callers are in
        the per-element path, and `yield from` on a fresh generator costs more
        per element than the call plus the `is not None` test it replaces.
        """
        cls = type(value)
        if id(cls) in _seen:
            return None
        _seen.add(id(cls))
        return cls if is_user_code_object(cls) else None

    def fold_code_args(
        self, args: tuple, kwargs: dict, state_hash: str, func_name: str = "?", owner_code: Any = None
    ) -> str:
        """Fold user code reached through the arguments into the key.

        ``args_hash`` is a digest of the PICKLED arguments, and pickle
        serializes a class or function by reference -- module plus qualname,
        never its code. So editing a passed class produced an identical key and
        a hit, and cash handed back the stale class object it had cached.

        Folded into ``state_hash`` rather than ``args_hash`` because this is
        code, and ``state_hash`` is already where code lives: function source,
        ``depends_on``, transitive helpers, module globals read.
        """
        parts: list[str] = []
        seen_carriers: set[int] = set()
        for param, value in (*((None, a) for a in args), *kwargs.items()):
            parts.extend(self.carrier_parts(value, func_name, param, seen_carriers, owner_code))
        if not parts:
            return state_hash
        payload = ":".join(sorted(set(parts)))
        return hashlib.sha256(f"{state_hash}:codeargs:{payload}".encode("utf-8")).hexdigest()

    def carrier_parts(
        self,
        value: Any,
        func_name: str = "?",
        param: str | None = None,
        seen_carriers: set[int] | None = None,
        owner_code: Any = None,
    ) -> list[str]:
        """Key parts for the user code *value* carries: each carrier's code and
        what that code reads. One walk for an argument and a data global.

        *seen_carriers* dedups across several values of one call:
        `f(a, b, c)` with three instances of one class reaches `is_opaque` +
        `CodeIdentity.code_surface_hash` once instead of three times. Safe by
        identity because every carrier stays reachable from the values for
        the whole key build, so no id can be recycled underneath us.
        *owner_code* is the cached function's code, for the drift guard.
        """
        if seen_carriers is None:
            seen_carriers = set()
        parts: list[str] = []
        # A clock test double's date is the date, not code (`fake_clock`).
        fake_dates = _plain_data.fake_clock()[0]
        for carrier in self.iter_code_carriers(value):
            if fake_dates and (type(carrier) in fake_dates or carrier in fake_dates):
                continue
            if id(carrier) in seen_carriers:
                continue
            seen_carriers.add(id(carrier))
            cached = cached_function_in(carrier)
            if cached is not None:
                # A cached function (bare, or under a partial) is what it
                # computes: its whole state, as a call of it keys it. Its
                # wrapper is cash's code, and the globals that code reads are
                # cash's own -- a constant key part, and a false
                # KEY-UNHASHABLE-GLOBAL naming them.
                parts.append(f"cached:{carrier_name(cached)}:{self._globals.data_callable_identity(cached)}")
                continue
            if is_opaque(carrier):
                continue
            digest = self._code.code_surface_hash(carrier)
            if digest is not None:
                parts.append(f"{carrier_name(carrier)}:{digest}")
                # Its CODE is in the key; the globals that code reads
                # were not. A callback reading a module
                # constant served the old result after the constant
                # changed, while the same read one call level deeper,
                # or in the cached function itself, invalidated.
                if is_user_code_carrier(carrier):
                    parts.extend(self._carrier_read_global_parts(carrier, func_name, owner_code))
                    self._warn_untrackable_in_carrier_once(carrier, func_name, param)
            elif is_user_code_carrier(carrier):
                # User code we could not hash: a C-extension type, an
                # exotic descriptor. We fall back to today's key, which means
                # an edit will NOT invalidate -- so say so once. This is
                # the residue where cash genuinely cannot determine the
                # answer, and silence is the danger.
                self._warn_unhashable_code_once(carrier, func_name, param)
        return parts

    def _carrier_read_global_parts(self, carrier: Any, func_name: str, owner_code: Any) -> list[str]:
        """Key parts for the data a code carrier's functions read: a
        function's or a bound method's through
        `GlobalsFold.fold_passed_function_reads`, a class's through
        `GlobalsFold.class_parts`. *owner_code* is the cached function's
        code, which the drift guard records under.
        """
        if isinstance(carrier, type):
            parts = [
                f"argclass:{label}:{h}"
                for label, h in self._globals.class_parts(carrier, func_name, owner_code=owner_code)
            ]
            seen: set = set()
            helpers = ""
            for member in class_surface_functions(carrier):
                if is_user_code_object(member):
                    helpers = self._globals.fold_passed_helper_reads(
                        member, func_name, helpers, owner_code=owner_code, seen=seen
                    )
            if helpers:
                parts.append(f"argclass:{carrier.__qualname__}#helpers:{helpers}")
            return parts
        fn = getattr(carrier, "__func__", carrier)
        if not isinstance(fn, types.FunctionType):
            return []
        digest = self._globals.fold_passed_function_reads(fn, func_name, owner_code)
        return [f"argglobal:{getattr(fn, '__qualname__', '?')}:{digest}"] if digest else []
