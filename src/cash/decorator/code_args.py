"""Functions and classes passed as arguments, or carried inside one: their code
is part of the key, not their pickled bytes."""

from __future__ import annotations

import hashlib
import logging
import types
from typing import Any

from .. import _plain_data
from ..dependency_state import EXPLAINING as _EXPLAINING
from ..diagnostics import log_diagnostic, warn_diagnostic
from ..exceptions import CashImpurityWarning
from ..purity_analyzer import ISSUE_UNTRACKABLE_DEP, get_analyzer
from ..source_norm import class_functions
from ..value_types import BUILTIN_CONTAINERS, CODELESS_PRIMS, PLAIN_SEQS
from .arg_hashing import plain_census
from .code_identity import CodeIdentityMixin

logger = logging.getLogger(__name__)


class CodeArgsMixin:
    """The code an argument carries, folded into the state segment."""

    @staticmethod
    def _carrier_name(carrier: Any) -> str:
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

    @staticmethod
    def _is_user_code_carrier(carrier: Any) -> bool:
        """``_is_user_code_object`` for the ADVISORY rather than for hashing.

        ``_is_user_code_object`` answers "could not confirm reachability ->
        treat as user code". That is the safe direction when deciding whether
        to HASH something and the wrong one when deciding whether to WARN about
        it: an object with no ``__qualname__`` of its own -- a
        ``functools.partial``, a ``weakref.ref`` -- can never be confirmed, so
        every single one was reported as un-hashable user code.

        Judge such an object by what it WRAPS (``.func``, the same attribute
        ``_class_surface_parts`` already follows for ``singledispatchmethod``
        and ``cached_property``), else by its TYPE. Measured:
        ``functools.partial(json.dumps)`` and ``weakref.ref(x)`` stop warning,
        while ``functools.partial(<a user function>)`` still warns -- and it
        must, because the wrapped body genuinely is absent from the key.
        """
        if getattr(carrier, "__qualname__", None) or getattr(carrier, "__name__", None):
            return CodeIdentityMixin._is_user_code_object(carrier)
        wrapped = getattr(carrier, "func", None)
        if wrapped is not None:
            return CodeIdentityMixin._is_user_code_object(wrapped)
        return CodeIdentityMixin._is_user_code_object(type(carrier))

    def _warn_untrackable_in_carrier_once(self, carrier: Any, func_name: str = "?", param: str | None = None) -> None:
        """Say once that code reached through an argument resolves a dependency
        at runtime, so an edit behind it will NOT invalidate.

        The cached function's own body refuses such a line
        (``untrackable_dep``); a method of an argument's class was never
        analysed, and ``getattr(MOD, name)()`` there served a stale result in
        silence. ``# @cash:assume-safe`` on the line waives it, as it does in
        the function itself.
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
                "line once you have checked a stale result cannot matter."
            )
            log_diagnostic(logger, "KEY-DYNAMIC-DEPENDENCY", what, fix)
            warn_diagnostic(CashImpurityWarning, "KEY-DYNAMIC-DEPENDENCY", what, fix)

    def _warn_unhashable_code_once(self, carrier: Any, func_name: str = "?", param: str | None = None) -> None:
        """Tell the user once that a reached type's code is NOT in the key.

        "reached", not "passed": the code channel keys off the BOUND arguments,
        so a carrier arriving as a parameter default the caller never typed
        gets here too.

        Names the parameter, the cached function and what the object wraps.
        Round 18: the text named only the object's type, so an object the
        user had already covered with depends_on= and a new one nobody had
        covered printed the same line -- a new hole looked like a handled one.
        Once per (object, function, parameter) for the same reason.
        """
        if _EXPLAINING.get():
            return
        name = self._carrier_name(carrier)
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

    def _iter_code_carriers(self, value: Any, _depth: int = 0, _seen: set | None = None):
        """Yield objects in *value* that carry user code.

        Depth-bounded at 8, matching ``_stabilize_for_global_hash``. ``_seen``
        guards self-referential containers, and doubles as a once-per-argument
        dedup for the classes yielded on behalf of instances: a list of 50k
        objects of one class must evaluate the user-code gate once, not 50k
        times. Dedup by identity is safe in both roles -- a class already
        yielded does not need yielding again, and the fold takes a ``set`` of
        the parts anyway.
        """
        if _depth > 8:
            return
        # Primitives carry no user code, and in a large argument they ARE the
        # argument. Returning before ``_seen`` is touched keeps a list of a
        # million numbers allocation-free; otherwise the id-set below would grow
        # to the container's length on every cached call. Mirrors
        # ``_iter_contained``'s first line. See `CODELESS_PRIMS` for why this
        # is an exact-type test against a tuple rather than an isinstance.
        if type(value) in CODELESS_PRIMS:
            return
        # A frozen function's list/tuple/dict result is keyed by the call that
        # produced it (`_remember_frozen_container`), code inside it included:
        # walking two million rows for functions was most of a hit's cost.
        # Checked BEFORE the plain-data census below, which is itself a walk:
        # in that order frozen=True on a list still cost a linear pass per call
        # (round 20, r20s2 F5).
        if (
            self._frozen_containers
            and id(value) in self._frozen_containers
            and self._frozen_containers[id(value)][0] is value
        ):
            return
        # Plain data carries no code (`_plain_data.is_plain`); walking two
        # million rows to find that out was 14% of a warm hit.
        if _depth == 0 and type(value) in PLAIN_SEQS and plain_census(value) is not None:
            return
        if _seen is None:
            _seen = set()
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
            # folds its class. Measured before this branch existed: adding
            # ``__call__`` to a class silently removed that class's code from
            # the key, and the instance then also tripped the unhashable
            # advisory. ``_is_opaque`` returns the same verdict for a class as
            # for one of its instances, so routing the class here rather than
            # the instance leaves opacity unchanged.
            call = getattr(type(value), "__call__", None)
            if getattr(call, "__code__", None) is not None:
                cls = self._instance_class_carrier(value, _seen)
                if cls is not None:
                    yield cls
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
                    yield from self._iter_code_carriers(k, _depth + 1, _seen)
                if type(v) not in CODELESS_PRIMS:
                    yield from self._iter_code_carriers(v, _depth + 1, _seen)
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
                    yield from self._iter_code_carriers(v, _depth + 1, _seen)
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
            yield from self._iter_attribute_carriers(value, _depth, _seen)

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
            verdict = (cls, self._is_user_code_object(cls))
            if len(self._attribute_walk_verdicts) < 4096:
                self._attribute_walk_verdicts[key] = verdict
        if not verdict[1]:
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
            yield from self._iter_code_carriers(v, _depth + 1, _seen)

    def _instance_class_carrier(self, value: Any, _seen: set) -> type | None:
        """``type(value)`` if it is user code and not already seen this walk.

        Split out because three branches need it, and because the dedup is the
        difference between one user-code gate evaluation per ARGUMENT and one
        per ELEMENT -- ``_is_user_code_object`` is a ``sys.modules`` lookup plus
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
        return cls if self._is_user_code_object(cls) else None

    def _fold_code_args(self, args: tuple, kwargs: dict, state_hash: str, func_name: str = "?") -> str:
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
        # A clock test double's date is the date, not code (`fake_clock`).
        fake_dates = _plain_data.fake_clock()[0]
        for param, value in (*((None, a) for a in args), *kwargs.items()):
            for carrier in self._iter_code_carriers(value):
                if fake_dates and (type(carrier) in fake_dates or carrier in fake_dates):
                    continue
                # Dedup ACROSS arguments too, not just within one walk:
                # `f(a, b, c)` with three instances of one class reaches
                # `_is_opaque` + `_code_surface_hash` once instead of three
                # times. Safe by identity because every carrier is
                # reachable from `args`/`kwargs` for this whole loop, so no
                # id can be recycled underneath us.
                if id(carrier) in seen_carriers:
                    continue
                seen_carriers.add(id(carrier))
                if self._is_opaque(carrier):
                    continue
                digest = self._code_surface_hash(carrier)
                if digest is not None:
                    parts.append(f"{self._carrier_name(carrier)}:{digest}")
                    # Its CODE is in the key; the globals that code reads
                    # were not (CAS-113). A callback reading a module
                    # constant served the old result after the constant
                    # changed, while the same read one call level deeper,
                    # or in the cached function itself, invalidated.
                    if self._is_user_code_carrier(carrier):
                        parts.extend(self._carrier_read_global_parts(carrier, func_name))
                        self._warn_untrackable_in_carrier_once(carrier, func_name, param)
                elif self._is_user_code_carrier(carrier):
                    # User code we could not hash: a C-extension type, an
                    # exotic descriptor, a ``functools.partial`` (whose
                    # wrapped function pickles by reference like any
                    # other). We fall back to today's key, which means an
                    # edit will NOT invalidate -- so say so once. This is
                    # the residue where cash genuinely cannot determine the
                    # answer, and silence is the danger.
                    self._warn_unhashable_code_once(carrier, func_name, param)
        if not parts:
            return state_hash
        payload = ":".join(sorted(set(parts)))
        return hashlib.sha256(f"{state_hash}:codeargs:{payload}".encode("utf-8")).hexdigest()

    def _carrier_read_global_parts(self, carrier: Any, func_name: str) -> list[str]:
        """Key parts for the module data a code carrier's functions read.

        The same channel the cached function's own globals go through
        (`_read_global_data_names` + `_safe_global_hash`, plus the
        ``module.ATTR`` fold), applied to code that arrived as an ARGUMENT: a
        function, a bound method's function, or a class's own methods -- which
        is how a callable instance's ``__call__`` is reached.
        """

        if isinstance(carrier, type):
            functions = class_functions(carrier)
        else:
            fn = getattr(carrier, "__func__", carrier)
            functions = [fn] if isinstance(fn, types.FunctionType) else []
        parts: list[str] = []
        for fn in functions:
            g = getattr(fn, "__globals__", None)
            if not isinstance(g, dict):
                continue
            owner = getattr(fn, "__qualname__", "?")
            for name in self._read_global_data_names(fn):
                if name not in g:
                    continue
                value = g[name]
                if isinstance(value, (types.ModuleType, type)):
                    continue
                if callable(value) and not isinstance(value, (dict, list, tuple, set)):
                    continue
                h = self._safe_global_hash(value, func_name, f"{owner}.{name}")
                if h is not None:
                    parts.append(f"argglobal:{owner}.{name}:{h}")
            for label, h in self._module_attr_parts(fn, func_name, g):
                parts.append(f"argglobal:{owner}:{label}:{h}")
        return parts
