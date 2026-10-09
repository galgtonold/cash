"""The digests of the values module data holds: a data global's value, the
data a callable carries besides its code, a function found inside data
identified by what calling it runs."""

from __future__ import annotations

import contextvars
import functools
import hashlib
import pickle
import types
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .._memo import CODE_OBJECTS, TABLES, LruMemo
from ..analysis.helper_code import callable_layers, is_mock, own_code_is_user, qualname_of
from ..analysis.purity_analyzer import get_analyzer
from ..analysis.purity_policy import REPORTED_METHODS
from ..dependency_state import SysModulesHelperResolver
from ..exceptions import CashImpurityWarning
from .code_tables import CodeTable
from .function_identity import hash_callable_source
from .key_values import carried_payload, held_partials, reduced_state, stabilize_for_global_hash
from .user_code import is_cash_wrapper

if TYPE_CHECKING:
    from .arg_hashing import ArgHasher
    from .closure_fold import HelperIdentity
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


#: ``(kind, id(value)) -> (value, digest)`` for the values hashed during one
#: key build; None outside one. Set by `KeyBuilder.build` for the length of
#: the build: nothing runs between two reads of a value there that could
#: change it, so a module constant every helper of another module reads
#: (``hm3.C1`` reached through fifty call sites) is hashed once, not once per
#: call site, on every hit. Dropped when the build ends, so the next build --
#: and the check after the call -- hashes the value as it is then. The value
#: is held, so its id is not reused while the entry stands.
KEY_BUILD_DIGESTS: contextvars.ContextVar[dict | None] = contextvars.ContextVar("_cash_key_build_digests", default=None)


class GlobalValues:
    """Digests of the values module data holds, with the memos that keep an
    unchanged value from being hashed again."""

    def __init__(self, args: ArgHasher, helpers: HelperIdentity, notices: Notices) -> None:
        self._args = args
        self._helpers = helpers
        self._notices = notices
        # The same live re-resolution the state hasher does, for functions
        # found inside data globals (`data_callable_identity`).
        self._data_helper_resolver = SysModulesHelperResolver(helpers.identity)
        self._carrier_verdicts: LruMemo[int, tuple[Any, bool | str]] = LruMemo(CODE_OBJECTS)
        # id(value) -> (value, digest) for immutable plain data globals; see
        # `global_value_digest`. The value is held, so its id is not reused.
        self._immutable_digests: LruMemo[int, tuple[Any, str]] = LruMemo(CODE_OBJECTS)
        # (kind, id(value)) -> (its `CodeTable` snapshot, digest) for a
        # table of plain functions; see `table_digest`.
        self._table_digests: LruMemo[tuple[str, int], tuple[CodeTable, str]] = LruMemo(TABLES)

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
        if isinstance(fn, type):
            # A class: its code, what it inherits and its metaclass.
            return self._helpers.identity(fn)
        if not isinstance(fn, types.FunctionType):
            return hash_callable_source(fn)
        own = self._helpers.identity(fn)

        if not own_code_is_user(fn, getattr(fn, "__module__", None)):
            return own
        try:
            report = get_analyzer().analyze_reached(fn)
        except (OSError, TypeError, SyntaxError, RecursionError):
            return own
        if not report.helper_source_hashes:
            return own
        live = self._data_helper_resolver.current_hashes(report)
        helpers = ",".join(f"{q}={live.get(q, h)}" for q, h in sorted(report.helper_source_hashes.items()))
        return hashlib.sha256(f"{own}|{helpers}".encode("utf-8")).hexdigest()

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
            return self.data_digest(value)
        args = self._args
        memo = plain == "immutable" and not (args.override_hashers or args.type_hashers)
        if memo:
            entry = self._immutable_digests.get(id(value))
            if entry is not None and entry[0] is value:
                return entry[1]
        build = KEY_BUILD_DIGESTS.get()
        if build is not None:
            entry = build.get(("plain", id(value)))
            if entry is not None and entry[0] is value:
                return entry[1]
        digest = args.plain_value_digest(value)
        if digest is None:
            digest = args.hash_payload((value,), {})
        if memo:
            self._immutable_digests[id(value)] = (value, digest)
        if build is not None:
            build[("plain", id(value))] = (value, digest)
        return digest

    def data_digest(self, value: Any) -> str:
        """The digest of a data value that may hold code: each function in it
        by what calling it runs (`data_callable_identity`). Kept per version
        of a table of plain functions (`table_digest`), and for the rest of
        one key build (`KEY_BUILD_DIGESTS`)."""
        build = KEY_BUILD_DIGESTS.get()
        if build is not None:
            entry = build.get(("data", id(value)))
            if entry is not None and entry[0] is value:
                return entry[1]
        digest = self.table_digest(
            "data",
            value,
            lambda: self._args.hash_payload((stabilize_for_global_hash(value, self.data_callable_identity),), {}),
        )
        if build is not None:
            build[("data", id(value))] = (value, digest)
        return digest

    def table_digest(self, kind: str, value: Any, compute: Callable[[], str]) -> str:
        """*compute*'s digest of *value*, kept while *value* is a table of
        plain functions whose `CodeTable` snapshot holds: a dispatch table of
        1000 functions read on every hit re-identified each one every time.
        Not with a hasher registered, which may answer differently for the
        same objects, nor for functions that call a helper, whose identity
        is looked up again on every call."""
        args = self._args
        memo = not (args.override_hashers or args.type_hashers)
        if memo:
            entry = self._table_digests.get((kind, id(value)))
            if entry is not None and entry[0].holds(value):
                return entry[1]
        digest = compute()
        if memo:
            table = CodeTable.of(value)
            if table is not None and all(self._calls_no_helper(fn) for fn in table.functions):
                self._table_digests[(kind, id(value))] = (table, digest)
        return digest

    @staticmethod
    def _calls_no_helper(fn: Any) -> bool:
        if not own_code_is_user(fn, getattr(fn, "__module__", None)):
            return True
        try:
            report = get_analyzer().analyze_reached(fn)
        except Exception:  # noqa: BLE001 - not kept, built every time
            return False
        return not (
            set(report.helper_source_hashes) - {qualname_of(fn)} or report.helper_bindings or report.cached_callees
        )

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

    def safe_global_hash(self, value: Any, func_name: str, label: str) -> str | None:
        """Hash *value* for the key, warning once and skipping if it cannot be."""
        try:
            return self.data_digest(value)
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
