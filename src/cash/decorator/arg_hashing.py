"""How a call's arguments become the args segment of its key."""

from __future__ import annotations

import functools
import hashlib
import inspect
import logging
import pickle
import sys
import threading
import types
import weakref
from typing import Any

from .. import _plain_data
from .._clock import perf_counter as _perf_counter
from ..exceptions import CashCacheIneffectiveWarning
from ..lineage_tag import own_tag
from ..object_hashing import builtin_hash, stable_key_repr
from ..value_types import BUILTIN_CONTAINERS, CODELESS_PRIMS, IMMUTABLE_PRIMS, PLAIN_SEQS

logger = logging.getLogger(__name__)

#: A census taken while one cache key is built, shared by the code fold and the
#: argument hash so a big argument is looked at once (`plain_census`). None
#: outside a key build: after the body has run, an argument may have changed.
PLAIN_CENSUS = threading.local()


def plain_census(value: Any) -> tuple[str, Any] | None:
    """What kind of plain data *value* is, memoized for the key build in progress.

    ``("plain", value)`` for lists and tuples of primitives (`_plain_data.is_plain`),
    ``("dict_rows", (keys, rows))`` for a list of dicts sharing their keys
    (`_plain_data.dict_rows`), None for anything else.
    """
    memo = getattr(PLAIN_CENSUS, "memo", None)
    if memo is not None:
        hit = memo.get(id(value))
        if hit is not None and hit[0] is value:
            return hit[1]
    found: tuple[str, Any] | None = None
    if _plain_data.is_plain(value):
        found = ("plain", value)
    else:
        rows = _plain_data.dict_rows(value)
        if rows is not None:
            found = ("dict_rows", rows)
    if memo is not None:
        memo[id(value)] = (value, found)
    return found


def plain_key_part(value: Any) -> Any:
    """*value*, or -- for plain data -- a marker holding the digest of its content.

    Each plain argument is keyed by its content on its own, pickled without the
    memo (`_plain_data.pickle_unshared`), so one small dict beside two million
    rows does not send the rows down the general path.
    """
    if type(value) not in PLAIN_SEQS:
        return value
    census = plain_census(value)
    if census is None:
        return value
    kind, data = census
    return (f"__cash_{kind}__", hashlib.sha256(_plain_data.pickle_unshared(data)).hexdigest())


#: Values whose identity is code plus what it captures. A hasher registered for
#: one of these types covers every such value in the process, and the obvious
#: one -- by name -- gives every closure one factory makes the same identity.
#: `Cash._first_unhashable_arg` found only built-in-typed arguments.
NO_SUSPECT = object()


CODE_VALUE_TYPES = (types.FunctionType, types.MethodType, functools.partial)


#: The fix for an unhashable code value. It must NOT suggest
#: `register_hasher(function, ...)`: following that advice is how a second
#: closure got the first one's result.
CODE_ARG_FIX = (
    "pass a module-level function in its place, and give the values it "
    "captures to the cached function as plain arguments, where they reach the "
    "key. Do not register a hasher for function: every closure one factory "
    "makes shares a name, so a hasher keyed on it hands one closure's result "
    "to another. See known-limitations.md, 'A closure or lambda passed as an "
    "argument'."
)


def unhashable_arg_fix(value: Any, type_name: str) -> str:
    """The fix line for an argument of *type_name* that could not be hashed."""
    if isinstance(value, CODE_VALUE_TYPES):
        return CODE_ARG_FIX
    return f"register a hasher with cash.register_hasher({type_name}, ...), or pass the argument by a hashable value."


#: Who wrote a value's ``_cash_lineage_hash``, in ``_cash_lineage_src``. Only the
#: notebook's statement layer keeps the tag current as the value changes, so
#: only its tag stands in for the value's content (see `_hash_arg_payload`).
LINEAGE_SRC_STATEMENT = "statement"
LINEAGE_SRC_DECORATOR = "decorator"
#: Written for a function decorated ``frozen=True``: the user's promise that the
#: result is not modified afterwards, trusted like the statement layer's tag and
#: audited now and then (`_audit_frozen`).
LINEAGE_SRC_FROZEN = "frozen"


_COW_PANDAS: bool | None = None


#: The costliest argument of the key most recently hashed on this thread:
#: ``(label, seconds, type name, producer, pandas without copy-on-write)``.
#: A description, never the value: a reference here would keep a large
#: argument alive after its caller dropped it.
ARG_COST = threading.local()


def is_cow_pandas(value: Any) -> bool:
    """Is *value* a pandas DataFrame/Series under copy-on-write?

    Copy-on-write is the only mode in pandas 3 and opt-in before. Checked
    without importing pandas: a pandas object means it is already loaded.
    """
    global _COW_PANDAS
    t = type(value)
    if t.__name__ not in ("DataFrame", "Series") or not (t.__module__ or "").startswith("pandas"):
        return False
    if _COW_PANDAS is None:
        try:
            import pandas as pd

            major = int(pd.__version__.split(".", 1)[0])
            _COW_PANDAS = major >= 3 or pd.options.mode.copy_on_write is True
        except Exception:  # noqa: BLE001 - unknown pandas: no memo, hash every time
            _COW_PANDAS = False
    return _COW_PANDAS


ARG_HASH_MEMO_CAP = 1024


FRAME_MEMO_CAP = 256


def frame_signature(obj: Any) -> tuple:
    """What must stay the same for a pandas object's content hash to hold.

    Under copy-on-write, a frame whose data another frame also references
    cannot be written in place: every write path (``loc``/``iloc``/``at``,
    column assignment, ``inplace=True`` methods, ``update``, ``insert``,
    ``pop``) first gives the written frame NEW block arrays, and writes
    through ``.values`` / ``to_numpy()`` raise (the arrays are read-only).
    So the identities of the block arrays, the manager and the axes are an
    exact change signal -- measured on 17 mutation forms, pandas 3.0.3. The
    axis NAMES are compared by value, because ``df.index.name = ...``
    renames the same Index object and the content hash includes them.
    """
    mgr = obj._mgr
    blocks = tuple(id(block.values) for block in mgr.blocks)
    if hasattr(obj, "columns"):
        return (id(mgr), blocks, id(obj.columns), tuple(obj.columns.names), id(obj.index), tuple(obj.index.names))
    return (id(mgr), blocks, id(obj.index), tuple(obj.index.names), obj.name)


def frame_borrows_its_data(obj: Any, held: Any = None) -> bool:
    """Whether *obj*'s blocks sit on memory something else may write.

    Copy-on-write is what makes the block identities an exact change
    signal, and it only governs writes through PANDAS. ``pd.DataFrame(arr,
    copy=False)`` keeps the caller's ndarray, and ``arr[0, 0] = 100`` goes
    straight past pandas: same blocks, changed data. The memo answered 10.0
    where the frame really summed to 109.0. Such a frame is re-hashed on every call.

    *held* is the memo's own shallow copy of *obj*. Its blocks are views
    whose ``base`` is *obj*'s array, one reference each. Those references
    are cash's, not an outside writer's, so they are not counted against
    the baseline; counting them made every memoised frame look borrowed,
    and it was re-hashed on every call.
    """
    try:
        ours = _held_block_refs(held) if held is not None else {}
        for block in obj._mgr.blocks:
            # Counted before this loop binds the array to a name of its
            # own, exactly as the baseline was measured.
            refcount = _block_refcount(block)
            values = block.values
            base = getattr(values, "base", None)
            if base is not None or not getattr(getattr(values, "flags", None), "owndata", True):
                return True
            # A 1-D block IS the caller's array (`pd.Series(arr,
            # copy=False)`), with no base and owning its data -- only the
            # extra reference the caller still holds tells them apart. A
            # count above the baseline can only make cash re-hash a frame
            # it could have memoised: slower, never wrong.
            if refcount > _block_refcount_baseline() + ours.get(id(values), 0):
                return True
            del values, base
    except Exception:  # noqa: BLE001 - a pandas internals change: re-hash, the safe answer
        return True
    return False


def _held_block_refs(held: Any) -> dict[int, int]:
    """``{id(array): n}``: the references *held*'s blocks keep to arrays.

    A function of its own so that no loop variable outlives it: one left
    pointing at an array would itself be a reference over the baseline.
    """
    refs: dict[int, int] = {}
    for block in held._mgr.blocks:
        values = block.values
        for ref in (values, getattr(values, "base", None)):
            if ref is not None:
                refs[id(ref)] = refs.get(id(ref), 0) + 1
    return refs


def _block_refcount(block: Any) -> int:
    """``sys.getrefcount`` of *block*'s array, taken the same way for the
    baseline and for every check."""
    return sys.getrefcount(block.values)


def _block_refcount_baseline() -> int:
    """What `_block_refcount` reads for an array only its block holds.

    Measured rather than written down: what ``sys.getrefcount`` counts
    besides the holders varies across Python versions (3.14 counts one
    fewer), and a baseline one too high lets a caller's array through
    as the frame's own -- the stale answer this check exists to stop.
    """
    global _BLOCK_REFCOUNT_BASELINE
    baseline = _BLOCK_REFCOUNT_BASELINE
    if baseline is None:
        import pandas as pd

        probe = pd.Series([0.0, 1.0, 2.0])
        baseline = _BLOCK_REFCOUNT_BASELINE = _block_refcount(probe._mgr.blocks[0])
    return baseline


#: See ``_block_refcount_baseline``; anything above it means something
#: outside can write to the array, see ``frame_borrows_its_data``.
_BLOCK_REFCOUNT_BASELINE: int | None = None


#: Types whose code must not participate in any cache key. Process-wide,
#: not per-instance: a marker is a property of the type, and a user who
#: marks it once should not have to repeat it per Cash instance.
#:
#: Holds STRONG references deliberately, so a registered class can never
#: be garbage collected. Considered and rejected a WeakSet: opaque types
#: are registered by hand, at import time, in the tens at most for any
#: real user -- not generated in volume -- so the leak this trades away
#: has no realistic scale to bite at. A WeakSet would also silently
#: un-register a type the moment nothing else references it, which is
#: the opposite of "mark it once and forget about it."
OPAQUE_TYPES: set = set()


def mark_opaque(*types_: type) -> None:
    """Exclude *types_* from code-surface hashing: what ``cash.opaque`` records."""
    OPAQUE_TYPES.update(types_)


def is_opaque(obj: Any) -> bool:
    """True when *obj* -- a class, or an instance of one -- must not have
    its code hashed into a cache key.

    The type itself must be in ``_OPAQUE_TYPES`` (``cash.opaque``); a
    subclass of an opaque class is not covered. It may carry its own
    freshly-written methods the user actively edits, and inheriting the
    mark would silently exempt that code from ever invalidating the cache.
    A subclass that wants the same treatment is marked itself (pinned by
    ``test_a_subclass_of_an_opaque_class_does_not_inherit_opacity``).

    Never raises. Measured, not assumed: a metaclass that defines
    ``__eq__`` without ``__hash__`` makes the CLASS ITSELF unhashable
    (Python's data-model default, not just its instances), so
    ``target in OPAQUE_TYPES`` can raise ``TypeError`` on a real,
    if unusual, class shape. An opacity check must not be the thing
    that breaks an otherwise-cacheable call.
    """
    try:
        if isinstance(obj, functools.partial):
            # A partial is the function it wraps plus arguments, both of
            # which are keyed now. `cash.opaque(functools.partial)` was the
            # old advice for silencing KEY-OPAQUE-CALLABLE, and it silenced
            # EVERY partial in the process, including ones over code the
            # user then edited.
            return False
        target = obj if isinstance(obj, type) else type(obj)
        return target in OPAQUE_TYPES
    except Exception as e:  # noqa: BLE001 - opacity check must never break a call
        logger.debug("[CORE] opacity check failed for %r: %s", obj, e)
        return False


class ArgHashingMixin:
    """Canonical arguments and their content hashes, with the memos that keep
    re-hashing an unchanged argument cheap."""

    def _warn_unhashable_args(self, func_name: str, args: tuple, kwargs: dict) -> None:
        """KEY-UNHASHABLE-ARG, naming the argument when one can be singled out."""
        arg_type_name = self._first_unhashable_arg_type(args, kwargs)
        if arg_type_name == "<unknown>":
            which = (
                "an argument could not be hashed, and cash cannot say which -- the value is nested inside a container"
            )
            suggestion = (
                "find the nested value, then register a hasher for its "
                "type with cash.register_hasher(SomeType, ...) or pass "
                "something hashable in its place."
            )
        else:
            which = f"an argument of type {arg_type_name} could not be hashed"
            suggestion = unhashable_arg_fix(self._first_unhashable_arg(args, kwargs), arg_type_name)
        self._notices.warn_once(
            CashCacheIneffectiveWarning,
            func_name,
            arg_type_name,
            f"@cash.cache on {func_name}: {which}, so this call and every call like it does not cache.",
            code="KEY-UNHASHABLE-ARG",
            fix=suggestion,
        )

    def _warn_key_build_failed(self, func_name: str, args: tuple, kwargs: dict, e: Exception) -> None:
        """KEY-BUILD-FAILED: a step of the key build raised where it did not expect to."""
        arg_type_name = self._first_unhashable_arg_type(args, kwargs)
        if arg_type_name == "<unknown>":
            hint = (
                "check the function's arguments -- cash could not identify "
                "the offending type; if the exception does not belong to "
                "your code, report it as a bug with the traceback."
            )
        elif isinstance(self._first_unhashable_arg(args, kwargs), CODE_VALUE_TYPES):
            hint = CODE_ARG_FIX
        else:
            hint = (
                f"register a hasher with "
                f"cash.register_hasher({arg_type_name}, ...) if "
                f"{arg_type_name} is the unhashable argument."
            )
        self._notices.warn_once(
            CashCacheIneffectiveWarning,
            func_name,
            arg_type_name,
            f"@cash.cache on {func_name}: cache-key generation raised "
            f"{type(e).__name__} ({e}) somewhere it did not anticipate, so "
            f"this call does not cache.",
            code="KEY-BUILD-FAILED",
            fix=hint,
        )

    def _first_unhashable_arg_type(self, args: tuple, kwargs: dict) -> str:
        """Return the qualname of the argument that could not be hashed, or '<unknown>'.

        Used to attribute CashCacheIneffectiveWarning to a concrete type name
        so the user knows which register_hasher() call to add. See
        `_first_unhashable_arg` for how the argument is found.
        """
        suspect = self._first_unhashable_arg(args, kwargs)
        return "<unknown>" if suspect is NO_SUSPECT else type(suspect).__qualname__

    def _first_unhashable_arg(self, args: tuple, kwargs: dict) -> Any:
        """The argument that could not be hashed, or ``NO_SUSPECT``.

        Each candidate is hashed ALONE and the first that fails is named, so
        ``score(df, lambda d: d * 2)`` blames the lambda, not the DataFrame
        (whose hasher cash rejects, and which with override=True would re-key
        every DataFrame function). This runs only on the failure path. Strings,
        numbers, None and built-in containers are skipped: a scalar always
        hashes, and a container holding the culprit is reported as "nested",
        which says more than naming the list. When no single candidate fails
        on its own, the first non-built-in is the best remaining guess.
        """
        candidates = [a for a in (*args, *kwargs.values()) if not isinstance(a, IMMUTABLE_PRIMS + BUILTIN_CONTAINERS)]
        for candidate in candidates:
            try:
                self._hash_arg_payload((candidate,), {})
            except Exception:  # noqa: BLE001 - exactly what we are looking for
                return candidate
        return candidates[0] if candidates else NO_SUSPECT

    def _normalize_call_args(
        self,
        func_name: str,
        args: tuple,
        kwargs: dict,
    ) -> tuple[tuple, dict]:
        """Bind ``(args, kwargs)`` to the function signature and apply defaults.

        Collapses logically-identical calls written in different forms - ``f(1)``
        vs ``f(1, y=10)`` (the default) vs ``f(x=1, y=10)``, and kwargs in any
        order - to one canonical argument shape so they share a cache key
        instead of producing wasteful misses.

        Best-effort: any introspection or bind failure (builtins with no
        signature, ``*args`` calls that don't match, deliberately mismatched
        calls) returns the inputs unchanged, so behavior never regresses.
        """
        # Read once per decoration: a notebook cell re-run with an edited
        # default gets a new `CachedFunction`, and with it the new signature.
        cf = self._cached.get(func_name)
        sig = cf.signature if cf is not None else None
        if sig is None:
            return args, kwargs
        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
        except TypeError:
            # The call doesn't match the signature (the function itself would
            # raise when invoked). Leave the raw form untouched.
            return args, kwargs
        # ``bound.arguments`` is ordered by parameter definition, so the result
        # is canonical regardless of how the caller wrote the call. Re-express
        # named params as kwargs; keep *args positional; sort **kwargs so its
        # order doesn't leak into the key. (We only build a payload to hash, so
        # routing named params through kwargs is purely for determinism.)
        canon_args: list[Any] = []
        canon_kwargs: dict[str, Any] = {}
        for name, param in sig.parameters.items():
            if name not in bound.arguments:
                continue
            val = bound.arguments[name]
            if param.kind is inspect.Parameter.VAR_POSITIONAL:
                canon_args.extend(val)
            elif param.kind is inspect.Parameter.VAR_KEYWORD:
                # Under its own name: a `**kwargs` entry may be called after a
                # parameter, and writing both into one dict let it overwrite
                # that parameter's value. `def request(url, /, **params)`
                # called as `request("/a", url="x")` then keyed on the kwargs
                # `url` alone, so every such call shared one entry and
                # `request("/b", url="x")` was served `GET /a`.
                for k in sorted(val):
                    canon_kwargs[f"{name}:{k}"] = val[k]
            else:
                canon_kwargs[name] = val
        return tuple(canon_args), canon_kwargs

    def _memo_arg_hash(self, arg: Any, lineage: str, content_hash: str) -> None:
        """Record ``id(arg) -> (weakref, lineage, content_hash)`` for the session,
        bounded so a long session can't grow the memo without limit. When full,
        drop it wholesale: the memo is a pure speedup, so an occasional cold
        start just re-hashes. Values that cannot be weak-referenced are skipped
        (they simply keep full-hashing).
        """
        try:
            wref = weakref.ref(arg)
        except TypeError:
            return
        memo = self._arg_hash_memo
        if len(memo) >= ARG_HASH_MEMO_CAP:
            memo.clear()
        memo[id(arg)] = (wref, lineage, content_hash)

    def _frame_memo_lookup(self, obj: Any) -> str | None:
        """The content hash recorded for *obj*, if *obj* has not changed since."""
        entry = self._frame_memo.get(id(obj))
        if entry is None:
            return None
        wref, held, signature, content_hash = entry
        if frame_borrows_its_data(obj, held):
            self._frame_memo.pop(id(obj), None)
            return None
        try:
            if wref() is obj and frame_signature(obj) == signature:
                return content_hash
        except Exception:  # noqa: BLE001 - a pandas internals change: just re-hash
            pass
        self._frame_memo.pop(id(obj), None)
        return None

    def _frame_memo_store(self, obj: Any, content_hash: str) -> None:
        """Remember *obj*'s content hash, and hold a shallow copy of it.

        The shallow copy shares the data and is what makes the signature
        exact: while cash references the blocks, pandas must copy before any
        write. Cost: the first in-place write to each block afterwards copies
        that block, once. The entry, copy included, goes when *obj* is
        collected, or when the memo fills.
        """
        try:
            held = obj.copy(deep=False)
            signature = frame_signature(obj)
            memo = self._frame_memo
            key = id(obj)
            wref = weakref.ref(obj, lambda _ref, key=key, memo=memo: memo.pop(key, None))
        except Exception:  # noqa: BLE001 - the memo is a speedup; hash every time
            return
        if len(self._frame_memo) >= FRAME_MEMO_CAP:
            self._frame_memo.clear()
        self._frame_memo[key] = (wref, held, signature, content_hash)

    def _hash_arg_payload(self, args: tuple, kwargs: dict) -> str:
        """Hash one concrete ``(args, kwargs)`` form. May raise on unpicklable
        values; the caller decides whether to retry with a different form."""

        def get_arg_hash(arg):
            # Content-authoritative builtin hashers FIRST. pandas /
            # numpy / polars / pyarrow / modin / dask hash the argument's
            # *content*, which is byte-stable across processes and kernel
            # restarts. The notebook's in-memory ``_cash_lineage_hash`` (checked
            # next) is recomputed per session and is NOT reproducible across a
            # restart -- keying a persisted @cash.cache entry on it makes the
            # decorator miss after a restart even though the argument is
            # byte-identical (re-training the model the docs promise survives a
            # restart). A value that has a content hash must key on content so
            # the entry survives; the modest extra hashing cost is the price of
            # the flagship "restart-and-run-all in seconds" guarantee. Mirrors
            # principle: the reproducible signal, not the volatile
            # in-memory one, is authoritative.
            # Fast path: skip re-hashing a possibly-huge argument we already
            # content-hashed this session, when it is provably the SAME,
            # unmutated object. Keyed on ``id`` (NOT lineage): two *different*
            # objects that happen to share a lineage string must still be
            # distinguished by content -- an explicit invariant
            # (test_arg_hash_restart_stable) -- and distinct live objects have
            # distinct ids. The entry is validated on read by BOTH a weakref
            # identity check (guards id reuse after GC) AND the object's
            # ``_cash_lineage_hash`` being unchanged (cash's own mutation signal,
            # the same one it trusts to cache every notebook statement). The
            # stored value is still the reproducible content hash, so the cache
            # key is byte-identical and restart-safe; the memo is a pure
            # within-session speedup, empty after a restart.
            #
            # Trusted only where something KEEPS it current: the notebook's
            # statement layer re-tags a variable on every assignment and
            # mutation. The decorator also tags what it returns, and nothing
            # ever moves that tag -- in a script, `q.F = 0.03; run(q)` or
            # `df.loc[0, "a"] = 100` left it as it was, and both the memo below
            # and the tag-as-identity shortcut further down served the result
            # for the unmutated object.
            # The instance's OWN tag: one inherited from a tagged class made
            # every instance key alike (see cash.lineage_tag).
            lineage = own_tag(arg)
            if lineage is not None:
                src = own_tag(arg, "_cash_lineage_src")
                if src == LINEAGE_SRC_FROZEN:
                    if not self._audit_frozen(arg):
                        lineage = None
                elif src != LINEAGE_SRC_STATEMENT:
                    lineage = None
            if self._frozen_arrays and id(arg) in self._frozen_arrays:
                frozen_hash = self._frozen_array_hash(arg)
                if frozen_hash is not None:
                    return frozen_hash
            if self._frozen_containers and id(arg) in self._frozen_containers:
                frozen_hash = self._frozen_container_hash(arg)
                if frozen_hash is not None:
                    return frozen_hash
            if lineage is not None:
                entry = self._arg_hash_memo.get(id(arg))
                if entry is not None:
                    wref, memo_lineage, content_hash = entry
                    if memo_lineage == lineage and wref() is arg:
                        return content_hash
            # pandas >= 3 copy-on-write: an exact "has this frame changed?"
            # check instead of a trusted tag. See `_frame_memo_lookup`.
            frame_memo = lineage is None and is_cow_pandas(arg)
            if frame_memo:
                content_hash = self._frame_memo_lookup(arg)
                if content_hash is not None:
                    return content_hash

            # Overriding hashers, ahead of everything cash would do itself.
            # The user has said their identity for this type beats content
            # hashing, which is the only way to stop re-reading a 800MB array
            # on every call. Guarded by the emptiness check so the ordinary
            # case pays one dict truth test, not a loop.
            if self._override_hashers:
                for type_, (hasher_fn, src_hash) in self._override_hashers.items():
                    if isinstance(arg, type_):
                        return f"{src_hash}:{hasher_fn(arg)}"

            content_digest = builtin_hash(arg)
            if content_digest is not None:
                if lineage is not None:
                    self._memo_arg_hash(arg, lineage, content_digest)
                elif frame_memo:
                    self._frame_memo_store(arg, content_digest)
                return content_digest
            # Notebook lineage hash: the authoritative, cheap identity for
            # values that carry NO content hasher (custom objects). Kept ahead
            # of registered hashers so a lineage-carrying object short-circuits
            # its (possibly expensive) registered hasher within a session
            # (test_hasher_priority_cash_hash_first).
            if lineage is not None:
                return lineage
            for type_, (hasher_fn, src_hash) in self._type_hashers.items():
                if isinstance(arg, type_):
                    # Embed the hasher source hash so that changing the
                    # hasher's body invalidates dependent cache entries
                    # even when the hasher's output coincidentally matches.
                    return f"{src_hash}:{hasher_fn(arg)}"
            return arg

        # Timed per argument -- two clock reads each -- so that a
        # CACHE-NET-LOSS verdict can name the argument that costs the time.
        costliest: tuple | None = None

        def timed(label: str, value: Any) -> Any:
            nonlocal costliest
            t0 = _perf_counter()
            digest = get_arg_hash(value)
            seconds = _perf_counter() - t0
            if costliest is None or seconds > costliest[1]:
                producer = getattr(value, "_cash_lineage_producer", None)
                if producer is None and self._frozen_arrays and id(value) in self._frozen_arrays:
                    producer = self._frozen_arrays[id(value)][1]
                if producer is None and self._frozen_containers and id(value) in self._frozen_containers:
                    producer = self._frozen_containers[id(value)][1]
                old_pandas = (
                    type(value).__name__ in ("DataFrame", "Series")
                    and (type(value).__module__ or "").startswith("pandas")
                    and not is_cow_pandas(value)
                )
                costliest = (label, seconds, type(value).__name__, producer, old_pandas)
            return digest

        hashed_args = tuple(timed(f"#{i}", a) for i, a in enumerate(args))
        hashed_kwargs = {k: timed(k, v) for k, v in kwargs.items()}
        # An argument with no hasher of its own goes into the payload AS IS,
        # and its cost is the walk and the pickle below, not the lookup timed
        # above -- so CACHE-NET-LOSS named a 2M-row list as taking "about 0ms
        # to hash". The payload's time is charged to the largest
        # such argument.
        raw = [
            (label, value)
            for (label, value), digest in zip(
                [(f"#{i}", a) for i, a in enumerate(args)] + list(kwargs.items()),
                list(hashed_args) + list(hashed_kwargs.values()),
            )
            if digest is value and type(value) not in CODELESS_PRIMS
        ]
        payload_t0 = _perf_counter()

        # One canonical form (`stable_key_repr`): sets and dicts in a stable
        # order, every container tagged with its type.
        payload = stable_key_repr(
            (tuple(map(plain_key_part, hashed_args)), {k: plain_key_part(v) for k, v in hashed_kwargs.items()})
        )
        args_bytes = _plain_data.key_dumps(payload)
        if raw:
            payload_seconds = _perf_counter() - payload_t0
            if costliest is None or payload_seconds > costliest[1]:
                label, value = max(raw, key=lambda r: len(r[1]) if hasattr(r[1], "__len__") else sys.getsizeof(r[1]))
                producer = getattr(value, "_cash_lineage_producer", None)
                if producer is None and self._frozen_containers and id(value) in self._frozen_containers:
                    producer = self._frozen_containers[id(value)][1]
                costliest = (label, payload_seconds, type(value).__name__, producer, False)
        ARG_COST.last = costliest
        return hashlib.sha256(args_bytes).hexdigest()

    def _serialize_args(
        self, func_name: str, args: tuple, kwargs: dict, normalized: tuple[tuple, dict] | None = None
    ) -> str | None:
        """Hash the arguments, canonicalised.

        *normalized* lets a caller that has ALREADY canonicalised pass the
        result in rather than have it recomputed. That is not an optimisation:
        the code channel (`_fold_code_args`) and this value channel must key
        off the SAME bound arguments, or `f()` and `f(<the default>)` -- the
        same logical call -- disagree in one channel and split into two cache
        entries. One canonicalisation, shared, is the only way that invariant
        holds by construction rather than by two call sites staying in step.
        """
        if normalized is None:
            normalized = self._normalize_call_args(func_name, args, kwargs)
        try:
            return self._hash_arg_payload(*normalized)
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError) as e:
            # Normalization can fold a default value into the payload (so
            # f(1) keys identically to f(1, y=<default>)). If that default is
            # unpicklable it must not make a call that hashed fine before stop
            # caching - retry with the raw, un-normalized form first.
            if normalized[0] is not args or normalized[1] is not kwargs:
                try:
                    return self._hash_arg_payload(args, kwargs)
                except (TypeError, pickle.PicklingError, AttributeError, OverflowError):
                    pass
            # Pickle failure here is surfaced via CashCacheIneffectiveWarning in
            # _resolve_cache_key (which sees the None return). Keep this log at
            # debug level so it's available when explicitly enabled but doesn't
            # double-warn.
            logger.debug("Could not serialize arguments for %s: %s", func_name, e)
            return None

    def _note_arg_cost(self, func_name: str) -> None:
        """Keep the costliest argument to hash seen for *func_name*.

        Only its description is kept -- parameter, type, seconds, the cached
        function that produced it -- never the value, which may be large.
        """
        cost = getattr(ARG_COST, "last", None)
        ARG_COST.last = None
        if cost is None:
            return
        label, seconds, type_name, producer, old_pandas = cost
        cf = self._cached.get(func_name)
        if cf is None or (cf.arg_cost is not None and cf.arg_cost[2] >= seconds):
            return
        cf.arg_cost = (label, type_name, seconds, producer, old_pandas)
