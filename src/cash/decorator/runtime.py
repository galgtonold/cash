"""One cached call, from its key to its result: build the key, look it up,
run the body on a miss, and hand back what to return."""

from __future__ import annotations

import concurrent.futures
import contextlib
import contextvars
import hashlib
import inspect
import itertools
import logging
import operator
import os
import pickle
import re
import sys
import threading
import time
import types
from collections.abc import Callable, Collection, Iterator
from typing import TYPE_CHECKING, Any, NamedTuple

from .. import _weak, argument_sharing, rewrite_in_place
from .. import phase_times as _phases
from .._active import EXPLAINING as _EXPLAINING
from .._clock import perf_counter as _perf_counter
from .._type_memo import class_dictoffset, class_name, class_qualname
from .._user_attrs import plain_getattr, unwrapped
from ..analysis.handed_callables import hands_a_state_changing_callable
from ..analysis.helper_walk import shared_nodes
from ..analysis.out_arguments import attribute_bound_parameter, mutated_parameter
from ..analysis.purity_report import UNREAD_MODULE, PurityReport, describe_untrackable, unread_module
from ..backends._base import ttl_expired, written_at
from ..canonical_form import WALKED_INDEXES
from ..code_swaps import FOLDED_CODE, FoldedCode, function_sets, note_owner
from ..dependency_state import STATE_LEDGER, ledger_note
from ..effect_observer import WaiverTrace, code_carries_waiver
from ..exceptions import (
    CashCacheIneffectiveWarning,
    CashCacheStoreFailedWarning,
    CashImpureFunctionError,
    CashImpurityWarning,
)
from ..handed_digests import Stored
from ..handed_digests import claim as claim_handed_digests
from ..loaded_code import code_objects_of
from ..mutation_fingerprint import mutation_fingerprint
from ..pint_values import BUILD_DIGESTS as PINT_DIGESTS
from ..tracking import process_registries
from ..tracking.env_reads import environment_digests, environment_unchanged, note_reads
from ..tracking.env_reads import recording as recording_env_reads
from ..tracking.file_tracker import FileAccessTracker
from ..tracking.randomness import (
    capture_argument_carrier_states,
    capture_reachable_carrier_states,
    carriers_put_back,
    moved_carriers,
    replay_argument_carriers,
    replayable,
)
from ..tracking.trace_recorder import other_thread_alive
from . import attribute_replay, class_memo, code_trace, hit_memo, library_reads, state_memo, walk_scope, warning_state
from .arg_hashing import (
    CALL_POLARS,
    CONTENT_KEYED,
    HANDED_DIGESTS,
    KEYED_PARTS,
    PLAIN_CENSUS,
    CallPolarsDigests,
    ContentKeyed,
    call_polars,
    content_only,
    quiet_digests,
)
from .arg_key import keyed_arguments
from .cache_metadata import CacheMetadata
from .cached_function import CachedFunction
from .call_state import (
    CACHE_MISS,
    CAPTURE_WATCH,
    CARRIER_LOOKUPS,
    HELD_BY_NAME,
    NESTED_CASH_SECONDS,
    REACHED_TTLS,
    THREADS_IN_CALLS,
    BodyRun,
    BuiltKey,
    Call,
    KeyBuildFailed,
    UnhashableArgs,
    UnhashableDefault,
    run_to_completion,
)
from .class_data import CLASSES_FOLDED, getter_fill_mark, note_getter_fills
from .closure_fold import EVERY_PARAMETER, bound_owner, bound_values, defaults_of
from .dynamic_sources import (
    Resolution,
    dynamic_sources_fresh,
    entry_resolutions,
    held_sources,
    recorded_sources,
)
from .explain import MissKind, MissReason, describe_stale_files
from .file_deps import (
    note_unentered_body,
    note_unresolved_dynamic,
    pass_dynamic_sources_up,
    propagate_file_deps_to_active_tracker,
    snapshot_tracked_deps,
)
from .function_identity import func_key, hash_callable_source
from .global_values import KEY_BUILD_DIGESTS
from .globals_fold import READS_FOLDED
from .iterators import ChunkedCachedIterator, StreamingCachedIterator, chunk_prefix, is_one_shot_iterator
from .refusal_verdicts import ArgumentRefusal, RefusalVerdicts
from .registry import resolve_dynamic_dependencies
from .rng import capture_rng_pre_state, replay_rng_state
from .store import StoreRequest

if TYPE_CHECKING:
    from ..dependency_state import DependencyStateHasher
    from .arg_hashing import ArgHasher
    from .backend_slot import BackendSlot
    from .closure_fold import ClosureFold
    from .code_args import CodeArgs
    from .environment_fold import EnvironmentFold
    from .explain import MissHistory
    from .file_deps import FileDeps
    from .function_identity import OwnSourcePins
    from .globals_fold import GlobalsFold
    from .method_deps import MethodClassDeps
    from .purity_checks import PurityChecks
    from .registry import FunctionRegistry
    from .reporting import CallLog, Notices
    from .rng import RngWatch
    from .store import ResultStore

logger = logging.getLogger(__name__)


#: Cached functions whose `KeyBuilder.callee_state` is being built on this
#: thread, so one that reaches itself folds its name instead of recursing.
CALLEE_STATES: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "_cash_callee_states", default=frozenset()
)


class Unkeyable(NamedTuple):
    """A call that gets no key, and the miss it is logged as."""

    reason: MissReason


_UNHASHABLE = MissReason(MissKind.UNHASHABLE, "an argument could not be hashed, so there is no key to look up")
_KEY_FAILED = MissReason(MissKind.KEY_FAILED, "building the key raised")
_HANDED = MissReason(
    MissKind.HANDED,
    "an argument is a function of yours, or holds one, that changes state when called, so the call ran uncached",
)


def _unhashable_callee_default(func_name: str) -> KeyBuildFailed:
    """The failure of a caller's key when cached *func_name*'s default cannot be hashed."""
    return KeyBuildFailed(
        "KEY-UNHASHABLE-DEFAULT",
        f"@cash.cache on {func_name}: a parameter default of this cached function, "
        f"which another cached function calls, could not be hashed, so the caller "
        f"cannot tell whether it changed and ran uncached.",
        "give the default a hashable value, or register a hasher for its type with cash.register_hasher.",
    )


def _iterator_position(value: Any) -> Any:
    """Where the one-shot iterator *value* stands, comparable before and
    after something read it; None when that cannot be told."""
    if inspect.isgenerator(value):
        frame = value.gi_frame
        return (inspect.getgeneratorstate(value), frame.f_lasti if frame is not None else None)
    try:
        hint = operator.length_hint(value, -1)
    except Exception:  # noqa: BLE001 - a user's __length_hint__ can raise anything
        return None
    return ("left", hint) if hint >= 0 else None


def iterator_arguments(signature: Any, args: tuple, kwargs: dict) -> list[tuple[str, Any, Any]]:
    """``(parameter, iterator, position)`` for each one-shot iterator argument."""
    if not any(is_one_shot_iterator(v) for v in (*args, *kwargs.values())):
        return []
    named: list[tuple[str, Any]] = [(f"argument {i}", v) for i, v in enumerate(args)] + list(kwargs.items())
    if signature is not None:
        try:
            bound = signature.bind(*args, **kwargs).arguments
        except TypeError:
            bound = None
        if bound is not None:
            named = []
            for name, value in bound.items():
                kind = signature.parameters[name].kind
                if kind is inspect.Parameter.VAR_POSITIONAL:
                    named.extend((f"*{name}[{i}]", v) for i, v in enumerate(value))
                elif kind is inspect.Parameter.VAR_KEYWORD:
                    named.extend(value.items())
                else:
                    named.append((name, value))
    return [(label, v, _iterator_position(v)) for label, v in named if is_one_shot_iterator(v)]


#: The most items of a generator argument a registered hasher reads that
#: are taken out to hand the hasher and the body a copy each
#: (`split_for_key`). Past it the call runs uncached, the body getting every
#: item still.
MATERIALISED_ITEMS = 100_000


class _KeyCopies(NamedTuple):
    """The iterator arguments of a call whose key would read them, each
    split in two (`split_for_key`): the key reads one copy, the body gets
    the other with every item, as in plain."""

    args: tuple
    kwargs: dict
    # (position or name, the caller's iterator, where it stood, the body's copy)
    splits: list[tuple[int | str, Any, Any, Any]]
    #: Set when a generator was longer than `MATERIALISED_ITEMS`: the key
    #: cannot be built without reading it all, so the call runs uncached.
    refused: KeyBuildFailed | None = None

    def body_arguments(self, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        """The arguments the body runs with: the caller's own iterator
        when the key did not read it, else the copy that still holds every
        item."""
        body_args, body_kwargs = list(args), dict(kwargs)
        for where, original, before, copy in self.splits:
            kept = before is not None and _iterator_position(original) == before
            if isinstance(where, int):
                body_args[where] = original if kept else copy
            else:
                body_kwargs[where] = original if kept else copy
        return tuple(body_args), body_kwargs

    def checked_arguments(self, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        """*args* and *kwargs* with each split iterator a placeholder: what
        the in-place-change check hashes before and after the body, which
        reads its copy and so leaves it changed."""
        checked_args, checked_kwargs = list(args), dict(kwargs)
        for where, _original, _before, _copy in self.splits:
            if isinstance(where, int):
                checked_args[where] = _SPLIT_ITERATOR
            else:
                checked_kwargs[where] = _SPLIT_ITERATOR
        return tuple(checked_args), checked_kwargs


#: What a split iterator argument stands as in the in-place-change check.
_SPLIT_ITERATOR = "<an iterator split for the key>"


def _generator_over(items: Any) -> Iterator[Any]:
    """A generator yielding *items*: of the type a hasher registered for
    generators matches, as the caller's was."""
    return (item for item in items)


def split_for_key(
    spec: CachedFunction, args: tuple, kwargs: dict, registered: Callable[[Any], bool] | None = None
) -> _KeyCopies | None:
    """`_KeyCopies` when an argument is a one-shot iterator the key would
    read; None otherwise.

    A ``key=`` function reads an `itertools.tee` copy: ``key=lambda rows:
    tuple(rows)`` read the iterator before the body ran, which then saw it
    emptied (``total(iter([1, 2, 3]))`` gave 0). A generator that a hasher
    registered for its type (*registered*) keys is read into a list once,
    and the hasher and the body each get a generator over it: a tee copy is
    not a generator, and the hasher would not match it. The key is what the
    hasher makes of the same items. Past `MATERIALISED_ITEMS` the call runs
    uncached (``refused``), the body getting every item.
    """
    key_fn = spec.arg_key is not None and spec.arg_key.key_fn is not None
    if not key_fn and registered is None:
        return None
    splits: list[tuple[int | str, Any, Any, Any]] = []
    key_args, key_kwargs = list(args), dict(kwargs)
    refused: KeyBuildFailed | None = None
    for where, value in [*enumerate(args), *kwargs.items()]:
        if not is_one_shot_iterator(value):
            continue
        before = _iterator_position(value)
        if key_fn:
            for_key, for_body = itertools.tee(value)
        elif registered is not None and inspect.isgenerator(value) and registered(value):
            items = list(itertools.islice(value, MATERIALISED_ITEMS + 1))
            if len(items) > MATERIALISED_ITEMS:
                for_key = for_body = _generator_over(itertools.chain(items, value))
                label = f"argument {where}" if isinstance(where, int) else repr(where)
                refused = refused or KeyBuildFailed(
                    "KEY-ITERATOR-CONSUMED",
                    f"@cash.cache on {spec.name}: the generator passed as {label} is keyed by a hasher "
                    f"registered for its type, which reads it, and it holds more than "
                    f"{MATERIALISED_ITEMS:,} items, too many to read ahead of the body, so the call "
                    f"ran uncached. The body got every item.",
                    "pass a list or a tuple instead of the generator, or key the call by something "
                    "that does not read it.",
                )
            else:
                for_key, for_body = _generator_over(items), _generator_over(items)
        else:
            continue
        if isinstance(where, int):
            key_args[where] = for_key
        else:
            key_kwargs[where] = for_key
        splits.append((where, value, before, for_body))
    if not splits:
        return None
    return _KeyCopies(tuple(key_args), key_kwargs, splits, refused)


def _unkeyed_parameters(spec: CachedFunction) -> frozenset[str]:
    """The parameters *spec* leaves out of its key, defaults and all: the
    ignored ones, or every one under ``key=``."""
    if spec.arg_key is None:
        return frozenset()
    if spec.arg_key.key_fn is not None:
        return EVERY_PARAMETER
    return spec.arg_key.ignored


def _resolution_of(
    spec: CachedFunction, resolvers: Any, args: tuple, kwargs: dict, sources: list[tuple[Any, str]]
) -> dict[tuple, Resolution]:
    """The resolver call that just resolved to *sources*, keyed so that the
    same call met again in one block is held once (`Resolution`)."""
    func_name = spec.name
    expected = [(source.get_id(), token) for source, token in sources]
    func = spec.func
    module, qualname = plain_getattr(func, "__module__", None), plain_getattr(func, "__qualname__", "")
    owner = (module, qualname) if module and "<locals>" not in qualname else None
    key = (
        func_name,
        id(resolvers),
        tuple(id(a) for a in args),
        tuple((k, id(v)) for k, v in kwargs.items()),
        tuple(expected),
    )
    return {key: Resolution(func_name, resolvers, args, kwargs, expected, owner)}


def decorator_key(func_name: str, state_hash: str, dynamic_hash: str, args_hash: str) -> str:
    return f"{func_name}:{state_hash}:{dynamic_hash}:{args_hash}"


def _ledger_moved(before: dict, after: dict | None, watch: Any) -> bool:
    """Does the ledger built after the body name a part that differs from
    *before*, other than the ones the drift guard watches (*watch*)?"""
    if after is None:
        return True
    watched = set(watch) if isinstance(watch, dict) else set()
    for label, now in after.items():
        if label == "@chain" or label not in before:
            continue
        then = before[label]
        if now == then:
            continue
        if isinstance(label, tuple) and label[0] == "globals":
            old_parts = dict(then)
            for name, digest in now:
                if name.partition("#")[0] in watched or name in watched:
                    continue
                if old_parts.get(name, digest) != digest:
                    return True
            continue
        return True
    return False


class KeyBuilder:
    """The cache key of a call: the state segment folded from the function's
    code, closure, defaults, globals, files, RNG epoch and environment, then
    the dynamic dependencies and the arguments."""

    def __init__(
        self,
        registry: FunctionRegistry,
        args: ArgHasher,
        method_deps: MethodClassDeps,
        pins: OwnSourcePins,
        files: FileDeps,
        closures: ClosureFold,
        globals_fold: GlobalsFold,
        environment: EnvironmentFold,
        rng: RngWatch,
        code_args: CodeArgs,
        state_hasher: DependencyStateHasher,
        misses: MissHistory,
        notices: Notices,
    ) -> None:
        self._registry = registry
        self._args = args
        self._method_deps = method_deps
        self._pins = pins
        self._files = files
        self._closures = closures
        self._globals = globals_fold
        self._environment = environment
        self._rng = rng
        self._code_args = code_args
        self._state_hasher = state_hasher
        self._misses = misses
        self._notices = notices

    def _hands_a_state_changing_callable(self, spec: CachedFunction, args: tuple, kwargs: dict) -> bool:
        """Whether the call hands the body a function of the user's that
        changes state when it is called -- ``run(record, 3)`` with ``record``
        appending to a global list, a bound method changing a global, one
        hidden in a ``functools.partial`` or a list the body calls the items
        of -- and has been warned about once.

        A hit would return the stored value without running it, so the change
        would happen on the first call only. Such a call runs uncached, which
        is what plain Python does. ``assume_safe=True`` says the function's
        effects are fine to skip and keeps it cached; ``strict=True`` raises
        `CashImpureFunctionError`, as it does for every other finding.

        A method changing its own object is left to the argument-mutation
        check (``objects=False``): that object is an argument, or held by
        one, and is hashed again after a miss. Nor is what the body reaches
        through its own globals looked at here (``through_globals=False``):
        that is the static purity analysis's to report.
        """
        mode = self._registry.purity_mode(spec.name)
        if mode == "silent" or not (args or kwargs):
            return False
        try:
            handed = hands_a_state_changing_callable(spec.func, args, kwargs, through_globals=False, objects=False)
        except Exception:  # cannot tell: the safe answer is "yes"
            logger.debug("handed-callable check failed for %s", spec.name, exc_info=True)
            handed = True
        if handed and mode == "strict":
            raise CashImpureFunctionError(
                f"@cash.cache(strict=True) on {spec.name}: an argument is a function "
                f"of yours, or holds one, that changes state when it is called, so a "
                f"cache hit would skip that change. Pass a function that returns its "
                f"result instead, or call it outside the cached function."
            )
        if handed:
            self._notices.warn_once(
                CashImpurityWarning,
                spec.name,
                "handed_callable",
                f"@cash.cache on {spec.name}: an argument changes state when it is "
                f"used, so calls like this run uncached, since a cache hit would skip "
                f"that change. It is a function of yours, or holds one, that changes a "
                f"global or what it closes over when it is called (or whose source "
                f"cannot be read), or an object of your class whose special methods "
                f"or properties (``with cm:``, ``b[k]``, ``b.value``) change a global.",
                code="IMPURE-HANDED-CALLABLE",
                fix="return the new value from the function you pass instead of "
                "changing state in it, or call it outside the cached function. "
                "If its effect is fine to skip on a hit, "
                "@cash.cache(assume_safe=True) caches the call again.",
            )
        return handed

    def split_for_key(self, spec: CachedFunction, args: tuple, kwargs: dict) -> _KeyCopies | None:
        """`split_for_key` with this cache's registered hashers."""
        hashers = self._args
        registered = hashers.keys_by_registration_only if hashers.type_hashers or hashers.override_hashers else None
        return split_for_key(spec, args, kwargs, registered)

    def resolve(
        self,
        spec: CachedFunction,
        args: tuple,
        kwargs: dict,
        split: frozenset[int] = frozenset(),
        *,
        refused: KeyBuildFailed | None = None,
    ) -> tuple[BuiltKey | Unkeyable, dict, dict]:
        """The key for a real call, or why it has none; its `CAPTURE_WATCH`;
        and its `STATE_LEDGER`.

        `KeyBuilder.build`, with a ledger of what the state segment is made of
        (`STATE_LEDGER`). A call with no key -- a mocked helper, an argument
        or default that cannot be hashed, a key build that raised -- has
        been warned about once; the caller runs it uncached.
        """
        func, func_name = spec.func, spec.name
        unkeyable = self._registry.refresh_helper_bindings(func, func_name)
        unread = self._registry.unread_imports.get(func_name, frozenset())
        # Every call's body is recorded and its entry served only while what
        # ran is unchanged (`code_trace`); one that picks code by a run-time
        # name is keyed as any call and needs that trace. Not a coroutine,
        # whose body other tasks' code runs in the middle of.
        coroutine = inspect.iscoroutinefunction(func)
        needs_trace = False
        if unkeyable is not None:
            if not unkeyable.traced or coroutine:
                return Unkeyable(unkeyable._replace(traced=False)), {}, {}
            needs_trace = True
        traced = needs_trace or (code_trace.on() and not coroutine)
        if self._hands_a_state_changing_callable(spec, args, kwargs):
            return Unkeyable(_HANDED), {}, {}
        ledger: dict = {}
        ledger_token = STATE_LEDGER.set(ledger)
        watch: dict = {}
        watch_token = CAPTURE_WATCH.set(watch)
        folded = FoldedCode()
        folded_token = FOLDED_CODE.set(folded)
        lookups: list = []
        lookups_token = CARRIER_LOOKUPS.set(lookups)
        try:
            if refused is not None:
                raise refused
            built = self.build(spec, args, kwargs, split)
        except UnhashableDefault:
            # `ClosureFold.fold_defaults` has warned: an unhashable default means cash
            # cannot tell whether it changed, so caching at all risks a stale
            # result.
            return Unkeyable(_UNHASHABLE), watch, ledger
        except UnhashableArgs as e:
            hashed_args, hashed_kwargs = e.keyed if e.keyed is not None else (args, kwargs)
            self._args.warn_unhashable_args(func_name, hashed_args, hashed_kwargs, e.args[0] if e.args else None)
            return Unkeyable(_UNHASHABLE), watch, ledger
        except KeyBuildFailed as e:
            self._notices.warn_once(CashCacheIneffectiveWarning, func_name, e.code, e.message, code=e.code, fix=e.fix)
            return Unkeyable(_KEY_FAILED), watch, ledger
        except Exception as e:  # noqa: BLE001 - any failure building the key means no key
            self._args.warn_key_build_failed(func_name, args, kwargs, e)
            return Unkeyable(_KEY_FAILED), watch, ledger
        finally:
            CARRIER_LOOKUPS.reset(lookups_token)
            FOLDED_CODE.reset(folded_token)
            CAPTURE_WATCH.reset(watch_token)
            STATE_LEDGER.reset(ledger_token)
        for_unread = False
        if lookups:
            unread = unread | frozenset(filter(None, map(unread_module, lookups)))
            picked = [i for i in lookups if UNREAD_MODULE not in i.description]
            if picked or (code_trace.on() and not coroutine):
                # Served on a trace, which holds an import that ran in the
                # call (``first_import``) and the values its module bound.
                refusal = self._carrier_refusal(func_name, lookups, coroutine)
                if refusal is not None:
                    return Unkeyable(refusal), watch, ledger
                for_unread = not needs_trace and not picked
                needs_trace = traced = True
            # An import of a module not loaded yet alone, with no trace: the
            # call is keyed, and not stored when it loads the module
            # (`CallRunner._loaded_unread`).
        built = built._replace(folded_code=folded, unread=unread, trace_for_unread=for_unread)
        if ledger:
            slot = (func_name, built.state_hash)
            if not self._misses.has_ledger(slot):
                self._misses.keep_state_ledger(slot, ledger)
        if traced:
            built = built._replace(traced=True, needs_trace=needs_trace)
        return built, watch, ledger

    @staticmethod
    def _carrier_refusal(func_name: str, lookups: list, coroutine: bool) -> MissReason | None:
        """Why a call whose arguments carry code that picks code by a
        run-time name (*lookups*, KEY-DYNAMIC-DEPENDENCY) runs uncached, or
        None when a trace of what it ran stands in for them, as for the same
        line in the body.

        A lazy import of a module not loaded yet whose top level cash cannot
        read (UNREAD_MODULE), in a method the call may never run, needs no
        key part when the call is traced: an import that runs is in the
        trace (``first_import``), with the code of that module that ran and
        the globals it read. Without a trace (Python < 3.12, ``CASH_TRACE=0``)
        an edit to that module could not be seen, so the call runs uncached.
        """
        traced = code_trace.on() and not coroutine
        rest = [i for i in lookups if UNREAD_MODULE not in i.description] if traced else lookups
        if not rest or (not coroutine and code_trace.covers(rest)):
            return None
        refusing = [i for i in rest if coroutine or not code_trace.covers([i])]
        unread = [i for i in refusing if UNREAD_MODULE in i.description]
        picked = [i for i in refusing if UNREAD_MODULE not in i.description]
        loads = (
            f"imports a module cash cannot key before it is loaded ({describe_untrackable(unread)})" if unread else ""
        )
        if not picked:
            return MissReason(
                MissKind.UNREAD_MODULE, f"code an argument of {func_name} carries {loads}, so the call ran uncached"
            )
        what = f"picks code by a run-time name ({describe_untrackable(picked)})" + (f" and {loads}" if loads else "")
        return MissReason(
            MissKind.UNTRACKABLE, f"code an argument of {func_name} carries {what}, so the call ran uncached"
        )

    def _code_state(self, spec: CachedFunction, chain: list[str], *, note: bool = False) -> str:
        """`_code_state_now`: its first part (`_code_state_pre`) as it gave
        last time while nothing it reads has moved since (`state_memo`),
        the rest (`_code_state_post`) built now."""
        memo = state_memo.memo_for(spec)
        if memo is None:
            state_hash = self._code_state_pre(spec, chain, note=note)
        else:
            state_hash = memo.code_state(self, spec, chain, note)
        return self._code_state_post(spec, state_hash, chain)

    def _code_state_now(self, spec: CachedFunction, chain: list[str], *, note: bool = False) -> str:
        """The state segment before anything the arguments decide: the
        function's code, helpers, cached callees, declared files, closure,
        defaults, globals (`_code_state_pre`), then the RNG epoch, the
        environment and ``key=`` (`_code_state_post`). Appends each stage
        to *chain*.

        Raises `UnhashableDefault` when a default cannot be hashed.
        """
        return self._code_state_post(spec, self._code_state_pre(spec, chain, note=note), chain)

    def _code_state_pre(self, spec: CachedFunction, chain: list[str], *, note: bool = False) -> str:
        """The part of the state segment that the program alone decides --
        code, helpers, cached callees, declared files, closure, defaults,
        globals -- which `state_memo` serves again while none of it moved.
        Appends each stage to *chain*.

        What the decoration itself says -- its declared files, ``key=`` and
        ignored parameters -- is read from *spec*, the wrapper's own record,
        never from the registry's slot for the name: two cached functions can
        share a module and qualified name (one function wrapped twice, the
        closures a factory makes, two lambdas), and the slot holds only the
        one decorated last.

        Raises `UnhashableDefault` when a default cannot be hashed.
        """
        func, func_name = spec.func, spec.name
        note_owner(func, code_objects_of(func))
        # Edges the dependency state left out as already walked as helpers:
        # the folds below over the same graph leave them out too.
        covered: set[tuple[str, str]] = set()
        state_hash = self._state_hasher.compute(
            func_name,
            own_source_override=self._pins.pin_own_source(func),
            own_report=self._registry.report_for(func, func_name),
            note=note,
            covered=covered,
        )
        state_hash = self._fold_callee_bindings(func_name, state_hash, covered)
        state_hash = self._files.fold_declared_files(spec, state_hash)
        chain.append(state_hash)
        state_hash = self._closures.fold_closure(func, func_name, state_hash)
        chain.append(state_hash)
        folded_defaults = self._closures.fold_defaults(func, func_name, state_hash, _unkeyed_parameters(spec))
        if folded_defaults is None:
            raise UnhashableDefault
        state_hash = folded_defaults
        chain.append(state_hash)
        state_hash = self._closures.fold_bound_self(func, func_name, state_hash)
        state_hash = self._closures.fold_bound_partial(func, func_name, state_hash)
        chain.append(state_hash)
        phase = _phases.sub_begin(_phases.KEY_GLOBALS) if _phases.ON else None
        state_hash = self._globals.fold_read_globals(func, func_name, state_hash)
        state_hash = self._globals.fold_helper_read_globals(func, func_name, state_hash)
        state_hash = self._globals.fold_dependency_read_globals(func, func_name, state_hash, covered)
        if phase is not None:
            _phases.sub_end(phase)
        chain.append(state_hash)
        return state_hash

    def _code_state_post(self, spec: CachedFunction, state_hash: str, chain: list[str]) -> str:
        """The rest of the state segment, folded onto `_code_state_pre`'s
        digest on every call: the RNG epoch (its seed moves without any
        change to the program), the environment and the installed versions
        of the libraries the call reaches, and ``key=``. Kept out of the
        part `state_memo` serves, so a function that draws from a global
        stream or reads a variable still has its code and globals served."""
        func, func_name = spec.func, spec.name
        state_hash = self._rng.fold_rng_epoch(func_name, state_hash)
        chain.append(state_hash)
        state_hash = self._environment.fold_environment(func, func_name, state_hash)
        chain.append(state_hash)
        if spec.arg_key is not None:
            # What decides the argument part is code too: a caller that
            # reaches this function keys it as well (`callee_state`).
            state_hash = self._fold_key_function(spec, state_hash)
        return state_hash

    def _fold_callee_bindings(self, func_name: str, state_hash: str, covered: Collection[tuple[str, str]] = ()) -> str:
        """Fold what each cached function *func_name* calls is bound to: its
        captured variables and its parameter defaults, transitively.

        The dependency state keys a cached callee by its code, helpers and
        callees. Its captures and defaults live on the function object, not in
        its code: ``def clip(x, limit=THRESHOLD)`` with ``THRESHOLD`` edited in
        another module, or ``add = make(n)`` built with another ``n``, served
        the caller's old result while the callee called directly recomputed.
        They are folded by the same rules as the caller's own
        (`ClosureFold.fold_closure`, `ClosureFold.fold_defaults`). A callee
        another registry describes (`FunctionRegistry.reached_callee`) is
        keyed whole by its own state already, and a *covered* edge's callee
        was keyed as a helper, defaults and captures included
        (`HelperIdentity.identity`).

        Raises `KeyBuildFailed` when a callee's default cannot be hashed.
        """
        parts: list[str] = []
        graph = self._registry.graph
        visited = {func_name}
        stack = [(func_name, dep) for dep in sorted(graph.get_dependencies(func_name), reverse=True)]
        # A callee's mutable capture is not this call's to watch.
        watch_token = CAPTURE_WATCH.set(None)
        try:
            while stack:
                node, dep = stack.pop()
                if dep in visited or (node, dep) in covered:
                    continue
                visited.add(dep)
                dep_func = self._registry.functions.get(dep)
                if dep_func is None or self._registry.reached_callee(node, dep) is not None:
                    continue
                stack.extend((dep, d) for d in sorted(graph.get_dependencies(dep), reverse=True))
                dep_spec = self._registry.cached.get(dep)
                leave_out = _unkeyed_parameters(dep_spec) if dep_spec is not None else frozenset()
                if not getattr(dep_func, "__closure__", None) and defaults_of(dep_func, leave_out) == ((), {}):
                    continue
                bound = self._closures.fold_closure(dep_func, dep, "")
                bound = self._closures.fold_defaults(dep_func, dep, bound, leave_out)
                if bound is None:
                    raise _unhashable_callee_default(dep)
                ledger_note(("captures and defaults of cached function", dep), bound)
                parts.append(f"{dep}={bound}")
        finally:
            CAPTURE_WATCH.reset(watch_token)
        if not parts:
            return state_hash
        return hashlib.sha256(f"{state_hash}:callee-bindings:{':'.join(parts)}".encode("utf-8")).hexdigest()

    @staticmethod
    def held_by_name_for(spec: CachedFunction) -> bool:
        """Is a call of *spec* keyed with the classes of module data by their
        surface (`HELD_BY_NAME`)? When its body is recorded and its entry
        served on that record (`code_trace.on`): not for a coroutine, whose
        body other tasks' code runs in the middle of, nor for a generator,
        whose body runs after the call, unrecorded."""
        func = spec.func
        return code_trace.on() and not (
            inspect.iscoroutinefunction(func) or inspect.isgeneratorfunction(func) or inspect.isasyncgenfunction(func)
        )

    def full_closure_state(self, spec: CachedFunction) -> str:
        """`callee_state` with every class of module data keyed by all its
        code, as for an untraced call (`HELD_BY_NAME` off): the check an
        entry whose call left no trace is stored with."""
        token = HELD_BY_NAME.set(False)
        try:
            return self.callee_state(spec)
        finally:
            HELD_BY_NAME.reset(token)

    def callee_state(self, spec: CachedFunction) -> str:
        """What a call of cached *func_name* depends on besides its arguments,
        as ONE digest: for a cached function another one reaches without a
        graph edge of its own registry.

        A cached function on another `Cash`, one passed in as an argument or
        held in a dict, a list or a closure, or one a caller still holds after
        ``importlib.reload`` put a new function under its name: its code was
        keyed by cash's own wrapper, or by the new function, and the globals
        and environment it reads not at all, so an edit to any of them served
        the caller's old result. This is the key's own state segment, built by
        the instance that owns *func*, for the function object the caller
        reaches.

        A cycle (a cached function that reaches itself through a table)
        contributes its name. The ledger and the capture watch of the key
        being built are the caller's, so they are set aside meanwhile.

        Raises `KeyBuildFailed` when the state cannot be built. Built inside
        a key build, with that build's `HELD_BY_NAME`; outside one, with
        the function's own (`held_by_name_for`).
        """
        class_memo.CALLEE_STATES_BUILT[0] += 1
        if HELD_BY_NAME.get() is not None:
            return self._callee_state(spec)
        token = HELD_BY_NAME.set(self.held_by_name_for(spec))
        try:
            return self._callee_state(spec)
        finally:
            HELD_BY_NAME.reset(token)

    def _callee_state(self, spec: CachedFunction) -> str:
        func, func_name = spec.func, spec.name
        reached_ttls = REACHED_TTLS.get()
        if reached_ttls is not None:
            # What the caller computes from this function is as fresh as it.
            reached_ttls.append(self._registry.effective_ttl(func_name, spec.ttl))
        active = CALLEE_STATES.get()
        if func_name in active:
            walk_scope.note_callees((func_name,))
            return f"cycle:{func_name}"
        # Reached from many places in one key build (a package whose module
        # namespace holds its cached functions: every function reading
        # ``nx.foo`` reaches it), it is built once per key build, in a walk
        # of its own, and served while the same digest would come out
        # (`walk_scope`). The TTLs its own callees added are added again for
        # each caller.
        build = KEY_BUILD_DIGESTS.get()
        item = ("callee", func_name)
        if build is not None:
            hit = build.get(item)
            if hit is not None and hit[0] is func and walk_scope.servable(hit[2]):
                if reached_ttls is not None:
                    reached_ttls.extend(hit[3])
                walk_scope.note_callees((func_name, *hit[2].touched))
                return hit[1]
        ttls_from = len(reached_ttls) if reached_ttls is not None else 0
        with walk_scope.isolated(item, func_name) as frame:
            state = self._callee_state_now(spec)
        touched = frozenset(frame.touched)
        if build is not None:
            ttls = tuple(reached_ttls[ttls_from:]) if reached_ttls is not None else ()
            build[item] = (func, state, walk_scope.Record(frozenset(), touched), ttls)
        walk_scope.note_callees((func_name, *touched))
        return state

    def _callee_state_now(self, spec: CachedFunction) -> str:
        """`callee_state` built now, for a function no running build of a
        callee state is building."""
        func, func_name = spec.func, spec.name
        active = CALLEE_STATES.get()
        token = CALLEE_STATES.set(active | {func_name})
        ledger_token = STATE_LEDGER.set(None)
        watch_token = CAPTURE_WATCH.set(None)
        # The callee's digest folds its own classes, whatever the caller folded.
        classes_token = CLASSES_FOLDED.set(set())
        reads_token = READS_FOLDED.set({})
        try:
            if self._registry.functions.get(func_name) is func:
                self._registry.ensure_closure_analyzed(func)
            if state_memo.memo_for(spec) is None:
                return self._code_state(spec, [])
            # Apart from the caller's containers, so the callee's own memo
            # can serve its segment (`state_memo.fresh_context`).
            return state_memo.fresh_context(lambda into: self._code_state(spec, into), [])
        except UnhashableDefault:
            raise _unhashable_callee_default(func_name) from None
        finally:
            READS_FOLDED.reset(reads_token)
            CLASSES_FOLDED.reset(classes_token)
            CAPTURE_WATCH.reset(watch_token)
            STATE_LEDGER.reset(ledger_token)
            CALLEE_STATES.reset(token)

    def build(self, spec: CachedFunction, args: tuple, kwargs: dict, split: frozenset[int] = frozenset()) -> BuiltKey:
        """The cache key for calling *spec*'s function with these arguments.

        The ONE key build: a real call (`KeyBuilder.resolve`) and ``explain()``
        (`Explainer.explain`) both use it, so the key explain() predicts is the key
        the call looks up.

        Raises when there is no key: `UnhashableDefault`, `UnhashableArgs`,
        `KeyBuildFailed`, or whatever else a step raised. Never keys the call
        without a part that failed. Warnings from the steps are silent while
        `_EXPLAINING` is set. *split* holds the ids of the iterator copies
        made for the key alone (`split_for_key`). Each helper the build's
        purity walks reach is read and audited once (`shared_nodes`).
        The classes of module data are keyed by their surface for a call
        whose body is recorded (`held_by_name_for`, `BuiltKey.narrowed`).
        """
        narrow = self.held_by_name_for(spec)
        token = HELD_BY_NAME.set(narrow)
        try:
            with shared_nodes():
                built = self._build(spec, args, kwargs, split)
        finally:
            HELD_BY_NAME.reset(token)
        return built._replace(narrowed=True) if narrow else built

    def full_key_folded(self, spec: CachedFunction, args: tuple, kwargs: dict) -> FoldedCode | None:
        """The code a key of this call built with the classes of module
        data keyed by all their code (`HELD_BY_NAME` off) folds, or None
        when it cannot be built.

        What the strict trace of a narrowed call accepts
        (`CallRunner._trace_of`): narrowing keys a held class by its
        surface and leaves its methods to the trace, which checks each
        one that ran with its own item; the strict check is for code
        that no key of the call covers at all, so it judges against what
        the key would fold unnarrowed, as it did before narrowing. One
        build per miss of such a call. No watch, ledger or warnings of its
        own: the call's key build gave them.
        """
        folded = FoldedCode()
        tokens = (
            (FOLDED_CODE, FOLDED_CODE.set(folded)),
            (STATE_LEDGER, STATE_LEDGER.set({})),
            (CAPTURE_WATCH, CAPTURE_WATCH.set({})),
            (CARRIER_LOOKUPS, CARRIER_LOOKUPS.set([])),
            (_EXPLAINING, _EXPLAINING.set(True)),
            (HELD_BY_NAME, HELD_BY_NAME.set(False)),
        )
        try:
            with shared_nodes():
                self._build(spec, args, kwargs, frozenset())
        except Exception:  # noqa: BLE001 - no full key: the strict trace judges against the call's own
            return None
        finally:
            for var, token in reversed(tokens):
                var.reset(token)
        return folded

    def _build(self, spec: CachedFunction, args: tuple, kwargs: dict, split: frozenset[int]) -> BuiltKey:
        """`build`'s work, as one round of checks (`hit_memo.probe_round`):
        data many class watches hold is read whole once."""
        with hit_memo.probe_round():
            return self._build_now(spec, args, kwargs, split)

    def _build_now(self, spec: CachedFunction, args: tuple, kwargs: dict, split: frozenset[int]) -> BuiltKey:
        func, func_name = spec.func, spec.name
        iterators = iterator_arguments(spec.signature, args, kwargs) if args or kwargs else []
        # One plain-data census per argument, shared across the key
        # (`plain_census`).
        previous = getattr(PLAIN_CENSUS, "memo", None)
        PLAIN_CENSUS.memo = {}
        # Each class's data is folded once per key (`ClassDataFold.class_parts`).
        classes_token = CLASSES_FOLDED.set(set())
        # Each function's globals are folded once per key (`READS_FOLDED`).
        reads_token = READS_FOLDED.set({})
        reached_ttls: list = []
        ttls_token = REACHED_TTLS.set(reached_ttls)
        # Each value module data holds is hashed once per key (`KEY_BUILD_DIGESTS`).
        digests_token = KEY_BUILD_DIGESTS.set({})
        pint_token = PINT_DIGESTS.set({})
        try:
            # The state after each fold, in `_STATE_STAGES` order: when no
            # named part moved, the first stage whose output did is the one
            # that changed (`describe_state_change`).
            chain: list[str] = []
            ledger_note("@chain", chain)
            phase = _phases.sub_begin(_phases.KEY_CODE) if _phases.ON else None
            state_hash = self._code_state(spec, chain, note=True)
            state_hash = self._method_deps.fold_method_class_deps(func, args, state_hash)
            if phase is not None:
                _phases.sub_end(phase)
            chain.append(state_hash)
            # ONE canonicalisation, fed to both the code channel and the value
            # channel. `CodeArgs.fold_code_args` on the RAW arguments saw a class
            # passed explicitly but not the identical class arriving as a
            # parameter DEFAULT, so `build()` and `build(Schema)` -- the same
            # logical call -- produced two cache keys and two executions.
            normalized_args = self._args.normalize_call_args(func_name, args, kwargs, signature=spec.signature)
            if spec.seed_params:
                self._rng.warn_if_seed_is_none(func, func_name, args, kwargs)
            # What the argument part of the key is made of: every argument,
            # or what ``key=`` / the ignored parameters leave of them.
            keyed = normalized_args
            if spec.arg_key is not None:
                keyed = self.key_arguments(spec, args, kwargs, normalized_args)
            # The code channel. ``key=`` replaces only the arguments' VALUE
            # part: the body still runs the code the real arguments carry (a
            # parameter object's methods, a passed function), so that code is
            # folded from them, and from what the key function returned on top
            # of it. Parameters left out with ``ignore=`` are declared not to
            # matter to the call, code and all.
            carriers = [keyed]
            if spec.arg_key is not None and spec.arg_key.key_fn is not None:
                carriers.insert(0, normalized_args)
            # A bound method's ``self`` (``cash.cache(est._fit)``) is not in
            # the arguments, but its code runs as if it were: what the body
            # reaches through it (a subclass override, the helpers that
            # calls) is folded as for an unbound method's ``self``. Its
            # value is folded by `ClosureFold.fold_bound_self`.
            bound, owner = bound_owner(func)
            if bound:
                carriers.append(((owner,), {}))
            phase = _phases.sub_begin(_phases.KEY_ARG_CODE) if _phases.ON else None
            for carrier_args in carriers:
                state_hash = self._code_args.fold_code_args(
                    *carrier_args,
                    state_hash,
                    func_name=func_name,
                    owner_code=plain_getattr(func, "__code__", None),
                    func=func,
                    report=self._registry.report_for(func, func_name),
                )
            if phase is not None:
                _phases.sub_end(phase)
            chain.append(state_hash)
            dynamic_sources: list = []
            try:
                dynamic_state_hash = resolve_dynamic_dependencies(
                    func_name, spec.dynamic_depends_on, args, kwargs, dynamic_sources
                )
            except KeyBuildFailed:
                # The call runs uncached; a cached call around it must not
                # be stored as if it had no such dependency.
                if not _EXPLAINING.get():
                    note_unresolved_dynamic(func_name)
                raise
            if spec.dynamic_depends_on and not _EXPLAINING.get():
                pass_dynamic_sources_up(
                    dynamic_sources, _resolution_of(spec, spec.dynamic_depends_on, args, kwargs, dynamic_sources)
                )
            failure: list = []
            # Which arguments it keys by all they hold, noted while it hashes
            # them and only then (`ContentKeyed`).
            content = CONTENT_KEYED.get()
            indexes_token = None
            if content is not None:
                content.armed = True
                indexes_token = WALKED_INDEXES.set(content.indexes)
            phase = _phases.sub_begin(_phases.KEY_ARGS) if _phases.ON else None
            try:
                if keyed is normalized_args:
                    args_hash = self._args.serialize_args(
                        func_name, args, kwargs, normalized=normalized_args, failure=failure
                    )
                else:
                    args_hash = self._keyed_args_hash(keyed, failure)
            finally:
                if content is not None:
                    content.armed = False
                if indexes_token is not None:
                    WALKED_INDEXES.reset(indexes_token)
                if phase is not None:
                    _phases.sub_end(phase)
            self._args.note_arg_cost(func_name)
        finally:
            PINT_DIGESTS.reset(pint_token)
            KEY_BUILD_DIGESTS.reset(digests_token)
            PLAIN_CENSUS.memo = previous
            REACHED_TTLS.reset(ttls_token)
            READS_FOLDED.reset(reads_token)
            CLASSES_FOLDED.reset(classes_token)
        if iterators:
            self._check_iterators_unread(spec, iterators, split)
        if args_hash is None:
            raise UnhashableArgs(*failure[:1], keyed=None if keyed is normalized_args else keyed)
        cache_key = decorator_key(func_name, state_hash, dynamic_state_hash, args_hash)
        # What the in-place-change check compares after the body: the key's
        # own argument hash, unless ``key=`` replaced it
        # (`PurityChecks.checked_arguments_hash`).
        call_args_hash = args_hash if spec.arg_key is None or spec.arg_key.key_fn is None else None
        ttls = [t for t in reached_ttls if t is not None]
        return BuiltKey(cache_key, state_hash, args_hash, normalized_args, call_args_hash, min(ttls) if ttls else None)

    def ledger_now(self, spec: CachedFunction, args: tuple) -> dict | None:
        """The `STATE_LEDGER` of the key's state segment built again now,
        after the body ran, or None when it cannot be built.

        Another thread can rebind or change in place a global the call read
        (a config reload in a service) while the body runs: the result then
        comes from the new value but would be stored under the key of the
        old one. No drift-guard watch and no warnings of its own: the key
        build before the body already gave them.
        """
        ledger: dict = {}
        previous = getattr(PLAIN_CENSUS, "memo", None)
        PLAIN_CENSUS.memo = {}
        tokens = (
            (CLASSES_FOLDED, CLASSES_FOLDED.set(set())),
            (READS_FOLDED, READS_FOLDED.set({})),
            (REACHED_TTLS, REACHED_TTLS.set([])),
            (KEY_BUILD_DIGESTS, KEY_BUILD_DIGESTS.set({})),
            (PINT_DIGESTS, PINT_DIGESTS.set({})),
            (STATE_LEDGER, STATE_LEDGER.set(ledger)),
            (CAPTURE_WATCH, CAPTURE_WATCH.set(None)),
            (_EXPLAINING, _EXPLAINING.set(True)),
            # As the key before the body held the classes of module data.
            (HELD_BY_NAME, HELD_BY_NAME.set(self.held_by_name_for(spec))),
        )
        try:
            with shared_nodes():
                state_hash = self._code_state(spec, [], note=True)
                self._method_deps.fold_method_class_deps(spec.func, args, state_hash)
        except Exception:  # noqa: BLE001 - cannot be built now: it cannot be shown unchanged
            return None
        finally:
            for var, token in reversed(tokens):
                var.reset(token)
            PLAIN_CENSUS.memo = previous
        return ledger

    def _check_iterators_unread(
        self, spec: CachedFunction, iterators: list[tuple[str, Any, Any]], split: frozenset[int] = frozenset()
    ) -> None:
        """Raise `KeyBuildFailed` when building the key read an iterator
        argument, or may have and cash cannot tell.

        A ``key=`` function (``key=lambda rows: tuple(rows)``) or a hasher
        registered for the iterator's type consumes it before the body runs:
        the body then sees it emptied, and its wrong result was stored and
        served, even for a list of the same rows.
        """
        readers = spec.arg_key is not None and spec.arg_key.key_fn is not None
        readers = readers or bool(self._args.type_hashers or self._args.override_hashers)
        for label, value, before in iterators:
            if id(value) in split:
                continue  # the key's own copy (`split_for_key`): the body has another
            after = _iterator_position(value)
            if before is not None and after == before:
                continue
            if before is None and not readers:
                continue
            what = "consumed" if before is not None else "may have consumed"
            raise KeyBuildFailed(
                "KEY-ITERATOR-CONSUMED",
                f"@cash.cache on {spec.name}: building the key {what} the iterator passed as "
                f"{label!r} ({class_name(type(value))}), so the body sees what is left of it and the "
                f"call ran uncached.",
                "pass a list or a tuple instead of an iterator, or key the call by something "
                "that does not read the iterator.",
            )

    def key_arguments(self, spec: CachedFunction, args: tuple, kwargs: dict, normalized: tuple[tuple, dict]) -> tuple:
        """`keyed_arguments` for *spec*'s ``key=`` or ignored parameters.

        The first time a key function runs, it runs under a file tracker of
        its own: a file it reads is an input the key cannot see, and the
        static check (`key_function_impurities`) only finds the readers it
        can name. Raises `KeyBuildFailed` when the key function raises.
        """
        arg_key = spec.arg_key
        if arg_key.key_fn is None or spec.key_reads_checked or _EXPLAINING.get():
            return keyed_arguments(arg_key, spec.signature, spec.name, args, kwargs, normalized)
        spec.key_reads_checked = True
        tracker = FileAccessTracker()
        with tracker:
            keyed = keyed_arguments(arg_key, spec.signature, spec.name, args, kwargs, normalized)
        read = sorted(tracker.get_accessed_files() | tracker.get_accessed_remote_urls())
        if read and spec.purity != "silent":
            self._notices.key_function_impure(spec.name, [f"reads {os.path.normpath(path)}" for path in read[:3]])
        return keyed

    def _fold_key_function(self, spec: CachedFunction, state_hash: str) -> str:
        """Fold the ``key=`` function's code into the state, as a function
        passed as an argument is folded: its code, the user code it reaches
        and the globals that code reads, plus what it holds -- captured
        values, defaults, a callable object's state
        (`ClosureFold.key_function_part`). An edit to any of them re-keys
        every call."""
        key_fn = spec.arg_key.key_fn
        if key_fn is None:
            part = f"ignore:{sorted(spec.arg_key.ignored)}"
            ledger_note(("ignored parameters", ", ".join(sorted(spec.arg_key.ignored))), part)
        else:
            parts = [f"{func_key(key_fn)}:{hash_callable_source(key_fn)}", self._closures.key_function_part(key_fn)]
            parts.extend(
                self._code_args.carrier_parts(
                    key_fn, spec.name, "key", owner_code=plain_getattr(spec.func, "__code__", None)
                )
            )
            part = "keyfn:" + hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()
            ledger_note(("key= function", plain_getattr(key_fn, "__qualname__", class_name(type(key_fn)))), part)
        return hashlib.sha256(f"{state_hash}:{part}".encode("utf-8")).hexdigest()

    def _keyed_args_hash(self, keyed: tuple[tuple, dict], failure: list) -> str | None:
        """`ArgHasher.hash_payload` of what ``key=`` / the ignored parameters
        left, or None (with what raised in *failure*). No retry with the raw
        arguments, as `ArgHasher.serialize_args` does for a default: that
        would key the parameters the user left out."""
        try:
            return self._args.hash_call_payload(*keyed)
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError) as e:
            failure.append(e)
            return None

    def call_args_hash(self, spec: CachedFunction, args: tuple, kwargs: dict, built: BuiltKey) -> str | None:
        """The hash the argument-mutation check compares after the body
        (`PurityChecks.checked_arguments_hash`). Under ``key=``, the hash of
        every argument, which an entry also records so explain() can say a
        hit was matched by ``key=``; None when they cannot all be hashed.
        Otherwise the key's own argument hash: a parameter left out with
        ``ignore=`` is not checked, so not hashed."""
        if built.call_args_hash is not None or spec.arg_key is None or spec.arg_key.key_fn is None:
            return built.call_args_hash
        try:
            digest = self._args.serialize_args(spec.name, args, kwargs, normalized=built.normalized_args)
        except Exception:  # a hash for the record only, never the key
            logger.debug("[CORE] could not hash every argument of %s", spec.name, exc_info=True)
            return None
        # ``key=`` leaves every argument out of the key, yet each is still
        # hashed here and after the body: when that is what a miss costs,
        # CACHE-NET-LOSS says so.
        self._args.note_arg_cost(spec.name, unkeyed=True)
        return digest

    def checked_arguments_hash(self, spec: CachedFunction, args: tuple, kwargs: dict) -> str | None:
        """The hash of *args* and *kwargs* the in-place-change check
        compares, for a call whose iterators were split for the key
        (`_KeyCopies.checked_arguments`); None when they cannot all be
        hashed."""
        try:
            return self._args.serialize_args(spec.name, args, kwargs)
        except Exception:  # a hash for the check only, never the key
            logger.debug("[CORE] could not hash every argument of %s", spec.name, exc_info=True)
            return None


class CallRunner:
    """The steps of a cached call, shared by the sync and async wrappers: the
    lookup, the body's scope, what a miss does after it, and the locked and
    single-flight paths of ``use_locking``."""

    def __init__(
        self,
        registry: FunctionRegistry,
        backend_slot: BackendSlot,
        keys: KeyBuilder,
        store: ResultStore,
        calls: CallLog,
        files: FileDeps,
        purity: PurityChecks,
        rng: RngWatch,
        misses: MissHistory,
        notices: Notices,
        verdicts: RefusalVerdicts | None = None,
    ) -> None:
        self._registry = registry
        self._backend_slot = backend_slot
        #: What earlier calls learned of arguments changed in place
        #: (`refusal_verdicts`).
        self._verdicts = verdicts
        #: func_name -> what a call refused for a first-seen argument change
        #: warned (`_settle_verdict`).
        self._first_refusal_warnings: dict[str, list] = {}
        self._keys = keys
        self._store = store
        self._calls = calls
        self._files = files
        self._purity = purity
        self._rng = rng
        self._misses = misses
        self._notices = notices
        # In-process async single-flight registry: cache_key ->
        # concurrent.futures.Future. When use_locking is set, concurrent awaits
        # of the same key coalesce - one coroutine computes, the rest wait and
        # then read the stored result.
        #
        # A plain future rather than an asyncio.Event, because an Event belongs
        # to the loop that made it: with one slot per key, a leader in a second
        # loop replaced the first loop's event and its followers -- unable to
        # await another loop's event -- each computed for themselves (4 loops x
        # 4 awaits ran the body 16 times). `asyncio.wrap_future` attaches the
        # wait to whichever loop is asking, so every await in the process
        # coalesces, which is what the docs promise.
        self._async_inflight: dict[str, Any] = {}
        self._async_inflight_lock = threading.Lock()
        #: ``function name -> the registries`` a call of it changed
        #: (`_refuse_registries`).
        self._registry_writers: dict[str, str] = {}
        #: func_name -> how many calls in a row ran code other than their
        #: key folded (`_code_swapped`).
        self._code_swaps: dict[str, int] = {}
        #: Functions a miss of which needed the code its key folds unnarrowed
        #: and could not build it after the body (`_late_full_fold`): their
        #: misses build it before the body.
        self._full_fold_first: set[str] = set()

    def _run_uncached(self, spec: CachedFunction, call: Call, why: MissReason) -> Any:
        """Run a call that has no key, log it as a miss, and hand back its result."""
        note_unentered_body(spec.name)
        # Nothing is stored, so nothing is written back: findings of an
        # argument changed in place stand as they are.
        self._purity.surface_held_mutations(spec.name)
        pt = _phases.CURRENT.get() if _phases.ON else None
        if pt is not None:
            _phases.mark(pt, _phases.REPLAY)  # what an uncached call paid before its body
        result = spec.func(*call.args, **call.kwargs)
        if pt is not None:
            _phases.mark(pt, _phases.BODY)
        self._calls.log(
            spec.name,
            cache_hit=False,
            execution_time=_perf_counter() - call.call_start,
            args_hash="",
            cache_key="",
            miss=why,
        )
        return result

    def _try_get_cached(
        self,
        cache_key: str,
        metadata: CacheMetadata | None,
        cached_data: Any,
        call_start: float,
        args_hash: str,
        func_name: str,
        ttl: int | None,
        needs_trace: bool = False,
        arguments: tuple[tuple, dict] | None = None,
        content_keyed: ContentKeyed | None = None,
    ) -> Any:
        """Return cached_data if valid, else CACHE_MISS sentinel.

        An entry stored with the state the call left its arguments in
        (`rewrite_in_place.ArgumentsLeft`) is served only when *arguments*,
        the call's, can take that state, which the hit then writes into them.

        Key-presence is determined by ``metadata is not None`` - the
        backend contract is that absent keys return ``(None, None)``,
        so a non-None metadata view with a ``None`` data value still
        counts as a hit (a function that legitimately returned ``None``).

        Whether a found entry is served is `CallRunner.entry_verdict`.
        """
        if metadata is None:
            self._misses.note_miss(func_name, cache_key, self._misses.absent_entry_reason(func_name, cache_key))
            return CACHE_MISS
        try:
            verdict = self.entry_verdict(cache_key, metadata, ttl, needs_trace=needs_trace)
            writes: list[tuple[Any, Any]] = []
            if verdict is None and isinstance(cached_data, rewrite_in_place.ArgumentsLeft):
                left, why = self._opened(func_name, cached_data, arguments)
                if why is None:
                    writes, why = self._arguments_to_write(func_name, left, arguments, args_hash, content_keyed)
                if why is not None:
                    verdict = MissReason(MissKind.ARGS, why)
                cached_data = left.result
                if isinstance(cached_data, rewrite_in_place.ResultIsArgument) and verdict is None:
                    # The argument handed in, which the writes below fill
                    # with the state the call left it in: a plain run
                    # returns that very object.
                    lives = dict(zip(left.states, (live for live, _ in writes)))
                    if cached_data.name not in lives:
                        verdict = MissReason(MissKind.ARGS, f"the argument {cached_data.name!r} it returns is missing")
                    else:
                        cached_data = lives[cached_data.name]
            if verdict is not None:
                self._misses.note_miss(func_name, cache_key, verdict)
                return CACHE_MISS
            ttl = self._backend_slot.entry_ttl(ttl, metadata)
            # If this hit happens *inside* another cached function's
            # computation, replay the files this entry depends on into the
            # enclosing tracker, so the outer function records them too.
            # Without this, a dependency that was already cached before the
            # consumer's first run hides its file deps behind a cache hit
            # and the consumer never invalidates when that file changes.
            propagate_file_deps_to_active_tracker(metadata, func_name)
            # And the sources its callees resolved, which the verdict just
            # found unchanged: the enclosing call depends on them too.
            if metadata.dynamic_sources or metadata.dynamic_resolvers:
                held = (
                    held_sources(cache_key, metadata.dynamic_sources, self._stored_source)
                    if metadata.dynamic_sources
                    else []
                )
                if held is not None:
                    pass_dynamic_sources_up(
                        [(source, record["token"]) for source, record in zip(held, metadata.dynamic_sources or [])],
                        entry_resolutions(cache_key, metadata.dynamic_resolvers, self._stored_source),
                    )
            # Re-attach the lineage hash to the restored value. It's a plain
            # attribute that doesn't survive pickling, so a value restored
            # from disk would otherwise lose it - and a downstream cached
            # function would fall back to content-hashing under a DIFFERENT
            # key than when the upstream was freshly computed, recomputing
            # needlessly. The hash is deterministic from (cache_key,
            # auto_file_deps), both available here.
            cached_data = self._store.restore_identity(func_name, metadata, cached_data)
            self._store.attach_lineage(cached_data, cache_key, metadata.auto_file_deps, ttl=ttl, func_name=func_name)
            self._store.carry_digest(metadata, cached_data)
            replay_rng_state(metadata)
            if metadata.shared_result:
                # Never silent: a process that only ever hits is told too.
                self._purity.warn_shared_on_hit(func_name, metadata.shared_result)
            self._registry.cached[func_name].last_key = cache_key
            self._calls.log(
                func_name,
                cache_hit=True,
                execution_time=_perf_counter() - call_start,
                args_hash=args_hash,
                cache_key=cache_key,
                time_saved=self._calls.credit_hit(func_name, metadata),
            )
            # Last, once nothing can turn the hit into a miss: a miss runs
            # the body on the arguments as the caller handed them in.
            for live, state in writes:
                rewrite_in_place.rewrite(live, state)
            return cached_data
        except (TypeError, KeyError) as e:
            self._notices.metadata_invalid(func_name, e)
            self._misses.note_miss(
                func_name, cache_key, MissReason(MissKind.INCOMPLETE, "the stored entry's metadata did not validate")
            )
        return CACHE_MISS

    def entry_verdict(
        self,
        cache_key: str,
        metadata: CacheMetadata,
        ttl: int | None,
        *,
        quiet: bool = False,
        needs_trace: bool = False,
    ) -> MissReason | None:
        """Why the entry found under *cache_key* is not served, or None to
        serve it: its age against the ttl, its recorded files, its chunks.

        The one judgement a lookup (`CallRunner._try_get_cached`) and
        ``explain()`` both apply, so a check added here holds for both.
        *quiet* leaves out the warnings the file check gives, for
        ``explain()``. *needs_trace*: the call picks code by a run-time name, so
        an entry without a trace of what it ran is not served. Raises
        ``TypeError``/``KeyError`` on metadata that does not validate.
        """
        ttl = self._backend_slot.entry_ttl(ttl, metadata)
        if ttl_expired(written_at(metadata), ttl):
            age = time.time() - (written_at(metadata) or 0)
            return MissReason(MissKind.TTL, f"the entry is {age:.1f}s old and ttl={ttl}s")
        if not self._files.auto_file_deps_fresh(metadata, quiet=quiet):
            return MissReason(MissKind.FILE, describe_stale_files(metadata))
        if (metadata.dynamic_sources or metadata.dynamic_resolvers) and not dynamic_sources_fresh(
            cache_key, metadata.dynamic_sources or [], self._stored_source, metadata.dynamic_resolvers
        ):
            return MissReason(
                MissKind.DYNAMIC,
                "a dynamic_depends_on= source of a cached function it calls changed, or could not be asked",
            )
        if metadata.env_reads and not environment_unchanged(metadata.env_reads):
            return MissReason(MissKind.ENVIRONMENT, "an environment variable an installed package read changed")
        if metadata.lib_reads:
            moved = library_reads.changed(metadata.lib_reads)
            if moved is not None:
                return MissReason(MissKind.LIBRARY, f"{moved}, a library setting the run read, changed")
        if metadata.warned_under is not None and not warning_state.holds(metadata.warned_under):
            return MissReason(
                MissKind.WARNINGS,
                "the run warned, and the warnings filters or a display hook of the user's changed since",
            )
        if not self._chunks_are_intact(cache_key, metadata):
            return MissReason(MissKind.INCOMPLETE, "a chunk of the stored result is missing")
        if metadata.code_trace is not None:
            with hit_memo.probe_round():
                changed = code_trace.verify(metadata.code_trace, self._cached_state, self._full_closure_state)
            if changed is not None:
                # Kept for a call that needs none too, as a check beside the
                # key: then nothing was picked by name.
                kind = MissKind.TRACE if needs_trace else MissKind.RAN
                return MissReason(kind, f"{changed} since the entry was stored")
        elif needs_trace:
            return MissReason(MissKind.TRACE, "the entry has no trace of the code the call picked by name")
        if metadata.env_reads:
            # Served inside another body being recorded: what it read counts there.
            note_reads(metadata.env_reads)
        if metadata.warned_under is not None:
            warning_state.note_warned()
        if metadata.lib_reads:
            library_reads.note_served(metadata.lib_reads)
        return None

    def _cached_state(self, func_name: str) -> str | None:
        """The state segment of cached function *func_name* now, for a trace
        item naming it (`code_trace`); None when there is none."""
        spec = self._registry.cached.get(func_name)
        if spec is None:
            return None
        try:
            return self._keys.callee_state(spec)
        except Exception:  # noqa: BLE001 - a state that cannot be built is not the one recorded
            return None

    def _full_closure_state(self, func_name: str) -> str | None:
        """`KeyBuilder.full_closure_state` of cached function *func_name*
        now, for a ``full_state`` trace item; None when there is none."""
        spec = self._registry.cached.get(func_name)
        if spec is None:
            return None
        try:
            return self._keys.full_closure_state(spec)
        except Exception:  # noqa: BLE001 - a state that cannot be built is not the one recorded
            return None

    def _closure_check(self, spec: CachedFunction) -> tuple[list | None, str]:
        """The trace an entry of a call keyed with the classes of module data
        by their surface (`BuiltKey.narrowed`) is stored with when its own
        trace is missing or could not be built: its state with those classes
        keyed by all their code (``full_state``), checked again on every
        lookup, so nothing the key left to the trace goes unchecked. ``(None,
        why)`` when that state cannot be built: the call is not stored."""
        state = self._full_closure_state(spec.name)
        if state is None:
            return None, "cash could not key the code of the classes it reads from module data in full"
        return [("full_state", spec.name, state)], ""

    def _stored_source(self, key: str) -> Any:
        """What the backend holds under *key*: a pickled source a caller's
        entry names (`dynamic_sources.source_key`), or None."""
        return self._backend_slot.backend.get(key)[1]

    def _chunks_are_intact(self, cache_key: str, metadata: CacheMetadata) -> bool:
        """True unless this is a chunked manifest missing some of its chunks.

        A manifest can outlive its chunks -- eviction reaches them separately
        -- and served as it is, such an entry returns FEWER items than it
        stored, or none at all, without a word. A truncated answer is worse
        than a slow one, so the entry is treated as absent and recomputed.

        Metadata-only reads: the point is to check presence, not to load the
        payload and undo the laziness chunking exists for.

        Scope, because the docs depend on it: BOTH read paths apply this --
        ``CallRunner._try_get_cached`` for the default one, and the double-checked re-read
        inside ``CallRunner.compute_with_lock`` for ``use_locking=True``, since both go
        through `CallRunner._try_get_cached`. Both paths are pinned by
        ``tests/test_core/results/test_iterator_caching.py``.
        """
        if getattr(metadata, "iterator_storage", None) != "chunked":
            return True
        try:
            prefix = chunk_prefix(cache_key, metadata.chunk_stream)
            for index in range(metadata.n_chunks or 0):
                if self._backend_slot.backend.get_metadata(f"{prefix}:chunk_{index}") is None:
                    logger.debug(
                        "[CORE] chunk %d of %s is missing; treating the entry "
                        "as a miss rather than serving a short result",
                        index,
                        cache_key,
                    )
                    return False
        except Exception:  # noqa: BLE001 - an integrity check must not break a call
            return True
        return True

    def _wrap_iterator_hit(self, call: Call, metadata: CacheMetadata | None, hit: Any) -> Any:
        """Wrap a cache-hit value in the right iterator class.

        Iterators (including the single-chunk case) are stored under
        an ``iterator_storage='chunked'`` manifest plus N chunk
        entries; on hit they're returned as a fresh
        ``ChunkedCachedIterator``, which recomputes the rest from the
        function if a chunk is gone. Non-iterator return types live as
        a single blob and are returned as *hit* directly.

        Every hit path goes through here -- the first lookup, the locked
        re-read (`CallRunner.compute_with_lock`) and the async single-flight follower
        (`CallRunner.single_flight`) -- and all of them pass the call's `recompute`, so a
        missing chunk is recomputed on every path rather than raised.
        """
        if metadata and metadata.iterator_storage == "chunked":
            n_chunks = metadata.n_chunks or 0
            prefix = chunk_prefix(call.cache_key, metadata.chunk_stream)
            returned = hit.get("return_value") if isinstance(hit, dict) else None
            return ChunkedCachedIterator(self._backend_slot, prefix, n_chunks, call.recompute, returned)
        return hit

    def _analyze_dependencies(self, func: Callable[..., Any]) -> None:
        """Populate analysis for *func* + its transitive cached-dependency
        closure, then surface *func*'s own purity issues.

        Populating the WHOLE closure (not just *func*) before the first cache
        key is computed is what keeps the key stable from the very first call.
        The state hash folds in each dependency's purity-report
        ``helper_source_hashes``; filled lazily on each dependency's own first
        call, the key would deepen only after the chain warmed, and a fresh
        process would miss the first call to every cached function even with a
        valid entry on disk.

        Surfacing stays per-function: each dependency warns/raises on its OWN
        first direct call, not here, so eager population doesn't change which
        warnings fire or when.
        """
        self._registry.ensure_closure_analyzed(func)
        func_name = func_key(func)
        report = self._registry.report_for(func, func_name) or PurityReport()
        mode = self._registry.purity_mode(func_name)
        self._purity.surface_purity(func_name, report, mode, per_report=bool(getattr(func, "__closure__", None)))

    def lookup(self, spec: CachedFunction, args: tuple, kwargs: dict, *, async_body: bool) -> Call:
        """Everything a call does before the body: analysis, key, lookup.

        Returns the call's state. ``call.outcome`` is what the wrapper returns
        now -- a hit, or the result of a call that has no key -- or
        ``CACHE_MISS`` when the body has to run.
        """
        func, func_name = spec.func, spec.name
        call = Call(args, kwargs)
        call.fill_mark = getter_fill_mark()
        call.call_start = _perf_counter()
        # Where this call's time goes, when asked for (`cash.phase_times`).
        pt = _phases.CURRENT.get() if _phases.ON else None
        if pt is not None:
            _phases.mark(pt, _phases.WRAPPER)
        if self._registry.needs_surfacing(func, func_name):
            # Double-checked under a per-function lock: the key is built
            # from what this populates, so two threads must not race it.
            # Once per name, and once per closure: another closure from a
            # factory already called is keyed by, and checked against, the
            # helpers it captures, not its sibling's.
            with self._registry.analysis_lock:
                if self._registry.needs_surfacing(func, func_name):
                    self._analyze_dependencies(func)
                    self._registry.mark_surfaced(func, func_name)
        # Inherit the shortest TTL of any TTL'd dependency (computed after
        # analysis populates the graph).
        call.ttl = self._registry.effective_ttl(func_name, spec.ttl)

        def recompute() -> Any:
            # The body runs here with no entry of its own (a stream whose
            # chunk went), under whatever tracker is active.
            note_unentered_body(func_name)
            if async_body:
                return run_to_completion(lambda: func(*args, **kwargs))
            return func(*args, **kwargs)

        call.recompute = recompute

        # Everything from here to the hit/miss verdict is cash's own cost,
        # not the user's work. Two perf_counter pairs cost under 1% of the
        # cheapest possible cached call, so this is not gated behind a
        # heuristic.
        overhead_t0 = _perf_counter()
        if pt is not None:
            _phases.mark(pt, _phases.SETUP)
        # Outside the key build, which turns any exception into "no key": an
        # exception from the body of an uncached call must propagate, not run
        # the body a second time.
        handed = claim_handed_digests()
        if self._verdicts is not None and not async_body and self._known_refusal(spec, call, handed):
            return call
        if pt is not None:
            window = _phases.open_window(pt, _phases.KEY_OTHER)
        # Before the key: a fast hit remembered from this call must not
        # outlive a change made while the key was built (`hit_memo`).
        epochs_before = hit_memo.epochs(spec, args, kwargs)
        token = HANDED_DIGESTS.set(handed) if handed is not None else None
        call.polars_args = CallPolarsDigests()
        polars_token = CALL_POLARS.set(call.polars_args)
        copies = self._keys.split_for_key(spec, args, kwargs)
        call.keyed_parts = {}
        keyed_token = KEYED_PARTS.set(call.keyed_parts)
        call.content_keyed = ContentKeyed()
        content_token = CONTENT_KEYED.set(call.content_keyed)
        try:
            if copies is None:
                built, call.capture_watch, call.ledger = self._keys.resolve(spec, args, kwargs)
            else:
                split = frozenset(id(copy) for copy in (*copies.args, *copies.kwargs.values()))
                built, call.capture_watch, call.ledger = self._keys.resolve(
                    spec, copies.args, copies.kwargs, split, refused=copies.refused
                )
        finally:
            CONTENT_KEYED.reset(content_token)
            KEYED_PARTS.reset(keyed_token)
            CALL_POLARS.reset(polars_token)
            if token is not None:
                HANDED_DIGESTS.reset(token)
            if copies is not None:
                args, kwargs = copies.body_arguments(args, kwargs)
                call.args, call.kwargs = args, kwargs
        call.watch_identities = self._purity.watched_identities(spec.func, call.capture_watch)
        if pt is not None:
            _phases.close_window(pt, window)
        if handed is not None:
            handed.ran_body = True  # until the call is seen to be a plain hit
            call.handed = handed
        if isinstance(built, Unkeyable):
            self._stop_probe(func_name, call)
            call.outcome = self._run_uncached(spec, call, built.reason)
            return call
        call.cache_key, call.state_hash, call.args_hash = built.cache_key, built.state_hash, built.args_hash
        call.traced, call.needs_trace = built.traced, built.needs_trace
        call.narrowed = built.narrowed
        call.folded_code = built.folded_code
        call.unread = built.unread
        call.trace_for_unread = built.trace_for_unread
        if built.reached_ttl is not None:
            # A cached function the key reached as a value, not by a graph
            # edge: its ttl is per call, not per function.
            call.ttl = built.reached_ttl if call.ttl is None else min(call.ttl, built.reached_ttl)

        if pt is not None:
            window = _phases.open_window(pt, _phases.READ_FETCH)
        raw_metadata, cached_data = self._backend_slot.read(call.cache_key)
        call.metadata = CacheMetadata.from_dict(raw_metadata) if raw_metadata is not None else None
        if pt is not None:
            _phases.close_window(pt, window)
        hit = self._try_get_cached(
            call.cache_key,
            call.metadata,
            cached_data,
            call.call_start,
            call.args_hash,
            func_name,
            call.ttl,
            # One stored by its key when the trace a lazy import asked for
            # was not needed (`finish_miss`) is served as such.
            call.needs_trace and not call.trace_for_unread,
            (args, kwargs),
            call.content_keyed,
        )
        call.cash_overhead = _perf_counter() - overhead_t0
        if pt is not None:
            _phases.mark(pt, _phases.CHECKS)
        if hit is not CACHE_MISS:
            self._stop_probe(func_name, call)
            if code_trace.active():
                # Served inside a traced call: its body did not run, so what
                # it stands for goes in that call's trace.
                code_trace.note_cached_hit(
                    func_name,
                    plain_getattr(spec.func, "__code__", None),
                    self._cached_state(func_name),
                    call.metadata.code_trace,
                )
            self._replay_argument_rng(func_name, args, kwargs, call.metadata)
            self._calls.note_effectiveness(
                func_name,
                call.cash_overhead,
                body_seconds=self._calls.credited_body(call.metadata),
                was_hit=True,
            )
            call.outcome = self._wrap_iterator_hit(call, call.metadata, hit)
            if (
                epochs_before is not None and handed is None and call.outcome is hit and not async_body
                # An entry with a trace is remembered once what the trace
                # names is watched (`hit_memo.scan_trace`).
            ):
                memo = hit_memo.memo_for(spec, self)
                if memo is not None:
                    memo.offer(call, args, kwargs, hit, epochs_before)
            if handed is not None and call.outcome is hit:
                # Served: the body does not run, nor anything else of the
                # user's until the statement binds the result.
                handed.ran_body = False
                self._hand_back_entry(call, hit, call.metadata, getattr(call.metadata, "storage", None))
        else:
            if call.needs_trace and call.narrowed and copies is None:
                # Not for an iterator argument, whose key copy is read (that
                # call's strict trace judges against its narrowed key alone).
                # Built after the body only when the strict trace refuses
                # what ran against the narrowed key's code (a fold of every
                # held class's code, hundreds of ms for a library's), and
                # only while nothing it reads moved since here; before the
                # body for a function that once moved it.
                gate = None if spec.name in self._full_fold_first else self._full_fold_gate()
                if gate is None:
                    call.full_folded = self._keys.full_key_folded(spec, args, kwargs)
                else:
                    call.full_fold_gate = gate
            # Before the body runs: the mutation check compares against it.
            with call_polars(call.polars_args):
                if copies is not None:
                    call.checked = copies.checked_arguments(args, kwargs)
                if call.content_keyed is not None and call.content_keyed.tagged:
                    # The key holds an argument by its statement tag, which
                    # the body's changes do not move: hashed by content.
                    call.call_args_hash = self._content_args_hash(spec, call)
                if not call.by_content:
                    if copies is None:
                        call.call_args_hash = self._keys.call_args_hash(spec, args, kwargs, built)
                    else:
                        call.call_args_hash = self._keys.checked_arguments_hash(spec, *call.checked)
            call.polars_args.sign()
        if pt is not None:
            _phases.mark(pt, _phases.REPLAY)
        return call

    @staticmethod
    def _attribute_changes_seen(observer: Any, attrs: attribute_replay.AttrVerdict | None) -> None:
        """Name the arguments whose attributes the body was recorded binding
        or deleting as changed, when the hash after the body did not, for a
        call whose key holds an argument by its statement tag and which
        could not hash its arguments by content (`_content_args_hash`):
        the tag alone does not move when the body changes it."""
        if observer is None or attrs is None or not attrs.changes or observer.mutated_args:
            return
        observer.mutated_args = attrs.names
        observer.record(
            "argument mutation",
            f"the call changed {', '.join(repr(n) for n in attrs.names)} in place -- "
            f"the result was not stored, so this call runs every time",
        )

    def _content_args_hash(self, spec: CachedFunction, call: Call) -> str | None:
        """The arguments' hash by their content (`content_only`), with
        ``call.by_content`` set, or None when they cannot be hashed so: the
        check then compares the tags, and only an attribute change the
        recording sees (`CallRunner.attribute_changes`) shows."""
        try:
            with content_only():
                found = self._keys.checked_arguments_hash(spec, *(call.checked or (call.args, call.kwargs)))
        except Exception:  # not hashed by content: the tags stand
            logger.debug("[PURITY] %s: arguments not hashed by content", spec.name, exc_info=True)
            return None
        call.by_content = found is not None
        return found

    def _known_refusal(self, spec: CachedFunction, call: Call, handed: Any) -> bool:
        """Run *call* uncached when an earlier call with the same code and
        argument types changed an argument in a way a hit cannot replay
        (`refusal_verdicts`): its result would not be stored, and building
        its key and checking it after the body is work thrown away. True
        when *call* ran; else the call is keyed as any other, and keeps or
        drops a verdict once checked (`_settle_verdict`)."""
        assert self._verdicts is not None
        func_name = spec.name
        try:
            self._backend_slot.backend  # the record lives beside the cache built for the lookup
        except Exception:  # noqa: BLE001 - the lookup itself says why there is no backend
            return False
        call.refusal_probe = True
        call.args_fingerprinted = spec.changes_arguments
        self._notices.record_warnings(func_name)
        if not self._verdicts.may_have(func_name):
            return False
        key = self._verdict_key(spec, call)
        if key is None:
            return False
        call.refusal_key = key
        verdict = self._verdicts.use(func_name, key)
        if verdict is None:
            return False
        self._stop_probe(func_name, call)
        if handed is not None:
            handed.ran_body = True
            call.handed = handed
        self._notices.replay_warnings(func_name, verdict.get("warnings") or [])
        refused = str(verdict.get("why") or "it changed its arguments in place")
        why = MissReason(MissKind.NOT_STORED, refused)
        note_unentered_body(func_name)
        # Timed as the uncached call's body (`_run_uncached`): reported
        # with the wrapper, the body of every such call counted as cash's.
        pt = _phases.CURRENT.get() if _phases.ON else None
        if pt is not None:
            _phases.mark(pt, _phases.REPLAY)
        call.outcome = spec.func(*call.args, **call.kwargs)
        if pt is not None:
            _phases.mark(pt, _phases.BODY)
        self._calls.log(
            func_name,
            cache_hit=False,
            execution_time=_perf_counter() - call.call_start,
            args_hash="",
            cache_key="",
            miss=why,
            not_stored=refused,
        )
        return True

    def _verdict_key(self, spec: CachedFunction, call: Call) -> str | None:
        """*call*'s `RefusalVerdicts.key_for`."""
        assert self._verdicts is not None
        report = self._registry.report_for(spec.func, spec.name)
        mode = self._registry.purity_mode(spec.name)
        return self._verdicts.key_for(spec, report, mode, call.args, call.kwargs)

    def _stop_probe(self, func_name: str, call: Call) -> None:
        """A call that is not checked after a body: it keeps no verdict."""
        if call.refusal_probe:
            self._notices.recorded_warnings(func_name)
            call.refusal_probe = False

    def _settle_verdict(self, spec: CachedFunction, call: Call, refusal: str | None, observer: Any = None) -> None:
        """Keep or drop the verdict of a checked call (`_known_refusal`).
        *observer* is the call's `EffectObserver`, which says whether a
        later call may be stored (`arguments_left`)."""
        if not call.refusal_probe or self._verdicts is None:
            return
        func_name = spec.name
        warnings = self._notices.recorded_warnings(func_name) or []
        call.refusal_probe = False
        if not isinstance(refusal, ArgumentRefusal):
            if call.refusal_key is not None:
                self._verdicts.drop(func_name, call.refusal_key)
            return
        earlier = self._first_refusal_warnings.pop(func_name, [])
        final = observer is not None and bool(observer.mutated_args) and not observer.next_may_store
        if not (refusal.unreplayable or call.args_fingerprinted or final):
            # The first change seen, on an argument nothing was taken of
            # before the body: the next call fingerprints it and may be
            # stored (`arguments_left`), so it says nothing yet. What it
            # warned goes with the verdict that call may leave. Not when
            # no later call can write the argument back either (*final*):
            # the next is refused too, and would build its key for nothing.
            self._first_refusal_warnings[func_name] = warnings
            return
        # Made from what the body changed: the types it was handed stay,
        # and the flags and strings in the key cannot change in place.
        key = call.refusal_key or self._verdict_key(spec, call)
        if key is not None:
            self._verdicts.refused(func_name, key, refusal, [*earlier, *warnings])

    def _hand_back_entry(self, call: Call, value: Any, metadata: Any, storage: list[str] | None) -> None:
        """Tell the notebook statement this call is (`cash.handed_digests`)
        which entry its result *value* is in, so the statement's own entry
        refers to it instead of holding a second copy: a 200 MB design matrix
        was copied and written twice, once per entry. Not for a result whose
        hit puts a held object back in place (`restore_identity`), nor one
        only this process can read."""
        handed = call.handed
        if handed is None or not handed.binds_result or not storage:
            return
        if metadata is not None and (
            getattr(metadata, "result_ref", None)
            or getattr(metadata, "held_refs", None)
            or getattr(metadata, "process_local", None)
            # Stored with the state it left its arguments in: the entry
            # holds more than the result, and a hit writes it back.
            or getattr(metadata, "args_left", None)
        ):
            return
        backend = self._backend_slot.backend
        tiers = getattr(backend, "backends", None)
        persisted = any(tier != "RAM" for tier in storage) or not (isinstance(tiers, list) and len(tiers) > 1)
        try:
            result = _weak.ref(value)
        except TypeError:  # a builtin value: its statement keeps its own copy
            return
        handed.stored = Stored(result, call.cache_key, backend, persisted)

    def _reread(self, spec: CachedFunction, call: Call) -> Any:
        """Look the key up again (another caller may have stored it meanwhile):
        the hit, wrapped like any other, or ``CACHE_MISS``.

        The SAME validity test as the first lookup, by calling the same
        function -- not a hand-rolled subset of it, which would lose a check
        (``CallRunner._chunks_are_intact``, ``FileDeps.auto_file_deps_fresh``) as they are added.
        One function decides whether an entry may be served.
        """
        raw_metadata, cached_data = self._backend_slot.read(call.cache_key)
        if raw_metadata is None:
            return CACHE_MISS
        metadata = CacheMetadata.from_dict(raw_metadata)
        hit = self._try_get_cached(
            call.cache_key,
            metadata,
            cached_data,
            call.call_start,
            call.args_hash,
            spec.name,
            call.ttl,
            call.needs_trace and not call.trace_for_unread,
            (call.args, call.kwargs),
            call.content_keyed,
        )
        if hit is CACHE_MISS:
            return CACHE_MISS
        self._replay_argument_rng(spec.name, call.args, call.kwargs, metadata)
        return self._wrap_iterator_hit(call, metadata, hit)

    def _checked_named(self, func_name: str, args: tuple, kwargs: dict) -> dict[str, Any] | None:
        """``{parameter: value}`` of the arguments the in-place-change check
        looks at, named as it names them (`PurityChecks.named_checked_arguments`),
        or None when they cannot be bound."""
        try:
            return self._purity.named_checked_arguments(func_name, args, kwargs)
        except Exception:  # noqa: BLE001 - unbindable: nothing is written back
            return None

    def arguments_before(self, spec: CachedFunction, call: Call, args: tuple, kwargs: dict) -> dict[str, tuple]:
        """``{parameter: (value, shape, before)}`` of each argument a hit
        could write a stored state back into (`rewrite_in_place`), taken
        before the body runs. *before* is what a hit checks the argument
        handed in against:

        * `rewrite_in_place.KeyedBefore`, the key's argument hash, for an
          argument the key holds by all it holds (`arg_hashing.ContentKeyed`):
          a hit's key match already says it is in the state the entry was
          stored from. Nothing is hashed for it, so even the first call
          seen changing it is stored.
        * a `mutation_fingerprint` for any other argument, once the function
          has been seen changing an argument in place: until then a call
          that changes nothing pays nothing.

        Neither is taken for a function whose misses so far changed nothing
        (`CachedFunction.leaves_arguments_alone`): one that never changes
        its arguments pays for this on its first miss alone, and one that
        starts changing them later is stored from the miss after.

        A call keyed by ``key=``, or on copies of its arguments, is never
        stored with them."""
        if call.checked is not None:
            return {}
        arg_key = getattr(spec, "arg_key", None)
        if arg_key is not None and arg_key.key_fn is not None:
            return {}
        if not spec.changes_arguments and spec.leaves_arguments_alone:
            return {}
        content = call.content_keyed
        keyed = content is not None and call.args_hash is not None
        if not spec.changes_arguments and not (keyed and any(map(_takes_state, (*args, *kwargs.values())))):
            return {}
        named = self._checked_named(spec.name, args, kwargs) or {}
        before: dict[str, tuple] = {}
        for name, value in named.items():
            if not rewrite_in_place.rewritable_kind(value):
                if keyed and _replays_attributes(value) and content.holds(value):
                    # An instance the key holds by all it holds: which of
                    # its attributes the body binds is recorded, for a hit
                    # to replay (`attribute_changes`).
                    try:
                        recording = rewrite_in_place.AttrRecording(value)
                    except Exception:  # noqa: BLE001 - not recorded: a change refuses the store
                        continue
                    before[name] = (value, recording, rewrite_in_place.KeyedBefore(call.args_hash))
                continue
            if keyed and content.covers(value):
                try:
                    # The memo's hold on a frame is let go only for a
                    # function seen changing its arguments, or whose source
                    # changes this one: one that changes nothing would pay
                    # a second read of the frame for it.
                    release = spec.changes_arguments or self._purity.reads_as_changing(spec.name, name)
                    before[name] = (
                        value,
                        rewrite_in_place.before_call(value, release=release),
                        rewrite_in_place.KeyedBefore(call.args_hash),
                    )
                except Exception:  # noqa: BLE001 - no shape: not stored with the result
                    pass
                continue
            if not spec.changes_arguments or rewrite_in_place.refusal(value) is not None:
                continue
            try:
                fingerprint = mutation_fingerprint(value)
                shape = rewrite_in_place.before_call(value)
            except Exception:  # noqa: BLE001 - not fingerprinted: not stored with the result
                continue
            if fingerprint is not None:
                before[name] = (value, shape, fingerprint)
        return before

    def attribute_changes(self, spec: CachedFunction, call: Call, run: BodyRun) -> attribute_replay.AttrVerdict | None:
        """What the body did to the attributes of the instances it was
        handed, and whether a hit can replay it (`attribute_replay`); None
        when it did nothing to them."""
        arguments = call.checked or (call.args, call.kwargs)
        func, func_name = spec.func, spec.name

        def unchanged() -> bool:
            with content_only(call.by_content):
                after = self._purity.checked_arguments_hash(func_name, *arguments)
            return after == call.call_args_hash and not self._purity.bound_moved(run.observer)

        try:
            verdict = attribute_replay.AttrVerdict.of(
                run,
                func,
                func_name,
                [*self._named_args(func_name, *arguments), *bound_values(func)],
                call.call_args_hash,
                unchanged,
                lambda value: self._purity.shared_outside(value, func, call.args, call.kwargs),
                lambda obj: self._purity.held_outside(
                    obj,
                    func,
                    # The instance a bound method is bound to is held by it,
                    # as it should be.
                    [
                        (what, value)
                        for what, value in self._reads_now(func, call.ran_unit)
                        if not (value is obj and any(held is obj for _, held in bound_values(func)))
                    ],
                ),
                lambda result: self._purity.warn_shared_result(func, func_name, result, call.args, call.kwargs),
            )
        except Exception:  # not judged: the change refuses the store, as without it
            logger.debug("[ATTRS] %s: attribute changes not judged", func_name, exc_info=True)
            return None
        if verdict is not None and verdict.why is not None and verdict.why.startswith(attribute_replay.ALSO_CHANGED):
            named = self._source_changes(spec, verdict.names, verdict.fresh_attributes())
            if named:
                verdict = verdict._replace(why=attribute_replay.also_changed({}, named))
        return verdict

    def _source_changes(
        self, spec: CachedFunction, params: list[str], fresh: dict[str, set[str]] | None = None
    ) -> list[str]:
        """The places in *spec*'s own body that reading its source found
        changing one of *params* in place other than by binding a top-level
        attribute (``self.history.append (line 12)``): what keeps a hit from
        replaying the attributes the call bound. A write into an attribute
        the call itself bound to a new object (*fresh*, ``{parameter:
        attributes}``: ``self.null_ = {}`` then ``self.null_[k] = v``)
        changed nothing the caller had, and is not one."""
        report = self._registry.report_for(spec.func, spec.name)
        own = {spec.name, plain_getattr(spec.func, "__qualname__", None)}
        found = []
        for issue in getattr(report, "issues", None) or ():
            if getattr(issue, "where", None) not in own or mutated_parameter(issue) not in params:
                continue
            if attribute_bound_parameter(issue) is not None:
                continue
            what = getattr(issue, "description", "").split(" - ", 1)[0].removesuffix("()")
            into = _WRITE_INTO_ATTRIBUTE.match(what)
            if into is not None and fresh and into.group(2) in fresh.get(into.group(1), ()):
                continue
            line = getattr(issue, "line", 0)
            found.append(f"{what} (line {line})" if line else what)
        return found

    @staticmethod
    def _reads_now(func: Callable, ran: Any) -> list[tuple[str, Any]]:
        """``(what, value)`` of what the code a call ran read outside its
        arguments (`argument_sharing.read_values`), or none when that was not
        recorded."""
        try:
            return list(argument_sharing.read_values(argument_sharing.reads_of(ran, func), func))
        except argument_sharing.Unknown:
            return []

    def _record_bound(self, spec: CachedFunction, call: Call, run: BodyRun) -> None:
        """Record the attributes the body binds on the instance the cached
        callable is bound to (``c.cache(est.fit)``), when the key holds it
        by all it holds (`PurityChecks.bound_by_content`): a hit replays
        them, as for an instance handed in (`arguments_before`)."""
        if call.checked is not None or (not spec.changes_arguments and spec.leaves_arguments_alone):
            return
        if not any(name == "self" and _replays_attributes(value) for name, value in bound_values(spec.func)):
            return
        for name, value in self._purity.bound_by_content(spec.func):
            if name in run.args_before or not _replays_attributes(value):
                continue
            try:
                recording = rewrite_in_place.AttrRecording(value)
            except Exception:  # noqa: BLE001 - not recorded: a change refuses the store
                continue
            run.args_before[name] = (value, recording, rewrite_in_place.BoundBefore(name))

    def arguments_left(
        self,
        spec: CachedFunction,
        run: BodyRun,
        arguments: tuple[tuple, dict],
        ran: Any = None,
        attrs: attribute_replay.AttrVerdict | None = None,
    ) -> dict[str, tuple[Any, Any]] | None:
        """``{parameter: (before, state after)}`` when the body changed
        arguments in place and every one of them can be written back by a
        hit (`arguments_before`): the result is then stored with them, and
        the change no longer refuses the store; the warning says what a hit
        does instead once it is stored (`CallRunner.finish_miss`). None
        otherwise, and the refusal stands.

        Nor when an argument it changed shares state with anything else the
        call sees, or cash cannot tell whether it does (`_shared_with`):
        *ran* is the record of the code the body ran, whose reads
        (`argument_sharing.reads_of`) are kept in ``run.args_reads`` for
        a hit to check again."""
        run.args_reads = None
        observer = run.observer
        mutated = observer.mutated_args if observer is not None else None
        if not mutated:
            if observer is not None:
                spec.leaves_arguments_alone = True
            return None
        learnt = not spec.changes_arguments
        spec.changes_arguments = True
        before = getattr(run, "args_before", None) or {}
        if any(name not in before for name in mutated):
            if learnt:
                # Its first change seen, on an argument nothing was taken of
                # before the body: this result is not stored; the next
                # calls fingerprint it first and are.
                named = self._checked_named(spec.name, *arguments) or {}
                values = [named.get(name) for name in mutated]
                if all(
                    (rewrite_in_place.rewritable_kind(v) and rewrite_in_place.refusal(v) is None)
                    or _replays_attributes(v)
                    for v in values
                ):
                    observer.stored_from_next = list(mutated)
                    observer.next_may_store = True
                    observer.reword(
                        "argument mutation",
                        "so this call runs every time",
                        "so this call ran; from the next call on, the state it leaves the argument in is stored "
                        "with the result and a hit writes it back",
                    )
            return None
        left: dict[str, tuple[Any, Any]] = {}
        replayed = [name for name in mutated if type(before[name][1]) is rewrite_in_place.AttrRecording]
        roots = (attrs.roots if attrs is not None else None) or {}
        if replayed:
            states = attrs.states if attrs is not None else None
            if states is None or any(name not in states and name not in roots.values() for name in replayed):
                if attrs is not None and attrs.why:
                    observer.unreplayable = attrs.why
                    observer.reword(
                        "argument mutation",
                        " in place -- the result was not stored",
                        f" in place, and a hit cannot replay it: {attrs.why} -- the result was not stored",
                    )
                return None
            for name, state in states.items():
                # The argument's own attributes, then those of what it held
                # one level down (`HeldBefore`), found again on a hit where
                # it held them.
                if roots.get(name, name) in replayed:
                    left[name] = (before[name][2], state)
        for name in mutated:
            if name in left or name in replayed:
                continue
            value, shape, fingerprint = before[name]
            if rewrite_in_place.refusal(value) is not None or rewrite_in_place.shape_change(shape, value) is not None:
                return None
            try:
                left[name] = (fingerprint, rewrite_in_place.snapshot(value, shape.arrays))
            except rewrite_in_place.HeldByCash:
                # Its first change seen, under the memo's hold: the next
                # call lets go of the frame first and is stored.
                observer.next_may_store = True
                observer.reword(
                    "argument mutation",
                    "so this call runs every time",
                    "so this call ran; from the next call on, the state it leaves the argument in is stored "
                    "with the result and a hit writes it back",
                )
                return None
            except Exception:  # noqa: BLE001 - not copied: not stored
                return None
        written = [name for name in left if name not in replayed and name not in roots]
        try:
            # Attributes rebound share no memory a write could reach: only
            # what is written into is looked at.
            reads = argument_sharing.reads_of(ran, spec.func) if written else ()
            shared = self._shared_with(spec.func, [(name, before[name][0]) for name in written], arguments, reads)
            if shared is not None:
                shared = (
                    f", and it shares memory with {shared}, which a hit writing it back would leave as no plain "
                    f"run does"
                )
        except argument_sharing.Unknown as why:
            shared = f", and cash cannot tell whether it shares memory with what else the call sees: {why}"
        if shared is not None:
            observer.reword(
                "argument mutation",
                " in place -- the result was not stored",
                f" in place{shared} -- the result was not stored",
            )
            return None
        run.args_reads = reads
        observer.mutated_args = None
        observer.forget("argument mutation")
        return left

    def result_argument(
        self, func_name: str, result: Any, args_left: dict[str, tuple[Any, Any]] | None, arguments: tuple[tuple, dict]
    ) -> str | None:
        """The parameter in *args_left* whose argument *result* is, by
        identity, or None. Such an argument is a NumPy array, a pandas
        frame or Series or a plain container (`arguments_left` keeps no
        other), whose state the entry keeps and a hit writes back in place,
        so a hit can hand back the argument itself as a plain run does; or
        an instance whose attributes the entry keeps, an argument or one an
        argument holds one level down (``return self.kernel``), named as
        *args_left* names it."""
        if not args_left:
            return None
        named = self._checked_named(func_name, *arguments) or {}
        spec = self._registry.cached.get(func_name)
        if spec is not None and any(
            isinstance(_outer(before), rewrite_in_place.BoundBefore) for before, _ in args_left.values()
        ):
            named = {**dict(bound_values(spec.func)), **named}
        for name, (before, _state) in args_left.items():
            if isinstance(before, rewrite_in_place.HeldBefore):
                # ``return self.kernel``: what the argument holds there, as
                # a hit finds it again.
                owner = named.get(before.param)
                held = plain_getattr(owner, "__dict__", None) if owner is not None else None
                if type(held) is dict and held.get(before.name, _NOT_HELD) is result:
                    return name
            elif name in named and named[name] is result:
                return name
        return None

    def _unwritable_args(
        self, func_name: str, arguments: tuple[tuple, dict], content: ContentKeyed | None = None
    ) -> list[str]:
        """The parameters whose argument no hit writes a state back into
        (`rewrite_in_place.refusal`), so a change cash may not see in it --
        an object keyed by ``__cash_key__`` or a registered hasher hashes
        alike before and after -- is not made by a hit. Every parameter
        when they cannot be bound."""
        named = self._checked_named(func_name, *arguments)
        if named is None:
            return [name for name, _ in self._named_args(func_name, *arguments)] or ["an argument"]
        return [
            name
            for name, value in named.items()
            if (not rewrite_in_place.rewritable_kind(value) or rewrite_in_place.refusal(value) is not None)
            and not (content is not None and _replays_attributes(value) and content.holds(value))
        ]

    def _shared_with(
        self, func: Callable, changed: list[tuple[str, Any]], arguments: tuple[tuple, dict], reads: tuple
    ) -> str | None:
        """What an argument in *changed*, ``(parameter, value)`` of those
        the call changes in place, is, is held by, or shares memory with,
        if anything: another argument, by its parameter's name, or a value
        the code the call ran reads (*reads*, `argument_sharing.reads_of`).
        The key hashes each argument by its content alone, so ``f(a,
        a[::-1])`` keys like ``f(a, b)`` with ``b`` an equal, independent
        array while the body sees different data, and a hit writing ``a``
        back would not leave the view as a plain run does. Exact, or
        `argument_sharing.Unknown` says why it cannot be."""
        bound = self._bound_arguments(func, arguments)
        values = None
        for name, value in changed:
            targets = argument_sharing.Targets(value)
            itself = False
            for other_name, other in bound:
                if other is value and not itself and other_name in (name, None):
                    itself = True  # the argument itself, once
                    continue
                if other_name == name and isinstance(other, tuple) and any(item is value for item in other):
                    # Its own ``*args``: the others in it are looked at.
                    other = tuple(item for item in other if item is not value)
                if argument_sharing.shared_in(targets, other):
                    return f"the argument {other_name!r}" if other_name else "another argument"
            if values is None:
                values = list(argument_sharing.read_values(reads, func))
            for what, other in values:
                if argument_sharing.shared_in(targets, other):
                    return what
        return None

    @staticmethod
    def _bound_arguments(func: Callable, arguments: tuple[tuple, dict]) -> list[tuple[str | None, Any]]:
        """``(parameter, value)`` of each argument as the signature binds
        it (``*args`` as one tuple); ``(None, value)`` each when it does not
        bind."""
        args, kwargs = arguments
        try:
            return list(inspect.signature(func).bind(*args, **kwargs).arguments.items())
        except (TypeError, ValueError):
            return [(None, value) for value in (*args, *kwargs.values())]

    def _opened(
        self, func_name: str, left: rewrite_in_place.ArgumentsLeft, arguments: tuple[tuple, dict] | None
    ) -> tuple[rewrite_in_place.ArgumentsLeft, str | None]:
        """*left* with its sealed parts unpickled onto the arguments this call
        is handed (`rewrite_in_place.seal`), or why they cannot be."""
        if left.sealed is None:
            return left, None
        if arguments is None:
            return left, "the arguments it changes in place could not be found"
        # Named as `CallRunner.attribute_changes` named them for the seal.
        spec = self._registry.cached.get(func_name)
        named = {
            **dict(bound_values(spec.func) if spec is not None else ()),
            **dict(self._named_args(func_name, *arguments)),
        }
        try:
            return left.opened(lambda relink: attribute_replay.find(relink, named)), None
        except LookupError:
            return left, "an argument an object it bound holds is not there"
        except Exception:  # not unpickled: not served
            logger.debug("[ATTRS] %s: sealed entry not opened", func_name, exc_info=True)
            return left, "the objects it bound could not be unpickled"

    def _arguments_to_write(
        self,
        func_name: str,
        left: rewrite_in_place.ArgumentsLeft,
        arguments: tuple[tuple, dict] | None,
        args_hash: str | None,
        content_keyed: ContentKeyed | None,
    ) -> tuple[list[tuple[Any, Any]], str | None]:
        """``(writes, why)``: the ``(argument, state)`` pairs a hit on *left*
        writes, or why it cannot: an argument missing, of another shape, or
        not in the state the entry was stored from. An argument stored with
        `rewrite_in_place.KeyedBefore` is in that state when this call's key,
        whose argument hash is *args_hash*, holds it by its content too
        (*content_keyed*) and that hash is the one stored; any other when its
        `mutation_fingerprint` is the stored one. All are checked before any
        is written."""
        named = self._checked_named(func_name, *arguments) if arguments is not None else None
        if named is None:
            return [], "the arguments it changes in place could not be found"
        writes: list[tuple[Any, Any]] = []
        bound: dict[str, Any] | None = None
        spec = self._registry.cached.get(func_name)
        if spec is not None and any(
            isinstance(_outer(before), rewrite_in_place.BoundBefore) for before, _ in left.states.values()
        ):
            # The instance the callable is bound to, which the key folded.
            named = {**dict(bound_values(spec.func)), **named}
        for name, (before, state) in left.states.items():
            # An instance an argument held one level down (`HeldBefore`) is
            # found where it held it, before anything is written; its state
            # is in the key as part of that argument's.
            held = before if isinstance(before, rewrite_in_place.HeldBefore) else None
            owner = held.param if held is not None else name
            if owner not in named:
                return [], f"the argument {owner!r} it changes in place was not handed in"
            before = _outer(before)
            live = named[owner]
            if held is not None:
                try:
                    live = attribute_replay.held_attribute(
                        live, held.name, owner, "the object whose attributes it bound"
                    )
                except attribute_replay.Unresolved as exc:
                    return [], f"the state the call left {name!r} in cannot be written back: {exc}"
            why = rewrite_in_place.mismatch(live, state)
            if why is None:
                if isinstance(before, rewrite_in_place.BoundBefore):
                    # Its whole state is in the key this hit matched.
                    same = True
                elif isinstance(before, rewrite_in_place.KeyedBefore):
                    holder = named[owner]
                    same = content_keyed is not None and content_keyed.holds(holder) and before.args_hash == args_hash
                else:
                    same = mutation_fingerprint(live) == before
                if not same:
                    why = "it is not in the state the entry was stored from"
            if why is None and isinstance(state, rewrite_in_place.AttrState) and state.relinks():
                if bound is None:
                    bound = {
                        **dict(bound_values(spec.func) if spec is not None else ()),
                        **dict(self._named_args(func_name, *arguments)),
                    }
                try:
                    state = state.resolved(lambda relink: attribute_replay.find(relink, bound))
                except attribute_replay.Unresolved as exc:
                    why = str(exc)
                except LookupError:
                    why = "an object it bound by identity is not there"
            if why is not None:
                return [], f"the state the call left {name!r} in cannot be written back: {why}"
            writes.append((live, state))
        written = [
            name for name, (_, state) in left.states.items() if not isinstance(state, rewrite_in_place.AttrState)
        ]
        if not written:
            return writes, None
        if spec is None or left.reads is None:
            return [], "the entry does not say what the call it was stored from read"
        live_named = [(name, named[name]) for name in written]
        try:
            shared = self._shared_with(spec.func, live_named, arguments, left.reads)
        except argument_sharing.Unknown as why:
            return [], f"cash cannot tell whether an argument it changes in place shares memory: {why}"
        if shared is not None:
            return [], f"an argument it changes in place shares memory with {shared}"
        return writes, None

    def _named_args(self, func_name: str, args: tuple, kwargs: dict) -> list[tuple[str, Any]]:
        """``(parameter, value)`` for each argument of the call, bound to the
        signature as the key binds it, so two spellings of one call agree."""
        try:
            canon_args, canon_kwargs = self._args.normalize_call_args(func_name, args, kwargs)
        except Exception:  # noqa: BLE001 - unbindable: positions are all there is
            canon_args, canon_kwargs = args, kwargs
        return [(f"*args[{i}]", v) for i, v in enumerate(canon_args)] + list(canon_kwargs.items())

    def _replay_argument_rng(self, func_name: str, args: tuple, kwargs: dict, metadata: Any) -> None:
        """Move a generator handed in as an argument to where the computed
        call left it (`RngWatch.replay_parts`)."""
        moved = (getattr(metadata, "rng_replay", None) or {}).get("arg_carriers")
        if moved:
            replay_argument_carriers(moved, dict(self._named_args(func_name, args, kwargs)))

    @contextlib.contextmanager
    def body_scope(self, spec: CachedFunction, call: Call) -> Iterator[BodyRun]:
        """Run the body inside this: file tracking, effect observation, RNG
        watch and timing, shared by the sync and async wrappers.

        The caller runs the body in the ``with`` block and puts what it
        returned in ``run.res``; on the way out this measures the body and
        reads what it drew, still inside the tracker. A body that raises is
        logged as such and the exception propagates.
        """
        # Wrap the function call in FileAccessTracker so any auto-tracked
        # file reads (pandas/numpy/joblib/open/...) are recorded as implicit
        # cache dependencies - a later content change forces a recompute.
        func, func_name, args, kwargs = spec.func, spec.name, call.args, call.kwargs
        pt = _phases.CURRENT.get() if _phases.ON else None
        run = BodyRun()
        run.tracker = FileAccessTracker(
            plain_getattr(func, "__globals__", None), propagate_to_parent=True, hash_on_read=True
        )
        # Watch for side effects the STATIC analyzer cannot see, which is
        # anything happening inside an installed library. Only on this
        # (missing) path: a hit runs no body, so there is nothing to observe
        # and nothing to pay for.
        run.observer = self._purity.make_effect_observer()
        with content_only(call.by_content):
            run.observer.arg_snapshot = self._purity.argument_snapshot(func_name, args, kwargs, call.content_keyed)
        run.observer.arg_identities, run.observer.arg_identities_cover_all = self._purity.argument_identities(
            func_name, args, kwargs, call.keyed_parts
        )
        run.observer.held_generators = self._purity.held_generators(func_name, args, kwargs)
        run.observer.bound_before = self._purity.bound_state(func)
        # Watch the global RNG across the call: a draw inside the body is an
        # input the key cannot see statically.
        run.rng_pre = capture_rng_pre_state()
        # And the generators it can reach through its globals and closure
        # (`rng = np.random.default_rng(42)` at module level, drawn from
        # inside): the module channel above cannot see those move.
        run.carriers_pre = capture_reachable_carrier_states(func)
        run.carriers_moved = []
        # Or handed in as an argument (`boot(x, rng)`): a hit must move the
        # caller's generator on just the same.
        run.arg_carriers_pre = capture_argument_carrier_states(self._named_args(func_name, args, kwargs))
        run.arg_carriers_moved = []
        run.args_before = self.arguments_before(spec, call, args, kwargs)
        self._record_bound(spec, call, run)
        if run.args_before:
            _record_held(run.args_before, (*args, *kwargs.values(), *(value for _, value in bound_values(func))))
        # And the registries the process keeps outside any variable: a hit
        # would not set ``np.seterr`` or a logger's level again.
        run.registries = process_registries.snapshot()
        run.observer.registries = run.registries
        # What a waived line, or a function waived from its ``def`` line,
        # changes of them is left out: only for code that carries a waiver.
        waiver_trace = WaiverTrace() if self._carries_waiver(spec) else None
        with run.tracker, run.observer:
            threads_at_start = THREADS_IN_CALLS[0]
            # What a call that picks code by name ran, for its trace. Begun
            # and ended outside the body's time: setting the recorder up
            # (installing it, on a process's first call) is cash's cost.
            call.trace_unit = (
                code_trace.begin(watch_globals=call.needs_trace, surface=call.narrowed and not call.needs_trace)
                if call.traced
                else None
            )
            if pt is not None:
                _phases.mark(pt, _phases.BODY_SETUP)
            body_t0 = _perf_counter()
            nested = [0.0]
            nested_token = NESTED_CASH_SECONDS.set(nested)
            try:
                self._files.track_declared_files(run.tracker, spec)
                self._files.track_scanned_files(run.tracker, args, kwargs)
                with recording_env_reads() as run.env_reads, warning_state.recording() as run.warned:
                    if waiver_trace is not None:
                        waiver_trace.begin()
                    try:
                        yield run
                    finally:
                        if waiver_trace is not None:
                            waiver_trace.end()
                        run.observer.drop_waivers()
            except Exception as exc:  # the user's body can raise anything; logged, then re-raised
                self._calls.log_raised(func_name, exc, call.call_start)
                raise
            finally:
                body_t1 = _perf_counter()
                if pt is not None:
                    _phases.mark(pt, _phases.BODY)
                code_trace.end(call.trace_unit)
                NESTED_CASH_SECONDS.reset(nested_token)
                # The nested calls' cash time is theirs, and this call's
                # caller's body ran it too: handed up as such, never as this
                # call's own overhead (`miss_overhead` takes it out).
                run.nested_cash_seconds = nested[0]
                outer = NESTED_CASH_SECONDS.get()
                if outer is not None:
                    outer[0] += nested[0]
                # What the body did to the attributes of an instance it was
                # handed: recorded across the body alone.
                for entry in run.args_before.values():
                    if type(entry[1]) is rewrite_in_place.AttrRecording:
                        entry[1].finish()
            # The user's own work, isolated. Everything cash does sits outside
            # this pair, which is the whole point: it is the only number that
            # can answer "did caching pay?".
            # The library settings the code that ran read, as it first read
            # them (`library_reads`).
            unit = call.trace_unit
            run.lib_reads = (
                library_reads.build(
                    code_trace.library_codes(unit),
                    unit.lib_items,
                    unit.lib_cells,
                    unit.lib_keys,
                    unit.lib_first,
                    {**unit.receivers, **unit.lib_receivers},
                    late=unit.lib_late,
                )
                if unit is not None
                else (None, None)
            )
            run.body_seconds = max(0.0, body_t1 - body_t0 - run.tracker.read_hash_seconds - nested[0])
            run.saves_seconds = run.body_seconds / max(threads_at_start, THREADS_IN_CALLS[0], 1)
            run.first_call = self._calls.credit.body_ran(func_name, run.saves_seconds)
            run.rng_drew = self._rng.drawn_streams(run.rng_pre)
            run.rng_new = self._rng.note_draw(func_name, run.rng_drew)
            run.carriers_moved = moved_carriers(run.carriers_pre)
            run.arg_carriers_moved = moved_carriers(run.arg_carriers_pre)

    def _inputs_moved_check(self, spec: CachedFunction, call: Call) -> Callable[[], bool] | None:
        """For `ResultStore.refusal`: did program state the key read change
        while the body ran? None when nothing else can have changed it: one
        thread, and a body that never awaits.

        Only another thread (or, for a coroutine, another task) can rebind a
        global the call read, or change it in place, without the body
        doing so itself. Checking costs a second build of the key's state
        segment, on a miss only, so a single-threaded script does not pay
        it. Cash's own threads (its writers, its flusher, a held-back write's
        timer) run no code of the user's and do not count
        (`other_thread_alive`): counted, they made every miss of a script
        that had stored anything pay the rebuild. What the drift guard
        watches (`CAPTURE_WATCH`) is left to it: the call itself moves a
        random generator it draws from, or a memo a callable keeps.
        """
        if not call.ledger:
            return None
        if not inspect.iscoroutinefunction(spec.func) and not other_thread_alive(None):
            return None
        return lambda: _ledger_moved(call.ledger, self._keys.ledger_now(spec, call.args), call.capture_watch)

    def finish_miss(self, spec: CachedFunction, call: Call, run: BodyRun) -> Any:
        """Everything a missed call does after its body: check, store, log."""
        func, func_name, args, kwargs = spec.func, spec.name, call.args, call.kwargs
        note_getter_fills(call.fill_mark)
        res = run.res
        if call.traced and is_one_shot_iterator(res):
            # Its body runs as the caller iterates, after the trace ended:
            # stored by its key alone, unless it needs a trace.
            call.trace_unit = None
            call.traced = False
            if call.needs_trace:
                return self._untraced(spec, call, "returns an iterator, whose code runs after the call", res)
        trace, untraced = self._trace_of(spec, call, run)
        if untraced and call.trace_for_unread and not any(name in sys.modules for name in call.unread):
            # Traced only for a lazy import the call did not run: none of
            # that module's code ran, so the key holds all the call ran.
            trace, untraced = None, ""
            call.needs_trace = False
        if untraced and not call.needs_trace:
            # A trace is a check on top of the key: a call that does not
            # need one is stored by its key, as without the recorder.
            trace, untraced = None, ""
        if trace is None and not untraced and call.narrowed:
            # Its key left the code of the classes module data holds to a
            # trace there is none of: stored with a check of that code in
            # full instead.
            trace, untraced = self._closure_check(spec)
            call.closure_check = trace
            if untraced and is_one_shot_iterator(res):
                return self._untraced(spec, call, untraced, res)
        # A generator is handed straight back, wrapped, and cached only once
        # the caller has drained it. Draining it here instead meant a streamed
        # response arrived in one lump after the full latency, so
        # `@cash.cache` changed how the function behaved. `ResultStore.stream_and_store`
        # carries the tracker into each production step so lazy file reads are
        # still recorded. An async function returning a SYNC generator streams
        # the same way.
        if is_one_shot_iterator(res):
            # Logged HERE, not at exhaustion. `stats_wrapper` counts this
            # call's entry the moment the wrapper returns, so an entry written
            # when the caller finishes iterating would never be counted. The
            # miss is a fact about the LOOKUP, which has already happened. The
            # produce time still reaches the entry, via the manifest, which is
            # what a later hit reports as saved.
            self._calls.log(
                func_name,
                cache_hit=False,
                execution_time=_perf_counter() - call.call_start,
                args_hash=call.args_hash,
                cache_key=call.cache_key,
            )
            return StreamingCachedIterator(self._store.stream_and_store(res, spec, call, run))

        # A generator argument the body drew from is moved on again by a hit
        # (`replay_argument_carriers`), so that draw is not a mutation the
        # caller would lose: the check sees it where the call found it.
        with (
            carriers_put_back(run.arg_carriers_pre if run.arg_carriers_moved else []),
            call_polars(call.polars_args),
            quiet_digests(),
        ):
            attrs = self.attribute_changes(spec, call, run)
            with content_only(call.by_content):
                self._purity.check_argument_mutation(
                    func_name,
                    *(call.checked or (args, kwargs)),
                    call.call_args_hash,
                    run.observer,
                    explained=attrs.names if attrs is not None and attrs.states is not None else None,
                )
        if call.content_keyed is not None and call.content_keyed.tagged and not call.by_content:
            self._attribute_changes_seen(run.observer, attrs)
        self._purity.check_bound_mutation(func_name, run.observer)
        args_left = self.arguments_left(spec, run, call.checked or (args, kwargs), call.ran_unit, attrs)
        self._files.credit_remembered_reads(func_name, run.tracker, args, kwargs)
        self._files.warn_unrecorded_reads(func_name, run.tracker)
        self._files.warn_unseen_native_reads(func_name, run.tracker)
        auto_file_deps = snapshot_tracked_deps(run.tracker, plain_getattr(func, "__module__", None))
        execution_time = _perf_counter() - call.call_start

        # A result that IS an argument the call changed and a hit writes
        # back (``stft(y, out=buf)``) is handed back as that argument by a
        # hit too, so it shares nothing a hit would hand back apart.
        result_is = self.result_argument(func_name, res, args_left, call.checked or (args, kwargs))
        if result_is:
            shared = None
        elif attrs is not None and attrs.result_checked:
            # Checked against the arguments as the caller handed them in:
            # an attribute the call bound comes back with the result.
            shared = attrs.result_shared
        else:
            shared = self._purity.warn_shared_result(func, func_name, res, args, kwargs)
        # The stream comparison above, read for a draw no seed stands behind:
        # one the source scan could not see (a library a helper calls).
        self._rng.warn_observed_draw(func_name, run.rng_drew, res)
        # Before the check that the state the key read held still: a
        # function whose code was replaced moved that state too.
        swapped = self._code_swapped(spec, call)
        refusal = self._store.refusal(
            func,
            func_name,
            res,
            run.rng_new,
            spec.cache_if,
            run.tracker,
            call.capture_watch,
            observer=run.observer,
            inputs_moved=None if swapped else self._inputs_moved_check(spec, call),
            cost=execution_time,
            watch_identities=call.watch_identities,
            narrowed=call.narrowed,
            file_deps=auto_file_deps,
            call_args=(args, kwargs),
        )
        if refusal is None:
            refusal = swapped
        if refusal is None:
            refusal = self._loaded_unread(spec, call)
        if refusal is None:
            refusal = self._refuse_registries(spec, run)
        if refusal is None and args_left and shared:
            # A hit writes the arguments back and hands the result back as a
            # copy, which no longer shares their memory as the call's did.
            refusal = (
                f"it changed {', '.join(repr(n) for n in dict.fromkeys(_root_of(n, args_left) for n in args_left))} "
                f"in place and its result shares an object "
                f"with what it was handed ({shared}), which a hit would hand back apart"
            )
        if refusal is None:
            refusal = self._refuse_library_reads(spec, run)
        if refusal is None and untraced:
            refusal = self._refuse_untraced(spec, untraced)
        if refusal is None and not replayable(run.carriers_moved + run.arg_carriers_moved):
            # A hit must leave the generator where the body did, or the
            # caller's next draw repeats what this call drew. One only a
            # closure holds cannot be found again by a later process.
            refusal = "it drew from a random generator a cached result cannot advance"
        if refusal is None and args_left and run.observer is not None:
            # Stored with the state it left them in: the warning says what
            # a hit does. Said after the refusals above, which a call
            # sharing its result with an argument meets.
            for name, (before, state) in args_left.items():
                if isinstance(before, rewrite_in_place.HeldBefore):
                    continue  # part of the argument that holds it, named as that
                if isinstance(state, rewrite_in_place.AttrState):
                    run.observer.replayed_args.append(name)
                else:
                    run.observer.record(rewrite_in_place.WRITTEN_BACK, name)
            for name in _held_roots(args_left):
                if name not in run.observer.replayed_args:
                    run.observer.replayed_args.append(name)
        elif args_left and run.observer is not None:
            # Changed and not stored after all: what reading the source
            # found on them stands (`PurityChecks.surface_held_mutations`).
            run.observer.mutated_args = list(dict.fromkeys(_root_of(name, args_left) for name in args_left))
        if run.observer is not None and self._purity.holds_mutations(func_name):
            run.observer.unwritable_args = self._unwritable_args(
                func_name, call.checked or (args, kwargs), call.content_keyed
            )
        self._purity.report_observed_effects(func_name, run.observer)
        pt = _phases.CURRENT.get() if _phases.ON else None
        if pt is not None:
            _phases.mark(pt, _phases.POST)
        self._settle_verdict(spec, call, refusal, run.observer)
        if refusal is not None:
            self._misses.note_not_stored(call.cache_key, refusal, func_name)
        else:
            # Attach lineage only when the value is actually stored: a lineage
            # hash points downstream at THIS cache entry, so a cache_if-rejected
            # (uncached) value must not carry one - it would reference an entry
            # that was never written.
            self._store.attach_lineage(res, call.cache_key, auto_file_deps, ttl=call.ttl, func_name=func_name)
            if not call.traced and not library_reads.UNSEEN_SAID:
                self._warn_settings_unseen(func, func_name)
            self._store.store(
                StoreRequest(
                    call,
                    func_name,
                    execution_time=execution_time,
                    auto_file_deps=auto_file_deps,
                    body_seconds=run.body_seconds,
                    saves_seconds=run.saves_seconds,
                    first_call=run.first_call,
                    rng_replay=self._rng.replay_parts(
                        bool(self._registry.cached[func_name].rng_modules),
                        run.rng_pre,
                        run.carriers_moved,
                        run.arg_carriers_moved,
                    ),
                    dynamic_sources=recorded_sources(run.tracker),
                    # An empty trace stands for nothing unless the call picks
                    # code by name, whose entry is served only on a trace.
                    code_trace=trace if trace or call.needs_trace else None,
                    # What the environment variables an installed package
                    # read held: a lookup checks them again (`env_reads`).
                    env_reads=environment_digests(run.env_reads) if run.env_reads else None,
                    # The warnings state, when the body warned: a lookup
                    # checks it again (`warning_state`).
                    warned_under=warning_state.stamp(run.warned) if run.warned else None,
                    # The library settings it read: a lookup checks them again.
                    lib_reads=run.lib_reads[0] or None,
                    shared_result=shared,
                    args_left=args_left,
                    args_reads=run.args_reads,
                    result_is=result_is,
                ),
                res,
            )
            if call.stored_in is not None:
                self._hand_back_entry(call, res, None, call.stored_in)
        if pt is not None:
            _phases.mark(pt, _phases.STORE)
        # A result the disk cap had evicted, computed again: say what that cost.
        if self._misses.pending_eviction(call.cache_key) is not None:
            self._notices.evicted_recompute(func_name, run.body_seconds)
        # Everything that was not the body: the key and lookup before it, the
        # checks and the store after it. Not the nested cached calls' own
        # cash time, which their entries count (and the body excludes).
        miss_overhead = max(
            call.cash_overhead, _perf_counter() - call.call_start - run.body_seconds - run.nested_cash_seconds
        )
        self._calls.log(
            func_name,
            cache_hit=False,
            execution_time=execution_time,
            args_hash=call.args_hash,
            cache_key=call.cache_key,
            body_seconds=run.body_seconds,
            cash_seconds=miss_overhead,
        )
        self._calls.note_effectiveness(func_name, miss_overhead, body_seconds=run.body_seconds, was_hit=False)
        return res

    def _trace_of(self, spec: CachedFunction, call: Call, run: BodyRun | None = None) -> tuple[list | None, str]:
        """The trace of what the call ran (`code_trace.build`), or why there
        is none: ``(None, "")`` for a call not recorded. An attribute the
        code that ran bound on an instance *run* recorded (an argument, or
        what the callable is bound to) is judged by `attribute_changes`,
        which stores and replays it or says why not, not refused here."""
        if not call.traced:
            return None, ""
        unit, call.trace_unit = call.trace_unit, None
        call.ran_unit = unit
        if unit is None:
            return None, "no sys.monitoring tool id was free to record what it ran"
        # What the key covers: the function, the cached functions it calls
        # (their state is in the key), and the helpers each walk read.
        functions, reports, names = [], [], [spec.name]
        seen: set[str] = set()
        while names:
            name = names.pop()
            fn = spec.func if name == spec.name else self._registry.functions.get(name)
            if name in seen or fn is None:
                continue
            seen.add(name)
            functions.append(fn)
            report = self._registry.report_for(fn, name)
            if report is not None:
                reports.append(report)
            names.extend(self._registry.graph.get_dependencies(name))
        # The code the key folded from the arguments and globals (a method
        # of an argument's class) is in the key as the function's own code
        # is: the strict check judges only code the key never saw.
        accepted = code_trace.folded_codes(call.folded_code) if call.needs_trace else None
        # A narrowed key holds a held class by its surface, and the trace
        # checks each of its methods that ran with its own item: the strict
        # check judges only code no key of the call covers, as for a key
        # built unnarrowed (`KeyBuilder.full_key_folded`).
        if call.needs_trace and call.full_folded is not None:
            accepted = (accepted or set()) | (code_trace.folded_codes(call.full_folded) or set())
        more = None
        if call.needs_trace and call.full_fold_gate is not None:

            def more() -> set | None:
                return code_trace.folded_codes(self._late_full_fold(spec, call))

        owners = frozenset(
            id(entry[1].obj)
            for entry in (getattr(run, "args_before", None) or {}).values()
            if type(entry[1]) is rewrite_in_place.AttrRecording
        )
        # One round of checks (`hit_memo.probe_round`), as the trace's
        # check on a hit is: the classes the trace names share the data
        # their watches probe, read once instead of once a class.
        with hit_memo.probe_round():
            return code_trace.build(
                unit,
                code_trace.static_codes(functions, reports),
                strict=call.needs_trace,
                accepted=accepted,
                owners=owners,
                # A call that needs its trace is checked on all of each class's
                # code, as before: the strict build judges what ran.
                narrow=call.narrowed and not call.needs_trace,
                also_accepted=more,
            )

    @staticmethod
    def _full_fold_gate() -> tuple[int, frozenset[str]] | None:
        """What a fold built after the body is compared against
        (`_late_full_fold`): how many times any function's code or defaults
        were set (`function_sets`), and the modules other than cash's own
        that are loaded. None without the extension's function watcher:
        the fold is built before the body then."""
        sets = function_sets()
        return None if sets is None else (sets, frozenset(sys.modules.copy()))

    def _late_full_fold(self, spec: CachedFunction, call: Call) -> FoldedCode | None:
        """The code the call's key would fold unnarrowed
        (`KeyBuilder.full_key_folded`), built after the body for a strict
        trace that refused what ran against the narrowed key's code; None
        when the body may have moved what it reads (`_world_held`). Built so, it could hold code the body bound
        that no key before it held, and the strict check would skip it:
        the call is refused, and the function's later misses build it
        before the body."""
        gate, call.full_fold_gate = call.full_fold_gate, None
        if gate is None or not self._world_held(spec, call, gate):
            self._full_fold_first.add(spec.name)
            return None
        return self._keys.full_key_folded(spec, call.args, call.kwargs)

    def _world_held(self, spec: CachedFunction, call: Call, gate: tuple[int, frozenset[str]]) -> bool:
        """Is what a key of *call* reads as it was at *gate*, before its
        body (`_full_fold_gate`): no function's code or defaults set, no
        module imported or dropped but cash's own, and the arguments' hash
        the same? A module global the code that ran read and the body
        rebound refuses the call by itself (`code_trace._changed_global`)."""
        sets, modules = gate
        if function_sets() != sets or call.call_args_hash is None:
            return False
        now = sys.modules.copy()
        if len(now) != len(modules) or not modules.issuperset(now):
            moved = modules.symmetric_difference(now)
            if any(name != "cash" and not name.startswith("cash.") for name in moved):
                return False
        try:
            with call_polars(call.polars_args), content_only(call.by_content), quiet_digests():
                after = self._purity.checked_arguments_hash(spec.name, *(call.checked or (call.args, call.kwargs)))
        except Exception:  # noqa: BLE001 - not hashed: taken as changed
            return False
        return after == call.call_args_hash

    def _code_swapped(self, spec: CachedFunction, call: Call) -> str | None:
        """Why a call is not stored when code its key folded was replaced
        while it ran (`cash.code_swaps`), or None.

        A library that compiles a function on its first call (networkx's
        ``argmap``) swaps a stub's ``__code__`` for the code that runs the
        body: the key folded the stub, and an entry stored under it would be
        found by every later process that starts with the same stub, however
        the body changed. Once is expected and said only in the miss reason:
        the next call keys the code that is there now. A function whose code
        is replaced on every call never gets an entry, and the warning says
        why.
        """
        folded = call.folded_code
        owner = folded.swapped() if folded is not None else None
        if owner is None:
            self._code_swaps.pop(spec.name, None)
            return None
        what = plain_getattr(owner, "__qualname__", None) or class_qualname(type(owner))
        times = self._code_swaps[spec.name] = self._code_swaps.get(spec.name, 0) + 1
        if times > 1:
            self._notices.warn_once(
                CashCacheStoreFailedWarning,
                spec.name,
                "code_swapped",
                f"@cash.cache on {spec.name}: the code of {what} is replaced while every call runs "
                f"(its __code__ is set anew), so the key, built before the call, never describes the "
                f"code that ran. The result was returned but not cached.",
                code="STORE-CODE-SWAPPED",
                fix="set the function's code once, before the first cached call, or pass it in as an argument.",
            )
        return f"the code of {what} was replaced while it ran, so its key described the code from before"

    def _loaded_unread(self, spec: CachedFunction, call: Call) -> str | None:
        """Why a call is not stored when a module its tree imports, which
        was not loaded when its key was built and whose top level cash
        cannot read without running it (UNREAD_MODULE), is loaded now: the
        call may have run its code, which the key does not hold. The next
        call's key walks it, loaded (`FunctionRegistry.refresh_helper_bindings`).
        Until then nothing of such a module can run, so a call that leaves
        it unloaded is stored by its key."""
        unread = call.unread
        if not unread or (call.needs_trace and code_trace.on()):
            # A call served on its trace has an import that ran in it, with
            # the code of that module that ran (``first_import``).
            return None
        loaded = sorted(name for name in unread if name in sys.modules)
        if not loaded:
            return None
        self._notices.warn_once(
            CashCacheIneffectiveWarning,
            spec.name,
            "unread_import:" + ",".join(loaded),
            f"{spec.name} imported {', '.join(loaded)}, {UNREAD_MODULE}: the call ran uncached, since what "
            "that top level bound is not in its key. Once loaded, the module is read like any other.",
            code="KEY-UNTRACKABLE-DEP",
            fix="import the module at the top of the file, or compute the value in a function instead of at "
            "the module's top level; `# @cash:assume-safe` on the import line caches anyway.",
        )
        if len(loaded) == 1:
            return (
                f"it imported {loaded[0]}, which was not loaded when its key was built and whose top level "
                "cash cannot read before it is; the next call keys its code"
            )
        return (
            f"it imported {', '.join(loaded)}, which were not loaded when its key was built and whose top "
            "levels cash cannot read before they are; the next call keys their code"
        )

    def _carries_waiver(self, spec: CachedFunction) -> bool:
        """Does ``# @cash:assume-safe`` waive something in the code a call of
        *spec* runs, as far as cash follows it (`code_functions`): its
        misses then trace those functions (`WaiverTrace`)."""
        if self._registry.purity_mode(spec.name) == "silent":
            return False
        for fn in self._registry.code_functions(spec.func, spec.name):
            try:
                code = plain_getattr(unwrapped(fn), "__code__", None)
            except ValueError:  # a cycle of __wrapped__
                continue
            if isinstance(code, types.CodeType) and code_carries_waiver(code):
                return True
        return False

    def _refuse_registries(self, spec: CachedFunction, run: BodyRun) -> str | None:
        """Why a call that set a registry the process keeps is not stored:
        ``np.seterr(...)``, ``logging.getLogger("lib").setLevel(10)``,
        ``atexit.register(f)`` (`process_registries`). A hit would not set
        it again, and a new process would find it unset. Every later call
        of the function too, also one that finds it set already: it still
        sets it. ``assume_safe=True`` waives it, and so does a waiver where
        the change is made: ``# @cash:assume-safe`` on its line or on the
        ``def`` line of a function on the way, or a ``with
        cash.assume_safe():`` block around it (`EffectObserver.open_waiver`)."""
        func_name = spec.name
        if self._registry.purity_mode(func_name) == "silent":
            return None
        found = process_registries.changed(run.registries)
        if found:
            self._registry_writers[func_name] = "; ".join(found)
        what = self._registry_writers.get(func_name)
        if what is None:
            return None
        self._notices.warn_once(
            CashImpurityWarning,
            func_name,
            "process_registry",
            f"@cash.cache on {func_name}: the call changed {what}, which no variable holds; a cache hit would "
            "not change it again (in a new process it would never be set), so calls run uncached.",
            code="IMPURE-OBSERVED-EFFECTS",
            fix="make the setting outside the cached function. If it does not matter that a hit skips it, put "
            "`# @cash:assume-safe` on the line that makes it (or on the `def` line), or `with "
            "cash.assume_safe():` around it; @cash.cache(assume_safe=True) waives the whole function instead.",
        )
        return f"it changed {what}, which no variable holds and a hit would not change again"

    def _refuse_library_reads(self, spec: CachedFunction, run: BodyRun) -> str | None:
        """Why a call that read library state cash cannot compare between
        calls is not stored (`library_reads.build`); said once."""
        why = run.lib_reads[1]
        if why is None:
            return None
        self._notices.warn_once(
            CashCacheIneffectiveWarning,
            spec.name,
            "library_state:" + why,
            f"{spec.name}: {why}, so a later call made under another value could be served this one's "
            "result: calls like this run uncached.",
            code="KEY-LIBRARY-STATE-UNREAD",
            fix="hold plain data (numbers, strings, tuples, dicts of them) in the context variable, or pass "
            "the value in as an argument.",
        )
        return why

    def _warn_settings_unseen(self, func: Callable, func_name: str) -> None:
        """Say once a process that a call stored without its body recorded
        does not have the library settings it read checked: below Python
        3.12, with ``CASH_TRACE=0`` and for a coroutine nothing sees them
        (`library_reads`). Said for the first such call of any kind: a
        library reached through an argument's method (``df.to_string()``)
        is not named in the function's own code."""
        library_reads.UNSEEN_SAID = True
        if inspect.iscoroutinefunction(func):
            where = "a coroutine's body is not recorded"
        elif not code_trace.TRACEABLE:
            where = "Python below 3.12 does not record what a body reads"
        elif not code_trace.on():
            where = "CASH_TRACE=0 turns off recording what a body reads"
        else:
            where = "it returns an iterator, whose code runs after the call and is not recorded"
        self._notices.warn_once(
            CashCacheIneffectiveWarning,
            func_name,
            "library_state_unseen",
            f"{func_name} was stored and {where}, so a library setting changed between calls "
            "(sklearn.set_config, np.seterr, pandas options, a ContextVar, an attribute set on a library's class, ...) "
            "is not detected: a later "
            "call could be served a result made under the old setting. Said once per process.",
            code="KEY-LIBRARY-STATE-UNSEEN",
            fix="run on Python 3.12+ with tracing on (the default), pass the setting in as an argument, "
            "or clear the cache after changing it.",
        )

    def _refuse_untraced(self, spec: CachedFunction, why: str) -> str:
        """Say once why a call that picks code by name is not stored; the
        refusal `MissHistory.note_not_stored` keeps."""
        self._notices.warn_once(
            CashCacheIneffectiveWarning,
            spec.name,
            "untrackable_trace:" + why,
            f"{spec.name} picks code by a run-time name and {why}, so cash cannot check the code a later "
            "call would pick: calls like this run uncached.",
            code="KEY-UNTRACKABLE-DEP",
            fix="call the function by name, or pass it in as an argument; "
            "`# @cash:assume-safe` on that line caches anyway.",
        )
        return f"it picks code by a run-time name and {why}"

    def _untraced(self, spec: CachedFunction, call: Call, why: str, res: Any) -> Any:
        """Hand back *res* of a traced call that cannot be stored."""
        self._misses.note_not_stored(call.cache_key, self._refuse_untraced(spec, why), spec.name)
        self._calls.log(
            spec.name,
            cache_hit=False,
            execution_time=_perf_counter() - call.call_start,
            args_hash=call.args_hash,
            cache_key=call.cache_key,
        )
        return res

    async def single_flight(self, spec: CachedFunction, call: Call, compute: Callable[[], Any]) -> Any:
        """Async single-flight for ``use_locking``: coalesce concurrent awaits
        of the same key in-process, so an expensive idempotent coroutine (a
        paid API call, say) under ``asyncio.gather`` computes once instead of
        N times. The leader computes and stores; followers wait for it and
        then read the stored result, and compute themselves when it stored
        nothing (cache_if rejected it, or it raised)."""
        # Imported HERE, not at module scope: asyncio costs ~76ms of a ~290ms
        # `import cash`, and a synchronous user never needs it. Inside a
        # running loop it is necessarily already imported, so this lookup is
        # free exactly where it is used.
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return await compute()
        with self._async_inflight_lock:
            existing = self._async_inflight.get(call.cache_key)
            if existing is None:
                # Leader: publish the future the followers wait on.
                leader = concurrent.futures.Future()
                self._async_inflight[call.cache_key] = leader
        if existing is not None:
            # Follower, in this loop or another: wait for the leader, then
            # read the stored value. The wait is wrapped per follower, so
            # cancelling one leaves the leader's computation running for the
            # rest.
            try:
                await asyncio.shield(asyncio.wrap_future(existing))
            except Exception:  # noqa: BLE001 - the leader's failure is its own
                pass
            hit = self._reread(spec, call)
            if hit is not CACHE_MISS:
                return hit
            return await compute()
        try:
            return await compute()
        finally:
            # Signal followers (success or failure) and free the slot.
            with self._async_inflight_lock:
                self._async_inflight.pop(call.cache_key, None)
            if not leader.done():
                leader.set_result(None)

    def compute_with_lock(self, spec: CachedFunction, call: Call, compute: Callable[[], Any]) -> Any:
        """Compute with double-checked locking; falls back to unlocked on error.

        Acquiring the lock is best-effort: if *any* backend raises while taking
        it (a Redis ``LockError`` on contention/timeout, a dropped connection,
        an OSError on a file lock), we degrade to an unlocked compute rather than
        crash the user's call. Acquisition, compute, and release are separated so
        a release failure can't re-run the compute, and a compute exception
        propagates normally (it is not mistaken for a lock failure). Under the
        lock the key is looked up again by `CallRunner._reread`, the same test as the first
        lookup."""
        lock_cm = self._backend_slot.backend.lock(call.cache_key)
        try:
            lock_cm.__enter__()
        except Exception as e:  # noqa: BLE001 - any acquisition failure -> unlocked
            self._notices.lock_failed(spec.name, e)
            return compute()
        try:
            hit = self._reread(spec, call)
            if hit is not CACHE_MISS:
                return hit
            return compute()
        finally:
            try:
                lock_cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001 - releasing failed; compute already done
                logger.debug("lock release failed for %s", spec.name)


#: A finding's target written through an attribute of a parameter
#: (``self.null_[...] = ...``, ``self.history.append``): the parameter and
#: the attribute.
_WRITE_INTO_ATTRIBUTE = re.compile(r"(\w+)\.(\w+)[.\[]")


_NOT_HELD = object()


def _outer(before: Any) -> Any:
    """What says the argument an entry of `ArgumentsLeft.states` is part of
    was in the state stored: *before*, or for an instance it held one level
    down (`rewrite_in_place.HeldBefore`), the argument's own."""
    return before.outer if isinstance(before, rewrite_in_place.HeldBefore) else before


def _root_of(name: str, args_left: dict[str, tuple[Any, Any]]) -> str:
    """The argument the entry *name* of *args_left* is part of: itself, or
    the one that held it (`rewrite_in_place.HeldBefore`)."""
    before = args_left[name][0]
    return before.param if isinstance(before, rewrite_in_place.HeldBefore) else name


def _held_roots(args_left: dict[str, tuple[Any, Any]]) -> list[str]:
    """The arguments in *args_left* only through what they held."""
    return list(
        dict.fromkeys(
            before.param for before, _ in args_left.values() if isinstance(before, rewrite_in_place.HeldBefore)
        )
    )


def _record_held(before: dict[str, tuple], arguments: tuple) -> None:
    """Record, in *before* (`CallRunner.arguments_before`), the attributes
    the body binds on each instance of a plain class that a recorded
    instance holds as a top-level attribute (``est.kernel_transformer``),
    one level down and no further: a hit replays them on the instance the
    argument it is handed holds there (`rewrite_in_place.HeldBefore`). One
    held at more than one place, or that is an argument itself, is not:
    which of them the call reached it through is not known, and a change to
    it keeps the refusal."""
    found: dict[int, list[tuple[str, str, Any, Any]]] = {}
    skip = {id(value) for value in arguments} | {id(entry[0]) for entry in before.values()}
    for param, entry in list(before.items()):
        recording = entry[1]
        if type(recording) is not rewrite_in_place.AttrRecording:
            continue
        for attribute, value in list(recording.state.items()):
            if type(attribute) is not str or id(value) in skip or not _replays_attributes(value):
                continue
            found.setdefault(id(value), []).append((param, attribute, value, entry[2]))
    for places in found.values():
        if len(places) != 1:
            continue
        param, attribute, value, outer = places[0]
        try:
            recording = rewrite_in_place.AttrRecording(value)
        except Exception:  # noqa: BLE001 - not recorded: a change refuses the store
            continue
        before[f"{param}.{attribute}"] = (value, recording, rewrite_in_place.HeldBefore(param, attribute, outer))


def _replays_attributes(value: Any) -> bool:
    """Whether a change to *value*'s attributes can be replayed by a hit
    (`rewrite_in_place.attribute_refusal`): an instance of a plain class."""
    return bool(class_dictoffset(type(value))) and rewrite_in_place.attribute_refusal(value) is None


def _takes_state(value: Any) -> bool:
    """Whether a hit could put the state a call leaves *value* in back:
    write it into the object, or replay its attributes."""
    return rewrite_in_place.rewritable_kind(value) or _replays_attributes(value)
