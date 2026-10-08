"""The ``dynamic_depends_on=`` sources of a cached callee, as an entry of its
cached caller records and checks them.

A callee folds the sources its resolver returns into its own key, which the
caller's key never sees: ``report()`` calling ``load("prices")`` kept serving
its old result after the source changed. A file source becomes a file the
caller's entry checks (`pass_dynamic_sources_up`). Any other source is
recorded in the caller's entry with the token it had when the callee keyed
on it, and every lookup of the caller asks the source for its token again,
as a static ``depends_on=`` source is asked when the caller's key is built
(`DependencyStateComputer.compute`): the same token serves the entry, any
other -- or a ``state_token()`` that raises -- recomputes it.

The source is pickled with the entry, so a later process can ask it. In the
process that recorded it, the object itself is asked, so a source whose
token moves with its own state is seen to move. The resolver is asked again
too, in any process: a resolver may hand out a new source object (a catalog
refresh that builds new handles), which the recorded object never answers
for (`Resolution`). Its call is pickled with the entry when small, and the
process that made it keeps a bounded number of them (`_HeldCalls`).
A source that cannot be pickled is kept by this process only: its entry
stays in RAM, and a backend with no RAM tier does not store it
(`ResultStore.refusal`).

A small pickle goes in the entry itself. A larger one -- a source that holds
data, such as an in-memory table handle -- is stored once, as an entry of
its own under `source_key`, and every caller's entry names it by digest: a
copy in each caller multiplied it, and a hit in a new process read and
unpickled it again for each caller.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import logging
import pickle
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

from .._memo import DYNAMIC_RESOLUTION_BYTES, DYNAMIC_RESOLUTIONS, DYNAMIC_SOURCE_ENTRIES, LruMemo
from ..data_source import DataSource, state_token_of

logger = logging.getLogger(__name__)

__all__ = [
    "DynamicSources",
    "Resolution",
    "dynamic_sources_fresh",
    "entry_resolutions",
    "held_sources",
    "recorded_sources",
    "remember_sources",
    "source_key",
]

#: A pickled source up to this many bytes is kept in the caller's entry;
#: a larger one is stored once under `source_key`.
INLINE_PICKLE_BYTES = 4096


def source_key(digest: str) -> str:
    """The backend key a pickled source of *digest* is stored under."""
    return f"cash-dynamic-source:{digest}"


class Resolution:
    """One cached call's ``dynamic_depends_on=`` resolvers, to be asked
    again: run with that call's arguments, do they resolve to the same
    sources with the same tokens (*expected*, ``[(id, token), ...]``)?

    The recorded source objects are asked too, but a resolver that hands
    out a NEW object on a refresh leaves the old one answering its old
    token. So the call itself is kept: pickled with the caller's entry
    (`recorded_sources`), so that a later process -- or the parent of a
    pool worker that ran it -- asks it again, as the process that stored
    the entry does."""

    __slots__ = ("_args", "_expected", "_func_name", "_kwargs", "_owner", "_resolvers")

    def __init__(
        self,
        func_name: str,
        resolvers: Any,
        args: tuple,
        kwargs: dict,
        expected: list[tuple[str, str]],
        owner: tuple[str, str] | None = None,
    ) -> None:
        self._func_name = func_name
        self._resolvers = resolvers
        self._args = args
        self._kwargs = kwargs
        self._expected = [tuple(pair) for pair in expected]
        #: ``(module, qualname)`` of the cached function the resolvers are
        #: declared on: they pickle by reference to it when they do not
        #: pickle themselves (``dynamic_depends_on=lambda name: ...``).
        self._owner = owner

    def __reduce__(self) -> tuple:
        state = (self._func_name, self._args, self._kwargs, self._expected, self._owner)
        if self._owner is not None:
            try:
                pickle.dumps(self._resolvers, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:  # noqa: BLE001 - a lambda, a local function
                return (_resolution_by_reference, state)
        return (Resolution, (self._func_name, self._resolvers, self._args, self._kwargs, self._expected, self._owner))

    def argument_bytes(self) -> int:
        """About how much the call's arguments hold: their pickled size."""
        try:
            return len(pickle.dumps((self._args, self._kwargs), protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:  # noqa: BLE001 - an argument that does not pickle
            return _UNSIZED_BYTES

    def resolve_now(self) -> list[tuple[str, str]]:
        """``[(id, token), ...]`` of what the resolvers resolve to now."""
        from .registry import resolve_dynamic_dependencies

        now: list[tuple[Any, str]] = []
        resolve_dynamic_dependencies(self._func_name, self._resolvers, self._args, self._kwargs, now)
        return [(source.get_id(), token) for source, token in now]

    def fresh(self) -> bool:
        """True when the resolvers resolve to the same sources with the same
        tokens; a resolver or token that raises is not shown unchanged."""
        try:
            return self.resolve_now() == self._expected
        except Exception:  # noqa: BLE001 - the user's resolver or state_token
            logger.debug("[CORE] a dynamic_depends_on resolver raised when asked again", exc_info=True)
            return False


def _resolution_by_reference(
    func_name: str, args: tuple, kwargs: dict, expected: list, owner: tuple[str, str]
) -> Resolution:
    """A `Resolution` read back with the resolvers of the cached function
    *owner* names, as this process defines it. Raises when it is not there:
    the call cannot be asked, and its caller recomputes."""
    module_name, qualname = owner
    target: Any = sys.modules[module_name]
    for part in qualname.split("."):
        target = getattr(target, part)
    resolvers = getattr(target, "_cash_dynamic_depends_on", None)
    if not resolvers:
        raise LookupError(f"{module_name}.{qualname} declares no dynamic_depends_on")
    return Resolution(func_name, resolvers, args, kwargs, expected, owner)


@dataclass(frozen=True)
class DynamicSources:
    """The non-file sources a call's cached callees resolved, ready to record.

    ``records`` is what the entry's metadata keeps: per source its id, its
    token at call time and, when it pickles, the pickle (``pickle``) or,
    for a large one, its digest (``blob``). ``live`` are the
    objects themselves, in the same order. ``picklable`` is False when one
    of them did not pickle: the entry can then only be checked by this
    process.
    """

    records: list[dict[str, str]]
    live: list[DataSource]
    picklable: bool
    resolutions: dict[tuple, Resolution]
    #: digest -> pickle of each source too large to keep in the entry, to
    #: be stored under `source_key` before the entry.
    blobs: dict[str, bytes]
    #: What the entry's metadata keeps of each of *resolutions*, in order:
    #: its pickle (``pickle``), or ``here_only`` for a call only this
    #: process can ask (`_resolver_record`).
    resolver_records: list[dict[str, str]] = dataclasses.field(default_factory=list)

    @property
    def unpicklable_ids(self) -> list[str]:
        return [r["id"] for r in self.records if "pickle" not in r and "blob" not in r]


#: cache key -> the source objects its entry's records name, in their order:
#: the ones this process recorded, or unpickled from an entry it read. One
#: dropped is unpickled again from the entry; the resolver calls behind it
#: are asked again either way (`dynamic_sources_fresh`), so a source that
#: was refreshed since is still seen. Bounded: a service calling a loader
#: with ever new arguments kept every caller entry's sources for good.
_LIVE: LruMemo[str, tuple[list[str], list[DataSource]]] = LruMemo(DYNAMIC_SOURCE_ENTRIES)
#: The pickle of a resolver call kept in an entry -> the call, unpickled once.
_FROM_PICKLE: LruMemo[str, Resolution] = LruMemo(DYNAMIC_RESOLUTIONS)
_LIVE_LOCK = threading.Lock()


def recorded_sources(tracker: Any) -> DynamicSources | None:
    """What *tracker* collected from the cached callees its block ran
    (`FileAccessTracker.add_dynamic_source`), pickled for the entry, or
    None when there is nothing to record."""
    collected = getattr(tracker, "dynamic_sources", None) or {}
    resolutions = getattr(tracker, "dynamic_resolutions", None) or {}
    if not collected and not resolutions:
        return None
    # Asked twice per call, by the refusal and by the store: pickled once.
    done = getattr(tracker, "_recorded_sources", None)
    if done is not None and done[0] == (len(collected), len(resolutions)):
        return done[1]
    records: list[dict[str, str]] = []
    live: list[DataSource] = []
    picklable = True
    seen: set[tuple[str, str, str]] = set()
    blobs: dict[str, bytes] = {}
    for key in sorted(collected, key=lambda k: k[:2]):
        source_id = key[0]
        source, token = collected[key]
        record = {"id": source_id, "token": token}
        try:
            data = pickle.dumps(source, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:  # noqa: BLE001 - pickle raises whatever __reduce__ raises
            logger.debug("[CORE] dynamic source %s does not pickle", source_id, exc_info=True)
            picklable = False
            data = None
        if data is not None:
            if len(data) > INLINE_PICKLE_BYTES:
                digest = hashlib.sha256(data).hexdigest()
                record["blob"] = digest
                blobs[digest] = data
                held = record["blob"]
            else:
                record["pickle"] = held = base64.b64encode(data).decode("ascii")
            # The same source met again, as a new object each call: once.
            if (source_id, token, held) in seen:
                continue
            seen.add((source_id, token, held))
        records.append(record)
        live.append(source)
    resolver_records = [_resolver_record(resolution) for resolution in resolutions.values()]
    result = DynamicSources(records, live, picklable, dict(resolutions), blobs, resolver_records)
    try:
        tracker._recorded_sources = ((len(collected), len(resolutions)), result)
    except AttributeError:  # a tracker that takes no attributes: pickle again
        pass
    return result


def _resolver_record(resolution: Resolution) -> dict[str, str]:
    """What a caller's entry keeps of one resolver call: its pickle when it
    is small. A large one -- an array or a frame passed to the loader -- is
    not copied into the store: only this process asks it, while it keeps it
    (`_HeldCalls`), as one that does not pickle at all."""
    try:
        data = pickle.dumps(resolution, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:  # noqa: BLE001 - a lambda resolver, an argument that does not pickle
        logger.debug("[CORE] a dynamic_depends_on resolver call does not pickle", exc_info=True)
        return {"here_only": "1", "bytes": str(resolution.argument_bytes())}
    if len(data) > INLINE_PICKLE_BYTES:
        return {"here_only": "1", "bytes": str(len(data))}
    return {"pickle": base64.b64encode(data).decode("ascii")}


#: What a resolver call whose arguments do not pickle counts as against
#: `DYNAMIC_RESOLUTION_BYTES`: their size is not known.
_UNSIZED_BYTES = 1 << 20


class _HeldCalls:
    """cache key -> the resolver calls this process keeps for that entry, by
    their place in its ``dynamic_resolvers``; the least recently used
    dropped past `DYNAMIC_RESOLUTIONS` entries or `DYNAMIC_RESOLUTION_BYTES`
    of arguments. Each holds its call's arguments: kept for good, a service
    passing arrays through a loader grew by each array. A small call
    dropped is read back from the entry; a large one can then not be asked,
    and its caller recomputes."""

    def __init__(self) -> None:
        self._data: OrderedDict[str, tuple[dict[int, Resolution], int]] = OrderedDict()
        self._bytes = 0

    def get(self, key: str) -> dict[int, Resolution] | None:
        held = self._data.get(key)
        if held is None:
            return None
        self._data.move_to_end(key)
        return held[0]

    def put(self, key: str, calls: dict[int, Resolution], nbytes: int) -> None:
        self.pop(key)
        if not calls:
            return
        self._data[key] = (calls, nbytes)
        self._bytes += nbytes
        while self._data and (len(self._data) > DYNAMIC_RESOLUTIONS or self._bytes > DYNAMIC_RESOLUTION_BYTES):
            _, (_, dropped) = self._data.popitem(last=False)
            self._bytes -= dropped

    def pop(self, key: str) -> None:
        held = self._data.pop(key, None)
        if held is not None:
            self._bytes -= held[1]


_RESOLUTIONS = _HeldCalls()


def remember_sources(cache_key: str, sources: DynamicSources) -> None:
    """Keep the objects behind the entry just stored under *cache_key*, and
    the resolver calls behind it (`_HeldCalls`)."""
    calls = dict(enumerate(sources.resolutions.values()))
    nbytes = sum(int(record.get("bytes", 0)) or len(record.get("pickle", "")) for record in sources.resolver_records)
    with _LIVE_LOCK:
        _LIVE[cache_key] = ([r["id"] for r in sources.records], list(sources.live))
        _RESOLUTIONS.put(cache_key, calls, nbytes)


def entry_resolutions(
    cache_key: str, resolver_records: list[dict[str, str]] | None, fetch: Callable[[str], Any] | None = None
) -> dict[tuple, Resolution] | None:
    """The resolver calls behind the entry under *cache_key*, by key: kept
    here, or unpickled from its *resolver_records*. None when one cannot be
    had (it did not pickle and is not kept here, or its pickle is gone)."""
    if not resolver_records:
        return {}
    with _LIVE_LOCK:
        kept = _RESOLUTIONS.get(cache_key) or {}
    found: dict[tuple, Resolution] = {}
    for i, record in enumerate(resolver_records):
        resolution = kept.get(i)
        if resolution is None:
            resolution = _unpickled_resolution(record, fetch)
        if resolution is None:
            return None
        found[("entry", cache_key, i)] = resolution
    return found


def _unpickled_resolution(record: dict[str, str], fetch: Callable[[str], Any] | None) -> Resolution | None:
    """The resolver call *record* keeps, unpickled; None when it cannot be had."""
    blob = record.get("pickle")
    if blob is None:
        return None  # kept by the process that stored the entry only
    with _LIVE_LOCK:
        known = _FROM_PICKLE.get(blob)
    if known is not None:
        return known
    data = base64.b64decode(blob)
    try:
        resolution = pickle.loads(data)
    except Exception:  # noqa: BLE001 - a resolver gone or renamed since
        logger.debug("[CORE] a dynamic_depends_on resolver call does not unpickle", exc_info=True)
        return None
    if not isinstance(resolution, Resolution):
        return None
    with _LIVE_LOCK:
        _FROM_PICKLE[blob] = resolution
    return resolution


#: digest -> the source unpickled from the entry stored under
#: `source_key`: read and unpickled once per process, whatever the number of
#: callers naming it.
_FROM_BLOB: dict[str, DataSource] = {}


def held_sources(
    cache_key: str, records: list[dict[str, str]], fetch: Callable[[str], Any] | None = None
) -> list[DataSource] | None:
    """The source objects *records* name: the ones this process holds for
    *cache_key*, else unpickled from the records -- or, for one stored on
    its own, from what *fetch* returns for its `source_key`. None when one
    cannot be had: not pickled and not held here, its own entry gone, or the
    pickle does not load."""
    ids = [r.get("id") for r in records]
    with _LIVE_LOCK:
        held = _LIVE.get(cache_key)
    if held is not None and held[0] == ids:
        return held[1]
    loaded: list[DataSource] = []
    for record in records:
        source = _unpickled(record, fetch)
        if source is None:
            return None
        loaded.append(source)
    with _LIVE_LOCK:
        _LIVE[cache_key] = (ids, loaded)
    return loaded


def _unpickled(record: dict[str, str], fetch: Callable[[str], Any] | None) -> DataSource | None:
    """The source *record* keeps, unpickled; None when it cannot be had."""
    digest = record.get("blob")
    if digest is not None:
        with _LIVE_LOCK:
            known = _FROM_BLOB.get(digest)
        if known is not None:
            return known
        try:
            data = fetch(source_key(digest)) if fetch is not None else None
        except Exception:  # noqa: BLE001 - a backend read that failed: not had
            logger.debug("[CORE] dynamic source %s could not be read", record.get("id"), exc_info=True)
            data = None
        if not isinstance(data, bytes) or hashlib.sha256(data).hexdigest() != digest:
            return None
    else:
        blob = record.get("pickle")
        if blob is None:
            return None
        data = base64.b64decode(blob)
    try:
        source = pickle.loads(data)
    except Exception:  # noqa: BLE001 - a class gone or renamed since: unusable
        logger.debug("[CORE] dynamic source %s does not unpickle", record.get("id"), exc_info=True)
        return None
    if not isinstance(source, DataSource):
        return None
    if digest is not None:
        with _LIVE_LOCK:
            _FROM_BLOB[digest] = source
    return source


def dynamic_sources_fresh(
    cache_key: str,
    records: list[dict[str, str]],
    fetch: Callable[[str], Any] | None = None,
    resolver_records: list[dict[str, str]] | None = None,
) -> bool:
    """True when every source *records* names gives the token it gave when
    the entry was written. A source that cannot be had, or whose
    ``state_token()`` raises, is not shown unchanged: False. The resolver
    calls behind the entry (*resolver_records*) are asked again first
    (`Resolution`), in any process: one that cannot be had is not shown
    unchanged either."""
    resolutions = entry_resolutions(cache_key, resolver_records, fetch)
    if resolutions is None or not all(r.fresh() for r in resolutions.values()):
        return False
    if not records:
        return True
    sources = held_sources(cache_key, records, fetch)
    if sources is None:
        return False
    for source, record in zip(sources, records):
        try:
            token = state_token_of(source)
        except Exception:  # noqa: BLE001 - the user's state_token
            logger.debug("[CORE] state_token() of %s raised", record.get("id"), exc_info=True)
            return False
        if token != record.get("token"):
            return False
    return True
