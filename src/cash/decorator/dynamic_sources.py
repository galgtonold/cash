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
token moves with its own state is seen to move. A source that cannot be
pickled is kept by this process only: its entry stays in RAM, and a backend
with no RAM tier does not store it (`ResultStore.refusal`).
"""

from __future__ import annotations

import base64
import logging
import pickle
import threading
from dataclasses import dataclass
from typing import Any

from ..data_source import DataSource, state_token_of

logger = logging.getLogger(__name__)

__all__ = ["DynamicSources", "dynamic_sources_fresh", "held_sources", "recorded_sources", "remember_sources"]


@dataclass(frozen=True)
class DynamicSources:
    """The non-file sources a call's cached callees resolved, ready to record.

    ``records`` is what the entry's metadata keeps: per source its id, its
    token at call time and, when it pickles, the pickle. ``live`` are the
    objects themselves, in the same order. ``picklable`` is False when one
    of them did not pickle: the entry can then only be checked by this
    process.
    """

    records: list[dict[str, str]]
    live: list[DataSource]
    picklable: bool

    @property
    def unpicklable_ids(self) -> list[str]:
        return [r["id"] for r in self.records if "pickle" not in r]


#: cache key -> the source objects its entry's records name, in their order:
#: the ones this process recorded, or unpickled from an entry it read. Never
#: evicted: a dropped object would be unpickled again from the entry, and a
#: source whose token moves with its own state would then answer as it stood
#: when the entry was written.
_LIVE: dict[str, tuple[list[str], list[DataSource]]] = {}
_LIVE_LOCK = threading.Lock()


def recorded_sources(tracker: Any) -> DynamicSources | None:
    """What *tracker* collected from the cached callees its block ran
    (`FileAccessTracker.add_dynamic_source`), pickled for the entry, or
    None when there is nothing to record."""
    collected = getattr(tracker, "dynamic_sources", None)
    if not collected:
        return None
    # Asked twice per call, by the refusal and by the store: pickled once.
    done = getattr(tracker, "_recorded_sources", None)
    if done is not None and done[0] == len(collected):
        return done[1]
    records: list[dict[str, str]] = []
    live: list[DataSource] = []
    picklable = True
    for source_id in sorted(collected):
        source, token = collected[source_id]
        record = {"id": source_id, "token": token}
        try:
            record["pickle"] = base64.b64encode(pickle.dumps(source, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")
        except Exception:  # noqa: BLE001 - pickle raises whatever __reduce__ raises
            logger.debug("[CORE] dynamic source %s does not pickle", source_id, exc_info=True)
            picklable = False
        records.append(record)
        live.append(source)
    result = DynamicSources(records, live, picklable)
    try:
        tracker._recorded_sources = (len(collected), result)
    except AttributeError:  # a tracker that takes no attributes: pickle again
        pass
    return result


def remember_sources(cache_key: str, sources: DynamicSources) -> None:
    """Keep the objects behind the entry just stored under *cache_key*."""
    with _LIVE_LOCK:
        _LIVE[cache_key] = ([r["id"] for r in sources.records], list(sources.live))


def held_sources(cache_key: str, records: list[dict[str, str]]) -> list[DataSource] | None:
    """The source objects *records* name: the ones this process holds for
    *cache_key*, else unpickled from the records. None when one cannot be
    had -- not pickled and not held here, or the pickle does not load."""
    ids = [r.get("id") for r in records]
    with _LIVE_LOCK:
        held = _LIVE.get(cache_key)
    if held is not None and held[0] == ids:
        return held[1]
    loaded: list[DataSource] = []
    for record in records:
        blob = record.get("pickle")
        if blob is None:
            return None
        try:
            source = pickle.loads(base64.b64decode(blob))
        except Exception:  # noqa: BLE001 - a class gone or renamed since: unusable
            logger.debug("[CORE] dynamic source %s does not unpickle", record.get("id"), exc_info=True)
            return None
        if not isinstance(source, DataSource):
            return None
        loaded.append(source)
    with _LIVE_LOCK:
        _LIVE[cache_key] = (ids, loaded)
    return loaded


def dynamic_sources_fresh(cache_key: str, records: list[dict[str, str]]) -> bool:
    """True when every source *records* names gives the token it gave when
    the entry was written. A source that cannot be had, or whose
    ``state_token()`` raises, is not shown unchanged: False."""
    sources = held_sources(cache_key, records)
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
