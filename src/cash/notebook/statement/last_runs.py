"""What each statement's last run in this session left to serve it from.

The reasons a miss names otherwise read records that are not there for
most statements: the input record by output name (a statement that binds
nothing, or only names it also reads, has none), a version stored on disk
(a statement too quick to store, or refused, has none), the note of an
unstored slow statement. A statement run again with none of them read
"computed, no reason given" on its badge.

So each run of a statement notes, in memory and by the statement's text,
the key it ran under, the lineages its inputs had and why it left no entry
(`LastRuns.note`); a miss that nothing else explains compares against it
(`LastRuns.reason`). Kept per process: after a restart it says nothing, as
for a statement never run.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ..lineage_formula import resolve_lineage_alias

__all__ = ["KEY_MOVED", "KeyParts", "LastRuns"]

#: How many statements the record keeps; the oldest go first.
_KEPT = 4096

#: The key changed while every input it reads, the functions and modules
#: it names and its place in the cell stayed as they were.
KEY_MOVED = (
    "something else its key reads changed: code or globals the functions it calls reach, "
    "an environment variable it reads, or code installed on a library class"
)


@dataclass(frozen=True)
class _LastRun:
    cache_key: str
    #: The lineage of each input it read, as it ran.
    entry: Mapping[str, str]
    #: Why it left nothing a later run under its key could be served from;
    #: None when it stored an entry, or was served one.
    unkept: str | None
    #: It was kept in the memory tier only.
    memory_only: bool
    #: The ``name:digest`` of each function and module its key named, and
    #: its occurrence in the cell (`KeyParts`).
    parts: KeyParts | None


@dataclass(frozen=True)
class KeyParts:
    """What a statement's key named besides its text and inputs' lineages
    (``CacheKeyResult``)."""

    functions: frozenset[str]
    modules: frozenset[str]
    occurrence: int

    def moved(self, then: KeyParts) -> str | None:
        """What of these differs from *then*, worded for a miss reason."""

        def names(now: frozenset[str], was: frozenset[str]) -> list[str]:
            return sorted({part.split(":", 1)[0] for part in now ^ was})

        said = []
        functions = names(self.functions, then.functions)
        if functions:
            said.append(f"code it calls changed: {', '.join(functions[:3])}")
        modules = names(self.modules, then.modules)
        if modules:
            said.append(f"module changed: {', '.join(modules[:3])}")
        if self.occurrence != then.occurrence:
            said.append("its place among the same statements in the cell changed")
        return "; ".join(said) or None


class LastRuns:
    """``statement text -> its last run in this session``."""

    def __init__(self) -> None:
        self._runs: dict[str, _LastRun] = {}

    def clear(self) -> None:
        self._runs.clear()

    def note(
        self,
        code: str,
        cache_key: str | None,
        entry: Mapping[str, str] | None,
        unkept: str | None = None,
        memory_only: bool = False,
        parts: KeyParts | None = None,
    ) -> None:
        """Note that *code* ran (or was served) under *cache_key* with its
        inputs at *entry*; *unkept* says why it left no entry."""
        if not cache_key:
            return
        self._runs.pop(code, None)
        self._runs[code] = _LastRun(cache_key, dict(entry or {}), unkept, memory_only, parts)
        while len(self._runs) > _KEPT:
            del self._runs[next(iter(self._runs))]

    def reason(
        self,
        code: str,
        cache_key: str | None,
        entry: Mapping[str, str] | None,
        has_entry: Callable[[str], bool | None],
        *,
        parts: KeyParts | None = None,
    ) -> str | None:
        """Why *code*, missing under *cache_key* with its inputs at *entry*,
        could not be served, against its last run in this session; None when
        it has not run in this session.

        *has_entry* says whether the cache holds an entry under a key (None:
        it could not tell)."""
        last = self._runs.get(code)
        if last is None or not cache_key:
            return None
        if last.cache_key == cache_key:
            if last.unkept:
                return f"not stored when it last ran under this key: {last.unkept}"
            held = has_entry(cache_key)
            if held is False:
                where = " (kept in memory only)" if last.memory_only else ""
                return f"the entry stored under this key when it last ran is no longer in the cache{where}"
            if held:
                return "an entry under this key is in the cache, but it was not used, and cash recorded no reason"
            return None
        now = entry or {}
        moved = sorted(
            name
            for name, was in last.entry.items()
            if isinstance(was, str)
            and isinstance(now.get(name), str)
            and resolve_lineage_alias(now[name]) != resolve_lineage_alias(was)
        )
        # A name with a lineage at one end only: bound where cash did not
        # see it (``user_ns["df"] = ...``), or dropped since.
        moved += [f"{name} (no recorded lineage then)" for name in sorted(set(now) - set(last.entry))]
        moved += [f"{name} (no recorded lineage now)" for name in sorted(set(last.entry) - set(now))]
        if moved:
            more = f" +{len(moved) - 3} more" if len(moved) > 3 else ""
            return f"input changed since it last ran: {', '.join(moved[:3])}{more}"
        if parts is not None and last.parts is not None:
            named = parts.moved(last.parts)
            if named:
                return f"{named} since it last ran"
        return f"its key changed since it last ran, with the same inputs: {KEY_MOVED}"
