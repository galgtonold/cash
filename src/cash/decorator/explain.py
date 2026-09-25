"""Why a call hit or missed: the miss reasons, `CacheExplanation`, and the
descriptions of what changed."""

from __future__ import annotations

import hashlib
import inspect
import os
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, NamedTuple

from .._active import EXPLAINING as _EXPLAINING
from .._memo import STATE_LEDGERS, LruMemo
from ..backends._base import ttl_expired
from ..data_source import DataSource
from ..tracking.file_dep_snapshot import dep_is_fresh, dep_path_for_this_process
from .arg_hashing import unhashable_arg_fix
from .cache_metadata import CacheMetadata
from .cached_function import CachedFunction
from .call_state import PROCESS_STARTED, KeyBuildFailed, UnhashableArgs, UnhashableDefault

if TYPE_CHECKING:
    from ..config import CashConfig
    from .arg_hashing import ArgHasher
    from .backend_slot import BackendSlot
    from .frozen import FrozenResults
    from .registry import FunctionRegistry
    from .runtime import KeyBuilder
    from .stored_keys import StoredKeyRecord

# Reason codes returned by `Explainer.explain` / ``f.explain(...)``.
# Kept as module-level constants so external code can match against them
# without string-typo risk: ``if e.reason == EXPLAIN_HIT: ...``.
EXPLAIN_HIT = "hit"
EXPLAIN_KEY_UNCOMPUTABLE = "key_uncomputable"
EXPLAIN_NO_ENTRY = "no_entry"
EXPLAIN_TTL_EXPIRED = "ttl_expired"
EXPLAIN_FILE_CHANGED = "file_changed"
EXPLAIN_DISABLED = "disabled"


@dataclass(frozen=True)
class CacheExplanation:
    """Why a specific call would hit or miss the cache *right now*.

    Returned by ``f.explain(*args, **kwargs)`` on any ``@cash.cache``-wrapped
    function. Inspecting an explanation does NOT mutate stats, call the
    underlying function, or write to the backend - it only reads what the
    cache already knows.

    Attributes:
        would_hit: True if the next call with these args would return a
            cached value (without recomputing).
        reason: Short stable string identifying the outcome. One of:
            ``"hit"``, ``"key_uncomputable"``, ``"no_entry"``,
            ``"ttl_expired"``, ``"file_changed"``, ``"disabled"``.
        func_name: Module-qualified name of the cached function.
        cache_key: The cache key computed for these args, or ``None``
            when key generation failed (``reason == "key_uncomputable"``).
        details: Reason-specific extras. Common keys:

            * ``hit``: ``cached_at`` (unix ts), ``execution_time_saved`` (s),
              ``cache_age_seconds``.
            * ``key_uncomputable``: ``arg_type`` (qualname or ``"<unknown>"``),
              ``error`` (exception type+message), ``hint``.
            * ``no_entry``: ``hint``, and ``why`` -- what this process
              knows about the key: never stored and why, stored and since
              evicted, or which part of the key moved since the last call
              (``new arguments``, ``code or state changed``, ...).
            * ``ttl_expired``: ``ttl_seconds``, ``age_seconds``, ``cached_at``
              when the decorator's ttl ran out; ``why`` when the entry
              expired under the ttl it was written with.
            * ``file_changed``: ``changed_files`` (dict of path -> reason),
              ``file_deps``.
            * ``file_deps`` (on ``hit`` and ``file_changed``): every file the
              entry recorded, with the fingerprint it was checked against.

    ``entry_id`` is the id ``cash inspect --function`` lists and
    ``cash clear --entry`` accepts. ``cache_dir`` is the directory the answer
    was read from (None for a cache with no directory): a nested
    ``pyproject.toml`` can point one project's functions at another cache,
    and an answer that does not say which one it read cannot show that.
    """

    would_hit: bool
    reason: str
    func_name: str
    cache_key: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    cache_dir: str | None = None

    @property
    def entry_id(self) -> str | None:
        """The id `cash inspect` lists and `cash clear --entry` takes."""
        return entry_id_of(self.cache_key) if self.cache_key else None

    def __str__(self) -> str:
        verdict = "HIT" if self.would_hit else "MISS"
        lines = [f"[{verdict}] {self.func_name} - {self.reason}"]
        if self.cache_dir:
            lines.append(f"  cache_dir: {self.cache_dir}")
        if self.cache_key:
            lines.append(f"  cache_key: {self.cache_key}")
            lines.append(f"  entry_id: {self.entry_id}")
        for k, v in self.details.items():
            if isinstance(v, dict):
                lines.append(f"  {k}:")
                for kk, vv in v.items():
                    lines.append(f"    {kk}: {vv}")
            else:
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()


class MissKind(str, Enum):
    """Why a call missed: what the summary and ``cache_info()`` count.

    A ``str`` so that it counts, compares and prints as the words below.
    """

    FIRST = "no entry yet"
    ARGS = "new arguments"
    CODE = "code or state changed"
    DYNAMIC = "dynamic dependency changed"
    FILE = "file changed"
    TTL = "ttl expired"
    NOT_STORED = "not stored last time"
    GONE = "entry gone"
    INCOMPLETE = "entry incomplete"
    UNHASHABLE = "unhashable argument"
    KEY_FAILED = "key could not be built"
    MOCKED = "a helper is a mock"
    RAISED = "raised"

    __str__ = str.__str__
    __format__ = str.__format__
    __hash__ = str.__hash__


_CODE_CHANGED = "the function's code, a helper it calls, or a value it reads changed"

#: A GONE miss the size cap's notes account for (`eviction_log`).
_EVICTED = "evicted to make room when the disk cache reached its size cap"


class MissReason(NamedTuple):
    """One miss: its kind, the detail the per-call line and explain() give,
    and for a code change, what changed (which the summary tallies)."""

    kind: MissKind
    detail: str = ""
    changed: str | None = None
    #: For an entry a size cap evicted, its `EvictionNote`.
    evicted: Any = None

    @property
    def text(self) -> str:
        return f"{self.detail} -- {self.changed}" if self.changed else self.detail

    def __str__(self) -> str:
        return f"{self.kind}: {self.text}" if self.text else str(self.kind)


#: What each link of the state chain folds (`KeyBuilder.build`), for a
#: change that no named part of the ledger accounts for.
_STATE_STAGES = (
    "its code",
    "a variable it captures",
    "a parameter default",
    "the instance it is bound to",
    "a global it reads",
    "the random-seed epoch",
    "the environment it reads",
    "the class of an argument",
    "a function or class passed as an argument",
)


#: `file_dep_is_fresh` reason codes, as the miss reason and explain() say them.
STALE_REASON_TEXT = {
    "unreadable": "file missing",
    "missing": "file missing",
    "unrecorded": "no usable snapshot of it was recorded",
    "size": "size changed",
    "content": "content changed",
    "mtime": "mtime changed",
    "mtime-sampled": "mtime changed (sampled file)",
    "ctime-sampled": "the file was written (sampled file)",
    "hash-mode": (
        "fingerprinted under a different file_hash_full_max_bytes, so "
        "it could not be compared -- the file itself may be unchanged"
    ),
    "appeared": "a file the call looked for and did not find now exists",
    "unresolved": "the file was read but could not be stat'ed, so it is never proven fresh",
    "remote-changed": "remote object changed",
    "remote-unresolved": "remote object could not be checked",
}


#: How many keys' store outcomes to remember. It explains the recent past;
#: a long-running service does not need the whole history to do that.
STORE_OUTCOMES_MAX = 4096


def same_file_key(path: str) -> str:
    """One spelling per file: the tracker can record a file under the relative
    path the code opened it by AND its absolute path, which listed it twice --
    "and 1 more" was the same file."""
    try:
        return os.path.normcase(os.path.realpath(path))
    except (OSError, ValueError, TypeError):
        return path


def describe_file_deps(deps: dict[str, Any] | None) -> dict[str, str]:
    """``{path: fingerprint}`` for the files an entry recorded, readably."""
    out: dict[str, str] = {}
    seen: set[str] = set()
    for path, rec in (deps or {}).items():
        same = same_file_key(path)
        if same in seen:
            continue
        seen.add(same)
        if not isinstance(rec, dict):
            out[path] = str(rec)
            continue
        if rec.get("absent"):
            out[path] = "absent when read"
            continue
        if rec.get("unresolved") and not rec.get("remote"):
            out[path] = "could not be stat'ed; never fresh"
            continue
        parts = ["remote"] if rec.get("remote") else []
        if rec.get("size") is not None:
            parts.append(f"{rec['size']} bytes")
        if rec.get("hash") and is_sampled_dep(rec):
            # Printed like a full hash, it read as proof of content that it is
            # not.
            parts.append(
                f"sampled hash {str(rec['hash'])[:12]} (head, middle and "
                f"tail only; the rest is trusted to its timestamps)"
            )
        elif rec.get("hash"):
            parts.append(f"hash {str(rec['hash'])[:12]}")
        out[path] = ", ".join(parts) or "recorded"
    return out


def is_sampled_dep(rec: Any) -> bool:
    """Was this file fingerprinted by sampling (larger than
    ``file_hash_full_max_bytes``)? The snapshot says so itself: every
    snapshot records a ctime now, so its presence says nothing."""
    return isinstance(rec, dict) and rec.get("sampled") is True


def describe_state_change(old: dict[str, str], new: dict[str, str]) -> str | None:
    """What differs between two flattened state ledgers, in words, or None.

    Named parts first -- its source, each helper and cached function it calls,
    each global it reads. When none of them moved, the first link of the state
    chain that did says which fold changed (a capture, a default, ...).
    """

    def named(ledger: dict[str, str]) -> dict[str, str]:
        return {k: v for k, v in ledger.items() if not k.startswith("@")}

    before, after = named(old), named(new)
    gone = [k for k in before if k not in after]
    added = [k for k in after if k not in before]
    phrases: list[str] = []
    # A helper whose code arrived under another name: moved, not edited.
    for name in list(gone):
        if not name.startswith("helper "):
            continue
        twin = next((k for k in added if k.startswith("helper ") and after[k] == before[name]), None)
        if twin is not None:
            gone.remove(name)
            added.remove(twin)
            phrases.append(f"{name} moved to {twin.split(' ', 1)[1]}")
    for name in after:
        if name in before and before[name] != after[name]:
            phrases.append("its own source changed" if name == "source" else f"{name} changed")
    phrases += [f"it no longer uses {name}" for name in gone]
    phrases += [f"it now uses {name}" for name in added]
    if not phrases:
        links = sorted((k for k in new if k.startswith("@")), key=lambda k: int(k[1:]))
        for link in links:
            i = int(link[1:])
            if old.get(link) != new[link] and i < len(_STATE_STAGES):
                return f"{_STATE_STAGES[i]} changed"
        return None
    shown = phrases[:3]
    if len(phrases) > 3:
        shown.append(f"and {len(phrases) - 3} more")
    return "; ".join(shown)


def entry_id_of(cache_key: str) -> str:
    """The id `cash inspect` and `cash clear --entry` use for *cache_key*."""
    return hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:12]


def stale_file_deps(metadata: CacheMetadata) -> dict[str, str]:
    """``{path: what changed}`` for each recorded dependency that moved.

    The same freshness check a lookup makes, so the answer cannot
    contradict the behaviour it explains.
    """

    stale: dict[str, str] = {}
    seen: set[str] = set()
    for path, recorded in (metadata.auto_file_deps or {}).items():
        here = dep_path_for_this_process(path, recorded)
        same = same_file_key(here)
        if same in seen:
            continue
        resolved, is_fresh, why = dep_is_fresh(path, recorded)
        if not is_fresh:
            seen.add(same)
            stale[resolved or here] = STALE_REASON_TEXT.get(why or "", "changed")
    return stale


def not_persisted_reason(stored_meta: dict[str, Any]) -> str | None:
    """Why a stored value reached only RAM, or ``None`` if it went further.

    Only a tiered backend says where a value landed; anything else reports
    nothing, and nothing is claimed.
    """
    tiers = stored_meta.get("storage")
    skipped = stored_meta.get("persist_skipped")
    if not isinstance(tiers, list) or skipped is None or any(t != "RAM" for t in tiers):
        return None
    if skipped == "size":
        return "too big for the persistent tier's size cap"
    # The notebook's other reasons (the persistence floor, the cost model,
    # the rate ceiling) cannot apply to a decorated result: `@cash.cache`
    # persists what it is given, and only a size cap stops it (see
    # `TieredBackend.set`).
    return None


def describe_dynamic_dependencies(
    dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None,
    args: tuple,
    kwargs: dict,
) -> list[str]:
    """Best-effort list of the ``DataSource`` ids a function's
    ``dynamic_depends_on`` resolves to for these args - so ``explain()`` can
    report *what* is being tracked. Returns ``[]`` when there are none or
    resolution fails (introspection must never raise)."""
    if not dynamic_depends_on:
        return []
    resolvers = dynamic_depends_on if isinstance(dynamic_depends_on, list) else [dynamic_depends_on]
    ids: list[str] = []
    for resolver in resolvers:
        try:
            ds_result = resolver(*args, **kwargs)
        except Exception:  # noqa: BLE001 - explain() is best-effort
            continue
        dss = ds_result if isinstance(ds_result, list) else [ds_result]
        for ds in dss:
            if isinstance(ds, DataSource):
                try:
                    ids.append(ds.get_id())
                except Exception:  # noqa: BLE001 - a user DataSource; explain() is best-effort
                    ids.append(repr(ds))
    return ids


def describe_stale_files(metadata: CacheMetadata) -> str:
    """The first file a stale entry read that has changed since, and why."""
    stale = stale_file_deps(metadata)
    if not stale:
        return "a file it read"
    path, why = next(iter(stale.items()))
    more = f" and {len(stale) - 1} more" if len(stale) > 1 else ""
    return f"{path} ({why}){more}"


class MissHistory:
    """What this process remembers about the keys it looked up and stored, so
    that a miss can say why it missed.

    In-process memory only, and bounded: it explains, it never decides
    anything. (The last key per function is on its `CachedFunction`.)
    """

    def __init__(
        self, cached: dict[str, CachedFunction], stored_keys: StoredKeyRecord, backend_slot: BackendSlot | None = None
    ) -> None:
        self._cached = cached
        self._stored_keys = stored_keys
        self._backend_slot = backend_slot
        #: cache_key -> what happened when it was last computed here.
        self.outcomes: OrderedDict[str, dict[str, Any]] = OrderedDict()
        # cache_key -> the reason for a lookup that just missed, taken by the
        # `CallLog.log` that reports it (`take_pending`).
        self._pending: dict[str, MissReason] = {}
        # (func_name, state segment) -> the ledger of the key build that first
        # produced it (`keep_state_ledger`).
        self._ledgers: LruMemo[tuple[str, str], dict] = LruMemo(STATE_LEDGERS)

    def note_miss(self, func_name: str, cache_key: str, reason: MissReason) -> None:
        """Hold *reason* for the `CallLog.log` that reports this miss."""
        if len(self._pending) > STORE_OUTCOMES_MAX:
            # Only a call that raised leaves one behind; never let those pile up.
            self._pending.clear()
        self._pending[cache_key] = reason
        cf = self._cached.get(func_name)
        if cf is not None:
            cf.last_key = cache_key

    def take_pending(self, cache_key: str) -> MissReason | None:
        """The reason `note_miss` held for *cache_key*, removed."""
        return self._pending.pop(cache_key, None)

    def pending_eviction(self, cache_key: str) -> Any:
        """The `EvictionNote` of the miss held for *cache_key*, if a size cap
        had evicted its entry. Leaves the reason for `take_pending`."""
        reason = self._pending.get(cache_key)
        return reason.evicted if reason is not None else None

    def _eviction_note(self, cache_key: str) -> Any:
        """Did the backend's size cap evict *cache_key*'s entry? Only asks a
        backend already built, and never fails a call over it."""
        backend = self._backend_slot.built if self._backend_slot is not None else None
        if backend is None:
            return None
        try:
            return backend.eviction_note(cache_key)
        except Exception:  # noqa: BLE001 - a miss reason is a diagnostic
            return None

    def outcome(self, cache_key: str) -> dict[str, Any] | None:
        """What happened when *cache_key* was last computed here, if known."""
        return self.outcomes.get(cache_key)

    def absent_entry_reason(self, func_name: str, cache_key: str) -> MissReason:
        """Why there is no entry for *cache_key*. Reads state; changes none.

        This process's own history first. With none -- the first call of a
        function in a fresh process, which is where a script's misses are --
        the keys earlier runs stored for this function, recorded beside the
        cache (`StoredKeyRecord`). Without them every such miss read "no
        earlier run left one on disk", including after a code edit and a TTL
        expiry, whose entries were in fact on disk.
        """
        outcome = self.outcomes.get(cache_key)
        if outcome is not None:
            if outcome.get("not_stored"):
                return MissReason(MissKind.NOT_STORED, outcome["not_stored"])
            written_ttl = outcome.get("ttl")
            age = time.time() - outcome.get("stored_at", 0)
            if ttl_expired(outcome.get("stored_at", 0), written_ttl):
                return MissReason(MissKind.TTL, f"written {age:.1f}s ago with ttl={written_ttl}s")
            evicted = self._eviction_note(cache_key)
            if evicted is not None:
                return MissReason(
                    MissKind.GONE, f"stored earlier in this process and since {_EVICTED}", evicted=evicted
                )
            return MissReason(MissKind.GONE, "stored earlier in this process and since evicted or cleared")
        cf = self._cached.get(func_name)
        previous = cf.last_key if cf is not None else None
        since = "since the last call"
        doc = self._stored_keys.read(func_name)
        record = doc["keys"]
        if cache_key in record:
            stored_at, written_ttl = record[cache_key][:2]
            age = time.time() - stored_at
            if ttl_expired(stored_at, written_ttl):
                return MissReason(MissKind.TTL, f"stored {age:.0f}s ago by an earlier run, with ttl={written_ttl}s")
            evicted = self._eviction_note(cache_key)
            if evicted is not None:
                return MissReason(MissKind.GONE, f"an earlier run stored it; it was {_EVICTED}", evicted=evicted)
            return MissReason(MissKind.GONE, "an earlier run stored it; it has since been evicted or cleared")
        if cache_key in doc["ram_only"]:
            why = doc["ram_only"][cache_key][1]
            return MissReason(
                MissKind.NOT_STORED,
                f"an earlier run computed it but kept it in RAM only ({why}), so this process recomputed it",
            )
        # Stored so long ago that the record has let it go, but the cap's own
        # notes still know it.
        evicted = self._eviction_note(cache_key)
        if evicted is not None:
            return MissReason(MissKind.GONE, f"an earlier run stored it; it was {_EVICTED}", evicted=evicted)
        # The same arguments stored under another state: the code or a value
        # it reads changed. Asked of the record BEFORE the call-to-call
        # comparison, which after a code edit blamed "new arguments" on every
        # call of a loop but the first.
        new_parts = cache_key.rsplit(":", 3)
        if len(new_parts) == 4:
            for key in reversed([*record, *doc["ram_only"]]):
                old_parts = key.rsplit(":", 3)
                if len(old_parts) == 4 and old_parts[2:] == new_parts[2:] and old_parts[1] != new_parts[1]:
                    return self._code_changed(
                        func_name, old_parts[1], new_parts[1], doc, "since an earlier run stored it"
                    )
            # Earlier runs stored entries, and none under the state this
            # process computes: every one of them is out of date, whatever the
            # arguments. A changed DEFAULT moves the arguments too (they are
            # keyed with defaults applied), so the match above cannot see it,
            # and the call-to-call comparison below called 3 of 4 such misses
            # "new arguments".
            earlier = {
                key: value
                for kind in ("keys", "ram_only")
                for key, value in doc[kind].items()
                if value and isinstance(value[0], (int, float)) and value[0] < PROCESS_STARTED
            }
            states = {key.rsplit(":", 3)[1] for key in earlier if key.count(":") >= 3}
            if states and new_parts[1] not in states:
                newest = max(earlier, key=lambda key: earlier[key][0])
                return self._code_changed(
                    func_name,
                    newest.rsplit(":", 3)[1],
                    new_parts[1],
                    doc,
                    "since an earlier run stored its entries, so none of them applies",
                )
        if previous is None or previous == cache_key:
            others = [key for key in record if key != cache_key]
            if previous is None and others:
                previous = others[-1]
                since = "since an earlier run stored it"
            if previous is None or previous == cache_key:
                return MissReason(
                    MissKind.FIRST, "the first call with these arguments in this process, and no earlier run stored one"
                )
        # Keys are `func:state:dynamic:args`; the parts that moved say why.
        old = previous.rsplit(":", 3)
        new = cache_key.rsplit(":", 3)
        if len(old) != 4 or len(new) != 4:
            return MissReason(MissKind.FIRST, "no entry for this key")
        moved: list[tuple[MissKind, str]] = []
        what = None
        if old[1] != new[1]:
            moved.append((MissKind.CODE, f"{_CODE_CHANGED} {since}"))
            what = self._what_changed(func_name, old[1], new[1], doc)
        if old[2] != new[2]:
            moved.append((MissKind.DYNAMIC, "a dynamic_depends_on source changed"))
        if old[3] != new[3]:
            last = "on the last call" if since == "since the last call" else "in the last run"
            moved.append((MissKind.ARGS, f"called with arguments not seen {last}"))
        if not moved:
            return MissReason(MissKind.FIRST, "no entry for this key")
        return MissReason(moved[0][0], "; and ".join(detail for _, detail in moved), what)

    def _code_changed(self, func_name: str, old_state: str, new_state: str, doc: dict, since: str) -> MissReason:
        """A "code or state changed" reason, naming what changed when known."""
        return MissReason(
            MissKind.CODE, f"{_CODE_CHANGED} {since}", self._what_changed(func_name, old_state, new_state, doc)
        )

    def keep_state_ledger(self, slot: tuple[str, str], ledger: dict) -> None:
        """Keep the ledger of the first key build that produced this
        ``(func_name, state)``."""
        self._ledgers[slot] = ledger

    def has_ledger(self, slot: tuple[str, str]) -> bool:
        """Is a ledger kept for ``(func_name, state)``?"""
        return slot in self._ledgers

    def flat_ledger(self, func_name: str, state: str, doc: dict | None = None) -> dict[str, str] | None:
        """``{part: short digest}`` for *state*: this process's ledger, else the record's."""
        ledger = self._ledgers.get((func_name, state))
        if ledger is None:
            recorded = (doc or {}).get("states", {}).get(state)
            return recorded if isinstance(recorded, dict) else None

        def short(value: Any) -> str:
            return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:10]

        flat: dict[str, str] = {}
        grouped: dict[str, list] = {}
        for label, value in list(ledger.items()):
            if label == "@chain":
                for i, link in enumerate(value):
                    flat[f"@{i}"] = short(link)
            elif label == "source":
                flat["source"] = short(value)
            elif isinstance(label, tuple) and label[0] == "globals":
                via = f" (read by {label[1]})" if label[1] else ""
                for name, digest in value:
                    # `X#carried`, `X#cls:C`: more of what global X is.
                    grouped.setdefault(f"global {name.split('#', 1)[0]}{via}", []).append((name, digest))
            elif isinstance(label, tuple) and label[0] == "env":
                # Already worded: "environment variable TENANT".
                flat[label[1]] = short(value)
            elif isinstance(label, tuple):
                kind = "cached function" if label[0] == "calls" else label[0]
                flat[f"{kind} {label[1]}"] = short(value)
        for name, parts in grouped.items():
            flat[name] = short(sorted(parts))
        return flat

    def _what_changed(self, func_name: str, old_state: str, new_state: str, doc: dict | None = None) -> str | None:
        """Name what moved between two states of *func_name*, or None if unknown.

        "code or state changed" alone sends the user to diff their own edits: a moved helper, an edited constant, a changed default, a
        path whose case differed by launch mode all read the same.
        """
        old = self.flat_ledger(func_name, old_state, doc)
        new = self.flat_ledger(func_name, new_state, doc)
        if not old or not new:
            return None
        return describe_state_change(old, new)

    def remember_outcome(self, cache_key: str, outcome: dict[str, Any]) -> None:
        outcome.setdefault("at", time.time())
        self.outcomes[cache_key] = outcome
        self.outcomes.move_to_end(cache_key)
        while len(self.outcomes) > STORE_OUTCOMES_MAX:
            self.outcomes.popitem(last=False)

    def note_not_stored(self, cache_key: str, refusal: str) -> None:
        self.remember_outcome(cache_key, {"not_stored": refusal})


def _owner_class(func: Callable[..., Any]) -> type | None:
    """The class a plain function was defined in, found through its
    ``__qualname__`` from its module's globals; None for a module-level or
    local function, or one whose class cannot be reached."""
    parts = getattr(func, "__qualname__", "").split(".")
    if len(parts) < 2 or "<locals>" in parts:
        return None
    owner: Any = getattr(func, "__globals__", {}).get(parts[0])
    for part in parts[1:-1]:
        owner = getattr(owner, part, None)
    return owner if isinstance(owner, type) else None


def check_explain_arguments(cf: CachedFunction, args: tuple, kwargs: dict) -> None:
    """Refuse arguments no call of *cf* could be made with.

    ``m.score.explain(2)`` for a cached method: the instance is not passed,
    since ``explain`` is an attribute of the function, not of the bound
    method, so ``2`` was taken as ``self`` and the answer (``no_entry``) was
    for another call than ``m.score(2)``, which then hit. A method is
    explained through its class, ``Model.score.explain(m, 2)``.

    Raises:
        TypeError: the arguments do not bind to the signature, or a method's
            first argument is not an instance (or subclass, for ``cls``) of
            the class it is defined in.
    """
    func = cf.func
    signature = cf.signature
    if signature is not None:
        try:
            signature.bind(*args, **kwargs)
        except TypeError as exc:
            raise TypeError(
                f"{cf.name}.explain(...): these arguments cannot call it: {exc}.{_method_hint(cf)}"
            ) from None
    if not inspect.isfunction(func) or signature is None:
        return
    owner = _owner_class(func)
    params = list(signature.parameters.values())
    if owner is None or not params or not args or params[0].name not in ("self", "cls"):
        return
    first = args[0]
    if params[0].name == "self" and isinstance(first, owner):
        return
    if params[0].name == "cls" and isinstance(first, type) and issubclass(first, owner):
        return
    raise TypeError(
        f"{cf.name}.explain(...): the first argument is {type(first).__name__} {first!r}, "
        f"not the {owner.__name__} {'class' if params[0].name == 'cls' else 'instance'} "
        f"the method is called on.{_method_hint(cf)}"
    )


def _method_hint(cf: CachedFunction) -> str:
    owner = _owner_class(cf.func) if inspect.isfunction(cf.func) else None
    if owner is None:
        return ""
    name = cf.func.__name__
    first = next(iter(cf.signature.parameters), None) if cf.signature is not None else None
    if first == "cls":
        return f" For a classmethod, pass the class: {owner.__name__}.{name}.__func__.explain({owner.__name__}, ...)."
    return (
        f" For a method, pass the instance: {owner.__name__}.{name}.explain(obj, ...). "
        f"obj.{name}.explain(...) does not pass it."
    )


class Explainer:
    """``f.explain()``: why the next call with some arguments would hit or miss."""

    def __init__(
        self,
        config: CashConfig,
        registry: FunctionRegistry,
        keys: KeyBuilder,
        args: ArgHasher,
        frozen: FrozenResults,
        backend_slot: BackendSlot,
        misses: MissHistory,
    ) -> None:
        self._config = config
        self._registry = registry
        self._keys = keys
        self._args = args
        self._frozen = frozen
        self._backend_slot = backend_slot
        self._misses = misses

    def explain(self, cf: CachedFunction, args: tuple, kwargs: dict) -> CacheExplanation:
        """Return why a call with these args would hit or miss the cache.

        Pure introspection - does NOT call ``func``, does NOT touch
        `Cash` stats, does NOT emit warnings, and does NOT
        mutate the backend. The key comes from `KeyBuilder.build`, the same
        build a real call uses, and the entry is judged by the rules
        `CallRunner._try_get_cached` applies, so the answer reflects what would
        actually happen on the next real call.

        See `CacheExplanation` for the return shape.
        """
        func, func_name, dynamic_depends_on, ttl = cf.func, cf.name, cf.dynamic_depends_on, cf.ttl
        check_explain_arguments(cf, args, kwargs)
        if self._config.disable:
            return CacheExplanation(
                would_hit=False,
                reason=EXPLAIN_DISABLED,
                func_name=func_name,
                details={"hint": "Caching is disabled (disable=True / CASH_DISABLE): every call runs the function."},
            )
        # Populate the dependency closure first so the state hash matches what
        # a real call computes (otherwise explain() reports a stale pre-analysis
        # key and a false `no_entry`). This only fills internal
        # analysis caches; it does not warn, run the function, or touch the
        # backend.
        self._registry.ensure_closure_analyzed(func)
        # Same binding check a real call makes first: a patched helper
        # changes the key, and a mock means the call would run uncached.
        unkeyable = self._registry.refresh_helper_bindings(func, func_name)
        if unkeyable is not None:
            return CacheExplanation(
                would_hit=False,
                reason=EXPLAIN_KEY_UNCOMPUTABLE,
                func_name=func_name,
                details={
                    "error": MissKind.MOCKED.value,
                    "hint": f"{unkeyable}, which has no code to key, so the call would run uncached.",
                },
            )

        # The key a real call builds, built the same way, with every warning
        # a step would give held back: explain() must stay silent.
        token = _EXPLAINING.set(True)
        try:
            built = self._keys.build(func, func_name, dynamic_depends_on, args, kwargs)
        except UnhashableDefault:
            return CacheExplanation(
                would_hit=False,
                reason=EXPLAIN_KEY_UNCOMPUTABLE,
                func_name=func_name,
                details={
                    "error": "unhashable parameter default",
                    "hint": (
                        "A parameter default could not be hashed, so cash "
                        "cannot detect a change to it and will not cache "
                        "this call."
                    ),
                },
            )
        except UnhashableArgs:
            arg_type_name = self._args.first_unhashable_arg_type(args, kwargs)
            return CacheExplanation(
                would_hit=False,
                reason=EXPLAIN_KEY_UNCOMPUTABLE,
                func_name=func_name,
                details={
                    "arg_type": arg_type_name,
                    "hint": (
                        unhashable_arg_fix(self._args.first_unhashable_arg(args, kwargs), arg_type_name)
                        if arg_type_name != "<unknown>"
                        else "Could not identify the offending argument; likely a nested unpicklable value."
                    ),
                },
            )
        except KeyBuildFailed as e:
            return CacheExplanation(
                would_hit=False,
                reason=EXPLAIN_KEY_UNCOMPUTABLE,
                func_name=func_name,
                details={"error": e.code, "hint": f"{e.message} {e.fix}"},
            )
        except Exception as e:  # noqa: BLE001 - explain() reports, never raises
            return CacheExplanation(
                would_hit=False,
                reason=EXPLAIN_KEY_UNCOMPUTABLE,
                func_name=func_name,
                details={
                    "error": f"{type(e).__name__}: {e}",
                    "hint": "Building the cache key raised, so the call would run uncached.",
                },
            )
        finally:
            _EXPLAINING.reset(token)
        cache_key = built.cache_key
        frozen_args = self._frozen.arg_names(built.normalized_args)

        # Looking, not reading: `get` would count this as a use (USES / LAST
        # USED in `cash inspect`) and make the file backend rewrite the entry.
        raw_metadata = self._backend_slot.backend.peek_metadata(cache_key)
        if raw_metadata is not None and raw_metadata.get("metadata_only"):
            raw_metadata = None  # nothing to restore: a real call misses
        if raw_metadata is None:
            details = {
                "hint": ("No matching cache entry. First call with these arguments, or the cache was cleared."),
            }
            # A tracked dynamic dependency that changed produces a NEW cache key,
            # so the miss surfaces as no_entry rather than file_changed. Make the
            # explanation say so and list what's tracked.
            dyn_ids = describe_dynamic_dependencies(dynamic_depends_on, args, kwargs)
            if dyn_ids:
                details["dynamic_dependencies"] = dyn_ids
                details["hint"] = (
                    "No matching cache entry. Either the first call with these "
                    "arguments, or a tracked dynamic dependency changed - a "
                    "dynamic_depends_on change yields a new cache key, so it "
                    "shows up here as no_entry, not file_changed. Tracked "
                    "dynamic dependencies: " + ", ".join(dyn_ids) + "."
                )
            # What this process knows about the key says more than "first call
            # or cleared": that it was never stored, why, or that it expired
            # under the ttl it was WRITTEN with -- which a backend drops on
            # read, so the entry looks absent.
            missed = self._misses.absent_entry_reason(func_name, cache_key)
            if missed.kind is MissKind.TTL:
                return CacheExplanation(
                    would_hit=False,
                    reason=EXPLAIN_TTL_EXPIRED,
                    func_name=func_name,
                    cache_key=cache_key,
                    details={"why": missed.text},
                )
            details["why"] = str(missed)
            if missed.kind is not MissKind.FIRST and "dynamic_dependencies" not in details:
                del details["hint"]  # the generic guess, now that we know
            if frozen_args:
                details["frozen_args"] = frozen_args
            return CacheExplanation(
                would_hit=False,
                reason=EXPLAIN_NO_ENTRY,
                func_name=func_name,
                cache_key=cache_key,
                details=details,
            )

        metadata = CacheMetadata.from_dict(raw_metadata)

        # TTL check - the same rule `CallRunner._try_get_cached` applies.
        ttl = self._backend_slot.entry_ttl(ttl, metadata)
        if ttl_expired(metadata.timestamp, ttl):
            timestamp = metadata.timestamp or 0
            age = time.time() - timestamp
            return CacheExplanation(
                would_hit=False,
                reason=EXPLAIN_TTL_EXPIRED,
                func_name=func_name,
                cache_key=cache_key,
                details={
                    "ttl_seconds": ttl,
                    "age_seconds": age,
                    "cached_at": timestamp,
                },
            )

        # Auto-tracked file deps freshness. Routed through the SAME
        # content-authoritative helper a real lookup uses - comparing
        # raw mtime/size here made explain() report file_changed / 'mtime
        # changed' after a touch while the actual call hit. A diagnostic that
        # contradicts the behavior it describes is worse than none.
        if metadata.auto_file_deps:
            stale = stale_file_deps(metadata)
            if stale:
                return CacheExplanation(
                    would_hit=False,
                    reason=EXPLAIN_FILE_CHANGED,
                    func_name=func_name,
                    cache_key=cache_key,
                    details={"changed_files": stale, "file_deps": describe_file_deps(metadata.auto_file_deps)},
                )

        timestamp = metadata.timestamp or 0
        details = {
            "cached_at": timestamp,
            "cache_age_seconds": time.time() - timestamp if timestamp else None,
            "execution_time_saved": metadata.execution_time or 0.0,
        }
        if metadata.auto_file_deps:
            details["file_deps"] = describe_file_deps(metadata.auto_file_deps)
        if frozen_args:
            details["frozen_args"] = frozen_args
        return CacheExplanation(
            would_hit=True,
            reason=EXPLAIN_HIT,
            func_name=func_name,
            cache_key=cache_key,
            details=details,
        )
