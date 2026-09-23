"""Why a call hit or missed: the miss reasons, `CacheExplanation`, and the
descriptions of what changed."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any

# Reason codes returned by `Cash._explain_call` / ``f.explain(...)``.
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


# Why a call missed. The KIND is what the summary counts; the detail goes to
# the per-call debug line and to explain().
MISS_FIRST = "no entry yet"
MISS_ARGS = "new arguments"
MISS_CODE = "code or state changed"
MISS_DYNAMIC = "dynamic dependency changed"
MISS_FILE = "file changed"
MISS_TTL = "ttl expired"
MISS_NOT_STORED = "not stored last time"
MISS_GONE = "entry gone"
MISS_INCOMPLETE = "entry incomplete"
MISS_UNHASHABLE = "unhashable argument"
MISS_KEY_FAILED = "key could not be built"
MISS_MOCKED = "a helper is a mock"
MISS_RAISED = "raised"


#: Separates a "code or state changed" detail from WHAT changed, which the
#: summary tallies on its own line.
WHAT_CHANGED = " -- "


#: What each link of the state chain folds (`_build_key`), for a
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
    "remote-changed": "remote object changed",
    "remote-unresolved": "remote object could not be checked",
}


#: How many keys' store outcomes to remember. It explains the recent past;
#: a long-running service does not need the whole history to do that.
STORE_OUTCOMES_MAX = 4096


def same_file_key(path: str) -> str:
    """One spelling per file: the tracker can record a file under the relative
    path the code opened it by AND its absolute path, which listed it twice --
    "and 1 more" was the same file (round 18)."""
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
        parts = ["remote"] if rec.get("remote") else []
        if rec.get("size") is not None:
            parts.append(f"{rec['size']} bytes")
        if rec.get("hash") and is_sampled_dep(rec):
            # Printed like a full hash, it read as proof of content that it is
            # not (round 20).
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
    ``file_hash_full_max_bytes``)? Only such snapshots record a ctime."""
    return isinstance(rec, dict) and ("ctime_ns" in rec or "ctime" in rec)


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
