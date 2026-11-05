"""A statement entry's file records, each kept once.

An entry records the files its value was built from
(``StatementCacheMetadata.file_dependencies``): those the statement read,
and those its inputs were built from, as the inputs' producing entries
recorded them. Copied into every entry below a folder read, a trivial
statement under a frame read from 15,000 files wrote a 5 MB record of
them, and a restart read it back to decide to run the statement again.

So an entry keeps in ``file_dependencies`` only the records it made, and
for the ones it took over whole from an entry above, a reference to that
entry (`REFS_FIELD`): its key and the token naming the records it holds
(`TOKEN_FIELD`), a digest of what they say of each file
(`records_token`). A reference holds while the entry under that key holds
records of that token; one stored again with records of the files in
another state, or dropped, leaves it unresolved, and the entry referring
to it is stale (``unresolved``): run again, it records its files afresh.

Each relative read's second spelling is folded into its absolute twin's
record as it is stored (`file_dep_snapshot.fold_relative_twins`) and
unfolded as it is read, so every reader sees the records as they were made.

Every reader of an entry's records goes through `entry_file_records`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, NamedTuple

from cash._memo import FILE_RECORDS, LruMemo
from cash.tracking.file_dep_snapshot import unfold_relative_twins

__all__ = [
    "REFS_FIELD",
    "TOKEN_FIELD",
    "UNRESOLVED_PREFIX",
    "Records",
    "entry_file_records",
    "records_token",
    "own_records",
    "referred_records",
    "with_file_records",
]

#: Metadata field: ``[[key, token], ...]``, the entries whose records this
#: one takes over whole, in the order they were taken.
REFS_FIELD = "file_refs"

#: Metadata field: the token naming the records this entry stores itself.
TOKEN_FIELD = "files_token"

#: The path a reference that could not be followed is recorded under, with
#: the reference's key after it, as an ``unresolved`` record.
UNRESOLVED_PREFIX = "<cash: file records of "


class Records(NamedTuple):
    """The records one entry stores itself, as read back."""

    key: str
    token: str
    records: Mapping[str, Any]
    #: Every path the records name, the relative spellings included.
    paths: frozenset[str]


#: Records read back, by the entry's key and their token.
_OWN: LruMemo[tuple[str, str], Records] = LruMemo(FILE_RECORDS)

#: An entry's records with those it refers to merged in, by the token of
#: its own and its references: ``(own token, *(key, token)) -> dict``.
_MERGED: LruMemo[tuple, Mapping[str, Any]] = LruMemo(FILE_RECORDS)


#: What a record with a content digest (``hash``) says of the file's stat
#: and of when it was hashed: a shortcut of its check, which takes the
#: content as the authority, so not part of what the token names.
_STAT_FIELDS = frozenset({"hashed_at", "mtime", "mtime_ns", "ctime_ns", "dev", "ino"})


def records_token(stored: Mapping[str, Any]) -> str:
    """The token naming *stored*, records about to be stored as they are
    stored (`file_dep_snapshot.fold_relative_twins`): a digest of what they
    say of each file, but for when it was last hashed (``hashed_at``),
    which a later check moves on in place, and, for a file recorded with
    its content digest, its stat (`_STAT_FIELDS`).

    Records of the same files in the same state are named alike, whenever
    they were made: a value kept only in RAM is computed again after a
    restart and stored with records of the files as they are, and the
    entries referring to the ones it stored before hold while those files
    are as they were. A file rewritten with the same bytes is as it was:
    its check of the content finds it fresh, so the records made of it
    after the rewrite name the same state."""
    parts: list[str] = []
    for path in sorted(stored):
        recorded = stored[path]
        parts.append(path)
        if type(recorded) is dict:
            drop = _STAT_FIELDS if recorded.get("hash") is not None else ("hashed_at",)
            parts.append(repr([(k, v) for k, v in recorded.items() if k not in drop]))
        else:
            parts.append(repr(recorded))
    return hashlib.sha256("\0".join(parts).encode("utf-8", "surrogateescape")).hexdigest()[:32]


def _field(meta: Any, name: str) -> Any:
    if isinstance(meta, Mapping):
        return meta.get(name)
    return getattr(meta, name, None)


def own_records(meta: Any, key: str | None = None) -> Records | None:
    """The records the entry *meta* (a metadata dict or
    `StatementCacheMetadata`) stores itself, read back, or None when it
    stores none. *key* is the entry's key when *meta* does not say."""
    stored = _field(meta, "file_dependencies")
    if not stored:
        return None
    token = _field(meta, TOKEN_FIELD)
    key = key or _field(meta, "key") or ""
    if isinstance(token, str):
        found = _OWN.get((key, token))
        if found is not None:
            return found
    records = unfold_relative_twins(stored)
    found = Records(key, token if isinstance(token, str) else "", records, frozenset(records))
    if isinstance(token, str):
        _OWN[(key, token)] = found
    return found


def _peek(backend: Any, key: str) -> Any:
    peek = getattr(backend, "peek_metadata", None) or getattr(backend, "get_metadata", None)
    if peek is None:
        return None
    try:
        return peek(key)
    except Exception:  # noqa: BLE001 - an entry it cannot read is a reference it cannot follow
        return None


def referred_records(backend: Any, key: str, token: str) -> Records | None:
    """The records the entry under *key* stores itself, while they are the
    ones *token* names; None when they are not, or it cannot be read."""
    found = _OWN.get((key, token))
    if found is not None:
        return found
    if backend is None:
        return None
    meta = _peek(backend, key)
    if not meta or _field(meta, TOKEN_FIELD) != token:
        return None
    return own_records(meta, key)


def _refs(meta: Any) -> list[tuple[str, str]]:
    refs = _field(meta, REFS_FIELD)
    if not refs:
        return []
    out = []
    for ref in refs:
        if isinstance(ref, (list, tuple)) and len(ref) == 2 and all(isinstance(part, str) for part in ref):
            out.append((ref[0], ref[1]))
        else:
            out.append((repr(ref), ""))  # unreadable: never resolved
    return out


def entry_file_records(meta: Any, backend: Any) -> Mapping[str, Any]:
    """Every file record of the entry *meta* (a metadata dict or
    `StatementCacheMetadata`), as made: those it stores, then those of each
    entry it refers to (`REFS_FIELD`, read through *backend*), a path
    recorded twice taking the first. A reference that cannot be followed
    is an ``unresolved`` record, which no check finds fresh.

    Shared between callers while the tokens are the same: read, not changed,
    but for what `file_dep_snapshot._note_settled` notes in a record.
    """
    if meta is None:
        return {}
    own = own_records(meta)
    refs = _refs(meta)
    if not refs:
        return own.records if own is not None else {}
    memo_key = (own.token if own is not None else None, *refs)
    # Records without a token (written by hand) are not named by one.
    cacheable = own is None or bool(own.token)
    if cacheable:
        merged = _MERGED.get(memo_key)
        if merged is not None:
            return merged
    parts: list[Mapping[str, Any]] = [own.records] if own is not None else []
    unresolved: dict[str, Any] = {}
    for key, token in refs:
        found = referred_records(backend, key, token) if token else None
        if found is None:
            unresolved[UNRESOLVED_PREFIX + key + ">"] = {"unresolved": True}
        else:
            parts.append(found.records)
    if unresolved:
        merged = dict(parts[0]) if parts else {}
        for part in parts[1:]:
            for path, recorded in part.items():
                merged.setdefault(path, recorded)
        merged.update(unresolved)
        return merged
    if len(parts) == 1:
        merged = parts[0]
    else:
        merged = dict(parts[0])
        for part in parts[1:]:
            for path, recorded in part.items():
                merged.setdefault(path, recorded)
    if cacheable:
        _MERGED[memo_key] = merged
    return merged


def with_file_records(metadata: Any, backend: Any) -> Any:
    """*metadata* (a `StatementCacheMetadata`) with every file record of
    its entry in ``file_dependencies`` (`entry_file_records`), as made, and
    no reference left: what the code that serves a hit reads."""
    if metadata is None or (not metadata.file_refs and not metadata.file_dependencies):
        return metadata
    return replace(
        metadata,
        file_dependencies=entry_file_records(metadata, backend) or None,
        file_refs=None,
        files_token=None,
    )
