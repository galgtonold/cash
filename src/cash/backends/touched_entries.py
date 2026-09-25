"""What a `FileBackend` knows about the entries this process has touched."""

from __future__ import annotations

import threading
from collections.abc import Callable

__all__ = ["TouchedEntries", "stat_signature"]


class TouchedEntries:
    """The metadata of the keys this process read or wrote, the way back from
    an entry's path to its key, and the keys whose reads are not yet flushed.

    A filename is a SHA-256 of the key, so nothing recovers the key from a
    path. Eviction ranks entries by walking the directory, which yields paths:
    `key_for` is how it finds the metadata that belongs to one. Only the keys
    this process touched are here, which is exactly the set that has any
    bookkeeping.

    Each key's metadata is held with the `entry_identity` of the write it was
    read from, and the ``os.stat`` signature of the file when that is known:
    other processes replace entries under this one, and metadata remembered
    from an earlier write must never be paired with a later write's payload.

    `FileBackend` and `FileEvictor` both go through this object. `lock`
    guards it and the evictor's accounting too, so forgetting an entry here
    and its bytes there is one step.
    """

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self._metadata: dict[str, dict] = {}
        self._keys: dict[str, str] = {}
        self._unflushed: set[str] = set()
        #: key -> (entry identity, stat signature or None).
        self._origin: dict[str, tuple[tuple | None, tuple | None]] = {}

    def __len__(self) -> int:
        return len(self._metadata)

    def remember(
        self, key: str, path: str, metadata: dict, identity: tuple | None = None, stat: tuple | None = None
    ) -> None:
        """Hold *metadata* as *key*'s, stored at *path* by the write *identity*
        names, in a file whose `stat_signature` was *stat*."""
        with self.lock:
            self._metadata[key] = metadata
            self._keys[path] = key
            self._origin[key] = (identity, stat)

    def identity(self, key: str) -> tuple | None:
        """The `entry_identity` of the write *key*'s metadata was read from."""
        origin = self._origin.get(key)
        return origin[0] if origin else None

    def stat_matches(self, key: str, stat: tuple) -> bool:
        """Is *stat* the signature the file had when *key*'s metadata was read?"""
        origin = self._origin.get(key)
        return origin is not None and origin[1] is not None and origin[1] == stat

    def note_stat(self, key: str, stat: tuple) -> None:
        """*key*'s file, now signed *stat*, still holds the write its metadata came from."""
        with self.lock:
            origin = self._origin.get(key)
            if origin is not None:
                self._origin[key] = (origin[0], stat)

    def metadata(self, key: str | None) -> dict | None:
        """The metadata this process holds for *key*, or None."""
        return self._metadata.get(key) if key is not None else None

    def key_for(self, path: str) -> str | None:
        """The key stored at *path*, if this process touched it."""
        return self._keys.get(path)

    def drop_metadata(self, key: str) -> None:
        """Stop holding *key*'s metadata: its file was not there."""
        with self.lock:
            self._metadata.pop(key, None)
            self._origin.pop(key, None)
            self._unflushed.discard(key)

    def note_read(self, key: str) -> None:
        """*key* was read: its access stamp is due to be written back."""
        with self.lock:
            self._unflushed.add(key)

    def has_unflushed_read(self, key: str | None) -> bool:
        return key in self._unflushed

    def take_unflushed(self, due: Callable[[str], bool] | None = None) -> list[str]:
        """The keys with unflushed reads that are *due* (all without it), no
        longer marked unflushed."""
        with self.lock:
            if due is None:
                keys = list(self._unflushed)
                self._unflushed.clear()
            else:
                keys = [k for k in self._unflushed if due(k)]
                self._unflushed.difference_update(keys)
            return keys

    def forget(self, key: str | None, path: str) -> None:
        """The entry at *path* (*key*'s, when known) is gone."""
        with self.lock:
            if key is not None:
                self._metadata.pop(key, None)
                self._origin.pop(key, None)
                self._unflushed.discard(key)
            self._keys.pop(path, None)

    def clear(self) -> None:
        """Every entry is gone."""
        with self.lock:
            self._metadata.clear()
            self._origin.clear()
            self._unflushed.clear()
            self._keys.clear()


def stat_signature(st) -> tuple:
    """What changes when a file is replaced or rewritten: inode, size, mtime."""
    return (st.st_ino, st.st_size, st.st_mtime_ns)
