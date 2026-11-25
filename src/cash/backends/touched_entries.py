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

    Only the `MAX_ENTRIES` keys written or read last are held, and every
    key whose read is not flushed yet (its metadata is the one place the
    read is counted). An older key is as one another process wrote: its
    metadata is read from its file when asked again, and eviction ranks
    it by the rank index. Held for every key, the table grew by 800 bytes
    with each distinct call for as long as the process ran.
    """

    #: Keys held at most, besides those with a read not flushed yet.
    MAX_ENTRIES = 20_000

    def __init__(self) -> None:
        self.lock = threading.RLock()
        #: In the order last written or read: the oldest first.
        self._metadata: dict[str, dict] = {}
        self._keys: dict[str, str] = {}
        self._unflushed: set[str] = set()
        #: key -> (entry identity, stat signature or None, path).
        self._origin: dict[str, tuple[tuple | None, tuple | None, str]] = {}
        #: Called with each key let go for being old (`_let_old_go`), with
        #: the lock held: the evictor drops what it keeps by key.
        self.on_let_go: Callable[[str], None] | None = None

    def __len__(self) -> int:
        return len(self._metadata)

    def remember(
        self, key: str, path: str, metadata: dict, identity: tuple | None = None, stat: tuple | None = None
    ) -> None:
        """Hold *metadata* as *key*'s, stored at *path* by the write *identity*
        names, in a file whose `stat_signature` was *stat*."""
        with self.lock:
            self._metadata.pop(key, None)  # last in the order
            self._metadata[key] = metadata
            self._keys[path] = key
            self._origin[key] = (identity, stat, path)
            # Past the bound by an eighth before the oldest go, so a table
            # at its bound is not looked through on every write.
            if len(self._metadata) > self.MAX_ENTRIES + max(self.MAX_ENTRIES >> 3, 1):
                self._let_old_go()

    def _let_old_go(self) -> None:
        """Down to `MAX_ENTRIES` keys, the oldest first, keeping each whose
        read is not flushed yet. With the lock held."""
        excess = len(self._metadata) - self.MAX_ENTRIES
        gone = []
        for key in self._metadata:
            if len(gone) >= excess:
                break
            if key not in self._unflushed:
                gone.append(key)
        for key in gone:
            del self._metadata[key]
            origin = self._origin.pop(key, None)
            if origin is not None and self._keys.get(origin[2]) == key:
                del self._keys[origin[2]]
            if self.on_let_go is not None:
                self.on_let_go(key)

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
                self._origin[key] = (origin[0], stat, origin[2])

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
            metadata = self._metadata.pop(key, None)
            if metadata is not None:
                self._metadata[key] = metadata  # last in the order

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
