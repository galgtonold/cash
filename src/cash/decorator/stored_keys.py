"""The record of which keys earlier runs stored, kept beside the cache so a
new process can say why its first call missed."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from typing import Any

from ..backends.file_backend import recreate_cache_dir
from ..tracking.file_tracker import untracked

logger = logging.getLogger(__name__)


class StoredKeysMixin:
    """Reads and writes the per-function record of stored keys."""

    #: Keys remembered per function beside the cache, most recent last.
    _STORED_KEYS_MAX = 64

    def _stored_keys_path(self, func_name: str) -> str | None:
        """Where this function's recently stored keys are recorded, or None.

        Beside the entries, in the directory of the backend actually built --
        never a configured path, so reading a miss reason cannot create a
        cache directory -- and only for a backend that has a local directory.
        """
        path = self._backend.local_dir if self._backend is not None else None
        if not path:
            return None
        name = hashlib.sha256(func_name.encode("utf-8")).hexdigest()[:32]
        return os.path.join(path, ".keys", f"{name}.json")

    def _stored_doc(self, func_name: str) -> dict[str, dict[str, list]]:
        """What earlier runs recorded for *func_name*, oldest first per kind.

        ``keys``: ``{cache_key: [stored_at, ttl]}`` for results that reached
        disk. ``ram_only``: ``{cache_key: [computed_at, why]}`` for results a
        run computed and kept in RAM only (see `_remember_ram_only`). Read
        through a memo on the file's (mtime, size), because every miss asks.
        """
        empty: dict[str, dict[str, list]] = {"keys": {}, "ram_only": {}, "states": {}, "warned": {}}
        path = self._stored_keys_path(func_name)
        if path is None:
            return empty

        with self._stored_doc_lock:
            try:
                st = os.stat(path)
            except OSError:
                return empty
            memo = self._stored_doc_memo.get(path)
            if memo is not None and memo[0] == (st.st_mtime_ns, st.st_size):
                return {kind: dict(value) for kind, value in memo[1].items()}
            try:
                # Cash's own bookkeeping: a nested call reads this while the
                # OUTER call's file tracker is live, and it must not become
                # that entry's dependency.
                with untracked(), open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, ValueError):
                # Another process mid-rewrite: the last record read beats none,
                # which reads every miss as "new arguments".
                if memo is not None:
                    return {kind: dict(value) for kind, value in memo[1].items()}
                return empty
            return self._memo_stored_doc(path, st, data)

    def _memo_stored_doc(self, path: str, st: os.stat_result, data: Any) -> dict[str, dict[str, list]]:
        """Keep *data* as the record at *path* as of *st*; return a copy."""
        doc = {}
        for kind in ("keys", "ram_only", "states", "warned"):
            value = data.get(kind) if isinstance(data, dict) else None
            doc[kind] = value if isinstance(value, dict) else {}
        if len(self._stored_doc_memo) >= 256:
            self._stored_doc_memo.clear()
        self._stored_doc_memo[path] = ((st.st_mtime_ns, st.st_size), doc)
        return {kind: dict(value) for kind, value in doc.items()}

    def _write_stored_doc(self, func_name: str, doc: dict[str, dict[str, list]]) -> None:
        """Replace the record for *func_name*. Raises; callers swallow."""
        path = self._stored_keys_path(func_name)
        if path is None:
            return
        with self._ram_only_lock:
            shown = self._warned_pending.pop(func_name, None)
        if shown:
            doc.setdefault("warned", {}).update(shown)
        for kind, most in (
            ("keys", self._STORED_KEYS_MAX),
            ("ram_only", self._STORED_KEYS_MAX),
            ("states", self._STORED_STATES_MAX),
            ("warned", self._STORED_STATES_MAX * 4),
        ):
            entries = doc.setdefault(kind, {})
            while len(entries) > most:
                entries.pop(next(iter(entries)))

        keys_dir = os.path.dirname(path)
        recreate_cache_dir(os.path.dirname(keys_dir))
        os.makedirs(keys_dir, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        with self._stored_doc_lock, untracked():
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"func": func_name, **doc}, fh)
            os.replace(tmp, path)
            # What this process just wrote is what its next miss reads.
            self._memo_stored_doc(path, os.stat(path), doc)

    def _remember_ram_only(self, func_name: str, cache_key: str, why: str) -> None:
        """Note a result this process kept in RAM only, for the NEXT run's reason.

        Without it the next run's miss read "new arguments: not seen in the
        last run" although the last run was called with exactly these (round
        19, 4 of 5 testers). Buffered and written once, at shutdown: a result
        kept in RAM is by definition a quick one, and rewriting the record per
        miss would cost more than the body.
        """
        with self._ram_only_lock:
            pending = self._ram_only_pending.setdefault(func_name, {})
            pending.pop(cache_key, None)
            pending[cache_key] = [time.time(), why]
            while len(pending) > self._STORED_KEYS_MAX:
                pending.pop(next(iter(pending)))

    def _flush_ram_only_keys(self) -> None:
        """Write what `_remember_ram_only` buffered. Never raises."""
        with self._ram_only_lock:
            pending, self._ram_only_pending = self._ram_only_pending, {}
            for func_name in self._warned_pending:
                pending.setdefault(func_name, {})  # its record takes the shown warnings
        for func_name, entries in pending.items():
            try:
                with self._stored_doc_lock:
                    doc = self._stored_doc(func_name)
                    for key, value in entries.items():
                        doc["keys"].pop(key, None)  # its disk copy is gone: this run recomputed it
                        doc["ram_only"].pop(key, None)
                        doc["ram_only"][key] = value
                        self._record_state(func_name, key, doc)
                    self._write_stored_doc(func_name, doc)
            except Exception:  # noqa: BLE001 - a diagnostic aid
                logger.debug("could not record RAM-only keys for %s", func_name, exc_info=True)

    def _record_stored_key(self, func_name: str, cache_key: str, ttl: int | None) -> None:
        """Remember that *cache_key* reached disk, for the next process's reasons.

        One small file per function, rewritten on each persisted store (the
        compute that just ran dwarfs it). A pool's threads take turns (see
        ``_stored_doc_lock``); concurrent PROCESSES race to the last rename,
        and the loser's key is missing from the record, which costs a vaguer
        reason, never a wrong answer. Never raises.
        """
        if self._stored_keys_path(func_name) is None:
            return
        with self._ram_only_lock:
            self._ram_only_pending.get(func_name, {}).pop(cache_key, None)
        try:
            with self._stored_doc_lock:
                doc = self._stored_doc(func_name)
                doc["ram_only"].pop(cache_key, None)
                doc["keys"].pop(cache_key, None)
                doc["keys"][cache_key] = [time.time(), ttl]
                self._record_state(func_name, cache_key, doc)
                self._write_stored_doc(func_name, doc)
        except Exception:  # noqa: BLE001 - a diagnostic aid; the store succeeded
            logger.debug("could not record the stored key for %s", func_name, exc_info=True)

    #: States whose ledger the record keeps per function, most recent last.
    _STORED_STATES_MAX = 8

    def _record_state(self, func_name: str, cache_key: str, doc: dict) -> None:
        """Put the ledger of *cache_key*'s state in *doc*, for the next run's
        "what changed". Once per state; kept most recent last."""
        parts = cache_key.rsplit(":", 3)
        if len(parts) != 4:
            return
        states = doc.setdefault("states", {})
        if parts[1] in states:
            states[parts[1]] = states.pop(parts[1])
            return
        flat = self._flat_ledger(func_name, parts[1])
        if flat:
            states[parts[1]] = flat
