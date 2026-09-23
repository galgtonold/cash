"""Checking an input's file dependencies costs one metadata read, not one per file.

Round 23 (r23s2): ``raw`` was read from 1,200 CSVs. Every statement reading
``raw`` checked all 1,200 of its files, and fetched ``raw``'s producer entry
with ``backend.get`` once PER FILE -- which deep-copies the cached frame. A
0.4 s cell took 42 s when served from the cache.
"""

import types

from cash.notebook.file_dep_snapshot import snapshot_file_deps
from cash.notebook.statement.freshness import CacheFreshnessChecker


class _CountingBackend:
    def __init__(self, metadata):
        self._metadata = metadata
        self.gets = 0
        self.peeks = 0

    def get(self, key):
        self.gets += 1
        return self._metadata, "a big cached frame"

    def peek_metadata(self, key):
        self.peeks += 1
        return dict(self._metadata)


def _files(tmp_path, n):
    paths = []
    for i in range(n):
        p = tmp_path / f"e{i:03d}.csv"
        p.write_text(f"g,v\n{i},1\n")
        paths.append(str(p))
    return paths


def _state(paths):
    return types.SimpleNamespace(
        executed_file_deps={"raw": set(paths)},
        variable_sources={"raw": "stmt:producer"},
    )


def test_one_metadata_read_per_input_and_no_value_copies(tmp_path):
    paths = _files(tmp_path, 50)
    backend = _CountingBackend({"key": "stmt:producer", "file_dependencies": snapshot_file_deps(set(paths))})
    checker = CacheFreshnessChecker(backend)
    assert checker._invalidate_if_input_file_changed(_state(paths), {"raw"}, "payload") == "payload"
    assert backend.gets == 0, "the producer's cached value was fetched to read its metadata"
    assert backend.peeks == 1, f"{backend.peeks} metadata reads for one input"


def test_a_changed_file_is_still_caught(tmp_path):
    paths = _files(tmp_path, 50)
    backend = _CountingBackend({"key": "stmt:producer", "file_dependencies": snapshot_file_deps(set(paths))})
    checker = CacheFreshnessChecker(backend)
    with open(paths[37], "w") as fh:
        fh.write("g,v\n9,9\n")
    assert checker._invalidate_if_input_file_changed(_state(paths), {"raw"}, "payload") is None
    assert paths[37].replace("\\", "/").split("/")[-1] in (checker.last_miss_reason or "")


def test_a_backend_without_peek_still_works(tmp_path):
    paths = _files(tmp_path, 3)

    class _GetOnly:
        def get(self, key):
            return {"key": key, "file_dependencies": snapshot_file_deps(set(paths))}, "v"

    checker = CacheFreshnessChecker(_GetOnly())
    assert checker._invalidate_if_input_file_changed(_state(paths), {"raw"}, "payload") == "payload"
