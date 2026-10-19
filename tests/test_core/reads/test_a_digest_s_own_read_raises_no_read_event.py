"""Cash's read of a small file for its digest raises no read ``open`` event.

Each file a statement reads is hashed as it is read. The hash opened the
file through ``io.FileIO``, whose ``open`` audit event names a read, so it ran
through cash's own read handlers only to be dropped there: ~15 us of each
2 KB file's digest in a loop reading 6,000. The digest opens a descriptor
now, whose event reports flags. The digest is the same.
"""

from __future__ import annotations

import os
import sys

from cash.bulk_digest import bulk_digest
from cash.tracking import file_dep_snapshot

EVENTS: list[tuple] = []


def _hook(event, args):
    if event == "open" and EVENTS and args[0] == EVENTS[0]:
        EVENTS.append(args)


sys.addaudithook(_hook)


def test_a_small_file_s_digest_opens_no_file_object(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_bytes(b"review text " * 200)
    EVENTS[:] = [str(path)]
    try:
        digest = file_dep_snapshot._read_digest(str(path), os.path.getsize(path))
        opened = EVENTS[1:]
    finally:
        EVENTS.clear()
    assert [mode for _, mode, _ in opened] == [None], opened
    assert digest == bulk_digest(path.read_bytes()).hex()


def test_a_file_that_grew_since_its_stat_is_read_to_its_end(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_bytes(b"a" * 5000)
    assert file_dep_snapshot._read_digest(str(path), 100) == bulk_digest(b"a" * 5000).hex()
