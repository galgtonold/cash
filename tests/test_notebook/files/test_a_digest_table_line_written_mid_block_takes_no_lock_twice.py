"""Writing the digest table's own log never settles a deferred digest.

`DigestTable.put` holds the table's lock while its waiting lines are
written (`LogLines.flush_locked`). That ``open`` went through cash's open
hook like any other: where the table's folder was not recognised as cash's
own storage (a cache folder registered under another spelling of its path),
the hook took the deferred digests first, and each digest's line went to
`put` on the same lock. The kernel hung, as E8's MNIST cell did on Windows.
The log write is cash's own work now, which the hook does not see.
"""

from __future__ import annotations

import threading

from cash.tracking import log_lines, pending_digest
from cash.tracking.digest_table import DigestTable
from cash.tracking.file_tracker import FileAccessTracker


def test_a_line_written_while_a_digest_waits_does_not_hang(tmp_path, monkeypatch):
    monkeypatch.setattr(log_lines, "FLUSH_LINES", 1)
    table = DigestTable(str(tmp_path))
    done = threading.Event()

    def run():
        with FileAccessTracker(hash_on_read=True):
            waiting = pending_digest.DeferredDigest(
                lambda _d: table.put(("b.txt", 1, 1, 1, 1, 1), "beta", 1.0) or "beta"
            )
            try:
                table.put(("a.txt", 1, 1, 1, 1, 1), "alpha", 1.0)
            finally:
                waiting.result()
        done.set()

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(10)
    assert done.is_set(), "put hung on the table's own lock"
    assert table.get(("a.txt", 1, 1, 1, 1, 1))[0] == "alpha"
