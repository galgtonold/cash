"""Review of claude/speed-file-deps d8d0dd41: the digest table's compaction.

Put this file in tests/test_core/internals/.

A table that kept more than `_COMPACT_AT` lines (every entry its process
used) is loaded by the next process with `_lines` already past
`_compact_at`, which starts again at `_COMPACT_AT`. That process's first
`put` then compacts while it has used nothing yet, and keeps only the newest
`_KEEP`: the restart case the branch fixes (E10, 50,000 files) comes back as
soon as the restarted notebook hashes any one file before the big read (a
new or edited input, the first changed file inside the loop). Within that
process the dropped digests are gone too, so the loop re-reads them.

Suggested fix: start `_compact_at` at ``max(_COMPACT_AT, 2 * lines loaded)``
(or count the loaded lines as kept), so a table a process kept is not
compacted by the next one before it had a chance to use it.
"""

from __future__ import annotations

from cash.tracking import digest_table


def _key(tag, i):
    return (tag, 1, i, 3, 4, 5)


def test_a_restart_that_hashes_one_new_file_first_keeps_what_the_last_run_used(tmp_path, monkeypatch):
    monkeypatch.setattr(digest_table, "_COMPACT_AT", 50)
    monkeypatch.setattr(digest_table, "_KEEP", 20)
    first = digest_table.DigestTable(str(tmp_path))  # run 1: one statement reads 60 files
    for i in range(60):
        first.put(_key("data", i), "a" * 64, float(i))
    first.flush()
    assert all(digest_table.DigestTable(str(tmp_path)).get(_key("data", i)) for i in range(60)), "run 1 kept all 60"

    second = digest_table.DigestTable(str(tmp_path))  # run 2: a new input is hashed first
    second.put(_key("new", 0), "b" * 64, 1000.0)
    missing = [i for i in range(60) if second.get(_key("data", i)) is None]
    assert missing == [], f"run 2 dropped {len(missing)} of the 60 digests run 1 used"


def test_a_restart_whose_loop_meets_one_changed_file_keeps_the_rest(tmp_path, monkeypatch):
    monkeypatch.setattr(digest_table, "_COMPACT_AT", 50)
    monkeypatch.setattr(digest_table, "_KEEP", 20)
    first = digest_table.DigestTable(str(tmp_path))
    for i in range(60):
        first.put(_key("data", i), "a" * 64, float(i))
    first.flush()

    second = digest_table.DigestTable(str(tmp_path))  # run 2: the same loop, file 5 was edited
    rehashed = []
    for i in range(60):
        key = _key("data", i) if i != 5 else _key("data-edited", 5)
        if second.get(key) is None:
            rehashed.append(i)
            second.put(key, "c" * 64, 2000.0 + i)
    assert rehashed == [5], f"run 2 read {len(rehashed)} files again for one edit"
