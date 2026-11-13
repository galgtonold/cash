"""Windows diagnostic: is a folder's stat from its parent's listing usable?

Confirms the cause of the CI failures of
tests/test_notebook/internals/test_a_folder_in_a_kept_listing_is_not_looked_at_again_each_cell.py
on windows-latest, fixed by leaving folders out of `listing_stats`
(cash/tracking/file_dep_snapshot.py, `_is_folder`).

Run from a checkout, with cash importable (stdlib + cash only):

    set PYTHONPATH=src
    python listed_folder_stat.py

Expected output on NTFS (what the fix assumes):

    [1] listing entry of a folder: st_ino=0 st_dev=0 st_nlink=0
    [2] os.stat of a folder: st_nlink=1 (the listing's link check takes it as one link)
    [3] after a write into the folder, its parent's listing mtime is stale: True
    [4] after os.stat opened the folder, the parent's listing is up to date: True (or False)
    [5] cash.listing_stats gives a stat for the folder: False   <- with the fix
    [6] watch keeps the folder's stat after one pass: True
    [7] lstats of the folders in the next 3 passes: [0, 0, 0]
    [8] a write into a folder is seen in its stat at the next pass: True

[1]-[3] True is the bug's mechanism: a listed folder stat has no file
identity (so `FileWatch.record` never keeps it and every cell lstat'ed
the folder again: the [80, 160, 160] lstats) and may hold the folder's
old times (the stale st_mtime_ns of test_a_folder_changed_since_...).
[5] True means the fix is not in this build. [7] not all 0 or [8] False
means a remaining problem: paste the whole output.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

from cash.notebook import run_memo
from cash.notebook.run_memo import forget_file_state_this_run, stats_this_run
from cash.tracking import file_dep_snapshot, file_watch


def listed(parent: str, name: str) -> os.stat_result:
    with os.scandir(parent) as it:
        for e in it:
            if e.name == name:
                return e.stat()
    raise FileNotFoundError(name)


def main() -> None:
    print(sys.version, sys.platform)
    root = tempfile.mkdtemp(prefix="cash-diag-")
    people = os.path.join(root, "people")
    folders = []
    for i in range(40):
        folder = os.path.join(people, f"p{i:02d}")
        os.makedirs(folder)
        for name in ("a.jpg", "b.jpg"):
            with open(os.path.join(folder, name), "wb") as f:
                f.write(b"x" * (i + 1))
        folders.append(folder)

    st = listed(people, "p07")
    print(f"[1] listing entry of a folder: st_ino={st.st_ino} st_dev={st.st_dev} st_nlink={st.st_nlink}")
    print(f"[2] os.stat of a folder: st_nlink={os.stat(folders[7]).st_nlink}")

    time.sleep(0.05)
    with open(os.path.join(folders[7], "c.jpg"), "wb") as f:
        f.write(b"new")
    after_write = listed(people, "p07").st_mtime_ns  # the parent's entry, before the folder is opened
    real = os.stat(folders[7]).st_mtime_ns  # opens the folder
    print(f"[3] after a write into the folder, its parent's listing mtime is stale: {after_write != real}")
    print(f"[4] after os.stat opened the folder, the parent's listing is up to date: {listed(people, 'p07').st_mtime_ns == real}")

    paths = []
    os.chdir(root)
    for folder in folders:
        rel = os.path.relpath(folder, root).replace("\\", "/")
        paths += [rel, folder, f"{rel}/a.jpg", os.path.join(folder, "a.jpg"), f"{rel}/b.jpg", os.path.join(folder, "b.jpg")]
    found = file_dep_snapshot.listing_stats([p for p in paths if os.path.isabs(p) and os.path.isdir(p)])
    print(f"[5] cash.listing_stats gives a stat for the folder: {bool(found)}")

    os.environ.pop("CASH_FILE_WATCH", None)
    forget_file_state_this_run()
    file_watch.reset()
    watch = file_watch.watcher()
    if watch is None:
        print("no folder watch here: stop")
        return
    run_memo.stats_from_listings = file_dep_snapshot.listing_stats
    taken: list[str] = []
    real_lstat = os.lstat

    def lstat(path, *a, **k):
        taken.append(os.fspath(path))
        return real_lstat(path, *a, **k)

    os.lstat = lstat
    file_dep_snapshot.begin_file_state_epoch()
    stats_this_run(paths)
    print(f"[6] watch keeps the folder's stat after one pass: {watch.current(folders[7]) is not None}")
    later = []
    for _ in range(3):
        file_dep_snapshot.end_file_state_epoch()
        file_dep_snapshot.begin_file_state_epoch()
        taken.clear()
        stats_this_run(paths)
        later.append(len(taken))
    print(f"[7] lstats of the folders in the next 3 passes: {later}")
    if any(later):
        print("    report:", watch.report())
        print("    first lstats:", taken[:6])

    with open(os.path.join(folders[3], "d.jpg"), "wb") as f:
        f.write(b"new")
    file_dep_snapshot.end_file_state_epoch()
    file_dep_snapshot.begin_file_state_epoch()
    found = stats_this_run(paths)
    os.lstat = real_lstat
    print(f"[8] a write into a folder is seen in its stat at the next pass: {found[folders[3]][1].st_mtime_ns == os.stat(folders[3]).st_mtime_ns}")
    file_dep_snapshot.end_file_state_epoch()


if __name__ == "__main__":
    main()
