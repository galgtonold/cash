"""When a file's stamps prove that any later write will move them.

A filesystem stamps a write with a clock that moves in ticks, so a second
write inside the tick of the first can leave mtime and ctime as they were.
Once a whole tick has passed since the last change, it cannot. The saved
notebook's read (`server_discovery`) and the VS Code backup scan
(`vscode_backup`) both rest on this rule: the first to know no write
overlapped a read, the second to keep what it read while the stamps hold.
"""

from __future__ import annotations

import os

#: The coarsest tick a filesystem with fine timestamps moves them in: the
#: 15.6 ms of the Windows clock (Linux: at most 10 ms; APFS: 1 ns). A write
#: at least this long after the last one gets other stamps.
STAMP_TICK_NS = 16_000_000
#: A change stamp on a whole multiple of this comes from a filesystem whose
#: tick is unknown here (whole seconds on HFS+ and ext3, two on FAT); a fine
#: stamp lands on one by chance once in a hundred thousand writes or less.
COARSE_STAMP_NS = 10_000_000
#: How old such a coarse stamp must be before a write is sure to move it.
COARSE_TICK_NS = 2_000_000_000


def changed_ns(st: os.stat_result) -> int:
    """When the file last changed, as its stamps tell it."""
    return max(st.st_mtime_ns, st.st_ctime_ns)


def a_write_now_moves_the_stamps(changed: int, now_ns: int) -> bool:
    """Whether every write from *now_ns* on gives a file last changed at
    *changed* (`changed_ns`) other stamps. Take *now_ns* before the stat
    *changed* comes from, so a write between the two is a write after it."""
    tick = STAMP_TICK_NS if changed % COARSE_STAMP_NS else COARSE_TICK_NS
    return now_ns - changed >= tick
