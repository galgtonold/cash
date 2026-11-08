"""When a digest taken earlier may still stand for a value (`unsettle`).

`value_hash.one_read` hands a digest taken in one window to a later window
only while the epoch here is the one it was taken in. Whatever may change
a value the notebook holds -- the user's code running, a restore writing
into the namespace or into objects, a random stream put back -- calls
`unsettle` first, and every digest taken before it is read again.
"""

from __future__ import annotations

#: Moves on at every `unsettle`; a kept digest names the one it was taken in.
EPOCH = [0]


def unsettle() -> None:
    """A value the notebook holds may change from here: no digest taken
    before is handed on (`value_hash.one_read`)."""
    EPOCH[0] += 1
