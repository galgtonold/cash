"""``frozen=True``: results keyed by the call that produced them, audited now
and then for changes."""

from __future__ import annotations

#: A frozen object is re-hashed at its 8th use as an argument and every 64th
#: after that (every use under CASH_DEBUG), and compared with the first audit.
FROZEN_AUDIT_FIRST = 8
FROZEN_AUDIT_EVERY = 64
