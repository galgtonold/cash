"""What the decorator tells the user: coded warnings once per function, and
the per-call log and debug lines."""

from __future__ import annotations

import logging

#: One line per decorated call -- hit or miss, and why -- when `debug=True` /
#: `CASH_DEBUG=1` or `verbose=True` asks for it.
calls_logger = logging.getLogger("cash.calls")
