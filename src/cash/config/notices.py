"""The one way configuration reports a setting it cannot use.

A config is resolved more than once -- the module default, then each
``Cash(...)`` -- so every notice is said once per process.
"""

from __future__ import annotations

import difflib
import logging
from typing import Any

from ..diagnostics import warn_diagnostic
from ..exceptions import CashCacheIneffectiveWarning

logger = logging.getLogger(__name__)

__all__ = ["config_notice", "did_you_mean"]


#: ``(code, message)`` pairs already said in this process: the one dedup for
#: every config notice. A config is resolved more than once -- the module
#: default, then each ``Cash(...)`` -- and must not repeat its complaints.
_CONFIG_NOTICES: set[tuple[str, str]] = set()


def config_notice(code: str, what: str, fix: str) -> None:
    """Warn, once per process, about a setting cash could not use."""
    if (code, what) in _CONFIG_NOTICES:
        return
    _CONFIG_NOTICES.add((code, what))
    try:
        warn_diagnostic(CashCacheIneffectiveWarning, code, what, fix)
    except Exception:  # a notice must never break a config load
        logger.debug("Could not emit %s", code, exc_info=True)


def did_you_mean(key: str, valid: Any) -> str:
    match = difflib.get_close_matches(key, sorted(valid), n=1, cutoff=0.6)
    return f" Did you mean `{match[0]}`?" if match else ""
