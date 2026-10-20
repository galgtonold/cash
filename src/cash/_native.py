"""Loads cash's optional C extension, ``cash._cwatch``, or says why not.

The extension tells cash which values provably did not change since it last
hashed them, so it can skip hashing them again. It is an accelerator only:
without it cash hashes every value as it always does, and every key and
digest is the same either way.

It is missing or switched off when:

* this is the pure-Python wheel (``py3-none-any``), or cash was built from
  source without a working C compiler;
* Python is older than 3.12, or not CPython;
* ``CASH_PURE_PYTHON=1`` is set in the environment;
* the compiled module does not load here (an old system library, say), was
  built for another version of cash, or reports itself unusable on this
  interpreter (``USABLE`` false, for example a free-threaded build).

Import ``cwatch`` from here; never ``cash._cwatch`` directly. It is the
module, or None and then ``REASON`` says why, which ``%cash_status`` shows
(``status_line``).
"""

from __future__ import annotations

import importlib
import logging
import os
import platform
import sys
from types import ModuleType

__all__ = ["AVAILABLE", "EXPECTED_API", "REASON", "cwatch", "kernels", "status_line"]

logger = logging.getLogger(__name__)

#: The ``API_VERSION`` this cash's Python code was written against. A module
#: with another number is a leftover from another build and is not used.
EXPECTED_API = 3

MODULE = "cash._cwatch"


def _switched_off() -> bool:
    return os.environ.get("CASH_PURE_PYTHON", "").strip().lower() in ("1", "true", "yes", "on")


def _load() -> tuple[ModuleType | None, str | None]:
    """``(module, None)``, or ``(None, why the fast path is off)``."""
    if _switched_off():
        return None, "CASH_PURE_PYTHON is set"
    implementation = platform.python_implementation()
    if implementation != "CPython":
        return None, f"no C extension for {implementation}"
    if sys.version_info < (3, 12):
        return None, f"it needs Python 3.12 or newer, this is {sys.version_info.major}.{sys.version_info.minor}"
    try:
        module = importlib.import_module(MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != MODULE:
            return None, f"the C extension does not load: {exc}"
        return None, "this install has no C extension (the pure-Python wheel, or built without a C compiler)"
    except Exception as exc:  # noqa: BLE001 - whatever stops the module loading, cash runs without it
        return None, f"the C extension does not load: {type(exc).__name__}: {exc}"
    api = getattr(module, "API_VERSION", None)
    if api != EXPECTED_API:
        return None, f"the C extension has API {api}, this cash needs {EXPECTED_API}; reinstall cash-lib"
    if getattr(module, "USABLE", True) is False:
        return None, str(getattr(module, "UNUSABLE_REASON", "") or "the C extension is not usable here")
    return module, None


cwatch, REASON = _load()
#: True when ``cwatch`` is the loaded extension.
AVAILABLE = cwatch is not None
if REASON is not None:
    logger.debug("cash fast path off: %s", REASON)

#: The module when it also has the walk kernels (``cwalk.c``): C versions of
#: cash's hottest Python walks, each giving exactly the Python version's
#: answer. None without them: the Python walks run, as they always did.
kernels = cwatch if cwatch is not None and hasattr(cwatch, "leaf_items") else None


def status_line() -> str:
    """One line for ``%cash_status``: is the fast path on, and if not, why."""
    if AVAILABLE:
        return f"on (C extension {MODULE}, API {EXPECTED_API})"
    return f"off: {REASON}"
