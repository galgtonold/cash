"""Byte sizes: reading ``"2GB"`` / ``"512MiB"`` and writing a count back.

Settings take sizes in these spellings, and every message that names a size
(caps, budgets, the CLI, the explorer) formats it here, so a size reads the
same everywhere.
"""

from __future__ import annotations

import re

__all__ = ["format_size", "human_bytes", "parse_size"]

_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]i?b|b)?\s*$", re.IGNORECASE)
_SIZE_UNITS = {
    "b": 1,
    "kb": 10**3,
    "mb": 10**6,
    "gb": 10**9,
    "tb": 10**12,  # SI
    "kib": 2**10,
    "mib": 2**20,
    "gib": 2**30,
    "tib": 2**40,  # binary
}


def parse_size(raw: str) -> int:
    """``"2GB"`` -> 2_000_000_000, ``"512MiB"`` -> 536_870_912, ``"1024"`` -> 1024.

    KB/MB/GB/TB are powers of 1000 and KiB/MiB/GiB/TiB powers of 1024, as the
    units say. Case-insensitive. Raises ``ValueError`` on anything else.
    """
    m = _SIZE_RE.match(raw)
    if m is None:
        raise ValueError(f"not a size: {raw!r} (write bytes, or e.g. '2GB' / '512MiB')")
    number, unit = m.group(1), (m.group(2) or "b").lower()
    return int(float(number) * _SIZE_UNITS[unit])


def human_bytes(n: int | None) -> str:
    """Format a byte count with a sensible unit (e.g. ``7.8 KiB``, ``8.0 GiB``).

    Used in the cap warnings so a message never reads ``~0 MiB`` for a small
    cap or an unwieldy raw byte count for a large one.
    """
    size = float(n or 0)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"  # unreachable; keeps type-checkers happy


def format_size(n: int) -> str:
    """*n* bytes the way `parse_size` reads it back: ``2000000000`` -> ``"2 GB"``.

    A size written ``"2GB"`` must not come back as ``1.9 GB``, which reads as
    mis-parsed: a whole number of a decimal or a binary unit is shown in that
    unit; anything else in binary units, labelled as such.
    """
    for name in ("TB", "TiB", "GB", "GiB", "MB", "MiB", "KB", "KiB"):
        unit = _SIZE_UNITS[name.lower()]
        if n >= unit and (n * 10) % unit == 0:
            return f"{n / unit:g} {name}"
    return human_bytes(n)
