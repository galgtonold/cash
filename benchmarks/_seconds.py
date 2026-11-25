"""How a duration is written on the docs pages. Standard library only:
``scripts/doc_numbers.py`` reads it to fill the per-call cost figures."""

from __future__ import annotations


def format_seconds(s: float) -> str:
    """The unit the page uses: µs below a millisecond, ms from there."""
    if s < 1e-3:
        us = s * 1e6
        return f"{us:.2f} µs" if us < 1 else f"{us:.0f} µs" if us >= 10 else f"{us:.1f} µs"
    ms = s * 1e3
    return f"{ms:.0f} ms" if ms >= 10 else f"{ms:.1f} ms"
