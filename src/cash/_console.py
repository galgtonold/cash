"""Console output that survives a stream which cannot encode emoji.

On Windows a plain ``python.exe`` writes through cp1252, which cannot encode
the status symbols cash prints (``✅``, ``⚙️``); `safe_text` swaps them for
ASCII there. Jupyter kernels and most other terminals are UTF-8, where it is
a pass-through.
"""

from __future__ import annotations

import sys

__all__ = ["encodable", "safe_text", "stdout_supports_unicode", "survive_narrow_streams"]

_ASCII_FALLBACKS: dict[str, str] = {
    # Status / outcome
    "✅": "[OK]",
    "❌": "[X]",
    "⚠️": "[!]",
    "⚠": "[!]",
    "✓": "[v]",
    "❓": "[?]",
    "🚫": "[no-cache]",
    # Cache lifecycle
    "⚡": "[cached]",
    "⚙️": "[run]",
    "⏩": "[skip]",
    "⏳": "[wait]",
    "🔄": "[refresh]",
    "♻️": "[reuse]",
    "⬆️": "[upstream]",
    "🔁": "[loop]",
    # Decorations / arrows
    "→": "->",
    "←": "<-",
    "↑": "^",
    "↓": "v",
    "↔": "<->",
    "↻": "~>",
    "└": "L",
    "─": "-",
    "│": "|",
    "├": "+",
    "…": "...",
    "∈": "in",
    # Dashboards / debug
    "🔧": "[computed]",
    "📦": "[restored]",
    "📋": "[provenance]",
    "📊": "[stats]",
    "📈": "[trend-up]",
    "📉": "[trend-down]",
    "📁": "[dir]",
    "🐛": "[bug]",
    "🏷️": "[tag]",
    "⏭️": "[next]",
    "🎯": "[target]",
}


def stdout_supports_unicode(stream: object | None = None) -> bool:
    """Return ``True`` if *stream* (default: ``sys.stdout``) can encode emojis.

    Cheap and side-effect free.  Used by :func:`safe_text` to decide whether
    to pass the input through unchanged or downgrade it to ASCII fallbacks.
    """
    if stream is None:
        stream = sys.stdout
    encoding = getattr(stream, "encoding", None) or "ascii"
    encoding_lc = encoding.lower()
    if encoding_lc.startswith("utf"):
        return True
    try:
        # ✅ and ⚙️ together cover both single-codepoint emoji and the
        # variation-selector form most likely to break under cp1252 / latin-1.
        "✅⚙️".encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def safe_text(s: str, *, stream: object | None = None) -> str:
    """Return *s* with characters un-encodable by *stream* replaced by ASCII.

    Pass-through when the stream can encode everything (the common case in
    Jupyter / on Linux / when ``PYTHONIOENCODING=utf-8``).  Otherwise replace
    each unsupported character with an entry from :data:`_ASCII_FALLBACKS`
    or, lacking a mapping, drop it.

    The function preserves all ASCII characters as-is, so log lines stay
    readable even on legacy Windows consoles.
    """
    if not s:
        return s
    if stream is None:
        stream = sys.stdout
    if stdout_supports_unicode(stream):
        return s
    encoding = getattr(stream, "encoding", None) or "ascii"
    try:
        s.encode(encoding)
        return s  # nothing to downgrade
    except UnicodeEncodeError:
        pass
    out: list[str] = []
    for ch in s:
        try:
            ch.encode(encoding)
        except UnicodeEncodeError:
            out.append(_ASCII_FALLBACKS.get(ch, ""))
        else:
            out.append(ch)
    return "".join(out)


def encodable(s: str, *, stream: object | None = None) -> str:
    """*s* as *stream* (default ``sys.stdout``) can write it: characters its
    encoding lacks become ``\\uXXXX`` escapes.

    For text that names things -- a function called ``数据``, a cache
    directory with an emoji in it -- where `safe_text`'s dropping would lose
    the name. A cp1252 pipe or file on Windows raised ``UnicodeEncodeError``
    out of ``print`` for them.
    """
    if stream is None:
        stream = sys.stdout
    encoding = getattr(stream, "encoding", None) or "ascii"
    try:
        s.encode(encoding)
        return s
    except UnicodeEncodeError:
        return s.encode(encoding, "backslashreplace").decode(encoding)
    except LookupError:
        return s


def survive_narrow_streams() -> None:
    """Make ``sys.stdout`` and ``sys.stderr`` escape what their encoding
    cannot write, instead of raising. For the ``cash`` command, which owns
    its streams: every name and path it prints can hold any character."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="backslashreplace")
        except (ValueError, OSError):  # a detached or closed stream
            pass
