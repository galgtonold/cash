"""Minify the badge stylesheet.

The badge inlines its own CSS so a saved notebook renders standalone, which
means the stylesheet is duplicated into every cell's output: measured at
354.9 KB across a twelve-cell notebook, 67% of the file. Minifying is the part
of that we can take back without cash having to know what the document already
contains.

String-aware on purpose. A blind ``re.sub(r"\\s*([{};:,>])\\s*", ...)`` also
rewrites the inside of ``content: "a ; b"`` and of quoted attribute values,
which is a silent corruption rather than a crash.
"""
from __future__ import annotations

import re


def minify_css(css: str) -> str:
    """Strip comments and redundant whitespace, leaving quoted text alone."""
    segments: list[tuple[str, str]] = []
    buf: list[str] = []
    i, n = 0, len(css)

    while i < n:
        ch = css[i]
        if ch in ('"', "'"):
            j = i + 1
            while j < n and css[j] != ch:
                j += 2 if css[j] == "\\" else 1
            segments.append(("code", "".join(buf)))
            buf = []
            segments.append(("str", css[i:j + 1]))
            i = j + 1
            continue
        if css.startswith("/*", i):
            end = css.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        buf.append(ch)
        i += 1
    segments.append(("code", "".join(buf)))

    out: list[str] = []
    for kind, text in segments:
        if kind == "str":
            out.append(text)
            continue
        text = re.sub(r"\s+", " ", text)
        out.append(re.sub(r"\s*([{};:,>])\s*", r"\1", text))
    return "".join(out).strip()
