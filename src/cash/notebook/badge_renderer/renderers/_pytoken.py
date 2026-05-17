"""Tiny Python tokenizer for code-line syntax highlighting in badges.

Ported directly from the v3 design kit's regex — same token classes,
same keyword set, same precedence. Renderers wrap each token in a
``<span class="c3-{kw|str|com|num|id|txt}">`` for the CSS-only theme to
colorize.

We deliberately don't use Python's ``tokenize`` stdlib module: badge
inputs are *single lines* of arbitrary user code (sometimes with %-magics,
sometimes broken syntax), and a heuristic regex degrades gracefully where
``tokenize`` would crash.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(
    r"(#[^\n]*)"
    r"|(\"[^\"]*\"|'[^']*')"
    r"|(\b(?:import|from|as|def|return|if|else|elif|for|while|in|with|"
    r"try|except|None|True|False|lambda|yield|raise|class|pass|break|"
    r"continue|and|or|not|is)\b)"
    r"|(\b\d+(?:\.\d+)?\b)"
    r"|(%%?[a-z_]+)"
    r"|(\b[A-Za-z_]\w*)"
)


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def highlight_python(src: str) -> str:
    """Return HTML with each Python token wrapped in a ``c3-`` class span.

    Unmatched runs are emitted as plain (escaped) text.
    """
    parts: list[str] = []
    last = 0
    for m in _TOKEN_RE.finditer(src):
        if m.start() > last:
            parts.append(_esc(src[last:m.start()]))
        if m.group(1):
            parts.append(f'<span class="c3-com">{_esc(m.group(1))}</span>')
        elif m.group(2):
            parts.append(f'<span class="c3-str">{_esc(m.group(2))}</span>')
        elif m.group(3):
            parts.append(f'<span class="c3-kw">{_esc(m.group(3))}</span>')
        elif m.group(4):
            parts.append(f'<span class="c3-num">{_esc(m.group(4))}</span>')
        elif m.group(5):
            parts.append(f'<span class="c3-kw">{_esc(m.group(5))}</span>')
        else:  # identifier
            parts.append(_esc(m.group(6)))
        last = m.end()
    if last < len(src):
        parts.append(_esc(src[last:]))
    return "".join(parts)
