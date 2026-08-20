from __future__ import annotations

"""Canonical source form for code-identity hashing.

Every code-identity channel in cash used to hash raw source text, so an
added comment, a stray blank line, or a ``black`` run invalidated cache
entries whose compiled behaviour had not changed at all. A repo-wide
reformat threw away every entry in the cache.

This module reduces source to a token stream, which drops comments,
blank lines, trailing whitespace and the exact indentation width while
keeping everything that can change behaviour.

Deliberately KEPT in the digest:

* ``# @cash:`` annotations. They are directives, not prose -- ``no-cache``,
  ``ttl``, ``persist`` and friends change how a statement is cached, so
  editing one must move the key. They survive as a normalized
  ``@cash:<directive>[=<value>]`` atom, so re-spacing a directive is free
  but re-targeting it is not.
* Docstrings. At the token level a docstring is an ordinary string
  constant, indistinguishable from a returned literal, and keeping it
  costs nothing.
* Block structure. INDENT/DEDENT enter the stream as width-independent
  markers, so a 4-space to 2-space reformat is free while dedenting a
  statement out of an ``if`` body -- a real behaviour change -- is not.
"""

import functools
import io
import re
import textwrap
import tokenize

__all__ = ["normalize_source_for_hash"]

# Populated on first use from ``cash.notebook.annotations``. Imported
# lazily because this module sits below the notebook package in the
# import graph, and hashing only runs long after imports have settled.
_ANNOTATION_PATTERN: re.Pattern[str] | None = None

# Structural markers. Chosen outside the range a Python token can carry
# so they cannot collide with real source text.
_INDENT = "\x02"
_DEDENT = "\x03"
_SEP = "\x01"


def _annotation_pattern() -> re.Pattern[str]:
    global _ANNOTATION_PATTERN
    if _ANNOTATION_PATTERN is None:
        from .notebook.annotations import ANNOTATION_PATTERN

        _ANNOTATION_PATTERN = ANNOTATION_PATTERN
    return _ANNOTATION_PATTERN


def _annotation_atom(comment: str) -> str | None:
    """Return a normalized atom for a ``# @cash:`` comment, else ``None``.

    Matches the regex directly rather than calling ``parse_annotation_line``:
    that parser *warns* on a malformed value (``ttl=5m``), and hashing runs
    on every cache lookup, so routing through it would emit a warning per
    call. Here an unparseable value is simply kept verbatim -- it still has
    to move the digest, because editing it is an edit the user meant.
    """
    match = _annotation_pattern().search(comment)
    if match is None:
        return None
    directive = match.group(1).lower()
    value = match.group(2)
    return f"@cash:{directive}" + (f"={value}" if value else "")


@functools.lru_cache(maxsize=2048)
def normalize_source_for_hash(source: str) -> str:
    """Return a canonical form of *source* for hashing.

    Falls back to the raw text when *source* does not tokenize -- a
    fragment, a syntax error mid-edit, an unterminated string. A coarse
    but never-wrong digest beats raising from inside a hasher, and the
    raw text still distinguishes different broken sources from each other.

    MEMOIZED, and load-bearingly so: tokenizing costs ~47us against the
    ~0.5us of hashing raw text, and this runs once per transitive helper
    per cached call -- a loop cached per statement pays it thousands of
    times, which showed up as a measurable CPU-overhead regression in
    ``test_cfd_loop_overhead`` before the cache went in. Keying on the
    source text is safe because the transform is pure: identical text
    always normalizes identically, and edited text is a different key.
    """
    try:
        readline = io.StringIO(textwrap.dedent(source)).readline
        tokens = list(tokenize.generate_tokens(readline))
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return source

    atoms: list[str] = []
    for tok in tokens:
        kind = tok.type
        if kind == tokenize.COMMENT:
            atom = _annotation_atom(tok.string)
            if atom is not None:
                atoms.append(atom)
        elif kind == tokenize.INDENT:
            atoms.append(_INDENT)
        elif kind == tokenize.DEDENT:
            atoms.append(_DEDENT)
        elif kind == tokenize.NEWLINE:
            atoms.append("\n")
        elif kind in (tokenize.NL, tokenize.ENDMARKER):
            # NL is a non-logical newline: a blank line, or a line break
            # inside brackets. Neither changes behaviour. (ENCODING never
            # appears here -- only ``tokenize.tokenize`` emits it.)
            continue
        else:
            atoms.append(tok.string)

    return _SEP.join(atoms)
