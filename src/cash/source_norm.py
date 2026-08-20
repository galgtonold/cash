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
import hashlib
import io
import re
import textwrap
import tokenize
import types

__all__ = ["bytecode_identity", "normalize_source_for_hash"]

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


# Consts that describe themselves exactly under ``repr`` -- no identity, no
# address, stable across processes and runs.
_PRIMITIVE_CONSTS = (bool, int, float, complex, str, bytes, type(None))

# Nested code nests: a comprehension inside a closure inside a method. The cap
# is a runaway guard, not a real limit -- eight levels is far past anything a
# human writes, and stopping early only makes the digest coarser, never wrong.
_MAX_CONST_DEPTH = 8


def _stabilize_const(const: object, depth: int) -> str:
    """Describe one const so the description never embeds an address."""
    if isinstance(const, _PRIMITIVE_CONSTS):
        return repr(const)
    if isinstance(const, types.CodeType):
        if depth >= _MAX_CONST_DEPTH:
            return "<code:depth>"
        return "code(" + _code_atoms(const, depth + 1) + ")"
    if isinstance(const, tuple):
        return "(" + ",".join(_stabilize_const(c, depth) for c in const) + ")"
    if isinstance(const, frozenset):
        return "{" + ",".join(sorted(_stabilize_const(c, depth) for c in const)) + "}"
    # Anything else (a rare exotic const) contributes its TYPE only: its repr
    # may carry an address, and a wrong-but-stable digest beats a right-but-
    # unstable one, which would miss forever.
    return f"<{type(const).__name__}>"


def _code_atoms(code: types.CodeType, depth: int = 0) -> str:
    """Serialize a code object's behaviour-bearing fields."""
    consts = ",".join(_stabilize_const(c, depth) for c in code.co_consts)
    return _SEP.join(
        (
            code.co_code.hex(),
            repr(code.co_names),
            repr(code.co_varnames),
            consts,
        )
    )


def bytecode_identity(fn: object) -> str | None:
    """Digest *fn*'s compiled body, or ``None`` when it has none.

    The identity of last resort, for callables whose source cannot be read
    (``exec``-defined, REPL, a source file that moved). Two requirements
    pull against each other and both are load-bearing:

    **It must see changes.** ``co_code`` ALONE does not. The operand of a
    const load is an INDEX into ``co_consts``, not the value, so on 3.14::

        return "alpha"  vs  return "omega"   -> co_code IDENTICAL
        return 100000   vs  return 200000    -> co_code IDENTICAL

    Measured, not assumed. Small ints happen to inline into the opcode and
    so do differ, which makes the blindness easy to miss when probing.

    **It must be stable across processes.** A nested code object's ``repr``
    embeds a memory address, so folding ``co_consts`` in via ``str()``
    yields a fresh digest every run and the cache never hits again. Nested
    code is therefore RECURSED into rather than repr'd -- and rather than
    dropped, which is the other way to stay safe but hides any edit made
    inside a nested function or lambda.
    """
    code = getattr(fn, "__code__", None)
    if code is None:
        # A callable instance: its behaviour lives in __call__.
        code = getattr(getattr(fn, "__call__", None), "__code__", None)
    if not isinstance(code, types.CodeType):
        return None
    try:
        return hashlib.sha256(_code_atoms(code).encode("utf-8")).hexdigest()
    except (AttributeError, TypeError, ValueError, RecursionError):
        return None
