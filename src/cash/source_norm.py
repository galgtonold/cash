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
* Block structure. INDENT/DEDENT enter the stream as width-independent
  markers, so a 4-space to 2-space reformat is free while dedenting a
  statement out of an ``if`` body -- a real behaviour change -- is not.

Deliberately COLLAPSED, beyond whitespace and comments:

* Numeric literals, to one spelling per value -- see `_canonical_number`.
  ``0.5`` and ``0.50`` compile to the same constant, so they are the same
  function. Type is kept, so ``1`` and ``1.0`` stay apart.
* Docstrings of functions and classes -- see `strip_docstrings`. They are
  documentation, like comments: rewording one re-ran every cached result
  built on the function.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import importlib.util
import inspect
import io
import os
import re
import sys
import textwrap
import tokenize
import types

from .analysis.annotations import ANNOTATION_PATTERN
from .exceptions import SOURCE_RETRIEVAL_ERRORS
from .value_types import IMMUTABLE_PRIMS

__all__ = [
    "bytecode_identity",
    "callable_identity",
    "compiled_identity",
    "module_identity",
    "opaque_identity",
    "own_source",
    "own_source_digest",
    "source_digest",
    "code_consts_without_docstring",
    "drop_docstrings",
    "normalize_source_for_hash",
    "source_identity_digest",
    "stat_has_settled",
    "strip_cache_decorator",
    "strip_docstrings",
    "unparse_without_docstrings",
]

# Populated on first use from ``cash.analysis.annotations``. Imported
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
        _ANNOTATION_PATTERN = ANNOTATION_PATTERN
    return _ANNOTATION_PATTERN


def _canonical_number(text: str) -> str:
    """Reduce a numeric literal to one spelling per value.

    ``0.5`` and ``0.50`` are the same double -- same bits, same entry in
    ``co_consts`` -- so a function that swaps one for the other is byte-for-byte
    the same function once compiled. Hashing the literal's TEXT made that swap
    throw the cache away, which is the numeric twin of hashing the decorator's
    arguments. Reported by a user who re-typed ``0.5`` as ``0.50``, watched a
    long computation re-run, and was told it was floating-point imprecision; it
    was not, and the two compare exactly equal.

    Covers the same value written as ``0.50`` / ``.5`` / ``5e-1``, and the
    readability spellings ``1_000`` / ``0x3e8`` / ``0o1750`` / ``0b1111101000``.

    TYPE is preserved, which is the line that matters: ``repr`` distinguishes
    ``1`` from ``1.0`` and ``1`` from ``1j``, and those really are different
    values that must keep different keys.

    Parses directly instead of going through ``ast.literal_eval``: this runs
    per numeric token on every cold normalize, and building an AST per literal
    is far more than the arithmetic costs. The ``0x``/``0b``/``0o`` test comes
    before the float test on purpose -- ``0x1e3`` contains an ``e`` and is 483,
    not a float.

    Falls back to the raw text on anything awkward, notably an integer past
    ``sys.set_int_max_str_digits`` whose ``repr`` refuses to render. A coarse
    digest is always safe here; a wrong one is not.
    """
    lowered = text.lower()
    try:
        if lowered.endswith("j"):
            return repr(complex(text))
        if lowered.startswith(("0x", "0b", "0o")):
            return repr(int(text, 0))
        if "." in lowered or "e" in lowered:
            return repr(float(text))
        return repr(int(text))
    except (ValueError, OverflowError, MemoryError):
        return text


#: Directives that only silence a purity warning. The function computes the
#: same thing with or without one, so it is prose to the key: adding
#: ``# @cash:assume-safe`` to a line of a cached function recomputed it, and
#: removing it again recomputed it once more.
_WAIVERS = frozenset({"assume-safe", "assumesafe"})


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
    if directive in _WAIVERS:
        return None
    value = match.group(2)
    return f"@cash:{directive}" + (f"={value}" if value else "")


_DOCSTRING_OWNERS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_DEF_OR_CLASS = re.compile(r"\b(?:def|class)\b")
_LINE = re.compile(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+\Z")


def _docstring_owners(tree: ast.AST, module: bool) -> list[ast.AST]:
    """Every function and class in *tree* whose body opens with a docstring.

    The same test as ``ast.get_docstring``: the first statement of the body,
    a bare ``str`` constant. A bytes literal or an f-string in that spot is
    not a docstring and stays.
    """
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, _DOCSTRING_OWNERS) or (module and isinstance(node, ast.Module))):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            found.append(node)
    return found


def drop_docstrings(tree: ast.AST, module: bool = False) -> bool:
    """Remove the docstrings from *tree* in place; True when there were any.

    The AST twin of `strip_docstrings`, for digests built from
    ``ast.unparse``. A body left empty unparses as a bare header, which is
    never compiled, only hashed.
    """
    owners = _docstring_owners(tree, module)
    for owner in owners:
        owner.body.pop(0)
    return bool(owners)


@functools.lru_cache(maxsize=4096)
def unparse_without_docstrings(code: str) -> str:
    """*code*, in ``ast.unparse`` form, with function and class docstrings gone.

    For statement text, which is always ``ast.unparse`` output and so has no
    comments left to lose. Returns *code* itself when it has no docstring or
    does not parse, which keeps every key that never had a docstring in it
    byte-identical. Re-rendering is what makes adding or removing a
    docstring free as well: the result is exactly the text the statement
    would have had without one.

    A module-level docstring is not touched. A notebook statement that is a
    bare string is the cell's displayed value, not documentation.
    """
    if not _may_have_docstring(code):
        return code
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return code
    if not drop_docstrings(tree):
        return code
    try:
        return ast.unparse(tree)
    except (ValueError, RecursionError):
        return code


def _may_have_docstring(source: str) -> bool:
    """Cheap reject before paying for a parse: this runs per lookup."""
    if '"' not in source and "'" not in source:
        return False
    return _DEF_OR_CLASS.search(source) is not None


def _char_col(line: str, byte_col: int) -> int:
    """``ast`` columns count UTF-8 bytes; string slicing counts characters."""
    return len(line.encode("utf-8")[:byte_col].decode("utf-8", errors="ignore"))


@functools.lru_cache(maxsize=2048)
def strip_docstrings(source: str) -> str:
    """Return *source* with the docstrings of its functions and classes cut out.

    A docstring is documentation, the same as a comment, and comments were
    already free. Rewording one re-ran every cached result built on the
    function, which is the opposite of what anyone editing prose expects.

    Adding or removing a docstring is free too, not only rewording one: a
    docstring on lines of its own takes those whole lines with it, so the
    function reads exactly as if it never had one.

    What this gives up: a program that reads ``f.__doc__`` at run time --
    ``docopt``, a CLI built from docstrings -- is not re-run when the text
    changes. That is the same trade ``python -OO`` makes.

    Kept: *source*'s own leading string, which is a module docstring only
    when *source* is a whole module -- and then `drop_docstrings` handles it.
    Kept too: a ``# @cash:`` directive sharing a docstring's line, since this
    cuts text rather than re-rendering it.

    Returns *source* unchanged when it has no docstring or does not parse (a
    fragment, a mid-edit syntax error). Coarse beats raising from inside a
    hasher, and leaving docstring-free source byte-identical keeps every
    existing cache key that never had a docstring in it.
    """
    if not _may_have_docstring(source):
        return source
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return source
    nodes = [owner.body[0] for owner in _docstring_owners(tree, module=False)]
    if not nodes:
        return source

    # Not ``splitlines``: it also breaks on form feeds and other separators
    # that can sit inside a string, which ``ast`` does not count as lines.
    lines = _LINE.findall(source)
    # Bottom-up, so an earlier cut never shifts a later one's coordinates.
    for node in sorted(nodes, key=lambda n: (n.lineno, n.col_offset), reverse=True):
        first, last = node.lineno - 1, (node.end_lineno or node.lineno) - 1
        if last >= len(lines):
            return source
        start = _char_col(lines[first], node.col_offset)
        end = _char_col(lines[last], node.end_col_offset or 0)
        before = lines[first][:start]
        after = lines[last][end:]
        # ``"doc"; x = 1`` -- the separator belongs to the docstring.
        if after.lstrip().startswith(";"):
            after = after.lstrip()[1:]
        if not after.strip():
            if before.strip():
                replacement = [before.rstrip() + "\n"]  # def f(): "doc"
            else:
                replacement = []  # a line of its own
        elif before.strip():
            replacement = [before.rstrip() + " " + after.lstrip()]
        else:
            replacement = [before + after.lstrip()]
        lines[first : last + 1] = replacement
    return "".join(lines)


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
        readline = io.StringIO(strip_docstrings(textwrap.dedent(source))).readline
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
        elif kind == tokenize.NUMBER:
            atoms.append(_canonical_number(tok.string))
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


# Every keyword ``Cash.cache`` accepts. A decorator call only counts as
# cash's own when every keyword it passes appears here, which is what keeps
# an unrelated third-party ``@something.cache(expire_after=60)`` out of the
# rule below. ``tests/test_core/test_decorator_arg_identity.py`` pins this
# against the real signature; a parameter added there and forgotten here
# merely reverts that parameter to the old over-invalidating behaviour,
# which is the safe direction to fail in.
_CACHE_DECORATOR_PARAMS = frozenset(
    {
        "depends_on",
        "dynamic_depends_on",
        "file_depends_on",
        "ttl",
        "cache_if",
        "chunk_max_items",
        "chunk_max_bytes",
        "strict",
        "assume_safe",
        "allow_random",
        "frozen",
    }
)
_CACHE_DECORATOR_NAME = "cache"


def _is_cache_decorator(node: ast.expr) -> bool:
    """True when *node* is cash's own ``@....cache`` decorator expression.

    Matches the spellings users actually write -- ``@c.cache``,
    ``@cash.cache(ttl=60)``, ``@get_cash().cache`` -- by looking at the
    trailing attribute rather than trying to resolve the receiver, which is
    a runtime value and not knowable from source.

    A call with positional arguments, or with a keyword cash does not
    accept, is NOT ours: ``Cash.cache`` is keyword-only past ``func``, so
    anything else is some other decorator that merely happens to be spelled
    ``.cache``. ``**kwargs`` likewise disqualifies, since its contents are
    invisible here.
    """
    target = node
    if isinstance(node, ast.Call):
        if node.args:
            return False
        if any(kw.arg is None or kw.arg not in _CACHE_DECORATOR_PARAMS for kw in node.keywords):
            return False
        target = node.func
    if isinstance(target, ast.Attribute):
        return target.attr == _CACHE_DECORATOR_NAME
    return isinstance(target, ast.Name) and target.id == _CACHE_DECORATOR_NAME


@functools.lru_cache(maxsize=2048)
def strip_cache_decorator(source: str) -> str:
    """Return *source* with cash's own ``@....cache`` decorator removed.

    ``inspect.getsource`` hands back the decorator lines along with the
    function, so without this the decorator's own arguments land in the
    function's identity digest and every cache entry keyed on it dies when
    they change. Nothing about that was designed: the purity analyzer
    already drops ``decorator_list`` from the same source before analyzing
    it, and the identity hash simply never got the same treatment.

    It made cash's own advice self-defeating. ``CashImpurityWarning`` says
    to add ``assume_safe=True`` after auditing -- and doing so recomputed
    everything, on exactly the expensive functions the warning fires for.
    An added ``()`` did it too, which is how you can tell this was never a
    decision about semantics.

    The arguments that MUST still move the key are not lost, because none
    of them travel through the decorator's text:

    * ``depends_on`` / ``dynamic_depends_on`` / ``file_depends_on`` become
      dependency-graph edges, and the state hash folds each edge's own
      source hash (or a file's content) in.
    * ``ttl`` is enforced against entry metadata at read time.
    * ``cache_if`` and the ``chunk_max_*`` pair decide whether and how a
      value is *stored*; an entry already on disk is equally valid either
      way.
    * ``strict`` / ``assume_safe`` / ``allow_random`` only ever choose
      between a warning, an exception, and silence.

    Decorators that are not cash's are kept verbatim: ``@inject(db=prod)``
    can absolutely change what the function returns, and there is no way to
    tell from here that it does not.

    Falls back to the input whenever the source will not parse -- a
    fragment, a mid-edit syntax error. Coarse beats raising from inside a
    hasher.
    """
    # Cheap reject before paying for a parse: an undecorated helper is the
    # overwhelmingly common case, and this runs per lookup.
    dedented = textwrap.dedent(source)
    if not dedented.lstrip().startswith("@"):
        return source

    try:
        tree = ast.parse(dedented)
    except (SyntaxError, ValueError, RecursionError):
        return source
    if not tree.body:
        return source
    node = tree.body[0]
    # Functions only. A class's decorators can change the class itself --
    # ``@dataclass(frozen=True)`` is not cosmetic -- and cash has no reason
    # to strip them.
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return source

    drop: set[int] = set()
    for decorator in node.decorator_list:
        if _is_cache_decorator(decorator):
            end = decorator.end_lineno or decorator.lineno
            # The ``@`` shares its line with the expression that follows it,
            # so the expression's own span covers the whole decorator.
            drop.update(range(decorator.lineno, end + 1))
    if not drop:
        return source

    lines = dedented.splitlines(keepends=True)
    return "".join(line for i, line in enumerate(lines, 1) if i not in drop)


def source_identity_digest(source: str) -> str:
    """Digest *source* as a callable's cache-key identity.

    The single spelling of "what makes this callable the same callable" for
    every channel that keys on source text: the decorated function's own
    registration hash, the live per-call hash of each transitive helper, the
    helper hashes snapshotted in a purity report, and the function hashes
    that reach a notebook statement's key. They were four copies of
    ``sha256(normalize_source_for_hash(src))``; they are one function now so
    that a rule like `strip_cache_decorator` cannot be applied to three of
    them.
    """
    normalized = normalize_source_for_hash(strip_cache_decorator(source))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# Nested code nests: a comprehension inside a closure inside a method. The cap
# is a runaway guard, not a real limit -- eight levels is far past anything a
# human writes, and stopping early only makes the digest coarser, never wrong.
_MAX_CONST_DEPTH = 8


_CO_OPTIMIZED = 0x0001
# Python 3.14 flags a code object whose ``co_consts[0]`` is its docstring.
_CO_HAS_DOCSTRING = 0x4000000


def code_consts_without_docstring(code: types.CodeType) -> tuple:
    """``code.co_consts`` with the docstring, if any, replaced by ``None``.

    Compiled code keeps a function's docstring as its first constant, so
    without this the bytecode identities -- the fallback when source cannot be
    read -- moved on a docstring edit while the source identity did not.

    Up to 3.13 every function reserves that slot, holding ``None`` when there
    is no docstring, so replacing it makes a docstring free to add, remove or
    reword. Lambdas and comprehensions (names in ``<...>``) never have one,
    and neither do class bodies or modules, which are not ``CO_OPTIMIZED``.

    3.14 only reserves the slot when there IS a docstring, and shifts every
    other constant's index to fit, which ``co_code`` shows. There rewording a
    docstring is free and adding or removing one is not.
    """
    consts = code.co_consts
    if not consts or not isinstance(consts[0], str):
        return consts
    if sys.version_info >= (3, 14):
        has_doc = bool(code.co_flags & _CO_HAS_DOCSTRING)
    else:
        has_doc = bool(code.co_flags & _CO_OPTIMIZED) and not code.co_name.startswith("<")
    return (None,) + consts[1:] if has_doc else consts


def _stabilize_const(const: object, depth: int) -> str:
    """Describe one const so the description never embeds an address."""
    if isinstance(const, IMMUTABLE_PRIMS):
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
    consts = ",".join(_stabilize_const(c, depth) for c in code_consts_without_docstring(code))
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


def own_source(fn: object) -> str:
    """``inspect.getsource``, without following ``__wrapped__`` for a function.

    ``getsource`` unwraps, so for a ``functools.wraps`` wrapper it returned
    the WRAPPED function's text. Keyed by that, every function one decorator
    wraps shares the wrapper's code object and was keyed by whichever wrapped
    body was read first (round 18); analysed by it, the wrapper's own body was
    never read (round 19). Reading the code object gives each half its own
    text; whoever needs the wrapped half reaches it through ``__wrapped__``.
    Raises what ``inspect.getsource`` raises (`SOURCE_RETRIEVAL_ERRORS`).
    """
    if isinstance(fn, types.FunctionType) and hasattr(fn, "__wrapped__"):
        return inspect.getsource(fn.__code__)
    return inspect.getsource(fn)


def _wrapped_of(fn: object) -> object | None:
    """What a ``functools.wraps`` wrapper FUNCTION wraps, or None."""
    if isinstance(fn, types.FunctionType):
        return getattr(fn, "__wrapped__", None)
    return None


def _with_wrapped(own: str, fn: object, depth: int) -> str:
    """*own*, the digest of *fn*'s own code, joined with the identity of the
    function it wraps, if it is a ``functools.wraps`` wrapper.

    A wrapper's own code is shared by every function its decorator wraps, so
    on its own it cannot tell ``@timed def a`` from ``@timed def b`` -- nor see
    an edit to either body. The wrapped function is what the wrapper runs.
    """
    wrapped = _wrapped_of(fn)
    if wrapped is None or depth >= _MAX_WRAP_DEPTH:
        return own
    inner = _callable_identity(wrapped, depth + 1)
    return hashlib.sha256(f"{own}:wraps:{inner}".encode("utf-8")).hexdigest()


#: How many ``__wrapped__`` layers an identity follows; a runaway guard.
_MAX_WRAP_DEPTH = 8


def source_digest(fn: object) -> str | None:
    """*fn*'s identity read from its source file -- `source_identity_digest`
    of its own source (`own_source`), with what a ``functools.wraps`` wrapper
    wraps folded in -- or ``None`` when there is no source to read."""
    return _source_digest(fn, 0)


def _source_digest(fn: object, depth: int) -> str | None:
    own = own_source_digest(fn)
    return None if own is None else _with_wrapped(own, fn, depth)


def own_source_digest(fn: object) -> str | None:
    """`source_identity_digest` of `own_source` alone -- the text in *fn*'s
    own file, without what a wrapper wraps -- or ``None`` without source.
    What a check that one FILE still holds the code that runs compares."""
    try:
        return source_identity_digest(own_source(fn))
    except SOURCE_RETRIEVAL_ERRORS:
        return None


def opaque_identity(fn: object) -> str:
    """A stable ``module.qualname`` for a callable with no source and no code:
    a builtin, a C-extension function, a ufunc, a ``functools.partial``.

    A partial reprs as ``functools.partial(<function slow at 0x...>, 1)``: an
    ADDRESS, so its identity differed in every process and a cached partial
    never hit across processes (found attacking the decorator before round
    26). What it wraps is stable; what it binds reaches a key through the
    arguments and the function's own namespace name.
    """
    depth = 0
    while isinstance(fn, functools.partial) and depth < 8:
        fn = fn.func
        depth += 1
    module = getattr(fn, "__module__", None) or "?"
    qualname = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or repr(fn)
    return f"{module}.{qualname}"


def compiled_identity(fn: object) -> str:
    """*fn*'s identity when its source cannot be read: its `bytecode_identity`
    (a wrapper's with what it wraps folded in), or for a callable with no
    code at all a digest of its `opaque_identity`."""
    return _compiled_identity(fn, 0)


def _compiled_identity(fn: object, depth: int) -> str:
    own = bytecode_identity(fn)
    if own is None:
        return hashlib.sha256(f"__cash_opaque__:{opaque_identity(fn)}".encode("utf-8")).hexdigest()
    return _with_wrapped(own, fn, depth)


def callable_identity(fn: object) -> str:
    """The one digest that stands for a callable's code, wherever cash keys on it.

    Its own source text reduced by `source_identity_digest` (a comment, a
    reformat or cash's own decorator arguments do not move it); failing that
    its compiled body (`bytecode_identity`: a REPL, ``exec``, a moved file);
    failing that its `opaque_identity`. For a ``functools.wraps`` wrapper,
    its OWN code (`own_source`) together with the identity of what it wraps,
    so an edit to either half moves it and two functions wrapped by one
    decorator never share it. Never raises.
    """
    return _callable_identity(fn, 0)


def _callable_identity(fn: object, depth: int) -> str:
    digest = _source_digest(fn, depth)
    return digest if digest is not None else _compiled_identity(fn, depth)


# ---------------------------------------------------------------------------
# Is the text on disk still the code that is running?
# ---------------------------------------------------------------------------
#
# A helper's digest is computed lazily, the first time a call needs it, from
# the source on disk. If the file was edited after this process imported it --
# new files land, the process restarts some time later: every deploy, every
# `git pull` under a long-running worker -- that first digest describes the NEW
# text while the code object executing is the OLD one. Key from the new text,
# result from the old code, stored together, served to the restarted process:
# round 17 measured 0.500504 served for 0.530876 (CAS-110).
#
# Recompiling just the function's text and comparing is NOT a valid check.
# Bytecode depends on the surrounding module: inside its module the compiler
# can see `hashlib` is an imported module and emits a different call form than
# for the same text compiled alone. Measured: 4 of 6 unedited functions
# "mismatched" that way. Compiling the whole FILE gives the compiler the same
# context the import had -- 0 of 7 mismatched -- and `bytecode_identity` ignores
# line numbers, so an edit to a neighbouring function does not read as one.

import time as _time

_IMPORT_TIME = _time.time()
_MODULE_CODE_CACHE: dict[str, tuple[int, int, types.CodeType | None]] = {}
_MODULE_CODE_CACHE_MAX = 256
_PROCESS_START: float | None = None

#: How long a file must have been left alone before something read from it is
#: memoised on its ``(mtime, size)``. See `stat_has_settled`.
_SETTLED_SECONDS = 2.0


def stat_has_settled(st: object) -> bool:
    """True when the file *st* describes may be memoised on its stat.

    A memo keyed on ``(mtime, size)`` cannot see an edit that keeps both, and
    a same-size edit moments after the last one can: the mtime moves in
    ticks -- ~15.6 ms on Windows, whole seconds on HFS+ and ext3, two on FAT
    -- so two saves inside one tick share it. The module digests served the
    first save's code for the second that way, and five tests that rewrite a
    file straight after reading it failed intermittently on Windows CI.

    Git's "racy git" rule, applied when the entry is made: a file whose mtime
    is within a tick of now may still be written again without the stat
    moving, so it is read every time; one untouched for longer than the
    coarsest tick cannot be, so what was read from it holds until the stat
    moves. Ask BEFORE reading the file, so an edit the read missed cannot be
    one that kept the stat. Costs a re-read for a couple of seconds after
    each save. (``file_dep_snapshot`` holds its input digests to the same
    rule, over a longer window.)
    """
    return _time.time() - st.st_mtime > _SETTLED_SECONDS


def _process_start_time() -> float:
    """Wall-clock time this process started, best effort, cached."""
    global _PROCESS_START
    if _PROCESS_START is not None:
        return _PROCESS_START

    started: float | None = None
    try:
        import psutil  # type: ignore[import-not-found]

        started = float(psutil.Process().create_time())
    except Exception:  # noqa: BLE001 - optional dependency, any failure
        started = None
    if started is None and sys.platform == "win32":
        try:
            # Local: Windows only.
            import ctypes
            from ctypes import wintypes

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            creation, exit_, kernel, user = (wintypes.FILETIME() for _ in range(4))
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            if k32.GetProcessTimes(
                k32.GetCurrentProcess(),
                ctypes.byref(creation),
                ctypes.byref(exit_),
                ctypes.byref(kernel),
                ctypes.byref(user),
            ):
                ticks = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
                started = ticks / 1e7 - 11644473600.0  # FILETIME epoch -> Unix
        except Exception:  # noqa: BLE001
            started = None
    if started is None and os.path.exists("/proc/self/stat"):
        try:
            with open("/proc/self/stat", encoding="ascii") as fh:
                fields = fh.read().rsplit(")", 1)[1].split()
            start_ticks = int(fields[19])  # field 22 overall
            with open("/proc/stat", encoding="ascii") as fh:
                btime = next(int(line.split()[1]) for line in fh if line.startswith("btime"))
            started = btime + start_ticks / os.sysconf("SC_CLK_TCK")
        except Exception:  # noqa: BLE001
            started = None
    if started is None:
        # cash's own import time. Misses only a file edited in the gap between
        # this process starting and cash being imported -- normally the first
        # lines of the program.
        started = _IMPORT_TIME
    _PROCESS_START = started
    return started


def read_code_file(path: str) -> bytes:
    """The bytes of the code file at *path*, read where no file tracker sees it.

    Cash reads a module to key or check the code that runs, not as data that
    code reads. Through ``open`` the read was recorded by whatever statement or
    cached call was running (the tracker patches ``open``), so the module
    became a raw-bytes input and a comment added to it re-ran the work. The
    memos in front of these reads hid it once a file had settled; for two
    seconds after a save (`stat_has_settled`) every key re-read the file.
    ``io.FileIO`` is not patched.
    """
    with io.FileIO(path, "rb") as fh:
        return fh.readall()


def read_code_text(path: str) -> str:
    """:func:`read_code_file` as UTF-8 text, newlines translated as text-mode ``open`` does."""
    return read_code_file(path).decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


def _compiled_module(path: str) -> types.CodeType | None:
    """The whole file at *path*, compiled, cached per (path, mtime, size)."""

    try:
        st = os.stat(path)
    except OSError:
        return None
    cached = _MODULE_CODE_CACHE.get(path)
    if cached is not None and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
        return cached[2]
    settled = stat_has_settled(st)
    try:
        source = read_code_file(path)
        code: types.CodeType | None = compile(source, path, "exec", dont_inherit=True)
    except (OSError, SyntaxError, ValueError):
        code = None
    if not settled:
        return code
    if len(_MODULE_CODE_CACHE) >= _MODULE_CODE_CACHE_MAX:
        _MODULE_CODE_CACHE.clear()
    _MODULE_CODE_CACHE[path] = (st.st_mtime_ns, st.st_size, code)
    return code


def _pyc_proves_unchanged(path: str, st: object) -> bool:
    """True when the module's ``.pyc`` shows *path* is what this process imported.

    A file's mtime alone proved nothing: ``shutil.copy2``, ``cp -p``, rsync,
    robocopy and Explorer all keep the SOURCE file's mtime, so a helper
    replaced under a running process by a copy made two hours earlier looked
    untouched since the process started -- and was keyed by the new text while
    the old code ran (round 19).

    The ``.pyc`` is a record of the import: importlib checks its header against
    the source's (mtime, size) and rewrites it when they differ. A ``.pyc``
    older than this process whose header still matches the source means the
    import saw exactly this (mtime, size). A newer one may have been written by
    a later import of a different file, and a missing one says nothing -- both
    fall back to compiling the file, once per (path, mtime, size).
    """

    started = _process_start_time()
    if st.st_mtime > started:
        return False
    try:
        pyc = importlib.util.cache_from_source(path)
        if os.stat(pyc).st_mtime > started:
            return False
        # FileIO, not `open`: read through `open` inside a cached call's body --
        # a nested cached call's key being built -- the module's .pyc became
        # that call's input, and editing ANY function in the module re-ran it
        # (round 20: a 25 s step on every deploy).
        with io.FileIO(pyc, "rb") as fh:
            header = fh.read(16)
    except (OSError, ValueError, NotImplementedError):
        return False
    if len(header) < 16 or header[:4] != importlib.util.MAGIC_NUMBER:
        return False
    if int.from_bytes(header[4:8], "little") != 0:
        return False  # hash-based pyc: no timestamp to compare
    return int.from_bytes(header[8:12], "little") == (int(st.st_mtime) & 0xFFFFFFFF) and int.from_bytes(
        header[12:16], "little"
    ) == (st.st_size & 0xFFFFFFFF)


def _code_objects(code: types.CodeType):
    yield code
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from _code_objects(const)


def loaded_code_matches_disk(fn: object) -> bool:
    """False when *fn*'s source file was edited after this process loaded it.

    True whenever that cannot be shown: no file (a REPL, ``exec``, a notebook
    cell), a file its ``.pyc`` shows is the one imported (the common case; see
    `_pyc_proves_unchanged`), or a file whose compiled form still contains this
    function unchanged. Anything else is compiled, once per file version.
    """

    if isinstance(fn, type):
        # A class has no code of its own; its methods do, and an edit to any
        # of them is an edit to the class.
        return all(loaded_code_matches_disk(m) for m in class_functions(fn))
    code = getattr(fn, "__code__", None)
    if not isinstance(code, types.CodeType):
        return True
    path = code.co_filename
    if not path or path.startswith("<"):
        return True
    if not path.endswith((".py", ".pyw")):
        # Code compiled from something that is not a Python file -- a doc
        # page's fence, a template -- cannot be recompiled whole to compare,
        # and "does not compile" would read as "edited".
        return True
    try:
        if _pyc_proves_unchanged(path, os.stat(path)):
            return True
    except OSError:
        return True
    module = _compiled_module(path)
    if module is None:
        return False  # the file no longer compiles: it is not what runs
    live = bytecode_identity(fn)
    if live is None:
        return True
    qualname = getattr(code, "co_qualname", None)
    for candidate in _code_objects(module):
        if qualname is not None:
            if getattr(candidate, "co_qualname", None) != qualname:
                continue
        elif candidate.co_name != code.co_name:
            continue
        # Not ``types.FunctionType(candidate, {})``: that raises for a nested
        # function with free variables (it needs a closure), and only the
        # code is read anyway.
        probe = types.SimpleNamespace(__code__=candidate)
        if bytecode_identity(probe) == live:
            return True
    return False


def class_functions(cls: type) -> list[types.FunctionType]:
    """The plain functions defined directly in *cls*, unwrapping descriptors."""
    found = []
    for value in vars(cls).values():
        func = getattr(value, "__func__", value)  # staticmethod / classmethod
        if isinstance(value, property):
            func = value.fget
        if isinstance(func, types.FunctionType) and getattr(func, "_cash_cached", False):
            # A cached method's class attribute is cash's wrapper, whose globals
            # are cash's own: walked as the class's code, it reported
            # KEY-UNHASHABLE-GLOBAL for `Model.fit.ACTIVE_CONFIG`, a name in
            # no file of the user's (round 19). The user's function is inside.
            func = getattr(func, "__wrapped__", func)
        if isinstance(func, types.FunctionType):
            found.append(func)
    return found


def loaded_class_identity(cls: type) -> str | None:
    """A digest of *cls* from its LOADED methods, for when disk is not them."""
    try:
        parts = [cls.__qualname__]
        for func in sorted(class_functions(cls), key=lambda f: f.__qualname__):
            parts.append(f"{func.__qualname__}={bytecode_identity(func)}")
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    except (AttributeError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# A module's identity
# ---------------------------------------------------------------------------

#: ``{path: (mtime_ns, size, identity_digest)}`` for `module_identity`. One
#: entry per file, replaced when it moves. The identity below parses the file, which is far too much to
#: repeat per statement -- and even the plain read it replaces was one file
#: read per statement per module.
_MODULE_IDENTITY_CACHE: dict[str, tuple[int, int, str]] = {}


def _module_text_identity(raw: bytes) -> bytes:
    """What a module file says, with what it merely looks like removed.

    The digest of this lands in the lineage of every name bound from the
    module and in the key of every statement that reads one, so anything it
    covers re-runs work when it moves. Hashing the FILE meant a comment, a
    blank line or a reformat re-ran everything built on the module: measured
    2026-09-21, adding one comment to a module re-executed a 1.2 s call that
    used a function the edit did not touch. Round 27 r27s2 reported the same
    thing at scale -- editing one helper re-read all 10,000 of their ticket
    files, 48.7 s against a 17.3 s control, later 9.1x.

    So: the module re-rendered from its AST, which drops comments and
    normalises formatting, and carries no line or column numbers -- without
    that last part, inserting a comment at the top would still move every
    node below it and nothing would be gained.

    ``ast.unparse`` rather than ``ast.dump``: both drop what we want dropped,
    but ``dump`` prints the AST's own field names, so a Python release that
    adds a field moves every module's digest. Generated source moves less.
    Neither is a promise across versions, and cache keys are already not
    portable across machines (see docs/how-it-works/cache-keys-and-lineage.md),
    but there is no reason to add a reason.

    Docstrings go too, the module's own included (``strip_docstrings``): they
    are prose, the same as comments.

    And ``@cash:`` directives stay, because they are instructions TO cash --
    ``# @cash:assume-safe`` on a line waives a purity check, and cash's own
    diagnostic says "@cash: directives are part of its source identity".

    Each is kept as its WHOLE line, in source order, which anchors it to the
    code it annotates: moving ``# @cash:assume-safe`` from one function to
    another moves the digest, where keeping only the directive text would
    have made that invisible -- the unsafe direction. The cost is that
    reformatting a line that carries a directive still re-runs work. That is
    a narrow class and it errs toward invalidating.

    Matched with ``annotations.ANNOTATION_PATTERN`` itself, so the set of
    directives that counts here cannot drift from the set cash parses.

    Falls back to the raw bytes for anything that will not decode or parse --
    a data file among the dependencies, a module written for a different
    Python. That is exactly the previous behaviour, so nothing that works
    today can be made worse by this.
    """
    try:
        text = raw.decode("utf-8")
        tree = ast.parse(text)
        drop_docstrings(tree, module=True)
        rendered = ast.unparse(tree)
    except (UnicodeDecodeError, SyntaxError, ValueError, AttributeError, RecursionError):
        return raw

    parts = [rendered]
    parts.extend(line.strip() for line in text.splitlines() if ANNOTATION_PATTERN.search(line))
    return "\n".join(parts).encode("utf-8")


def module_identity(module: object) -> str | None:
    """The identity digest of a module's source file -- see
    `_module_text_identity` for what it covers -- or ``None`` when the file
    cannot be read. *module* is a module object or the path of its file.
    Memoised on the file's stat."""
    path = module if isinstance(module, str) else getattr(module, "__file__", None)
    if not path:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    cached = _MODULE_IDENTITY_CACHE.get(path)
    if cached is not None and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
        return cached[2]
    settled = stat_has_settled(st)
    try:
        raw = read_code_file(path)
    except OSError:
        return None
    # Keyed on the same signal `FunctionTracker.check_tracked_modules` uses to
    # notice a module changed at all, so a change this memo would miss is one
    # cash would not have reloaded for either -- once the file has settled,
    # since a same-size save inside one mtime tick keeps that stat too.
    digest = hashlib.sha256(_module_text_identity(raw)).hexdigest()
    if settled:
        _MODULE_IDENTITY_CACHE[path] = (st.st_mtime_ns, st.st_size, digest)
    return digest
