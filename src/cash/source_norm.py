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

Deliberately COLLAPSED, beyond whitespace and comments:

* Numeric literals, to one spelling per value -- see `_canonical_number`.
  ``0.5`` and ``0.50`` compile to the same constant, so they are the same
  function. Type is kept, so ``1`` and ``1.0`` stay apart.
"""

import ast
import functools
import hashlib
import io
import re
import textwrap
import tokenize
import types

__all__ = [
    "bytecode_identity",
    "normalize_source_for_hash",
    "source_identity_digest",
    "strip_cache_decorator",
]

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
_CACHE_DECORATOR_PARAMS = frozenset({
    "depends_on", "dynamic_depends_on", "file_depends_on", "ttl", "cache_if",
    "chunk_max_items", "chunk_max_bytes", "strict", "assume_safe",
    "allow_random", "frozen",
})
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
        if any(kw.arg is None or kw.arg not in _CACHE_DECORATOR_PARAMS
               for kw in node.keywords):
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


def _process_start_time() -> float:
    """Wall-clock time this process started, best effort, cached."""
    global _PROCESS_START
    if _PROCESS_START is not None:
        return _PROCESS_START
    import os
    import sys

    started: float | None = None
    try:
        import psutil  # type: ignore[import-not-found]
        started = float(psutil.Process().create_time())
    except Exception:  # noqa: BLE001 - optional dependency, any failure
        started = None
    if started is None and sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            creation, exit_, kernel, user = (wintypes.FILETIME() for _ in range(4))
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            if k32.GetProcessTimes(k32.GetCurrentProcess(), ctypes.byref(creation),
                                   ctypes.byref(exit_), ctypes.byref(kernel),
                                   ctypes.byref(user)):
                ticks = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
                started = ticks / 1e7 - 11644473600.0     # FILETIME epoch -> Unix
        except Exception:  # noqa: BLE001
            started = None
    if started is None and os.path.exists("/proc/self/stat"):
        try:
            with open("/proc/self/stat", encoding="ascii") as fh:
                fields = fh.read().rsplit(")", 1)[1].split()
            start_ticks = int(fields[19])                 # field 22 overall
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


def _compiled_module(path: str) -> types.CodeType | None:
    """The whole file at *path*, compiled, cached per (path, mtime, size)."""
    import os

    try:
        st = os.stat(path)
    except OSError:
        return None
    cached = _MODULE_CODE_CACHE.get(path)
    if cached is not None and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
        return cached[2]
    try:
        with open(path, "rb") as fh:
            source = fh.read()
        code: types.CodeType | None = compile(source, path, "exec", dont_inherit=True)
    except (OSError, SyntaxError, ValueError):
        code = None
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
    import importlib.util
    import os

    started = _process_start_time()
    if st.st_mtime > started:
        return False
    try:
        pyc = importlib.util.cache_from_source(path)
        if os.stat(pyc).st_mtime > started:
            return False
        with open(pyc, "rb") as fh:
            header = fh.read(16)
    except (OSError, ValueError, NotImplementedError):
        return False
    if len(header) < 16 or header[:4] != importlib.util.MAGIC_NUMBER:
        return False
    if int.from_bytes(header[4:8], "little") != 0:
        return False                  # hash-based pyc: no timestamp to compare
    return (int.from_bytes(header[8:12], "little") == (int(st.st_mtime) & 0xFFFFFFFF)
            and int.from_bytes(header[12:16], "little") == (st.st_size & 0xFFFFFFFF))


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
    import os

    if isinstance(fn, type):
        # A class has no code of its own; its methods do, and an edit to any
        # of them is an edit to the class.
        return all(loaded_code_matches_disk(m) for m in _class_functions(fn))
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
        return False          # the file no longer compiles: it is not what runs
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


def _class_functions(cls: type) -> list[types.FunctionType]:
    """The plain functions defined directly in *cls*, unwrapping descriptors."""
    found = []
    for value in vars(cls).values():
        func = getattr(value, "__func__", value)          # staticmethod / classmethod
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
        for func in sorted(_class_functions(cls), key=lambda f: f.__qualname__):
            parts.append(f"{func.__qualname__}={bytecode_identity(func)}")
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    except (AttributeError, TypeError, ValueError):
        return None
