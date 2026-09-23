"""What reading some names from a module depends on -- and when that cannot be bounded.

A statement that reads ``lib.load`` used to be keyed on the whole of ``lib``,
so editing any other function in the file re-ran it. Round 27, r27s2: editing
one helper re-read all 10,000 of their ticket files, 48.7 s against a 17.3 s
control, later 9.1x, because ``corpus = tl.load_corpus(ROOT, MONTH)`` was
keyed on a module that had changed while ``load_corpus`` had not.

:func:`closure_digest` answers the narrower question: given the module's
source and the names a statement reads from it, a digest of exactly the
source those names depend on. The closure is deliberately generous, because
leaving something out means serving a value computed from code that has since
changed:

* every top-level statement that BINDS a name in the closure -- all of them,
  when a name is bound more than once or conditionally;
* everything those statements read that the module binds, transitively --
  a helper a function calls, a constant it reads, the function run at import
  time to build a value it reads (``TABLE = build_table()``);
* every top-level statement that is not a plain definition -- a call, an
  ``if``, a loop, an augmented or attribute assignment -- whatever it names,
  because code run at import time can rebind or mutate anything;
* every ``@cash:`` directive line in the file, because they are instructions
  to cash rather than commentary.

And it returns None -- "depend on the whole module", the old behaviour -- the
moment it cannot bound the answer: a name the module does not bind literally,
``from x import *``, a module-level ``__getattr__``, anything in the closure
that reaches the module's namespace dynamically (``globals()``, ``exec``,
``setattr``, ``sys.modules``, ``__dict__`` ...), or anything in it whose
result differs between two runs of the same source (``time.time()``,
``random``, ``uuid4()`` ...). That last is not caution for its own sake:
narrowing reloads a module on an edit to any part of it, so import-time code
like ``STAMP = time.time()`` would hand a new value to an old key -- a stale
answer keying on the whole module never gave.

Pure AST over the file, so the runtime and the upstream simulation, which
must compute identical keys, get identical answers from the same file.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass

from ..source_norm import read_code_text, stat_has_settled, unparse_without_docstrings

__all__ = ["closure_digest", "static_attribute_reads"]

#: Names whose use means the code can reach the module namespace by string.
_DYNAMIC_CALLS = frozenset(
    {
        "globals",
        "locals",
        "vars",
        "exec",
        "eval",
        "compile",
        "setattr",
        "delattr",
        "__import__",
    }
)

#: Attributes whose use means the same, through an object.
_DYNAMIC_ATTRS = frozenset({"__dict__", "modules"})

#: Calls whose result differs between two runs of identical source. Import-
#: time code calling one (``STAMP = time.time()``) gets a new value on every
#: reload, and per-symbol keying reloads a module on an edit to ANY of it --
#: so an edit elsewhere in the file would bring a new STAMP under the old key.
#: Keying on the whole module never met this: it reloaded only on a change,
#: which re-keyed everything. Any of these in a closure makes it unbounded, the
#: old behaviour, rather than open a stale value the old scheme could not.
_NONDETERMINISTIC_TAILS = frozenset(
    {
        "now",
        "utcnow",
        "today",
        "time",
        "time_ns",
        "perf_counter",
        "monotonic",
        "process_time",
        "uuid1",
        "uuid4",
        "urandom",
        "getpid",
        "token_hex",
        "token_bytes",
        "token_urlsafe",
    }
)
_NONDETERMINISTIC_ROOTS = frozenset({"random", "secrets"})

_DIRECTIVE = re.compile(r"#\s*@cash:")


@dataclass(frozen=True)
class _Analysis:
    statements: tuple[ast.stmt, ...]
    #: name -> indices of the top-level statements that bind it.
    binders: dict[str, tuple[int, ...]]
    #: per statement, the names it reads that the module binds.
    reads: tuple[frozenset[str], ...]
    #: indices always in every closure.
    effectful: frozenset[int]
    #: per statement: does it reach the namespace dynamically?
    dynamic: tuple[bool, ...]
    directives: tuple[str, ...]
    #: the module cannot be bounded at all.
    opaque: bool


def _bound_names(stmt: ast.stmt) -> tuple[set[str], bool]:
    """Names *stmt* binds at module level, and whether it is a plain definition."""
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {stmt.name}, True
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        names = set()
        for alias in stmt.names:
            if alias.name == "*":
                return set(), False
            names.add(alias.asname or alias.name.split(".")[0])
        return names, True
    if isinstance(stmt, ast.Assign):
        names = set()
        for target in stmt.targets:
            for node in ast.walk(target):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    names.add(node.id)
                elif isinstance(node, (ast.Attribute, ast.Subscript, ast.Starred)):
                    # `CONFIG["x"] = 1` mutates; `a, *b = ...` is fine but rare.
                    if not isinstance(node, ast.Starred):
                        return names, False
        return names, True
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return {stmt.target.id}, True
    # Anything else runs code at import time. It is always in the closure, and
    # it may still bind names (`if FLAG: def f(): ...`, `for x in ...`).
    names = {n.id for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    for n in ast.walk(stmt):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
    return names, False


def _dotted(func: ast.expr) -> list[str]:
    """``np.random.rand`` -> ['np', 'random', 'rand']; [] if not a name chain."""
    parts: list[str] = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
        return parts[::-1]
    return []


_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _is_nondeterministic_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    chain = _dotted(node.func)
    return bool(chain) and (chain[-1] in _NONDETERMINISTIC_TAILS or any(p in _NONDETERMINISTIC_ROOTS for p in chain))


def _is_dynamic(stmt: ast.stmt, runs_at_import: bool = True) -> bool:
    """Can *stmt* make a closure unboundable? See the module docstring.

    Reaching the namespace by string counts wherever it is. A result that
    differs run to run counts only in code that runs at import time: a
    function's body runs when it is called, and a reload gives it nothing new
    -- a helper timing its own steps (``t0 = time.perf_counter()``) keyed
    every statement using it on the whole module, so any edit to the file
    re-ran them (round 29, r29s1 and r29s3). *runs_at_import* is False for a
    ``def`` nothing at import time calls; its decorators and default values
    run at import time all the same.
    """
    for node in ast.walk(stmt):
        if isinstance(node, ast.Call):
            chain = _dotted(node.func)
            if len(chain) == 1 and chain[0] in _DYNAMIC_CALLS:
                return True
        if isinstance(node, ast.Attribute) and node.attr in _DYNAMIC_ATTRS:
            return True
    if runs_at_import or not isinstance(stmt, _FUNCTIONS):
        return any(_is_nondeterministic_call(node) for node in ast.walk(stmt))
    at_import = [*stmt.decorator_list, *stmt.args.defaults, *(d for d in stmt.args.kw_defaults if d is not None)]
    return any(_is_nondeterministic_call(node) for part in at_import for node in ast.walk(part))


def _called_at_import(statements: tuple[ast.stmt, ...]) -> set[str]:
    """Names of the module's functions that code run at import time calls,
    directly or through one another. Everything but a ``def``'s body runs at
    import time; a class body does too, methods and all -- over-counting only
    keeps the old, whole-module answer."""
    defs = {s.name: s for s in statements if isinstance(s, _FUNCTIONS)}

    def calls(nodes) -> set[str]:
        return {
            n.func.id
            for part in nodes
            for n in ast.walk(part)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in defs
        }

    pending = calls(s for s in statements if not isinstance(s, _FUNCTIONS))
    for fn in defs.values():
        pending |= calls([*fn.decorator_list, *fn.args.defaults, *(d for d in fn.args.kw_defaults if d is not None)])
    reached: set[str] = set()
    while pending:
        name = pending.pop()
        if name not in reached:
            reached.add(name)
            pending |= calls(defs[name].body) - reached
    return reached


def _analyse(source: str) -> _Analysis | None:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    statements = tuple(tree.body)
    binders: dict[str, list[int]] = {}
    effectful: set[int] = set()
    opaque = False
    for i, stmt in enumerate(statements):
        names, plain = _bound_names(stmt)
        if isinstance(stmt, ast.ImportFrom) and any(a.name == "*" for a in stmt.names):
            opaque = True
        if not plain:
            effectful.add(i)
        for name in names:
            binders.setdefault(name, []).append(i)
            if name in ("__getattr__", "__dir__"):
                opaque = True
    bound = frozenset(binders)
    import_called = _called_at_import(statements)
    reads = tuple(
        frozenset(
            n.id for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in bound
        )
        for stmt in statements
    )
    return _Analysis(
        statements=statements,
        binders={k: tuple(v) for k, v in binders.items()},
        reads=reads,
        effectful=frozenset(effectful),
        dynamic=tuple(_is_dynamic(s, not isinstance(s, _FUNCTIONS) or s.name in import_called) for s in statements),
        directives=tuple(ln.strip() for ln in source.splitlines() if _DIRECTIVE.search(ln)),
        opaque=opaque,
    )


#: ``{path: (mtime_ns, size, analysis)}``, one entry per file.
_ANALYSES: dict[str, tuple[int, int, _Analysis | None]] = {}


def analysis_for(path: str) -> _Analysis | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    cached = _ANALYSES.get(path)
    if cached is not None and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
        return cached[2]
    settled = stat_has_settled(st)
    try:
        analysis = _analyse(read_code_text(path))
    except (OSError, UnicodeDecodeError):
        analysis = None
    if settled:
        _ANALYSES[path] = (st.st_mtime_ns, st.st_size, analysis)
    return analysis


def closure_digest_of_source(source: str, names: Iterable[str]) -> str | None:
    """:func:`closure_digest` for source text in hand. None: cannot be bounded."""
    analysis = _analyse(source)
    return _digest(analysis, names) if analysis is not None else None


#: ``{(path, names): (mtime_ns, size, digest)}``. Keys are computed for every
#: statement by both the runtime and the simulation, so the walk and the
#: re-rendering of the closure must not repeat while the file stands still.
_DIGESTS: dict[tuple[str, frozenset[str]], tuple[int, int, str | None]] = {}
_MAX_DIGESTS = 4096


def closure_digest(path: str, names: Iterable[str]) -> str | None:
    """Digest of what reading *names* from the module at *path* depends on.

    None when it cannot be bounded; the caller then depends on the whole
    module, which is always correct and only ever slower.
    """
    key = (path, frozenset(names))
    try:
        st = os.stat(path)
    except OSError:
        return None
    cached = _DIGESTS.get(key)
    if cached is not None and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
        return cached[2]
    settled = stat_has_settled(st)
    analysis = analysis_for(path)
    digest = _digest(analysis, key[1]) if analysis is not None else None
    if not settled:
        return digest
    if len(_DIGESTS) >= _MAX_DIGESTS:
        _DIGESTS.clear()
    _DIGESTS[key] = (st.st_mtime_ns, st.st_size, digest)
    return digest


def _digest(analysis: _Analysis, names: Iterable[str]) -> str | None:
    names = sorted(set(names))
    if not names or analysis.opaque:
        return None
    if any(name not in analysis.binders for name in names):
        return None  # not bound literally: cannot say what it is
    # Start from the requested names AND everything import-time code reads:
    # effectful statements are in every closure, so what they reach is too.
    included: set[int] = set(analysis.effectful)
    queue = list(names)
    for i in analysis.effectful:
        queue.extend(analysis.reads[i])
    seen: set[str] = set()
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        # EVERY statement binding the name: a redefinition or a conditional
        # one below the first is as much its source as the first.
        for i in analysis.binders.get(name, ()):
            included.add(i)
            queue.extend(analysis.reads[i])
    if any(analysis.dynamic[i] for i in included):
        return None
    h = hashlib.sha256()
    h.update(("names:" + ",".join(names) + "\n").encode("utf-8"))
    for i in sorted(included):
        h.update(unparse_without_docstrings(ast.unparse(analysis.statements[i])).encode("utf-8"))
        h.update(b"\n")
    for line in analysis.directives:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


@functools.lru_cache(maxsize=8192)
def _static_attribute_reads(code: str, name: str) -> frozenset[str] | None:
    reads = _static_attribute_reads_uncached(code, name)
    return frozenset(reads) if reads is not None else None


def static_attribute_reads(code: str, name: str) -> set[str] | None:
    """The attributes *code* reads from *name*, when that is ALL it does with it.

    Memoised: a statement's text does not change, and this runs for every key.
    """
    reads = _static_attribute_reads(code, name)
    return set(reads) if reads is not None else None


def _static_attribute_reads_uncached(code: str, name: str) -> set[str] | None:
    """The attributes *code* reads from *name*, when that is ALL it does with it.

    None when *code* uses *name* any other way -- passes it, returns it, binds
    it, stores or deletes an attribute on it, reads a dunder -- because then
    what it depends on cannot be read off the attributes. An empty set is
    never returned: a name read with no attribute is a bare use.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    attrs: set[str] = set()
    covered: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == name:
            if not isinstance(node.ctx, ast.Load) or node.attr.startswith("__"):
                return None
            attrs.add(node.attr)
            covered.add(id(node.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name and id(node) not in covered:
            return None  # a bare use, or a rebinding
    return attrs or None
