"""Decorator-side purity analyzer.

Walks the body of a ``@cash.cache``-decorated function plus any
*module-bounded* helpers it calls, and reports:

* **Known-impure calls** - ``requests.post``, ``os.system``,
  ``logging.info``, file-write methods (``to_csv``, ``write``, ...).
* **Explicit dynamism** - ``eval``/``exec``/``compile``,
  ``getattr(obj, name)(...)`` where *name* is not a constant,
  calling a *parameter* directly (``def f(cb): cb(x)``).
* **Discarded-return calls** - ``f(x)`` as a statement (return
  value thrown away). Strong syntactic signal of "I'm calling this
  for the side effect". Skipped for callees in
  :data:`KNOWN_PURE_BUILTINS` (where discarding is just dead code).
* **Scope mutations** - ``global``/``nonlocal``, assignment to
  ``Attribute`` or ``Subscript`` targets, augmented-assign to same.

As a side benefit, the same walk captures source hashes of every
analyzed user-code helper. Those hashes become part of the cache
key, so editing a helper invalidates the parent cache automatically
on the next run.

**Boundary rule** - recursion follows callees that are *user code*:
the callable's source file is outside stdlib / site-packages
(``is_local_module``) **or** the callable shares the cached
function's top-level package. Everything else is treated as opaque
and optimistic (not flagged). Users who want a library call flagged
call ``cash.stateful(library_func)`` on it.

The analyzer is pure: no side effects of its own, no warnings
emitted from here. The decorator layer turns the report into
warnings / exceptions / cache-key components.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import hashlib
import importlib
import importlib.util
import inspect
import logging
import re
import sqlite3
import sys
import textwrap
import threading
import time
import types
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ._annotation_refs import annotation_referents
from ._paths import MAIN_MODULE_NAMES, resolve_main_module
from .analysis.ast_util import called_names, resolve_callee
from .analysis.file_effects import get_base_name, get_call_module, get_call_name
from .analysis.mutations import PANDAS_INPLACE_METHODS
from .effects import (
    CLOCK_WHEN_ARG_CALLS,
    ENVIRON_NAMES,
    METHOD_VERBS,
    MODULE_CALLS,
    MUTATOR_METHODS,
    Action,
    EffectKind,
    classify_call,
    dotted_name,
    environment_input,
)
from .exceptions import SOURCE_RETRIEVAL_ERRORS
from .purity import (
    KNOWN_PURE_BUILTINS,
    is_pure,
    is_stateful,
)
from .purity_flow import (
    LogOnlyFlow,
    fresh_name_nodes,
    is_log_helper,
    is_log_line,
    receiver_is_fresh,
)
from .source_norm import (
    callable_identity,
    compiled_identity,
    normalize_source_for_hash,
    own_source,
)
from .tracking.function_tracker import is_local_module
from .value_types import BUILTIN_NAMES

logger = logging.getLogger(__name__)

__all__ = [
    "DECORATOR_POLICY",
    "REPORTED_METHODS",
    "PurityAnalyzer",
    "PurityReport",
    "PurityIssue",
    "ISSUE_IMPURE_CALL",
    "ISSUE_DYNAMIC_PATTERN",
    "ISSUE_UNTRACKABLE_DEP",
    "ISSUE_DISCARDED_CALL",
    "ISSUE_SCOPE_MUTATION",
    "ISSUE_MUTABLE_GLOBAL",
    "ISSUE_AMBIENT_READ",
    "ISSUE_NETWORK_READ",
]

ISSUE_IMPURE_CALL = "impure_call"
ISSUE_DYNAMIC_PATTERN = "dynamic_pattern"
# Patterns where a dependency is resolved from a runtime value, so cash cannot
# see an edit to it and a cached result can go silently stale: eval/exec/compile,
# getattr(obj, name)() dynamic dispatch, importlib.import_module. Caching
# correctness cannot be guaranteed, so these RAISE by default (opt in with
# assume_safe=True). Distinct from ISSUE_DYNAMIC_PATTERN, which stays advisory.
ISSUE_UNTRACKABLE_DEP = "untrackable_dep"
ISSUE_DISCARDED_CALL = "discarded_call"
ISSUE_SCOPE_MUTATION = "scope_mutation"
ISSUE_MUTABLE_GLOBAL = "mutable_global"
# Reading ambient state -- the clock, the environment, the working directory, a
# fresh UUID. Not a side effect: nothing about the world changes. The result
# depends on something the cache key cannot see, so the FIRST call's answer is
# what every later call gets, in this process and every process after it.
# Advisory like ISSUE_IMPURE_CALL (warn, still cache), because freezing is
# sometimes exactly what the user wants -- but it is never what they want by
# accident, and it is invisible without this.
ISSUE_AMBIENT_READ = "ambient_read"
# Fetching from a server or querying a database: `requests.get(url)`,
# `cur.execute("SELECT ...")`, `pd.read_sql(...)`. Also not a side effect -- the
# hazard is that the server's answer is an input the key cannot see, so the
# first answer is served until something changes the key. Unlike the clock,
# there is a knob made for exactly this: `ttl=` bounds how old a served answer
# may be, and setting one silences the advisory.
ISSUE_NETWORK_READ = "network_read"
#: What a `network_read` finding names as the source of the answer.
_SOURCE: dict[EffectKind, str] = {EffectKind.NETWORK_READ: "server", EffectKind.DB_READ: "database"}


def _opens_tracked_database(func: ast.expr, namespace: dict[str, Any] | None) -> bool:
    """Is *func* ``sqlite3.connect``, however it is spelled?

    The file tracker records the path a SQLite connection opens as a file read
    (``FileTracker``), so a query over a connection the body opens itself is
    already keyed by the database file.
    """
    name = dotted_name(func)
    if name in ("sqlite3.connect", "sqlite3.dbapi2.connect"):
        return True
    if not namespace:
        return False
    if isinstance(func, ast.Name):
        return namespace.get(func.id) is sqlite3.connect
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return getattr(namespace.get(func.value.id), func.attr, None) is sqlite3.connect
    return False


#: Builtins whose whole job is to run code chosen at runtime. Reaching one of
#: these through `getattr(x, "<name>")` is the same hazard as calling it
#: directly, and the constant-name form otherwise slips past the dynamic
#: dispatch rule (which only fires on a NON-constant name).
_DYNAMIC_BUILTIN_NAMES = frozenset({"eval", "exec", "compile", "__import__"})

#: What a ``@cash.cache`` function's first call does about each kind of effect
#: its body has. What a kind IS lives in :mod:`cash.effects`, shared with the
#: notebook, whose own table is ``cash.analysis.file_effects.NOTEBOOK_POLICY``.
#: A decorated function is always cached -- refusing would cost the user the
#: compute and prevent nothing, since the body has run -- so the choice here is
#: only what to say. A test keeps this covering every kind.
DECORATOR_POLICY: dict[EffectKind, Action] = {
    EffectKind.FILE_WRITE: Action.WARN,
    EffectKind.FILE_READ: Action.CACHE_AS_INPUT,
    # What the server returns is an input the key cannot see: advise `ttl=`,
    # which silences it (KEY-NETWORK-READ). A database is a server too.
    EffectKind.NETWORK_READ: Action.SUGGEST_TTL,
    EffectKind.NETWORK_WRITE: Action.WARN,
    EffectKind.NETWORK: Action.WARN,
    EffectKind.DB_READ: Action.SUGGEST_TTL,
    EffectKind.DB_WRITE: Action.WARN,
    EffectKind.SUBPROCESS: Action.WARN,
    # These two are reported as ambient reads (KEY-AMBIENT-READ), not as
    # side effects: a hidden input is frozen, nothing is skipped.
    EffectKind.CLOCK: Action.WARN,
    # A read whose name is written out is folded into the key by value
    # (`Cash._fold_environment`); one whose name is only known at run time
    # still warns, as an ambient read.
    EffectKind.ENVIRONMENT: Action.CACHE_AS_INPUT,
    # A hit drops what the first call printed. A log line (`is_log_line`) is
    # exempt: a hit skipping it is what caching means.
    EffectKind.CONSOLE: Action.WARN,
    EffectKind.DISPLAY: Action.WARN,
    EffectKind.INTERACTIVE: Action.WARN,
}

#: Kinds reported as ambient reads rather than as side effects.
_AMBIENT_KINDS = frozenset({EffectKind.CLOCK, EffectKind.ENVIRONMENT})

#: Bare builtins whose discarded result is not worth a word: each is reported
#: by another rule already (an effect, or explicit dynamic execution).
_DISCARD_REPORTED_BUILTINS = frozenset(name for name in MODULE_CALLS if "." not in name) | {
    "open",
    "exec",
    "eval",
    "compile",
}


#: Method names the decorator reports on any receiver: a mutator, or a verb
#: whose kind it warns about. The discarded-call rule skips these (the call is
#: already reported), and "does this change a global?" reads them.
REPORTED_METHODS: frozenset[str] = MUTATOR_METHODS | frozenset(
    name for name, kind in METHOD_VERBS.items() if DECORATOR_POLICY[kind] is Action.WARN
)


@dataclass(frozen=True)
class PurityIssue:
    """A single issue surfaced by :class:`PurityAnalyzer`.

    Attributes:
        kind: One of ``impure_call``, ``dynamic_pattern``,
            ``untrackable_dep``, ``discarded_call``, ``scope_mutation``,
            ``mutable_global``, ``ambient_read``, ``network_read``.
        description: Human-readable summary (e.g. ``"requests.post()"``,
            ``"global X"``, ``"discards return of helper(...)``).
        where: Qualified name + line of the function containing the
            issue. For helpers, this is the helper's qualname so the
            user can fix the source of the problem, not just the
            outermost decorator.
        line: Line of the issue in the FILE that defines the function --
            what an editor's go-to-line takes. 0 for a finding about the
            whole function.
        filename: That file, so a finding in a helper names the helper's
            module rather than the file of the call that surfaced it.
        subject: The name the finding is about, where there is one -- the
            global of a ``mutable_global``.
        effect_kind: The :class:`~cash.effects.EffectKind` of the call an
            ``impure_call`` is about, where it has one.
    """

    kind: str
    description: str
    where: str
    line: int = 0
    filename: str = ""
    subject: str = ""
    effect_kind: EffectKind | None = None


@dataclass(frozen=True)
class PurityReport:
    """Result of analyzing a callable + its module-bounded helpers.

    Attributes:
        issues: All findings, in stable order (kind, line).
        helper_source_hashes: ``qualname -> digest`` for every user-code
            helper actually walked, captured at analysis time. Used as the
            fallback when per-call re-resolution fails (helper was
            deleted/renamed since analysis). The digest is over NORMALIZED
            source, so a comment or reformat in a helper does not
            invalidate its callers; for a helper with no readable source
            it is ``bytecode_identity``, matching what the per-call rehash
            computes for the same object.
        helper_resolution_paths: ``qualname -> (module_name, attr_chain)``
            for every walked helper. Used by the decorator to
            re-resolve the helper from ``sys.modules`` on each
            call and re-hash its source, so in-process
            redefinitions (notebook cells, REPL) invalidate the
            parent's cache key. The path is the one its CALLER uses:
            the caller's module and the name or attribute chain written
            at the call site (``("app", ("_sieve",))`` for
            ``from sievelib import sieve as _sieve``), so rebinding that
            name -- ``monkeypatch``, ``mock.patch`` -- reaches the key.
            Only a helper reached some other way (a class body's calls)
            falls back to its own home: ``(module, qualname chain)``.
        helper_bindings: ``(module_name, attr_chain, ref)`` for every
            call-site binding the walk followed -- helpers, cached callees
            and mocks alike -- where ``ref()`` is the object it held when
            analysed. ``bindings_changed`` compares them per call; a
            binding that no longer holds that object means the tree below
            it is not the one analysed.
        unkeyable: One description per binding that held a mock when
            analysed. A mock has no code to key and its answer is whatever
            the test configured, so a call reaching one runs uncached.
        opaque_callees: Qualified names of callees we encountered
            but couldn't read source for (C extensions, missing source,
            partial application). Treated as pure by default; strict
            mode promotes their presence to an issue.

            Opaque for PURITY only. One that still has a ``__code__``
            also gets an entry in ``helper_source_hashes`` and
            ``helper_resolution_paths``, digested from its compiled form,
            so editing it invalidates its callers.
    """

    issues: tuple[PurityIssue, ...] = ()
    helper_source_hashes: dict[str, str] = field(default_factory=dict)
    helper_resolution_paths: dict[str, tuple[str, tuple[str, ...]]] = field(default_factory=dict)
    #: ``qualname -> weakref`` for walked helpers that have NO resolution path
    #: -- a closure from a factory (``_make.<locals>.scaled``) cannot be looked
    #: up by qualname, so it used to be keyed by the analysis-time snapshot
    #: forever, which never sees its parameter defaults. Holding a
    #: weak reference lets the per-call rehash reach the live object.
    helper_objects: dict[str, Any] = field(default_factory=dict)
    opaque_callees: tuple[str, ...] = ()
    helper_bindings: tuple[tuple[str, tuple[str, ...], Any], ...] = ()
    unkeyable: tuple[str, ...] = ()
    #: Binding paths every call site of which is on a ``# @cash:assume-safe``
    #: line (``LEDGER.record(r)  # @cash:assume-safe``). The code is still
    #: followed; the data the bound object carries is not keyed, because the
    #: audited effect is what moves it (a ledger's count, a client's stats).
    waived_bindings: frozenset[tuple[str, tuple[str, ...]]] = frozenset()
    #: Environment reads, in the function and its helpers, whose current value
    #: the key folds on every call: ``("env", NAME)`` or ``("cwd", "")``
    #: (`cash.effects.environment_input`).
    environment_reads: frozenset[tuple[str, str]] = frozenset()

    @property
    def is_clean(self) -> bool:
        """True when no issues were flagged."""
        return not self.issues

    def format(self) -> str:
        """Human-readable multi-line summary."""
        if self.is_clean:
            return "(no purity issues detected)"
        by_where: dict[str, list[PurityIssue]] = {}
        for issue in self.issues:
            by_where.setdefault(issue.where, []).append(issue)
        lines = []
        for where, issues in by_where.items():
            lines.append(f"  in {where}{_file_part(issues)}:")
            for i in issues:
                lines.append(f"    line {i.line}: [{i.kind}] {i.description}")
        return "\n".join(lines)


def _file_part(issues: list[PurityIssue]) -> str:
    """`` (path/to/file.py)`` for a group of issues, or nothing."""
    name = next((i.filename for i in issues if i.filename), "")
    return f" ({name})" if name else ""


class _PurityVisitor(ast.NodeVisitor):
    """Single-function-body visitor that collects :class:`PurityIssue`s.

    Does NOT recurse into callees - the analyzer handles that. Just
    flags everything visible in this one body and records called
    names so the analyzer can resolve and recurse.
    """

    __slots__ = (
        "issues",
        "called_callable_nodes",
        "_param_names",
        "_qualname",
        "_line_offset",
        "read_names",
        "_assign_kinds",
        "_name_call_nodes",
        "_subscript_call_nodes",
        "_fresh_nodes",
        "_log_only",
        "_namespace",
        "_log_helpers",
    )

    def __init__(
        self,
        qualname: str,
        param_names: frozenset[str],
        line_offset: int = 0,
        fresh_nodes: frozenset[int] = frozenset(),
        log_only: frozenset[int] = frozenset(),
        namespace: dict[str, Any] | None = None,
        log_helpers: frozenset[str] = frozenset(),
    ) -> None:
        self.issues: list[PurityIssue] = []
        self.called_callable_nodes: list[ast.AST] = []
        #: Calls reported as known I/O (``requests.get``, ``open``). Not walked,
        #: but their bindings are noted, so a mock put in their place is seen.
        self.impure_call_nodes: list[ast.AST] = []
        #: The body opens a SQLite file itself (`sqlite3.connect(path)`), which
        #: the file tracker records as a read of that file: what a query on it
        #: returns IS in the key, so its reads are not advised on.
        self.opens_tracked_database = False
        #: Environment reads whose value the key folds (`environment_input`).
        self.environment_reads: set[tuple[str, str]] = set()
        # Bare names read (Load context) in this body - used to detect reads of
        # mutable module globals.
        self.read_names: set[str] = set()
        # For each simple ``name = ...`` target, the kinds of RHS it was ever
        # assigned ({"dynamic"} / {"other"} / both). A name assigned ONLY from a
        # dynamic source (getattr(obj,name), eval, importlib) and then CALLED is
        # untrackable dispatch obscured by a local; requiring "dynamic only"
        # keeps a name later reassigned to a safe value from false-positiving.
        self._assign_kinds: dict[str, set[str]] = {}
        # Calls whose function is a bare Name, checked against the taint set in
        # :meth:`finalize_taint` after the whole body is walked.
        self._name_call_nodes: list[ast.Call] = []
        # Calls whose function is a Subscript, judged in `finalize_taint`
        # once every local assignment in the body is known.
        self._subscript_call_nodes: list[ast.Call] = []
        self._param_names = param_names
        #: Loop variables over a parameter (``for r in rows``): changing one
        #: changes an element of the caller's object.
        self._param_elements: dict[str, str] = {}
        self._qualname = qualname
        # When source comes from inspect.getsource on a method, line
        # numbers in the parsed AST are 1-based relative to the
        # dedented source, not the original file. We just report them
        # as-is - the qualname tells the user where to look.
        self._line_offset = line_offset
        # Fresh locals: in-place mutation of these is pure (escape analysis).
        # The same question per POINT (`purity_flow.fresh_name_nodes`): ids of
        # the Name nodes that hold an object this function made, where read.
        self._fresh_nodes = fresh_nodes
        # Ambient reads (by node id) whose value reaches only a log line.
        self._log_only = log_only
        # What the body's names are bound to, so an aliased ambient read
        # (`_dt.datetime.now()`) is recognised (`_ambient_call`).
        self._namespace = namespace
        # The module's own log helpers (`_log_helper_names`): a call to one is
        # a print, not a call made for an effect a hit would skip.
        self._log_helpers = log_helpers

    # --- impure / dynamic / called-name detection on Call nodes ---

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if isinstance(node.ctx, ast.Load):
            self.read_names.add(node.id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if isinstance(node.func, ast.Name):
            self._name_call_nodes.append(node)
        self._record_call(node)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:  # noqa: N802
        """``os.environ["KEY"]`` -- the one ambient read that is not a call.

        Needed as its own visitor because the call rule cannot see it: this is
        a subscript on a mapping, and it is the form most people actually
        write. ``os.environ.get("KEY")`` goes through the call table instead.

        Load context only. ``os.environ["KEY"] = ...`` is a side effect rather
        than a frozen input, a different issue with a different fix.
        """
        env = environment_input(node)
        if env is not None and DECORATOR_POLICY[EffectKind.ENVIRONMENT] is Action.CACHE_AS_INPUT:
            if id(node) not in self._log_only:
                self.environment_reads.add(env)
        elif (
            isinstance(node.ctx, ast.Load)
            and get_base_name(node.value) in ENVIRON_NAMES
            and id(node) not in self._log_only
        ):
            self.issues.append(
                PurityIssue(
                    kind=ISSUE_AMBIENT_READ,
                    description=(
                        "os.environ[...] - reads ambient state, which is not in the "
                        "cache key, so the first call's value is frozen into every "
                        "later result"
                    ),
                    where=self._qualname,
                    line=getattr(node, "lineno", 0),
                )
            )
        self.generic_visit(node)

    @staticmethod
    def _is_dynamic_source(value: ast.AST) -> bool:
        """True when *value* resolves a callable from a runtime value.

        ``eval``/``exec``/``compile`` bound by name; ``getattr(obj, name)`` with
        a non-constant name; ``importlib.import_module(...)`` / ``__import__``.
        Assigning one of these to a local and calling it is untrackable dispatch.
        """
        if isinstance(value, ast.Name) and value.id in {"eval", "exec", "compile"}:
            return True
        if isinstance(value, ast.Call):
            f = value.func
            if (
                isinstance(f, ast.Name)
                and f.id == "getattr"
                and len(value.args) >= 2
                and not (isinstance(value.args[1], ast.Constant) and isinstance(value.args[1].value, str))
            ):
                return True
            if isinstance(f, ast.Attribute) and f.attr == "import_module":
                return True
            if isinstance(f, ast.Name) and f.id == "__import__":
                return True
        return False

    def finalize_taint(self) -> None:
        """Flag calling a local that was bound ONLY from a dynamic source.

        Run once after the whole body is walked (assignments may follow or
        precede the call in source order). A name is untrackable only if every
        assignment to it was dynamic -- so ``f = getattr(o,n); f()`` and
        ``ev = eval; ev(x)`` flag, but ``f = getattr(...); f = helper; f()``
        does not (it was rebound to a tracked value).
        """
        tainted = {n for n, kinds in self._assign_kinds.items() if kinds == {"dynamic"}}
        # A local bound only from a subscript is the temporary-variable
        # spelling of `TABLE[k]()`, and carries that rule's severity rather
        # than eval's: advisory, with `depends_on=` named as the remedy.
        looked_up = {n for n, kinds in self._assign_kinds.items() if kinds == {"lookup"}}
        for node in self._name_call_nodes:
            name = node.func.id  # type: ignore[attr-defined]
            if name in tainted:
                self.issues.append(
                    PurityIssue(
                        kind=ISSUE_UNTRACKABLE_DEP,
                        description=(
                            f"calls {name!r}, which was bound from a runtime value (dynamic dispatch through a local)"
                        ),
                        where=self._qualname,
                        line=getattr(node, "lineno", 0),
                    )
                )
            elif name in looked_up:
                self.issues.append(
                    PurityIssue(
                        kind=ISSUE_DYNAMIC_PATTERN,
                        description=(
                            f"calls {name!r}, which was looked up at runtime - "
                            f"editing the callable it names will not invalidate; "
                            f"name it with depends_on=[...]"
                        ),
                        where=self._qualname,
                        line=getattr(node, "lineno", 0),
                    )
                )

        for node in self._subscript_call_nodes:
            base = node.func.value  # type: ignore[attr-defined]
            if self._table_is_reachable_from_the_key(base):
                continue
            self.issues.append(
                PurityIssue(
                    kind=ISSUE_DYNAMIC_PATTERN,
                    description=(
                        f"calls {_describe_subscript(node.func)} - that table does "  # type: ignore[arg-type]
                        f"not reach the cache key, so editing the callable it holds "
                        f"will not invalidate; name it with depends_on=[...]"
                    ),
                    where=self._qualname,
                    line=getattr(node, "lineno", 0),
                )
            )

    def _table_is_reachable_from_the_key(self, base: ast.AST) -> bool:
        """True when cash already folds this dispatch table into the key.

        A module-level table -- ``MODELS[k]()``, ``mod.MODELS[k]()`` -- is read
        as a global, and hashing that global hashes the functions inside it.
        Measured: editing any function IN the table invalidates (even one never
        dispatched to), while editing one outside it does not. Warning there
        would tell the user to declare something cash already tracks, on the
        single most common dispatch idiom there is.

        A table that is a body-local, a parameter, an attribute of one, or a
        runtime namespace (``globals()[k]``, ``vars(mod)[k]``) never reaches the
        key. Those measurably serve STALE results, and are what this rule is
        for.
        """
        root = base
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Call):
            return False  # globals()[k], vars(mod)[k]
        if not isinstance(root, ast.Name):
            return False
        return not (root.id in self._param_names or root.id in self._assign_kinds)

    def _record_call(self, node: ast.Call) -> None:
        func_node = node.func
        line = getattr(node, "lineno", 0)

        # Explicit dynamism: eval / exec / compile by bare name.
        if isinstance(func_node, ast.Name) and func_node.id in {"eval", "exec", "compile"}:
            self.issues.append(
                PurityIssue(
                    kind=ISSUE_UNTRACKABLE_DEP,
                    description=f"{func_node.id}(...) - explicit dynamic execution",
                    where=self._qualname,
                    line=line,
                )
            )
            return

        # getattr(obj, name)(...) where name is not a constant string.
        if (
            isinstance(func_node, ast.Call)
            and isinstance(func_node.func, ast.Name)
            and func_node.func.id == "getattr"
            and len(func_node.args) >= 2
            and not (isinstance(func_node.args[1], ast.Constant) and isinstance(func_node.args[1].value, str))
        ):
            self.issues.append(
                PurityIssue(
                    kind=ISSUE_UNTRACKABLE_DEP,
                    description="getattr(obj, name)(...) with non-constant name - dynamic dispatch",
                    where=self._qualname,
                    line=line,
                )
            )
            return

        # Dynamic import: importlib.import_module(...) / __import__(...). The
        # imported module's members are resolved from a runtime value, so an
        # edit to that module is invisible to the cache key.
        _is_import_module = isinstance(func_node, ast.Attribute) and func_node.attr == "import_module"
        _is_dunder_import = isinstance(func_node, ast.Name) and func_node.id == "__import__"
        if _is_import_module or _is_dunder_import:
            self.issues.append(
                PurityIssue(
                    kind=ISSUE_UNTRACKABLE_DEP,
                    description=(
                        f"{'importlib.import_module' if _is_import_module else '__import__'}"
                        "(...) - dynamic import; the imported module's code is not tracked"
                    ),
                    where=self._qualname,
                    line=line,
                )
            )
            return

        # Calling a parameter -- `def f(cb): cb(x)` -- is NOT flagged.
        #
        # It used to be, and that warning outlived the mechanism that made it
        # true. A callable reaching a cached call as an argument is now hashed
        # by its SOURCE, so editing it invalidates: measured across a named
        # function, a lambda, a bound method, and even a helper called by the
        # passed function two levels down. Where cash genuinely cannot hash one
        # (`functools.partial`) it already says so precisely, at the argument
        # that failed, naming depends_on= / mark_opaque(). Warning here as well
        # would fire on every callback-taking function in the codebase to
        # report a hazard that no longer exists.

        # getattr(x, "exec")(...) -- a CONSTANT name, so the dynamic-dispatch
        # rule above does not fire, yet what it reaches is the very thing that
        # rule exists to stop. Measured: `getattr(builtins, "exec")("z = 5")`
        # executed arbitrary source in silence while a bare `exec(...)` raised.
        if (
            isinstance(func_node, ast.Call)
            and isinstance(func_node.func, ast.Name)
            and func_node.func.id == "getattr"
            and len(func_node.args) >= 2
            and isinstance(func_node.args[1], ast.Constant)
            and func_node.args[1].value in _DYNAMIC_BUILTIN_NAMES
        ):
            self.issues.append(
                PurityIssue(
                    kind=ISSUE_UNTRACKABLE_DEP,
                    description=(
                        f"getattr(..., {func_node.args[1].value!r})(...) - reaches {func_node.args[1].value} indirectly"
                    ),
                    where=self._qualname,
                    line=line,
                )
            )
            return

        # getattr(obj, "name")(...) with a constant identifier is obj.name(...)
        # spelled differently. Analysed as written it reached nothing: an edit
        # to the function it names was served stale, with no warning
        # (`getattr(helpers, "fun1")()`). Judged, and followed as a helper, as
        # the attribute call it is.
        if (
            isinstance(func_node, ast.Call)
            and isinstance(func_node.func, ast.Name)
            and func_node.func.id == "getattr"
            and len(func_node.args) == 2
            and not func_node.keywords
            and isinstance(func_node.args[1], ast.Constant)
            and isinstance(func_node.args[1].value, str)
            and func_node.args[1].value.isidentifier()
        ):
            spelled = ast.copy_location(
                ast.Call(
                    func=ast.copy_location(
                        ast.Attribute(value=func_node.args[0], attr=func_node.args[1].value, ctx=ast.Load()), func_node
                    ),
                    args=node.args,
                    keywords=node.keywords,
                ),
                node,
            )
            self._record_call(spelled)
            return

        # Calling whatever a subscript yields -- but ONLY when the table
        # itself cannot reach the cache key. Deferred to `finalize_taint`,
        # because whether the base is a body-local depends on assignments
        # that may appear after this call in source order.
        if isinstance(func_node, ast.Subscript):
            self._subscript_call_nodes.append(node)
            return

        # Calls with an effect (requests.post, os.system, df.to_csv, ...).
        func_name = get_call_name(func_node)
        module_name = get_call_module(func_node)
        if func_name:
            dotted = f"{module_name}.{func_name}" if module_name else func_name

            # Ambient reads (datetime.now, os.getenv, uuid4, ...). Only ever
            # matched DOTTED, or through what a name is bound to: every entry
            # carries its module, so a method named `now` on the user's own
            # object is not this.
            if DECORATOR_POLICY[EffectKind.ENVIRONMENT] is Action.CACHE_AS_INPUT:
                env = environment_input(node, self._namespace)
                if env is not None:
                    if id(node) not in self._log_only:
                        self.environment_reads.add(env)
                    return
            ambient = _ambient_call(node, self._namespace)
            if (
                ambient is not None
                and isinstance(func_node, ast.Name)
                and self._namespace
                and _clock_helper_read(self._namespace.get(func_node.id)) is not None
            ):
                # A clock helper is still the user's code: walked, so an edit
                # to it reaches the key like any helper's.
                self.called_callable_nodes.append(node)
            if ambient is not None and id(node) in self._log_only:
                return  # only ever printed or logged: cannot reach a result
            if ambient is not None:
                shown = ambient if ambient.endswith(")") else f"{ambient}()"
                self.issues.append(
                    PurityIssue(
                        kind=ISSUE_AMBIENT_READ,
                        description=(
                            f"{shown} - reads ambient state, which is not in the "
                            f"cache key, so the first call's value is frozen into "
                            f"every later result"
                        ),
                        where=self._qualname,
                        line=line,
                    )
                )
                return

            if is_log_line(node):
                return  # a diagnostic line: a hit skipping it is what caching means

            if _opens_tracked_database(func_node, self._namespace):
                self.opens_tracked_database = True
            effect = classify_call(node, self._namespace)
            if effect is not None and DECORATOR_POLICY[effect.kind] is Action.SUGGEST_TTL:
                self.issues.append(
                    PurityIssue(
                        kind=ISSUE_NETWORK_READ,
                        description=f"{dotted}() - what the {_SOURCE[effect.kind]} returns is not in the cache key",
                        where=self._qualname,
                        line=line,
                        effect_kind=effect.kind,
                    )
                )
                self.impure_call_nodes.append(node)
                return
            if (
                effect is not None
                and effect.kind not in _AMBIENT_KINDS
                and DECORATOR_POLICY[effect.kind] is Action.WARN
                and not effect.method
            ):
                what = (
                    "draws on pyplot's current figure, which a hit does not redraw"
                    if effect.kind is EffectKind.DISPLAY
                    else "known I/O / side-effecting"
                )
                self.issues.append(
                    PurityIssue(
                        kind=ISSUE_IMPURE_CALL,
                        description=f"{dotted}() - {what}",
                        where=self._qualname,
                        line=line,
                        effect_kind=effect.kind,
                    )
                )
                self.impure_call_nodes.append(node)
                return
            # A method with an effect on any receiver (to_csv, write, post,
            # execute, ...), or a container mutator. Skipped when the receiver
            # is a fresh local (``lines.append`` where ``lines = []``):
            # mutating a local accumulator is pure.
            reported_method = (
                effect is not None and effect.method and DECORATOR_POLICY[effect.kind] is Action.WARN
            ) or func_name in MUTATOR_METHODS
            if (
                isinstance(func_node, ast.Attribute)
                and reported_method
                and not self._receiver_is_fresh(func_node.value)
                and not self._is_module_function_named_like_a_mutator(func_node)
            ):
                base = get_base_name(func_node.value)
                base_str = f"{base}." if base else ""
                what = "write method"
                if func_node.attr in MUTATOR_METHODS:
                    # `rows.sort()` on a parameter changes the caller's list:
                    # say so, rather than the label a local's `.sort()` gets.
                    kind = self._mutation_kind(base, "method")
                    if kind != "method mutation":
                        what = kind
                self.issues.append(
                    PurityIssue(
                        kind=ISSUE_IMPURE_CALL,
                        description=f"{base_str}{func_node.attr}() - {what}",
                        where=self._qualname,
                        line=line,
                        effect_kind=effect.kind if effect is not None else None,
                    )
                )
                return

            # Pandas inplace=True kwarg - mutates the receiver.
            if (
                isinstance(func_node, ast.Attribute)
                and func_node.attr in PANDAS_INPLACE_METHODS
                and not self._receiver_is_fresh(func_node.value)
            ):
                for kw in node.keywords:
                    if kw.arg == "inplace" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        base = get_base_name(func_node.value)
                        base_str = f"{base}." if base else ""
                        self.issues.append(
                            PurityIssue(
                                kind=ISSUE_IMPURE_CALL,
                                description=f"{base_str}{func_node.attr}(inplace=True) - in-place mutation",
                                where=self._qualname,
                                line=line,
                            )
                        )
                        return

        # Not flagged as anything - record for recursion attempt.
        self.called_callable_nodes.append(node)

    #: Container mutators (`MUTATOR_METHODS`) called on a MODULE (`np.sort`,
    #: `np.append`, `np.insert`) return a new array and change nothing --
    #: users got "np.sort() - write method". A module's real writes
    #: (`np.save`, `plt.savefig`, `os.write`) keep being reported.

    def _reports_effect(self, call: ast.Call) -> bool:
        """Is *call* reported by the effect rule (`plt.plot(...)`, say)?"""
        effect = classify_call(call, self._namespace)
        return effect is not None and DECORATOR_POLICY[effect.kind] in (Action.WARN, Action.SUGGEST_TTL)

    def _is_module_function_named_like_a_mutator(self, func_node: ast.Attribute) -> bool:
        if func_node.attr not in MUTATOR_METHODS or not self._namespace:
            return False
        chain = _callee_chain(func_node.value)
        if not chain or chain[0] not in self._namespace:
            return False
        obj: Any = self._namespace[chain[0]]
        for attr in chain[1:]:
            if not isinstance(obj, types.ModuleType):
                return False
            obj = getattr(obj, attr, None)
        return isinstance(obj, types.ModuleType)

    # --- discarded-call detection on Expr statements ---

    def _discard_is_expected(self, func_node: ast.AST) -> bool:
        """A discarded call whose result nobody wants and whose effect a hit
        may skip: a log helper of the module's own, or `time.sleep` however it
        is spelled. The quickstart's own `time.sleep(5)` was reported
        as `discarded_call`, a label documented for calls made for an effect."""
        if isinstance(func_node, ast.Name) and func_node.id in self._log_helpers:
            return True
        chain = _callee_chain(func_node)
        if not chain or not self._namespace or chain[0] not in self._namespace:
            return False
        obj: Any = self._namespace[chain[0]]
        for attr in chain[1:]:
            if not isinstance(obj, (types.ModuleType, type)):
                return False
            obj = getattr(obj, attr, None)
        return obj is time.sleep

    def visit_Expr(self, node: ast.Expr) -> None:  # noqa: N802
        if isinstance(node.value, ast.Call) and (self._discard_is_expected(node.value.func) or is_log_line(node.value)):
            self.generic_visit(node)
            return
        if isinstance(node.value, ast.Call):
            call = node.value
            func_node = call.func
            # Plain bare-name call: foo(x)
            if isinstance(func_node, ast.Name):
                name = func_node.id
                # `print(...)` is already an impure_call; saying it twice, once
                # as a discarded return, was the same line counted two ways.
                if name not in KNOWN_PURE_BUILTINS and name not in _DISCARD_REPORTED_BUILTINS:
                    self.issues.append(
                        PurityIssue(
                            kind=ISSUE_DISCARDED_CALL,
                            description=f"discards return of {name}(...)",
                            where=self._qualname,
                            line=getattr(node, "lineno", 0),
                        )
                    )
            elif isinstance(func_node, ast.Attribute):
                method = func_node.attr
                # If the method is already flagged elsewhere (write-methods,
                # impure module calls), it's recorded by _record_call. The
                # discarded-return flag here adds nothing useful - skip to
                # avoid double-counting. Skip known-pure idioms too.
                if (
                    method not in REPORTED_METHODS
                    and method not in PANDAS_INPLACE_METHODS
                    and not self._reports_effect(call)
                ):
                    base = get_base_name(func_node.value)
                    base_str = f"{base}." if base else ""
                    self.issues.append(
                        PurityIssue(
                            kind=ISSUE_DISCARDED_CALL,
                            description=f"discards return of {base_str}{method}(...)",
                            where=self._qualname,
                            line=getattr(node, "lineno", 0),
                        )
                    )
        self.generic_visit(node)

    # --- scope mutation detection ---

    def visit_Global(self, node: ast.Global) -> None:  # noqa: N802
        for name in node.names:
            self.issues.append(
                PurityIssue(
                    kind=ISSUE_SCOPE_MUTATION,
                    description=f"global {name} - reads/writes module-level state",
                    where=self._qualname,
                    line=getattr(node, "lineno", 0),
                )
            )
        self.generic_visit(node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:  # noqa: N802
        for name in node.names:
            self.issues.append(
                PurityIssue(
                    kind=ISSUE_SCOPE_MUTATION,
                    description=f"nonlocal {name} - writes enclosing-scope state",
                    where=self._qualname,
                    line=getattr(node, "lineno", 0),
                )
            )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        for target in node.targets:
            self._maybe_flag_mutation_target(target, node.lineno)
        # Record the RHS kind for a simple ``name = ...`` so a dynamically-bound
        # local that is later called can be flagged (see finalize_taint).
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if self._is_dynamic_source(node.value):
                kind = "dynamic"
            elif isinstance(node.value, ast.Subscript):
                # `cls = REGISTRY[k]; cls()` is the same runtime lookup as
                # `REGISTRY[k]()` one line apart, and must not be punished
                # harder for using a temporary: it warns, it does not raise.
                kind = "lookup"
            else:
                kind = "other"
            self._assign_kinds.setdefault(node.targets[0].id, set()).add(kind)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:  # noqa: N802
        self._maybe_flag_mutation_target(node.target, node.lineno)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:  # noqa: N802
        for target in node.targets:
            self._maybe_flag_mutation_target(target, node.lineno)
        self.generic_visit(node)

    def _receiver_is_fresh(self, value: ast.AST) -> bool:
        """True when *value* holds an object this function made -- mutating
        it in place is pure (escape analysis, `purity_flow`).

        Whatever the flow pass shows is fresh at this point: a local bound to
        a new object (``rows = []``), a view of one (``inner = u[1:-1]``), a
        name rebound to a copy (``df = df.merge(...)``), an unpacked fresh
        literal. An ELEMENT of a fresh container is not: ``d["k"]`` may be
        anyone's object.
        """
        return receiver_is_fresh(value, self._fresh_nodes)

    def _maybe_flag_mutation_target(self, target: ast.AST, line: int) -> None:
        # Mutating a fresh local (``pos[i] = ...`` where ``pos = np.zeros(n)``)
        # is pure: the object can't reach caller-visible state and returning it
        # just hands the caller a new object. Skip those.
        if isinstance(target, (ast.Attribute, ast.Subscript)) and self._receiver_is_fresh(target.value):
            return
        if isinstance(target, ast.Attribute):
            base = get_base_name(target.value)
            base_str = f"{base}." if base else ""
            self.issues.append(
                PurityIssue(
                    kind=ISSUE_SCOPE_MUTATION,
                    description=f"{base_str}{target.attr} = ... - " + self._mutation_kind(base, "attribute"),
                    where=self._qualname,
                    line=line,
                )
            )
        elif isinstance(target, ast.Subscript):
            base = get_base_name(target.value)
            base_str = f"{base}[...]" if base else "[...]"
            self.issues.append(
                PurityIssue(
                    kind=ISSUE_SCOPE_MUTATION,
                    description=f"{base_str} = ... - " + self._mutation_kind(base, "subscript"),
                    where=self._qualname,
                    line=line,
                )
            )

    def _mutation_kind(self, base: str | None, kind: str) -> str:
        """Name what a mutation of *base* does. For a PARAMETER that is not
        "a scope mutation": it changes the CALLER's object, on a miss only -- a
        hit returns the stored result and the caller's object stays as it was,
        so everything downstream of the call sees two different objects
        depending on whether it hit."""
        root = (base or "").split(".")[0].split("[")[0]
        if root and root in self._param_names:
            return (
                f"{kind} mutation that changes the argument '{root}' in place; "
                f"a cache hit would not make that change, so a call that makes "
                f"it is not stored and runs every time"
            )
        if root and root in self._param_elements:
            return (
                f"{kind} mutation that changes an element of the argument "
                f"'{self._param_elements[root]}' in place; a cache hit would not "
                f"make that change, so a call that makes it is not stored and "
                f"runs every time"
            )
        return f"{kind} mutation"

    def visit_For(self, node: ast.For) -> None:
        """Note loop variables over a parameter, then walk the loop as usual."""
        iterable, target = node.iter, node.target
        if (
            isinstance(iterable, ast.Call)
            and isinstance(iterable.func, ast.Name)
            and iterable.func.id == "enumerate"
            and iterable.args
            and isinstance(target, ast.Tuple)
            and len(target.elts) == 2
        ):
            iterable, target = iterable.args[0], target.elts[1]
        if isinstance(iterable, ast.Name) and iterable.id in self._param_names and isinstance(target, ast.Name):
            self._param_elements[target.id] = iterable.id
        self.generic_visit(node)

    visit_AsyncFor = visit_For


def _defining_module(obj: Any) -> Any:
    """The module *obj*'s code was written in.

    For a function, its ``__globals__`` say so. ``__module__`` does not
    always: ``functools.wraps`` copies the WRAPPED function's ``__module__``
    onto the wrapper, so a library's wrapper (tenacity's, torch's) claimed to
    be user code and a user's wrapper claimed to be the helper's module.
    """
    if isinstance(obj, types.FunctionType):
        name = obj.__globals__.get("__name__")
        module = sys.modules.get(name) if isinstance(name, str) else None
        if module is not None:
            return module
    return inspect.getmodule(obj)


def own_code_is_user(obj: Any, root_module: str | None) -> bool:
    """`_is_user_code`, for callables that may be wrappers."""
    try:
        return _is_user_code(obj, root_module)
    except Exception:  # noqa: BLE001 - a probe of arbitrary objects
        return False


#: Upper bounds on `callable_layers`, per callable: a wrong answer is worth a
#: few more objects visited, a cycle through a registry is not.
_LAYER_DEPTH = 6
_LAYER_COUNT = 32


def _function_like(value: Any) -> bool:
    return isinstance(value, (types.FunctionType, types.MethodType, functools.partial)) or (
        callable(value)
        and not isinstance(value, (type, types.ModuleType, types.BuiltinFunctionType))
        and hasattr(value, "__wrapped__")
    )


def callable_layers(obj: Any) -> list[Any]:
    """The functions *obj* will run besides its own code, outermost first.

    A decorated helper is two or more functions, and the key has to see all of
    them: with ``functools.wraps`` only the wrapped function was followed, so
    an edit to the wrapper's body was served stale; without it, only the
    wrapper was, so an edit to the wrapped function was. Followed:

    * ``__wrapped__`` (``functools.wraps``, ``update_wrapper``, ``lru_cache``);
    * function-valued closure cells (a wrapper written without ``wraps``, and
      the ``decorator`` package, which keeps the caller in a closure);
    * a bound method's ``__func__``, a ``functools.partial``'s ``func``;
    * a callable instance's class ``__call__`` and its function-valued
      attributes (``np.vectorize.pyfunc``, a class-based decorator's
      ``self.fn``, ``toolz.curry``'s partial, wrapt's ``_self_wrapper``);
    * a function's own ``__dict__`` values and mappings of functions
      (``functools.singledispatch``'s ``registry``).

    Returns FUNCTION objects only, deduplicated, never *obj* itself; whether
    each is user code is the caller's decision. Bounded in depth and count.
    """
    found: list[Any] = []
    seen: set[int] = {id(obj)}

    def add(value: Any, depth: int) -> None:
        if len(found) >= _LAYER_COUNT or id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, types.FunctionType):
            found.append(value)
        expand(value, depth + 1)

    def candidates(value: Any):
        if isinstance(value, types.MethodType):
            yield value.__func__
            return
        if isinstance(value, functools.partial):
            yield value.func
            return
        wrapped = getattr(value, "__wrapped__", None) if not isinstance(value, type) else None
        if wrapped is not None:
            yield wrapped
        # wrapt's proxies forward `__class__`, so one passes for a plain
        # function below; the user's wrapper function sits here.
        wrapper = getattr(value, "_self_wrapper", None) if not isinstance(value, type) else None
        if wrapper is not None:
            yield wrapper
        if isinstance(value, types.FunctionType):
            for cell in value.__closure__ or ():
                try:
                    inner = cell.cell_contents
                except ValueError:
                    continue
                if _function_like(inner):
                    yield inner
            attrs = getattr(value, "__dict__", None) or {}
        else:
            if callable(value) and not isinstance(value, (type, types.ModuleType)):
                call = getattr(type(value), "__call__", None)
                if isinstance(call, types.FunctionType):
                    yield call
            try:
                attrs = dict(vars(value))
            except TypeError:
                attrs = {}
        for key, attr in list(attrs.items()):
            if key == "__wrapped__":
                continue
            if _function_like(attr):
                yield attr
            elif isinstance(attr, (dict, types.MappingProxyType)):
                for item in list(attr.values())[:_LAYER_COUNT]:
                    if _function_like(item):
                        yield item

    def expand(value: Any, depth: int) -> None:
        if depth > _LAYER_DEPTH:
            return
        try:
            for candidate in candidates(value):
                add(candidate, depth)
        except Exception:  # noqa: BLE001 - arbitrary objects; best effort
            return

    expand(obj, 0)
    return found


def _is_user_code(callee: Any, root_module: str | None) -> bool:
    """Decide whether to recurse into *callee* during purity analysis.

    The boundary rule from the design discussion: user code is
    anything that (a) shares the cached function's top-level package
    OR (b) lives outside stdlib/site-packages.

    Args:
        callee: Resolved callable from the cached function's globals.
        root_module: ``__module__`` of the cached function. Used for
            the same-top-level-package shortcut.

    Returns:
        True when the analyzer should attempt to read source and
        recurse into *callee*. False for library code we trust
        unless explicitly marked stateful.
    """
    module = _defining_module(callee)
    if module is None:
        return False

    # Top-level package shortcut - handles common case fast and
    # works for editable installs (both pieces share the same
    # top-level package by construction).
    callee_mod = getattr(module, "__name__", "") or ""
    if root_module:
        root_top = root_module.split(".", 1)[0]
        callee_top = callee_mod.split(".", 1)[0]
        if root_top and callee_top and root_top == callee_top:
            return True

    # Never analyse cash's own code on a user's behalf. Under ``%cash_on``
    # the file tracker replaces ``open`` and the pandas readers with cash
    # shims, so a user function that reads a file resolves its callee to
    # ``cash.tracking.file_tracker``. In a NORMAL install that lands in
    # site-packages and the fallback below rejects it; in an EDITABLE
    # install it does not, so the analyzer walked the shim and reported
    # cash's own ``_tracker._track_path(...)`` as the user's side effect.
    # The report is about the user's function, and cash's instrumentation
    # is never part of it. Placed after the shortcut above so cash
    # analysing its own functions still recurses.
    if callee_mod == "cash" or callee_mod.startswith("cash."):
        return False

    # Fall back to the file-path-based check used by the notebook
    # subsystem. Catches editable installs that DON'T share the
    # cached function's package (e.g. user's project depends on a
    # locally-developed sibling lib also installed `-e`).
    try:
        return is_local_module(module)
    except (TypeError, AttributeError):
        return False


def local_import_map(func_def: ast.AST, func: Any) -> dict[str, tuple[str, tuple[str, ...]]]:
    """``local name -> (module, attribute prefix)`` for imports in a function body.

    ``from helpmod import scale`` inside the body binds a LOCAL, so the helper
    walk, which resolves names in the module's globals, found nothing and an
    edit to ``scale`` was served stale -- in the common shape of an
    import moved into the function to break an import cycle. Each import runs
    on every call, so the binding it makes is ``helpmod.scale`` as the module
    holds it at call time: that is the path recorded for the per-call check.
    """
    package = None
    g = getattr(func, "__globals__", None)
    if isinstance(g, dict):
        package = g.get("__package__")
    if package is None:
        package = (getattr(func, "__module__", "") or "").rpartition(".")[0]
    found: dict[str, tuple[str, tuple[str, ...]]] = {}
    for node in ast.walk(func_def):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    found[alias.asname] = (alias.name, ())
                else:
                    top = alias.name.split(".")[0]
                    found[top] = (top, ())
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                try:
                    module = importlib.util.resolve_name("." * node.level + (node.module or ""), package or None)
                except (ImportError, ValueError):
                    continue
            else:
                module = node.module or ""
            if not module:
                continue
            for alias in node.names:
                if alias.name != "*":
                    found[alias.asname or alias.name] = (module, (alias.name,))
    return found


def _module_is_user_code(module_name: str, root_module: str | None) -> bool:
    """Is *module_name* user code, decided WITHOUT importing it?

    The top-level package is checked first, with a spec lookup that imports
    nothing, so a library imported inside a function to defer its cost
    (``import torch``) is never imported early on its behalf.
    """
    top = module_name.split(".")[0]
    if root_module and root_module.split(".")[0] == top:
        return True
    try:
        spec = importlib.util.find_spec(top)
    except (ImportError, ValueError):
        return False
    origin = getattr(spec, "origin", None) if spec is not None else None
    if not origin or origin in ("built-in", "frozen"):
        return False
    return is_local_module(types.SimpleNamespace(__file__=origin))


def resolve_local_import(module_name: str, prefix: tuple[str, ...], root_module: str | None) -> Any:
    """The object a function-body import binds, importing a USER module if the
    body has not run yet. That import is the one the body is about to make;
    doing it now is what lets the first call's key see the helper. A library
    module is only read if it is already loaded."""
    module = sys.modules.get(module_name)
    if module is None:
        if not _module_is_user_code(module_name, root_module):
            return None
        try:
            module = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001 - the body will raise it, not the analysis
            return None
    obj: Any = module
    for attr in prefix:
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    return obj


def _callee_chain(node: ast.AST) -> tuple[str, ...] | None:
    """The name chain a call site uses: ``_sieve`` -> ``("_sieve",)``,
    ``mod.sub.f`` -> ``("mod", "sub", "f")``; None for anything else."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None
    parts.append(cur.id)
    return tuple(reversed(parts))


def _ambient_call(node: ast.Call, namespace: dict[str, Any] | None) -> str | None:
    """The ambient read *node* makes, spelled canonically, or None.

    The spelling in the source first (``datetime.now()``), then what its names
    are bound to in *namespace* (:func:`cash.effects.classify_call`): users
    wrote ``import datetime as _dt; _dt.datetime.now()``, ``from datetime
    import datetime as DateTime``, ``import time as _time``, ``import os as
    _os`` and ``pd.Timestamp.now()`` freezing a timestamp with no warning,
    while the canonical spellings warned. Also ``pd.to_datetime("today")`` and
    ``pd.Timestamp("now")``.
    """
    effect = classify_call(node, namespace)
    if effect is not None and effect.kind in _AMBIENT_KINDS:
        if effect.name in CLOCK_WHEN_ARG_CALLS:
            return f"{effect.name}({node.args[0].value!r})"  # type: ignore[attr-defined]
        return effect.name
    chain = _callee_chain(node.func)
    if namespace and chain and len(chain) == 1 and chain[0] in namespace:
        inner = _clock_helper_read(namespace[chain[0]])
        if inner is not None:
            shown = inner if inner.endswith(")") else f"{inner}()"
            return f"{chain[0]}() (which returns {shown})"
    return None


def _clock_helper_read(value: Any) -> str | None:
    """The ambient read a CLOCK HELPER returns, or None.

    A clock helper is a function of the user's whose body is log lines and one
    ``return <ambient read>``: ``def mark(name): print(..., file=sys.stderr);
    return time.perf_counter()``. Calling it IS the ambient read, so it is
    judged where it is called -- where ``t0 = mark("step")`` handed only to a
    ``done(name, t0)`` that prints it cannot reach a result.
    Inside the helper it is not
    reported at all when the helper is reached from a cached function.
    """
    code = getattr(value, "__code__", None)
    if not isinstance(value, types.FunctionType) or code is None:
        return None
    if code in _CLOCK_HELPER_CACHE:
        return _CLOCK_HELPER_CACHE[code]
    _CLOCK_HELPER_CACHE[code] = None  # a helper that calls itself
    found = None
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(value)))
        func_def = tree.body[0] if tree.body else None
        body = list(getattr(func_def, "body", []))
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        if (
            body
            and isinstance(body[-1], ast.Return)
            and isinstance(body[-1].value, ast.Call)
            and all(
                isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and is_log_line(s.value) for s in body[:-1]
            )
        ):
            namespace = _build_namespace(value)
            # A read the key folds (`environment_input`) is an input, not a
            # frozen value: the helper's own walk lists it.
            if environment_input(body[-1].value, namespace) is None:
                found = _ambient_call(body[-1].value, namespace)
    except SOURCE_RETRIEVAL_ERRORS + (SyntaxError, ValueError):
        found = None
    if len(_CLOCK_HELPER_CACHE) >= 4096:
        _CLOCK_HELPER_CACHE.clear()
    _CLOCK_HELPER_CACHE[code] = found
    return found


#: code object -> the ambient read that clock helper returns, or None.
_CLOCK_HELPER_CACHE: dict[Any, str | None] = {}


def is_mock(obj: Any) -> bool:
    """A ``unittest.mock`` object (``pytest-mock`` uses the same classes).

    Checked before anything reads an attribute from a callee: a mock answers
    every attribute truthily, so ``_cash_cached`` or a purity marker would
    read as set. Never imports ``unittest.mock`` itself.
    """
    module = sys.modules.get("unittest.mock")
    if module is None:
        return False
    if isinstance(obj, module.NonCallableMock):
        return True
    # `create_autospec` / `patch(..., autospec=True)` on a function makes a
    # real function that carries its mock. Walked as code, it led into the
    # TEST's side_effect, analysed as production code -- an `__import__` in a
    # fake raised CashImpureFunctionError out of the test.
    return isinstance(obj, types.FunctionType) and isinstance(obj.__dict__.get("mock"), module.NonCallableMock)


def _binding_path(caller: Any, chain: tuple[str, ...] | None) -> tuple[str, tuple[str, ...]] | None:
    """``(module_name, chain)`` when *chain* starts at a name *caller* looks up
    in its module's globals, so ``sys.modules[module_name]`` + the chain finds
    what the call site finds. None for a closure cell or a namespace that is
    not a registered module (``exec``, a class body)."""
    if not chain:
        return None
    code = getattr(caller, "__code__", None)
    if code is not None and chain[0] in (getattr(code, "co_freevars", ()) or ()):
        return None
    module_name = getattr(caller, "__module__", None)
    module = sys.modules.get(module_name or "")
    if module is None or getattr(module, "__dict__", None) is not getattr(caller, "__globals__", None):
        return None
    return module_name, chain


def _ref(obj: Any) -> Callable[[], Any]:
    """A weak reference where the object allows one, a strong one otherwise."""
    try:
        return weakref.ref(obj)
    except TypeError:
        return lambda: obj


_UNRESOLVED = object()


def resolve_binding(module_name: str, chain: tuple[str, ...]) -> Any:
    """What ``sys.modules[module_name]`` + *chain* holds now, or ``_UNRESOLVED``."""
    obj: Any = sys.modules.get(module_name)
    if obj is None:
        return _UNRESOLVED
    for attr in chain:
        obj = getattr(obj, attr, _UNRESOLVED)
        if obj is _UNRESOLVED:
            return _UNRESOLVED
    return obj


def bindings_changed(report: PurityReport) -> bool:
    """Does any call-site binding the report followed hold a different object now?

    A module that has left ``sys.modules`` proves nothing either way and is
    skipped. Identity, not equality: a re-created function with the same
    code is still a different object, whose globals may differ.
    """
    for module_name, chain, ref in report.helper_bindings:
        live = resolve_binding(module_name, chain)
        if live is _UNRESOLVED:
            continue
        if live is not ref():
            return True
    return False


def _called_names_in_tree(tree: ast.AST) -> list[str]:
    """Bare names called anywhere in *tree*, including inside lambdas.

    Used for a CLASS body, where the interesting call can sit inside a
    default-factory lambda: ``field(default_factory=lambda: B(0))``. Only
    ``ast.Name`` callees -- an attribute call (``mod.f()``) is resolved by
    the ordinary callee machinery, not here.
    """
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.append(node.func.id)
    return names


def _class_namespaces(cls: type) -> list[dict[str, Any]]:
    """Every globals dict in which *cls*'s body names might resolve.

    Returns a LIST, and callers must try all of them, because no single one
    is reliably right:

    * The defining module is the obvious candidate, but a class built by
      ``exec`` into a namespace that never reaches ``sys.modules`` (notebook
      cell, REPL) has none.
    * A member's ``__globals__`` covers that case -- but picking the FIRST
      member with one is wrong. A dataclass's GENERATED methods carry the
      ``dataclasses`` machinery's globals, not the user's, and whether such
      a method comes first in ``vars(cls)`` varies by Python version. On
      3.13 it does, so ``B`` in ``field(default_factory=lambda: B(10))``
      resolved against the wrong namespace and the class was never folded;
      on 3.14 it happened to work. CI caught it on all three 3.13 runners.

    Ordering is best-first (module, then member globals), but correctness
    does not depend on it -- the caller searches until a name resolves.
    """
    namespaces: list[dict[str, Any]] = []
    seen: set[int] = set()

    def add(candidate: Any) -> None:
        if isinstance(candidate, dict) and id(candidate) not in seen:
            seen.add(id(candidate))
            namespaces.append(candidate)

    module = sys.modules.get(getattr(cls, "__module__", "") or "")
    add(getattr(module, "__dict__", None))

    members: list[Any] = list(vars(cls).values())
    fields_map = getattr(cls, "__dataclass_fields__", None)
    if isinstance(fields_map, dict):
        for fld in fields_map.values():
            factory = getattr(fld, "default_factory", None)
            if factory is not None and factory is not dataclasses.MISSING:
                # Put field factories FIRST among members: a factory lambda is
                # written in the user's module, while a generated __init__ is
                # not, so it is the more likely place for the name to resolve.
                members.insert(0, factory)
    for member in members:
        if isinstance(member, (classmethod, staticmethod)):
            member = member.__func__
        add(getattr(member, "__globals__", None))
    return namespaces


def _resolve_in_class_namespaces(cls: type, name: str) -> Any:
    """First binding of *name* across *cls*'s candidate namespaces, else None."""
    for namespace in _class_namespaces(cls):
        if name in namespace:
            return namespace[name]
    return None


def _build_namespace(func: Callable[..., Any]) -> dict[str, Any]:
    """Return a merged ``__globals__`` + closure-cell namespace for *func*.

    Lets :func:`~cash.analysis.ast_util.resolve_callee` see helpers defined as closures
    (nested function definitions) - not just module-level names.
    Without this, a ``@cash.cache``d function inside another
    function couldn't recurse into its sibling helpers, and any
    impurity those helpers contained would be missed.

    Closure cells are pulled from ``func.__code__.co_freevars`` paired
    with ``func.__closure__``. An empty cell (rare - happens when a
    closure variable is never assigned) is silently skipped.

    The returned dict is a shallow copy of ``__globals__`` with
    closure entries layered on top - same-name closure variables
    shadow globals, matching Python's normal scoping.
    """
    ns = dict(getattr(func, "__globals__", None) or {})
    code = getattr(func, "__code__", None)
    closure = getattr(func, "__closure__", None) or ()
    if code is not None and closure:
        freevars = getattr(code, "co_freevars", ()) or ()
        for name, cell in zip(freevars, closure):
            try:
                ns[name] = cell.cell_contents
            except ValueError:
                # Cell exists but has no value yet (forward reference
                # in mutually-recursive closures). Skip.
                continue
    return ns


class PurityAnalyzer:
    """Walks a callable's body + its module-bounded helpers and
    returns a :class:`PurityReport`.

    Results are cached by the analyzed callable's source hash.
    Multiple :class:`Cash` instances share a single process-wide
    analyzer via :func:`get_analyzer`.
    """

    _MAX_DEPTH = 6  # Defensive depth cap. Real call hierarchies
    # rarely exceed 3-4 levels before hitting library code; the cap
    # bounds pathological cases (recursive helpers that resolve
    # through different code paths).

    _CACHE_SIZE_LIMIT = 500

    def __init__(self) -> None:
        self._cache: dict[str, PurityReport] = {}
        self._cache_lock = threading.Lock()

    def analyze(self, func: Callable[..., Any]) -> PurityReport:
        """Return a :class:`PurityReport` for *func*.

        Idempotent and cached by source hash. A function marked ``@pure`` or
        ``@stateful`` is walked for the cache key like any other, but its body
        is not audited: ``@pure`` reports nothing, ``@stateful`` reports the
        function itself.
        """
        source_hash = _try_source_hash(func)
        if source_hash is not None:
            # Keyed by the namespace the names resolve in as well as the text:
            # `def run(): return step()` written identically in two modules
            # calls two different `step`s, and sharing one report handed the
            # second module the first one's helpers -- editing its own `step`
            # then changed nothing its key could see.
            source_hash = f"{source_hash}:{id(getattr(func, '__globals__', None))}"
            with self._cache_lock:
                cached = self._cache.get(source_hash)
            # The source is the same, but a name it calls through may hold a
            # different object now (a patched helper, or a real one restored):
            # the tree below that binding is not the one this report walked.
            if cached is not None and not bindings_changed(cached):
                return cached

        report = self._analyze_uncached(func)
        if is_stateful(func):
            # The user has spoken: one finding for the function itself.
            report = dataclasses.replace(
                report,
                issues=(
                    PurityIssue(
                        kind=ISSUE_IMPURE_CALL,
                        description="explicitly marked @stateful",
                        where=_qualname_of(func),
                        line=0,
                    ),
                ),
            )

        if source_hash is not None:
            with self._cache_lock:
                if len(self._cache) >= self._CACHE_SIZE_LIMIT:
                    # Drop the oldest entry; insertion-order dict.
                    oldest = next(iter(self._cache))
                    del self._cache[oldest]
                self._cache[source_hash] = report
        return report

    def _analyze_uncached(self, root_func: Callable[..., Any]) -> PurityReport:
        root_module = getattr(root_func, "__module__", None)

        all_issues: list[PurityIssue] = []
        helper_hashes: dict[str, str] = {}
        helper_paths: dict[str, tuple[str, tuple[str, ...]]] = {}
        helper_objects: dict[str, Any] = {}
        opaque: list[str] = []
        visited: set[str] = set()
        bindings: list[tuple[str, tuple[str, ...], Any]] = []
        seen_bindings: set[tuple[str, tuple[str, ...]]] = set()
        unkeyable: list[str] = []
        # id(callee) -> the first call-site binding that reached it
        caller_paths: dict[int, tuple[str, tuple[str, ...]]] = {}
        waived_paths: set[tuple[str, tuple[str, ...]]] = set()
        unwaived_paths: set[tuple[str, tuple[str, ...]]] = set()
        environment_reads: set[tuple[str, str]] = set()

        def _note_binding(callee: Any, path: tuple[str, tuple[str, ...]] | None) -> None:
            if path is None or path in seen_bindings:
                return
            seen_bindings.add(path)
            bindings.append((path[0], path[1], _ref(callee)))

        def _record_resolution_path(func: Callable[..., Any], qualname: str) -> None:
            """Note where to re-resolve *func* from ``sys.modules`` per call.

            Lets the decorator pick up in-process redefinitions (notebook
            cells, REPL) and rebinding (``monkeypatch``, ``mock.patch``).
            Preferably through the name the CALLER uses; otherwise the
            helper's own ``__qualname__`` split on '.', so methods like
            ``Klass.method`` resolve. The root function is skipped -- it is
            not a "helper" and the decorator holds its own reference.
            """
            path = caller_paths.get(id(func))
            if func is not root_func and path is not None and resolve_binding(*path) is func:
                helper_paths[qualname] = path
                return
            helper_module = getattr(func, "__module__", None)
            helper_inner_qualname = getattr(func, "__qualname__", None)
            home = (
                (helper_module, tuple(helper_inner_qualname.split(".")))
                if helper_module and helper_inner_qualname and "<locals>" not in helper_inner_qualname
                else None
            )
            # The home must lead back to THIS object. A function wrapped by a
            # decorator does not: its module name now holds the wrapper, so a
            # per-call re-resolution hashed the wrapper in its place and the
            # wrapped function's own edits never reached the key.
            if home is not None and resolve_binding(*home) is not func:
                home = None
            if func is not root_func and home is not None:
                helper_paths[qualname] = home
            elif func is not root_func:
                try:
                    helper_objects[qualname] = weakref.ref(func)
                except TypeError:  # not weak-referenceable
                    pass

        # The third element is HASH_ONLY: fold this callable's code into the
        # cache key, but do not analyze it for purity. Set for classes reached
        # from another class's body -- they are followed for correctness, not
        # audited, and analyzing them reports every ``self.x = x`` in an
        # ordinary __init__ as a scope mutation.
        def _queue_hash_only(target: Any, owner: Any, depth: int) -> None:
            if target is None or target is owner or not callable(target):
                return
            if getattr(target, "_cash_cached", False):
                return
            if not _is_user_code(target, root_module):
                return
            # ``reported`` is the entry being walked when this runs.
            stack.append((target, depth + 1, True, reported))

        def _queue_annotation_refs(obj: Any, depth: int) -> None:
            """Queue, hash-only, the user classes and functions *obj*'s
            annotations name (see ``cash._annotation_refs``): pydantic runs a
            field type's validators, a ``get_type_hints`` builder constructs it."""
            if depth >= self._MAX_DEPTH:
                return
            for target in annotation_referents(obj, lambda o: _is_user_code(o, root_module)):
                _queue_hash_only(target, obj, depth)

        def _queue_class_refs(cls: Any, tree: ast.AST, depth: int) -> None:
            """Queue user-code objects a CLASS body constructs, hash-only."""
            if not isinstance(cls, type) or depth >= self._MAX_DEPTH:
                return
            for called in _called_names_in_tree(tree):
                _queue_hash_only(_resolve_in_class_namespaces(cls, called), cls, depth)
            _queue_annotation_refs(cls, depth)

        # Each entry is (callable, depth, hash_only, reported). Every entry is
        # walked for the cache key. ``reported`` is False below a callable marked
        # ``@pure`` or ``@stateful``: the marker settles what it and everything
        # it calls may do, so their findings are not reported -- but their code
        # still decides the result, so it is keyed like any other helper's.
        root_reported = not (is_pure(root_func) or is_stateful(root_func))
        stack: list[tuple[Callable[..., Any], int, bool, bool]] = [(root_func, 0, False, root_reported)]
        if (
            isinstance(root_func, types.FunctionType)
            and hasattr(root_func, "__wrapped__")
            and not own_code_is_user(root_func, root_module)
        ):
            # `@cash.cache` over a LIBRARY decorator (`@retry(...)`,
            # `@torch.no_grad()`): the wrapper's own body is someone else's
            # code, so start from the user functions it runs instead.
            starts = [
                (layer, 0, False, root_reported)
                for layer in callable_layers(root_func)
                if own_code_is_user(layer, root_module)
            ]
            if starts:
                stack = starts
        # id -> whether that walk reported findings, and the name it took.
        visited_ids: dict[int, bool] = {}
        walked_names: dict[int, str] = {}
        while stack:
            func, depth, hash_only, reported = stack.pop()
            reported = reported and not (is_pure(func) or is_stateful(func))
            walked_reported = visited_ids.get(id(func))
            if walked_reported is not None and (walked_reported or not reported):
                continue
            visited_ids[id(func)] = reported
            if walked_reported is not None:
                # Walked below a marker first, reached now from an unmarked
                # caller too: walk it again so its findings are reported. Its
                # key part is the same, under the same name.
                qualname = walked_names[id(func)]
            else:
                qualname = _qualname_of(func)
                # Visited by OBJECT: a library wrapper can copy the name of the
                # function it wraps (`toolz.curry`, `np.vectorize`), and visiting
                # by name walked only whichever of the two came first. A second
                # object under a name already taken gets a numbered one, in walk
                # order, which is the same in every process.
                if qualname in visited:
                    n = 2
                    while f"{qualname}#{n}" in visited:
                        n += 1
                    qualname = f"{qualname}#{n}"
                visited.add(qualname)
                walked_names[id(func)] = qualname

            # Read source. Failure -> opaque leaf for PURITY: we cannot see
            # what it does, so we decline to judge it.
            #
            # It must NOT become invisible to the CACHE KEY as well, which is
            # what dropping it outright used to do. A helper whose source
            # cannot be read contributed nothing to its callers' state hash,
            # so ANY edit to it went unnoticed -- not merely a constant.
            # Measured with an exec-defined helper under a filename absent
            # from linecache: replacing its entire body still served the
            # stale result. Fall back to the compiled identity, which is
            # exactly what ``Cash._hash_callable_source`` recomputes live for
            # the same object, so the snapshot and the per-call value agree
            # instead of disagreeing forever.
            try:
                src = own_source(func)
            except SOURCE_RETRIEVAL_ERRORS:
                if reported:
                    opaque.append(qualname)
                helper_hashes[qualname] = compiled_identity(func)
                _record_resolution_path(func, qualname)
                continue
            src = textwrap.dedent(src)

            # Hash the NORMALIZED source for cache-key invalidation of
            # helpers. Root function's hash is captured separately by the
            # decorator via _hash_callable_source - we record all walked
            # callables here so the decorator can fold them into the state
            # hash uniformly. Normalizing means a comment or reformat in a
            # helper no longer invalidates its callers, which was the more
            # surprising half of the old behaviour: users expect editing a
            # function to recompute it, not editing something it calls.
            # `callable_identity`, the digest the live check recomputes: for a
            # wrapper it folds in what it wraps, which the text alone does not.
            helper_hashes[qualname] = callable_identity(func)

            _record_resolution_path(func, qualname)

            try:
                tree = ast.parse(src)
            except SyntaxError:
                if reported:
                    opaque.append(qualname)
                continue

            if hash_only:
                # Followed for the cache key, not audited: reporting every
                # ``self.x = x`` in an ordinary __init__ as a scope mutation
                # would bury the real findings. Keep walking what IT builds,
                # so the closure stays transitive.
                if reported:
                    opaque.append(qualname)
                _queue_class_refs(func, tree, depth)
                continue

            func_def = _find_first_function_def(tree)
            if func_def is None:
                # A CLASS, not a function: a ClassDef has no FunctionDef at
                # the top, so there is no body to analyze for purity. Bailing
                # here also ended the WALK, which lost the code the class
                # reaches. A dataclass field like
                # ``field(default_factory=lambda: B(0))`` constructs B, so
                # editing B changes what every instance holds -- measured, the
                # cache returned ``A(value=B(value=10))`` where a fresh call
                # produced ``A(value=B(value=1000))``. A wrong answer, not a
                # stale one.
                #
                # Queue what the class body CALLS. Names only, from actual
                # Call nodes: annotations are deliberately not consulted,
                # since ``value: B`` never runs and following it would
                # invalidate on a type hint.
                if reported:
                    opaque.append(qualname)
                _queue_class_refs(func, tree, depth)
                continue

            # Drop the decorator expressions: ``inspect.getsource`` includes the
            # ``@c.cache(...)`` / ``@get_cash().cache`` lines, and analyzing them
            # as if they were body statements walks into the decorator factory's
            # source (cash's own internals), flagging its mutations as the
            # user's. We only want to analyze the function body.
            func_def.decorator_list = []

            param_names = frozenset(
                arg.arg for arg in (func_def.args.args + func_def.args.posonlyargs + func_def.args.kwonlyargs)
            )
            if func_def.args.vararg:
                param_names = param_names | {func_def.args.vararg.arg}
            if func_def.args.kwarg:
                param_names = param_names | {func_def.args.kwarg.arg}

            own_issues_from = len(all_issues)
            # What the body's names are bound to: the callee resolution below,
            # and aliased ambient reads, both need it. Imports written inside
            # the body bind locals the module's globals never see; a local
            # shadows a global of the same name.
            namespace = _build_namespace(func)
            local_imports = local_import_map(func_def, func)
            for _local, (_mod, _prefix) in local_imports.items():
                _obj = resolve_local_import(_mod, _prefix, root_module)
                if _obj is not None:
                    namespace[_local] = _obj
            visitor = _PurityVisitor(
                qualname=qualname,
                param_names=param_names,
                fresh_nodes=fresh_name_nodes(func_def),
                log_only=_log_only_ambient_reads(func_def, func, namespace),
                namespace=namespace,
                log_helpers=_log_helper_names(func_def, func),
            )
            visitor.visit(func_def)
            visitor.finalize_taint()
            if visitor.opens_tracked_database:
                visitor.issues = [i for i in visitor.issues if i.effect_kind is not EffectKind.DB_READ]
            if depth > 0 and _clock_helper_read(func) is not None:
                # Judged where it is called (`_clock_helper_read`).
                visitor.issues = [i for i in visitor.issues if i.kind != ISSUE_AMBIENT_READ]
            all_issues.extend(visitor.issues)
            environment_reads |= visitor.environment_reads

            # Flag reads of module globals that are reassigned/mutated somewhere
            # in the module - a silent staleness footgun (the cached result won't
            # change when the global does). Constants (never written) are not
            # flagged, so this stays quiet on dispatch tables / lookup maps.
            self._flag_mutable_global_reads(
                func,
                func_def,
                qualname,
                visitor.read_names,
                all_issues,
            )

            # Drop what THIS function's source says it has already audited.
            # Filtered per function, against that function's own source, so a
            # waiver written in a helper covers the helper and nothing else.
            _drop_audited(all_issues, own_issues_from, src)
            # Only now, after the waivers matched against the function's own
            # source: report lines as the FILE numbers them. Relative to the
            # decorator line, "line 4" sent users to the wrong line.
            _anchor_issue_lines(all_issues, own_issues_from, func)
            if not reported:
                # Under ``@pure`` / ``@stateful``: walked for the key only.
                del all_issues[own_issues_from:]

            if depth >= self._MAX_DEPTH:
                continue

            # Resolve callees and queue user-code helpers. The merged
            # namespace (built above) includes closure cells so nested-function
            # helpers (defined inside another function) are visible for
            # recursion.

            def _call_site_path(chain: tuple[str, ...] | None) -> tuple[str, tuple[str, ...]] | None:
                if chain and chain[0] in local_imports:  # noqa: B023 - loop var, used within iteration
                    module_name, prefix = local_imports[chain[0]]  # noqa: B023
                    return (module_name, prefix + chain[1:]) if module_name in sys.modules else None
                return _binding_path(func, chain)  # noqa: B023

            def _queue_helper(callee: Any, line: int, path: tuple[str, tuple[str, ...]] | None = None) -> None:
                if callee is None or not callable(callee):
                    return
                # First, before any attribute read: a mock answers every
                # attribute truthily, so it would pass for a cached function
                # or a @pure one below. It has no code to key and returns
                # whatever the test configured -- the call runs uncached.
                if is_mock(callee):
                    _note_binding(callee, path)
                    where = f"{path[0]}.{'.'.join(path[1])}" if path else "a callee"
                    unkeyable.append(f"{where} is a {type(callee).__name__}")
                    return
                # A call to another @cash.cache-decorated function is a
                # dependency-graph edge, not a helper to walk: its own source
                # hash and purity are tracked as a separate node. Recursing
                # would read cash's wrapper machinery (which ``functools.wraps``
                # makes look like same-package user code) and flag cash's own
                # internal mutations as the user's.
                #
                # Its binding is still noted: rebinding the name the caller
                # calls it by (``app.inner = fake``) replaces the edge.
                if getattr(callee, "_cash_cached", False):
                    _note_binding(callee, path)
                    return
                # A ``@pure`` or ``@stateful`` callee settles what the helper
                # may DO, not what it computes: it is walked below like any
                # helper, so an edit to it moves its callers' keys, and only
                # its findings are left out (``reported`` in the walk).
                if is_stateful(callee) and reported:  # noqa: B023 - loop var, called within iteration
                    all_issues.append(
                        PurityIssue(
                            kind=ISSUE_IMPURE_CALL,
                            description=f"calls @stateful {_qualname_of(callee)}()",
                            where=qualname,  # noqa: B023 - loop var, called within iteration
                            line=line,
                        )
                    )
                # The functions it runs besides its own code: the other half
                # of a decorated helper, the user function inside a library
                # wrapper (np.vectorize, toolz.curry, lru_cache), a
                # singledispatch implementation. Each user-code one is walked
                # in its own right, under its own name and namespace.
                layers = [
                    layer
                    for layer in callable_layers(callee)
                    if own_code_is_user(layer, root_module) and not getattr(layer, "_cash_cached", False)
                ]
                own = _is_user_code(callee, root_module)
                if not own and not layers:
                    # A library function the call site names by a module
                    # attribute (`requests.get`, `pd.read_csv`): not walked,
                    # but its binding is noted, so `mock.patch("requests.get")`
                    # -- or the whole module swapped for a MagicMock -- is seen
                    # on the next call and runs it uncached. It was not: the
                    # fake answer was stored under the real key and served to
                    # every later, unpatched run.
                    #
                    # Plain and builtin functions only, reached back through
                    # their path: a bound method is a new object on every
                    # attribute read and would look rebound on every call.
                    if (
                        path is not None
                        and isinstance(callee, (types.FunctionType, types.BuiltinFunctionType))
                        and resolve_binding(*path) is callee
                    ):
                        _note_binding(callee, path)
                    return
                _note_binding(callee, path)
                if own:
                    if path is not None:
                        caller_paths.setdefault(id(callee), path)
                    stack.append((callee, depth + 1, False, reported))  # noqa: B023 - same
                for layer in layers:
                    stack.append((layer, depth + 1, False, reported))  # noqa: B023 - same

            audited = audited_lines(src)[0] if "@cash:" in src else frozenset()
            for call_node in visitor.called_callable_nodes + visitor.impure_call_nodes:
                site_path = _call_site_path(_callee_chain(call_node.func))
                if site_path is not None:
                    start = getattr(call_node, "lineno", 0)
                    end = getattr(call_node, "end_lineno", None) or start
                    on_waived = any(n in audited for n in range(start, end + 1))
                    (waived_paths if on_waived else unwaived_paths).add(site_path)
                _queue_helper(
                    resolve_callee(call_node.func, namespace, modules_only=False),
                    getattr(call_node, "lineno", 0),
                    site_path,
                )

            # A helper referenced by NAME but reached through a value -- not in
            # call position -- was never walked, so an edit to it silently
            # served a stale result: ``fn = helper; fn(x)``, ``_apply(helper,
            # x)``, storing a function in a local then calling it. Any read name
            # that resolves to a user FUNCTION is a source dependency; walk it
            # too. Restricted to functions/methods so modules,
            # classes, and arbitrary attribute reads are not folded in, and the
            # existing user-code / purity / cached-node filters still apply. The
            # direction is safe: at worst it over-invalidates (a referenced-but-
            # unused function recomputes), never serves stale.
            for _name in visitor.read_names:
                _val = namespace.get(_name)
                if inspect.isfunction(_val) or inspect.ismethod(_val):
                    _queue_helper(_val, 0, _call_site_path((_name,)))
                elif isinstance(_val, type):
                    # A user class named but not called by a user path:
                    # `A.model_validate(d)` (a library method) or `build(A, d)`.
                    # Its code shapes the result all the same; followed for the
                    # key, not audited.
                    _queue_hash_only(_val, func, depth)
            _queue_annotation_refs(func, depth)

        # Stable order: by where (insertion) then line then kind.
        all_issues_sorted = tuple(
            sorted(
                all_issues,
                key=lambda i: (i.where, i.line, i.kind, i.description),
            )
        )
        return PurityReport(
            issues=all_issues_sorted,
            helper_source_hashes=helper_hashes,
            helper_objects=helper_objects,
            helper_resolution_paths=helper_paths,
            opaque_callees=tuple(sorted(set(opaque))),
            helper_bindings=tuple(bindings),
            waived_bindings=frozenset(waived_paths - unwaived_paths),
            unkeyable=tuple(unkeyable),
            environment_reads=frozenset(environment_reads),
        )

    def _flag_mutable_global_reads(
        self,
        func: Callable[..., Any],
        func_def: ast.AST,
        qualname: str,
        read_names: set[str],
        all_issues: list[PurityIssue],
    ) -> None:
        """Append an issue for each module global *func* reads that is
        reassigned/mutated elsewhere in its module - the result would go stale
        when that global changes. Reads of never-written globals (constants,
        dispatch tables) are not flagged."""
        if not read_names:
            return
        module = inspect.getmodule(func)
        if module is None:
            return
        modified = _module_modified_globals(module)
        if not modified:
            return
        module_ns = getattr(func, "__globals__", None) or {}
        locals_ = _function_locals(func_def)
        freevars = set(getattr(getattr(func, "__code__", None), "co_freevars", ()) or ())
        own_name = getattr(func, "__name__", None)
        candidates = (read_names & modified) - locals_ - freevars - BUILTIN_NAMES
        for name in sorted(candidates):
            if name == own_name or name not in module_ns:
                continue
            all_issues.append(
                PurityIssue(
                    kind=ISSUE_MUTABLE_GLOBAL,
                    description=(
                        f"reads module global {name!r} that is reassigned or mutated "
                        f"elsewhere - cached results won't reflect changes to it; pass "
                        f"it as an argument or declare it via depends_on"
                    ),
                    where=qualname,
                    line=0,
                    subject=name,
                )
            )


_global_analyzer: PurityAnalyzer | None = None
_global_analyzer_lock = threading.Lock()


def get_analyzer() -> PurityAnalyzer:
    """Return the process-wide :class:`PurityAnalyzer` singleton.

    Multiple :class:`Cash` instances share one analyzer cache so
    redundant AST walks are avoided across instances.
    """
    global _global_analyzer
    if _global_analyzer is not None:
        return _global_analyzer
    with _global_analyzer_lock:
        if _global_analyzer is None:
            _global_analyzer = PurityAnalyzer()
        return _global_analyzer


#: ``# @cash:assume-safe`` -- a waiver scoped to ONE statement.
#
# ``assume_safe=True`` on the decorator silences the whole function, for good.
# Audit a call today, add an unrelated ``session.post(...)`` next month, and
# nothing says a word: the waiver outlived the audit it was granted for.
# Measured -- a POST added after the fact was detected by the analyzer and
# suppressed by the flag.
#
# A waiver written NEXT TO the statement cannot do that. New code arrives
# unannotated, so it is reported. The scope of the exemption is visible in the
# diff that grants it, which is the property blanket suppression cannot have.
ASSUME_SAFE_RE = re.compile(r"#\s*@cash:\s*assume-safe\b")


def audited_lines(src: str) -> tuple[frozenset[int], bool]:
    """Line numbers waived by ``# @cash:assume-safe``, and the function flag.

    1-based against *src*, the same frame ``PurityIssue.line`` uses -- both
    come from the dedented function source.

    An annotation on its own line waives the statement BELOW it as well as
    itself, because that is how people write ``# noqa`` once the line is long.
    On the ``def`` line it waives the function-scoped findings instead: a read
    of a mutated global is a property of the whole body and carries no line, so
    there is no statement to attach it to.
    """
    lines = src.splitlines()
    marked: set[int] = set()
    for index, line in enumerate(lines, start=1):
        if not ASSUME_SAFE_RE.search(line):
            continue
        marked.add(index)
        if line.strip().startswith("#"):
            marked.add(index + 1)
    function_scope = any(
        lines[i - 1].lstrip().startswith(("def ", "async def ")) for i in marked if 1 <= i <= len(lines)
    )
    return frozenset(marked), function_scope


def _drop_audited(issues: list[PurityIssue], start: int, src: str) -> None:
    """Remove issues in ``issues[start:]`` that *src* waives, in place."""
    # Substring test before the line scan: almost no function carries one of
    # these, and this runs for every function the analyzer walks.
    if "@cash:" not in src:
        return
    audited, function_scope = audited_lines(src)
    if not audited and not function_scope:
        return
    kept = [issue for issue in issues[start:] if not (issue.line in audited if issue.line else function_scope)]
    del issues[start:]
    issues.extend(kept)


def _anchor_issue_lines(issues: list[PurityIssue], start: int, func: Any) -> None:
    """Rewrite ``issues[start:]`` in the defining file's line numbers, in place.

    Anchored on the same object `own_source` read: for a ``functools.wraps``
    wrapper, its own code. ``getsourcelines`` unwraps, so a print in the
    wrapper was numbered from the WRAPPED function's first line, in another
    file -- "line 46" of an 11-line deco.py.
    """
    try:
        target = func.__code__ if isinstance(func, types.FunctionType) and hasattr(func, "__wrapped__") else func
        first = inspect.getsourcelines(target)[1]
        filename = inspect.getsourcefile(target) or ""
    except SOURCE_RETRIEVAL_ERRORS:
        return
    for index in range(start, len(issues)):
        issue = issues[index]
        issues[index] = dataclasses.replace(
            issue,
            line=issue.line + first - 1 if issue.line else 0,
            filename=filename,
        )


def _log_only_ambient_reads(
    func_def: ast.AST, func: Any = None, namespace: dict[str, Any] | None = None
) -> frozenset[int]:
    """ids of the ambient reads in *func_def* whose value is only logged.

    "Logged" includes being passed to one of the module's own log helpers
    (`_log_helper_names`): one project counted ~20 KEY-AMBIENT-READ lines per
    worker start from ``_log(f"... {time.perf_counter() - t0:.2f}s")``,
    none of which could reach a result.
    """
    candidates = []
    for node in ast.walk(func_def):
        if isinstance(node, ast.Call):
            if _ambient_call(node, namespace) is not None:
                candidates.append(node)
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Load)
            and get_base_name(node.value) in ENVIRON_NAMES
        ):
            candidates.append(node)
    if not candidates:
        return frozenset()  # the common case pays for no parent map
    flow = LogOnlyFlow(func_def, _log_helper_names(func_def, func))
    return frozenset(id(n) for n in candidates if flow.only_logged(n))


def _log_helper_names(func_def: ast.AST, func: Any) -> frozenset[str]:
    """Names *func_def* calls that are, in *func*'s globals, log helpers."""
    module_ns = getattr(func, "__globals__", None)
    if not isinstance(module_ns, dict):
        return frozenset()
    return frozenset(name for name in called_names(func_def) if _is_log_helper_function(module_ns.get(name)))


def _is_log_helper_function(value: Any) -> bool:
    code = getattr(value, "__code__", None)
    if not isinstance(value, types.FunctionType) or code is None:
        return False
    known = _LOG_HELPER_CACHE.get(code)
    if known is None:
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(value)))
            known = bool(tree.body) and is_log_helper(tree.body[0])
        except SOURCE_RETRIEVAL_ERRORS + (SyntaxError, ValueError):
            known = False
        if len(_LOG_HELPER_CACHE) >= 4096:  # a notebook redefines freely
            _LOG_HELPER_CACHE.clear()
        _LOG_HELPER_CACHE[code] = known
    return known


#: code object -> "is it a log helper?". Code objects are immutable, so a
#: redefined helper is a new key.
_LOG_HELPER_CACHE: dict[Any, bool] = {}


def _describe_subscript(node: ast.Subscript) -> str:
    """A short, readable name for `HANDLERS[k]` / `globals()[n]` in a message."""
    base = node.value
    if isinstance(base, ast.Name):
        return f"{base.id}[...]"
    if isinstance(base, ast.Call) and isinstance(base.func, ast.Name):
        return f"{base.func.id}()[...]"
    if isinstance(base, ast.Attribute):
        return f"{base.attr}[...]"
    return "a subscript"


def _target_root_name(node: ast.AST) -> str | None:
    """Root ``Name`` id of an Attribute/Subscript chain (``a.b[c]`` -> ``a``)."""
    cur = node
    while isinstance(cur, (ast.Attribute, ast.Subscript)):
        cur = cur.value
    return cur.id if isinstance(cur, ast.Name) else None


def _collect_bound_names(target: ast.AST, out: set[str]) -> None:
    """Names bound by an assignment target (``Name`` / nested tuple/list)."""
    if isinstance(target, ast.Name):
        out.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List, ast.Starred)):
        for el in ast.iter_child_nodes(target):
            _collect_bound_names(el, out)


def _function_locals(func_node: ast.AST) -> frozenset[str]:
    """Names local to a function scope: parameters plus names it binds, minus
    any declared ``global``/``nonlocal``. Does not descend into nested scopes."""
    args = getattr(func_node, "args", None)
    locs: set[str] = set()
    decl: set[str] = set()
    if args is not None:
        for a in args.posonlyargs + args.args + args.kwonlyargs:
            locs.add(a.arg)
        if args.vararg:
            locs.add(args.vararg.arg)
        if args.kwarg:
            locs.add(args.kwarg.arg)
    body = getattr(func_node, "body", [])
    # A lambda's body is a single expression, not a statement list.
    stack = list(body) if isinstance(body, list) else [body]
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue  # separate scope
        if isinstance(n, (ast.Global, ast.Nonlocal)):
            decl.update(n.names)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                _collect_bound_names(t, locs)
        elif isinstance(n, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            if isinstance(n.target, ast.Name):
                locs.add(n.target.id)
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            _collect_bound_names(n.target, locs)
        elif isinstance(n, (ast.With, ast.AsyncWith)):
            for item in n.items:
                if item.optional_vars:
                    _collect_bound_names(item.optional_vars, locs)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                locs.add(al.asname or al.name.split(".")[0])
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for child in ast.iter_child_nodes(n):
            stack.append(child)
    return frozenset(locs - decl)


def _imported_module_names(tree: ast.AST) -> frozenset[str]:
    """Names bound by a plain ``import x`` / ``import x as y`` in *tree*.

    ``import os.path`` binds ``os``, so the top-level segment is what counts.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return frozenset(names)


class _GlobalMutationScanner(ast.NodeVisitor):
    """Collects module-global names that are reassigned or mutated *inside a
    function body* (i.e. reachable at runtime), scope-aware so a function's
    local that merely shares a name with a global is not mistaken for a
    mutation of that global.

    Mutations at module top level are ignored on purpose: they run once at
    import, before any cached function is called, so the global is effectively
    constant during runtime and reading it is safe. This avoids flagging
    registries/config dicts that are populated at import and then never change.
    (A mutation inside a function that only ever runs at import - e.g. a
    decorator body - is still flagged conservatively, since we cannot prove
    statically that the function never runs at call time. A false flag is a
    harmless warning; a missed one would be a silent stale cache.)"""

    def __init__(self, module_names: frozenset[str] = frozenset()) -> None:
        self.modified: set[str] = set()
        self._locals_stack: list[frozenset[str]] = []  # enclosing function locals
        # Names bound by a plain ``import x`` / ``import x as y``. A
        # write-METHOD call on one of these does not mutate it: `net.post(...)`
        # calls a function that lives on the module, it does not change the
        # module. Without this, one `requests.post(...)` anywhere in a file made
        # every function in that file that merely READS `requests` report
        # "reads module global 'requests' that is reassigned or mutated
        # elsewhere" -- measured, and on a finding that carries no line number,
        # so it could not even be waived per statement.
        #
        # Only plain module imports are excluded. `from config import SETTINGS`
        # binds an object that `SETTINGS.update(...)` really does mutate, so
        # those still flag.
        self._module_names = module_names

    @property
    def _in_function(self) -> bool:
        return bool(self._locals_stack)

    def _is_global(self, name: str) -> bool:
        # Inside a function, a name is the global only if it isn't shadowed by
        # a local there. (Only consulted when _in_function is True.)
        return all(name not in loc for loc in self._locals_stack)

    def visit_FunctionDef(self, node):  # noqa: N802
        self._locals_stack.append(_function_locals(node))
        self.generic_visit(node)
        self._locals_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):  # noqa: N802
        self._locals_stack.append(_function_locals(node))
        self.generic_visit(node)
        self._locals_stack.pop()

    def visit_Global(self, node):  # noqa: N802
        # An explicit `global G` inside a function is intent to rebind the
        # module global at runtime. (`global` at module scope is a no-op.)
        if self._in_function:
            self.modified.update(node.names)
        self.generic_visit(node)

    def visit_AugAssign(self, node):  # noqa: N802
        base = _target_root_name(node.target)
        if self._in_function and base and self._is_global(base):
            self.modified.add(base)
        self.generic_visit(node)

    def visit_Assign(self, node):  # noqa: N802
        for t in node.targets:
            if isinstance(t, (ast.Subscript, ast.Attribute)):
                base = _target_root_name(t)
                if self._in_function and base and self._is_global(base):
                    self.modified.add(base)
        self.generic_visit(node)

    def visit_Delete(self, node):  # noqa: N802
        for t in node.targets:
            base = t.id if isinstance(t, ast.Name) else _target_root_name(t)
            if self._in_function and base and self._is_global(base):
                self.modified.add(base)
        self.generic_visit(node)

    def visit_Call(self, node):  # noqa: N802
        f = node.func
        if (
            self._in_function
            and isinstance(f, ast.Attribute)
            and f.attr in REPORTED_METHODS
            and isinstance(f.value, ast.Name)
            and self._is_global(f.value.id)
            and f.value.id not in self._module_names
        ):
            self.modified.add(f.value.id)
        self.generic_visit(node)


def _module_modified_globals(module: Any) -> frozenset[str]:
    """Module-global names that are reassigned/mutated somewhere in *module*.

    Empty when the source can't be read, so nothing is flagged on incomplete
    information. The scan is memoised on the source text, so a module edited
    under a running process is scanned again.
    """
    try:
        source = inspect.getsource(module)
    except SOURCE_RETRIEVAL_ERRORS:
        return frozenset()
    return _modified_globals_in_source(source)


@functools.lru_cache(maxsize=256)
def _modified_globals_in_source(source: str) -> frozenset[str]:
    try:
        tree = ast.parse(textwrap.dedent(source))
    except (SyntaxError, ValueError):
        return frozenset()
    try:
        scanner = _GlobalMutationScanner(_imported_module_names(tree))
        scanner.visit(tree)
    except RecursionError:
        logger.debug("global-mutation scan gave up on a deeply nested module")
        return frozenset()
    return frozenset(scanner.modified)


def _qualname_of(func: Callable[..., Any]) -> str:
    """Name a callable for ``helper_source_hashes``.

    ``__main__`` is resolved the same way ``Cash.get_func_key`` resolves it.
    These keys are folded into the state hash as ``helper:{qual}:{digest}``, so
    leaving this one alone made a direct run and an import disagree on the KEY
    while agreeing on the digest -- the function name matched, the state hash
    did not, and the entry still missed. Deliberately NOT applied to
    ``helper_paths``, whose module string is looked up in ``sys.modules`` at
    runtime and has to stay ``__main__`` to resolve.
    """
    module = getattr(func, "__module__", None) or "<unknown>"
    qualname = getattr(func, "__qualname__", None) or getattr(func, "__name__", "<callable>")
    if isinstance(func, types.FunctionType) and hasattr(func, "__wrapped__"):
        # `functools.wraps` copied the wrapped function's names onto this one,
        # so both halves of a decorated helper answered to the same name and
        # the walk, which visits each name once, followed only one of them.
        # Name the wrapper by where its code was written. Only for wrappers:
        # every other function's name is unchanged, and so are their keys.
        module = func.__globals__.get("__name__") or module
        code = func.__code__
        qualname = getattr(code, "co_qualname", None) or f"{code.co_name}@wrapper"
    if module in MAIN_MODULE_NAMES:
        module = resolve_main_module(func)
    return f"{module}.{qualname}"


def _try_source_hash(func: Callable[..., Any]) -> str | None:
    """Memo key for the analyzer's own report cache -- NOT a cache key.

    Deliberately the un-stripped form, unlike every channel that goes through
    ``source_identity_digest``. Nothing downstream keys on this, so folding
    the decorator in only means two spellings of one function get analyzed
    twice instead of once, and the narrower input keeps this from quietly
    becoming a correctness surface.
    """
    try:
        src = inspect.getsource(func)
    except SOURCE_RETRIEVAL_ERRORS:
        return None
    return hashlib.sha256(normalize_source_for_hash(src).encode("utf-8")).hexdigest()


def _find_first_function_def(tree: ast.AST) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    return None
