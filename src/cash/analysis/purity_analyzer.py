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
import inspect
import logging
import sqlite3
import sys
import textwrap
import threading
import time
import types
import weakref
from collections.abc import Callable
from typing import Any

from .._annotation_refs import annotation_referents
from .._memo import MODULE_ANALYSES, PURITY_REPORTS, LruMemo
from ..diagnostics import warn_diagnostic
from ..effects import (
    ENVIRON_KEYED_METHODS,
    ENVIRON_NAMES,
    MODULE_CALLS,
    MUTATOR_METHODS,
    STDIN_NAMES,
    Action,
    EffectKind,
    classify_call,
    dotted_name,
    environ_membership,
    environment_input,
)
from ..exceptions import SOURCE_RETRIEVAL_ERRORS, CashCacheIneffectiveWarning
from ..purity import (
    KNOWN_PURE_BUILTINS,
    is_pure,
    is_stateful,
)
from ..source_norm import (
    callable_identity,
    compiled_identity,
    extension_file_digest,
    getsource,
    getsourcelines,
    own_source,
    settled_source_version,
    source_version_unchanged,
)
from ..value_types import BUILTIN_NAMES
from .ambient_reads import ambient_call, clock_helper_of, log_helper_names, log_only_ambient_reads, method_namespace
from .annotations import assume_safe_block_lines, audited_lines
from .ast_util import bytecode_global_refs, resolve_callee
from .callee_effects import module_function_global_changes, scope_locals
from .file_effects import get_base_name, get_call_module, get_call_name
from .helper_bindings import (
    binding_path,
    bindings_changed,
    build_namespace,
    called_names_in_tree,
    callee_chain,
    held_ref,
    local_import_map,
    resolve_binding,
    resolve_callee_chain,
    resolve_in_class_namespaces,
    resolve_local_import,
)
from .helper_code import UnwalkableLayers, callable_layers, is_mock, is_user_code, own_code_is_user, qualname_of
from .mutations import PANDAS_INPLACE_METHODS
from .purity_flow import (
    fresh_name_nodes,
    is_log_line,
    receiver_is_fresh,
)
from .purity_policy import AMBIENT_KINDS, DECORATOR_POLICY, REPORTED_METHODS
from .purity_report import (
    ISSUE_AMBIENT_READ,
    ISSUE_DISCARDED_CALL,
    ISSUE_DYNAMIC_PATTERN,
    ISSUE_IMPURE_CALL,
    ISSUE_MUTABLE_GLOBAL,
    ISSUE_NETWORK_READ,
    ISSUE_SCOPE_MUTATION,
    ISSUE_UNTRACKABLE_DEP,
    PurityIssue,
    PurityReport,
)
from .static_dispatch import MODULE_NAME_PREFIX, spell_static_dispatch

logger = logging.getLogger(__name__)

__all__ = ["PurityAnalyzer"]

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


#: Bare builtins whose discarded result is not worth a word: each is reported
#: by another rule already (an effect, or explicit dynamic execution).
_DISCARD_REPORTED_BUILTINS = frozenset(name for name in MODULE_CALLS if "." not in name) | {
    "open",
    "exec",
    "eval",
    "compile",
}


#: Methods that set up or train the object they are called on, in place,
#: and whose return is dropped by design (sklearn's ``fit`` returns self).
_IN_PLACE_SETUP_METHODS = frozenset(
    {
        "fit",
        "partial_fit",
        "set_params",
        "shuffle",
        "seed",
        "add_argument",
        "add_argument_group",
        "add_mutually_exclusive_group",
        "add_subparsers",
        "set_defaults",
    }
)


def _constructed_locals(func_def: ast.AST | None, params: frozenset[str]) -> frozenset[str]:
    """Locals every assignment of which is a new instance of a class, named as
    one is (``m = LinearRegression()``, ``p = argparse.ArgumentParser()``)."""
    if func_def is None:
        return frozenset()
    kinds: dict[str, bool] = {}
    for node in ast.walk(func_def):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            targets, value = [node.target], getattr(node, "value", None)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.withitem, ast.comprehension)):
            target = getattr(node, "target", None) or getattr(node, "optional_vars", None)
            targets, value = ([target] if target is not None else []), None
        else:
            continue
        for target in targets:
            for name in ast.walk(target):
                if isinstance(name, ast.Name):
                    made = (
                        target is name
                        and isinstance(node, ast.Assign)
                        and isinstance(value, ast.Call)
                        and _names_a_class(value.func)
                    )
                    kinds[name.id] = kinds.get(name.id, True) and made
    return frozenset(n for n, made in kinds.items() if made and n not in params)


def _names_a_class(func: ast.AST) -> bool:
    last = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
    return last[:1].isupper() and not last.isupper()


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
        "read_names",
        "read_attributes",
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
        fresh_nodes: frozenset[int] = frozenset(),
        log_only: frozenset[int] = frozenset(),
        namespace: dict[str, Any] | None = None,
        log_helpers: frozenset[str] = frozenset(),
        ambient_namespace: dict[str, Any] | None = None,
        func_def: ast.AST | None = None,
    ) -> None:
        self.issues: list[PurityIssue] = []
        #: *namespace* plus, in a method, its ``self`` / ``cls`` bound to the
        #: class: what a call site's clock helper is looked up in
        #: (`ambient_call`), so ``self.stamp()`` is judged like ``stamp()``.
        self._ambient_namespace = namespace if ambient_namespace is None else ambient_namespace
        #: Code objects of the clock helpers this body's call sites judged
        #: (`clock_helper_of`): the walk leaves their own read to that judgment.
        self.judged_helpers: set[Any] = set()
        #: ids of ``sys.stdin`` nodes reached as a method's receiver.
        self._stdin_attributes: set[int] = set()
        #: The function being visited, and the locals only ever bound to a
        #: new instance (`_constructed_locals`), worked out when first asked.
        self._func_def = func_def
        self._constructed: frozenset[str] | None = None
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
        # Bare names read (Load context) in this body, in source order - used
        # to detect reads of mutable module globals and to find helpers named
        # as values. Ordered, so the walk queues helpers in the same order in
        # every process (a set's order moves with PYTHONHASHSEED).
        self.read_names: dict[str, None] = {}
        #: Attribute reads (Load context) on a name or another attribute: a
        #: helper named as a value through its module (``map(helper.g, xs)``).
        self.read_attributes: list[ast.Attribute] = []
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
        # Line numbers are those of the parsed (dedented) source, not the
        # file; the qualname tells the user where to look.
        # Fresh locals: in-place mutation of these is pure (escape analysis).
        # The same question per POINT (`purity_flow.fresh_name_nodes`): ids of
        # the Name nodes that hold an object this function made, where read.
        self._fresh_nodes = fresh_nodes
        # Ambient reads (by node id) whose value reaches only a log line.
        self._log_only = log_only
        # What the body's names are bound to, so an aliased ambient read
        # (`_dt.datetime.now()`) is recognised (`ambient_call`).
        self._namespace = namespace
        # The module's own log helpers (`log_helper_names`): a call to one is
        # a print, not a call made for an effect a hit would skip.
        self._log_helpers = log_helpers
        # ``os.environ`` nodes read for one named variable (a subscript, a
        # ``get``, an ``in``) or changed through a method: judged there, not
        # as a read of the whole environment (`visit_Attribute`).
        self._environ_keyed: set[int] = set()

    # --- impure / dynamic / called-name detection on Call nodes ---

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.read_names[node.id] = None
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            self._name_call_nodes.append(node)
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr in ENVIRON_KEYED_METHODS
            and dotted_name(func.value) in ENVIRON_NAMES
        ):
            self._environ_keyed.add(id(func.value))
        for keyword in node.keywords:
            # `subprocess.run(cmd, env=os.environ)`: handed to a child, which
            # is reported as what it is.
            if keyword.arg == "env":
                self._environ_keyed.add(id(keyword.value))
        self._record_call(node)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        """``"DEBUG" in os.environ``: whether a variable is set, read by name."""
        environ = environ_membership(node)
        if environ is not None:
            self._environ_keyed.add(id(environ))
            env = environment_input(node, self._namespace, resolve_constants=True)
            if env is not None and DECORATOR_POLICY[EffectKind.ENVIRONMENT] is Action.CACHE_AS_INPUT:
                if id(node) not in self._log_only:
                    self.environment_reads.add(env)
            elif id(node) not in self._log_only:
                self._whole_environment_read(node, "... in os.environ with a name computed at run time")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """``os.environ`` used as a whole: ``.copy()``, ``.items()``,
        ``dict(os.environ)``, ``env = os.environ``. Which variables it reads is
        not known, so no key can fold them."""
        if (
            isinstance(node.ctx, ast.Load)
            and id(node) not in self._environ_keyed
            and dotted_name(node) in ENVIRON_NAMES
            and id(node) not in self._log_only
        ):
            self._whole_environment_read(node, f"{dotted_name(node)} as a whole")
            return
        if isinstance(node.ctx, ast.Load) and isinstance(node.value, (ast.Name, ast.Attribute)):
            self.read_attributes.append(node)
        if dotted_name(node.value) in STDIN_NAMES:
            # `sys.stdin.read()` is judged as a call; `.isatty()` reads nothing.
            self._stdin_attributes.add(id(node.value))
        elif (
            isinstance(node.ctx, ast.Load)
            and id(node) not in self._stdin_attributes
            and dotted_name(node) in STDIN_NAMES
        ):
            self.issues.append(
                PurityIssue(
                    kind=ISSUE_IMPURE_CALL,
                    description=f"{dotted_name(node)} - reads standard input, which a cache hit does not read",
                    where=self._qualname,
                    line=getattr(node, "lineno", 0),
                    effect_kind=EffectKind.INTERACTIVE,
                )
            )
        self.generic_visit(node)

    def _whole_environment_read(self, node: ast.AST, what: str) -> None:
        self.issues.append(
            PurityIssue(
                kind=ISSUE_AMBIENT_READ,
                description=(
                    f"{what} - reads the environment, which is not in the cache "
                    f"key, so the first call's environment is frozen into every "
                    f"later result; read each variable by name "
                    f'(os.environ.get("NAME")) to have it keyed'
                ),
                where=self._qualname,
                line=getattr(node, "lineno", 0),
            )
        )

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """``os.environ["KEY"]`` -- the one ambient read that is not a call.

        Needed as its own visitor because the call rule cannot see it: this is
        a subscript on a mapping, and it is the form most people actually
        write. ``os.environ.get("KEY")`` goes through the call table instead.

        Load context only. ``os.environ["KEY"] = ...`` is a side effect rather
        than a frozen input, a different issue with a different fix.
        """
        if get_base_name(node.value) in ENVIRON_NAMES:
            self._environ_keyed.add(id(node.value))
        env = environment_input(node, self._namespace, resolve_constants=True)
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
                env = environment_input(node, self._namespace, resolve_constants=True)
                if env is not None:
                    if id(node) not in self._log_only:
                        self.environment_reads.add(env)
                    if dotted in ("os.environ.setdefault", "os.environb.setdefault"):
                        # Also a write: it sets the variable when it is unset,
                        # which a cache hit skips.
                        self.issues.append(
                            PurityIssue(
                                kind=ISSUE_IMPURE_CALL,
                                description=f"{dotted}() - write method",
                                where=self._qualname,
                                line=line,
                            )
                        )
                    return
            ambient = ambient_call(node, self._ambient_namespace)
            helper = clock_helper_of(node, self._ambient_namespace) if ambient is not None else None
            if helper is not None:
                # A clock helper is still the user's code: walked, so an edit
                # to it reaches the key like any helper's.
                self.judged_helpers.add(helper.__code__)
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
                and effect.kind not in AMBIENT_KINDS
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
        chain = callee_chain(func_node.value)
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
        chain = callee_chain(func_node)
        if not chain or not self._namespace or chain[0] not in self._namespace:
            return False
        obj: Any = self._namespace[chain[0]]
        for attr in chain[1:]:
            if not isinstance(obj, (types.ModuleType, type)):
                return False
            obj = getattr(obj, attr, None)
        return obj is time.sleep

    def visit_Expr(self, node: ast.Expr) -> None:
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
                    and not self._works_on_its_own_object(func_node)
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

    def visit_Global(self, node: ast.Global) -> None:
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

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
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

    def visit_Assign(self, node: ast.Assign) -> None:
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

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._maybe_flag_mutation_target(node.target, node.lineno)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            self._maybe_flag_mutation_target(target, node.lineno)
        self.generic_visit(node)

    def _works_on_its_own_object(self, func_node: ast.Attribute) -> bool:
        """Is a discarded ``obj.method(...)`` work on an object this function
        made? Then dropping the return is how it is written: ``d =
        deque(xs); d.popleft()``, ``r = random.Random(k); r.shuffle(xs)``,
        and, on a local an unknown class made, the methods that configure or
        train it in place (``m = LinearRegression(); m.fit(X, y)``,
        ``p = ArgumentParser(); p.add_argument("--x")``)."""
        receiver = func_node.value
        if self._receiver_is_fresh(receiver):
            return True
        if not isinstance(receiver, ast.Name) or func_node.attr not in _IN_PLACE_SETUP_METHODS:
            return False
        if self._constructed is None:
            self._constructed = _constructed_locals(self._func_def, self._param_names)
        return receiver.id in self._constructed

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


class PurityAnalyzer:
    """Walks a callable's body + its module-bounded helpers and
    returns a :class:`PurityReport`.

    Results are cached by the analyzed callable's source hash.
    Multiple :class:`Cash` instances share a single process-wide
    analyzer via :func:`get_analyzer`.
    """

    #: How many callables one walk may visit. Not a depth cap: every helper
    #: the cached function reaches, however deep, is keyed, and the visited
    #: set ends cycles. Code that is finite never reaches this. What can is
    #: code that makes a NEW function on every read (a module ``__getattr__``
    #: or a class building a closure per attribute access, reached from the
    #: function it builds): the walk would never end. Stopping there silently
    #: would leave the rest out of the key, so the function runs uncached
    #: instead (``PurityReport.unwalkable``), with a warning.
    _WALK_LIMIT = 5_000

    def __init__(self) -> None:
        # memo key -> (report, the function it was built from); see `analyze`
        self._cache: LruMemo[str, tuple[PurityReport, weakref.ref | None]] = LruMemo(PURITY_REPORTS)
        self._cache_lock = threading.Lock()

    def analyze(self, func: Callable[..., Any]) -> PurityReport:
        """Return a :class:`PurityReport` for *func*.

        Idempotent and cached by source hash. A function marked ``@pure`` or
        ``@stateful`` is walked for the cache key like any other, but its body
        is not audited: ``@pure`` reports nothing, ``@stateful`` reports the
        function itself.
        """
        owner: weakref.ref | None = None
        target = getattr(func, "__func__", func)  # a bound method is made anew per access
        closure = bool(getattr(func, "__closure__", None))
        source_hash = _try_source_hash(func)
        if source_hash is not None:
            # Keyed by the namespace the names resolve in as well as the text:
            # `def run(): return step()` written identically in two modules
            # calls two different `step`s, and sharing one report handed the
            # second module the first one's helpers -- editing its own `step`
            # then changed nothing its key could see.
            source_hash = f"{source_hash}:{id(getattr(func, '__globals__', None))}"
            # An id outlives nothing: a module dropped from `sys.modules` frees
            # its namespace, and a new module with the same text can be given
            # the same address. Its function was then handed the dead one's
            # report, bindings and all -- and a binding into a module that has
            # gone proves nothing (`bindings_changed`), so a helper patched
            # with a mock was never seen and the call was served from the
            # cache. So an entry holds the function it was built from, and
            # serves only while that function is alive in the same namespace.
            try:
                owner = weakref.ref(target)
            except TypeError:
                source_hash = None
            else:
                # A closure's names also resolve in its cells: two closures
                # with the same text in one module (one factory called twice,
                # or two factories) can capture different helpers, and sharing
                # a report keyed the second by the first one's helpers. A
                # report of a closure belongs to that function object alone.
                if closure:
                    source_hash = f"{source_hash}:{id(func)}"
        if source_hash is not None:
            with self._cache_lock:
                entry = self._cache.get(source_hash)
            cached = None
            if entry is not None:
                cached, cached_owner = entry
                built_from = cached_owner() if cached_owner is not None else None
                if built_from is None:
                    cached = None
                elif closure and built_from is not target:
                    cached = None
                elif getattr(built_from, "__globals__", None) is not getattr(func, "__globals__", None):
                    cached = None
            # The source is the same, but a name it calls through may hold a
            # different object now (a patched helper, or a real one restored):
            # the tree below that binding is not the one this report walked.
            if cached is not None and not bindings_changed(cached):
                return cached

        report = self._analyze_uncached(func)
        if report.unwalkable:
            warn_diagnostic(
                CashCacheIneffectiveWarning,
                "KEY-HELPERS-UNWALKABLE",
                f"cash cannot key {qualname_of(func)}: {report.unwalkable}. It runs uncached.",
                "Name the helpers it reaches with depends_on=[...] instead of creating them on every read.",
            )
        if is_stateful(func):
            # The user has spoken: one finding for the function itself.
            report = dataclasses.replace(
                report,
                issues=(
                    PurityIssue(
                        kind=ISSUE_IMPURE_CALL,
                        description="explicitly marked @stateful",
                        where=qualname_of(func),
                        line=0,
                    ),
                ),
            )

        if source_hash is not None:
            with self._cache_lock:
                self._cache[source_hash] = (report, owner)
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
        cached_callees: list[Any] = []
        cached_seen: set[int] = set()

        def _note_cached(callee: Any) -> None:
            if id(callee) not in cached_seen:
                cached_seen.add(id(callee))
                cached_callees.append(held_ref(callee))

        #: Clock helpers judged at a call site: their own read is not reported.
        judged_helpers: set[Any] = set()

        def _note_binding(callee: Any, path: tuple[str, tuple[str, ...]] | None) -> None:
            if path is None or path in seen_bindings:
                return
            seen_bindings.add(path)
            bindings.append((path[0], path[1], held_ref(callee)))

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
            if getattr(target, "_cash_cached", False) is True:
                if not is_mock(target):
                    _note_cached(target)
                return
            if not is_user_code(target, root_module):
                return
            # ``reported`` is the entry being walked when this runs.
            stack.append((target, depth + 1, True, reported))

        def _queue_annotation_refs(obj: Any, depth: int) -> None:
            """Queue, hash-only, the user classes and functions *obj*'s
            annotations name (see ``cash._annotation_refs``): pydantic runs a
            field type's validators, a ``get_type_hints`` builder constructs it."""
            for target in annotation_referents(obj, lambda o: is_user_code(o, root_module)):
                _queue_hash_only(target, obj, depth)

        def _queue_class_refs(cls: Any, tree: ast.AST, depth: int) -> None:
            """Queue user-code objects a CLASS body constructs, hash-only."""
            if not isinstance(cls, type):
                return
            for called in called_names_in_tree(tree):
                _queue_hash_only(resolve_in_class_namespaces(cls, called), cls, depth)
            _queue_annotation_refs(cls, depth)

        # Each entry is (callable, depth, hash_only, reported). Every entry is
        # walked for the cache key. ``reported`` is False below a callable marked
        # ``@pure`` or ``@stateful``: the marker settles what it and everything
        # it calls may do, so their findings are not reported -- but their code
        # still decides the result, so it is keyed like any other helper's.
        root_reported = not (is_pure(root_func) or is_stateful(root_func))
        stack: list[tuple[Callable[..., Any], int, bool, bool]] = [(root_func, 0, False, root_reported)]
        # Why `callable_layers` could not find every function one callable
        # runs: the walk stops and the function runs uncached.
        layer_failures: list[str] = []
        if (
            isinstance(root_func, types.FunctionType)
            and hasattr(root_func, "__wrapped__")
            and not own_code_is_user(root_func, root_module)
        ):
            # `@cash.cache` over a LIBRARY decorator (`@retry(...)`,
            # `@torch.no_grad()`): the wrapper's own body is someone else's
            # code, so start from the user functions it runs instead.
            try:
                root_layers = callable_layers(root_func)
            except UnwalkableLayers as e:
                root_layers = []
                layer_failures.append(str(e))
            starts = [(layer, 0, False, root_reported) for layer in root_layers if own_code_is_user(layer, root_module)]
            if starts:
                stack = starts
        # id -> whether that walk reported findings, and the name it took.
        visited_ids: dict[Any, bool] = {}
        walked_names: dict[Any, str] = {}
        # Every object walked stays alive until the walk ends: `id()` is
        # unique only among live objects, and a bound method is made anew on
        # each attribute read, so a collected one could hand its id to a
        # different method, which was then skipped as already walked.
        keep_alive: list[Any] = []
        unwalkable = ""
        while stack:
            func, depth, hash_only, reported = stack.pop()
            reported = reported and not (is_pure(func) or is_stateful(func))
            # A bound method is visited as its function and the object it is
            # bound to: `Model.run` read twice gives two method objects, one
            # method.
            if isinstance(func, types.MethodType):
                walk_id: Any = (id(func.__func__), id(func.__self__))
            else:
                walk_id = id(func)
            walked_reported = visited_ids.get(walk_id)
            if walked_reported is not None and (walked_reported or not reported):
                continue
            if layer_failures:
                unwalkable = layer_failures[0]
                break
            if len(keep_alive) >= self._WALK_LIMIT:
                unwalkable = (
                    f"the helpers {qualname_of(root_func)} reaches do not end (over {self._WALK_LIMIT} "
                    "functions walked; code that makes a new function on every read can cause this)"
                )
                break
            keep_alive.append(func)
            visited_ids[walk_id] = reported
            if walked_reported is not None:
                # Walked below a marker first, reached now from an unmarked
                # caller too: walk it again so its findings are reported. Its
                # key part is the same, under the same name.
                qualname = walked_names[walk_id]
            else:
                qualname = qualname_of(func)
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
                walked_names[walk_id] = qualname

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
            # exactly what ``hash_callable_source`` recomputes live for
            # the same object, so the snapshot and the per-call value agree
            # instead of disagreeing forever.
            try:
                src = own_source(func)
            except SOURCE_RETRIEVAL_ERRORS:
                if reported:
                    opaque.append(qualname)
                helper_hashes[qualname] = compiled_identity(func)
                _record_resolution_path(func, qualname)
                # Its helpers still decide the result. Without source there
                # is no AST to find them in, so the bytecode's global names
                # stand in: a cached function run from `python - <<EOF` or
                # `python -c` keyed its own bytecode and nothing it called,
                # and an edited helper was served the old result.
                if isinstance(func, types.FunctionType):
                    for chain in bytecode_global_refs(func):
                        callee = resolve_callee_chain(func.__globals__, chain)
                        if not isinstance(callee, types.FunctionType) or callee is func:
                            continue
                        if is_mock(callee):
                            continue
                        if getattr(callee, "_cash_cached", False) is True:
                            _note_cached(callee)
                            continue
                        if not own_code_is_user(callee, root_module):
                            continue
                        path = binding_path(func, chain)
                        _note_binding(callee, path)
                        if path is not None:
                            caller_paths.setdefault(id(callee), path)
                        stack.append((callee, depth + 1, False, reported))
                continue
            src = textwrap.dedent(src)

            # Hash the NORMALIZED source for cache-key invalidation of
            # helpers. Root function's hash is captured separately by the
            # decorator via hash_callable_source - we record all walked
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
            namespace = build_namespace(func)
            local_imports = local_import_map(func_def, func)
            for _local, (_mod, _prefix) in local_imports.items():
                _obj = resolve_local_import(_mod, _prefix, root_module)
                if _obj is not None:
                    namespace[_local] = _obj
            dispatch_issues = spell_static_dispatch(func_def, namespace, qualname)
            ambient_namespace = method_namespace(func, func_def, namespace)
            visitor = _PurityVisitor(
                qualname=qualname,
                param_names=param_names,
                fresh_nodes=fresh_name_nodes(func_def),
                log_only=log_only_ambient_reads(func_def, func, ambient_namespace),
                namespace=namespace,
                log_helpers=log_helper_names(func_def, func),
                ambient_namespace=ambient_namespace,
                func_def=func_def,
            )
            visitor.visit(func_def)
            visitor.finalize_taint()
            visitor.issues.extend(dispatch_issues)
            if visitor.opens_tracked_database:
                visitor.issues = [i for i in visitor.issues if i.effect_kind is not EffectKind.DB_READ]
            if depth > 0 and getattr(func, "__code__", None) in judged_helpers:
                # Judged where it is called (`clock_helper_read`). Reached
                # any other way (``fn = now; fn()``), it reports its own read.
                visitor.issues = [i for i in visitor.issues if i.kind != ISSUE_AMBIENT_READ]
            judged_helpers |= visitor.judged_helpers
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
            audited, function_scope = _waived_lines(src, tree, func)
            _drop_audited(all_issues, own_issues_from, audited, function_scope)
            # Only now, after the waivers matched against the function's own
            # source: report lines as the FILE numbers them. Relative to the
            # decorator line, "line 4" sent users to the wrong line.
            _anchor_issue_lines(all_issues, own_issues_from, func)
            if not reported:
                # Under ``@pure`` / ``@stateful``: walked for the key only.
                del all_issues[own_issues_from:]

            # Resolve callees and queue user-code helpers. The merged
            # namespace (built above) includes closure cells so nested-function
            # helpers (defined inside another function) are visible for
            # recursion.

            def _call_site_path(chain: tuple[str, ...] | None) -> tuple[str, tuple[str, ...]] | None:
                if chain and chain[0].startswith(MODULE_NAME_PREFIX):  # sys.modules["mod"]
                    return (chain[0][len(MODULE_NAME_PREFIX) : -1], chain[1:]) if len(chain) > 1 else None
                if chain and chain[0] in local_imports:  # loop var, used within iteration
                    module_name, prefix = local_imports[chain[0]]
                    return (module_name, prefix + chain[1:]) if module_name in sys.modules else None
                return binding_path(func, chain)

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
                if getattr(callee, "_cash_cached", False) is True:
                    _note_binding(callee, path)
                    _note_cached(callee)
                    return
                # A classmethod reached as `module.Model.run`, or bound to a
                # name (`run = Model.run`), is a method bound to the CLASS.
                # Its function is walked below; the class it reads through
                # `cls` is not named anywhere in the body, so its constants
                # (`factor = 2`) reached no channel and an edit was served
                # the old result. Key the class, as a class read by name is.
                if isinstance(callee, types.MethodType) and isinstance(callee.__self__, type):
                    _queue_hash_only(callee.__self__, func, depth)  # loop var, used within iteration
                # A ``@pure`` or ``@stateful`` callee settles what the helper
                # may DO, not what it computes: it is walked below like any
                # helper, so an edit to it moves its callers' keys, and only
                # its findings are left out (``reported`` in the walk).
                if is_stateful(callee) and reported:  # loop var, called within iteration
                    all_issues.append(
                        PurityIssue(
                            kind=ISSUE_IMPURE_CALL,
                            description=f"calls @stateful {qualname_of(callee)}()",
                            where=qualname,  # loop var, called within iteration
                            line=line,
                        )
                    )
                # The functions it runs besides its own code: the other half
                # of a decorated helper, the user function inside a library
                # wrapper (np.vectorize, toolz.curry, lru_cache), a
                # singledispatch implementation. Each user-code one is walked
                # in its own right, under its own name and namespace.
                layers = []
                try:
                    callee_layers = callable_layers(callee)
                except UnwalkableLayers as e:
                    layer_failures.append(str(e))
                    return
                for layer in callee_layers:
                    if getattr(layer, "_cash_cached", False) is True and not is_mock(layer):
                        _note_cached(layer)
                    elif own_code_is_user(layer, root_module):
                        layers.append(layer)
                # A function of a compiled extension built in the project
                # (`build_ext --inplace`) is no Python source, but it is the
                # user's code: keyed by its built file (`compiled_identity`).
                own = is_user_code(callee, root_module) or extension_file_digest(callee) is not None
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
                    stack.append((callee, depth + 1, False, reported))
                for layer in layers:
                    stack.append((layer, depth + 1, False, reported))

            for call_node in visitor.called_callable_nodes + visitor.impure_call_nodes:
                site_path = _call_site_path(callee_chain(call_node.func))
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
            # The same through a module or a class: `map(helper.g, xs)`,
            # `fn = helper.g`, `df.apply(features.row)`. Only the call-position
            # spelling was followed, so an edit to `g` served the old result.
            # Resolved statically (no property runs), from a module or class
            # the name is bound to, not a local or parameter that shadows it.
            shadowed = (param_names | scope_locals(func_def)) - local_imports.keys()
            for _node in visitor.read_attributes:
                _chain = callee_chain(_node)
                if _chain is None or _chain[0] in shadowed:
                    continue
                if not isinstance(namespace.get(_chain[0]), (types.ModuleType, type)):
                    continue
                _val = resolve_callee(_node, namespace, modules_only=False)
                if _val is None or is_mock(_val):
                    continue
                if getattr(_val, "_cash_cached", False) is True or (
                    (inspect.isfunction(_val) or inspect.ismethod(_val)) and is_user_code(_val, root_module)
                ):
                    _queue_helper(_val, getattr(_node, "lineno", 0), _call_site_path(_chain))
                elif isinstance(_val, type):
                    _queue_hash_only(_val, func, depth)
            _queue_annotation_refs(func, depth)
        if layer_failures and not unwalkable:
            unwalkable = layer_failures[0]

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
            cached_callees=tuple(cached_callees),
            unwalkable=unwalkable,
        )

    def _flag_mutable_global_reads(
        self,
        func: Callable[..., Any],
        func_def: ast.AST,
        qualname: str,
        read_names: dict[str, None],
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
        locals_ = scope_locals(func_def)
        freevars = set(getattr(getattr(func, "__code__", None), "co_freevars", ()) or ())
        own_name = getattr(func, "__name__", None)
        candidates = (read_names.keys() & modified) - locals_ - freevars - BUILTIN_NAMES
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


def _waived_lines(src: str, tree: ast.AST, func: Any) -> tuple[frozenset[int], bool]:
    """The lines of *src* a waiver covers, and whether one covers the function.

    ``# @cash:assume-safe`` (`audited_lines`) and the lines inside a ``with
    cash.assume_safe():`` block (`assume_safe_block_lines`), both numbered
    as *src* and *tree* are. Only the comment on the ``def`` line waives the
    function-scoped findings: a block is lines, and they carry none.
    """
    # Substring test before the line scan: almost no function carries a
    # comment waiver, and this runs for every function the analyzer walks.
    marked, function_scope = audited_lines(src) if "@cash:" in src else (frozenset(), False)
    if "with" in src:
        marked = marked | assume_safe_block_lines(tree, func)
    return marked, function_scope


def _drop_audited(issues: list[PurityIssue], start: int, audited: frozenset[int], function_scope: bool) -> None:
    """Remove issues in ``issues[start:]`` that the waivers cover, in place."""
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
        first = getsourcelines(target)[1]
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


def _module_modified_globals(module: Any) -> frozenset[str]:
    """Module-global names that are reassigned/mutated somewhere in *module*.

    Empty when the source can't be read, so nothing is flagged on incomplete
    information. The scan is memoised on the source text, so a module edited
    under a running process is scanned again; and, once its file has settled,
    on the file's version, so each helper the walk meets in a big module does
    not read and hash the whole file again to find the scan it already did.
    """
    version = settled_source_version(module)
    if version is not None:
        hit = _MODULE_MUTATIONS.get(version)
        if hit is not None:
            return hit
    try:
        source = inspect.getsource(module)
    except SOURCE_RETRIEVAL_ERRORS:
        return frozenset()
    modified = _modified_globals_in_source(source)
    if version is not None and source_version_unchanged(version):
        _MODULE_MUTATIONS[version] = modified
    return modified


#: `_module_modified_globals` per module file version: (path, mtime_ns, size).
_MODULE_MUTATIONS: LruMemo[tuple[str, int, int], frozenset[str]] = LruMemo(MODULE_ANALYSES)


@functools.lru_cache(maxsize=256)
def _modified_globals_in_source(source: str) -> frozenset[str]:
    try:
        tree = ast.parse(textwrap.dedent(source))
    except (SyntaxError, ValueError):
        return frozenset()
    # Changes made by code at module level run once, at import, before any
    # cached function is called, so a registry filled at import reads as
    # constant; only function bodies count. A function that only ever runs
    # at import (a decorator body) still counts: a false flag is a warning, a
    # missed one a stale cache. A method call on a plainly imported module
    # (``requests.post``) calls a function and does not change the module;
    # ``from config import SETTINGS`` binds an object, which still counts.
    try:
        return module_function_global_changes(tree, _imported_module_names(tree))
    except RecursionError:
        logger.debug("global-mutation scan gave up on a deeply nested module")
        return frozenset()


def _try_source_hash(func: Callable[..., Any]) -> str | None:
    """Memo key for the analyzer's own report cache -- NOT a cache key.

    Deliberately the raw text, unlike every channel that goes through
    ``source_identity_digest``. Nothing downstream keys on this, so a comment
    or a reformat only means a function is analyzed again. The normalized
    form drops what the key ignores and the report does not: a waiver, the
    ``# @cash:assume-safe`` comment or a ``with cash.assume_safe():`` block.
    Two methods named alike in one module, one of them waived, shared one
    report, and the second got the first one's findings.
    """
    try:
        src = getsource(func)
    except SOURCE_RETRIEVAL_ERRORS:
        return None
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def _find_first_function_def(tree: ast.AST) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    return None
