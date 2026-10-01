"""The rules the purity analysis applies to one function body.

`PurityVisitor` walks a single body and reports:

* **Known-impure calls** - ``requests.post``, ``os.system``,
  ``logging.info``, file-write methods (``to_csv``, ``write``, ...).
* **Explicit dynamism** - ``eval``/``exec``/``compile``,
  ``getattr(obj, name)(...)`` where *name* is not a constant.
* **Discarded-return calls** - ``f(x)`` as a statement (return
  value thrown away). Strong syntactic signal of "I'm calling this
  for the side effect". Skipped for callees in
  :data:`KNOWN_PURE_BUILTINS` (where discarding is just dead code).
* **Scope mutations** - ``global``/``nonlocal``, assignment to
  ``Attribute`` or ``Subscript`` targets, augmented-assign to same.
* **Ambient reads** - the clock, the environment, standard input.

It does not follow callees: it records the calls and names it saw, and the
helper walk (`helper_walk`) resolves and visits them.
"""

from __future__ import annotations

import ast
import sqlite3
import time
import types
from typing import Any

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
from ..purity import KNOWN_PURE_BUILTINS
from .ambient_reads import ambient_call, clock_helper_of
from .file_effects import get_base_name, get_call_module, get_call_name
from .helper_bindings import callee_chain
from .mutations import PANDAS_INPLACE_METHODS
from .purity_flow import is_log_line, receiver_is_fresh
from .purity_policy import AMBIENT_KINDS, DECORATOR_POLICY, REPORTED_METHODS
from .purity_report import (
    ISSUE_AMBIENT_READ,
    ISSUE_DISCARDED_CALL,
    ISSUE_DYNAMIC_PATTERN,
    ISSUE_IMPURE_CALL,
    ISSUE_NETWORK_READ,
    ISSUE_SCOPE_MUTATION,
    ISSUE_UNTRACKABLE_DEP,
    PurityIssue,
)

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


class PurityVisitor(ast.NodeVisitor):
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
        """Judge one call by the first rule in `_CALL_RULES` that claims it; a
        call no rule claims is recorded for the walk to resolve and follow.

        Calling a parameter (``def f(cb): cb(x)``) is not a rule: a callable
        that reaches a cached call as an argument is hashed by its source, so
        an edit to it invalidates, and one cash cannot hash
        (``functools.partial``) is reported at the argument.
        """
        for rule in self._CALL_RULES:
            if rule(self, node):
                return
        self.called_callable_nodes.append(node)

    def _issue(self, kind: str, description: str, node: ast.AST, effect_kind: EffectKind | None = None) -> None:
        self.issues.append(
            PurityIssue(
                kind=kind,
                description=description,
                where=self._qualname,
                line=getattr(node, "lineno", 0),
                effect_kind=effect_kind,
            )
        )

    def _dynamic_execution(self, node: ast.Call) -> bool:
        """``eval`` / ``exec`` / ``compile`` by bare name."""
        func = node.func
        if isinstance(func, ast.Name) and func.id in {"eval", "exec", "compile"}:
            self._issue(ISSUE_UNTRACKABLE_DEP, f"{func.id}(...) - explicit dynamic execution", node)
            return True
        return False

    def _dynamic_getattr(self, node: ast.Call) -> bool:
        """``getattr(obj, name)(...)`` where *name* is not a constant string."""
        func = node.func
        if (
            isinstance(func, ast.Call)
            and isinstance(func.func, ast.Name)
            and func.func.id == "getattr"
            and len(func.args) >= 2
            and not (isinstance(func.args[1], ast.Constant) and isinstance(func.args[1].value, str))
        ):
            self._issue(
                ISSUE_UNTRACKABLE_DEP, "getattr(obj, name)(...) with non-constant name - dynamic dispatch", node
            )
            return True
        return False

    def _dynamic_import(self, node: ast.Call) -> bool:
        """``importlib.import_module(...)`` / ``__import__(...)``. The imported
        module's members are resolved from a runtime value, so an edit to that
        module is invisible to the cache key."""
        func = node.func
        is_import_module = isinstance(func, ast.Attribute) and func.attr == "import_module"
        is_dunder_import = isinstance(func, ast.Name) and func.id == "__import__"
        if not (is_import_module or is_dunder_import):
            return False
        self._issue(
            ISSUE_UNTRACKABLE_DEP,
            f"{'importlib.import_module' if is_import_module else '__import__'}"
            "(...) - dynamic import; the imported module's code is not tracked",
            node,
        )
        return True

    def _getattr_of_a_dynamic_builtin(self, node: ast.Call) -> bool:
        """``getattr(x, "exec")(...)``: a CONSTANT name, so `_dynamic_getattr`
        does not fire, yet what it reaches is the very thing that rule exists
        to stop (``getattr(builtins, "exec")("z = 5")`` runs arbitrary
        source)."""
        func = node.func
        if (
            isinstance(func, ast.Call)
            and isinstance(func.func, ast.Name)
            and func.func.id == "getattr"
            and len(func.args) >= 2
            and isinstance(func.args[1], ast.Constant)
            and func.args[1].value in _DYNAMIC_BUILTIN_NAMES
        ):
            name = func.args[1].value
            self._issue(ISSUE_UNTRACKABLE_DEP, f"getattr(..., {name!r})(...) - reaches {name} indirectly", node)
            return True
        return False

    def _constant_getattr(self, node: ast.Call) -> bool:
        """``getattr(obj, "name")(...)`` with a constant identifier is
        ``obj.name(...)`` spelled differently: judged, and followed as a
        helper, as the attribute call it is (``getattr(helpers, "fun1")()``
        must reach ``fun1``'s code)."""
        func = node.func
        if not (
            isinstance(func, ast.Call)
            and isinstance(func.func, ast.Name)
            and func.func.id == "getattr"
            and len(func.args) == 2
            and not func.keywords
            and isinstance(func.args[1], ast.Constant)
            and isinstance(func.args[1].value, str)
            and func.args[1].value.isidentifier()
        ):
            return False
        spelled = ast.copy_location(
            ast.Call(
                func=ast.copy_location(
                    ast.Attribute(value=func.args[0], attr=func.args[1].value, ctx=ast.Load()), func
                ),
                args=node.args,
                keywords=node.keywords,
            ),
            node,
        )
        self._record_call(spelled)
        return True

    def _subscript_call(self, node: ast.Call) -> bool:
        """Calling whatever a subscript yields. Judged in `finalize_taint`:
        only a table that cannot reach the cache key is reported, and whether
        the base is a body-local depends on assignments that may appear after
        this call in source order."""
        if isinstance(node.func, ast.Subscript):
            self._subscript_call_nodes.append(node)
            return True
        return False

    def _named_call(self, node: ast.Call) -> bool:
        """A call by a name or a dotted name, judged by what it does: an
        ambient read, a log line, an effect (``requests.post``, ``os.system``,
        ``df.to_csv``), a mutator or an ``inplace=True`` method."""
        func_name = get_call_name(node.func)
        if not func_name:
            return False
        module_name = get_call_module(node.func)
        dotted = f"{module_name}.{func_name}" if module_name else func_name
        if self._environment_read(node, dotted) or self._ambient_read(node):
            return True
        if is_log_line(node):
            return True  # a diagnostic line: a hit skipping it is what caching means
        if _opens_tracked_database(node.func, self._namespace):
            self.opens_tracked_database = True
        effect = classify_call(node, self._namespace)
        return (
            self._read_from_a_server(node, dotted, effect)
            or self._known_effect(node, dotted, effect)
            or self._reported_method(node, func_name, effect)
            or self._inplace_true(node)
        )

    def _environment_read(self, node: ast.Call, dotted: str) -> bool:
        """An environment read the key folds by value (`environment_input`).
        ``os.environ.setdefault`` is also a write: it sets the variable when
        it is unset, which a cache hit skips."""
        if DECORATOR_POLICY[EffectKind.ENVIRONMENT] is not Action.CACHE_AS_INPUT:
            return False
        env = environment_input(node, self._namespace, resolve_constants=True)
        if env is None:
            return False
        if id(node) not in self._log_only:
            self.environment_reads.add(env)
        if dotted in ("os.environ.setdefault", "os.environb.setdefault"):
            self._issue(ISSUE_IMPURE_CALL, f"{dotted}() - write method", node)
        return True

    def _ambient_read(self, node: ast.Call) -> bool:
        """An ambient read (``datetime.now``, ``os.getenv``, ``uuid4``, a clock
        helper). Only ever matched DOTTED, or through what a name is bound to:
        every entry carries its module, so a method named ``now`` on the
        user's own object is not this."""
        ambient = ambient_call(node, self._ambient_namespace)
        if ambient is None:
            return False
        helper = clock_helper_of(node, self._ambient_namespace)
        if helper is not None:
            # A clock helper is still the user's code: walked, so an edit
            # to it reaches the key like any helper's.
            self.judged_helpers.add(helper.__code__)
            self.called_callable_nodes.append(node)
        if id(node) in self._log_only:
            return True  # only ever printed or logged: cannot reach a result
        shown = ambient if ambient.endswith(")") else f"{ambient}()"
        self._issue(
            ISSUE_AMBIENT_READ,
            f"{shown} - reads ambient state, which is not in the "
            f"cache key, so the first call's value is frozen into "
            f"every later result",
            node,
        )
        return True

    def _read_from_a_server(self, node: ast.Call, dotted: str, effect: Any) -> bool:
        """A network or database read: what it returns is an input the key
        cannot see. Not walked, but its binding is noted."""
        if effect is None or DECORATOR_POLICY[effect.kind] is not Action.SUGGEST_TTL:
            return False
        self._issue(
            ISSUE_NETWORK_READ,
            f"{dotted}() - what the {_SOURCE[effect.kind]} returns is not in the cache key",
            node,
            effect_kind=effect.kind,
        )
        self.impure_call_nodes.append(node)
        return True

    def _known_effect(self, node: ast.Call, dotted: str, effect: Any) -> bool:
        """A function call with an effect the decorator warns about."""
        if (
            effect is None
            or effect.kind in AMBIENT_KINDS
            or DECORATOR_POLICY[effect.kind] is not Action.WARN
            or effect.method
        ):
            return False
        what = (
            "draws on pyplot's current figure, which a hit does not redraw"
            if effect.kind is EffectKind.DISPLAY
            else "known I/O / side-effecting"
        )
        self._issue(ISSUE_IMPURE_CALL, f"{dotted}() - {what}", node, effect_kind=effect.kind)
        self.impure_call_nodes.append(node)
        return True

    def _reported_method(self, node: ast.Call, func_name: str, effect: Any) -> bool:
        """A method with an effect on any receiver (``to_csv``, ``write``,
        ``post``, ``execute``, ...), or a container mutator. Skipped when the
        receiver is a fresh local (``lines.append`` where ``lines = []``):
        mutating a local accumulator is pure."""
        func = node.func
        reported = (
            effect is not None and effect.method and DECORATOR_POLICY[effect.kind] is Action.WARN
        ) or func_name in MUTATOR_METHODS
        if (
            not isinstance(func, ast.Attribute)
            or not reported
            or self._receiver_is_fresh(func.value)
            or self._is_module_function_named_like_a_mutator(func)
        ):
            return False
        base = get_base_name(func.value)
        base_str = f"{base}." if base else ""
        what = "write method"
        if func.attr in MUTATOR_METHODS:
            # `rows.sort()` on a parameter changes the caller's list:
            # say so, rather than the label a local's `.sort()` gets.
            kind = self._mutation_kind(base, "method")
            if kind != "method mutation":
                what = kind
        self._issue(
            ISSUE_IMPURE_CALL,
            f"{base_str}{func.attr}() - {what}",
            node,
            effect_kind=effect.kind if effect is not None else None,
        )
        return True

    def _inplace_true(self, node: ast.Call) -> bool:
        """A pandas method called with ``inplace=True`` mutates its receiver."""
        func = node.func
        if (
            not isinstance(func, ast.Attribute)
            or func.attr not in PANDAS_INPLACE_METHODS
            or self._receiver_is_fresh(func.value)
        ):
            return False
        for kw in node.keywords:
            if kw.arg == "inplace" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                base = get_base_name(func.value)
                base_str = f"{base}." if base else ""
                self._issue(ISSUE_IMPURE_CALL, f"{base_str}{func.attr}(inplace=True) - in-place mutation", node)
                return True
        return False

    #: The call rules, tried in this order; the first that returns True has
    #: judged the call. The order is part of the verdict: a dynamic
    #: ``getattr`` is reported before its constant form is rewritten, and an
    #: ambient read is named before the effect table is consulted.
    _CALL_RULES = (
        _dynamic_execution,
        _dynamic_getattr,
        _dynamic_import,
        _getattr_of_a_dynamic_builtin,
        _constant_getattr,
        _subscript_call,
        _named_call,
    )

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
