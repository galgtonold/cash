"""Python's scoping rules, written once: which names a piece of code binds
and reads, and in which scope.

:func:`scope_model` walks a syntax tree one time and returns a
:class:`ScopeModel`: the tree of scopes (module, function, lambda, class,
comprehension, type parameters, annotation), the names each binds and reads,
its ``global`` / ``nonlocal`` declarations and star imports, and every read,
binding and call site as one list of events in the order the code runs them.
Each analysis that needs to know "is this name the notebook's or a local of
the function", "which names does this statement bind", "what does this
function read from its globals" asks the model, so a new piece of syntax
(a ``match`` capture, a type parameter, a template string) is taught in one
place. ``tests/test_core/analysis/test_scope_model_matches_symtable.py``
checks the model against Python's own ``symtable`` on every supported
version.

What the model follows, as Python does:

* a function's parameters and everything its body binds are its locals,
  unless declared ``global`` or ``nonlocal``; decorators, defaults and base
  classes are evaluated in the enclosing scope;
* a class body is a scope whose names the functions inside it do not see;
* a comprehension binds its targets for itself, evaluates its first iterable
  in the enclosing scope, and a ``:=`` inside it binds in the nearest scope
  that is not a comprehension;
* a ``match`` pattern binds its captures (``case [a, *rest]``,
  ``case {'k': v, **others}``, ``case Point(x=a) as whole``), ``except E as
  e`` binds ``e``, ``import a.b`` binds ``a``, ``del x`` unbinds ``x``;
* ``from m import *`` binds names the text does not show: the scope's
  ``star_imports`` lists the modules;
* type parameters (``def f[T: Bound]``, ``class C[T]``, ``type A[T] = ...``)
  are bound in a scope of their own around the definition;
* annotations are read in an :data:`ANNOTATION` scope. Python evaluates a
  function's annotations when the ``def`` runs (up to 3.13), on first use
  (3.14), or never by itself (``from __future__ import annotations``, a
  quoted annotation, a variable annotated inside a function); the scope's
  ``evaluated`` says which kind it is, and a quoted annotation is parsed so
  the names in it are known.

The walk uses a work list, not recursion: a sum of a thousand terms nests a
thousand levels deep and is walked like any other.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterator
from typing import NamedTuple

from .. import _weak

__all__ = [
    "ANNOTATION",
    "BIND",
    "CLASS",
    "COMPREHENSION",
    "DECLARED",
    "DELETE",
    "EVALUATED",
    "EXCEPT",
    "FUNCTION",
    "LAZY",
    "MATCH",
    "NEVER",
    "QUOTED",
    "LAMBDA",
    "MODULE",
    "READ",
    "SITE",
    "TYPE_PARAMS",
    "Event",
    "Scope",
    "ScopeModel",
    "bound_names",
    "function_locals",
    "scope_model",
]

# --- scope kinds -----------------------------------------------------------
MODULE = "module"
FUNCTION = "function"
LAMBDA = "lambda"
CLASS = "class"
COMPREHENSION = "comprehension"
#: The scope around a generic ``def`` / ``class`` / ``type`` that holds its
#: type parameters.
TYPE_PARAMS = "type parameters"
#: Where an annotation, a type parameter's bound or default, or a type
#: alias's value is read.
ANNOTATION = "annotation"

# --- event kinds -----------------------------------------------------------
#: A name is read (``Event.name``).
READ = "read"
#: A name is bound or unbound (``Event.name``; ``Event.how`` says by what).
BIND = "bind"
#: A node other analyses hang rules on is reached, before its parts: a call,
#: a subscript, an attribute, an augmented assignment, a ``def`` or a
#: ``class``.
SITE = "site"

# --- how a name is bound (``Event.how``, ``Scope.stores``) ------------------
ASSIGN = "assign"
AUGMENTED = "augmented"
#: ``x: int`` with no value: declares ``x`` local to its scope, binds nothing.
DECLARED = "declared"
IMPORT = "import"
DEF = "def"
CLASS_DEF = "class"
FOR = "for"
WITH = "with"
EXCEPT = "except"
MATCH = "match"
WALRUS = "walrus"
DELETE = "del"
PARAMETER = "parameter"
COMPREHENSION_TARGET = "comprehension"
TYPE_PARAMETER = "type parameter"
TYPE_ALIAS = "type alias"

#: A function's annotations and a module or class level variable annotation:
#: evaluated when the statement runs, or on first use (Python 3.14).
EVALUATED = "evaluated"
#: A type parameter's bound or default and a type alias's value: evaluated
#: on first use, on every version.
LAZY = "lazy"
#: A variable annotated inside a function: Python never evaluates it.
NEVER = "never"
#: The text of a quoted annotation (``x: "Kind"``): Python keeps the string;
#: whoever resolves the hints (``typing.get_type_hints``, a dispatch or
#: validation decorator) reads the names.
QUOTED = "quoted"

_FUNCTION_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef)
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
_SITES = (ast.Call, ast.Subscript, ast.Attribute)
_TYPE_ALIAS = getattr(ast, "TypeAlias", ())
_TRY = (ast.Try, *((ast.TryStar,) if hasattr(ast, "TryStar") else ()))


class Scope:
    """One scope of the code: what it binds, reads and declares.

    ``stores`` maps each name the scope's own code binds or unbinds to how
    (the ``how`` values above); a name declared ``global`` or ``nonlocal``
    is stored here and bound elsewhere (:meth:`owner`). ``reads`` are the
    names its own code reads, wherever they resolve.
    """

    __slots__ = (
        "kind",
        "name",
        "node",
        "parent",
        "children",
        "params",
        "stores",
        "reads",
        "declared_global",
        "declared_nonlocal",
        "star_imports",
        "evaluated",
    )

    def __init__(self, kind: str, name: str, node: ast.AST | None, parent: Scope | None) -> None:
        self.kind = kind
        self.name = name
        self.node = node
        self.parent = parent
        self.children: list[Scope] = []
        self.params: set[str] = set()
        self.stores: dict[str, set[str]] = {}
        self.reads: set[str] = set()
        self.declared_global: set[str] = set()
        self.declared_nonlocal: set[str] = set()
        #: The modules of each ``from m import *`` here.
        self.star_imports: list[str] = []
        #: For an :data:`ANNOTATION` scope: when Python evaluates it.
        self.evaluated: str | None = None
        if parent is not None:
            parent.children.append(self)

    def __repr__(self) -> str:
        return f"<Scope {self.kind} {self.name!r}>"

    @property
    def top(self) -> Scope:
        """The module scope."""
        scope = self
        while scope.parent is not None:
            scope = scope.parent
        return scope

    @property
    def local_names(self) -> set[str]:
        """The names that are this scope's own: its parameters and what it
        binds, minus what it declares ``global`` or ``nonlocal``."""
        names = self.params | self.stores.keys()
        if self.declared_global or self.declared_nonlocal:
            names -= self.declared_global
            names -= self.declared_nonlocal
        return names

    @property
    def runs_later(self) -> bool:
        """Is this the body of a function or lambda, or inside one? Its code
        runs when the function is called, not with the statement."""
        scope: Scope | None = self
        while scope is not None:
            if scope.kind in (FUNCTION, LAMBDA):
                return True
            scope = scope.parent
        return False

    def owner(self, name: str) -> Scope:
        """The scope *name* belongs to when code of this scope uses it: this
        scope (a local), an enclosing function's (a free variable), or the
        module's (a global or builtin)."""
        if self.parent is None:
            return self
        if name in self.declared_global:
            return self.top
        if name not in self.declared_nonlocal and (name in self.params or name in self.stores):
            return self
        if name == "__class__" and self.kind != CLASS and self._in_class():
            return self  # the cell Python gives every method, read by `super()`
        scope = self.parent
        while scope.parent is not None:
            if name in scope.declared_global:
                return scope.top
            # A class body's names are not seen from the scopes inside it.
            if scope.kind != CLASS and name not in scope.declared_nonlocal:
                if name in scope.params or name in scope.stores:
                    return scope
            scope = scope.parent
        return scope

    def _in_class(self) -> bool:
        scope = self.parent
        while scope is not None:
            if scope.kind == CLASS:
                return True
            scope = scope.parent
        return False

    def walk(self) -> Iterator[Scope]:
        """This scope and every scope inside it, outermost first."""
        todo = [self]
        while todo:
            scope = todo.pop()
            yield scope
            todo.extend(reversed(scope.children))


class Event(NamedTuple):
    """One step of the code, in the order it runs (see :class:`ScopeModel`)."""

    kind: str
    #: The name read or bound; None for a :data:`SITE`.
    name: str | None
    node: ast.AST
    #: The scope whose code this is. For a binding, the scope the name is
    #: stored in: a ``:=`` in a comprehension stores in the scope around it.
    scope: Scope
    #: For a :data:`BIND`: by what.
    how: str | None = None


class ScopeModel:
    """The scopes of one tree and what happens in them.

    ``events`` lists every read, binding and site in the order the code runs
    them, with one exception: a ``def`` or ``class`` binds its name before
    its body is listed, since the body of a function runs after the name
    exists (a function that calls itself does not read its name from
    before the statement).
    """

    __slots__ = ("module", "events", "_by_node", "__weakref__")

    def __init__(self, module: Scope, events: list[Event], by_node: dict[int, Scope]) -> None:
        self.module = module
        self.events = events
        self._by_node = by_node

    def scope_of(self, node: ast.AST) -> Scope | None:
        """The scope *node* (a ``def``, lambda, class or comprehension of the
        tree) opens for its body."""
        return self._by_node.get(id(node))

    @property
    def scopes(self) -> Iterator[Scope]:
        return self.module.walk()

    def bindings(self, *, later: bool = False) -> dict[str, set[str]]:
        """The module-level names the tree binds, each with how.

        With ``later=False``, what running the code binds: at the top level,
        in a class body through ``global``, by a ``:=`` in a comprehension.
        With ``later=True``, what its functions bind through ``global`` when
        they are called. ``x: int`` alone binds nothing and is left out.
        """
        found: dict[str, set[str]] = {}
        for scope in self.scopes:
            if scope.runs_later is not later:
                continue
            for name, hows in scope.stores.items():
                if scope.owner(name).parent is None and hows != {DECLARED}:
                    found.setdefault(name, set()).update(hows - {DECLARED})
        return found

    def global_reads(self, *, annotations: bool = True) -> set[str]:
        """The names the tree reads from the module's namespace or the
        builtins, now or when a function of it is called. With
        ``annotations=False``, not those only an annotation names."""
        found: set[str] = set()
        for scope in self.scopes:
            if not annotations and scope.kind == ANNOTATION:
                continue
            for name in scope.reads:
                if scope.owner(name).parent is None:
                    found.add(name)
        return found

    @property
    def star_imports(self) -> list[str]:
        """The modules of every ``from m import *`` in the tree: each binds
        names the text does not show."""
        return [module for scope in self.scopes for module in scope.star_imports]


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------

# Tasks on the work list.
_VISIT, _TARGET, _BIND, _PATTERN = range(4)


class _Builder:
    def __init__(self, inline: Callable[[ast.Call], ast.Module | None] | None) -> None:
        self.module = Scope(MODULE, "top", None, None)
        self.events: list[Event] = []
        self.by_node: dict[int, Scope] = {}
        self.inline = inline
        self.todo: list[tuple] = []

    def build(self, tree: ast.AST) -> ScopeModel:
        self.module.node = tree
        todo = self.todo
        todo.append((_VISIT, tree, self.module, None))
        while todo:
            op, node, scope, how = todo.pop()
            if op == _VISIT:
                self.visit(node, scope)
            elif op == _BIND:
                self.bind(node, scope, how)
            elif op == _TARGET:
                self.target(node, scope, how)
            else:
                self.pattern(node, scope)
        return ScopeModel(self.module, self.events, self.by_node)

    # -- helpers -----------------------------------------------------------

    def then(self, *tasks: tuple) -> None:
        """Queue *tasks* to run next, in the order given."""
        self.todo.extend(reversed(tasks))

    def bind(self, at: tuple[str, ast.AST], scope: Scope, how: str) -> None:
        name, node = at
        scope.stores.setdefault(name, set()).add(how)
        self.events.append(Event(BIND, name, node, scope, how))

    def open(self, kind: str, name: str, node: ast.AST | None, parent: Scope, evaluated: str | None = None) -> Scope:
        scope = Scope(kind, name, node, parent)
        scope.evaluated = evaluated
        if node is not None and kind not in (ANNOTATION, TYPE_PARAMS):
            self.by_node[id(node)] = scope
        return scope

    @staticmethod
    def visits(nodes, scope: Scope) -> list[tuple]:
        return [(_VISIT, node, scope, None) for node in nodes if node is not None]

    # -- targets -----------------------------------------------------------

    def target(self, node: ast.AST, scope: Scope, how: str) -> None:
        """*node* is assigned to (or deleted): bind the names in it."""
        if isinstance(node, ast.Name):
            self.bind((node.id, node), scope, how)
        elif isinstance(node, (ast.Tuple, ast.List)):
            self.then(*[(_TARGET, elt, scope, how) for elt in node.elts])
        elif isinstance(node, ast.Starred):
            self.then((_TARGET, node.value, scope, how))
        else:  # ``a[i] = ...``, ``a.b = ...``: reads ``a`` and ``i``
            self.visit(node, scope)

    # -- match patterns ----------------------------------------------------

    def pattern(self, node: ast.AST, scope: Scope) -> None:
        if isinstance(node, ast.MatchValue):
            self.then((_VISIT, node.value, scope, None))
        elif isinstance(node, ast.MatchSequence) or isinstance(node, ast.MatchOr):
            self.then(*[(_PATTERN, p, scope, None) for p in node.patterns])
        elif isinstance(node, ast.MatchMapping):
            tasks = self.visits(node.keys, scope) + [(_PATTERN, p, scope, None) for p in node.patterns]
            if node.rest:
                tasks.append((_BIND, (node.rest, node), scope, MATCH))
            self.then(*tasks)
        elif isinstance(node, ast.MatchClass):
            self.then(
                (_VISIT, node.cls, scope, None),
                *[(_PATTERN, p, scope, None) for p in (*node.patterns, *node.kwd_patterns)],
            )
        elif isinstance(node, ast.MatchStar):
            if node.name:
                self.bind((node.name, node), scope, MATCH)
        elif isinstance(node, ast.MatchAs):
            tasks = [(_PATTERN, node.pattern, scope, None)] if node.pattern is not None else []
            if node.name:
                tasks.append((_BIND, (node.name, node), scope, MATCH))
            self.then(*tasks)
        # MatchSingleton: nothing to read or bind

    # -- nodes -------------------------------------------------------------

    def visit(self, node: ast.AST, scope: Scope) -> None:  # noqa: C901, PLR0912, PLR0915 - one branch per kind of node
        kind = type(node)
        if kind is ast.Name:
            if isinstance(node.ctx, ast.Load):
                scope.reads.add(node.id)
                self.events.append(Event(READ, node.id, node, scope))
            else:
                self.bind((node.id, node), scope, DELETE if isinstance(node.ctx, ast.Del) else ASSIGN)
        elif kind is ast.Constant:
            if scope.kind == ANNOTATION and scope.evaluated != QUOTED and isinstance(node.value, str):
                self.quoted_annotation(node, scope)
        elif kind in _SITES:
            self.events.append(Event(SITE, None, node, scope))
            if kind is ast.Call:
                self.call(node, scope)
            else:
                self.then(*self.visits(ast.iter_child_nodes(node), scope))
        elif kind in _FUNCTION_DEFS or kind is ast.Lambda:
            self.function(node, scope)
        elif kind is ast.ClassDef:
            self.class_def(node, scope)
        elif kind in _COMPREHENSIONS:
            self.comprehension(node, scope)
        elif kind is ast.NamedExpr:
            into = scope
            while into.kind == COMPREHENSION and into.parent is not None:
                into = into.parent
            self.then((_VISIT, node.value, scope, None), (_TARGET, node.target, into, WALRUS))
        elif kind is ast.Assign:
            self.then((_VISIT, node.value, scope, None), *[(_TARGET, target, scope, ASSIGN) for target in node.targets])
        elif kind is ast.AugAssign:
            self.events.append(Event(SITE, None, node, scope))
            if isinstance(node.target, ast.Name):
                name = node.target.id
                scope.reads.add(name)
                self.events.append(Event(READ, name, node.target, scope))
                self.then((_VISIT, node.value, scope, None), (_BIND, (name, node.target), scope, AUGMENTED))
            else:
                self.then((_VISIT, node.target, scope, None), (_VISIT, node.value, scope, None))
        elif kind is ast.AnnAssign:
            self.annotated_assignment(node, scope)
        elif kind is ast.For or kind is ast.AsyncFor:
            self.then(
                (_VISIT, node.iter, scope, None),
                (_TARGET, node.target, scope, FOR),
                *self.visits([*node.body, *node.orelse], scope),
            )
        elif kind is ast.With or kind is ast.AsyncWith:
            tasks: list[tuple] = []
            for item in node.items:
                tasks.append((_VISIT, item.context_expr, scope, None))
                if item.optional_vars is not None:
                    tasks.append((_TARGET, item.optional_vars, scope, WITH))
            self.then(*tasks, *self.visits(node.body, scope))
        elif kind is ast.Delete:
            self.then(*[(_TARGET, target, scope, DELETE) for target in node.targets])
        elif kind is ast.Import:
            for alias in node.names:
                self.bind((alias.asname or alias.name.split(".")[0], node), scope, IMPORT)
        elif kind is ast.ImportFrom:
            for alias in node.names:
                if alias.name == "*":
                    scope.star_imports.append("." * node.level + (node.module or ""))
                else:
                    self.bind((alias.asname or alias.name, node), scope, IMPORT)
        elif kind is ast.Global:
            scope.declared_global.update(node.names)
        elif kind is ast.Nonlocal:
            scope.declared_nonlocal.update(node.names)
        elif kind is ast.ExceptHandler:
            tasks = self.visits([node.type], scope)
            if node.name:
                tasks.append((_BIND, (node.name, node), scope, EXCEPT))
            self.then(*tasks, *self.visits(node.body, scope))
        elif kind is ast.Match:
            tasks = [(_VISIT, node.subject, scope, None)]
            for case in node.cases:
                tasks.append((_PATTERN, case.pattern, scope, None))
                tasks.extend(self.visits([case.guard, *case.body], scope))
            self.then(*tasks)
        elif _TYPE_ALIAS and kind is _TYPE_ALIAS:
            inner, params = self.type_params(node, scope)
            value = self.open(ANNOTATION, node.name.id, None, inner, LAZY)
            self.then((_TARGET, node.name, scope, TYPE_ALIAS), *params, (_VISIT, node.value, value, None))
        else:
            self.then(*self.visits(ast.iter_child_nodes(node), scope))

    def call(self, node: ast.Call, scope: Scope) -> None:
        tasks: list[tuple] = []
        if self.inline is not None and scope.parent is None:
            spliced = self.inline(node)
            if spliced is not None:
                # Code the call runs in the namespace it is called from.
                tasks.extend(self.visits(spliced.body, scope))
        tasks.append((_VISIT, node.func, scope, None))
        tasks.extend(self.visits(node.args, scope))
        tasks.extend(self.visits([keyword.value for keyword in node.keywords], scope))
        self.then(*tasks)

    def quoted_annotation(self, node: ast.Constant, scope: Scope) -> None:
        try:
            parsed = ast.parse(node.value.strip(), mode="eval")
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            return
        inner = self.open(ANNOTATION, scope.name, None, scope, QUOTED)
        self.then((_VISIT, parsed.body, inner, None))

    def type_params(self, node: ast.AST, scope: Scope) -> tuple[Scope, list[tuple]]:
        """The scope the type parameters of *node* live in (*scope* itself
        when it has none), and the tasks that bind each parameter and read
        its bound and default."""
        params = getattr(node, "type_params", None)
        if not params:
            return scope, []
        name = node.name.id if isinstance(node.name, ast.Name) else node.name
        inner = self.open(TYPE_PARAMS, name, node, scope)
        tasks: list[tuple] = []
        for param in params:
            tasks.append((_BIND, (param.name, param), inner, TYPE_PARAMETER))
            for part in (getattr(param, "bound", None), getattr(param, "default_value", None)):
                if part is not None:
                    tasks.append((_VISIT, part, self.open(ANNOTATION, param.name, param, inner, LAZY), None))
        return inner, tasks

    def function(self, node: ast.AST, scope: Scope) -> None:
        """A ``def`` or a lambda: decorators, defaults and annotations in the
        scope around it, then its name, then its body."""
        args = node.args
        is_def = not isinstance(node, ast.Lambda)
        inner, tasks = scope, []
        if is_def:
            self.events.append(Event(SITE, None, node, scope))
            inner, params = self.type_params(node, scope)
            tasks = self.visits(node.decorator_list, scope) + params
        tasks.extend(self.visits([*args.defaults, *args.kw_defaults], scope))
        every = [a for a in (*args.posonlyargs, *args.args, args.vararg, *args.kwonlyargs, args.kwarg) if a is not None]
        if is_def:
            annotations = [a.annotation for a in every] + [node.returns]
            if any(a is not None for a in annotations):
                hints = self.open(ANNOTATION, node.name, None, inner, EVALUATED)
                tasks.extend(self.visits(annotations, hints))
            tasks.append((_BIND, (node.name, node), scope, DEF))
        body = self.open(FUNCTION if is_def else LAMBDA, node.name if is_def else "lambda", node, inner)
        body.params.update(a.arg for a in every)
        tasks.extend(self.visits(node.body if is_def else [node.body], body))
        self.then(*tasks)

    def class_def(self, node: ast.ClassDef, scope: Scope) -> None:
        self.events.append(Event(SITE, None, node, scope))
        inner, params = self.type_params(node, scope)
        tasks = self.visits(node.decorator_list, scope) + params
        tasks.extend(self.visits([*node.bases, *(keyword.value for keyword in node.keywords)], inner))
        tasks.append((_BIND, (node.name, node), scope, CLASS_DEF))
        tasks.extend(self.visits(node.body, self.open(CLASS, node.name, node, inner)))
        self.then(*tasks)

    def comprehension(self, node: ast.AST, scope: Scope) -> None:
        first, *rest = node.generators
        inner = self.open(COMPREHENSION, type(node).__name__.lower(), node, scope)
        # The first iterable is evaluated in the scope around the comprehension.
        tasks: list[tuple] = [(_VISIT, first.iter, scope, None), (_TARGET, first.target, inner, COMPREHENSION_TARGET)]
        tasks.extend(self.visits(first.ifs, inner))
        for generator in rest:
            tasks.append((_VISIT, generator.iter, inner, None))
            tasks.append((_TARGET, generator.target, inner, COMPREHENSION_TARGET))
            tasks.extend(self.visits(generator.ifs, inner))
        parts = [node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]
        tasks.extend(self.visits(parts, inner))
        self.then(*tasks)

    def annotated_assignment(self, node: ast.AnnAssign, scope: Scope) -> None:
        tasks = self.visits([node.value], scope)
        within = scope
        while within.kind == COMPREHENSION and within.parent is not None:
            within = within.parent
        evaluated = NEVER if within.kind in (FUNCTION, LAMBDA) else EVALUATED
        name = node.target.id if isinstance(node.target, ast.Name) else ""
        tasks.append((_VISIT, node.annotation, self.open(ANNOTATION, name, node, scope, evaluated), None))
        if isinstance(node.target, ast.Name):
            tasks.append((_BIND, (node.target.id, node.target), scope, ASSIGN if node.value is not None else DECLARED))
        else:
            tasks.append((_TARGET, node.target, scope, ASSIGN))
        self.then(*tasks)


#: `scope_model`'s answers by the tree, held weakly: {inline: model}.
_MODELS: _weak.WeakKeyDictionary = _weak.WeakKeyDictionary()


def scope_model(tree: ast.AST, inline: Callable[[ast.Call], ast.Module | None] | None = None) -> ScopeModel:
    """The :class:`ScopeModel` of *tree*, built once per tree.

    *tree* may be a module, a statement, an expression or a single ``def``:
    whatever it is stands at the model's module level. *inline* names code a
    module-level call runs in the namespace it is called from
    (``exec("w = 1")``): it returns that code parsed, or None, and the code
    is walked where the call stands. The tree must not change afterwards.
    """
    try:
        models = _MODELS[tree]
    except KeyError:
        models = _MODELS[tree] = {}
    except TypeError:  # a node that cannot be weakly referenced
        return _Builder(inline).build(tree)
    model = models.get(inline)
    if model is None:
        model = models[inline] = _Builder(inline).build(tree)
    return model


def function_locals(node: ast.AST) -> set[str]:
    """The names the ``def`` or lambda *node* binds for itself: its
    parameters and what its own body assigns, imports, defines, catches or
    captures in a ``match``, minus what it declares ``global`` or
    ``nonlocal``. Not the names of the functions, classes and
    comprehensions inside it; a ``:=`` in a comprehension of its body is
    its own."""
    scope = scope_model(node).scope_of(node)
    return set(scope.local_names) if scope is not None else set()


def bound_names(tree: ast.AST) -> set[str]:
    """Every name *tree* binds or unbinds in any of its scopes, parameters
    and comprehension targets included: the names that may not mean, inside
    the tree, what they mean outside it."""
    found: set[str] = set()
    for scope in scope_model(tree).scopes:
        found |= scope.params
        found |= scope.stores.keys()
    return found
