"""The user's own code a statement reaches when it runs: the functions it
calls, the local modules whose code and data those functions read, and that
data itself.

A statement's inputs name what it mentions. ``m = mylib.mode()`` mentions
``mylib``; what decides its value is ``mode``'s body, the environment that
body reads and the data ``mylib`` holds. Both engines call this with the same
statement text and the live namespace, so they reach the same answer.
"""

from __future__ import annotations

import ast
import dis
import functools
import importlib
import inspect
import operator
import os
import sqlite3
import sys
import textwrap
import types
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from itertools import repeat
from typing import Any, NamedTuple

from .._memo import NOTEBOOK_STATEMENTS
from ..analysis.ast_util import parse_cached
from ..analysis.callee_effects import source_global_mutations
from ..analysis.handed_callables import handed_callable_values
from ..analysis.mutations import MUTATING_METHODS
from ..effects import ENVIRON_NAMES, dotted_name
from ..exceptions import SOURCE_RETRIEVAL_ERRORS
from ..analysis.helper_code import own_code_is_user
from ..tracking.function_tracker import is_local_module
from ..tracking.randomness import get_seeding_rng_modules
from ..value_types import IMMUTABLE_PRIMS, INTERPRETER_MANAGED_GLOBALS, is_runtime_machinery

__all__ = [
    "CWD",
    "ENVIRON",
    "Reach",
    "helper_seeded_modules",
    "module_globals",
    "module_holders",
    "module_state_names",
    "import_state_writes",
    "module_state_writes",
    "one_walk",
    "process_state_writes",
    "reached_user_code",
    "rebound_modules",
    "state_holders",
]


class Reach(NamedTuple):
    """What :func:`reached_user_code` found."""

    #: User functions the statement can call, in the order first met.
    functions: tuple[types.FunctionType, ...]
    #: Names of the local modules whose code or data those functions use.
    modules: frozenset[str]
    #: ``(label, value)`` of each piece of a local module's data the code
    #: reads, sorted by label: ``mylib.K`` read by ``mylib.from_k``,
    #: ``mylib.Settings.scale``. What the module's source says is not the
    #: whole story: ``mylib.K = 7`` in a cell, or ``mylib.set_k(7)``, changes
    #: what ``from_k`` returns and leaves the file as it was. A global one of
    #: the reached functions changes itself (``_counter += n``) is left out:
    #: keyed on its value before the call, the statement would key anew on
    #: every run, as a notebook global a called function changes is not.
    data: tuple[tuple[str, Any], ...] = ()


_EMPTY = Reach((), frozenset())

#: Inside `one_walk`: what each class was found to be (`_Found._class`),
#: keyed by the class and the namespace asked from. Empty outside it.
_WALKS: list[dict[tuple[str, type, int], bool]] = []


@contextmanager
def one_walk() -> Iterator[None]:
    """Decide each class a walk meets at most once inside the block.

    For a stretch where no code of the notebook runs -- the upstream
    simulation, which asks every cell above the one being checked what it
    reaches. A cell reading a pandas frame meets ``DataFrame``, whose 500
    members `_Found._users_class` reads for a function of the notebook's:
    in a 300-cell notebook of such cells that was 66-80 ms before every cell.
    Outside the block every walk decides again, as a cell may have given a
    class a method since (``pd.DataFrame.report = report``)."""
    _WALKS.append(_WALKS[-1] if _WALKS else {})
    try:
        yield
    finally:
        _WALKS.pop()



def reached_user_code(code: str, namespace: Mapping[str, Any] | None, *, close: bool = True) -> Reach:
    """The user functions *code* can call and the local modules it reaches.

    Followed: a name bound to a function, to a class (its ``__init__``), or
    to a local module; ``module.attr`` chains through local modules; the
    globals a notebook function reads when called, and ``module.attr`` in its
    body, transitively; and from each local module reached, the local modules
    its own names come from. A function belongs to the user when it was defined in *namespace*
    (a cell) or in their own code (`own_code_is_user`).

    Without *close*, the local modules the reached ones take their names
    from are left out: what calling the reached functions can change, not
    all the data they may lead to (`module_globals`).
    """
    if not code or not namespace:
        return _EMPTY
    steps = names_read(code)
    if not steps or all(_leads_nowhere(namespace, root, attrs) for root, attrs in steps):
        return _EMPTY
    found = _Found(namespace)
    for root, attrs in steps:
        if attrs:
            found.chain(root, attrs)
        else:
            found.value(namespace.get(root))
    if close:
        found.close_modules()
    data = tuple(sorted(item for item in found.data.items() if item[0] not in found.written))
    return Reach(tuple(found.functions), frozenset(found.modules), data)


def _leads_nowhere(namespace: Mapping[str, Any], root: str, attrs: tuple[str, ...]) -> bool:
    """Whether `_Found` would take nothing from the step ``root.attrs``:
    every value it meets is ``None``, a library module, a C routine
    (``print``, ``np.arange``), a C type or an instance of one
    (`_FOREIGN_C_TYPES`). Asked before a walk is set up: the upstream scan
    asks for every cell above on every cell, and most steps are ``x1``,
    ``np.arange`` or ``print``. Decides each value as `_Found.value` and
    `_Found.chain` would, in their order; anything else is walked."""
    obj = namespace.get(root)
    if not attrs:
        return _inert(obj)
    for attr in reversed(attrs):
        if not isinstance(obj, types.ModuleType):
            return True  # `_Found.chain` stops here
        if _is_local(obj):
            return False
        obj = vars(obj).get(attr)
        if not _inert(obj):
            return False
    return True


def _inert(value: Any) -> bool:
    """`_Found.value` of *value*, found with no label, adds nothing."""
    if isinstance(value, types.ModuleType):
        return not _is_local(value)
    if isinstance(value, (types.MethodType, types.FunctionType)):
        return False
    if isinstance(value, type):
        # A C type of no local module: its ``__init__`` is a slot, if any.
        return value in _FOREIGN_C_TYPES
    return value is None or type(value) in _FOREIGN_C_TYPES or inspect.isroutine(value)


@functools.lru_cache(maxsize=NOTEBOOK_STATEMENTS)
def names_read(code: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """What :func:`reached_user_code` looks up for *code*, in the order its
    walk meets it: ``(name, ())`` for a name read, ``(root, attrs)`` for
    each ``root.a.b`` chain (``attrs`` innermost first); empty when *code*
    does not parse.

    The text alone decides it, so it is found once per text. The upstream
    scan asks for every cell above on every cell: walking each cell's tree
    again was 0.14 s of a cell at the end of a 400-cell notebook.
    """
    tree = parse_cached(code)
    if tree is None:
        return ()
    steps: list[tuple[str, tuple[str, ...]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            steps.append((node.id, ()))
        elif isinstance(node, ast.Attribute):
            attrs: list[str] = []
            root: ast.expr = node
            while isinstance(root, ast.Attribute):
                attrs.append(root.attr)
                root = root.value
            if isinstance(root, ast.Name):
                steps.append((root.id, tuple(attrs)))
    return tuple(steps)


def module_state_writes(code: str, namespace: Mapping[str, Any] | None) -> frozenset[str]:
    """The local modules whose state *code* sets, by name.

    ``mylib.K = 7``, ``mylib.CONFIG["k"] = 7``, ``mylib.K += 1``,
    ``del mylib.K``, ``setattr(mylib, "K", 7)``, ``mylib.REGISTRY.update(...)``
    and a call of a module function that changes the module's globals
    (``mylib.set_k(7)``, or ``set_k(7)`` imported from it), or of a notebook
    function whose body does any of these, and ``importlib.reload(mylib)``,
    which sets every global anew. The same through a name holding
    what the module holds (``CONFIG["k"] = 7`` after ``from mylib import
    CONFIG``, ``cfg["k"] = 7`` after ``cfg = mylib.CONFIG``), through a bare
    decorator (``@mylib.register``), and through a method of the module's
    class whose body stores on the class (``cls.k = k``, ``type(self).k =
    k``) or on an instance the module holds. A method of the user's is
    followed into its body; only a library's method (``dict.update``) is
    judged by its name. A reload runs the module's top level again and drops
    all of these; the notebook's cells that made them are what puts them
    back. A ``def`` sets nothing: its body runs when the function is called.
    """
    if not code or not namespace:
        return frozenset()
    tree = parse_cached(code)
    if tree is None:
        return frozenset()
    found: set[str] = set()
    _state_writes(tree.body, namespace, found, set())
    return frozenset(found)


#: What a statement can change of the process the notebook runs in, which a
#: kernel restart puts back as the shell started it: its environment
#: variables, and its working directory.
ENVIRON = "environ"
CWD = "cwd"

#: Methods of ``os.environ`` that change it.
_ENVIRON_SETTING_METHODS = frozenset({"update", "pop", "popitem", "setdefault", "clear", "__setitem__", "__delitem__"})
#: The functions that change the environment or the working directory, and
#: how they are spelled where the namespace cannot say (after a restart).
_PROCESS_SETTERS: tuple[tuple[Any, str], ...] = (
    (os.chdir, CWD),
    (getattr(os, "fchdir", None), CWD),
    (os.putenv, ENVIRON),
    (getattr(os, "unsetenv", None), ENVIRON),
)
_PROCESS_SPELLINGS = {"os.chdir": CWD, "os.fchdir": CWD, "os.putenv": ENVIRON, "os.unsetenv": ENVIRON}


def process_state_writes(code: str, namespace: Mapping[str, Any] | None) -> frozenset[str]:
    """What of the process *code* changes: `ENVIRON` for ``os.environ["K"] =
    v``, ``del os.environ["K"]``, ``os.environ.update(...)``, ``os.putenv``;
    `CWD` for ``os.chdir(d)``. Written in it, or in a function it calls: a
    notebook function is followed into its body (``setup()`` doing
    ``os.environ["MODE"] = "b"``), and so is the user's module function and
    the helpers it calls (``mylib.go(d)`` doing ``os.chdir(d)``), as for the
    state of a module (`module_state_writes`).

    A kernel restart puts both back as the shell started them; the
    statements that changed them are what puts the notebook's values back.
    """
    if not code:
        return frozenset()
    tree = parse_cached(code)
    if tree is None:
        return frozenset()
    process: set[str] = set()
    _state_writes(tree.body, namespace or {}, None, set(), process)
    return frozenset(process)


def _is_environ(expr: ast.expr, namespace: Mapping[str, Any]) -> bool:
    """Whether *expr* is the process environment: ``os.environ``, an alias
    of it, or (with nothing in the namespace to say) one spelled so."""
    value, _ = _static_value(expr, namespace)
    if value is not None:
        return value is os.environ or value is getattr(os, "environb", None)
    return dotted_name(expr) in ENVIRON_NAMES


def _process_call(func: ast.expr, callee: Any, namespace: Mapping[str, Any]) -> str | None:
    """`ENVIRON` or `CWD` when calling *func* (resolved to *callee*) changes
    that of the process."""
    if callee is not None:
        for setter, kind in _PROCESS_SETTERS:
            if setter is not None and callee is setter:
                return kind
    if isinstance(func, ast.Attribute) and func.attr in _ENVIRON_SETTING_METHODS and _is_environ(func.value, namespace):
        return ENVIRON
    return _PROCESS_SPELLINGS.get(dotted_name(func) or "") if callee is None else None


def _state_writes(
    statements: Iterable[ast.AST],
    namespace: Mapping[str, Any],
    found: set[str] | None,
    followed: set[int],
    process: set[str] | None = None,
    *,
    follow: bool = True,
) -> None:
    """Add to *found* (unless None) the local modules *statements* set state
    on, run with *namespace* as their globals; *followed* are the notebook
    functions already walked. With *process*, add to it what of the process
    they change (`process_state_writes`). Without *follow*, a call is judged by
    what it calls, not followed into a body.

    This is the one walk that follows a statement into the notebook
    functions it calls: whatever else a statement's helpers can change is
    looked for here, so the bodies are read once, the same way."""

    def rooted(node: ast.expr) -> None:
        """Add the local module the store target *node* sets something on:
        through the module (``mylib.CONFIG["k"]``), or through a name holding
        what the module holds (``CONFIG["k"]`` after ``from mylib import
        CONFIG``, ``cfg["k"]`` after ``cfg = mylib.CONFIG``, ``Cfg.k``)."""
        if found is None or not isinstance(node, (ast.Attribute, ast.Subscript)):
            return
        changed_in_place(node.value)

    def changed_in_place(root: ast.expr) -> None:
        """Add the local module whose state changes when *root* is changed in place."""
        if found is None:
            return
        while isinstance(root, (ast.Attribute, ast.Subscript)):
            root = root.value
        if isinstance(root, ast.Name):
            home = _state_home(namespace.get(root.id), namespace)
            if home is not None:
                found.add(home)

    def called(func: ast.expr, node: ast.Call | None) -> None:
        """What calling *func* sets state on: the function it names, followed
        into its body -- a method or a class method of a local module's class
        too (``Cfg.tune(5)`` doing ``cls.k = k``) -- or, for a method no
        Python code of the user's runs (``CONFIG.update(...)``), the
        receiver when the method's name says it changes it."""
        callee, kind, receiver = _resolve_call(func, namespace)
        if process is not None:
            changed = _process_call(func, callee, namespace)
            if changed is not None:
                process.add(changed)
        users = followed_into(callee, kind, receiver)
        if node is None or users or found is None:
            return
        if isinstance(func, ast.Name) and func.id in ("setattr", "delattr") and node.args:
            target = namespace.get(node.args[0].id) if isinstance(node.args[0], ast.Name) else None
            home = _state_home(target, namespace)
            if home is not None:
                found.add(home)
        elif isinstance(func, ast.Attribute) and func.attr in MUTATING_METHODS:
            changed_in_place(func.value)

    def followed_into(callee: Any, kind: str, receiver: Any) -> bool:
        """Follow the function *callee* into its body; whether it is the
        user's (its body, not its name, says what it changes)."""
        users = isinstance(callee, types.FunctionType) and _is_users_function(callee, namespace)
        if isinstance(callee, types.FunctionType) and follow:
            modules = _modules_changed_by(callee, namespace, process)
            if found is not None:
                found.update(modules)
                if users:
                    found.update(_class_state_written(callee, kind, receiver, namespace))
            if callee.__globals__ is namespace and id(callee) not in followed:
                followed.add(id(callee))
                _state_writes(_function_body(callee), namespace, found, followed, process)
        return users

    def handed(node: ast.Call) -> None:
        """Follow what the call is handed that the callee calls: a function
        of a local module handed to ``s.apply`` runs as much as one called by
        name (``s.apply(helpers.record)``, ``s.map(count)``)."""
        for value in handed_callable_values(node, namespace):
            followed_into(*_as_handed_callee(value))

    pending = list(statements)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # A bare decorator (``@mylib.register``) is a call of what it
            # names, made when the definition runs.
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    called(decorator, None)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            # What runs at definition: the decorators and the defaults.
            pending.extend(getattr(node, "decorator_list", ()))
            pending.extend(d for d in (*node.args.defaults, *node.args.kw_defaults) if d is not None)
            continue
        pending.extend(ast.iter_child_nodes(node))
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
            for target in targets:
                for part in ast.walk(target):
                    if isinstance(part, ast.expr) and isinstance(getattr(part, "ctx", None), (ast.Store, ast.Del)):
                        rooted(part)
                        if (
                            process is not None
                            and isinstance(part, ast.Subscript)
                            and _is_environ(part.value, namespace)
                        ):
                            process.add(ENVIRON)
        elif isinstance(node, ast.Call):
            if found is not None and node.args and _static_value(node.func, namespace)[0] is importlib.reload:
                # A reload runs the module's top level again: every global it
                # has is set anew, whatever the cells above set on it.
                target = _static_value(node.args[0], namespace)[0]
                if isinstance(target, types.ModuleType) and _is_local(target):
                    found.add(target.__name__)
            called(node.func, node)
            if node.args or node.keywords:
                handed(node)


def _as_handed_callee(value: Any) -> tuple[Any, str, Any]:
    """``(function, kind, receiver)`` calling the handed callable *value*
    runs, as `_resolve_call` names a call target."""
    if isinstance(value, types.MethodType):
        receiver = value.__self__
        return value.__func__, "classmethod" if isinstance(receiver, type) else "method", receiver
    if isinstance(value, type):
        try:
            return inspect.getattr_static(value, "__init__"), "method", value
        except AttributeError:
            return None, "function", None
    if isinstance(value, types.FunctionType):
        return value, "function", None
    try:
        return inspect.getattr_static(type(value), "__call__"), "method", value
    except AttributeError:
        return None, "function", None


def import_state_writes(code: str, namespace: Mapping[str, Any] | None) -> frozenset[str]:
    """The local modules, other than the ones it imports, whose state the
    import statements in *code* set by running the imported modules' top
    level: ``import plugin`` where ``plugin.py`` does ``@mylib.register``.

    Asked after the statement ran, when what it imported is loaded: a
    restart drops the registration with ``mylib``, and the import is one of
    the statements that put it there."""
    if not code or "import" not in code or not namespace:
        return frozenset()
    tree = parse_cached(code)
    if tree is None:
        return frozenset()
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.append(node.module)
            names.extend(f"{node.module}.{alias.name}" for alias in node.names if alias.name != "*")
    found: set[str] = set()
    imported: set[str] = set()
    for name in names:
        module = sys.modules.get(name)
        if not isinstance(module, types.ModuleType) or not _is_local(module):
            continue
        imported.add(module.__name__)
        try:
            body = parse_cached(inspect.getsource(module))
        except SOURCE_RETRIEVAL_ERRORS:
            continue
        if body is not None:
            _state_writes(body.body, vars(module), found, set())
    return frozenset(found - imported)


def _is_users_function(fn: types.FunctionType, namespace: Mapping[str, Any]) -> bool:
    """Defined in a cell (its globals are *namespace*) or in a local module:
    its body says what it changes, not its name."""
    if fn.__globals__ is namespace:
        return True
    home = _loaded(fn.__globals__.get("__name__"))
    return home is not None and _is_local(home)


def _state_home(value: Any, namespace: Mapping[str, Any]) -> str | None:
    """The local module whose state a store into *value* changes: *value*
    is the module, a class defined in it, or an object one of its globals
    holds (by identity: ``from mylib import CONFIG``, ``cfg =
    mylib.CONFIG``). None for anything else -- a value of the notebook's own,
    an instance it made of the module's class."""
    if value is None or isinstance(value, (*IMMUTABLE_PRIMS, tuple, frozenset)):
        return None
    if isinstance(value, types.ModuleType):
        return value.__name__ if _is_local(value) else None
    if inspect.isroutine(value):
        return None
    if isinstance(value, type):
        home = _loaded(getattr(value, "__module__", None))
        return home.__name__ if home is not None and vars(home) is not namespace and _is_local(home) else None
    # Every local module loaded but the one whose globals *namespace* is:
    # ``from mylib import CONFIG`` alone leaves no other trace of ``mylib``.
    for module in _local_modules():
        # By identity, at C speed: a module may hold hundreds of names.
        if vars(module) is not namespace and any(map(operator.is_, list(vars(module).values()), repeat(value))):
            return module.__name__
    return None


#: ``[stamp, entries, modules]``: the ``sys.modules`` stamp the local
#: modules were found at, the ``(key, module)`` entries they were found
#: under, and the modules (each once).
_LOCAL_MODULES: list[Any] = [None, (), ()]


def _local_modules() -> tuple[types.ModuleType, ...]:
    """The local modules loaded, found again only when ``sys.modules`` has
    another size or another last entry, or an entry they were found under
    holds another object now: asked for a store through any name that is
    not a module (``df["a"] = 1``), it must not walk the thousand library
    modules each time.

    Checked by the key each was found under, not by its ``__name__``: a
    module can sit under another key than its name, and the name's own
    entry then holds something else for good. ``multiprocessing`` keeps a
    script's ``__main__`` as ``__mp_main__``, and IPython puts its own
    ``__main__`` in its place, so a check by name found the cache stale on
    every call and walked every loaded module again, per statement."""
    modules = sys.modules
    try:
        last = next(reversed(modules.keys()))
    except (StopIteration, RuntimeError):
        last = None
    stamp = (len(modules), last)
    if _LOCAL_MODULES[0] != stamp or any(modules.get(key) is not module for key, module in _LOCAL_MODULES[1]):
        entries = tuple(
            (key, module)
            for key, module in list(modules.items())
            if isinstance(module, types.ModuleType) and _is_local(module)
        )
        _LOCAL_MODULES[1] = entries
        _LOCAL_MODULES[2] = tuple({id(module): module for _key, module in entries}.values())
        _LOCAL_MODULES[0] = stamp
    return _LOCAL_MODULES[2]


def _resolve_call(func: ast.expr, namespace: Mapping[str, Any]) -> tuple[Any, str, Any]:
    """``(callee, kind, receiver)`` of the call target *func*: the function
    it runs, ``"function"``, ``"classmethod"`` (*receiver* the class) or
    ``"method"`` (*receiver* the instance, or the class when only the class
    is known: ``mylib.Cfg().setk(5)``). Attributes are looked up statically:
    no property or ``__getattr__`` runs."""
    if isinstance(func, ast.Name):
        return _as_callee(namespace.get(func.id), None, False)
    if not isinstance(func, ast.Attribute):
        return None, "function", None
    owner, instance_of = _static_value(func.value, namespace)
    if instance_of is not None:
        try:
            member = inspect.getattr_static(instance_of, func.attr)
        except AttributeError:
            return None, "function", None
        return _as_callee(member, instance_of, True)
    if owner is None:
        return None, "function", None
    if isinstance(owner, types.ModuleType):
        return _as_callee(vars(owner).get(func.attr), None, False)
    try:
        member = inspect.getattr_static(owner, func.attr)
    except Exception:  # noqa: BLE001 - an object's attribute lookup
        return None, "function", None
    if isinstance(owner, type):
        return _as_callee(member, owner, False)
    return _as_callee(member, owner, True)


def _as_callee(member: Any, owner: Any, on_instance: bool) -> tuple[Any, str, Any]:
    if isinstance(member, classmethod):
        cls = owner if isinstance(owner, type) else type(owner)
        return member.__func__, "classmethod", cls
    if isinstance(member, staticmethod):
        return member.__func__, "function", None
    if isinstance(member, types.MethodType):
        receiver = member.__self__
        kind = "classmethod" if isinstance(receiver, type) else "method"
        return member.__func__, kind, receiver
    if isinstance(member, types.FunctionType) and on_instance:
        return member, "method", owner
    return member, "function", None


def _static_value(expr: ast.expr, namespace: Mapping[str, Any]) -> tuple[Any, type | None]:
    """``(value, None)`` of *expr* (a name or a chain of attributes from
    one), or ``(None, cls)`` when it is a call of the class *cls* -- an
    instance of it, not made here -- or ``(None, None)`` when unknown."""
    if isinstance(expr, ast.Call):
        value, _ = _static_value(expr.func, namespace)
        return (None, value) if isinstance(value, type) else (None, None)
    attrs: list[str] = []
    while isinstance(expr, ast.Attribute):
        attrs.append(expr.attr)
        expr = expr.value
    if not isinstance(expr, ast.Name):
        return None, None
    value = namespace.get(expr.id)
    for attr in reversed(attrs):
        if isinstance(value, types.ModuleType):
            value = vars(value).get(attr)
            continue
        if not isinstance(value, type) and not _of_a_local_class(value):
            return None, None
        try:
            value = inspect.getattr_static(value, attr)
        except Exception:  # noqa: BLE001 - an object's attribute lookup
            return None, None
        if isinstance(value, (classmethod, staticmethod, property)) or not _plain_attribute(value):
            return None, None
    return value, None


def _of_a_local_class(value: Any) -> bool:
    """An instance whose class a local module defines: its attributes are
    followed (``mylib.CFG.sub.setk``)."""
    home = _loaded(getattr(type(value), "__module__", None))
    return home is not None and _is_local(home)


def _plain_attribute(value: Any) -> bool:
    """A value held in a ``__dict__``, not a descriptor that computes one."""
    return not hasattr(type(value), "__get__") or isinstance(value, (types.FunctionType, type))


def _class_state_written(fn: types.FunctionType, kind: str, receiver: Any, namespace: Mapping[str, Any]) -> set[str]:
    """The local module whose state calling the method *fn* sets: a class
    method storing on ``cls``, a method storing on ``type(self)`` or
    ``self.__class__``, of a class a local module defines; or a method
    storing on ``self`` when the instance is one a local module holds
    (``mylib.CFG.setk(5)``)."""
    if kind not in ("classmethod", "method") or receiver is None:
        return set()
    cls = receiver if isinstance(receiver, type) else type(receiver)
    stores = _receiver_stores(fn.__code__)
    if not stores:
        return set()
    found: set[str] = set()
    on_class = "class" in stores if kind == "method" else ("self" in stores or "class" in stores)
    if on_class:
        home = _state_home(cls, namespace)
        if home is not None:
            found.add(home)
    if kind == "method" and "self" in stores and not isinstance(receiver, type):
        home = _state_home(receiver, namespace)
        if home is not None:
            found.add(home)
    return found


@functools.lru_cache(maxsize=4096)
def _receiver_stores(code: types.CodeType) -> frozenset[str]:
    """What the method whose code is *code* stores attributes or items on
    through its first parameter: ``"self"`` for ``self.k = v`` (``cls.k =
    v`` in a class method), ``"class"`` for ``type(self).k = v`` and
    ``self.__class__.k = v``."""
    if not code.co_varnames or code.co_argcount < 1:
        return frozenset()
    first = code.co_varnames[0]
    try:
        tree = parse_cached(textwrap.dedent(inspect.getsource(code)))
    except SOURCE_RETRIEVAL_ERRORS:
        return frozenset()
    if tree is None:
        return frozenset()
    found: set[str] = set()

    def target_root(node: ast.expr) -> None:
        while isinstance(node, (ast.Attribute, ast.Subscript)):
            inner = node.value
            if isinstance(inner, ast.Name) and inner.id == first:
                if isinstance(node, ast.Attribute) and node.attr == "__class__":
                    found.add("class")
                else:
                    found.add("self")
                return
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id == "type"
                and len(inner.args) == 1
                and isinstance(inner.args[0], ast.Name)
                and inner.args[0].id == first
            ):
                found.add("class")
                return
            node = inner

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
            for target in targets:
                for part in ast.walk(target):
                    if isinstance(part, (ast.Attribute, ast.Subscript)) and isinstance(part.ctx, (ast.Store, ast.Del)):
                        target_root(part)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("setattr", "delattr")
            and node.args
        ):
            target_root(ast.Attribute(value=node.args[0], attr="_", ctx=ast.Store()))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in MUTATING_METHODS:
            target_root(node.func)
    return frozenset(found)


def helper_seeded_modules(
    code: str,
    namespace: Mapping[str, Any] | None,
    def_source: Callable[[str], str | None] | None = None,
) -> frozenset[str]:
    """The RNG modules *code* seeds through a notebook function it calls.

    ``set_seed(42)``, with ``def set_seed(s): random.seed(s);
    np.random.seed(s)`` in a cell, seeds ``random`` and ``numpy.random`` as
    surely as the two calls written out, so it writes their RNG variables as
    they would. Followed: a call of a function defined in a cell (its globals
    are *namespace*), and the notebook functions its body calls in turn. A
    name not bound yet (the simulation after a restart reaches the call before
    the kernel ran the ``def``) is looked up with *def_source*, which gives
    the source of the ``def`` the simulation saw. A ``def`` seeds nothing: its
    body runs when the function is called. Both engines call this with the
    same statement text and the live namespace.
    """
    if not code or (not namespace and def_source is None):
        return frozenset()
    tree = parse_cached(code)
    if tree is None:
        return frozenset()
    namespace = namespace or {}
    found: set[str] = set()
    followed: set[str] = set()
    pending: list[ast.AST] = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        pending.extend(ast.iter_child_nodes(node))
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        if name in followed:
            continue
        followed.add(name)
        source = _notebook_function_source(name, namespace, def_source)
        if source is None:
            continue
        seeded, calls = _seeds_and_calls(source)
        found |= seeded
        pending.extend(ast.Call(func=ast.Name(id=called, ctx=ast.Load()), args=[], keywords=[]) for called in calls)
    return frozenset(found)


def _notebook_function_source(
    name: str, namespace: Mapping[str, Any], def_source: Callable[[str], str | None] | None
) -> str | None:
    """The source of the notebook function *name*, or None when it is not one."""
    fn = namespace.get(name)
    if fn is None:
        return def_source(name) if def_source is not None else None
    if not isinstance(fn, types.FunctionType) or fn.__globals__ is not namespace:
        return None
    return _code_source(fn.__code__)


@functools.lru_cache(maxsize=1024)
def _code_source(code: types.CodeType) -> str | None:
    """The source of the function whose code is *code*: a def run again
    makes a new code object, so an edited one is read again."""
    try:
        return textwrap.dedent(inspect.getsource(code))
    except SOURCE_RETRIEVAL_ERRORS:
        return None


@functools.lru_cache(maxsize=1024)
def _seeds_and_calls(source: str) -> tuple[frozenset[str], tuple[str, ...]]:
    """What the function defined by *source* seeds itself, and the plain
    names it calls (the notebook functions among them are followed)."""
    try:
        seeded = frozenset(get_seeding_rng_modules(source))
        tree = ast.parse(source)
    except (SyntaxError, ValueError, AttributeError, RecursionError):
        return frozenset(), ()
    calls = sorted(
        {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    )
    return seeded, tuple(calls)


def _function_body(fn: types.FunctionType) -> list[ast.stmt]:
    """The statements of *fn*'s body, from its source; none when it has none.

    Read once per code object (`_code_body`): every statement that calls a
    notebook function walks its body, for the module state and again for
    the process state it sets, and reading the source is what costs."""
    try:
        target = inspect.unwrap(fn)
    except ValueError:  # a __wrapped__ cycle
        target = fn
    code = getattr(target, "__code__", None)
    if not isinstance(code, types.CodeType):
        return []
    return list(_code_body(code))


@functools.lru_cache(maxsize=1024)
def _code_body(code: types.CodeType) -> tuple[ast.stmt, ...]:
    """The body statements of the function whose code is *code*: a def run
    again makes a new code object, so an edited one is read again."""
    source = _code_source(code)
    tree = parse_cached(source) if source is not None else None
    if tree is None:
        return ()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return tuple(node.body)
    return ()


def module_state_names(code: str, namespace: Mapping[str, Any] | None, *, structure: bool = False) -> frozenset[str]:
    """The names that see the state of a local module *code* sets state on
    (`module_state_writes`, `module_holders`): ``mylib`` for ``mylib.K =
    k``, ``mylib.cfg["a"] = v``, ``mylib.CACHE.append(x)`` or
    ``mylib.set_k(k)``, under every name the namespace holds it by, and
    ``from_k`` taken from it by ``from mylib import from_k``.

    The module is a value of the notebook that ``import mylib`` made and
    these statements change in place, so each of them is one of its
    producers: its lineage moves with them, as a list's moves with
    ``items.append(x)``. A reload of the edited module puts back the
    import's state only, and the upstream check rebuilds the rest from these
    producers, the way it rebuilds any variable.

    A loop or branch is answered for as a whole only with *structure*: it
    changes the module as it changes a list it appends to, through the
    names its body changes (``control_structure_mutations``), so both
    engines ask from there, never as of a plain statement.
    """
    if not code or not namespace:
        return frozenset()
    if not structure:
        from .control_structures.common import is_control_structure  # noqa: PLC0415 - imports this module

        tree = parse_cached(code)
        if tree is not None and len(tree.body) == 1 and is_control_structure(tree.body[0]):
            return frozenset()
    mentioned = {root for root, _ in names_read(code)}
    # Only a module, a function that may call into one, or what a module
    # holds (``CONFIG`` after ``from mylib import CONFIG``) can lead there.
    if not any(_may_lead_to_module_state(namespace.get(root), namespace) for root in mentioned):
        return frozenset()
    modules = module_state_writes(code, namespace)
    if not modules:
        return frozenset()
    return state_holders(modules, code, namespace)


def _may_lead_to_module_state(value: Any, namespace: Mapping[str, Any]) -> bool:
    if isinstance(value, (types.ModuleType, types.FunctionType, types.MethodType, type)):
        return True
    return _state_home(value, namespace) is not None or _of_a_local_class(value)


def state_holders(modules: Iterable[str], code: str, namespace: Mapping[str, Any]) -> frozenset[str]:
    """The names of *namespace* through which a statement *code* that sets
    state on the local *modules* changes what the notebook sees
    (`module_holders`). *code* is kept for the callers' symmetry."""
    del code
    names: set[str] = set()
    for module in modules:
        value = sys.modules.get(module)
        if isinstance(value, types.ModuleType):
            names |= module_holders(value, namespace)
    return frozenset(names)


def module_holders(module: types.ModuleType, namespace: Mapping[str, Any]) -> frozenset[str]:
    """The names of *namespace* that see the state of *module*: the module
    itself under any name, and what ``from mylib import ...`` took from it
    that reads or is that state -- a function or class defined in it
    (``from_k`` reads ``K``) and a mutable object it holds (``CFG``).

    A statement that sets state on the module changes each of them, as
    ``items.append(x)`` changes every name holding ``items``: a cell reading
    ``from_k(2)`` depends on ``set_k(5)`` above it as surely as one reading
    ``mylib.from_k(2)``. ``from mylib import K`` took a copy: setting
    ``mylib.K`` later leaves it as it was, so it is none of them.
    """
    own = vars(module)
    data_ids = {
        id(value)
        for value in list(own.values())
        if not isinstance(value, (types.ModuleType, type, *IMMUTABLE_PRIMS, tuple, frozenset))
        and not inspect.isroutine(value)
    }
    names: set[str] = set()
    for name, held in list(namespace.items()):
        if held is module:
            if name == module.__name__ or not name.startswith("_"):
                names.add(name)
        elif name.startswith("_"):
            continue
        elif isinstance(held, types.FunctionType):
            if held.__globals__ is own:
                names.add(name)
        elif isinstance(held, type):
            if getattr(held, "__module__", None) == module.__name__:
                names.add(name)
        elif id(held) in data_ids:
            names.add(name)
    return frozenset(names)


def module_globals(code: str, namespace: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """What the globals of each local module *code* reaches hold before it
    runs, to tell after it which of them it rebound (`rebound_modules`).

    ``module_state_writes`` reads a write in the text: ``global K`` in a
    function, ``CFG["k"] = v``. One it cannot read -- ``globals()[name] =
    v``, ``setattr(sys.modules[__name__], ...)``, ``global K`` in a method
    of an instance's class -- is seen by running it. Empty for an import or
    a reload, which put the module's own top level in place, and for a
    statement that reaches no local module.
    """
    if not code or not namespace:
        return {}
    tree = parse_cached(code)
    if tree is None or "get_ipython()" in code:
        return {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return {}
        if isinstance(node, ast.Call) and _call_name(node.func) == "reload":
            return {}
    try:
        modules = reached_user_code(code, namespace, close=False).modules
    except Exception:  # noqa: BLE001 - an analysis of arbitrary code
        return {}
    found: dict[str, dict[str, Any]] = {}
    for name in modules:
        module = sys.modules.get(name)
        if isinstance(module, types.ModuleType) and vars(module) is not namespace:
            found[name] = dict(vars(module))
    return found


_ABSENT = object()


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def rebound_modules(before: Mapping[str, Mapping[str, Any]]) -> frozenset[str]:
    """The modules of *before* (`module_globals`) whose globals now hold
    another object than they did, or one more or fewer."""
    changed = set()
    for name, held in before.items():
        module = sys.modules.get(name)
        if not isinstance(module, types.ModuleType):
            continue
        now = vars(module)
        # By identity, at C speed: *held* keeps every object it saw alive,
        # so an id cannot be reused for another one meanwhile.
        if (now.keys() != held.keys() or not all(map(operator.is_, now.values(), held.values()))) and any(
            key not in INTERPRETER_MANAGED_GLOBALS and now.get(key, _ABSENT) is not held.get(key, _ABSENT)
            for key in now.keys() | held.keys()
        ):
            changed.add(name)
    return frozenset(changed)


def _modules_changed_by(
    fn: types.FunctionType, namespace: Mapping[str, Any], process: set[str] | None = None
) -> set[str]:
    """The local modules whose globals calling *fn* changes: its own, or
    those of the helpers it calls, at any depth. ``mylib.add(5)`` calling
    ``_bump(5)``, which does ``global COUNT; COUNT += n``, changes ``mylib``
    as surely as a body that does it itself. A notebook function is followed
    to the module functions it calls; its own writes are the notebook's.
    With *process*, add to it what of the process each module function
    reached changes (`process_state_writes`)."""
    found: set[str] = set()
    seen: set[int] = set()
    pending = [fn]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        home_ns = current.__globals__
        if home_ns is not namespace:
            home = _loaded(home_ns.get("__name__"))
            if home is None or not _is_local(home):
                continue
            if _changed_globals(current.__code__):
                found.add(home.__name__)
            if process is not None:
                process.update(_module_function_process_writes(current))
        for co in _code_objects(current.__code__):
            for name in _loaded_globals(co):
                value = home_ns.get(name)
                if isinstance(value, types.FunctionType):
                    pending.append(value)
                elif isinstance(value, types.ModuleType) and _is_local(value):
                    attrs = vars(value)
                    pending.extend(attrs[a] for a in co.co_names if isinstance(attrs.get(a), types.FunctionType))
    return found


@functools.lru_cache(maxsize=4096)
def _module_function_process_writes(fn: types.FunctionType) -> frozenset[str]:
    """What of the process the body of the module function *fn* itself
    changes (`process_state_writes`), its calls judged by what they call:
    `_modules_changed_by` reaches its helpers. Once per function: asked for
    every statement that calls it."""
    process: set[str] = set()
    _state_writes(_function_body(fn), fn.__globals__, None, set(), process, follow=False)
    return frozenset(process)


class _Found:
    """What the walk has found so far, deduplicated."""

    def __init__(self, namespace: Mapping[str, Any]) -> None:
        self.namespace = namespace
        self.functions: list[types.FunctionType] = []
        self._function_ids: set[int] = set()
        self.modules: set[str] = set()
        self.data: dict[str, Any] = {}
        #: ``module.name`` of each global a reached module function changes.
        self.written: set[str] = set()

    def value(self, value: Any, label: str | None = None) -> None:
        """Take in *value*, found at *label* (``module.name``) when it is a
        name of a local module."""
        if isinstance(value, types.ModuleType):
            if _is_local(value):
                self.modules.add(value.__name__)
        elif isinstance(value, types.MethodType):
            self.value(value.__func__)
        elif isinstance(value, type):
            self._class(value)
        elif isinstance(value, types.FunctionType) and self._is_user(value):
            if id(value) in self._function_ids:
                return
            self._function_ids.add(id(value))
            self.functions.append(value)
            self._owner(value.__globals__.get("__name__"))
            if value.__globals__ is self.namespace:
                self._body_reads(value.__code__)
            else:
                self._module_body_reads(value)
            # A decorator's wrapper reaches the function it wraps through
            # its closure (``def w(*a): return f(*a)``).
            for cell in value.__closure__ or ():
                try:
                    self.value(cell.cell_contents)
                except ValueError:  # an empty cell
                    pass
            self._wrapped(value)
        elif isinstance(value, types.FunctionType):
            # A wrapper that is not the user's code (``@cash.cache``'s): what
            # calling it runs is the function it wraps.
            self._wrapped(value)
        else:
            if label is not None and _is_data(value):
                self.data.setdefault(label, value)
            cls = type(value)
            if value is None or cls in _FOREIGN_C_TYPES or self._known_to_lead_nowhere(cls):
                return
            if not inspect.isroutine(value):
                # An instance: its methods run when the statement calls them.
                self._class(cls)

    def _wrapped(self, value: Any) -> None:
        """The function *value* wraps (``functools.wraps``' ``__wrapped__``)."""
        try:
            wrapped = inspect.getattr_static(value, "__wrapped__", None)
        except Exception:  # noqa: BLE001 - an object's attribute lookup
            return
        if isinstance(wrapped, types.FunctionType):
            self.value(wrapped)

    def _class(self, cls: type) -> None:
        """A class: its ``__init__``; when it is the user's, its methods and
        those it inherits from the user's classes (``model.predict(2)`` runs
        ``Model.predict``, and what that reads), and when it is a local
        module's, the data it holds (``Settings.scale``), which its methods
        read through ``self``."""
        memo = _WALKS[-1] if _WALKS else None
        if memo is not None:
            # Whether the class leads anywhere, told by a walk of its own: one
            # that finds nothing finds nothing in this walk either, which
            # only holds more already.
            key = ("leads", cls, id(self.namespace))
            leads = memo.get(key)
            if leads is None:
                # Taken to lead somewhere while it is told, so that a class
                # met again inside its own walk is walked, not told again.
                memo[key] = True
                alone = _Found(self.namespace)
                alone._walk_class(cls)
                leads = memo[key] = bool(alone.functions or alone.modules or alone.data or alone.written)
            if not leads:
                return
        self._walk_class(cls)

    def _known_to_lead_nowhere(self, cls: type) -> bool:
        """Inside `one_walk`: *cls* is known to add nothing to a walk."""
        return bool(_WALKS) and _WALKS[-1].get(("leads", cls, id(self.namespace))) is False

    def _walk_class(self, cls: type) -> None:
        if not self._users_class(cls):
            self.value(vars(cls).get("__init__"))
            return
        if id(cls) in self._function_ids:
            return
        self._function_ids.add(id(cls))
        home = _loaded(getattr(cls, "__module__", None))
        if home is not None and _is_local(home):
            self.modules.add(home.__name__)
            for attr, attr_value in list(vars(cls).items()):
                if not attr.startswith("__") and _is_data(attr_value):
                    self.data.setdefault(f"{home.__name__}.{cls.__qualname__}.{attr}", attr_value)
        for klass in cls.__mro__:
            if not self._users_class(klass):
                continue
            for attr_value in list(vars(klass).values()):
                for fn in _class_member_functions(attr_value):
                    self.value(fn)

    def _users_class(self, cls: type) -> bool:
        """Defined in a local module, or in a cell: its methods' globals are
        the namespace. Once per class inside `one_walk`."""
        memo = _WALKS[-1] if _WALKS else None
        if memo is None:
            return self._decide_users_class(cls)
        key = ("users", cls, id(self.namespace))
        known = memo.get(key)
        if known is None:
            known = memo[key] = self._decide_users_class(cls)
        return known

    def _decide_users_class(self, cls: type) -> bool:
        home = _loaded(getattr(cls, "__module__", None))
        if home is not None and _is_local(home):
            return True
        if _is_c_type(cls):
            # No ``class`` statement made it, so no cell's function is its
            # member. Asked of every instance a statement reads (an int, an
            # array): walking ``vars(np.ndarray)`` for each, for every cell
            # above, grew a cell at the end of a 400-cell notebook by 60 ms.
            _FOREIGN_C_TYPES.add(cls)
            return False
        return any(
            fn.__globals__ is self.namespace
            for member in list(vars(cls).values())
            for fn in _class_member_functions(member)
            if isinstance(fn, types.FunctionType)
        )

    def chain(self, root: str, attrs: tuple[str, ...]) -> None:
        """``mod.sub.attr`` (*root* ``mod``, *attrs* innermost first): each
        local module on the way, and what it ends at."""
        obj = self.namespace.get(root)
        for attr in reversed(attrs):
            if not isinstance(obj, types.ModuleType):
                return
            label = f"{obj.__name__}.{attr}" if _is_local(obj) else None
            obj = vars(obj).get(attr)
            self.value(obj, label)

    def _body_reads(self, code: types.CodeType) -> None:
        """What a notebook function reads when it is called: the globals its
        body names, and ``mod.f`` for a local module it names. Its globals
        are resolved when it runs, so a function defined below the caller
        counts as well."""
        stack = [code]
        while stack:
            co = stack.pop()
            stack.extend(c for c in co.co_consts if isinstance(c, types.CodeType))
            for name in co.co_names:
                value = self.namespace.get(name)
                self.value(value)
                if isinstance(value, types.ModuleType) and _is_local(value):
                    self._module_attributes(value, co.co_names)

    def _module_body_reads(self, fn: types.FunctionType) -> None:
        """What a function of a local module reads when it is called: the
        module's globals its code loads, and ``mod.attr`` of a local module it
        names. Its helpers are followed through :meth:`value`."""
        home = fn.__globals__
        home_name = home.get("__name__")
        self.written.update(f"{home_name}.{name}" for name in _changed_globals(fn.__code__))
        for co in _code_objects(fn.__code__):
            for name in _loaded_globals(co):
                if name.startswith("__") or name not in home:
                    continue
                value = home[name]
                self.value(value, f"{home_name}.{name}")
                if isinstance(value, types.ModuleType) and _is_local(value):
                    self._module_attributes(value, co.co_names)

    def _module_attributes(self, module: types.ModuleType, names: Iterable[str]) -> None:
        """``module.attr`` for each of *names* the local *module* has."""
        namespace = vars(module)
        for attr in names:
            if attr in namespace and not attr.startswith("__"):
                self.value(namespace[attr], f"{module.__name__}.{attr}")

    def close_modules(self) -> None:
        """Add the local modules the reached ones take their names from."""
        pending = list(self.modules)
        while pending:
            module = _loaded(pending.pop())
            if module is None:
                continue
            for value in list(vars(module).values()):
                for name in _local_homes(value):
                    if name not in self.modules:
                        self.modules.add(name)
                        pending.append(name)

    def _owner(self, module_name: Any) -> None:
        module = _loaded(module_name)
        if module is not None and _is_local(module):
            self.modules.add(module.__name__)

    def _is_user(self, fn: types.FunctionType) -> bool:
        # No root package: a library function is never the user's because
        # of the package it is in, only because it lives in their files.
        return fn.__globals__ is self.namespace or own_code_is_user(fn, None)


#: C types of no local module: an instance of one leads nowhere (`_Found._class`
#: takes only its ``__init__``, a slot wrapper), so it is not asked again.
_FOREIGN_C_TYPES: set[type] = set()

_HEAPTYPE = 1 << 9
_IMMUTABLETYPE = 1 << 8


def _is_c_type(cls: type) -> bool:
    """Was *cls* made by C code rather than a ``class`` statement? A static
    type, or a heap type an extension created immutable; Python cannot make
    either, nor give one a method."""
    flags = getattr(cls, "__flags__", 0)
    return not flags & _HEAPTYPE or bool(flags & _IMMUTABLETYPE)


def _class_member_functions(member: Any) -> Iterable[Any]:
    """The functions a class attribute runs: a method, a static or class
    method, a property's accessors."""
    if isinstance(member, (staticmethod, classmethod)):
        yield member.__func__
    elif isinstance(member, property):
        yield from (f for f in (member.fget, member.fset, member.fdel) if f is not None)
    elif isinstance(member, types.FunctionType):
        yield member


def _is_data(value: Any) -> bool:
    """Whether *value* is data a function reads, rather than code or a
    module, which other channels key, or a handle no result is computed
    from: a lock, a logger, a stream (`is_runtime_machinery`) or a database
    connection, whose answers are read through it, not held in it. Hashed
    by identity, such a handle gave the statement a new key in every
    process."""
    return not (
        value is None
        or isinstance(value, (types.ModuleType, type, sqlite3.Connection, sqlite3.Cursor))
        or inspect.isroutine(value)
        or is_runtime_machinery(value)
        or getattr(value, "_is_file_tracker_patch", False) is True
    )


def _code_objects(code: types.CodeType) -> Iterable[types.CodeType]:
    """*code* and the code nested in it: comprehensions, inner functions."""
    stack = [code]
    while stack:
        co = stack.pop()
        yield co
        stack.extend(c for c in co.co_consts if isinstance(c, types.CodeType))


@functools.lru_cache(maxsize=4096)
def _loaded_globals(code: types.CodeType) -> frozenset[str]:
    """The global names *code* loads (not its attribute names)."""
    try:
        return frozenset(
            ins.argval
            for ins in dis.get_instructions(code)
            if ins.opname in ("LOAD_GLOBAL", "LOAD_NAME") and isinstance(ins.argval, str)
        )
    except (TypeError, ValueError):
        return frozenset(code.co_names)


@functools.lru_cache(maxsize=4096)
def _changed_globals(code: types.CodeType) -> frozenset[str]:
    """The module globals the function whose code is *code* changes:
    ``global K; K = v``, ``CONFIG["k"] = v``."""
    try:
        return source_global_mutations(inspect.getsource(code))
    except SOURCE_RETRIEVAL_ERRORS:
        return frozenset()


def _loaded(name: Any) -> types.ModuleType | None:
    return sys.modules.get(name) if isinstance(name, str) else None


def _is_local(module: types.ModuleType) -> bool:
    try:
        return is_local_module(module)
    except (TypeError, AttributeError):
        return False


def _local_homes(value: Any) -> Iterable[str]:
    """The local module *value* is, or was defined in."""
    if isinstance(value, types.ModuleType):
        if _is_local(value):
            yield value.__name__
        return
    if isinstance(value, (types.FunctionType, type)):
        home = value.__globals__.get("__name__") if isinstance(value, types.FunctionType) else value.__module__
        module = _loaded(home)
        if module is not None and _is_local(module):
            yield module.__name__
