"""A user module a function body imports, keyed before it has been imported.

``from helpmod import scale`` inside a cached function binds a local the
helper walk follows only once ``helpmod`` is loaded. Importing it early to
look would run its code, which the body may never get to (a branch not
taken, a ``__setstate__`` only a rebuild runs), and that code may write
files or change state. So the module is found and read WITHOUT running it:
its spec through the import system's finders (`_origin`), its source and the
source of every user module that source imports (`source_digest`). The
files' bytes stand for the helpers they define and for the values their top
level binds, which holds only when every such value follows from the source
alone (`_SourceOnly`): a literal, a ``def``, a class, a call cash knows reads
nothing outside its arguments. Anything else -- a value read from the
environment or a file, a call into code whose effects cash does not know --
makes the digest None: the key cannot vouch for the module, and the call
runs with a warning.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import importlib.util
import os
import sys
import types

from .._user_attrs import attribute_or
from ..install_paths import in_own_package, top_package
from ..tracking.function_tracker import is_local_module
from ..value_types import BUILTIN_NAMES

#: Calls at a module's top level whose result follows from their arguments
#: alone. A call not listed makes the module's values unvouched for.
_SOURCE_ONLY_CALLS = frozenset(
    {
        *(
            f"builtins.{n}"
            for n in (
                "abs all any bool bytes chr classmethod complex dict divmod enumerate filter float format frozenset "
                "int isinstance issubclass len list map max min object ord pow property range repr reversed round "
                "set slice sorted staticmethod str sum tuple type zip"
            ).split()
        ),
        "abc.abstractmethod",
        "collections.Counter",
        "collections.OrderedDict",
        "collections.defaultdict",
        "collections.deque",
        "collections.namedtuple",
        "contextlib.contextmanager",
        "dataclasses.dataclass",
        "dataclasses.field",
        "decimal.Decimal",
        "enum.auto",
        "enum.unique",
        "fractions.Fraction",
        "functools.cache",
        "functools.cached_property",
        "functools.lru_cache",
        "functools.partial",
        "functools.singledispatch",
        "functools.total_ordering",
        "functools.wraps",
        "logging.getLogger",
        "re.compile",
        "typing.NamedTuple",
        "typing.NewType",
        "typing.ParamSpec",
        "typing.TypeVar",
        "typing.TypedDict",
        "typing.cast",
        "typing.final",
        "typing.overload",
        "typing.runtime_checkable",
        "cash.cache",
        "cash.pure",
        # numpy's array makers and reductions: functions of their
        # arguments. Not ``numpy.empty``, whose items are whatever memory held.
        *(
            f"numpy.{n}"
            for n in (
                "abs all any arange array asarray concatenate cumsum dot exp eye full identity linspace log max "
                "mean min ones prod sqrt stack sum where zeros"
            ).split()
        ),
    }
)
#: Modules whose every function is a function of its arguments.
_SOURCE_ONLY_MODULES = frozenset({"math", "operator", "string"})
#: Functions of `_SOURCE_ONLY_MODULES` that look up an attribute by the name
#: they are given, as ``getattr`` does: ``operator.attrgetter('__import__')``
#: on ``builtins`` reaches ``os.environ``. Only with public names written out.
_NAME_TAKING = frozenset({"operator.attrgetter", "operator.methodcaller", "operator.call"})
#: Modules whose attributes describe the machine, the process or the moment.
_AMBIENT_MODULES = frozenset(
    {
        "os",
        "sys",
        "time",
        "datetime",
        "random",
        "uuid",
        "secrets",
        "socket",
        "platform",
        "getpass",
        "tempfile",
        "shutil",
        "glob",
        "subprocess",
        "pathlib",
        "locale",
        "io",
    }
)
#: ``sys`` attributes fixed for an interpreter, which the key already holds.
_INTERPRETER_FACTS = frozenset({"sys.version_info", "sys.platform", "sys.maxsize", "sys.byteorder"})
#: More user modules than this behind one import: not read through.
_MAX_FILES = 200

#: ``(module, own package)`` -> ``(digest, what was looked at)`` (`source_digest`).
_DIGESTS: dict[tuple[str, str | None], tuple[str | None, tuple]] = {}


def is_user_code(module_name: str, root_module: str | None) -> bool:
    """Is *module_name* user code, decided WITHOUT importing it or its parents?"""
    if in_own_package(module_name, top_package(root_module)):
        return True
    return _users_file(_origin(module_name))


def _users_file(origin: str | None) -> bool:
    """Is *origin* a file of the user's code? Bytecode with no source beside
    it is judged by where it is, so that `source_digest`, which has no source
    to read, leaves it unvouched rather than taking it for a library."""
    if origin and origin.endswith((".pyc", ".pyo")):
        origin = origin[:-1]
    return bool(origin) and is_local_module(types.SimpleNamespace(__file__=origin))


def unloaded_import(module_name: str, prefix: tuple[str, ...], root_module: str | None) -> str | None:
    """The user module a function-body import would load that is not loaded
    yet, or None: *module_name* itself, or for ``from pkg import sub`` the
    submodule ``pkg.sub`` when ``pkg`` is loaded without it."""
    module = sys.modules.get(module_name)
    if module is None:
        if prefix and _namespace_package(module_name):
            # ``from nsp import mod`` with ``nsp`` a folder with no
            # ``__init__.py``: the module the import loads is ``nsp.mod``.
            sub = f"{module_name}.{prefix[0]}"
            return sub if _origin(sub) is not None and is_user_code(sub, root_module) else None
        return module_name if is_user_code(module_name, root_module) else None
    # Its own ``__dict__``: ``hasattr`` runs a package's ``__getattr__``
    # (PEP 562), which imports the very submodule asked about.
    if not prefix or prefix[0] in (attribute_or(module, "__dict__", None) or ()):
        return None
    sub = f"{module_name}.{prefix[0]}"
    if sub in sys.modules or _origin(sub) is None:
        return None
    return sub if is_user_code(sub, root_module) else None


def source_digest(module_name: str, root_module: str | None) -> str | None:
    """A digest of the source of user module *module_name*, of the packages
    importing it runs, and of every user module that source imports, read
    without running any of it; None when one of them cannot be vouched for
    that way (module docstring).

    Asked on every call until the body imports the module, so the answer is
    kept while every file it read is where it was, unchanged, and every name
    it found no file for still has none."""
    own_pkg = top_package(root_module)
    slot = (module_name, own_pkg)
    kept = _DIGESTS.get(slot)
    if kept is not None and all(
        (name is None or _origin(name) == origin) and _stat(origin) == st for name, origin, st in kept[1]
    ):
        return _with_live_values(kept[0], kept[2])
    digest, looked, taken = _source_digest(module_name, own_pkg)
    if len(_DIGESTS) > 1024:
        _DIGESTS.clear()
    _DIGESTS[slot] = (digest, looked, taken)
    return _with_live_values(digest, taken)


def _with_live_values(digest: str | None, taken: tuple[tuple[str, str | None], ...]) -> str | None:
    """*digest*, the source's, with the values the source takes from a user
    module loaded already: importing it runs no source of that module but
    takes what it holds now, which a statement since its import may have
    changed (``conf.BASE = 5``). ``from m import name`` takes *name*'s value;
    ``import m`` the whole module, every name it binds. None when one cannot
    be digested."""
    if digest is None or not taken:
        return digest
    from ..tracking.trace_items import value_digest

    parts = []
    for module, name in taken:
        mod = sys.modules.get(module)
        ns = getattr(mod, "__dict__", None) if mod is not None else None
        if not isinstance(ns, dict):
            # Not loaded: importing it runs its source, read above.
            continue
        if name is not None:
            value = ns.get(name, _NOT_BOUND)
            if value is _NOT_BOUND or (isinstance(value, types.ModuleType) and value.__name__ == f"{module}.{name}"):
                # A submodule (or nothing yet): read as a module of its own.
                continue
            values = [(name, value)]
        else:
            values = sorted((k, v) for k, v in list(ns.items()) if isinstance(k, str) and not k.startswith("__"))
        for key, value in values:
            if isinstance(value, types.ModuleType):
                # A module it imported: its own values are taken by name.
                got = "module:" + str(getattr(value, "__name__", ""))
            else:
                got = value_digest(value)
                if got.startswith("opaque:"):
                    return None
            parts.append(f"{module}:{key}={got}")
    if not parts:
        return digest
    blob = digest + "\n" + "\n".join(parts)
    return hashlib.blake2b(blob.encode(), digest_size=16).hexdigest()


_NOT_BOUND = object()


def _source_digest(module_name: str, own_pkg: str | None) -> tuple[str | None, tuple, tuple]:
    files: dict[str, str] = {}
    #: ``(module, name)`` the sources import from a user module, name None
    #: for the module itself (`_with_live_values`).
    taken: dict[tuple[str, str | None], None] = {}
    looked: list[tuple[str, str | None, tuple | None]] = []
    todo = [module_name]
    seen: set[str] = set()
    #: loaded packages read only as the parent of a module below them.
    parents_only: set[str] = set()
    while todo:
        name = todo.pop()
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            prefix = ".".join(parts[:i])
            if prefix in seen:
                continue
            if i < len(parts) and prefix in sys.modules:
                # A loaded package on the way to a module below it: importing
                # that module does not run the package again, so nothing of its
                # top level is the import's to vouch for. Its file is keyed, so
                # an edit to it still moves the key; read in full only when a
                # source names the package itself (``from pkg import VALUE``).
                if prefix not in parents_only:
                    parents_only.add(prefix)
                    origin = _origin(prefix)
                    looked.append((prefix, origin, _stat(origin)))
                    if origin is not None:
                        files[f"{prefix}#file"] = _file_digest(origin)
                continue
            seen.add(prefix)
            origin = _origin(prefix)
            looked.append((prefix, origin, _stat(origin)))
            if origin is None:
                if prefix == module_name:
                    if _namespace_package(prefix):
                        return None, tuple(looked), ()
                    # No finder knows it: the import fails. Keyed by the files
                    # that could become it (`_absent_digest`).
                    absent = _absent_digest(prefix, looked)
                    if absent is None:
                        return None, tuple(looked), ()
                    files[prefix] = absent
                    continue
                if _namespace_package(prefix):
                    # A folder with no ``__init__.py``: no code of its own to
                    # read; the modules in it are read as the parts below.
                    continue
                # Not on disk to import: a name `from pkg import name` reads off
                # the package, or a module the body's import would fail on.
                break
            if not (in_own_package(prefix, own_pkg) or _users_file(origin)):
                # A library: followed no further, as the helper walk does not.
                break
            if not origin.endswith(".py") or len(files) >= _MAX_FILES:
                return None, tuple(looked), ()
            try:
                with open(origin, "rb") as fh:
                    source = fh.read()
                tree = ast.parse(source)
            except (OSError, SyntaxError, ValueError):
                return None, tuple(looked), ()
            is_package = os.path.basename(origin) == "__init__.py"
            reader = _SourceOnly(prefix if is_package else prefix.rpartition(".")[0])
            if not reader.module(tree):
                return None, tuple(looked), ()
            files[prefix] = hashlib.blake2b(source, digest_size=16).hexdigest()
            todo.extend(reader.imported)
            taken.update(dict.fromkeys(reader.taken))
    if module_name not in files:
        # The module itself was not read: its source cannot stand for it.
        return None, tuple(looked), ()
    blob = "\n".join(f"{name}={digest}" for name, digest in sorted(files.items()))
    users = [(m, n) for m, n in taken if m != module_name and is_user_code(m, own_pkg)]
    return hashlib.blake2b(blob.encode(), digest_size=16).hexdigest(), tuple(looked), tuple(users)


def _file_digest(path: str) -> str:
    try:
        with open(path, "rb") as fh:
            return hashlib.blake2b(fh.read(), digest_size=16).hexdigest()
    except OSError:
        return "unreadable"


def _absent_digest(name: str, looked: list) -> str | None:
    """A stable digest for module *name*, which no finder knows: the files
    in its parent's search path that could become it -- an unbuilt ``.pyx``,
    its C source, a stale extension -- by name and bytes, and the folders'
    listings through *looked*, so that creating, building or editing one
    moves the key. A finder installed later (``pyximport``) turns the
    ``.pyx`` into the module: its bytes are in the key already. None when
    the search path is not known."""
    parent, _, last = name.rpartition(".")
    path = _locate(parent)[1] if parent else sys.path
    if path is None:
        return None
    try:
        entries = [p for p in path if isinstance(p, str)]
    except TypeError:
        return None
    found = []
    for entry in entries:
        folder = entry or os.getcwd()
        # The folder's own stat: a file created in it moves its mtime.
        looked.append((None, folder, _stat(folder)))
        try:
            names = sorted(os.listdir(folder))
        except OSError:
            continue
        for fname in names:
            if fname == last or fname.startswith(f"{last}."):
                full = os.path.join(folder, fname)
                looked.append((None, full, _stat(full)))
                found.append(f"{fname}:{_file_digest(full) if os.path.isfile(full) else 'dir'}")
    return "absent:" + hashlib.blake2b("\n".join(found).encode(), digest_size=16).hexdigest()


def _stat(path: str | None) -> tuple | None:
    if path is None:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def module_file(name: str) -> str | None:
    """The file module *name* is, or would be, loaded from, found without
    running any module (`_origin`)."""
    return _origin(name)


def _origin(name: str) -> str | None:
    """The file module *name* is (or would be) loaded from, found without
    running any module: a loaded parent's ``__path__``, else the finders'
    spec for each part in turn. ``importlib.util.find_spec`` would import
    the parents of a dotted name, running their ``__init__``."""
    return _locate(name)[0]


def _namespace_package(name: str) -> bool:
    """Is *name* a namespace package: found, with a search path but no file?"""
    origin, path = _locate(name)
    return origin is None and path is not None


def _locate(name: str) -> tuple[str | None, object]:
    """``(file, search path)`` of module *name*, as `_origin` finds them;
    ``(None, None)`` when it is not found."""
    path = None
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        full = ".".join(parts[:i])
        mod = sys.modules.get(full)
        if mod is not None:
            if i == len(parts):
                origin = getattr(mod, "__file__", None)
                return (origin if isinstance(origin, str) else None), getattr(mod, "__path__", None)
            path = getattr(mod, "__path__", None)
        else:
            spec = _find_spec(full, path)
            if spec is None:
                return None, None
            if i == len(parts):
                origin = spec.origin if spec.has_location and isinstance(spec.origin, str) else None
                return origin, spec.submodule_search_locations
            path = spec.submodule_search_locations
        if path is None:
            return None, None
    return None, None


def _find_spec(name: str, path) -> importlib.machinery.ModuleSpec | None:
    for finder in sys.meta_path:
        if (type(finder).__module__ or "").startswith("cash."):
            # cash's own hooks look a spec up through `importlib.util.find_spec`.
            continue
        find = getattr(finder, "find_spec", None)
        if find is None:
            continue
        try:
            spec = find(name, path)
        except Exception:  # noqa: BLE001 - a finder that cannot say has not found it
            return None
        if spec is not None:
            return spec
    return None


class _SourceOnly:
    """Does every value a module's top level binds follow from its source?

    Statements a module or class body runs at import are judged; a function
    body runs only when called, and its code is in the source. ``imported``
    collects the modules the source imports, anywhere in it.
    """

    def __init__(self, package: str, depth: int = 0) -> None:
        self.package = package
        self.depth = depth
        self.imported: list[str] = []
        #: ``(module, name)`` each import takes: a ``from`` import's names,
        #: name None for a module ``import`` binds whole.
        self.taken: list[tuple[str, str | None]] = []
        #: local name -> qualified name, from the source's imports.
        self.aliases: dict[str, str] = {}
        #: names the top level defines with ``def`` or ``class``.
        self.defined: set[str] = set()
        #: names the top level binds in its own scope (``for input in ...``).
        self.bound: dict[str, int] = {}
        #: names the class or function body being read binds in its own
        #: scope (a parameter ``id``), innermost last.
        self.scopes: list[dict[str, int]] = []
        #: whether each of `scopes` is a class body's, whose names a
        #: comprehension inside it does not see.
        self.class_scopes: list[bool] = []
        #: names the module's own imports bind, not one inside a function or
        #: class; and the function and class bodies being read, innermost last.
        self.module_aliases: set[str] = set()
        self.import_scopes: list[ast.AST] = []
        #: the top level's functions, by name.
        self.functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        #: user functions asked about (`_vouches`), and the answers.
        self.vouched: dict[str, bool] = {}
        #: inside the body of a user function called at import (`_runs_source_only`).
        self.in_body = 0

    def module(self, tree: ast.Module) -> bool:
        """Does *tree*, a module's source, bind only what follows from it,
        and read nothing from outside it anywhere, a function body included?"""
        self.names(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and (_calls_an_importer(node) or self._reads_outside(node)):
                return False
            if isinstance(node, (ast.Name, ast.Attribute)) and self._qualified(node) in _OUTSIDE_VALUES:
                return False
            if isinstance(node, ast.Name) and self._qualified(node) in _RUNS_OR_READS_UNSEEN:
                # ``map(eval, ...)``, ``map(open, paths)``: called all the same.
                return False
            if isinstance(node, ast.Name) and node.id == "__builtins__":
                # ``map(__builtins__['eval'], ...)``: every builtin, by any name.
                return False
        return self._body(tree.body)

    def names(self, tree: ast.Module) -> None:
        """What the names of *tree* are bound to by its imports and its top level."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self._note_import(node)
        self.module_aliases = _import_names(tree)
        self.taken = self._taken(tree)
        self.defined = {
            n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        self.functions = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.bound = _scope_names(tree)

    def _taken(self, tree: ast.AST) -> list[tuple[str, str | None]]:
        """What the imports that run at import take, a function body's left
        out: ``(module, name)`` for ``from module import name``, name None
        for a module bound whole (``import a.b`` binds ``a``, through which
        ``a.b`` is read)."""
        out: list[tuple[str, str | None]] = []
        todo = list(ast.iter_child_nodes(tree))
        while todo:
            node = todo.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            todo.extend(ast.iter_child_nodes(node))
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    out.extend((".".join(parts[: i + 1]), None) for i in range(len(parts)))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    try:
                        module = importlib.util.resolve_name("." * node.level + module, self.package or None)
                    except (ImportError, ValueError):
                        continue
                if module:
                    out.extend((module, None if a.name == "*" else a.name) for a in node.names)
        return out

    def _reads_outside(self, node: ast.Call) -> bool:
        """Does *node* read or change something outside the source (the
        clock, the environment, a file, the network), as `classify_call`
        judges a function body's calls -- by the name its imports give it?"""
        from ..effects import EffectKind, classify_call

        effect = classify_call(node, None)
        if effect is None:
            name = self._qualified(node.func)
            if name is None or name.startswith(("own:", "builtins.")) or name == _spelled(node.func):
                return False
            try:
                spelled = ast.parse(name, mode="eval").body
            except SyntaxError:
                return False
            effect = classify_call(ast.Call(func=spelled, args=node.args, keywords=node.keywords), None)
        return effect is not None and effect.kind is not EffectKind.CONSOLE

    def _note_import(self, node: ast.Import | ast.ImportFrom) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                self.imported.append(alias.name)
                if alias.asname:
                    self.aliases[alias.asname] = alias.name
                else:
                    top = alias.name.split(".")[0]
                    self.aliases[top] = top
            return
        module = node.module or ""
        if node.level:
            try:
                module = importlib.util.resolve_name("." * node.level + module, self.package or None)
            except (ImportError, ValueError):
                return
        if not module:
            return
        self.imported.append(module)
        for alias in node.names:
            if alias.name != "*":
                # A submodule as often as not: `source_digest` reads it if one is there.
                self.imported.append(f"{module}.{alias.name}")
                self.aliases[alias.asname or alias.name] = f"{module}.{alias.name}"

    def _body(self, body: list[ast.stmt]) -> bool:
        return all(self._statement(s) for s in body)

    def _statement(self, node: ast.stmt) -> bool:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Pass)):
            return True
        if isinstance(node, ast.Expr):
            return isinstance(node.value, ast.Constant) or self._expr(node.value)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            defaults = [*args.defaults, *(d for d in args.kw_defaults if d is not None)]
            return all(self._decorator(d) for d in node.decorator_list) and all(map(self._expr, defaults))
        if isinstance(node, ast.ClassDef):
            return (
                all(self._decorator(d) for d in node.decorator_list)
                and all(map(self._expr, node.bases))
                and all(self._expr(k.value) for k in node.keywords)
                and self._in_scope(node, lambda: self._body(node.body))
            )
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not all(_binds_names(t) for t in targets):
                return False
            return node.value is None or self._expr(node.value)
        if isinstance(node, ast.If):
            if _is_main_guard(node.test):
                return self._body(node.orelse)
            return self._expr(node.test) and self._body(node.body) and self._body(node.orelse)
        if isinstance(node, ast.Try):
            return (
                self._body(node.body)
                and all((h.type is None or self._expr(h.type)) and self._body(h.body) for h in node.handlers)
                and self._body(node.orelse)
                and self._body(node.finalbody)
            )
        return False

    def _decorator(self, node: ast.expr) -> bool:
        target = node.func if isinstance(node, ast.Call) else node
        name = self._qualified(target)
        if name is None:
            return False
        if isinstance(node, ast.Call) and not all(map(self._expr, [*node.args, *(k.value for k in node.keywords)])):
            return False
        return name in _SOURCE_ONLY_CALLS or self._vouches(name)

    def _expr(self, node: ast.expr) -> bool:
        if isinstance(node, _COMPREHENSIONS) and self.class_scopes and self.class_scopes[-1]:
            # Its own scope, which the class body's names do not reach:
            # ``getattr`` in it is the module's, or the builtin. All but its
            # first iterable, which runs in the class body before the scope
            # is entered: ``[x for x in map(getattr, ...)]`` reads the
            # class's ``getattr = abs``.
            if not self._expr(node.generators[0].iter):
                return False
            self.import_scopes.append(_NO_IMPORTS)
            try:
                return self._scoped({}, False, lambda: self._comprehension_body(node))
            finally:
                self.import_scopes.pop()
        if isinstance(node, ast.Lambda):
            # Runs only when called; its code is in the source.
            return all(map(self._expr, [*node.args.defaults, *(d for d in node.args.kw_defaults if d is not None)]))
        if isinstance(node, (ast.NamedExpr, ast.Await, ast.Yield, ast.YieldFrom)):
            # In the body of a function called at import these run as its
            # other expressions do; at the top level they are not followed.
            return bool(self.in_body) and (node.value is None or self._expr(node.value))
        if isinstance(node, ast.Call):
            name = self._qualified_here(node.func)
            if name in _NAME_TAKING and not _public_names_only(name, node):
                return False
            if name is None or not (
                name in _SOURCE_ONLY_CALLS
                or name.partition(".")[0] in _SOURCE_ONLY_MODULES
                and "." in name
                or self._vouches(name)
            ):
                return False
            return all(map(self._expr, [*node.args, *(k.value for k in node.keywords)]))
        if isinstance(node, (ast.Attribute, ast.Name)):
            name = self._qualified_here(node)
            if name is not None and name.partition(".")[0] in _AMBIENT_MODULES and name not in _INTERPRETER_FACTS:
                return False
            if name in _NAME_TAKING:
                # Handed on as a value (``map(operator.attrgetter, ...)``),
                # the names it is given are not seen.
                return False
            if (
                name is not None
                and name.startswith("builtins.")
                and name not in _BUILTIN_VALUES
                and not _plain_method(name)
                and not (_head(node) not in self.aliases and self._binds(_head(node), node))
            ):
                # A builtin handed on as a value (``map(eval, ...)``) is
                # called all the same: only those `_SOURCE_ONLY_CALLS` lists,
                # by its whole name (``object.__getattribute__`` is not
                # ``object``). A name an import bound is no rebinding away
                # from it: ``import builtins`` or ``from builtins import
                # getattr`` names the builtin, which *name* already says.
                return False
            if isinstance(node, ast.Attribute):
                return name is not None or self._expr(node.value)
            return True
        return all(self._expr(child) for child in ast.iter_child_nodes(node) if isinstance(child, ast.expr)) and all(
            self._comprehension(child) for child in ast.iter_child_nodes(node) if isinstance(child, ast.comprehension)
        )

    def _comprehension(self, node: ast.comprehension) -> bool:
        return self._expr(node.iter) and all(map(self._expr, node.ifs))

    def _comprehension_body(self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp) -> bool:
        """What of comprehension *node* runs in its own scope: all but its
        first iterable."""
        first, *later = node.generators
        values = [node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]
        return all(map(self._expr, [*values, *first.ifs])) and all(map(self._comprehension, later))

    def _qualified(self, node: ast.expr) -> str | None:
        """``np.linalg.norm`` -> ``numpy.linalg.norm`` through the top level's
        imports; ``own:<name>`` for a name it defines; a builtin as
        ``builtins.<name>``; None for anything else."""
        chain: list[str] = []
        while isinstance(node, ast.Attribute):
            chain.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return None
        head = node.id
        if head in self.aliases:
            root = self.aliases[head]
        elif head in self.defined:
            root = f"own:{head}"
        elif head in BUILTIN_NAMES:
            root = f"builtins.{head}"
        else:
            return None
        return ".".join([root, *reversed(chain)])

    def _vouches(self, name: str) -> bool:
        """Does the user function *name* (``own:f``, ``pkg.util.timed``),
        called at import (a decorator, ``TABLE = build()``), return what
        follows from its arguments and the source alone? So when what its
        body runs as it is called -- not the functions it defines, which run
        later -- calls only what `_expr` accepts, `_MAX_DEPTH` calls deep."""
        known = self.vouched.get(name)
        if known is not None:
            return known
        self.vouched[name] = False  # a function that calls itself is not followed round
        if name.startswith("own:"):
            reader, fn = self, self.functions.get(name[4:])
        else:
            reader, fn = self._users_function(name)
        known = fn is not None and self.depth < _MAX_DEPTH and reader._runs_source_only(fn)
        self.vouched[name] = known
        return known

    def _users_function(self, name: str) -> tuple[_SourceOnly, ast.FunctionDef | ast.AsyncFunctionDef | None]:
        module, _, attr = name.rpartition(".")
        origin = _origin(module) if module else None
        if origin is None or not origin.endswith(".py") or not is_local_module(types.SimpleNamespace(__file__=origin)):
            return self, None
        try:
            with open(origin, "rb") as fh:
                tree = ast.parse(fh.read())
        except (OSError, SyntaxError, ValueError):
            return self, None
        is_package = os.path.basename(origin) == "__init__.py"
        reader = _SourceOnly(module if is_package else module.rpartition(".")[0], self.depth + 1)
        reader.names(tree)
        return reader, reader.functions.get(attr)

    def _runs_source_only(self, fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        args = fn.args
        defaults = [*args.defaults, *(d for d in args.kw_defaults if d is not None)]
        if not all(self._decorator(d) for d in fn.decorator_list) or not all(map(self._expr, defaults)):
            return False
        self.in_body += 1
        try:
            # Every expression the body runs, not only its calls: an
            # attribute read (``sys.path[-1]``) is outside the source too.
            return self._in_scope(fn, lambda: all(map(self._expr, _expressions_as_run(fn.body))))
        finally:
            self.in_body -= 1

    def _in_scope(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef, read) -> bool:
        self.import_scopes.append(node)
        try:
            return self._scoped(_scope_names(node), isinstance(node, ast.ClassDef), read)
        finally:
            self.import_scopes.pop()

    def _qualified_here(self, node: ast.expr) -> str | None:
        """`_qualified`, but a builtin's name that only an import in another
        function or class binds is the builtin here: ``def _u(): from
        functools import reduce as getattr`` binds ``_u``'s local, not the
        top level's ``getattr``."""
        name = self._qualified(node)
        head = _head(node)
        if head in BUILTIN_NAMES and name != f"builtins.{head}" and self._only_imported_elsewhere(head):
            spelled = _spelled(node)
            return None if spelled is None else f"builtins.{spelled}"
        return name

    def _only_imported_elsewhere(self, name: str) -> bool:
        """Is *name* in `aliases` only from an import of a function or class
        body other than the ones being read, so that here it is not bound?"""
        if name not in self.aliases or name in self.module_aliases:
            return False
        scopes = [s for s in self.import_scopes if not isinstance(s, ast.ClassDef)]
        if self.import_scopes and isinstance(self.import_scopes[-1], ast.ClassDef):
            scopes.append(self.import_scopes[-1])
        return not any(name in _import_names(scope) for scope in scopes)

    def _scoped(self, names: dict[str, int], is_class: bool, read) -> bool:
        self.scopes.append(names)
        self.class_scopes.append(is_class)
        try:
            return read()
        finally:
            self.scopes.pop()
            self.class_scopes.pop()

    def _binds(self, name: str | None, node: ast.AST) -> bool:
        """Does the scope being read bind *name* by the time *node* runs, so
        that it is not the builtin: its own body, or the top level? Not a
        function elsewhere in the module: ``def _u(getattr)`` leaves a
        top-level ``getattr`` the builtin; nor a binding below the use
        (``getattr(...)`` then ``getattr = x``)."""
        line = getattr(node, "lineno", 0)
        scopes = [self.bound, *self.scopes[-1:]]
        return name is not None and any(name in scope and scope[name] < line for scope in scopes)


#: A scope with no imports of its own: a comprehension in a class body.
_NO_IMPORTS = ast.Module(body=[], type_ignores=[])
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
#: How many user functions deep a call at import is followed (`_SourceOnly._vouches`).
_MAX_DEPTH = 8
#: Values read from outside the source wherever a module names them.
_OUTSIDE_VALUES = frozenset({"os.environ", "os.environb", "sys.argv", "sys.stdin", "sys.stdin.buffer"})
#: Builtins that run code the source does not show, or read outside it,
#: wherever a module names them -- called or handed on as a value.
_RUNS_OR_READS_UNSEEN = frozenset(f"builtins.{n}" for n in ("eval", "exec", "__import__", "open", "breakpoint"))
#: Builtins a top level may hand on as a value: those it may call, the
#: exception classes, and constants.
_BUILTIN_VALUES = frozenset(
    {
        *(n for n in _SOURCE_ONLY_CALLS if n.startswith("builtins.")),
        *(f"builtins.{n}" for n, v in vars(builtins).items() if isinstance(v, type) and issubclass(v, BaseException)),
        "builtins.Ellipsis",
        "builtins.NotImplemented",
        "builtins.__name__",
        "builtins.__debug__",
    }
)


#: Builtin types whose methods return what follows from their arguments:
#: ``map(str.upper, names)``.
_PLAIN_TYPES = frozenset({"str", "bytes", "int", "float", "complex", "bool", "tuple", "frozenset", "range"})


def _plain_method(name: str) -> bool:
    """``builtins.str.upper``: a public method of a type in `_PLAIN_TYPES`."""
    parts = name.split(".")
    return len(parts) == 3 and parts[1] in _PLAIN_TYPES and not parts[2].startswith("_")


def _public_names_only(name: str, call: ast.Call) -> bool:
    """Does the call *name* of `_NAME_TAKING` look up only public names
    written out in it: ``attrgetter('a', 'b.c')``, ``methodcaller('m',
    ...)``? ``operator.call`` calls what it is given, and never is."""
    if name == "operator.call" or call.keywords and name == "operator.attrgetter":
        return False
    looked_up = call.args if name == "operator.attrgetter" else call.args[:1]
    return bool(looked_up) and all(
        isinstance(arg, ast.Constant)
        and isinstance(arg.value, str)
        and all(part and not part.startswith("_") for part in arg.value.split("."))
        for arg in looked_up
    )


def _scope_names(node: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, int]:
    """The names *node* binds in its own scope, so that a use of one there
    never reaches the builtin of that name, each with the line a use must
    come after (0 for any).

    In a function, any binding does: a parameter, an assignment's, a loop's
    or an ``except ... as`` target, a ``def`` or a ``class`` make the name
    local, and a use before it raises. A module's or a class's names are
    looked up as the code runs and fall back to the builtins, so there only
    a statement of its own body that always binds the name counts: an
    assignment, an import, a ``def`` or a ``class``; not one under an
    ``if``, a ``try`` or a loop (``if False: getattr = 1``), nor an ``except
    ... as`` name, which the handler's end unbinds; and only for a use on a
    line below that statement. Not the names a function, class, lambda or
    comprehension inside binds."""
    if isinstance(node, ast.Module):
        return _module_bound(node)
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return _always_bound(node.body)
    args = node.args
    names = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg) if a}
    pending: list[ast.AST] = list(node.body)
    while pending:
        child = pending.pop()
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(child.name)
            continue
        if isinstance(child, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            continue
        if isinstance(child, ast.Name) and not isinstance(child.ctx, ast.Load):
            names.add(child.id)
        elif isinstance(child, ast.ExceptHandler) and child.name:
            names.add(child.name)
        pending.extend(ast.iter_child_nodes(child))
    return dict.fromkeys(names, 0)


def _module_bound(tree: ast.Module) -> dict[str, int]:
    """`_always_bound` for a module, less what may be bound again by the
    time a use runs: a name a function of it declares ``global`` (``del
    getattr`` there, called at import, brings the builtin back), and, after
    ``from m import *`` anywhere in it, every builtin's name (``from
    builtins import *`` binds them all)."""
    bound = _always_bound(tree.body)
    rebound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            rebound.update(node.names)
        elif isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
            rebound |= BUILTIN_NAMES
    return {name: line for name, line in bound.items() if name not in rebound}


def _always_bound(body: list[ast.stmt]) -> dict[str, int]:
    """The names the statements of *body* itself always bind, and that no
    statement of it deletes, each with the last line of the first statement
    that binds it."""
    first: dict[str, int] = {}
    deleted: set[str] = set()
    for stmt in body:
        names: set[str] = set()
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(stmt.name)
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            names |= {(alias.asname or alias.name).partition(".")[0] for alias in stmt.names if alias.name != "*"}
        elif isinstance(stmt, ast.Assign):
            names |= {
                n.id
                for t in stmt.targets
                for n in ast.walk(t)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
            }
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None and isinstance(stmt.target, ast.Name):
            names.add(stmt.target.id)
        elif isinstance(stmt, ast.Delete):
            deleted |= {
                n.id
                for t in stmt.targets
                for n in ast.walk(t)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Del)
            }
        for name in names:
            first.setdefault(name, stmt.end_lineno or stmt.lineno)
    return {name: line for name, line in first.items() if name not in deleted}


def _import_names(scope: ast.AST) -> set[str]:
    """The names the imports of *scope*'s own body bind: not those of a
    function, class or lambda inside it."""
    names: set[str] = set()
    pending = list(ast.iter_child_nodes(scope))
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).partition(".")[0] for a in node.names if a.name != "*"}
        pending.extend(ast.iter_child_nodes(node))
    return names


def _head(node: ast.expr) -> str | None:
    """The name ``a.b.c`` starts with, or None."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _spelled(node: ast.expr) -> str | None:
    """``a.b.c`` as written, or None."""
    chain: list[str] = []
    while isinstance(node, ast.Attribute):
        chain.append(node.attr)
        node = node.value
    return ".".join([node.id, *reversed(chain)]) if isinstance(node, ast.Name) else None


def _expressions_as_run(body: list[ast.stmt]) -> list[ast.expr]:
    """The outermost expressions *body* runs: not those inside a function or
    lambda it defines, which run when that is called."""
    found: list[ast.expr] = []
    todo: list[ast.AST] = list(body)
    while todo:
        node = todo.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            args = node.args
            todo.extend([*args.defaults, *(d for d in args.kw_defaults if d is not None)])
            todo.extend(getattr(node, "decorator_list", ()))
            continue
        if isinstance(node, ast.expr):
            found.append(node)
            continue
        todo.extend(ast.iter_child_nodes(node))
    return found


def _calls_an_importer(node: ast.Call) -> bool:
    """``importlib.import_module(...)``, ``__import__(...)``, ``exec``/``eval``:
    the modules such a source runs are not named in it."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in ("__import__", "exec", "eval")
    return isinstance(func, ast.Attribute) and func.attr == "import_module"


def _binds_names(target: ast.expr) -> bool:
    if isinstance(target, ast.Name):
        return True
    if isinstance(target, (ast.Tuple, ast.List)):
        return all(_binds_names(e) for e in target.elts)
    if isinstance(target, ast.Starred):
        return _binds_names(target.value)
    return False


def _is_main_guard(test: ast.expr) -> bool:
    """``if __name__ == "__main__":``, which an import does not run."""
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )
