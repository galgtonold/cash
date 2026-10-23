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

from ..install_paths import in_own_package, top_package
from ..tracking.function_tracker import is_local_module

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
    }
)
#: Modules whose every function is a function of its arguments.
_SOURCE_ONLY_MODULES = frozenset({"math", "operator", "string"})
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
    origin = _origin(module_name)
    return bool(origin) and is_local_module(types.SimpleNamespace(__file__=origin))


def unloaded_import(module_name: str, prefix: tuple[str, ...], root_module: str | None) -> str | None:
    """The user module a function-body import would load that is not loaded
    yet, or None: *module_name* itself, or for ``from pkg import sub`` the
    submodule ``pkg.sub`` when ``pkg`` is loaded without it."""
    module = sys.modules.get(module_name)
    if module is None:
        return module_name if is_user_code(module_name, root_module) else None
    if not prefix or hasattr(module, prefix[0]):
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
    if kept is not None and all(_origin(name) == origin and _stat(origin) == st for name, origin, st in kept[1]):
        return kept[0]
    digest, looked = _source_digest(module_name, own_pkg)
    if len(_DIGESTS) > 1024:
        _DIGESTS.clear()
    _DIGESTS[slot] = (digest, looked)
    return digest


def _source_digest(module_name: str, own_pkg: str | None) -> tuple[str | None, tuple]:
    files: dict[str, str] = {}
    looked: list[tuple[str, str | None, tuple | None]] = []
    todo = [module_name]
    seen: set[str] = set()
    while todo:
        name = todo.pop()
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            prefix = ".".join(parts[:i])
            if prefix in seen:
                continue
            seen.add(prefix)
            origin = _origin(prefix)
            looked.append((prefix, origin, _stat(origin)))
            if origin is None:
                if prefix == module_name:
                    return None, tuple(looked)
                # Not on disk to import: a name `from pkg import name` reads off
                # the package, or a module the body's import would fail on.
                break
            if not (in_own_package(prefix, own_pkg) or is_local_module(types.SimpleNamespace(__file__=origin))):
                # A library: followed no further, as the helper walk does not.
                break
            if not origin.endswith(".py") or len(files) >= _MAX_FILES:
                return None, tuple(looked)
            try:
                with open(origin, "rb") as fh:
                    source = fh.read()
                tree = ast.parse(source)
            except (OSError, SyntaxError, ValueError):
                return None, tuple(looked)
            is_package = os.path.basename(origin) == "__init__.py"
            reader = _SourceOnly(prefix if is_package else prefix.rpartition(".")[0])
            if not reader.module(tree):
                return None, tuple(looked)
            files[prefix] = hashlib.blake2b(source, digest_size=16).hexdigest()
            todo.extend(reader.imported)
    blob = "\n".join(f"{name}={digest}" for name, digest in sorted(files.items()))
    return hashlib.blake2b(blob.encode(), digest_size=16).hexdigest(), tuple(looked)


def _stat(path: str | None) -> tuple | None:
    if path is None:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def _origin(name: str) -> str | None:
    """The file module *name* is (or would be) loaded from, found without
    running any module: a loaded parent's ``__path__``, else the finders'
    spec for each part in turn. ``importlib.util.find_spec`` would import
    the parents of a dotted name, running their ``__init__``."""
    path = None
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        full = ".".join(parts[:i])
        mod = sys.modules.get(full)
        if mod is not None:
            if i == len(parts):
                origin = getattr(mod, "__file__", None)
                return origin if isinstance(origin, str) else None
            path = getattr(mod, "__path__", None)
        else:
            spec = _find_spec(full, path)
            if spec is None:
                return None
            if i == len(parts):
                return spec.origin if spec.has_location and isinstance(spec.origin, str) else None
            path = spec.submodule_search_locations
        if path is None:
            return None
    return None


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

    def __init__(self, package: str) -> None:
        self.package = package
        self.imported: list[str] = []
        #: local name -> qualified name, from the top level's imports.
        self.aliases: dict[str, str] = {}
        #: names the top level defines with ``def`` or ``class``.
        self.defined: set[str] = set()

    def module(self, tree: ast.Module) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self._note_import(node)
            elif isinstance(node, ast.Call) and _calls_an_importer(node):
                return False
        self.defined = {
            n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        return self._body(tree.body)

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
                and self._body(node.body)
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
        # The user's own decorator: its code is in the source read.
        return name in _SOURCE_ONLY_CALLS or name.startswith("own:") or self._is_users(name)

    def _expr(self, node: ast.expr) -> bool:
        if isinstance(node, ast.Lambda):
            # Runs only when called; its code is in the source.
            return all(map(self._expr, [*node.args.defaults, *(d for d in node.args.kw_defaults if d is not None)]))
        if isinstance(node, (ast.NamedExpr, ast.Await, ast.Yield, ast.YieldFrom)):
            return False
        if isinstance(node, ast.Call):
            name = self._qualified(node.func)
            if name is None or not (
                name in _SOURCE_ONLY_CALLS or name.partition(".")[0] in _SOURCE_ONLY_MODULES and "." in name
            ):
                return False
            return all(map(self._expr, [*node.args, *(k.value for k in node.keywords)]))
        if isinstance(node, (ast.Attribute, ast.Name)):
            name = self._qualified(node)
            if name is not None and name.partition(".")[0] in _AMBIENT_MODULES and name not in _INTERPRETER_FACTS:
                return False
            if isinstance(node, ast.Attribute):
                return name is not None or self._expr(node.value)
            return True
        return all(self._expr(child) for child in ast.iter_child_nodes(node) if isinstance(child, ast.expr)) and all(
            self._comprehension(child) for child in ast.iter_child_nodes(node) if isinstance(child, ast.comprehension)
        )

    def _comprehension(self, node: ast.comprehension) -> bool:
        return self._expr(node.iter) and all(map(self._expr, node.ifs))

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
        elif head in _BUILTIN_NAMES:
            root = f"builtins.{head}"
        else:
            return None
        return ".".join([root, *reversed(chain)])

    def _is_users(self, name: str) -> bool:
        module = name.rpartition(".")[0]
        while module:
            origin = _origin(module)
            if origin is not None:
                return is_local_module(types.SimpleNamespace(__file__=origin))
            module = module.rpartition(".")[0]
        return False


_BUILTIN_NAMES = frozenset(dir(builtins))


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
