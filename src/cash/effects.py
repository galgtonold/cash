"""One vocabulary for what a call does to the world, shared by both paths.

The notebook's statement analysis and the decorator's purity analyzer ask the
same question of a call -- does running it do something a cache hit would not
do, or read something the key does not see? -- and answer it with different
consequences: a notebook statement with a file write runs every time, a
decorated function with one caches and warns. Those consequences belong to
each path. The question does not. When each path kept its own list of names,
the lists drifted: a name added for one path was never added to the other, and
the same call was a side effect in a notebook and silent in a function.

So this module says only *what* a call is, as an :class:`EffectKind`, and each
path keeps a policy table next to its own code saying what it does about each
kind (``NOTEBOOK_POLICY`` in :mod:`cash.analysis.file_effects`,
``DECORATOR_POLICY`` in :mod:`cash.analysis.purity_analyzer`). A test asserts that both
tables cover every kind, so adding a kind forces a decision in both paths.

**Mutation is a different question.** ``rows.append(x)`` changes an object,
not the world, and whether that matters depends on who owns the object. The
names for it are :data:`MUTATOR_METHODS`, kept apart from the effect verbs.

**Matching is by name.** A call is matched by how it is spelled
(``requests.post``), by a method name that means the same thing on every
receiver that has it (``df.to_csv``, ``session.post``), and -- when a namespace
is given -- by what its names are bound to (``import time as t; t.time()``).
Nothing here imports a library the program has not imported itself, and no
attribute is read from anything but a module or a class, so classifying a call
never runs the user's code.

Nothing here imports from ``cash.notebook``: both paths depend on it.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import os
import sys
import types
from collections.abc import Callable, Iterable, Mapping
from enum import Enum
from typing import Any, NamedTuple

__all__ = [
    "Action",
    "EffectKind",
    "MODULE_CALLS",
    "METHOD_VERBS",
    "MUTATOR_METHODS",
    "CLOCK_WHEN_ARG_CALLS",
    "ENVIRON_NAMES",
    "classify_call",
    "dotted_name",
    "environment_component",
    "environment_input",
    "is_open_write_mode",
    "writes_to_console",
]


class EffectKind(str, Enum):
    """What a call does that a cache hit would not repeat, or reads that a
    key would not see."""

    #: Writes, moves or removes a file or folder.
    FILE_WRITE = "file_write"
    #: Reads a file. Owned by the file tracker, which folds the file into the
    #: key; listed so that no call is classified twice.
    FILE_READ = "file_read"
    #: Fetches from a server without changing it: ``requests.get``.
    NETWORK_READ = "network_read"
    #: Changes something on a server: ``requests.post``, ``session.post``,
    #: ``client.publish``, ``s3.upload_file``.
    NETWORK_WRITE = "network_write"
    #: A socket connection seen at run time. Its direction is unknown, so each
    #: policy treats it like a write.
    NETWORK = "network"
    #: A database query that only reads: ``cur.execute("SELECT ...")``.
    DB_READ = "db_read"
    #: Any other database statement: ``cur.execute(sql)``, ``conn.commit()``,
    #: ``df.to_sql(...)``.
    DB_WRITE = "db_write"
    #: Starts another program: ``subprocess.run``, ``os.system``.
    SUBPROCESS = "subprocess"
    #: Reads the clock or makes a fresh id: ``time.time``, ``uuid4``.
    CLOCK = "clock"
    #: Reads the process environment: ``os.getenv``, ``os.getcwd``.
    ENVIRONMENT = "environment"
    #: Writes to the console: ``print``, ``logging.info``, ``sys.stdout.write``.
    CONSOLE = "console"
    #: Draws on or shows pyplot's current figure: ``plt.plot``, ``plt.show``.
    DISPLAY = "display"
    #: Talks to the person at the keyboard: ``input``, ``getpass``.
    INTERACTIVE = "interactive"

    def __str__(self) -> str:
        return self.value


class Action(str, Enum):
    """What a path does about one kind of effect."""

    #: Nothing to do: cache as if the call were not there.
    CACHE = "cache"
    #: The value read is an input: it goes into the key.
    CACHE_AS_INPUT = "cache_as_input"
    #: Cache, and say that a hit will not repeat it.
    WARN = "warn"
    #: Do not cache: run every time.
    REFUSE = "refuse"
    #: Cache, and say how to bound the age of what was fetched (``ttl=``).
    SUGGEST_TTL = "suggest_ttl"

    def __str__(self) -> str:
        return self.value


class Effect(NamedTuple):
    """One classified call."""

    kind: EffectKind
    #: The callee as matched: the dotted spelling (``requests.post``,
    #: ``open``), the canonical name it resolved to through a namespace
    #: (``time.time`` for ``t.time``), or, when *method* is set, the method
    #: name alone (``to_csv``).
    name: str
    #: Matched by a method name on any receiver rather than by a full name.
    method: bool = False


_W = EffectKind.FILE_WRITE
_NR = EffectKind.NETWORK_READ
_NW = EffectKind.NETWORK_WRITE
_DR = EffectKind.DB_READ
_DW = EffectKind.DB_WRITE
_SP = EffectKind.SUBPROCESS
_CK = EffectKind.CLOCK
_EV = EffectKind.ENVIRONMENT
_CO = EffectKind.CONSOLE
_IN = EffectKind.INTERACTIVE

#: Calls matched by their full dotted name (or a bare builtin name).
MODULE_CALLS: dict[str, EffectKind] = {
    # -- files and folders --
    "os.remove": _W,
    "os.unlink": _W,
    "os.rmdir": _W,
    "os.removedirs": _W,
    "os.mkdir": _W,
    "os.makedirs": _W,
    "os.rename": _W,
    "os.renames": _W,
    "os.replace": _W,
    "os.symlink": _W,
    "os.link": _W,
    "os.chmod": _W,
    "os.truncate": _W,
    "shutil.copy": _W,
    "shutil.copy2": _W,
    "shutil.copyfile": _W,
    "shutil.copytree": _W,
    "shutil.move": _W,
    "shutil.rmtree": _W,
    "json.dump": _W,
    "pickle.dump": _W,
    "csv.writer": _W,
    # -- other programs --
    "os.system": _SP,
    "os.popen": _SP,
    "os.execv": _SP,
    "os.execve": _SP,
    "os.execl": _SP,
    "os.execlp": _SP,
    "os.execvp": _SP,
    "os.spawnv": _SP,
    "os.spawnl": _SP,
    "os.posix_spawn": _SP,
    "subprocess.run": _SP,
    "subprocess.call": _SP,
    "subprocess.Popen": _SP,
    "subprocess.check_call": _SP,
    "subprocess.check_output": _SP,
    "subprocess.getoutput": _SP,
    "subprocess.getstatusoutput": _SP,
    # -- the network --
    "requests.get": _NR,
    "requests.head": _NR,
    "requests.options": _NR,
    "requests.post": _NW,
    "requests.put": _NW,
    "requests.patch": _NW,
    "requests.delete": _NW,
    # `requests.request(method, url)`: the kind follows the method argument.
    "requests.request": _NW,
    "httpx.get": _NR,
    "httpx.head": _NR,
    "httpx.options": _NR,
    "httpx.post": _NW,
    "httpx.put": _NW,
    "httpx.patch": _NW,
    "httpx.delete": _NW,
    "httpx.request": _NW,
    # `urlopen(url, data)` posts; without data it fetches.
    "urllib.request.urlopen": _NR,
    "urlopen": _NR,
    # -- the clock and fresh ids --
    "datetime.now": _CK,
    "datetime.utcnow": _CK,
    "datetime.today": _CK,
    "datetime.datetime.now": _CK,
    "datetime.datetime.utcnow": _CK,
    "datetime.datetime.today": _CK,
    "datetime.date.today": _CK,
    "date.today": _CK,
    "time.time": _CK,
    "time.time_ns": _CK,
    "time.monotonic": _CK,
    "time.perf_counter": _CK,
    "time.process_time": _CK,
    "uuid.uuid1": _CK,
    "uuid.uuid4": _CK,
    "pandas.Timestamp.now": _CK,
    "pandas.Timestamp.today": _CK,
    "pandas.Timestamp.utcnow": _CK,
    # -- the environment --
    "os.getcwd": _EV,
    "os.getcwdb": _EV,
    "os.getenv": _EV,
    "os.getenvb": _EV,
    "os.environ.get": _EV,
    "os.environb.get": _EV,
    # The working directory, read to make a path absolute. Not when the path
    # is already absolute (`_surely_absolute`).
    "pathlib.Path.cwd": _EV,
    "Path.cwd": _EV,
    "os.path.abspath": _EV,
    "os.path.realpath": _EV,
    # -- the console --
    "print": _CO,
    "logging.debug": _CO,
    "logging.info": _CO,
    "logging.warning": _CO,
    "logging.error": _CO,
    "logging.critical": _CO,
    "logging.exception": _CO,
    "logging.log": _CO,
    # -- the person at the keyboard --
    "input": _IN,
    "breakpoint": _IN,
    "getpass.getpass": _IN,
    "exit": _IN,
    "quit": _IN,
}

#: Method names that mean the same effect on essentially every receiver that
#: has them, matched whatever the receiver is.
#:
#: A receiver's type is not knowable from source, and a name is the only thing
#: that reaches an effect inside an installed library: ``session.post(...)``
#: is invisible except through ``post``. The bar for adding a name is that it
#: is effect-shaped on every type that defines it. ``get`` fails it
#: (``dict.get``), which is why a read through a client object is not matched
#: at all; ``rename``, ``replace`` and ``touch``-like names that collide with
#: ``str`` methods fail it too.
METHOD_VERBS: dict[str, EffectKind] = {
    # -- files: pandas, numpy, matplotlib, PIL, torch, file objects, pathlib --
    "to_csv": _W,
    "to_excel": _W,
    "to_parquet": _W,
    "to_json": _W,
    "to_pickle": _W,
    "to_hdf": _W,
    "to_feather": _W,
    "to_stata": _W,
    "to_latex": _W,
    "to_html": _W,
    "to_clipboard": _W,
    "to_markdown": _W,
    "savefig": _W,
    "save": _W,
    "write": _W,
    "writelines": _W,
    "write_text": _W,
    "write_bytes": _W,
    # `OUT.mkdir(exist_ok=True)`: restored instead of run, it left an output
    # folder the user had emptied missing, and the first savefig into it
    # raised (with every result persisted). A write on every type
    # that has it (Path, ZipFile, SFTP clients).
    "mkdir": _W,
    "touch": _W,
    "unlink": _W,
    "symlink_to": _W,
    "hardlink_to": _W,
    # -- the network, through a client object (a requests Session, httpx, an
    #    SDK wrapper, a socket) --
    "post": _NW,
    "put": _NW,
    "patch": _NW,
    "send": _NW,
    "sendall": _NW,
    "sendto": _NW,
    "publish": _NW,
    "upload": _NW,
    "upload_file": _NW,
    "upload_fileobj": _NW,
    "put_object": _NW,
    # -- a database, through a cursor, connection or frame --
    "execute": _DW,
    "executemany": _DW,
    "executescript": _DW,
    "commit": _DW,
    "rollback": _DW,
    "to_sql": _DW,
    "to_gbq": _DW,
    "read_sql": _DR,
    "read_sql_query": _DR,
    "read_sql_table": _DR,
    "read_gbq": _DR,
}

#: Method names that change their receiver in place: a container's mutators.
#:
#: Not effects. Whether one matters depends on who owns the receiver -- a list
#: the function built itself is its own business, the caller's list is not --
#: so each path asks this question separately from the effect question.
MUTATOR_METHODS: frozenset[str] = frozenset(
    {
        "append",
        "extend",
        "insert",
        "pop",
        "remove",
        "sort",
        "reverse",
        "clear",
        "update",
        "add",
        "discard",
    }
)

#: Constructors that read the clock only when a string argument says so:
#: ``pd.to_datetime("today")``, ``pd.Timestamp("now")``, ``np.datetime64("now")``.
CLOCK_WHEN_ARG_CALLS: frozenset[str] = frozenset(
    {
        "pandas.to_datetime",
        "pandas.Timestamp",
        "numpy.datetime64",
    }
)
CLOCK_ARG_VALUES: frozenset[str] = frozenset({"now", "today"})

#: Functions that read the clock when their time argument is LEFT OUT: called
#: with at most this many positional arguments. ``time.strftime("%Y-%m")``
#: froze a report's period with no warning, while
#: ``time.strftime("%Y-%m", t)`` only formats ``t`` -- as ``time.localtime(ts)``
#: only converts.
CLOCK_WHEN_ARGS_OMITTED: dict[str, int] = {
    "time.strftime": 1,
    "time.asctime": 0,
    "time.ctime": 0,
    "time.localtime": 0,
    "time.gmtime": 0,
}

#: How the process environment is spelled at a subscript: ``os.environ[...]``,
#: or bare ``environ[...]`` after ``from os import environ``; and its bytes
#: twin.
ENVIRON_NAMES: frozenset[str] = frozenset({"os.environ", "environ", "os.environb", "environb"})

#: Methods of the environment that change it (a side effect, not a read of
#: it) or read one named variable (folded like a subscript).
ENVIRON_KEYED_METHODS: frozenset[str] = frozenset(
    {"get", "setdefault", "pop", "popitem", "update", "clear", "putenv", "unsetenv", "__setitem__", "__delitem__"}
)

#: Calls whose answer is the working directory, whatever their arguments.
_CWD_CALLS: frozenset[str] = frozenset({"os.getcwd", "os.getcwdb", "pathlib.Path.cwd", "Path.cwd"})
#: Calls that read the working directory to resolve a RELATIVE path.
_CWD_RESOLVERS: frozenset[str] = frozenset({"os.path.abspath", "os.path.realpath"})
#: Path methods that do the same (``Path(p).resolve()``).
_CWD_RESOLVER_METHODS: frozenset[str] = frozenset({"resolve", "absolute"})

# matplotlib.pyplot module aliases. EVERY module-level ``plt.*`` call operates on
# pyplot's PROCESS-GLOBAL current figure -- drawing (``plt.plot``, ``plt.hist``),
# styling (``plt.title``, ``plt.legend``) or displaying (``plt.show``) -- state
# cash does not track. ``plt.savefig`` is a file write (a method verb), not a
# display.
PYPLOT_MODULE_ALIASES: frozenset[str] = frozenset({"plt", "pyplot", "matplotlib.pyplot"})

# pyplot calls that CREATE or FETCH a Figure/Axes rather than draw on / style the
# current one. They return identity-coupled objects, which the notebook refuses
# (and explains) on their own, so they are not a display effect.
PYPLOT_FIGURE_ACCESSORS: frozenset[str] = frozenset(
    {
        "figure",
        "subplots",
        "subplot",
        "subplot_mosaic",
        "subplot2grid",
        "axes",
        "gca",
        "gcf",
        "get_current_fig_manager",
    }
)

#: The kinds a namespace is consulted for: a call to one is recognised through
#: what its names are bound to (``from time import time as now; now()``), not
#: only by its spelling.
_RESOLVED_KINDS: frozenset[EffectKind] = frozenset({EffectKind.CLOCK, EffectKind.ENVIRONMENT, EffectKind.INTERACTIVE})


def dotted_name(func: ast.AST) -> str | None:
    """``a.b.c`` for a call's callee written as a name chain, else None.

    A chain rooted at anything but a name (a call, a subscript) has no dotted
    name; its last attribute is still a method name, matched separately.
    """
    parts: list[str] = []
    node = func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def is_open_write_mode(call: ast.Call) -> bool:
    """True when an ``open()`` call's mode is statically a write mode.

    A computed mode (``open(p, mode)``) proves nothing and reports False.
    """
    mode: ast.expr | None = call.args[1] if len(call.args) >= 2 else None
    if mode is None:
        mode = next((kw.value for kw in call.keywords if kw.arg == "mode"), None)
    return isinstance(mode, ast.Constant) and isinstance(mode.value, str) and any(c in mode.value for c in "wax")


def writes_to_console(call: ast.Call) -> bool:
    """``os.write(1 | 2, ...)``, ``sys.stdout.write(...)``, ``sys.stderr.write(...)``.

    Output, like ``print``, not a file. Counted as a write whose repeatability
    was unknown, a step marker (``os.write(2, f"RUN {step}")``) inside a helper
    made every statement calling it a file writer.
    """
    func = call.func
    if not (isinstance(func, ast.Attribute) and func.attr in ("write", "writelines")):
        return False
    base = func.value
    if isinstance(base, ast.Name) and base.id == "os" and func.attr == "write":
        fd = call.args[0] if call.args else None
        return isinstance(fd, ast.Constant) and fd.value in (1, 2)
    return (
        isinstance(base, ast.Attribute)
        and base.attr in ("stdout", "stderr", "__stdout__", "__stderr__")
        and isinstance(base.value, ast.Name)
        and base.value.id == "sys"
    )


_SQL_WRITES = (
    "INSERT",
    "UPDATE",
    "DELETE",
    "REPLACE",
    "MERGE",
    "UPSERT",
    "CREATE",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "ATTACH",
    "DETACH",
    "VACUUM",
    "PRAGMA",
)


def is_read_only_sql(call: ast.Call) -> bool:
    """``con.execute("SELECT ...")``: a query that reads, written as a literal.

    ``execute`` is how most database writes happen, but a literal SELECT (or a
    WITH that only selects) changes nothing, yet a sqlite lookup was
    reported as a "write method". A query built at run time, or any statement
    naming a write verb, is still a write.
    """
    if not (isinstance(call.func, ast.Attribute) and call.func.attr == "execute" and call.args):
        return False
    sql = call.args[0]
    if not (isinstance(sql, ast.Constant) and isinstance(sql.value, str)):
        return False
    text = " ".join(line.split("--", 1)[0] for line in sql.value.splitlines()).upper()
    words = set(text.replace("(", " ").replace(")", " ").replace(";", " ").split())
    head = text.lstrip()
    return head.startswith(("SELECT", "WITH")) and not words.intersection(_SQL_WRITES)


def is_environ_read(node: ast.AST) -> bool:
    """``os.environ["KEY"]`` read in Load context: the environment read that is
    not a call."""
    return (
        isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load) and dotted_name(node.value) in ENVIRON_NAMES
    )


#: One environment read the key can fold: ``("env", NAME)`` for a variable,
#: ``("cwd", "")`` for the working directory.
EnvironmentInput = tuple[str, str]


def _variable_name(node: ast.AST | None) -> str | None:
    """An environment variable's name written out: a str, or bytes for
    ``os.environb``."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return node.value
        if isinstance(node.value, bytes):
            return os.fsdecode(node.value)
    return None


def environ_membership(node: ast.AST) -> ast.expr | None:
    """``"NAME" in os.environ`` (or ``not in``): the environment operand."""
    if (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], (ast.In, ast.NotIn))
        and dotted_name(node.comparators[0]) in ENVIRON_NAMES
    ):
        return node.comparators[0]
    return None


#: What a path built from these is anchored to: already absolute.
_ABSOLUTE_PATH_MAKERS = frozenset(
    {"Path", "PurePath", "pathlib.Path", "pathlib.PurePath", "os.path.dirname", "os.path.join", "str", "os.fspath"}
)
_ABSOLUTE_PATH_METHODS = frozenset({"joinpath", "with_name", "with_suffix", "with_stem"})


def _surely_absolute(node: ast.AST | None) -> bool:
    """Is the path *node* builds absolute however the code runs?

    A literal absolute path, ``__file__``, and what is built from one
    (``Path(__file__).parent / "data"``, ``os.path.dirname(__file__)``):
    resolving those reads no working directory. Anything else might be
    relative.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
        text = os.fsdecode(node.value) if isinstance(node.value, bytes) else node.value
        return text.startswith(("/", "\\")) or (len(text) > 2 and text[1] == ":" and text[2] in "/\\")
    if isinstance(node, ast.Name):
        return node.id == "__file__"
    if isinstance(node, ast.Attribute):
        return node.attr in ("parent", "parents") and _surely_absolute(node.value)
    if isinstance(node, ast.Subscript):
        return _surely_absolute(node.value)
    if isinstance(node, ast.BinOp):
        return isinstance(node.op, ast.Div) and _surely_absolute(node.left)
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in _CWD_RESOLVER_METHODS:
            return True  # what it returns is absolute; the call is judged on its own
        if isinstance(func, ast.Attribute) and func.attr in _ABSOLUTE_PATH_METHODS:
            return _surely_absolute(func.value)
        name = dotted_name(func)
        if name in _CWD_CALLS or name in _CWD_RESOLVERS:
            return True
        return name in _ABSOLUTE_PATH_MAKERS and bool(node.args) and _surely_absolute(node.args[0])
    return False


def _resolves_relative_path(call: ast.Call) -> bool:
    """``Path(p).resolve()`` / ``.absolute()`` on a path that may be relative."""
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr in _CWD_RESOLVER_METHODS
        and not call.args
        and all(kw.arg == "strict" for kw in call.keywords)
        and not _surely_absolute(func.value)
    )


def environment_input(node: ast.AST, namespace: Mapping[str, Any] | None = None) -> EnvironmentInput | None:
    """What an environment read reads, when its value can go into a key.

    ``os.getenv("NAME")``, ``os.environ.get("NAME")``, ``os.environ["NAME"]``
    and ``"NAME" in os.environ`` with the name written out give
    ``("env", "NAME")`` (``os.environb`` too); ``os.getcwd()``, ``Path.cwd()``,
    and making a path that may be relative absolute (``os.path.abspath(p)``,
    ``Path(p).resolve()``) give ``("cwd", "")``. Anything else is None --
    including a read whose name is only known at run time, which no key can
    fold.
    """
    if is_environ_read(node):
        name = _variable_name(node.slice)  # type: ignore[attr-defined]
        return ("env", name) if name is not None else None
    if environ_membership(node) is not None:
        name = _variable_name(node.left)  # type: ignore[attr-defined]
        return ("env", name) if name is not None else None
    if not isinstance(node, ast.Call):
        return None
    if _resolves_relative_path(node):
        return ("cwd", "")
    effect = classify_call(node, namespace)
    if effect is None or effect.kind is not EffectKind.ENVIRONMENT:
        return None
    if effect.name in _CWD_CALLS or effect.name in _CWD_RESOLVERS:
        return ("cwd", "")
    name = _variable_name(_literal_arg(node, 0, "key"))
    return ("env", name) if name is not None else None


def environment_label(entry: EnvironmentInput) -> str:
    """How a folded environment read is named when it is why a key changed."""
    kind, name = entry
    return "the working directory" if kind == "cwd" else f"environment variable {name}"


def _environment_digest(entry: EnvironmentInput) -> str:
    kind, name = entry
    if kind == "cwd":
        try:
            value: str | None = os.getcwd()
        except OSError:  # the directory was removed under the process
            value = None
    else:
        value = os.environ.get(name)
    # A digest, never the value: an environment variable is where secrets live,
    # and a key component can end up in a log line or an explain() report.
    return "unset" if value is None else hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def environment_component(entries: Iterable[EnvironmentInput], note: Callable[[str, str], None] | None = None) -> str:
    """The key component for the environment reads *entries*: their current
    values, digested. Empty when there are none, so a key that reads no
    environment is exactly what it was. *note* is told each read's label and
    digest, for the account of why a key changed."""
    parts = []
    for entry in sorted(set(entries)):
        digest = _environment_digest(entry)
        if note is not None:
            note(environment_label(entry), digest)
        parts.append(f"{entry[0]}:{entry[1]}={digest}")
    return f":env:{':'.join(parts)}" if parts else ""


def _reads_clock_when_omitted(name: str, call: ast.Call) -> bool:
    """``time.strftime(fmt)`` reads the clock; ``time.strftime(fmt, t)`` does not."""
    most = CLOCK_WHEN_ARGS_OMITTED.get(name)
    return (
        most is not None
        and len(call.args) <= most
        and not call.keywords
        and not any(isinstance(a, ast.Starred) for a in call.args)
    )


def _literal_arg(call: ast.Call, position: int, keyword: str) -> ast.expr | None:
    if len(call.args) > position:
        return call.args[position]
    return next((kw.value for kw in call.keywords if kw.arg == keyword), None)


_READ_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def _named_kind(name: str, call: ast.Call) -> EffectKind | None:
    """The kind of a call to the fully named *name*, with its argument rules."""
    kind = MODULE_CALLS.get(name)
    if kind is None:
        if _reads_clock_when_omitted(name, call):
            return EffectKind.CLOCK
        arg = call.args[0] if call.args else None
        if (
            name in CLOCK_WHEN_ARG_CALLS
            and isinstance(arg, ast.Constant)
            and isinstance(arg.value, str)
            and arg.value.strip().lower() in CLOCK_ARG_VALUES
        ):
            return EffectKind.CLOCK
        return None
    if name.endswith(".request") and name.split(".")[0] in ("requests", "httpx"):
        method = _literal_arg(call, 0, "method")
        if isinstance(method, ast.Constant) and isinstance(method.value, str):
            return _NR if method.value.upper() in _READ_METHODS else _NW
        return EffectKind.NETWORK  # a computed method: the direction is unknown
    if name.endswith("urlopen") and _literal_arg(call, 1, "data") is not None:
        return _NW
    if name in _CWD_RESOLVERS and _surely_absolute(_literal_arg(call, 0, "path")):
        return None  # `os.path.abspath(__file__)` reads no working directory
    return kind


# -- resolving names through a namespace ------------------------------------

_ROOTS: dict[str, Any] = {"loaded": None, "roots": {}}


def _resolution_table() -> set[str]:
    return (
        {name for name, kind in MODULE_CALLS.items() if kind in _RESOLVED_KINDS}
        | set(CLOCK_WHEN_ARG_CALLS)
        | set(CLOCK_WHEN_ARGS_OMITTED)
    )


def _roots() -> dict[int, tuple[Any, str]]:
    """``id(obj) -> (obj, canonical name)`` for every module, class and plain
    function a resolvable name passes through (``datetime``,
    ``datetime.datetime``, ``time.time``, ``pandas.Timestamp``), among the
    modules loaded now. Rebuilt when that set changes; never imports one."""
    table = _resolution_table()
    loaded = tuple(sorted({entry.split(".", 1)[0] for entry in table} & set(sys.modules)))
    if _ROOTS["loaded"] == loaded:
        return _ROOTS["roots"]
    roots: dict[int, tuple[Any, str]] = {}
    for entry in table:
        parts = entry.split(".")
        if len(parts) == 1:  # a builtin: `ask = input; ask()`
            builtin = getattr(builtins, entry, None)
            if builtin is not None:
                roots.setdefault(id(builtin), (builtin, entry))
            continue
        obj: Any = sys.modules.get(parts[0])
        if obj is None:
            continue
        roots.setdefault(id(obj), (obj, parts[0]))
        for i in range(1, len(parts)):
            try:
                nxt = getattr(obj, parts[i])
                # Only objects with a stable identity: `datetime.datetime.now`
                # is a new bound method on every read, and a recycled id would
                # match something else.
                if nxt is not getattr(obj, parts[i]):
                    break
            except AttributeError:
                break
            obj = nxt
            roots.setdefault(id(obj), (obj, ".".join(parts[: i + 1])))
    _ROOTS["loaded"], _ROOTS["roots"] = loaded, roots
    return roots


def _canonical_names(chain: list[str], namespace: Mapping[str, Any]) -> list[str]:
    """What the name chain *chain* is, spelled from a known root, innermost
    candidate last. Attributes are read from modules and classes only."""
    roots = _roots()
    obj: Any = namespace[chain[0]]
    candidates: list[str] = []
    for i in range(len(chain)):
        if i:
            if not isinstance(obj, (types.ModuleType, type)):
                break
            try:
                obj = getattr(obj, chain[i])
            except Exception:  # noqa: BLE001 - a probe of user namespaces
                break
        hit = roots.get(id(obj))
        if hit is not None and hit[0] is obj:
            candidates.append(".".join((hit[1], *chain[i + 1 :])))
        elif i == len(chain) - 1 and isinstance(obj, (types.MethodType, types.BuiltinMethodType)):
            # `now = datetime.now` bound in an earlier cell: a method of a
            # known class, whose bound object is new on every read.
            owner = roots.get(id(getattr(obj, "__self__", None)))
            if owner is not None and owner[0] is obj.__self__:
                candidates.append(f"{owner[1]}.{obj.__name__}")
    return candidates


def _shadowed(name: str, namespace: Mapping[str, Any] | None) -> bool:
    """Is the builtin *name* rebound in *namespace* to something else?"""
    if namespace is None or name not in namespace:
        return False
    return namespace[name] is not getattr(builtins, name, None)


def classify_call(call: ast.Call, namespace: Mapping[str, Any] | None = None) -> Effect | None:
    """What *call* does, or None when it is none of the :class:`EffectKind`\\ s.

    *namespace*, when given, is what the call's names are bound to; with it,
    ``import time as t; t.time()`` is recognised as ``time.time``. Only the
    kinds in ``_RESOLVED_KINDS`` are looked up that way.
    """
    func = call.func
    if isinstance(func, ast.Name) and func.id == "open":
        if _shadowed("open", namespace):
            return None
        return Effect(EffectKind.FILE_WRITE if is_open_write_mode(call) else EffectKind.FILE_READ, "open")
    if isinstance(func, ast.Name) and func.id in MODULE_CALLS and _shadowed(func.id, namespace):
        return None
    if writes_to_console(call):
        return Effect(EffectKind.CONSOLE, func.attr, method=True)  # type: ignore[attr-defined]
    dotted = dotted_name(func)
    if dotted is not None:
        kind = _named_kind(dotted, call)
        if kind is not None:
            return Effect(kind, dotted)
        chain = dotted.split(".")
        if namespace is not None and chain[0] in namespace:
            for canonical in reversed(_canonical_names(chain, namespace)):
                kind = _named_kind(canonical, call)
                if kind in _RESOLVED_KINDS:
                    return Effect(kind, canonical)
    if not isinstance(func, ast.Attribute):
        return None
    method = func.attr
    base = dotted_name(func.value)
    if base in PYPLOT_MODULE_ALIASES and method not in METHOD_VERBS and method not in PYPLOT_FIGURE_ACCESSORS:
        return Effect(EffectKind.DISPLAY, f"{base}.{method}")
    kind = METHOD_VERBS.get(method)
    if kind is None:
        return None
    if kind is EffectKind.DB_WRITE and is_read_only_sql(call):
        kind = EffectKind.DB_READ
    return Effect(kind, method, method=True)
