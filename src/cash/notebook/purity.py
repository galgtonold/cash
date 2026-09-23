"""@pure and @stateful decorator system for caching decisions."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

__all__ = [
    "pure",
    "stateful",
    "is_pure",
    "is_stateful",
    "is_known_pure",
    "KNOWN_PURE_BUILTINS",
]

F = TypeVar("F", bound=Callable[..., Any])

# Attribute names used to mark functions
_PURE_ATTR = "_cash_pure"
_STATEFUL_ATTR = "_cash_stateful"


def pure(func: F) -> F:
    """Mark a function as pure (no side effects) for the purity analyzer.

    This is a promise to the analyzer, not a caching switch. Its load-bearing
    effect is on the ``@cash.cache`` decorator: a callee marked pure is trusted,
    so the ``CashImpurityWarning`` that would otherwise fire for it is
    suppressed (see :mod:`cash.purity_analyzer`). In the notebook *statement*
    path it changes one verdict: a call to a helper that writes a file (a
    chart, an export) runs every time, as the write itself would, unless the
    helper is marked pure. Otherwise an unmarked helper's statements already
    cache. Use :func:`stateful` when you need to stop a statement from caching.

    Args:
        func: The function to mark as pure.

    Returns:
        The same function with a ``_cash_pure`` attribute set to True.

    Example::

        @pure
        def compute(x, y):
            return x + y
    """
    setattr(func, _PURE_ATTR, True)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    setattr(wrapper, _PURE_ATTR, True)
    return wrapper  # type: ignore[return-value]  # wrapper preserves F's signature via @wraps


def stateful(func: F) -> F:
    """Mark a function as stateful (has side effects).

    Stateful functions should never be cached because their return value
    alone does not capture their full effect. The notebook caching system
    will skip caching for statements that call stateful functions.

    Args:
        func: The function to mark as stateful.

    Returns:
        The same function with a ``_cash_stateful`` attribute set to True.

    Example::

        @stateful
        def train_model(data):
            model.fit(data)
            return model.score(data)
    """
    setattr(func, _STATEFUL_ATTR, True)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    setattr(wrapper, _STATEFUL_ATTR, True)
    return wrapper  # type: ignore[return-value]  # wrapper preserves F's signature via @wraps


def is_pure(func: Any) -> bool:
    """Check if a function is marked as pure.

    Args:
        func: The function or callable to check.

    Returns:
        True if the function has the ``_cash_pure`` attribute set to True.
    """
    return getattr(func, _PURE_ATTR, False) is True


def is_stateful(func: Any) -> bool:
    """Check if a function is marked as stateful.

    Args:
        func: The function or callable to check.

    Returns:
        True if the function has the ``_cash_stateful`` attribute set to True.
    """
    return getattr(func, _STATEFUL_ATTR, False) is True


# ============================================================================
# Known-pure built-in and stdlib functions
# ============================================================================

# These built-in functions have no side effects and always return
# the same output for the same inputs.
KNOWN_PURE_BUILTINS: frozenset[str] = frozenset(
    {
        # Type constructors / conversions
        "int",
        "float",
        "str",
        "bool",
        "bytes",
        "complex",
        "list",
        "tuple",
        "set",
        "frozenset",
        "dict",
        # Numeric / math
        "abs",
        "round",
        "pow",
        "divmod",
        "min",
        "max",
        "sum",
        # Sequence / iteration
        "len",
        "sorted",
        "reversed",
        "enumerate",
        "zip",
        "range",
        "map",
        "filter",
        "all",
        "any",
        # Object introspection
        "type",
        "isinstance",
        "issubclass",
        "id",
        "hash",
        "callable",
        "hasattr",
        "getattr",
        "repr",
        "ascii",
        "format",
        "chr",
        "ord",
        "hex",
        "oct",
        "bin",
        # Containers
        "iter",
        "next",
        "slice",
    }
)


def is_known_pure(name: str) -> bool:
    """Check if a function name is a known-pure built-in.

    This allows the statement processor to skip mutation detection
    for statements that only call known-pure built-in functions,
    even without explicit ``@pure`` annotations.

    Args:
        name: The function name to check.

    Returns:
        True if the name is in the known-pure builtins list.
    """
    return name in KNOWN_PURE_BUILTINS


# ============================================================================
# Name tables shared by the purity analyzer and the decorator's effect checks
# ============================================================================

# Operations that indicate impurity (side effects or global state access)
_IMPURE_FUNCTION_CALLS = frozenset(
    {
        "print",
        "input",
        "open",
        "exec",
        "eval",
        "compile",
        "exit",
        "quit",
        "breakpoint",
    }
)

_IMPURE_MODULE_CALLS = frozenset(
    {
        "os.system",
        "os.remove",
        "os.unlink",
        "os.mkdir",
        "os.makedirs",
        "os.rename",
        "os.replace",
        "os.rmdir",
        "subprocess.run",
        "subprocess.call",
        "subprocess.Popen",
        "subprocess.check_call",
        "subprocess.check_output",
        "shutil.copy",
        "shutil.copy2",
        "shutil.move",
        "shutil.rmtree",
        "logging.info",
        "logging.debug",
        "logging.warning",
        "logging.error",
        "logging.critical",
        "requests.get",
        "requests.post",
        "requests.put",
        "requests.delete",
        # `patch` was missing beside its four siblings; `request` is the generic
        # form they all delegate to, so naming only the verbs left a hole.
        "requests.patch",
        "requests.request",
        "json.dump",
        "pickle.dump",
    }
)

#: Calls that READ ambient state — the clock, the environment, the working
#: directory, a fresh UUID.
#:
#: A different category from everything above, which is why it is a separate
#: table. Those are side effects: they change the world, and a cache hit means
#: the change silently stops happening. These change nothing — they make the
#: function's RESULT depend on hidden state, so a cache hit freezes it. A
#: nightly job's `date.today()` returns the night it first ran, forever; a
#: multi-tenant job reading `os.environ['TENANT']` serves the first tenant's
#: answer to every other tenant, exit 0.
#:
#: Not routed through the randomness detector even though the hazard is the
#: same shape, because that filters through a seed ledger and there is no
#: `seed()` that makes `datetime.now()` reproducible — these can only ever be
#: unseeded, so they would fight the machinery they were folded into.
#:
#: The bar for adding a name: reading it twice in one program can give two
#: answers, through no argument the caller passed. `os.path.exists` fails that
#: bar (it is about a file, which the file-dependency tracker already owns);
#: `os.getpid` passes it but is not worth the noise.
_AMBIENT_READ_CALLS = frozenset(
    {
        "datetime.now",
        "datetime.utcnow",
        "datetime.today",
        "datetime.datetime.now",
        "datetime.datetime.utcnow",
        "datetime.datetime.today",
        "datetime.date.today",
        "date.today",
        "time.time",
        "time.time_ns",
        "time.monotonic",
        "time.perf_counter",
        "os.getcwd",
        "os.getenv",
        "os.environ.get",
        "uuid.uuid1",
        "uuid.uuid4",
        "pandas.Timestamp.now",
        "pandas.Timestamp.today",
        "pandas.Timestamp.utcnow",
    }
)

#: Constructors that read the clock only when a string argument says so:
#: ``pd.to_datetime("today")``, ``pd.Timestamp("now")``, ``np.datetime64("now")``.
_AMBIENT_WHEN_ARG_CALLS = frozenset(
    {
        "pandas.to_datetime",
        "pandas.Timestamp",
        "numpy.datetime64",
    }
)
_AMBIENT_ARG_VALUES = frozenset({"now", "today"})

#: Functions that read the clock when their time argument is LEFT OUT: called
#: with at most this many positional arguments. ``time.strftime("%Y-%m")``
#: froze a report's period with no warning (round 20), while
#: ``time.strftime("%Y-%m", t)`` only formats ``t`` -- as ``time.localtime(ts)``
#: only converts, which the flat list above used to warn about.
_AMBIENT_WHEN_ARGS_OMITTED: dict[str, int] = {
    "time.strftime": 1,
    "time.asctime": 0,
    "time.ctime": 0,
    "time.localtime": 0,
    "time.gmtime": 0,
}

#: Method names meaning "this call changed something outside the function".
#:
#: Matched on ANY receiver, because a receiver's type is not knowable from
#: source. That is deliberate, and it is the only thing that can reach a side
#: effect inside an INSTALLED library: the analyzer stops at library
#: boundaries, so `session.post(...)` is invisible to it except through this
#: name.
#:
#: Without the verbs below, a library effect was caught only when its return
#: value was DISCARDED (by the separate discarded-call rule) -- so
#: `requests.post(url)` warned while `r = session.post(url); return r.json()`
#: did not, though they are the same operation and the second is how
#: production code is written.
#:
#: The bar for adding a name: effect-shaped on essentially every type that
#: defines it. `get` fails that bar (`dict.get`), which is why a *read* over a
#: client object still cannot be reached by name at all. A false positive here
#: costs an advisory warning; a false negative costs a write that silently
#: stops happening on every cache hit.
_WRITE_METHODS = frozenset(
    {
        "write",
        "writelines",
        "send",
        "sendall",
        "sendto",
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
        "to_csv",
        "to_excel",
        "to_parquet",
        "to_json",
        "to_pickle",
        "savefig",
        "save",
        "write_text",
        "write_bytes",  # pathlib.Path
        # HTTP verbs that change server state, reached through a client object
        # (a requests Session, httpx, an SDK wrapper) rather than the
        # module-qualified form already listed above.
        "post",
        "put",
        "patch",
        # Database writes, reached through a cursor / connection / collection.
        "execute",
        "executemany",
        "executescript",
        "commit",
        "rollback",
        # Object stores and message buses.
        "upload",
        "upload_file",
        "upload_fileobj",
        "put_object",
        "publish",
    }
)
