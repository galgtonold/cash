"""One decorated function: what it was decorated with, and what this process
has learned about its calls."""

from __future__ import annotations

import ast
import datetime
import inspect
import math
import textwrap
import types
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from ..exceptions import SOURCE_RETRIEVAL_ERRORS
from ..source_reading import own_source

__all__ = [
    "CHUNK_MAX_BYTES",
    "CHUNK_MAX_ITEMS",
    "WARNINGS_MAX",
    "CachedFunction",
    "PurityMode",
    "checked_ttl",
    "new_stats",
]

#: A returned iterator's chunk closes at whichever of these it reaches first.
CHUNK_MAX_ITEMS = 1_000_000
CHUNK_MAX_BYTES = 1_000_000_000

#: Recent warnings `cache_info()` reports per function.
WARNINGS_MAX = 20

#: ``strict`` raises on a purity issue, ``silent`` (``assume_safe=True``)
#: suppresses the warnings, ``warn`` is the default.
PurityMode = Literal["warn", "strict", "silent"]

#: `CachedFunction.signature` before it is first read.
_UNREAD: Any = object()


_PASS_THROUGH = frozenset({inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD})


def _def_node(func: types.FunctionType) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """The ``def`` of *func* itself (not of what it wraps), or None."""
    try:
        tree = ast.parse(textwrap.dedent(own_source(func)))
    except (*SOURCE_RETRIEVAL_ERRORS, SyntaxError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func.__code__.co_name:
            return node
    return None


def _own_calls(node: ast.AST) -> list[ast.Call]:
    """The calls in *node*'s own body, not in a def, lambda or class inside it."""
    calls = []
    stack = list(node.body)
    while stack:
        current = stack.pop()
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        if isinstance(current, ast.Call):
            calls.append(current)
        stack.extend(ast.iter_child_nodes(current))
    return calls


def _is_name(node: ast.AST, name: str | None) -> bool:
    return name is not None and isinstance(node, ast.Name) and node.id == name


def _names_object(func: types.FunctionType, name: str, target: Any) -> bool:
    """Does *name*, read in *func*'s body, hold *target*?"""
    code = func.__code__
    if name in code.co_freevars and func.__closure__:
        try:
            return func.__closure__[code.co_freevars.index(name)].cell_contents is target
        except ValueError:
            return False
    return func.__globals__.get(name) is target


def passthrough_shape(func: Any, wrapped: Any) -> tuple[int, frozenset[str]] | None:
    """What the ``*args, **kwargs`` wrapper *func* adds when it calls *wrapped*:
    the number of positional arguments it puts before ``*args``, and the
    keywords it passes besides ``**kwargs`` -- ``(0, frozenset())`` for a
    wrapper that passes the call straight on. None when its source does not
    show one consistent way it passes them on.

    A wrapper that injects an argument (``f(LOG, *args, **kwargs)``, a
    session, click's ``pass_obj``) shifts the binding: the caller's first
    argument is the wrapped function's SECOND parameter.
    """
    if not isinstance(func, types.FunctionType):
        return None
    params = inspect.signature(func, follow_wrapped=False).parameters.values()
    star = next((p.name for p in params if p.kind is inspect.Parameter.VAR_POSITIONAL), None)
    starstar = next((p.name for p in params if p.kind is inspect.Parameter.VAR_KEYWORD), None)
    node = _def_node(func)
    if node is None:
        return None
    forwarding = [
        call
        for call in _own_calls(node)
        if any(isinstance(a, ast.Starred) and _is_name(a.value, star) for a in call.args)
        or any(k.arg is None and _is_name(k.value, starstar) for k in call.keywords)
    ]
    to_wrapped = [c for c in forwarding if isinstance(c.func, ast.Name) and _names_object(func, c.func.id, wrapped)]
    shapes = set()
    for call in to_wrapped or forwarding:
        leading = list(call.args)
        if star is not None:
            if not leading or not (isinstance(leading[-1], ast.Starred) and _is_name(leading[-1].value, star)):
                return None
            leading.pop()
        if any(isinstance(a, ast.Starred) for a in leading):
            return None
        spread = [k for k in call.keywords if k.arg is None]
        if [_is_name(k.value, starstar) for k in spread] != ([True] if starstar is not None else []):
            return None
        shapes.add((len(leading), frozenset(k.arg for k in call.keywords if k.arg is not None)))
    return shapes.pop() if len(shapes) == 1 else None


def follow_passthrough(func: Callable[..., Any]) -> tuple[Any, inspect.Signature, int, frozenset[str]]:
    """``(owner, its signature, injected positionals, injected keywords)``:
    the function whose parameters a call to *func* binds to, through every
    ``*args, **kwargs`` wrapper on the way, and what those wrappers add.

    A wrapper is followed only while it takes nothing but
    ``*args``/``**kwargs`` and its source shows how it passes them on
    (`passthrough_shape`); a ``__signature__`` set on the way is honoured,
    as ``inspect.signature`` does.
    """
    leading, keywords = 0, frozenset()
    for _ in range(32):
        own = inspect.signature(func, follow_wrapped=False)
        wrapped = getattr(func, "__wrapped__", None)
        kinds = {p.kind for p in own.parameters.values()}
        if (
            wrapped is None
            or "__signature__" in getattr(func, "__dict__", {})
            or not kinds
            or not kinds <= _PASS_THROUGH
        ):
            return func, own, leading, keywords
        shape = passthrough_shape(func, wrapped)
        if shape is None:
            return func, own, leading, keywords
        leading, keywords = leading + shape[0], keywords | shape[1]
        func = wrapped
    return func, inspect.signature(func), leading, keywords


def call_signature(func: Callable[..., Any]) -> inspect.Signature:
    """The signature a call to *func* binds to, for canonicalising its arguments.

    ``inspect.signature`` follows ``__wrapped__`` to the innermost function,
    which is right for a ``functools.wraps`` wrapper that passes
    ``*args, **kwargs`` straight through, and wrong for one with parameters
    of its own: ``def wrapper(x, factor=1)`` over ``def price(x,
    currency=3)`` bound ``price(10, 3)`` as ``currency=3`` -- the default --
    so it shared ``price(10)``'s entry. Wrong too for one that injects an
    argument: under ``f(LOG, *args, **kwargs)`` the caller's first argument
    is the wrapped function's second, so the parameters the wrappers fill
    are dropped (`follow_passthrough`). Without that, ignoring the injected
    ``log`` left the caller's real first argument out of the key.
    """
    _owner, sig, leading, keywords = follow_passthrough(func)
    if not leading and not keywords:
        return sig
    kept = []
    for param in sig.parameters.values():
        if leading and param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            leading -= 1
            continue
        if param.name in keywords and param.kind is not inspect.Parameter.VAR_KEYWORD:
            continue
        kept.append(param)
    return sig.replace(parameters=kept)


def checked_ttl(ttl: Any) -> float | None:
    """``ttl=`` as the decorator stores it: ``None``, or a finite number of
    seconds ``>= 0``. A `datetime.timedelta` becomes its seconds.

    Checked when the function is decorated, because the value is written
    into every entry's metadata, and every later lookup of those entries
    reads it: a ``ttl="300"`` read from a config file is refused here.

    Raises:
        TypeError: not a number, ``None`` or a timedelta (a str, a bool).
        ValueError: negative, NaN or infinite.
    """
    if ttl is None:
        return None
    if isinstance(ttl, datetime.timedelta):
        seconds = ttl.total_seconds()
        ttl = int(seconds) if seconds.is_integer() else seconds
    if isinstance(ttl, bool) or not isinstance(ttl, (int, float)):
        hint = " Convert it with int(...) first." if isinstance(ttl, str) else ""
        raise TypeError(
            f"@cash.cache: ttl must be a number of seconds, a datetime.timedelta or None, "
            f"not {type(ttl).__name__} {ttl!r}.{hint}"
        )
    if math.isnan(ttl) or math.isinf(ttl) or ttl < 0:
        raise ValueError(
            f"@cash.cache: ttl must be a finite number of seconds >= 0, not {ttl!r}. "
            "Use ttl=None for entries that never expire, ttl=0 to recompute every call."
        )
    return ttl


def new_stats() -> dict[str, Any]:
    """The counters `cache_info()` and the run summary read."""
    return {
        "hits": 0,
        "misses": 0,
        "total_time_saved": 0.0,
        "lookup_seconds": 0.0,
        # What the misses cost besides their bodies: key, checks, store.
        "miss_overhead_seconds": 0.0,
        # What the misses were, and which results did not reach disk.
        "miss_reasons": Counter(),
        "not_persisted": Counter(),
        "not_stored": Counter(),
        # For "code or state changed" misses: WHAT changed.
        "changed": Counter(),
        # Calls that went straight through because caching is off.
        "bypassed": 0,
    }


@dataclass(eq=False)
class CachedFunction:
    """The options of one ``@cache`` decoration and its per-process state.

    A name decorated again (a notebook cell re-run) gets a new record; what
    was learned about the name's calls rather than its code carries over
    (`carry_over`).
    """

    func: Callable
    name: str
    dynamic_depends_on: Any = None
    #: Seconds, as `checked_ttl` accepted it.
    ttl: float | None = None
    cache_if: Callable[[Any], bool] | None = None
    chunk_max_items: int = CHUNK_MAX_ITEMS
    chunk_max_bytes: int = CHUNK_MAX_BYTES
    purity: PurityMode = "warn"
    frozen: bool = False
    allow_random: bool = False
    #: ``file_depends_on=`` as written.
    declared_files: tuple[str, ...] = ()
    #: ``key=`` or the ignored parameters (`arg_key.ArgKey`); None keys every argument.
    arg_key: Any = None

    #: The wrapper `Cash.cache` returned.
    wrapper: Callable | None = field(default=None, repr=False)
    stats: dict[str, Any] = field(default_factory=new_stats, repr=False)
    #: Recent warnings, newest last, at most `WARNINGS_MAX`.
    warnings: list[dict[str, Any]] = field(default_factory=list, repr=False)
    #: The key of the last call, for "what changed since the last call".
    last_key: str | None = None
    #: The costliest argument hash seen: ``(label, type, seconds, producer, old_pandas)``.
    arg_cost: tuple | None = None
    #: RNG modules its calls were seen drawing from; None until known.
    rng_modules: set[str] | None = None
    #: Seeding expressions fed by a parameter or a global (`seed_parameters`).
    seed_params: dict[str, tuple] = field(default_factory=dict, repr=False)
    #: Naming a changed argument cost more than its budget and is off (the
    #: check that some argument changed still runs).
    argument_naming_retired: bool = False
    #: The ``key=`` function has been watched for file reads (`KeyBuilder.key_arguments`).
    key_reads_checked: bool = False
    _signature: Any = field(default=_UNREAD, repr=False)

    @property
    def signature(self) -> inspect.Signature | None:
        """`call_signature` of the function, read once; None when it has none."""
        if self._signature is _UNREAD:
            try:
                self._signature = call_signature(self.func)
            except (ValueError, TypeError):
                self._signature = None
        return self._signature

    def carry_over(self, previous: CachedFunction) -> None:
        """Keep what *previous*, the name's earlier definition, learned about
        its calls: the last key (so the next miss can say the code changed),
        the warning log, the argument cost, the RNG verdict."""
        self.last_key = previous.last_key
        self.warnings = previous.warnings
        self.arg_cost = previous.arg_cost
        self.rng_modules = previous.rng_modules
        self.argument_naming_retired = previous.argument_naming_retired
