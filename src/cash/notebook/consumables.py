"""Classification and divergence probing for *consumable, unrestorable* inputs.

A **consumable** is an object that is advanced/drained IN PLACE by reading it:
a generator, a ``queue.Queue``, an ``itertools`` iterator, an open file handle.
An **unrestorable** consumable is one the cache cannot faithfully snapshot:
``InMemoryBackend._safe_deep_copy`` (``backends/memory_backend.py``) deep-copies
every stored value, but falls back to storing the *reference* when the copy
raises — so "restoring" such a value hands back the very same, already-drained
object.

Together those two properties break the isolated re-run of a consumer cell: the
cell's required input is a live object that its producer left in one state and
the previous run left in another, and nothing can put it back. The fix is to
re-execute the producer, which is exactly what ``run_all`` does (and why
``run_all`` is already correct for these cases).

**Why a bespoke probe instead of the existing content-hash check?**
``value_hash.compute_hash`` falls back to ``sha256(str(id(obj)))`` for
unpicklable objects. Consumables drain in place, so their identity never
changes and their hash is byte-identical before and after being drained. The
simulator's content-base staleness check therefore *cannot* fire for them; each
type needs its own state observation. These probes are deliberately kept
separate from cache-key derivation — they never touch ``source_hash`` and never
recompute a cache key (see the unified-cache-key rule in
``.github/copilot-instructions.md``).

**Load-bearing distinction: "unpicklable" is NOT "consumable".**
``map`` / ``zip`` / ``filter`` / ``enumerate`` / ``iter(list)`` /
``iter(range(n))`` / ``reversed`` / ``io.StringIO`` are all self-iterators that
drain in place, but they are *deep-copyable*, so the store snapshots them fresh
at ``set`` time and they restore correctly. They must NOT be classified here or
their producers would be re-executed for nothing. Conversely a ``dict.keys()``
view is not deep-copyable but is not a consumable either (re-iterating a view
works fine). Classification therefore requires **both** signals: the object is
a self-iterator *and* it hits the by-ref fallback.
"""

from __future__ import annotations

import ast
import collections
import functools
import inspect
import io
import itertools
import logging
import queue
import sys
import types
from collections.abc import Iterable, Mapping
from typing import Any

from cash._user_attrs import attribute_or, has_attribute, instance_of

from ..analysis.mutation_effects import captured_call_receivers
from ..analysis.mutations import consumed_input_names
from ..install_paths import is_user_code_file, is_user_code_module

logger = logging.getLogger(__name__)

__all__ = [
    "drawn_stream_inputs",
    "held_stream_paths",
    "is_consumable_unrestorable",
    "consumable_state",
    "has_diverged",
    "is_write_stream",
]

# ``itertools`` iterators that hold advanceable state and are not deep-copyable.
# Resolved to real types once at import; every public itertools iterator is a
# C-level self-iterator, so the generic test below also catches any we miss.
_ITERTOOLS_TYPES: tuple[type, ...] = tuple(
    t
    for t in (
        attribute_or(itertools, name, None)
        for name in (
            "count",
            "cycle",
            "chain",
            "repeat",
            "accumulate",
            "compress",
            "dropwhile",
            "filterfalse",
            "groupby",
            "islice",
            "permutations",
            "combinations",
            "combinations_with_replacement",
            "product",
            "starmap",
            "takewhile",
            "zip_longest",
            "pairwise",
            "batched",
        )
    )
    if instance_of(t, type)
)

_GENERATOR_TYPES: tuple[type, ...] = (
    types.GeneratorType,
    types.AsyncGeneratorType,
    types.CoroutineType,
)

_QUEUE_TYPES: tuple[type, ...] = tuple(
    t for t in (attribute_or(queue, n, None) for n in ("Queue", "SimpleQueue")) if instance_of(t, type)
)


def _is_self_iterator(obj: Any) -> bool:
    """True when ``iter(obj) is obj`` — i.e. reading *obj* advances *obj*.

    Excludes re-iterable containers and live views (``list``, ``dict.keys()``),
    whose ``__iter__`` hands back a fresh cursor each time.
    """
    cls = type(obj)
    if not has_attribute(cls, "__iter__") or not has_attribute(cls, "__next__"):
        return False
    try:
        return iter(obj) is obj
    except (TypeError, ValueError, AttributeError):
        return False


def _hits_byref_fallback(obj: Any) -> bool:
    """True when the cache store cannot snapshot *obj* and keeps a reference.

    ``InMemoryBackend._safe_deep_copy`` decides this with ``copy.deepcopy``, but
    deep-copying is far too expensive to use as a classifier probe: a
    ``map`` over a 2M-element list deep-copies the whole list (~0.1s) where
    ``__reduce_ex__`` is ~3µs, because it only *describes* how to rebuild the
    object rather than rebuilding it. ``__reduce_ex__(4)`` is the protocol
    ``deepcopy`` itself consults for objects without ``__deepcopy__`` /
    ``__copy__``, and it was verified to agree with ``deepcopy`` on every
    iterator type in this module's remit.

    (It disagrees for ``queue.Queue`` — a plain Python object whose ``reduce``
    succeeds while ``deepcopy`` chokes on its internal ``threading.Lock`` — so
    queues are classified by an explicit ``isinstance`` branch instead.)
    """
    if has_attribute(type(obj), "__deepcopy__") or has_attribute(type(obj), "__copy__"):
        return False
    try:
        obj.__reduce_ex__(4)
    except Exception:  # noqa: BLE001 - any failure means deepcopy would fall back
        return True
    return False


def is_consumable_unrestorable(obj: Any) -> bool:
    """True when *obj* is drained in place AND cannot be faithfully restored.

    See the module docstring for why both halves are required.
    """
    if instance_of(obj, _QUEUE_TYPES):
        return True
    if instance_of(obj, _GENERATOR_TYPES):
        return True
    if not _is_self_iterator(obj):
        # Not a self-iterator: re-reading does not advance it (list, dict view,
        # ndarray), so there is nothing to diverge.
        return False
    if instance_of(obj, _ITERTOOLS_TYPES):
        return True
    return _hits_byref_fallback(obj)


# ---------------------------------------------------------------------------
# Per-type divergence probes
# ---------------------------------------------------------------------------
#
# ``consumable_state`` returns a cheap, comparable, NON-CONSUMING token for the
# object's drain position, or ``None`` when no probe exists for its type. The
# token is compared against a baseline recorded at the consumer cell's ENTRY
# (see ``TrackingState.consumable_bases``), which is what makes the check
# self-disabling: on ``run_all`` the producer re-runs first and hands the cell
# the same state it saw last time (token == baseline -> no-op), while on an
# isolated re-run the object still holds the previous run's drained state
# (token != baseline -> re-execute the producer).


def consumable_state(obj: Any) -> Any | None:
    """Return a non-consuming token describing how far *obj* has been drained.

    ``None`` means "no probe exists for this type" — the caller decides the
    policy for that case (see ``has_diverged``).
    """
    # Generators/coroutines expose their exact position as a state enum, and
    # reading it neither advances nor closes them.
    if instance_of(obj, types.GeneratorType):
        try:
            return ("gen", inspect.getgeneratorstate(obj))
        except (TypeError, AttributeError):
            return None
    if instance_of(obj, types.CoroutineType):
        try:
            return ("coro", inspect.getcoroutinestate(obj))
        except (TypeError, AttributeError):
            return None
    if instance_of(obj, types.AsyncGeneratorType):
        try:
            return ("agen", inspect.getasyncgenstate(obj))
        except (TypeError, AttributeError):
            return None
    # Queues report their depth directly. ``qsize`` is documented as
    # approximate under concurrency, but a notebook cell draining a queue is
    # the single reader, so it is exact here.
    if instance_of(obj, _QUEUE_TYPES):
        try:
            return ("qsize", obj.qsize())
        except (NotImplementedError, AttributeError, ValueError):
            return None
    # Open file handles: byte offset is exact and cheap. Closed/unseekable
    # streams raise -> no probe.
    if has_attribute(obj, "tell") and has_attribute(obj, "read"):
        try:
            return ("tell", obj.tell(), bool(attribute_or(obj, "closed", False)))
        except (OSError, ValueError, AttributeError):
            return None
    # ``itertools.count`` renders its next value (``count(5)``); the other
    # itertools iterators keep their cursor entirely in C with no observable
    # handle, so they fall through to ``None``.
    if instance_of(obj, attribute_or(itertools, "count", ())):
        try:
            return ("count", repr(obj))
        except Exception:  # noqa: BLE001 - repr of a foreign object
            return None
    return None


def has_diverged(obj: Any, baseline: Any, *, had_baseline: bool) -> bool:
    """True when *obj* is not in the state its consumer cell last found it in.

    ``baseline``/``had_baseline`` come from ``TrackingState.consumable_bases``,
    recorded at the cell's entry on its previous run.

    Policy for the unprobeable case (``consumable_state`` -> ``None``): report
    NOT diverged and leave the producer alone. ``itertools.cycle`` / ``chain`` /
    ``tee`` keep their cursor in C with nothing to observe, so the only
    alternative would be to assume divergence and re-execute their producer on
    every isolated re-run — trading a silent wrong answer for unconditional
    recompute of anything that touches them. That is a real (narrower) gap of
    the same family, and is deliberately left for a later stage rather
    than paid for by every cell here.
    """
    token = consumable_state(obj)
    if token is None:
        return False
    if not had_baseline:
        # First run of this cell (or the base was cleared): nothing to compare
        # against, so there is no evidence of staleness. Self-disables.
        return False
    return token != baseline


def is_write_stream(obj: Any) -> bool:
    """True for a file open for writing (``'w'``, ``'a'``, ``'r+'``, a gzip
    writer), whether or not it has been closed since.

    Its producer must never be re-run to reset it: re-opening the path and
    replaying the writes before a cell repeats side effects that a top-to-bottom
    run performs once. An append-mode log gets its earlier lines again, and a
    gzip writer opened a second time over a live one leaves a corrupt file.
    An in-memory buffer (``io.StringIO``, ``io.BytesIO``) writes nowhere else,
    so rebuilding it is safe and it does not count.
    """
    if instance_of(obj, (io.StringIO, io.BytesIO)):
        return False
    writable = attribute_or(obj, "writable", None)
    if not callable(writable) or not has_attribute(obj, "write"):
        return False
    try:
        return writable() is True
    except ValueError:
        # Closed: ``writable()`` refuses, the mode it was opened in remains.
        mode = attribute_or(obj, "mode", None)
        return isinstance(mode, str) and any(c in mode for c in "wax+")
    except (OSError, AttributeError):
        return False


def drawn_stream_inputs(
    tree: ast.Module | None,
    inputs: Iterable[str],
    outputs: Iterable[str],
    user_ns: Mapping[str, Any],
) -> list[str]:
    """The inputs a statement draws from that a cache hit could not advance.

    An iterator or stream held in a variable (``iter(rows)``, a
    ``csv.reader``, an open file, ``io.StringIO``, an ``islice``, a
    generator, a queue) moves when the statement reads from it:
    ``header = parse(next(rows))``. A stored entry holds the statement's
    outputs only, so a hit would leave the iterator where it was and the
    next reader would get the header again. Such a statement runs every
    time. A name the statement also rebinds is restored with the other
    outputs and does not count.
    """
    names = set(inputs) - set(outputs)
    if not names:
        return []
    drawn = consumed_input_names(tree) & names
    found = []
    for name in sorted(drawn):
        value = user_ns.get(name)
        if value is None:
            continue
        if is_stream(value):
            found.append(name)
    return found


#: How far the look for an iterator held out of sight goes: below what the
#: statement calls, through functions' globals, closure cells and defaults,
#: an instance's or class's attributes, items; and how many objects it looks
#: at in all.
_HELD_DEPTH = 6
_HELD_BUDGET = 2000
#: A container a variable indexes is looked through up to this many items.
_HELD_WIDTH = 64
#: Containers whose items are read in order.
_CONTAINERS = (list, tuple, set, frozenset, collections.deque)
#: pandas' indexers: ``s.iloc[i]`` reads an item of ``s``.
_INDEXERS = frozenset({"iloc", "loc", "iat", "at"})
#: ``Py_TPFLAGS_HEAPTYPE``: a class made by a ``class`` statement.
_HEAP_TYPE = 1 << 9


def held_stream_paths(tree: ast.Module | None, user_ns: Mapping[str, Any]) -> list[str]:
    """The iterators and generators a statement can draw from that no name
    it reads holds itself, as paths (``m.g``, ``f.gen``).

    One an attribute or item it spells holds (``next(m.g)``, ``next(gens[i])``,
    ``next(s.iloc[0])``), or one what it calls reaches -- a function through
    its globals, closure or defaults, a method through its instance or class,
    a callable instance, a ``partial``, a generator's own ``__next__``, a
    function it hands to a call -- moves when it runs, and no lineage follows
    it: a hit would leave it where it was, and the statement's outputs, served
    under the same key, would not be what a run gives. An open file is left
    to the file checks, which see what is read from it.

    Inside a library's objects and functions only a generator of the user's
    counts: a library keeps its own counters (matplotlib's callback ids). A
    look that runs out of objects before it is done gives the path it
    stopped at, so the statement runs rather than be served unlooked.
    """
    if tree is None:
        return []
    found: list[str] = []
    frontier: list[tuple[str, Any, int, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Attribute, ast.Subscript)) and not isinstance(node.ctx, ast.Store):
            found.extend(path for path, value in _spelled(node, user_ns) if _held_stream(value))
        if isinstance(node, ast.Call):
            frontier.extend((path, callee, 0, _owned(callee)) for path, callee in _callees(node.func, user_ns))
            for arg in [*node.args, *(k.value for k in node.keywords)]:
                frontier.extend((path, value, 0, True) for path, value in _spelled(arg, user_ns) if _callable(value))
    seen: set[int] = set()
    budget = _HELD_BUDGET
    while frontier and budget > 0:
        path, obj, depth, owned = frontier.pop()
        if id(obj) in seen or instance_of(obj, (str, bytes, int, float, complex, bool, type(None))):
            continue
        seen.add(id(obj))
        budget -= 1
        if depth > 0 and _held_stream(obj):
            if owned or _users_generator(obj):
                found.append(path)
            continue
        if depth >= _HELD_DEPTH:
            continue
        try:
            for label, child in _reached(obj):
                frontier.append(
                    (
                        f"{path}{label}" if label.startswith(("[", ".")) else f"{path}.{label}",
                        child,
                        depth + 1,
                        owned and _owned(child),
                    )
                )
        except Exception:  # noqa: BLE001 - a look into arbitrary objects must not break the statement
            continue
    if frontier and budget <= 0:
        found.append(f"{frontier[-1][0]} (the look stopped after {_HELD_BUDGET} objects)")
    return sorted(set(found))


def _callable(value: Any) -> bool:
    return instance_of(value, (types.FunctionType, types.MethodType, functools.partial))


def _owned(obj: Any) -> bool:
    """Whether what *obj* holds is the user's to draw from: anything but a
    library's object or function."""
    if instance_of(obj, types.FunctionType):
        return is_user_code_file(obj.__code__.co_filename)
    if instance_of(obj, (types.MethodType, functools.partial, classmethod, staticmethod, *_CONTAINERS)):
        return True
    if instance_of(obj, (types.SimpleNamespace, types.ModuleType)):
        return True
    if instance_of(obj, type):
        return _users_class(obj)
    return _users_class(type(obj))


def _users_generator(obj: Any) -> bool:
    """Whether *obj* is a generator the user's code made."""
    code = getattr(obj, "gi_code", None) or getattr(obj, "ag_code", None)
    return code is not None and is_user_code_file(code.co_filename)


def _reached(obj: Any) -> Iterable[tuple[str, Any]]:
    """What calling or reading *obj* can reach: `hidden_generators.children`,
    and a function's defaults, a ``partial``'s function and arguments, a
    builtin method's object, a class of the user's and its attributes, an
    instance's class, the items of a container a library makes (a deque, a
    pandas object column)."""
    from .statement.hidden_generators import children

    if instance_of(obj, type):
        if _users_class(obj):
            for name, value in list(object.__getattribute__(obj, "__dict__").items()):
                if not name.startswith("__") or name == "__call__":
                    yield f".{name}", _unwrapped(value)
        return
    if instance_of(obj, (classmethod, staticmethod)):
        yield "__func__", obj.__func__
        return
    if instance_of(obj, functools.partial):
        yield "func", obj.func
        for i, arg in enumerate(obj.args[:_HELD_WIDTH]):
            yield f"args[{i}]", arg
        for key, arg in list(obj.keywords.items())[:_HELD_WIDTH]:
            yield f"keywords[{key!r}]", arg
        return
    if instance_of(obj, (types.BuiltinMethodType, types.MethodWrapperType)):
        owner = obj.__self__
        if not instance_of(owner, types.ModuleType):
            yield "__self__", owner
        return
    items = _items(obj)
    if items is not None:
        yield from items
        if not instance_of(obj, (dict, list, tuple)):
            return
    yield from children(obj)
    if instance_of(obj, types.FunctionType):
        for i, value in enumerate(obj.__defaults__ or ()):
            yield f"__defaults__[{i}]", value
        for key, value in (obj.__kwdefaults__ or {}).items():
            yield f"__kwdefaults__[{key!r}]", value
        return
    cls = type(obj)
    if not instance_of(obj, types.MethodType) and _users_class(cls):
        yield "__class__", cls


def _items(owner: Any) -> list[tuple[str, Any]] | None:
    """The items of a container, up to `_HELD_WIDTH` of them, read without
    running the user's code: a list, tuple, deque or dict of any kind; a
    pandas object column or a numpy array of objects. ``None`` for anything
    else."""
    if instance_of(owner, dict):
        return [(f"[{k!r}]", v) for k, v in itertools.islice(dict.items(owner), _HELD_WIDTH)]
    if instance_of(owner, _CONTAINERS):
        return [(f"[{i}]", v) for i, v in enumerate(itertools.islice(owner, _HELD_WIDTH))]
    module = type(owner).__module__ or ""
    if module.startswith("numpy") and getattr(getattr(owner, "dtype", None), "kind", None) == "O":
        return [(f"[{i}]", v) for i, v in enumerate(itertools.islice(owner.flat, _HELD_WIDTH))]
    if module.split(".")[0] == "pandas":
        name = type(owner).__name__
        if name == "Series" and owner.dtype.kind == "O":
            return [(f".iloc[{i}]", v) for i, v in enumerate(itertools.islice(owner.array, _HELD_WIDTH))]
        if name == "DataFrame":
            found = []
            for col in [c for c, dtype in owner.dtypes.items() if dtype.kind == "O"][:_HELD_WIDTH]:
                found.extend(
                    (f"[{col!r}].iloc[{i}]", v) for i, v in enumerate(itertools.islice(owner[col].array, _HELD_WIDTH))
                )
            return found
    return None


def _users_class(cls: type) -> bool:
    """Whether *cls* was defined in the user's code: a function of its own
    was compiled from a notebook cell or a file of theirs, or, with none, its
    module is theirs (a notebook's ``__main__``). A library's class keeps its
    counters and iterators to itself, as `hidden_generators` treats them."""
    try:
        attrs = object.__getattribute__(cls, "__dict__")
    except (AttributeError, TypeError):
        return False
    for value in list(attrs.values())[:_HELD_WIDTH]:
        if instance_of(value, (types.FunctionType, classmethod, staticmethod)):
            code = getattr(_unwrapped(value), "__code__", None)
            if code is not None:
                return is_user_code_file(code.co_filename)
    name = getattr(cls, "__module__", None) or ""
    if name == "builtins":
        # A class ``exec``'d without a module of its own; the interpreter's
        # are static types.
        return bool(cls.__flags__ & _HEAP_TYPE)
    module = sys.modules.get(name)
    return module is None or is_user_code_module(module)


def _unwrapped(value: Any) -> Any:
    return value.__func__ if instance_of(value, (classmethod, staticmethod)) else value


def _callees(func: ast.expr, user_ns: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """What the call ``func(...)`` runs, with what it runs on: the function a
    name, an item or an attribute holds; a method with its instance (or
    class); read statically, so nothing of the objects' runs."""
    if isinstance(func, ast.Attribute):
        found = []
        for path, owner in _spelled(func.value, user_ns):
            try:
                attr = inspect.getattr_static(owner, func.attr)
            except (AttributeError, TypeError):
                continue
            found.append((f"{path}.{func.attr}", _unwrapped(attr)))
            if instance_of(attr, (types.FunctionType, classmethod)):
                found.append((path, owner))
        return found
    return _spelled(func, user_ns)


def _spelled(node: ast.expr, user_ns: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """The paths and values of an attribute or item chain over a name
    (``m.g``, ``d['g']``, ``gens[i]``: an item a variable picks stands for
    each item of a small list, tuple or dict), read from the objects' own
    dicts so nothing of theirs runs; empty when it cannot be read so."""
    if isinstance(node, ast.Name):
        return [(node.id, user_ns[node.id])] if node.id in user_ns else []
    if isinstance(node, ast.Attribute):
        found = []
        for path, owner in _spelled(node.value, user_ns):
            if node.attr in _INDEXERS and (type(owner).__module__ or "").split(".")[0] == "pandas":
                # ``s.iloc[0]``: the accessor indexes the object itself.
                found.append((f"{path}.{node.attr}", owner))
                continue
            try:
                attrs = object.__getattribute__(owner, "__dict__")
            except (AttributeError, TypeError):
                continue
            if instance_of(attrs, (dict, types.MappingProxyType)) and node.attr in attrs:
                found.append((f"{path}.{node.attr}", _unwrapped(attrs[node.attr])))
        return found
    if isinstance(node, ast.Subscript):
        found = []
        for path, owner in _spelled(node.value, user_ns):
            if type(owner) in (dict, list, tuple) and isinstance(node.slice, ast.Constant):
                try:
                    found.append((f"{path}[{node.slice.value!r}]", owner[node.slice.value]))
                except (KeyError, IndexError, TypeError):
                    pass
                continue
            # An item a variable picks, or one of a container whose lookup
            # runs its own code: any item it holds.
            items = _items(owner)
            if items is not None:
                found.extend((f"{path}{label}", item) for label, item in items)
        return found
    return []


def _held_stream(obj: Any) -> bool:
    return not instance_of(obj, io.IOBase) and is_stream(obj)


def is_stream(obj: Any) -> bool:
    """True when reading *obj* moves it: an iterator, generator, queue or
    open file, the values `drawn_stream_inputs` answers for."""
    try:
        return _is_self_iterator(obj) or is_consumable_unrestorable(obj)
    except (TypeError, ValueError, AttributeError, RecursionError):
        return False


def watched_call_receivers(tree: ast.Module | None, user_ns: Mapping[str, Any]) -> frozenset[str]:
    """`captured_call_receivers` without the streams: a call on a queue, an
    iterator or a file (`n = q.qsize()`, `line = fh.readline()`) is judged by
    `drawn_stream_inputs`, which tells a draw from an inspection. Watching
    it as well would fingerprint an object that cannot be pickled, count it
    as changed, and re-run an inspection a hit serves correctly."""
    return frozenset(name for name in captured_call_receivers(tree, user_ns) if not is_stream(user_ns.get(name)))
