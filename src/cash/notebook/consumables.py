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
import inspect
import io
import itertools
import logging
import queue
import types
from collections.abc import Iterable, Mapping
from typing import Any

from cash._user_attrs import attribute_or, has_attribute, instance_of, own_dict

from ..analysis.mutation_effects import captured_call_receivers
from ..analysis.mutations import consumed_input_names

logger = logging.getLogger(__name__)

__all__ = [
    "called_stream_readers",
    "drawn_stream_inputs",
    "reaches_a_stream",
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


#: How many objects down from a function's globals `reaches_a_stream` looks.
_REACH_DEPTH = 4
#: How many functions it follows.
_REACH_FUNCTIONS = 64
_ABSENT = object()


def called_stream_readers(tree: ast.Module | None, inputs: Iterable[str], user_ns: Mapping[str, Any]) -> list[str]:
    """The functions among *inputs* the statement *tree* calls that can draw
    from an iterator or stream the notebook holds (`reaches_a_stream`)."""
    if tree is None:
        return []
    inputs = set(inputs)
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in inputs
    }
    return sorted(
        name
        for name in called
        if instance_of(user_ns.get(name), types.FunctionType) and reaches_a_stream(user_ns[name])
    )


def reaches_a_stream(fn: Any) -> bool:
    """Can a call of *fn* draw from an iterator or stream (`is_stream`) the
    notebook holds, which a stored result could not move on?

    Looked for along the names *fn*'s code reads: a global (``next(it)``), an
    attribute of one (``next(Rec.it)``, ``next(R.it)``), an item it names by
    a constant (``b['c'].it`` with ``b=B``), in its defaults and closure too,
    and the same in every notebook function those reach. Attributes are read
    from the objects' own dicts, so no property runs.
    """
    seen_fns: set[int] = set()
    stack = [fn]
    while stack and len(seen_fns) < _REACH_FUNCTIONS:
        f = stack.pop()
        f = attribute_or(f, "__func__", f)
        code = attribute_or(f, "__code__", None)
        if not instance_of(code, types.CodeType) or id(f) in seen_fns:
            continue
        seen_fns.add(id(f))
        names: set[str] = set()
        keys: set[Any] = set()
        codes = [code]
        while codes:
            c = codes.pop()
            names.update(c.co_names)
            for const in c.co_consts:
                if isinstance(const, types.CodeType):
                    codes.append(const)
                elif isinstance(const, (str, int)) and not isinstance(const, bool):
                    keys.add(const)
        f_globals = attribute_or(f, "__globals__", None) or {}
        values = [f_globals[name] for name in names if isinstance(f_globals, dict) and name in f_globals]
        values.extend(_cell_value(cell) for cell in attribute_or(f, "__closure__", None) or ())
        values.extend(attribute_or(f, "__defaults__", None) or ())
        values.extend((attribute_or(f, "__kwdefaults__", None) or {}).values())
        seen: set[int] = set()
        if any(_holds_a_stream(v, names, keys, _REACH_DEPTH, seen, stack) for v in values):
            return True
    return False


def _holds_a_stream(value: Any, names: set[str], keys: set[Any], depth: int, seen: set[int], fns: list) -> bool:
    if value is None or id(value) in seen:
        return False
    seen.add(id(value))
    if instance_of(value, (str, bytes, int, float, bool, types.ModuleType)):
        return False
    if instance_of(value, types.FunctionType):
        # Followed as code: the names its own body reads.
        fns.append(value)
        return False
    if is_stream(value):
        return True
    if depth <= 0:
        return False
    children: list[Any] = []
    try:
        if type(value) is dict:
            children.extend(dict.get(value, k) for k in keys if dict.__contains__(value, k))
        elif type(value) in (list, tuple):
            children.extend(value[k] for k in keys if isinstance(k, int) and -len(value) <= k < len(value))
        for name in names:
            child = _static_attr(value, name)
            if child is not _ABSENT:
                children.append(child)
    except (TypeError, ValueError, RecursionError):
        return False
    return any(_holds_a_stream(child, names, keys, depth - 1, seen, fns) for child in children)


def _static_attr(obj: Any, name: str) -> Any:
    """``obj.name`` as its own dict or its class's holds it, or `_ABSENT`;
    nothing of the object's runs."""
    if instance_of(obj, type):
        klasses = type.__dict__["__mro__"].__get__(obj)
    else:
        own = own_dict(obj)
        if own is not None and dict.__contains__(own, name):
            return dict.get(own, name)
        klasses = type.__dict__["__mro__"].__get__(type(obj))
    for klass in klasses:
        found = type.__dict__["__dict__"].__get__(klass)
        if name in found:
            return found[name]
    return _ABSENT


def _cell_value(cell: Any) -> Any:
    try:
        return cell.cell_contents
    except ValueError:
        return None
