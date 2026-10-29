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
from ..analysis.mutations import NON_CONSUMING_FUNCS, consumed_input_names
from .read_handles import reaches_read_handle

logger = logging.getLogger(__name__)

__all__ = [
    "GENERATOR_TYPES",
    "QUEUE_TYPES",
    "drawn_stream_inputs",
    "is_consumable_unrestorable",
    "consumable_state",
    "has_diverged",
    "is_write_stream",
    "with_other_names",
    "without_read_handles",
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

GENERATOR_TYPES: tuple[type, ...] = (
    types.GeneratorType,
    types.AsyncGeneratorType,
    types.CoroutineType,
)

QUEUE_TYPES: tuple[type, ...] = tuple(
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
    if instance_of(obj, QUEUE_TYPES):
        return True
    if instance_of(obj, GENERATOR_TYPES):
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
    if instance_of(obj, QUEUE_TYPES):
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
    return list(drawn_stream_inputs_why(tree, inputs, outputs, user_ns))


#: Why `drawn_stream_inputs_why` counts an input: it is or holds a stream
#: the statement draws from, or it holds more than cash reads to tell.
STREAM = "stream"
UNREAD = "unread"


def drawn_stream_inputs_why(
    tree: ast.Module | None,
    inputs: Iterable[str],
    outputs: Iterable[str],
    user_ns: Mapping[str, Any],
) -> dict[str, str]:
    """`drawn_stream_inputs`, each name with why it counts: `STREAM`, or
    `UNREAD` for a holder the statement draws from by a path cash cannot
    follow that holds more than cash reads to rule a stream out."""
    names = set(inputs) - set(outputs)
    if not names:
        return {}
    drawn = consumed_input_names(tree) & names
    found: dict[str, str] = {}
    for name in sorted(drawn):
        value = user_ns.get(name)
        if value is None:
            continue
        why = STREAM if is_stream(value) else _holds_drawn_stream(tree, name, value, user_ns)
        if why:
            found[name] = why
    return found


# What cash reads of a holder when the statement's path into it cannot be
# followed exactly (``next(h[i])``, a property, ``h.get()``): objects other
# than numbers and strings looked into, items in all, and how deep. Past
# any of them it counts as a draw (`UNREAD`).
_HELD_SCAN = 20_000
_HELD_ITEMS = 2_000_000
_HELD_DEPTH = 6
_PLAIN_CONTAINERS = (list, tuple, dict, collections.deque)
_NO_STREAM = frozenset({int, float, complex, bool, str, bytes, type(None)})
_UNKNOWN = object()


def _holds_drawn_stream(tree: ast.Module | None, name: str, value: Any, user_ns: Mapping[str, Any]) -> str | None:
    """True when the statement reaches a stream inside *value* (``next(h[0])``,
    ``list(h['k'])``, ``next(b.it)``) and so draws from it as it would from a
    name that held it. Followed through a list, tuple, dict or deque by a
    constant index (or a name the statement does not bind) and through an
    attribute stored on the object. Where the path cannot be followed without
    running code (another index, a property, a slot, a method call) and what
    it gives is drawn from, any stream *value* holds counts (`STREAM`), and
    so does a holder too big to read (`UNREAD`). None when it draws none."""
    if tree is None:
        return None
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    bound = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load)}
    for node in ast.walk(tree):
        chain = _reach_chain(node, name)
        parent = parents.get(node)
        if not chain or (isinstance(parent, (ast.Attribute, ast.Subscript)) and parent.value is node):
            continue
        if isinstance(node, ast.Attribute) and isinstance(parent, ast.Call) and parent.func is node:
            node, parent = parent, parents.get(parent)
        drawn = not _left_where_it_is(node, parent)
        known = {k: v for k, v in user_ns.items() if k not in bound and type(v) in (int, str)}
        why = _chain_meets_stream(value, chain, drawn, known)
        if why:
            return why
    return None


def _left_where_it_is(node: ast.AST, parent: ast.AST | None) -> bool:
    """True when what *node* gives is not drawn from where *parent* puts it:
    dropped, bound to a name, compared, or only inspected (``len(..)``)."""
    if isinstance(parent, (ast.Expr, ast.Compare, ast.BinOp, ast.UnaryOp, ast.FormattedValue)):
        return True
    if isinstance(parent, (ast.Assign, ast.AnnAssign)) and parent.value is node:
        targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
        return all(isinstance(t, ast.Name) for t in targets)
    return (
        isinstance(parent, ast.Call)
        and isinstance(parent.func, ast.Name)
        and parent.func.id in NON_CONSUMING_FUNCS
        and node is not parent.func
    )


def _reach_chain(node: ast.AST, name: str) -> list[ast.AST] | None:
    """The subscripts and attributes, outermost last, between *node* and the
    name it starts from, or None when *node* does not start from *name*."""
    chain: list[ast.AST] = []
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        chain.append(node)
        node = node.value
    if not chain or not (isinstance(node, ast.Name) and node.id == name):
        return None
    chain.reverse()
    return chain


def _chain_meets_stream(value: Any, chain: list[ast.AST], drawn: bool, known: Mapping[str, Any]) -> str | None:
    for step in chain:
        if isinstance(step, ast.Attribute):
            try:
                found = inspect.getattr_static(value, step.attr)
            except (AttributeError, TypeError):
                return None
            if has_attribute(type(found), "__get__") and not instance_of(value, type):
                # A method, property or slot: what it gives cannot be read
                # without running it.
                return _holds_stream(value) if drawn and _notebook_holder(value) else None
            value = found
        elif not instance_of(value, _PLAIN_CONTAINERS):
            return None
        else:
            index = _constant_index(step.slice, known)
            if index is _UNKNOWN:
                return _holds_stream(value) if drawn else None
            value = _item(value, index)
            if value is _UNKNOWN:
                return None
        if is_stream(value) and not _an_output_stream(value):
            return STREAM
    return None


def _an_output_stream(obj: Any) -> bool:
    """The process's own output streams (``sys.stderr`` handed to a log
    handler): written to, never read, so reaching one draws nothing."""
    return any(obj is attribute_or(sys, name, None) for name in ("stdout", "stderr", "__stdout__", "__stderr__"))


def with_other_names(names: Iterable[str], user_ns: Mapping[str, Any], known: Iterable[str]) -> set[str]:
    """*names* and the names of *known* bound to the same object as one of
    them: ``it2 = it`` above, a draw through ``it`` moved ``it2`` as well.
    The simulation moves ``it2`` on through the edge ``it2 = it`` linked; the
    runtime moves it with the drawn name, so both leave it alike."""
    held = {id(user_ns[name]): user_ns[name] for name in names if name in user_ns}
    found = set(names)
    for name in known:
        value = user_ns.get(name, held)
        if value is not held and held.get(id(value), held) is value:
            found.add(name)
    return found


def without_read_handles(names: Iterable[str], user_ns: Mapping[str, Any]) -> list[str]:
    """*names* but those bound to a file open for reading, or to what reaches
    one (``rd = f.readline``). Such a handle is named by its file and where it
    stands (`read_handle_digest`), and what reads it runs every time, so it
    is not moved as a generator is: a cell that only looks at it, re-run on
    its own, reads it where it stands, and the ``open()`` above stays as it
    ran."""
    return [name for name in names if not reaches_read_handle(user_ns.get(name))]


def _constant_index(node: ast.AST, known: Mapping[str, Any]) -> Any:
    """The index *node* spells as a literal (``0``, ``-1``, ``'k'``) or a name
    in *known*, else _UNKNOWN."""
    if isinstance(node, ast.Name):
        return known.get(node.id, _UNKNOWN)
    if isinstance(node, ast.Slice):
        return _UNKNOWN
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return _UNKNOWN


def _item(container: Any, index: Any) -> Any:
    """``container[index]`` read without a subclass's own lookup (a
    ``defaultdict`` would add the key), or _UNKNOWN when it is not there."""
    try:
        if isinstance(container, dict):
            return dict.__getitem__(container, index) if dict.__contains__(container, index) else _UNKNOWN
        if isinstance(container, list):
            return list.__getitem__(container, index)
        if isinstance(container, tuple):
            return tuple.__getitem__(container, index)
        return collections.deque.__getitem__(container, index)
    except (LookupError, TypeError):
        return _UNKNOWN


def _notebook_holder(value: Any) -> bool:
    return instance_of(value, _PLAIN_CONTAINERS) or attribute_or(type(value), "__module__", None) == "__main__"


def _holds_stream(value: Any) -> str | None:
    """`STREAM` when a stream sits anywhere in *value* that cash reads (the
    items of a list, tuple, set, dict or deque; the stored attributes and
    slots of an object of a class the notebook defines), `UNREAD` when
    *value* holds more than cash reads, else None. Numbers and strings are
    passed over in bulk, at C speed."""
    seen: set[int] = set()
    stack = [(value, 0)]
    left, items_left = _HELD_SCAN, _HELD_ITEMS
    while stack:
        obj, depth = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if is_stream(obj) and not _an_output_stream(obj):
            return STREAM
        items = _held_items(obj)
        if items is None:
            continue
        if depth >= _HELD_DEPTH:
            return UNREAD
        taken = list(itertools.islice(items, items_left + 1))
        items_left -= len(taken)
        if items_left < 0:
            return UNREAD
        if not _NO_STREAM.issuperset(map(type, taken)):
            # Only what is not a number or a string is looked into.
            taken = [item for item in taken if type(item) not in _NO_STREAM]
            left -= len(taken)
            if left < 0:
                return UNREAD
            stack.extend(zip(taken, itertools.repeat(depth + 1)))
    return None


def _held_items(obj: Any) -> Iterable[Any] | None:
    """What *obj* holds that `_holds_stream` reads, or None for nothing to read."""
    if isinstance(obj, dict):
        return dict.values(obj)
    if isinstance(obj, list):
        return list.__iter__(obj)
    if isinstance(obj, tuple):
        return tuple.__iter__(obj)
    if isinstance(obj, frozenset):
        return frozenset.__iter__(obj)
    if isinstance(obj, set):
        return set.__iter__(obj)
    if isinstance(obj, collections.deque):
        return collections.deque.__iter__(obj)
    cls = type(obj)
    if attribute_or(cls, "__module__", None) != "__main__" or instance_of(obj, type):
        return None
    held: list[Any] = []
    try:
        held.extend(dict.values(object.__getattribute__(obj, "__dict__")))
    except (AttributeError, TypeError):
        pass
    for klass in cls.__mro__:
        slots = klass.__dict__.get("__slots__", ())
        for slot in (slots,) if isinstance(slots, str) else slots:
            member = klass.__dict__.get(slot)
            if isinstance(member, types.MemberDescriptorType):
                try:
                    held.append(member.__get__(obj, cls))
                except AttributeError:
                    pass
    return held


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
