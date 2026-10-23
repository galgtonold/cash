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
import enum
import functools
import inspect
import io
import itertools
import logging
import operator
import queue
import sys
import types
import weakref
from collections.abc import Callable, Iterable, Mapping
from typing import Any, NamedTuple

from cash import _plain_data
from cash._memo import LruMemo
from cash._user_attrs import attribute_or, has_attribute, instance_of

from ..analysis.mutation_effects import captured_call_receivers
from ..analysis.mutations import consumed_input_names
from ..install_paths import is_user_code_file, is_user_code_module
from ..value_types import IMMUTABLE_VALUE_TYPES

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
#: an instance's or class's attributes, items; how many objects that can hold
#: something it looks into; and how many items of the user's containers it
#: reads in all. Past any of them on the user's objects, the look did not
#: finish, and the statement runs (`HeldStreams.stopped`).
_HELD_DEPTH = 16
_HELD_BUDGET = 2000
_HELD_ITEMS = 200_000
#: Inside a library's object, only this many of its items or attributes are
#: looked at: only a generator of the user's counts there.
_HELD_WIDTH = 64
#: Containers whose items are read in order.
_CONTAINERS = (list, tuple, set, frozenset, collections.deque)
#: Values that hold nothing: a container of only these is read for free.
_LEAVES = frozenset({str, bytes, bytearray, int, float, complex, bool, type(None)})
#: pandas' indexers: ``s.iloc[i]`` reads an item of ``s``.
_INDEXERS = frozenset({"iloc", "loc", "iat", "at"})
#: ``Py_TPFLAGS_HEAP_TYPE``: a class made by a ``class`` statement.
_HEAP_TYPE = 1 << 9


class HeldStreams(NamedTuple):
    """What `held_stream_paths` found: the paths of the iterators a
    statement can draw from that no name it reads holds, and, when the look
    could not finish, where it stopped and why."""

    paths: list[str]
    stopped: str | None = None


class _Look:
    """The budgets of one look (`held_stream_paths`)."""

    def __init__(self) -> None:
        self.objects = _HELD_BUDGET
        self.items = _HELD_ITEMS
        self.stopped: str | None = None
        #: Names the statement binds itself, a comprehension's, a lambda's,
        #: a loop's: what one holds when it runs is not what the notebook's
        #: variable of that name holds now (`_plain_index`).
        self.bound: frozenset[str] = frozenset()
        self.lineage_of: Callable[[Any], str | None] = _no_lineage

    def stop(self, why: str) -> None:
        if self.stopped is None:
            self.stopped = why


def held_stream_paths(
    tree: ast.Module | None, user_ns: Mapping[str, Any], lineages: Mapping[str, str] | None = None
) -> HeldStreams:
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

    Every item of the user's containers and every attribute of the user's
    objects is looked at; a value that holds nothing (an ``int``, a ``str``)
    costs no budget. Any stream the user's own function, container or object
    holds counts (an ``itertools.count()`` a function reads); inside a
    library's objects and functions only a
    generator of the user's counts: a library keeps its own counters
    (matplotlib's callback ids). A look that runs out of objects, items or
    depth on the user's objects says where it stopped (``stopped``), so the
    statement runs rather than be served unlooked.

    With *lineages*, the notebook's, a big container a variable holds is
    read once per lineage (`_holds_nothing_at`): 250,000 records cost
    0.6 s a look, paid on every hit.
    """
    if tree is None:
        return HeldStreams([])
    look = _Look()
    look.lineage_of = _lineage_finder(user_ns, lineages)
    look.bound = _bound_inside(tree)
    found: list[str] = []
    # (path, object, depth, whether what it holds is the user's, whether its
    # holder is the user's: a stream the user's object holds is the user's to
    # draw from, whoever made it)
    frontier: list[tuple[str, Any, int, bool, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Attribute, ast.Subscript)) and not isinstance(node.ctx, ast.Store):
            found.extend(path for path, value in _spelled(node, user_ns, look) if _held_stream(value))
        if isinstance(node, ast.Call):
            frontier.extend(
                (path, callee, 0, _owned(callee), True) for path, callee in _callees(node.func, user_ns, look)
            )
            for arg in [*node.args, *(k.value for k in node.keywords)]:
                frontier.extend(
                    (path, value, 0, True, True) for path, value in _spelled(arg, user_ns, look) if _callable(value)
                )
    seen: set[int] = set()
    while frontier:
        path, obj, depth, owned, held_by_user = frontier.pop()
        if _inert(type(obj)) or id(obj) in seen or instance_of(obj, (str, bytes, int, float, complex)):
            continue
        seen.add(id(obj))
        residue = _residue_at(obj, look.lineage_of) if depth > 0 else None
        if residue is not None:
            # Data, or records of data: only their classes can reach code.
            frontier.extend((f"{path}[...]", cls, depth + 1, owned and _owned(cls), owned) for cls in residue)
            continue
        if depth > 0 and _held_stream(obj):
            if held_by_user or _users_generator(obj):
                found.append(path)
            continue
        if depth >= _HELD_DEPTH:
            if owned:
                look.stop(f"{path}, deeper than {_HELD_DEPTH} steps")
            continue
        if type(obj) not in _CONTAINERS and type(obj) is not dict and not _plain_record(obj):
            look.objects -= 1
            if look.objects < 0:
                look.stop(f"{path}, past {_HELD_BUDGET} objects")
                break
        try:
            for label, child in _reached(obj, owned, look, path):
                if _inert(type(child)):
                    continue
                frontier.append(
                    (
                        f"{path}{label}" if label.startswith(("[", ".")) else f"{path}.{label}",
                        child,
                        depth + 1,
                        owned and _owned(child),
                        owned,
                    )
                )
        except Exception:  # noqa: BLE001 - a look into arbitrary objects must not break the statement
            continue
        if look.items < 0:
            break
    return HeldStreams(sorted(set(found)), look.stopped)


def held_consumable_path(name: str, value: Any, lineage: str | None = None) -> str | None:
    """Where inside *value* a generator, iterator or queue sits that a store
    could not copy (`is_consumable_unrestorable`), as a path (``s.iloc[70]``,
    ``reg['last'].g``); None when there is none.

    A statement binding such a container (``s = pd.Series([..., gen])``, a
    dict of the user's objects one of which holds a generator) was served:
    the restored value held the generator the earlier runs had already
    drawn from, and a reader below drew on from there. Read are every item
    of the user's containers, every attribute of the user's objects, and the
    items of a numpy or pandas object container, whose types are read at C
    speed; a library's own objects are not looked into. With *lineage*,
    *value*'s, a big one found to hold nothing is not read again."""
    if _inert(type(value)) or _residue_at(value, lambda obj: lineage if obj is value else None) is not None:
        return None
    stack: list[tuple[str, Any]] = [(name, value)]
    seen: set[int] = set()
    while stack:
        path, obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if obj is not value and is_consumable_unrestorable(obj):
            return path
        if obj is not value and _holds_nothing(obj):
            continue
        for label, child in _stored_children(obj):
            if not _inert(type(child)):
                stack.append((f"{path}{label}", child))
    return None


def _stored_children(obj: Any) -> list[tuple[str, Any]]:
    """What a store of *obj* stores with it, for `held_consumable_path`."""
    if type(obj) is dict or type(obj) in _CONTAINERS:
        items = dict.values(obj) if type(obj) is dict else obj
        if all(map(_inert, set(map(type, items)))):
            return []
        return _items_up_to(obj, sys.maxsize) or []
    if instance_of(obj, (types.FunctionType, types.MethodType, types.ModuleType, type)):
        return []
    found = _items_up_to(obj, sys.maxsize) if _object_container(obj) else None
    if found is not None:
        return found
    if instance_of(obj, (dict, *_CONTAINERS)) or _users_class(type(obj)):
        children = _items_up_to(obj, sys.maxsize) or []
        try:
            attrs = object.__getattribute__(obj, "__dict__")
        except (AttributeError, TypeError):
            attrs = None
        if isinstance(attrs, dict):
            children.extend((f".{k}", v) for k, v in list(attrs.items()))
        return children
    return []


def _module_of(kind: type) -> str:
    """The module a class says it comes from, or "". A metaclass that keeps
    ``__module__`` as a descriptor for its own classes (Cython's) answers
    with the descriptor when asked about itself."""
    name = attribute_or(kind, "__module__")
    return name if isinstance(name, str) else ""


def _object_container(obj: Any) -> bool:
    """A numpy array or pandas object holding Python objects, whose items
    are read at C speed: True only when one of them is not a leaf."""
    module = _module_of(type(obj)).split(".")[0]
    if module == "numpy":
        if attribute_or(attribute_or(obj, "dtype"), "kind") != "O":
            return False
        return not _only_inert(obj.flat)
    if module != "pandas":
        return False
    name = type(obj).__name__
    if name == "Series":
        return obj.dtype.kind == "O" and not _only_inert(obj.array)
    if name == "DataFrame":
        return any(dtype.kind == "O" and not _only_inert(obj[col].array) for col, dtype in obj.dtypes.items())
    return False


#: Value types that hold nothing a statement could draw from, tested by
#: subclass: dates, decimals, fractions, paths, UUIDs. An ``Enum`` member
#: is left out: its class is the user's and can hold anything.
_VALUE_TYPES = tuple(t for t in IMMUTABLE_VALUE_TYPES if t is not enum.Enum)

_INERT: weakref.WeakKeyDictionary[type, bool] = weakref.WeakKeyDictionary()


def _inert(kind: type) -> bool:
    """Whether a value of *kind* holds nothing: a str or a number, a date,
    a ``Decimal``, a numpy scalar. Such a value costs a look nothing, where
    2,500 dates had run it out of objects and the statement ran every
    time. Answered once per type."""
    if kind in _LEAVES:
        return True
    try:
        return _INERT[kind]
    except (KeyError, TypeError):
        pass
    numpy = sys.modules.get("numpy")
    try:
        answer = issubclass(kind, _VALUE_TYPES) or (numpy is not None and issubclass(kind, numpy.generic))
    except TypeError:
        answer = False
    try:
        _INERT[kind] = answer
    except TypeError:
        pass
    return answer


def _only_inert(values: Iterable[Any]) -> bool:
    """Whether every one of *values* is `_inert`, its types read at C speed."""
    return all(map(_inert, set(map(type, values))))


def _holds_nothing(obj: Any) -> bool:
    """Whether *obj* is data with nothing inside to draw from: JSON-like
    data, a builtin container of inert values or of tuples of them (a set of
    300,000 pairs), a numpy array or pandas object of no Python objects but
    inert ones. Read at C speed, a level at a time, never an item at a time:
    a look over 250,000 records took 1.4 s and then gave up."""
    kind = type(obj)
    if kind in (list, dict, tuple) and _plain_data.is_tree(obj):
        return True
    if kind in _CONTAINERS or kind is dict:
        items = list(dict.values(obj)) if kind is dict else obj
        kinds = set(map(type, items))
        rows = {k for k in kinds if not _inert(k)}
        if not rows:
            return True
        if not all(issubclass(k, tuple) and k.__iter__ is tuple.__iter__ for k in rows):
            return False
        if rows == kinds:
            return _only_inert(itertools.chain.from_iterable(items))
        return _only_inert(itertools.chain.from_iterable(item for item in items if type(item) in rows))
    module = _module_of(kind).split(".")[0]
    if module == "numpy":
        return attribute_or(obj, "dtype") is not None and not _object_container(obj)
    if module == "pandas" and kind.__name__ in ("Series", "DataFrame"):
        return not _object_container(obj)
    return False


#: What `_residue` found of a container, by its identity and the lineage
#: of the variable holding it: the same object under the same lineage was
#: not changed by any statement since.
_RESIDUES: LruMemo[tuple[int, str], tuple[type, ...] | None] = LruMemo(256)
#: A container this long is read once per lineage.
_BIG = 256
_UNREAD = object()


def _residue_at(obj: Any, lineage_of: Callable[[Any], str | None]) -> tuple[type, ...] | None:
    """`_residue`, remembered for a container a variable of the notebook
    holds (`_RESIDUES`)."""
    try:
        big = type(obj) in _MEMO_KINDS and len(obj) >= _BIG
    except TypeError:
        big = False
    lineage = lineage_of(obj) if big else None
    if lineage is None:
        return _residue(obj)
    key = (id(obj), lineage)
    answer = _RESIDUES.get(key, _UNREAD)
    if answer is _UNREAD:
        answer = _RESIDUES[key] = _residue(obj)
    return answer  # type: ignore[return-value]


def _residue(obj: Any) -> tuple[type, ...] | None:
    """What of *obj* a look must still read: nothing (``()``) when it
    `_holds_nothing`; the classes of its items when they are records of
    inert values (`_plain_record`), whose methods may reach a stream; None
    when it is to be looked into as usual."""
    if _holds_nothing(obj):
        return ()
    kind = type(obj)
    if kind not in _CONTAINERS and kind is not dict:
        return None
    classes: dict[type, None] = {}
    for item in dict.values(obj) if kind is dict else obj:
        item_kind = type(item)
        if _inert(item_kind) or item_kind in classes:
            if item_kind in classes and not _plain_record(item):
                return None
            continue
        if not _plain_record(item):
            return None
        classes[item_kind] = None
    return tuple(classes)


_MEMO_KINDS = frozenset({list, dict, tuple, set, frozenset, collections.deque})


def _no_lineage(_obj: Any) -> str | None:
    return None


def _lineage_finder(user_ns: Mapping[str, Any], lineages: Mapping[str, str] | None) -> Callable[[Any], str | None]:
    """The lineage of the variable holding an object, by the object; the
    map is built the first time one is asked for."""
    if not lineages:
        return _no_lineage
    by_id: dict[int, str] | None = None

    def lineage_of(obj: Any) -> str | None:
        nonlocal by_id
        if by_id is None:
            by_id = {}
            for name, lineage in list(lineages.items()):
                value = user_ns.get(name)
                if type(value) in _MEMO_KINDS:
                    by_id[id(value)] = lineage
        return by_id.get(id(obj))

    return lineage_of


def _plain_record(obj: Any) -> bool:
    """Whether *obj* is an object whose attributes are all `_inert` (a
    dataclass point, a record): it costs the look no object, so 3,000 of
    them do not run it out. Its class is still looked into."""
    if instance_of(obj, (type, types.FunctionType, types.MethodType, types.ModuleType)):
        return False
    try:
        attrs = object.__getattribute__(obj, "__dict__")
    except (AttributeError, TypeError):
        attrs = None
    if attrs is not None and not (isinstance(attrs, dict) and attrs and _only_inert(attrs.values())):
        return False
    read = attrs is not None
    for cls in type(obj).__mro__:
        for slot in attribute_or(cls, "__slots__", ()) or ():
            if not isinstance(slot, str) or slot in ("__dict__", "__weakref__"):
                continue
            try:
                value = object.__getattribute__(obj, slot)
            except (AttributeError, TypeError):
                continue
            if not _inert(type(value)):
                return False
            read = True
    return read


def _callable(value: Any) -> bool:
    return instance_of(value, (types.FunctionType, types.MethodType, functools.partial))


def _owned(obj: Any) -> bool:
    """Whether what *obj* holds is the user's to draw from: anything but a
    library's object or function."""
    if instance_of(obj, types.FunctionType):
        return is_user_code_file(obj.__code__.co_filename)
    if instance_of(obj, (types.MethodType, functools.partial, classmethod, staticmethod, dict, *_CONTAINERS)):
        return True
    if instance_of(obj, (types.SimpleNamespace, types.ModuleType)):
        return True
    if instance_of(obj, type):
        return _users_class(obj)
    return _users_class(type(obj))


def _users_generator(obj: Any) -> bool:
    """Whether *obj* is a generator the user's code made."""
    code = attribute_or(obj, "gi_code") or attribute_or(obj, "ag_code")
    return code is not None and is_user_code_file(code.co_filename)


def _reached(obj: Any, owned: bool, look: _Look, path: str) -> Iterable[tuple[str, Any]]:
    """What calling or reading *obj* can reach: `hidden_generators.children`,
    and a function's defaults, a ``partial``'s function and arguments, a
    builtin method's object, a class of the user's and its attributes, an
    instance's class, the items of a container a library makes (a deque, a
    pandas object column). All of an *owned* object's items, attributes and
    globals; a library's up to `_HELD_WIDTH`."""
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
        for i, arg in enumerate(obj.args):
            yield f"args[{i}]", arg
        for key, arg in list(obj.keywords.items()):
            yield f"keywords[{key!r}]", arg
        return
    if instance_of(obj, (types.BuiltinMethodType, types.MethodWrapperType)):
        owner = obj.__self__
        if not instance_of(owner, types.ModuleType):
            yield "__self__", owner
        return
    items = _items(obj, look, path if owned else None)
    if items is not None:
        yield from items
        if type(obj) is dict or type(obj) in _CONTAINERS or not instance_of(obj, (dict, *_CONTAINERS)):
            return
        # A subclass of the user's: its attributes too.
        try:
            attrs = object.__getattribute__(obj, "__dict__")
        except (AttributeError, TypeError):
            return
        if isinstance(attrs, dict):
            yield from ((f".{k}", v) for k, v in list(attrs.items()))
        return
    if owned:
        yield from _all_children(obj)
    else:
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


def _all_children(obj: Any) -> Iterable[tuple[str, Any]]:
    """`hidden_generators.children` of a user's object, with no width: every
    global a function of theirs names, every attribute of their instance,
    every name of their module."""
    from .statement.hidden_generators import children

    if instance_of(obj, types.ModuleType):
        if is_user_code_module(obj):
            yield from ((f".{k}", v) for k, v in list(obj.__dict__.items()) if not k.startswith("__"))
        return
    if instance_of(obj, types.FunctionType):
        code = obj.__code__
        for name, cell in zip(code.co_freevars, obj.__closure__ or (), strict=False):
            try:
                yield name, cell.cell_contents
            except ValueError:
                continue
        g = obj.__globals__
        for name in code.co_names:
            if name in g:
                yield name, g[name]
        return
    if instance_of(obj, types.MethodType):
        yield from children(obj)
        return
    try:
        attrs = object.__getattribute__(obj, "__dict__")
    except (AttributeError, TypeError):
        return
    if isinstance(attrs, dict):
        yield from ((f".{k}", v) for k, v in list(attrs.items()))


def _items(owner: Any, look: _Look | None = None, path: str | None = None) -> list[tuple[str, Any]] | None:
    """The items of a container that can hold something, read without
    running the user's code: of a list, tuple, set, deque or dict of any
    kind; a pandas object column or a numpy array of objects. ``None`` for
    anything else. With *path* (an object of the user's), and for a numpy or
    pandas object container wherever it is (where the user's data lives),
    all of them, counted against *look*'s items; past them the look stops
    there. A library's other containers, up to `_HELD_WIDTH`."""
    whole = (path is not None and look is not None) or _object_container(owner)
    width = max(look.items, 0) + 1 if whole and look is not None else sys.maxsize if whole else _HELD_WIDTH
    found = _items_up_to(owner, width)
    if found is None:
        return None
    found = [(label, value) for label, value in found if not _inert(type(value))]
    if whole and look is not None:
        look.items -= len(found)
        if look.items < 0:
            look.stop(f"{path or 'an object container'}, past {_HELD_ITEMS} items")
    return found


def _items_up_to(owner: Any, width: int) -> list[tuple[str, Any]] | None:
    if instance_of(owner, dict):
        return [(f"[{k!r}]", v) for k, v in itertools.islice(dict.items(owner), width)]
    if instance_of(owner, _CONTAINERS):
        return [(f"[{i}]", v) for i, v in enumerate(itertools.islice(owner, width))]
    module = _module_of(type(owner))
    if module.startswith("numpy") and attribute_or(attribute_or(owner, "dtype"), "kind") == "O":
        return _not_inert("[{}]", owner.flat, width)
    if module.split(".")[0] == "pandas":
        name = type(owner).__name__
        if name == "Series" and owner.dtype.kind == "O":
            return _not_inert(".iloc[{}]", owner.array, width)
        if name == "DataFrame":
            found = []
            for col in [c for c, dtype in owner.dtypes.items() if dtype.kind == "O"]:
                found.extend(_not_inert(f"[{col!r}].iloc[{{}}]", owner[col].array, width - len(found)))
                if len(found) >= width:
                    break
            return found
    return None


def _not_inert(label: str, values: Iterable[Any], width: int) -> list[tuple[str, Any]]:
    """The items of a numpy or pandas object container that can hold
    something, up to *width*, labelled by position: picked at C speed, so a
    column of a million strings and dates costs no Python step per item."""
    values = list(values)
    picked = itertools.compress(range(len(values)), map(operator.not_, map(_inert, map(type, values))))
    return [(label.format(i), values[i]) for i in itertools.islice(picked, max(width, 0))]


_USERS_CLASS: weakref.WeakKeyDictionary[type, bool] = weakref.WeakKeyDictionary()


def _users_class(cls: type) -> bool:
    """Whether *cls* was defined in the user's code: a function of its own
    was compiled from a notebook cell or a file of theirs, or, with none, its
    module is theirs (a notebook's ``__main__``). A library's class keeps its
    counters and iterators to itself, as `hidden_generators` treats them.
    Answered once per class: the look asks it of every object it passes."""
    try:
        return _USERS_CLASS[cls]
    except (KeyError, TypeError):
        pass
    answer = _decide_users_class(cls)
    try:
        _USERS_CLASS[cls] = answer
    except TypeError:
        pass
    return answer


def _decide_users_class(cls: type) -> bool:
    try:
        attrs = object.__getattribute__(cls, "__dict__")
    except (AttributeError, TypeError):
        return False
    for value in list(attrs.values())[:_HELD_WIDTH]:
        if instance_of(value, (types.FunctionType, classmethod, staticmethod)):
            code = attribute_or(_unwrapped(value), "__code__")
            if code is not None:
                return is_user_code_file(code.co_filename)
    name = _module_of(cls)
    if name == "builtins":
        # A class ``exec``'d without a module of its own; the interpreter's
        # are static types.
        return bool(cls.__flags__ & _HEAP_TYPE)
    module = sys.modules.get(name)
    return module is None or is_user_code_module(module)


def _unwrapped(value: Any) -> Any:
    return value.__func__ if instance_of(value, (classmethod, staticmethod)) else value


def _callees(func: ast.expr, user_ns: Mapping[str, Any], look: _Look | None = None) -> list[tuple[str, Any]]:
    """What the call ``func(...)`` runs, with what it runs on: the function a
    name, an item or an attribute holds; a method with its instance (or
    class); read statically, so nothing of the objects' runs."""
    if isinstance(func, ast.Attribute):
        found = []
        for path, owner in _spelled(func.value, user_ns, look):
            try:
                attr = inspect.getattr_static(owner, func.attr)
            except (AttributeError, TypeError):
                continue
            found.append((f"{path}.{func.attr}", _unwrapped(attr)))
            if instance_of(attr, (types.FunctionType, classmethod)):
                found.append((path, owner))
        return found
    return _spelled(func, user_ns, look)


def _spelled(node: ast.expr, user_ns: Mapping[str, Any], look: _Look | None = None) -> list[tuple[str, Any]]:
    """The paths and values of an attribute or item chain over a name
    (``m.g``, ``d['g']``, ``gens[i]``: an item a variable of a plain value
    picks is that item; one picked otherwise stands for every item), read
    from the objects' own dicts so nothing of theirs runs; empty when it
    cannot be read so."""
    if isinstance(node, ast.Name):
        return [(node.id, user_ns[node.id])] if node.id in user_ns else []
    if isinstance(node, ast.Attribute):
        found = []
        for path, owner in _spelled(node.value, user_ns, look):
            if node.attr in _INDEXERS and _module_of(type(owner)).split(".")[0] == "pandas":
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
        index = _plain_index(node.slice, user_ns, look.bound if look is not None else frozenset())
        for path, owner in _spelled(node.value, user_ns, look):
            if type(owner) in (dict, list, tuple) and index is not _NO_INDEX:
                try:
                    found.append((f"{path}[{index!r}]", owner[index]))
                except (KeyError, IndexError, TypeError):
                    pass
                continue
            # An item an expression picks, or one of a container whose lookup
            # runs its own code: any item it holds.
            items = _items(owner, look, path)
            if items is not None:
                found.extend((f"{path}{label}", item) for label, item in items)
        return found
    return []


_NO_INDEX = object()


def _plain_index(node: ast.expr, user_ns: Mapping[str, Any], bound: frozenset[str] = frozenset()) -> Any:
    """The index a constant, or a name holding a plain value (``i``, ``k``),
    gives, which a lookup in a dict, list or tuple uses without running the
    user's code; `_NO_INDEX` for any other. A name in *bound* is the
    statement's own (``[next(gens[j]) for j in [1]]``): the notebook's ``j``
    says nothing of it."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id not in bound and node.id in user_ns and type(user_ns[node.id]) in _LEAVES:
        return user_ns[node.id]
    return _NO_INDEX


def _bound_inside(tree: ast.Module) -> frozenset[str]:
    """The names *tree* binds anywhere in it: its assignments and loops,
    and the variables of its comprehensions, lambdas and functions."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return frozenset(names)


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
