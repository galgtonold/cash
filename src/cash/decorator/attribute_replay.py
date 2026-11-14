"""Which attribute changes of a missed call a hit can replay.

``est.fit(X)`` setting ``self.coef_`` changed its argument in place, which
a hit would not do, so it ran every time. When the change is only a
binding, rebinding or deletion of top-level attributes of an instance of a
plain class (`rewrite_in_place.attribute_refusal`), and the key holds that
instance by all it holds, the result is stored with the attributes the call
left (`rewrite_in_place.AttrState`) and a hit binds them on the instance
handed in.

`AttrVerdict.of` decides, after the body, whether it can:

* the arguments, with each recorded instance's attributes put back as they
  were (`rewrite_in_place.put_back`), hash as the key hashed them: nothing
  else changed, in place or anywhere in them (``self.history.append(x)``,
  ``X[:] = 0``, a slot);
* each value bound is either an immutable atom, one of the call's
  arguments or the instance itself, a module global (bound again by
  identity on a hit, never a copy), or a fresh object: nothing outside the
  instance, the other bound attributes and the result references it
  (counted by ``sys.getrefcount``), it is none of the values the instance
  held before, and the share check of a result finds nothing it holds that
  the caller or a module also holds.

Anything else keeps the refusal, with the reason in the warning.
"""

from __future__ import annotations

import collections
import enum
import gc
import importlib
import sys
import types
from typing import TYPE_CHECKING, Any, NamedTuple

from ..rewrite_in_place import AttrChange, AttrRecording, AttrState, Relink, put_back, seal
from ..value_types import IMMUTABLE_LEAF_TYPES

if TYPE_CHECKING:
    from collections.abc import Callable

    from .call_state import BodyRun

__all__ = ["AttrVerdict", "recordings"]

#: Bound by value: copies are the same value, whatever their identity.
_ATOMS = IMMUTABLE_LEAF_TYPES

#: Found again by name, so bound again as that very object.
_NAMED = (type, types.FunctionType, types.BuiltinFunctionType)


def recordings(args_before: dict | None) -> dict[str, AttrRecording]:
    """``{parameter: recording}`` of the instances whose attributes the
    body was recorded changing (`CallRunner.arguments_before`)."""
    return {name: entry[1] for name, entry in (args_before or {}).items() if type(entry[1]) is AttrRecording}


class AttrVerdict(NamedTuple):
    """What a missed call did to the attributes of the instances it was
    handed, and whether a hit can replay it."""

    #: ``{parameter: change}`` of each recorded instance the call changed.
    changes: dict[str, AttrChange]
    #: ``{parameter: state}`` a hit replays, or None when it cannot.
    states: dict[str, AttrState] | None
    #: Why it cannot, or None.
    why: str | None
    #: Whether the arguments, with those attributes put back, hash as the
    #: key hashed them: the change is that and nothing else.
    explained: bool
    #: What the result shares with the arguments as they were before the
    #: call, as `PurityChecks.warn_shared_result` says it (None: nothing),
    #: when *result_checked*.
    result_shared: str | None = None
    result_checked: bool = False

    @property
    def names(self) -> list[str]:
        return list(self.changes)

    @classmethod
    def of(
        cls,
        run: BodyRun,
        func: Callable,
        func_name: str,
        named: list[tuple[str, Any]],
        keyed_hash: str | None,
        unchanged: Callable[[], bool],
        shared_outside: Callable[[Any], str | None],
        held_outside: Callable[[Any], str | None],
        warn_shared_result: Callable[[Any], str | None],
    ) -> AttrVerdict | None:
        """The verdict on the attribute changes of a call whose body ran
        (*run*), or None when it changed none. *named* is ``(parameter,
        value)`` of each argument as the key binds them and of the instance
        the callable is bound to, *keyed_hash* the arguments' hash before the
        body, *unchanged* says whether they (and what the callable binds)
        hash as before it;
        *shared_outside* and *warn_shared_result* are the share checks of a
        bound value and of the result, run while the attributes are put back."""
        recorded = recordings(run.args_before)
        if not recorded:
            return None
        changes = {name: rec.change for name, rec in recorded.items() if rec.change is not None}
        refused = [rec.why for rec in recorded.values() if rec.why is not None]
        if not changes and not refused:
            return None
        if refused:
            return cls(changes, None, refused[0], False)
        if keyed_hash is None:
            return cls(changes, None, "its arguments were not hashed before the call", False)
        for param, change in changes.items():
            holder = held_outside(change.obj)
            if holder is not None:
                return cls(
                    changes, None, f"{holder} also holds {param}, which the call may have reached that way", False
                )
        result = run.res
        classified = _classify(changes, func, named)
        if type(classified) is str:
            return cls(changes, None, classified, False)
        values, fresh = classified
        pinned: dict[int, tuple[Any, Relink]] = {}
        for name, value in named:
            if type(value) not in _ATOMS:
                pinned.setdefault(id(value), (value, Relink("argument", (name,))))
        hidden: list[tuple[Any, str]] = []
        why = _held_elsewhere(
            fresh, result, {id(value) for _, value in named} | {id(c.obj) for c in changes.values()}, pinned, hidden
        )
        objects = {id(change.obj) for change in changes.values()}
        with put_back(changes.values()):
            try:
                explained = unchanged()
            except Exception:  # noqa: BLE001 - not hashed again: nothing says what else changed
                explained = False
            if not explained:
                return cls(changes, None, "it also changed something else in its arguments in place", False)
            if why is None:
                why = _shared(fresh, shared_outside, hidden)
            result_checked = id(result) not in objects
            result_shared = warn_shared_result(result) if result_checked else None
        if why is not None:
            return cls(changes, None, why, True, result_shared, result_checked)
        states = {
            param: AttrState(type(change.obj), values[param], change.deleted, change.order)
            for param, change in changes.items()
        }
        if pinned:
            for state in states.values():
                state.pinned = pinned
        return cls(changes, states, None, True, result_shared, result_checked)


def _classify(
    changes: dict[str, AttrChange], func: Callable, named: list[tuple[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[int, tuple[Any, int, str]]] | str:
    """``(values, fresh)``: per parameter, what each bound attribute is
    stored as (the value, or a `Relink`), and ``{id: (value, number of
    attributes bound to it, label)}`` of the values stored that must be the
    call's own new objects. Or why one cannot be replayed."""
    arguments = {}
    for name, value in named:
        arguments.setdefault(id(value), (value, name))
    held_before = {id(value) for change in changes.values() for value in change.before.values()}
    fresh: dict[int, tuple[Any, int, str]] = {}
    values: dict[str, dict[str, Any]] = {}
    for param, change in changes.items():
        mine: dict[str, Any] = {}
        for key, value in change.bound.items():
            label = f"{param}.{key}"
            if type(value) in _ATOMS:
                mine[key] = value
                continue
            found = arguments.get(id(value))
            if found is not None and found[0] is value:
                mine[key] = Relink("argument", (found[1],))
                continue
            ref = _global_ref(value, func, change.obj)
            if ref is not None:
                mine[key] = ref
                continue
            if id(value) in held_before or isinstance(value, enum.Enum):
                return f"it bound {label} to an object that was there before the call"
            seen = fresh.get(id(value))
            fresh[id(value)] = (value, (seen[1] if seen else 0) + 1, label)
            mine[key] = value
        values[param] = mine
    return values, fresh


def _global_ref(value: Any, func: Callable, obj: Any) -> Relink | None:
    """A `Relink` to the module global *value* is, or None: a class or
    function found by its module and qualified name, or a name bound to it
    in the function's module or in the module of *obj*'s class."""
    if isinstance(value, _NAMED):
        module, qualname = getattr(value, "__module__", None), getattr(value, "__qualname__", None)
        if isinstance(module, str) and isinstance(qualname, str) and _by_path(module, qualname) is value:
            return Relink("global", (module, qualname))
    for namespace in (getattr(func, "__globals__", None), _module_dict(type(obj).__module__)):
        if not isinstance(namespace, dict):
            continue
        module = namespace.get("__name__")
        if not isinstance(module, str) or _module_dict(module) is not namespace:
            continue
        for name, held in list(namespace.items()):
            if held is value:
                return Relink("global", (module, name))
    return None


def _module_dict(module: Any) -> dict | None:
    held = sys.modules.get(module) if isinstance(module, str) else None
    namespace = getattr(held, "__dict__", None)
    return namespace if isinstance(namespace, dict) else None


def _by_path(module: str, qualname: str) -> Any:
    """What ``module.qualname`` names, read from the namespaces alone (no
    attribute lookup runs code), or None."""
    held: Any = sys.modules.get(module)
    for part in qualname.split("."):
        namespace = getattr(held, "__dict__", None)
        if not isinstance(namespace, (dict, types.MappingProxyType)) or part not in namespace:
            return None
        held = namespace[part]
    return held


def find(relink: Relink, named: dict[str, Any]) -> Any:
    """The object *relink* names on a hit: the argument bound to its
    parameter (*named*), or the module global. Raises `LookupError`."""
    if relink.kind == "argument":
        return named[relink.where[0]]
    module, name = relink.where
    held = _by_path(module, name)
    if held is None:
        raise LookupError(f"{module}.{name}")
    return held


#: Objects a walk of a bound value visits at most (`_held_elsewhere`):
#: past it, cash cannot say nothing else holds part of it.
WALK_LIMIT = 200_000

_ATOM_SET = frozenset(_ATOMS)


def _by_identity_anyway(value: Any) -> bool:
    """Whether a copy of *value* is as good as *value*: an atom, a tuple or
    frozenset of atoms (a constant the code holds too), or what pickling
    finds again by name (a class, a function, a module, an enum member)."""
    kind = type(value)
    if kind in _ATOM_SET:
        return True
    if kind is tuple or kind is frozenset:
        return set(map(type, value)) <= _ATOM_SET
    return isinstance(value, (*_NAMED, types.ModuleType, enum.Enum))


def _children(value: Any) -> Any:
    """What *value* holds, as the garbage collector sees it (a dict's keys
    and values, a container's items, an object's ``__dict__`` or slots, what
    a C object visits), and an object array's members, which it does not
    visit, and the arrays a scikit-learn tree or metric holds through Cython
    memoryviews, which it does not visit either (`_sklearn_buffers`).
    Nothing for an object that holds no other (a number, an array of
    numbers): a C object that holds references and does not say so is not
    seen into."""
    kind = type(value)
    if kind.__name__ == "ndarray" and getattr(getattr(value, "dtype", None), "kind", None) == "O":
        return [*gc.get_referents(value), *value.flat]
    module = getattr(kind, "__module__", "")
    if module == _SKLEARN_METRICS or module in _SKLEARN_TREES:
        return [*gc.get_referents(value), *_sklearn_buffers(value)]
    return gc.get_referents(value)


def _sklearn_buffers(value: Any) -> list:
    """The arrays a scikit-learn tree or distance metric holds through Cython
    memoryviews, each as many times as one memoryview holds it
    (`_memoryview_holds`), so the reference count of each adds up; and a
    ``PyFuncDistance``'s function and keyword arguments. Its
    ``__getstate__`` hands back a tree's arrays themselves, and a metric's
    as new arrays over its memoryviews, whose ``base`` is the array."""
    try:
        state = value.__getstate__()
    except Exception:  # noqa: BLE001 - not read: seen as the collector sees it
        return []
    if type(state) is not tuple:
        return []
    found = []
    if type(value).__module__ == _SKLEARN_METRICS:
        for view in state[1:3]:
            held = getattr(getattr(getattr(view, "base", None), "obj", None), "base", None)
            if held is not None:
                found.append(held)
        extra = list(state[3:5])
    else:
        found = [item for item in (*state[:4], state[12]) if type(item).__name__ == "ndarray"]
        extra = []
    holds = _memoryview_holds()
    return [held for held in found for _ in range(holds)] + extra


_MEMORYVIEW_HOLDS: list[int] = []


def _memoryview_holds() -> int:
    """How many references a Cython memoryview of scikit-learn's keeps to
    the array it views (its ``obj`` and its buffer's), measured once on a
    small tree."""
    if not _MEMORYVIEW_HOLDS:
        np = sys.modules["numpy"]
        # Imported only when a tree or a metric is met, so scikit-learn is.
        kd_tree = importlib.import_module("sklearn.neighbors._kd_tree")

        probe = np.zeros((2, 1))
        before = sys.getrefcount(probe)
        tree = kd_tree.KDTree64(probe, leaf_size=1)
        _MEMORYVIEW_HOLDS.append(sys.getrefcount(probe) - before)
        del tree
    return _MEMORYVIEW_HOLDS[0]


def _walk(
    fresh: dict[int, tuple[Any, int, str]], arguments: set[int]
) -> tuple[dict[int, int], list, dict[int, str], dict[int, str]] | str:
    """``(inner, nodes, labels, met)``: how many times the objects the values
    in *fresh* hold are held among them (by id), those objects (each once),
    the attribute each was found under, and ``{id: attribute}`` of the
    call's arguments (*arguments*) met inside them, not walked into. Or why
    it cannot be told. A function of its own, so no loop variable of it
    holds an object when the references are counted."""
    inner: dict[int, int] = {}
    nodes: list = []
    labels: dict[int, str] = {}
    met: dict[int, str] = {}
    stack = [(value, label) for value, _holders, label in fresh.values()]
    visited = set(fresh)
    while stack:
        value, label = stack.pop()
        for child in _children(value):
            if _by_identity_anyway(child):
                continue
            key = id(child)
            if key in arguments:
                met.setdefault(key, label)
                continue
            inner[key] = inner.get(key, 0) + 1
            if key not in visited:
                visited.add(key)
                if len(visited) > WALK_LIMIT:
                    return f"the object it bound to {label} is too large to tell what else holds it"
                nodes.append(child)
                labels[key] = label
                stack.append((child, label))
    return inner, nodes, labels, met


#: Containers whose pickle holds all they hold: the builtin ones, and the
#: ordered, defaulting and double-ended ones of `collections` (exactly
#: these types; a subclass of one may pickle less): an estimator's
#: ``sampling_strategy_`` is an ``OrderedDict``.
_EXACT_CONTAINERS = (
    dict,
    list,
    tuple,
    set,
    frozenset,
    collections.OrderedDict,
    collections.defaultdict,
    collections.deque,
)

#: Modules whose objects cash's store pickles exactly (their own reducers
#: carry all they hold).
_EXACT_PICKLERS = ("numpy", "pandas", "scipy")

#: scikit-learn's neighbour trees by their exact module, and the class name
#: each of its trees starts with: Cython classes whose own ``__reduce__``
#: carries all they hold (the data, the index and node arrays, the sample
#: weights, the counters and the distance metric). A fitted
#: ``NearestNeighbors`` or a resampler's ``nn_k_`` holds one.
_SKLEARN_TREES = {"sklearn.neighbors._kd_tree": "KDTree", "sklearn.neighbors._ball_tree": "BallTree"}
#: The module of scikit-learn's distance metrics, which a tree holds: their
#: ``__reduce__`` carries ``p``, the weight vector and matrix, and a
#: ``PyFuncDistance``'s function and keyword arguments.
_SKLEARN_METRICS = "sklearn.metrics._dist_metrics"


def _sklearn_refusal(value: Any, kind: type) -> str | None | bool:
    """False when *kind* is none of scikit-learn's trees or metrics; else
    None, or why a stored copy of *value* would differ. A float32 tree on the
    Euclidean metric comes back from its pickle without its Euclidean flag
    (its ``__setstate__`` looks for the float64 metric's name): its queries
    give the same results, through the metric instead of the inline path,
    but it is not the object the call left."""
    module = getattr(kind, "__module__", "") or ""
    if module == _SKLEARN_METRICS:
        return None if any(base.__name__ == "DistanceMetric" for base in kind.__mro__) else False
    prefix = _SKLEARN_TREES.get(module)
    if prefix is None or not kind.__name__.startswith(prefix):
        return False
    if kind.__name__.endswith("32"):
        try:
            metric = value.__getstate__()[11]
        except Exception:  # noqa: BLE001 - not read: not trusted
            return f"is a {kind.__name__}, whose state could not be read"
        if type(metric).__name__ == "EuclideanDistance32":
            return (
                f"is a {kind.__name__} on the Euclidean metric, which its pickle brings back without its Euclidean flag"
            )
    return None


def _pickling_refusal(value: Any) -> str | None:
    """Why a stored copy of *value* might not hold all it holds, or None.

    A builtin container, an ``OrderedDict``, ``defaultdict`` or ``deque``
    or a NumPy, pandas or SciPy object pickles all it holds, as do scikit-learn's neighbour trees and distance metrics
    (`_sklearn_refusal`). Any other object must pickle as a plain instance
    does -- no ``__reduce__`` or ``__reduce_ex__`` of its own -- and a
    ``__getstate__`` of its own must hand back every attribute as it is
    (scikit-learn's adds its version and keeps the rest); one that leaves
    state out would come back from a hit without it."""
    kind = type(value)
    if kind in _EXACT_CONTAINERS or kind in _ATOM_SET:
        return None
    if (getattr(kind, "__module__", "") or "").split(".")[0] in _EXACT_PICKLERS:
        return None
    sklearn = _sklearn_refusal(value, kind)
    if sklearn is not False:
        return sklearn
    if kind.__reduce_ex__ is not object.__reduce_ex__ or kind.__reduce__ is not object.__reduce__:
        return f"is a {kind.__name__}, which pickles through its own __reduce__"
    getstate = getattr(kind, "__getstate__", None)
    if getstate is None or getstate is getattr(object, "__getstate__", None):
        return None
    try:
        state = value.__getstate__()
    except Exception:  # noqa: BLE001 - not read: not trusted
        return f"is a {kind.__name__}, whose __getstate__ could not be read"
    attrs = getattr(value, "__dict__", None)
    if type(state) is not dict or type(attrs) is not dict:
        return f"is a {kind.__name__}, whose __getstate__ does not hand back its attributes"
    if any(state.get(key, _MISSING) is not held for key, held in attrs.items()):
        return f"is a {kind.__name__}, whose __getstate__ leaves out or changes some of its attributes"
    return None


_MISSING = object()


def _not_pickled_whole(fresh: dict[int, tuple[Any, int, str]], nodes: list, labels: dict[int, str]) -> str | None:
    """`_pickling_refusal` of each value bound and each object inside one.
    A function of its own, so no loop variable of it holds an object when
    `_held_elsewhere` counts the references."""
    for value, _holders, label in fresh.values():
        why = _pickling_refusal(value)
        if why is not None:
            return f"the object it bound to {label} {why}"
    for node in nodes:
        why = _pickling_refusal(node)
        if why is not None:
            return f"an object inside what it bound to {labels[id(node)]} {why}"
    return None


def _held_elsewhere(
    fresh: dict[int, tuple[Any, int, str]],
    result: Any,
    arguments: set[int],
    pinned: dict[int, tuple[Any, Relink]],
    hidden: list[tuple[Any, str]],
) -> str | None:
    """Why a value in *fresh* (`_classify`), or an object inside one, is not
    the call's own new object, or None.

    Every object a walk of the values reaches (`_children`) must be held by
    nothing but what the walk saw hold it: for a value bound, the
    instance's ``__dict__`` and the change's records of it, and the result
    (once if the tuple, list or dict it is holds it; a result that is the
    value itself is left to the result's own share check). One held by more
    is held by something outside -- a module, a class, a library's cache --
    which a hit's copy would no longer be.

    An argument inside a value (a tree built over the array handed in) is
    pickled as a reference a hit resolves to the argument it is handed
    (`rewrite_in_place.seal`): *pinned*, handed in with each argument's
    `Relink` by id, is left with those met. One the value's pickle does not
    reach as itself (a reducer that copies it), or the instance itself when
    it was not handed in as an argument, is refused: a hit would bind a
    copy of it. *hidden* is filled with ``(array, attribute)`` of each
    array a scikit-learn tree or metric holds out of the collector's sight
    (`_sklearn_buffers`), for `_shared`."""
    candidates = dict(pinned)
    pinned.clear()
    if not fresh:
        return None
    walked = _walk(fresh, arguments)
    if type(walked) is str:
        return walked
    inner, nodes, labels, met = walked
    why = _not_pickled_whole(fresh, nodes, labels)
    if why is not None:
        return why
    if met:
        why = _pin(fresh, met, candidates, pinned)
        if why is not None:
            return why
    items: list = []
    if type(result) in (tuple, list):
        items = list(result)
    elif type(result) is dict:
        items = list(result.values())
    # Measured on a probe held as each object is -- by one container, by a
    # record like `fresh`'s and by this loop, or by `nodes` -- since what
    # ``sys.getrefcount`` counts besides the holders differs across
    # Python versions.
    probe_holder = [object()]
    nested_base = max(map(sys.getrefcount, [probe_holder[0]]))  # the list read as `nodes` is
    probe = {0: (probe_holder[0], 1, "")}
    base = 0
    for value, _holders, _label in probe.values():
        base = max(map(sys.getrefcount, [value]))
    del probe, probe_holder
    for value, holders, label in fresh.values():
        if value is result:
            continue
        # Each attribute bound to it: the instance's dict, the change's
        # record of it and the state's (`_classify`), against the probe's
        # one container; each item of the result: the result and `items`.
        expected = base + 3 * holders - 1 + 2 * sum(1 for item in items if item is value) + inner.get(id(value), 0)
        if max(map(sys.getrefcount, [value])) > expected:
            return f"something outside the call also holds the object it bound to {label}"
    if nodes:
        counts = list(map(sys.getrefcount, nodes))
        for node, count in zip(nodes, counts):
            key = id(node)
            # The probe's one container against the walk's count; a node
            # the result also holds is held once more by it and `items`.
            expected = nested_base - 1 + inner[key] + 2 * sum(1 for item in items if item is node)
            if node is result:
                continue
            if count > expected:
                return f"something outside the call also holds an object inside what it bound to {labels[key]}"
    # Counted: now the arrays the scikit-learn objects hold out of the
    # collector's sight, for `_shared`.
    for value, label in [(value, label) for value, _holders, label in fresh.values()] + [
        (node, labels[id(node)]) for node in nodes
    ]:
        module = getattr(type(value), "__module__", "")
        if module == _SKLEARN_METRICS or module in _SKLEARN_TREES:
            hidden.extend(
                (held, label)
                for held in _sklearn_buffers(value)
                if type(held).__name__ == "ndarray" and id(held) not in arguments
            )
    return None


def _pin(
    fresh: dict[int, tuple[Any, int, str]],
    met: dict[int, str],
    candidates: dict[int, tuple[Any, Relink]],
    pinned: dict[int, tuple[Any, Relink]],
) -> str | None:
    """Put in *pinned* the arguments the values hold (*met*), out of every
    argument the call was handed (*candidates*, by id), or say why one
    cannot be pickled as a reference."""
    for key, label in met.items():
        if key not in candidates:
            return f"the object it bound to {label} holds the instance it was bound on"
        pinned[key] = candidates[key]
    try:
        _sealed, reached = seal([value for value, _holders, _label in fresh.values()], pinned)
    except Exception:  # noqa: BLE001 - not pickled: nothing says a hit could rebuild it
        pinned.clear()
        return "the objects it bound could not be pickled"
    for key, label in met.items():
        if key not in reached:
            pinned.clear()
            return f"the object it bound to {label} holds one of the call's arguments, which its pickle copies"
    return None


def _shared(
    fresh: dict[int, tuple[Any, int, str]],
    shared_outside: Callable[[Any], str | None],
    hidden: list[tuple[Any, str]] = (),  # type: ignore[assignment]
) -> str | None:
    """Why a value in *fresh*, or an array a scikit-learn tree in one holds
    (*hidden*: a tree over ``X[10:]`` views the argument), holds an object
    the caller or a module also holds (`PurityChecks.shared_outside`), or
    None. An argument itself, which the entry holds as a reference
    (`_pin`), is not among them. Run while the attributes are put back, so
    the instance is as the caller handed it."""
    for value, _holders, label in fresh.values():
        why = shared_outside(value)
        if why is not None:
            return f"the object it bound to {label} {why}"
    for value, label in hidden:
        why = shared_outside(value)
        if why is not None:
            return f"an array inside the object it bound to {label} {why}"
    return None
