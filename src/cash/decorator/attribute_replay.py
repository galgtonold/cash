"""Which attribute changes of a missed call a hit can replay.

``est.fit(X)`` setting ``self.coef_`` changed its argument in place, which
a hit would not do, so it ran every time. When the change is only a
binding, rebinding or deletion of top-level attributes of an instance of a
plain class (`rewrite_in_place.attribute_refusal`), and the key holds that
instance by all it holds, the result is stored with the attributes the call
left (`rewrite_in_place.AttrState`) and a hit binds them on the instance
handed in. The same holds one level down, and no deeper, for an instance of
a plain class that such an argument held as a top-level attribute before
the call and still holds there after it (NiMARE's
``self.kernel_transformer`` gaining ``memory_``): it is recorded beside the
argument (`runtime._record_held`, `rewrite_in_place.HeldBefore`), judged by
the same rules, and a hit binds its attributes on the instance the
argument it is handed holds there.

`AttrVerdict.of` decides, after the body, whether it can:

* the arguments, with each recorded instance's attributes put back as they
  were (`rewrite_in_place.put_back`), hash as the key hashed them: nothing
  else changed, in place or anywhere in them (``self.history.append(x)``,
  ``X[:] = 0``, a slot);
* each value bound is either an immutable atom, one of the call's
  arguments or the instance itself, a module global, the value of another
  attribute of a recorded instance that held it before the call and still
  does (``img._data_cache = img._dataobj``; held under several such names,
  bound while they all hold one object on the hit; all bound again by
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

from .._type_memo import class_dict, class_module, class_mro, class_name, mro_of
from .._user_attrs import plain_getattr
from ..kept_state import joblib_memory_state
from ..rewrite_in_place import (
    ABSENT,
    AttrChange,
    AttrRecording,
    AttrState,
    HeldBefore,
    Relink,
    held_changes,
    put_back,
    seal,
)
from ..value_types import IMMUTABLE_LEAF_TYPES

if TYPE_CHECKING:
    from collections.abc import Callable

    from .call_state import BodyRun

__all__ = ["ALSO_CHANGED", "AttrVerdict", "Unresolved", "also_changed", "held_paths", "recordings"]

#: Bound by value: copies are the same value, whatever their identity.
_ATOMS = IMMUTABLE_LEAF_TYPES

#: Found again by name, so bound again as that very object.
_NAMED = (type, types.FunctionType, types.BuiltinFunctionType)


def recordings(args_before: dict | None) -> dict[str, AttrRecording]:
    """``{parameter: recording}`` of the instances whose attributes the
    body was recorded changing (`CallRunner.arguments_before`)."""
    return {name: entry[1] for name, entry in (args_before or {}).items() if type(entry[1]) is AttrRecording}


def held_paths(args_before: dict | None) -> dict[str, tuple[str, str]]:
    """``{recording: (parent, attribute)}`` of the instances recorded one
    level down (`rewrite_in_place.HeldBefore`): what the recorded argument
    *parent* held as its top-level *attribute* before the call."""
    return {
        name: (entry[2].param, entry[2].name)
        for name, entry in (args_before or {}).items()
        if type(entry[1]) is AttrRecording and type(entry[2]) is HeldBefore
    }


#: The refusal of a call that changed more in its arguments than the
#: attributes it bound; `also_changed` and `CallRunner.attribute_changes`
#: add what that was, where they can tell.
ALSO_CHANGED = "it also changed something else in its arguments in place"


def also_changed(recorded: dict[str, AttrRecording], named: list[str] = ()) -> str:
    """`ALSO_CHANGED`, naming what else changed: *named* (the calls of the
    body that change an argument in place, as reading the source found
    them: ``self.history.append``), or else each container an instance held
    as an attribute and still holds, which the call grew or shrank (``the
    list self.history``)."""
    found = list(named)
    if not found:
        for param, rec in recorded.items():
            found += [f"an in-place change of the {kind} {param}.{key} held" for key, kind in held_changes(rec)]
    if not found:
        return (
            f"{ALSO_CHANGED}, inside a value one of its attributes holds (a list, a dict, an array), "
            "which a hit cannot replay by binding attributes"
        )
    return f"{ALSO_CHANGED}: {'; '.join(dict.fromkeys(found))}"


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
    #: ``{recording: parameter}`` of each change recorded on an instance an
    #: argument held one level down (``self.kernel_transformer``), naming
    #: the argument it is part of.
    roots: dict[str, str] | None = None

    @property
    def names(self) -> list[str]:
        """The arguments the call changed: a change one level down is a
        change of the argument that holds it."""
        roots = self.roots or {}
        return list(dict.fromkeys(roots.get(name, name) for name in self.changes))

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
        held = held_paths(run.args_before)
        changes = {name: rec.change for name, rec in recorded.items() if rec.change is not None}
        refused = [rec.why for rec in recorded.values() if rec.why is not None]
        if not changes and not refused:
            return None
        roots = {name: _root(name, held) for name in changes if name in held} or None

        def verdict(
            states: dict[str, AttrState] | None,
            why: str | None,
            explained: bool,
            shared: str | None = None,
            checked: bool = False,
        ) -> AttrVerdict:
            return cls(changes, states, why, explained, shared, checked, roots)

        if refused:
            return verdict(None, refused[0], False)
        if keyed_hash is None:
            return verdict(None, "its arguments were not hashed before the call", False)
        for name in changes:
            if name in held:
                parent, attribute = held[name]
                if recorded[parent].state.get(attribute, ABSENT) is not changes[name].obj:
                    return verdict(
                        None,
                        f"it changed the attributes of {name} and also rebound or deleted {name}, "
                        "so a hit cannot find that object where it changed it",
                        False,
                    )
        for param, change in changes.items():
            holder = held_outside(change.obj)
            if holder is not None:
                return verdict(None, f"{holder} also holds {param}, which the call may have reached that way", False)
        result = run.res
        paths = _paths(changes, recorded, held, named)
        classified = _classify(changes, func, named, paths)
        if type(classified) is str:
            return verdict(None, classified, False)
        values, fresh = classified
        pinned: dict[int, tuple[Any, Relink]] = {}
        for name, value in named:
            if type(value) not in _ATOMS:
                pinned.setdefault(id(value), (value, Relink("argument", (name,))))
        for name, change in changes.items():
            # An instance an argument holds, which a value bound may hold
            # too: found again where the argument holds it.
            path = paths.get(name)
            if path is not None and len(path) > 1:
                pinned.setdefault(id(change.obj), (change.obj, Relink("attribute", path)))
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
                return verdict(None, also_changed(recorded), False)
            if why is None:
                why = _shared(fresh, shared_outside, hidden)
            result_checked = id(result) not in objects
            result_shared = warn_shared_result(result) if result_checked else None
        if why is not None:
            return verdict(None, why, True, result_shared, result_checked)
        states = {
            param: AttrState(type(change.obj), values[param], change.deleted, change.order)
            for param, change in changes.items()
        }
        if pinned:
            for state in states.values():
                state.pinned = pinned
        return verdict(states, None, True, result_shared, result_checked)

    def fresh_attributes(self) -> dict[str, set[str]]:
        """``{parameter: attributes}`` the call bound on an argument to an
        object that was not there before it (the call's own new dict, say):
        a write into one of them changed nothing the caller had."""
        held_before = {id(value) for change in self.changes.values() for value in change.before.values()}
        found: dict[str, set[str]] = {}
        for name, change in self.changes.items():
            if self.roots and name in self.roots:
                continue
            found[name] = {
                key for key, value in change.bound.items() if type(value) not in _ATOMS and id(value) not in held_before
            }
        return found


def _root(name: str, held: dict[str, tuple[str, str]]) -> str:
    """The argument the recording *name* is part of."""
    while name in held:
        name = held[name][0]
    return name


def _paths(
    changes: dict[str, AttrChange],
    recorded: dict[str, AttrRecording],
    held: dict[str, tuple[str, str]],
    named: list[tuple[str, Any]],
) -> dict[str, tuple[str, ...]]:
    """``{recording: path}`` of each changed instance a hit finds again
    among its arguments: ``(parameter,)`` for an argument (or the instance
    the callable is bound to) by identity, ``(parameter, attribute)`` for
    what one held as its attribute (`held_paths`). Named as *named* names
    the arguments, which a hit's `find` reads."""
    arguments: dict[int, tuple[Any, str]] = {}
    for name, value in named:
        arguments.setdefault(id(value), (value, name))

    def of(obj: Any) -> tuple[str, ...] | None:
        found = arguments.get(id(obj))
        return (found[1],) if found is not None and found[0] is obj else None

    paths: dict[str, tuple[str, ...]] = {}
    for name, change in changes.items():
        if name in held:
            parent, attribute = held[name]
            base = of(recorded[parent].obj)
            if base is not None:
                paths[name] = (*base, attribute)
        else:
            path = of(change.obj)
            if path is not None:
                paths[name] = path
    return paths


def _classify(
    changes: dict[str, AttrChange], func: Callable, named: list[tuple[str, Any]], paths: dict[str, tuple[str, ...]]
) -> tuple[dict[str, dict[str, Any]], dict[int, tuple[Any, int, str]]] | str:
    """``(values, fresh)``: per parameter, what each bound attribute is
    stored as (the value, or a `Relink`), and ``{id: (value, number of
    attributes bound to it, label)}`` of the values stored that must be the
    call's own new objects. Or why one cannot be replayed."""
    arguments = {}
    for name, value in named:
        arguments.setdefault(id(value), (value, name))
    held_before = {id(value) for change in changes.values() for value in change.before.values()}
    kept = _kept_attributes(changes, paths)
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
            paths = kept.get(id(value), ())
            if len(paths) == 1 and not isinstance(value, enum.Enum):
                mine[key] = Relink("attribute", paths[0][0])
                continue
            if len(paths) > 1 and not isinstance(value, enum.Enum):
                # ``self.memory_ = self.kernel.memory`` where ``self.memory``
                # is that object too: a hit binds it when every one of them
                # still holds one object (`find`).
                mine[key] = Relink("attributes", tuple(path for path, _label in paths))
                continue
            if id(value) in held_before or isinstance(value, enum.Enum):
                return f"it bound {label} to an object that was there before the call"
            seen = fresh.get(id(value))
            fresh[id(value)] = (value, (seen[1] if seen else 0) + 1, label)
            mine[key] = value
        values[param] = mine
    return values, fresh


def _kept_attributes(
    changes: dict[str, AttrChange], paths: dict[str, tuple[str, ...]]
) -> dict[int, list[tuple[tuple[str, ...], str]]]:
    """``{id: [(path, label), ...]}`` of each object a recorded instance
    held as its top-level attribute *name* before the call and still holds
    there after it, the call having neither rebound nor deleted that name;
    *path* is where a hit finds it again (`find`): the instance's own path
    (*paths*, `_paths`) and *name*. nibabel's
    ``img._data_cache = np.asanyarray(img._dataobj)`` aliases the array the
    image already held, and a hit leaves that alias, not a copy."""
    kept: dict[int, list[tuple[tuple[str, ...], str]]] = {}
    seen: set[int] = set()
    for recorded, change in changes.items():
        path = paths.get(recorded)
        if path is None or id(change.obj) in seen:
            continue
        seen.add(id(change.obj))
        for name, value in change.before.items():
            if type(value) in _ATOMS or name in change.bound or change.state.get(name, _ABSENT) is not value:
                continue
            kept.setdefault(id(value), []).append(((*path, name), f"{recorded}.{name}"))
    return kept


_ABSENT = object()


class Unresolved(LookupError):
    """A `Relink` to an attribute of an argument that the argument a hit is
    handed no longer holds as a plain attribute; says which and why."""


def _global_ref(value: Any, func: Callable, obj: Any) -> Relink | None:
    """A `Relink` to the module global *value* is, or None: a class or
    function found by its module and qualified name, or a name bound to it
    in the function's module or in the module of *obj*'s class."""
    if isinstance(value, _NAMED):
        module, qualname = plain_getattr(value, "__module__", None), plain_getattr(value, "__qualname__", None)
        if isinstance(module, str) and isinstance(qualname, str) and _by_path(module, qualname) is value:
            return Relink("global", (module, qualname))
    for namespace in (plain_getattr(func, "__globals__", None), _module_dict(class_module(type(obj)))):
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
    namespace = plain_getattr(held, "__dict__", None)
    return namespace if isinstance(namespace, dict) else None


def _by_path(module: str, qualname: str) -> Any:
    """What ``module.qualname`` names, read from the namespaces alone (no
    attribute lookup runs code), or None."""
    held: Any = sys.modules.get(module)
    for part in qualname.split("."):
        namespace = plain_getattr(held, "__dict__", None)
        if not isinstance(namespace, (dict, types.MappingProxyType)) or part not in namespace:
            return None
        held = namespace[part]
    return held


def find(relink: Relink, named: dict[str, Any]) -> Any:
    """The object *relink* names on a hit: the argument bound to its
    parameter (*named*), what that argument holds as a top-level attribute
    (or, one level down, what that attribute holds as one: read from each
    ``__dict__``, as the store read it, before the hit binds anything), or
    the module global. Raises `LookupError` (`Unresolved`, saying why, for
    an attribute that is not there or that a property or other descriptor
    of its class now answers for)."""
    if relink.kind == "argument":
        return named[relink.where[0]]
    if relink.kind == "attribute":
        return _held_at(relink.where, named)
    if relink.kind == "attributes":
        found = [_held_at(path, named) for path in relink.where]
        if any(obj is not found[0] for obj in found):
            held = " and ".join(".".join(path) for path in relink.where)
            raise Unresolved(f"{held} held the object it bound by identity, and hold different objects now")
        return found[0]
    module, name = relink.where
    held = _by_path(module, name)
    if held is None:
        raise LookupError(f"{module}.{name}")
    return held


def _held_at(path: tuple[str, ...], named: dict[str, Any]) -> Any:
    """What *path*, a parameter and the attributes under it, holds."""
    param, *names = path
    obj, label = named[param], param
    for name in names:
        obj = held_attribute(obj, name, label, "the object it bound by identity")
        label = f"{label}.{name}"
    return obj


def held_attribute(obj: Any, name: str, label: str, what: str) -> Any:
    """What *obj* (called *label*) holds as its plain attribute *name*,
    read from its ``__dict__``; `Unresolved` saying that *what* is not
    there, or that a property or other descriptor of its class now answers
    for it."""
    try:
        state = object.__getattribute__(obj, "__dict__")
    except AttributeError:
        state = None
    if type(state) is not dict or name not in state:
        raise Unresolved(f"{label}.{name}, {what}, is not an attribute of {label} now")
    for klass in mro_of(type(obj)):
        found = class_dict(klass).get(name, _ABSENT)
        if found is not _ABSENT:
            if hasattr(type(found), "__set__") or hasattr(type(found), "__delete__"):
                raise Unresolved(f"{label}.{name}, {what}, is a property of its class now")
            break
    return state[name]


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
    if class_name(kind) == "ndarray" and getattr(getattr(value, "dtype", None), "kind", None) == "O":
        return [*gc.get_referents(value), *value.flat]
    module = plain_getattr(kind, "__module__", "")
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
    if class_module(type(value)) == _SKLEARN_METRICS:
        for view in state[1:3]:
            held = getattr(getattr(getattr(view, "base", None), "obj", None), "base", None)
            if held is not None:
                found.append(held)
        extra = list(state[3:5])
    else:
        found = [item for item in (*state[:4], state[12]) if class_name(type(item)) == "ndarray"]
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
    module = plain_getattr(kind, "__module__", "") or ""
    if module == _SKLEARN_METRICS:
        return None if any(base.__name__ == "DistanceMetric" for base in class_mro(kind)) else False
    prefix = _SKLEARN_TREES.get(module)
    if prefix is None or not class_name(kind).startswith(prefix):
        return False
    if class_name(kind).endswith("32"):
        try:
            metric = value.__getstate__()[11]
        except Exception:  # noqa: BLE001 - not read: not trusted
            return f"is a {class_name(kind)}, whose state could not be read"
        if class_name(type(metric)) == "EuclideanDistance32":
            return f"is a {class_name(kind)} on the Euclidean metric, which its pickle brings back without its Euclidean flag"
    return None


def _pickling_refusal(value: Any) -> str | None:
    """Why a stored copy of *value* might not hold all it holds, or None.

    A builtin container, an ``OrderedDict``, ``defaultdict`` or ``deque``
    or a NumPy, pandas or SciPy object pickles all it holds, as do scikit-learn's neighbour trees and distance metrics
    (`_sklearn_refusal`), and joblib's ``Memory`` but for the clock read it
    was built with, which a stored copy reads again
    (`kept_state.joblib_memory_state`). Any other object must pickle as a plain instance
    does -- no ``__reduce__`` or ``__reduce_ex__`` of its own -- and a
    ``__getstate__`` of its own must hand back every attribute as it is
    (scikit-learn's adds its version and keeps the rest); one that leaves
    state out would come back from a hit without it."""
    kind = type(value)
    if kind in _EXACT_CONTAINERS or kind in _ATOM_SET:
        return None
    if (plain_getattr(kind, "__module__", "") or "").split(".")[0] in _EXACT_PICKLERS:
        return None
    sklearn = _sklearn_refusal(value, kind)
    if sklearn is not False:
        return sklearn
    if joblib_memory_state(value) is not None:
        # Its pickle leaves out the clock read it was built with, which a
        # stored copy reads again as it is rebuilt (`kept_state`).
        return None
    if kind.__reduce_ex__ is not object.__reduce_ex__ or kind.__reduce__ is not object.__reduce__:
        return f"is a {class_name(kind)}, which pickles through its own __reduce__"
    getstate = plain_getattr(kind, "__getstate__", None)
    if getstate is None or getstate is object.__getstate__:
        return None
    try:
        state = value.__getstate__()
    except Exception:  # noqa: BLE001 - not read: not trusted
        return f"is a {class_name(kind)}, whose __getstate__ could not be read"
    attrs = plain_getattr(value, "__dict__", None)
    if type(state) is not dict or type(attrs) is not dict:
        return f"is a {class_name(kind)}, whose __getstate__ does not hand back its attributes"
    if any(state.get(key, _MISSING) is not held for key, held in attrs.items()):
        return f"is a {class_name(kind)}, whose __getstate__ leaves out or changes some of its attributes"
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
                if class_name(type(held)) == "ndarray" and id(held) not in arguments
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
