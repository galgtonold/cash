"""Values as the key sees them: which hold no code and cannot change, what
a callable carries besides its code, and a value rewritten so the code it
holds is keyed by what that code does."""

from __future__ import annotations

import functools
import threading
import types
from typing import Any

from ..analysis.helper_code import is_mock
from ..value_types import IMMUTABLE_LEAF_TYPES
from .user_code import is_user_class, own_package


def iter_contained(obj: Any):
    """Yield *obj*, or its members if it is a plain container, skipping
    primitives outright (they can hold no user class and are common)."""
    if isinstance(obj, (str, bytes, bytearray, int, float, bool, complex, type(None))):
        return
    if isinstance(obj, (list, tuple, set, frozenset)):
        yield from obj
    elif isinstance(obj, dict):
        yield from obj.values()
    else:
        yield obj


def is_immutable_capture(v: Any) -> bool:
    """True for values that are immutable and so define a closure's
    behaviour without drifting between calls. Mutable captures (dict/list/
    set/objects) are excluded: they are typically side-effect accumulators
    (e.g. a hit counter) whose value changes every call - folding those into
    the key would make every call miss. Tuples are looked through however deep
    they nest: one cannot hold itself."""
    if isinstance(v, (bool, int, float, complex, str, bytes, type(None))):
        return True
    if isinstance(v, (tuple, frozenset)):
        return all(is_immutable_capture(x) for x in v)
    return False


#: Synchronization objects (``threading.Lock()``): no result is computed
#: from one, so a lock held as a class attribute, captured by a closure or
#: given as a default is left out of the key.
SYNC_TYPES: tuple[type, ...] = (
    type(threading.Lock()),
    type(threading.RLock()),
    threading.Condition,
    threading.Event,
    threading.Semaphore,
)


def reduced_state(value: Any) -> Any:
    """What ``__reduce_ex__`` says *value* was built with, or None.

    For a C callable with no ``__dict__`` -- ``operator.itemgetter("n")``
    reduces to ``(itemgetter, ("n",))`` -- that is the only place its data
    lives. None when the reduction is just a global name (``np.add``, ``len``:
    nothing carried) or the object refuses to be reduced.
    """
    try:
        reduced = value.__reduce_ex__(4)
    except Exception:  # noqa: BLE001 - not reducible: nothing to fold
        return None
    if isinstance(reduced, str) or not isinstance(reduced, tuple) or len(reduced) < 2:
        return None
    return reduced[:3]


def held_partials(value: Any) -> list[tuple[tuple, dict]]:
    """The arguments of the ``functools.partial`` objects a wrapper instance
    holds as attributes (``np.vectorize.pyfunc``), for a wrapper that is not
    itself a partial."""
    if isinstance(value, functools.partial):
        return []
    state = getattr(value, "__dict__", None)
    if not isinstance(state, dict):
        return []
    return [(p.args, dict(p.keywords)) for p in state.values() if isinstance(p, functools.partial)]


def carried_payload(value: Any) -> Any:
    """The data a callable carries besides its code, or None when it carries none.

    A partial's function and arguments; a bound method's instance (any
    class -- the same as that instance read as a global); a callable
    instance's own attributes, for USER classes only: a library's callable
    instance (`np.vectorize`) keeps lazy caches that change after its first
    call, which would make every call miss -- of those, only the partials
    they hold, which are data the user built them with. A function, a
    class, a method of a class or module, a mock: None.
    """
    if isinstance(value, functools.partial):
        return ("partial", value.func, value.args, dict(value.keywords))
    if isinstance(value, types.MethodType):
        owner = value.__self__
        if isinstance(owner, (type, types.ModuleType)):
            return None
        return ("method", owner)
    if isinstance(value, (types.FunctionType, types.BuiltinFunctionType, type, types.ModuleType)) or is_mock(value):
        return None
    if not callable(value):
        return None
    if is_user_class(type(value), own_package(type(value))):
        state = getattr(value, "__dict__", None)
        if isinstance(state, dict):
            return ("instance", dict(state))
        return ("instance", reduced_state(value))
    held = held_partials(value)
    return ("wrapped partials", held) if held else None


def stabilize_for_global_hash(
    v: Any, hash_callable, _path: frozenset[int] = frozenset(), *, carried: bool = True
) -> Any:
    """Rewrite *v* so callables (incl. lambdas held in containers) are
    replaced by their code identity, making a container of callables
    hashable and content-sensitive (dict-dispatch channel).

    A callable is its code AND what it carries (`carried_payload`): a
    ``Scaler(10)`` with ``__call__`` and ``Scaler(11)`` -- held in a dict, a
    list, or read as a global -- key apart, and so do
    ``{"scale": partial(mul, k=10)}`` and ``k=11``.
    *carried* False keeps the code alone, the fallback for a carried state
    that cannot be hashed.

    However deep the containers nest: a callable left as it is pickles by
    name, which an edit does not move. *_path* (the containers and
    callables being rewritten) ends one that holds itself.

    An object whose state is its ``__dict__`` (`_state_is_its_dict`) and that
    holds code is rewritten as its class's name and that dict:
    ``CFG = {"b": Box(lambda x: x + 1)}`` cannot be pickled as it is, and
    its lambda is an input. An object that holds no code is left as it is.
    """
    return _stabilized(v, hash_callable, _path, carried)[0]


#: `_state_is_its_dict` per class: asked of every object a global holds.
_STATE_IS_DICT: dict[type, bool] = {}


#: Values with nothing in them that can hold code.
_NO_CODE_LEAVES = frozenset(IMMUTABLE_LEAF_TYPES)


# None before Python 3.11, where ``object`` has no ``__getstate__``.
_OBJECT_GETSTATE = getattr(object, "__getstate__", None)


def _state_is_its_dict(t: type) -> bool:
    """Does an instance of *t* pickle as its class and its ``__dict__``, and
    nothing else? No custom reduce or getstate, no slots: then its dict,
    rewritten, stands for all of it. ``SimpleNamespace`` reduces to exactly
    that."""
    try:
        return _STATE_IS_DICT[t]
    except KeyError:
        known = _STATE_IS_DICT[t] = _reads_as_its_dict(t)
        return known
    except TypeError:  # a class whose metaclass makes it unhashable
        return _reads_as_its_dict(t)


def _reads_as_its_dict(t: type) -> bool:
    """`_state_is_its_dict`, worked out."""
    if t is types.SimpleNamespace:
        return True
    if t.__reduce_ex__ is not object.__reduce_ex__ or t.__reduce__ is not object.__reduce__:
        return False
    if getattr(t, "__getstate__", _OBJECT_GETSTATE) is not _OBJECT_GETSTATE:
        return False
    return all(set(getattr(k, "__slots__", ())) <= {"__dict__", "__weakref__"} for k in t.__mro__)


def _stabilized(v: Any, hash_callable, _path: frozenset[int], carried: bool) -> tuple[Any, bool]:
    """`stabilize_for_global_hash`, and whether any code was rewritten."""
    if type(v) in _NO_CODE_LEAVES:
        return v, False
    if id(v) in _path:
        return ("__cash_cycle__", type(v).__qualname__), False
    if callable(v) and not isinstance(v, type):
        try:
            ident: Any = hash_callable(v)
        except (OSError, TypeError, ValueError):
            ident = getattr(v, "__qualname__", repr(v))
        payload = carried_payload(v) if carried else None
        if payload is None:
            return ("__cash_callable__", ident), True
        return ("__cash_callable__", ident, _stabilized(payload, hash_callable, _path | {id(v)}, True)[0]), True
    if isinstance(v, dict):
        inner = _path | {id(v)}
        out, changed = {}, False
        for k, val in v.items():
            out[k], moved = _stabilized(val, hash_callable, inner, carried)
            changed = changed or moved
        return out, changed
    if isinstance(v, (list, tuple)):
        if all(type(x) in _NO_CODE_LEAVES for x in v):
            return v, False
        inner = _path | {id(v)}
        parts = [_stabilized(x, hash_callable, inner, carried) for x in v]
        return type(v)(x for x, _ in parts), any(moved for _, moved in parts)
    state = getattr(v, "__dict__", None)
    if (
        type(state) is dict
        and state
        and not isinstance(v, (type, types.ModuleType))
        and _state_is_its_dict(type(v))
        # A record of plain values holds no code: nothing to rewrite.
        and not all(type(x) in _NO_CODE_LEAVES for x in state.values())
    ):
        rewritten, changed = _stabilized(state, hash_callable, _path | {id(v)}, carried)
        if changed:
            t = type(v)
            return ("__cash_object__", f"{t.__module__}.{t.__qualname__}", rewritten), True
    return v, False


#: Leaves of plain data (`plain_data_kind`): exact types that cannot change.
_PLAIN_LEAVES = frozenset(IMMUTABLE_LEAF_TYPES)
#: Containers and depth `plain_data_kind` looks through before giving up.
_PLAIN_BUDGET = 4096
_PLAIN_DEPTH = 16


def plain_data_kind(value: Any) -> str | None:
    """``"immutable"`` for a number, string, date or ``None``, or tuples and
    frozensets of them; ``"plain"`` for such data in exact dicts, lists and
    sets too; None for anything else (an object, a callable, a subclass,
    something too big or too deep to look through, a container that holds
    itself).

    Plain data holds no code and no instance, so a data global of it needs
    no callable replaced and no code searched for; immutable plain data
    cannot change while the global holds the same object.
    """
    leaves = _PLAIN_LEAVES
    if type(value) in leaves:
        return "immutable"
    budget = _PLAIN_BUDGET
    mutable = False
    level = [value]
    for _ in range(_PLAIN_DEPTH):
        nested = []
        for x in level:
            t = type(x)
            if t in leaves:
                continue
            budget -= 1
            if budget < 0:
                return None
            if t is tuple or t is frozenset:
                nested.extend(x)
            elif t is list or t is set:
                mutable = True
                nested.extend(x)
            elif t is dict:
                mutable = True
                nested.extend(x.keys())
                nested.extend(x.values())
            else:
                return None
        if not nested:
            return "plain" if mutable else "immutable"
        if len(nested) > _PLAIN_BUDGET * 16:
            return None
        level = nested
    return None
