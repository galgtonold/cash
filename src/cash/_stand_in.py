"""The object cash puts in a library function's place when a wrapper
function cannot pass for it.

Cash replaces the seed functions of the global random streams
(``random.seed``, ``numpy.random.seed``, the get/set-state functions) from
``import cash`` on. The program around cash still holds and passes those
names, so what stands there has to behave as the function it replaced
wherever it is not called:

- **pickle**: ``random.seed`` is a bound method, which pickles by its
  receiver, and ``numpy.random.seed`` is not where its own name points. A
  wrapper function is stored by module and name instead, and one defined
  inside another function cannot be pickled at all, so
  ``pool.map(np.random.seed, seeds)`` raised. A stand-in pickles as the
  function it replaced: by its receiver for a bound method, by the
  original's own name where that name gives the stand-in, and otherwise by
  the module attribute cash installed it under. Reading the pickle never
  needs cash.
- ``__name__``, ``__qualname__``, ``__module__``, ``__doc__``,
  ``__wrapped__``, the signature and every other attribute are the
  original's, and ``isinstance`` answers as for the original, so ``help()``
  and ``inspect`` show what they showed before.

One type is made per stand-in, with the handler as its ``__call__``: calling
the stand-in runs the handler with no frame of cash's in between.

The readers and probes replaced while a cached call runs
(`cash.tracking.reader_patches`) are module-level functions and stay plain
wrapper functions, named as their originals: cash's keys and ``help()``
treat them as the functions they are.
"""

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Callable
from typing import Any

__all__ = ["StandIn", "is_stand_in", "stand_in"]

_MISSING = object()


class _Named:
    """Pickles as "import *module*, then walk *path*": what stands under
    that name where the pickle is read, with only the standard library."""

    __slots__ = ("module", "path")

    def __init__(self, module: str, path: tuple[str, ...]) -> None:
        self.module = module
        self.path = path

    def __reduce__(self) -> tuple:
        if not self.path:
            return (importlib.import_module, (self.module,))
        return (getattr, (_Named(self.module, self.path[:-1]), self.path[-1]))


def _named(module: Any, qualname: Any) -> Any:
    """What ``module.qualname`` gives in this process now, or `_MISSING`."""
    if not isinstance(module, str) or not isinstance(qualname, str):
        return _MISSING
    found = sys.modules.get(module, _MISSING)
    for part in qualname.split("."):
        if found is _MISSING or part == "<locals>":
            return _MISSING
        found = getattr(found, part, _MISSING)
    return found


class StandIn:
    """Base of every stand-in; `stand_in` makes them."""

    def __init__(self, original: Any, handler: Callable[..., Any]) -> None:
        state = self.__dict__
        # What a wrapper made with ``functools.wraps`` carries: its markers.
        state.update(getattr(handler, "__dict__", {}))
        state["__wrapped__"] = original
        state["_stand_in_homes"] = []
        for name in ("__name__", "__qualname__", "__module__", "__doc__"):
            try:
                state[name] = getattr(original, name)
            except AttributeError:
                pass

    # ``isinstance(os.path.getsize, types.FunctionType)``, `inspect.isfunction`
    # and pydoc ask the class; pickle and the interpreter ask the type.
    @property
    def __class__(self) -> type:  # type: ignore[override]
        return type(self.__dict__["__wrapped__"])

    def __getattr__(self, name: str) -> Any:
        # Only what the stand-in does not carry itself: ``__code__``,
        # ``__defaults__``, ``__self__``, ``__text_signature__``, ...
        try:
            original = object.__getattribute__(self, "__dict__")["__wrapped__"]
        except KeyError:
            raise AttributeError(name) from None
        return getattr(original, name)

    def __repr__(self) -> str:
        return repr(self.__dict__["__wrapped__"])

    def __copy__(self) -> StandIn:
        return self  # as a function: copied by reference

    def __deepcopy__(self, memo: dict) -> StandIn:
        return self

    def installed_at(self, module: str, name: str) -> None:
        """Note that cash set ``module.name`` to this stand-in."""
        home = (module, name)
        homes = self.__dict__["_stand_in_homes"]
        if home not in homes:
            homes.append(home)

    def __reduce_ex__(self, protocol: int) -> Any:
        state = self.__dict__
        original = state["__wrapped__"]
        receiver = getattr(original, "__self__", None)
        if receiver is not None and not isinstance(receiver, types.ModuleType):
            # A bound method (``random.seed``): by its receiver, as plain
            # Python pickles it.
            return original.__reduce_ex__(protocol)
        module, qualname = state.get("__module__"), state.get("__qualname__", state.get("__name__"))
        if _named(module, qualname) is self:
            return qualname  # by the original's own name: the bytes plain Python writes
        for home_module, home_name in state["_stand_in_homes"]:
            if _named(home_module, home_name) is self:
                # Where cash installed it: the stand-in while the watch is
                # on (in a pool's worker too), the original once it is off.
                return _Named(home_module, (home_name,)).__reduce__()
        if _named(module, qualname) is original:
            return _Named(module, tuple(qualname.split("."))).__reduce__()
        return original.__reduce_ex__(protocol)

    def __reduce__(self) -> Any:
        return self.__reduce_ex__(2)


def _bind(self: StandIn, instance: Any, owner: Any = None) -> Any:
    """A Python function set as a class attribute binds; so does its stand-in."""
    return self if instance is None else types.MethodType(self, instance)


def stand_in(original: Any, handler: Callable[..., Any]) -> StandIn:
    """A callable that runs *handler* and is *original* to everything else.

    *handler* takes what *original* takes. Attributes set on it (the markers
    of a ``functools.wraps`` wrapper) are carried over.
    """
    namespace: dict[str, Any] = {"__call__": staticmethod(handler), "__slots__": ()}
    if isinstance(original, types.FunctionType):
        namespace["__get__"] = _bind
    kind = type("StandIn", (StandIn,), namespace)
    return kind(original, handler)


def is_stand_in(obj: Any) -> bool:
    """Is *obj* one of cash's stand-ins? (``isinstance`` cannot say: it
    answers as for the original.)"""
    return issubclass(type(obj), StandIn)
