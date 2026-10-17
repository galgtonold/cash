"""The state a notebook class holds on itself, saved where a cell starts and
put back for the cell's re-run.

A constructor or method that changes the class (``Widget.count += 1`` in
``__init__``, ``type(self).registry.append(self)``) is reset for a cell
re-run on its own by re-running the class def. When another cell changes
the class too, that would undo the other cell's change, so the def is not
re-run (`CellEffects.class_state_kept`). Instead the class's own attributes
are saved as the cells above leave them, where the cell starts, and put back
before each re-run of the cell: the class stays the same object, and the
re-run starts where a run from the top does.

Every notebook class in the class's MRO is saved, so an attribute a base
owns is put back on the base, and so is every notebook subclass, where a
method run on a subclass's instance (``type(self).count += 1``) adds the
attribute. Attributes are put back with ``type.__setattr__`` and
``type.__delattr__``, past a metaclass's own ``__setattr__``.
"""

from __future__ import annotations

import copy
import types
import weakref
from typing import Any

from ...sizing import estimate_object_size

#: Most bytes a cell's saved class state may take (estimated): above it the
#: state is not saved, and the re-run keeps the class as it is, with a warning.
CLASS_STATE_BUDGET = 64 * 1024 * 1024

#: Containers put back in place, so a name that holds one keeps holding the
#: class's own.
_IN_PLACE = (list, dict, set)


def _is_state(name: str, value: Any) -> bool:
    """Whether *name* in a class's ``__dict__`` is state, not behaviour."""
    if name.startswith("__") and name.endswith("__"):
        return False
    if isinstance(value, (types.FunctionType, staticmethod, classmethod, property, type)):
        return False
    return not hasattr(type(value), "__get__")


def _state_of(cls: type) -> dict[str, Any]:
    return {name: value for name, value in vars(cls).items() if _is_state(name, value)}


def notebook_classes(cls: type, module: str) -> list[type]:
    """*cls*, the bases in its MRO and the subclasses below it that the
    notebook (module *module*) defines."""
    found: list[type] = [c for c in cls.__mro__ if c.__module__ == module]
    pending = list(type.__subclasses__(cls))
    while pending:
        sub = pending.pop()
        if sub in found:
            continue
        if sub.__module__ == module:
            found.append(sub)
        pending.extend(type.__subclasses__(sub))
    return found


class ClassState:
    """What the notebook classes around one class hold on themselves."""

    def __init__(self, cls: type, module: str, lineage: str | None = None) -> None:
        self.cls = weakref.ref(cls)
        #: The class's lineage when saved: the state a run in order starts from.
        self.lineage = lineage
        self._saved: list[tuple[weakref.ref, dict[str, Any], dict[str, int]]] = []
        size = 0
        for c in notebook_classes(cls, module):
            state = _state_of(c)
            for value in state.values():
                size += estimate_object_size(value)
            if size > CLASS_STATE_BUDGET:
                raise MemoryError(f"its class state is over {CLASS_STATE_BUDGET // (1024 * 1024)} MiB")
            ids = {name: id(value) for name, value in state.items()}
            self._saved.append((weakref.ref(c), copy.deepcopy(state), ids))

    def is_for(self, cls: Any) -> bool:
        return self.cls() is cls

    def put_back(self) -> None:
        """Give each class the attributes it had when saved: changed ones
        back, added ones removed. A list, dict or set the class still holds
        is refilled in place."""
        for ref, saved, ids in self._saved:
            c = ref()
            if c is None:
                continue
            for name, value in list(vars(c).items()):
                if name not in saved and _is_state(name, value):
                    type.__delattr__(c, name)
            for name, value in saved.items():
                fresh = copy.deepcopy(value)
                live = vars(c).get(name)
                if id(live) == ids[name] and type(live) is type(fresh) and isinstance(live, _IN_PLACE):
                    if isinstance(live, list):
                        live[:] = fresh
                    else:
                        live.clear()
                        live.update(fresh)
                    continue
                type.__setattr__(c, name, fresh)
