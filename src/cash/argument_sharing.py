"""Whether an argument a call changed in place shares state with anything
else the call can see: another argument, or a value the code it ran reads.

A hit writes the state a call left an argument in back into the argument
handed in (`cash.rewrite_in_place`), and the key holds each argument by its
content alone. Both are right only when nothing else the call sees is, or
holds, or shares memory with that argument: ``f(a, a[::-1])`` keys like
``f(a, b)`` with ``b`` an equal, independent array, though its body sees
``b`` change as it writes ``a``, and a global the function reads that is
the argument changes with it.

The answer here is exact or there is none: `Unknown` names why it cannot
be given (code that looks names up by a string, a closure made outside the
call), and the caller then neither stores the state nor serves it. It
walks all of what it looks at, however much that is. Memory is compared
by the address ranges arrays span, which says "shared" for two arrays
interleaved in one buffer: a refusal, never a wrong hit.

What is walked: containers, an object's attributes and what the garbage
collector says any other object refers to, NumPy arrays and their memory,
pandas objects through their blocks, pyarrow arrays through their buffers,
and anything exporting a buffer. Not modules, not code, not a library's
classes: a module is reached by the names the code reads, a function by its
defaults and closure.
"""

from __future__ import annotations

import gc
import sys
import types
from collections.abc import Callable, Iterator
from typing import Any

from .value_types import IMMUTABLE_LEAF_TYPES

__all__ = ["Unknown", "Targets", "memory_of", "reads_of", "read_values", "shared_in"]

#: Global names whose use means the code may reach any global by a string.
BY_STRING = frozenset({"globals", "vars", "locals", "eval", "exec", "__import__", "import_module"})
#: Attributes read off a module that reach other namespaces by a string
#: or a frame (``sys.modules``, ``sys._getframe().f_globals``).
BY_STRING_ATTRS = frozenset({"modules", "_getframe", "currentframe", "import_module", "stack"})

_LEAVES = frozenset(IMMUTABLE_LEAF_TYPES)


class Unknown(Exception):
    """Why whether an argument shares state cannot be told exactly."""


def _bounds(array: Any) -> tuple[int, int] | None:
    """The address range an ndarray's elements span, or None when empty."""
    if array.size == 0:
        return None
    np = sys.modules["numpy"]
    try:
        low, high = np.lib.array_utils.byte_bounds(array)
    except AttributeError:  # NumPy 1.x
        low, high = np.byte_bounds(array)
    return (low, high)


def _ndarray(obj: Any) -> bool:
    np = sys.modules.get("numpy")
    return np is not None and isinstance(obj, np.ndarray)


def _arrow_ranges(obj: Any) -> list[tuple[int, int]] | None:
    """The address ranges of a pyarrow array's buffers, None for anything
    not a pyarrow array."""
    module = type(obj).__module__ or ""
    if not module.startswith("pyarrow"):
        return None
    chunks = getattr(obj, "chunks", None)
    arrays = chunks if isinstance(chunks, list) else [obj]
    ranges: list[tuple[int, int]] = []
    for array in arrays:
        buffers = getattr(array, "buffers", None)
        if not callable(buffers):
            return [] if chunks is None and not hasattr(obj, "buffers") else ranges
        for buffer in buffers():
            if buffer is not None and buffer.size:
                ranges.append((buffer.address, buffer.address + buffer.size))
    return ranges


def _buffer_range(obj: Any) -> tuple[int, int] | None:
    """The address range of the buffer *obj* exports, or None."""
    np = sys.modules.get("numpy")
    if np is None or (sys.version_info >= (3, 12) and not hasattr(type(obj), "__buffer__")):
        return None
    try:
        view = np.frombuffer(memoryview(obj).cast("B"), dtype=np.uint8)
    except (TypeError, ValueError, BufferError):
        return None
    return _bounds(view)


_USER_CLASS: dict[type, bool] = {}


def _user_class(cls: type) -> bool:
    known = _USER_CLASS.get(cls)
    if known is None:
        from .decorator.user_code import is_user_class

        known = _USER_CLASS[cls] = bool(is_user_class(cls))
        if len(_USER_CLASS) > 4096:
            _USER_CLASS.clear()
    return known


class Targets:
    """What an argument a call changed in place is: the objects a hit
    writes into (by identity) and the memory it writes (address ranges)."""

    def __init__(self, changed: Any) -> None:
        self.ids: set[int] = {id(changed)}
        self.ranges: list[tuple[int, int]] = []
        self._keep = [changed]
        if _ndarray(changed):
            self._add_array(changed)
        elif type(changed).__module__.startswith("pandas"):
            mgr = getattr(changed, "_mgr", None)
            blocks = getattr(mgr, "blocks", None)
            if mgr is None or blocks is None:
                raise Unknown(f"cash cannot find the memory of a {type(changed).__name__}")
            self.ids.add(id(mgr))
            for block in blocks:
                self.ids.add(id(block))
                self._add_values(block.values)

    def _add_array(self, array: Any) -> None:
        span = _bounds(array)
        if span is not None:
            self.ranges.append(span)

    def _add_values(self, values: Any) -> None:
        """A block's values: an ndarray, or an extension array whose own
        arrays and buffers are its memory."""
        if _ndarray(values):
            self._add_array(values)
            return
        self.ids.add(id(values))
        self._keep.append(values)
        for found in _Walk(self, collect=True).memory_of(values):
            self.ranges.append(found)

    def overlaps(self, span: tuple[int, int]) -> bool:
        low, high = span
        return any(low < their_high and their_low < high for their_low, their_high in self.ranges)


class _Walk:
    """One walk of values for *targets* (`shared_in`), or, with *collect*,
    for the memory an object holds (`Targets`)."""

    def __init__(self, targets: Targets, collect: bool = False) -> None:
        self.targets = targets
        self.collect = collect
        self.seen: set[int] = set()
        self.found: list[tuple[int, int]] = []
        #: With *collect*: what owns each memory found (an ndarray, a
        #: pyarrow array, an object exporting a buffer), in order.
        self.owners: list[Any] = []

    def memory_of(self, value: Any) -> list[tuple[int, int]]:
        self.run(value)
        return self.found

    def _memory(self, span: tuple[int, int] | None, owner: Any = None) -> bool:
        if span is None:
            return False
        if self.collect:
            self.found.append(span)
            self.owners.append(owner)
            return False
        return self.targets.overlaps(span)

    def run(self, value: Any) -> bool:
        """Whether *value* is, holds or shares memory with the targets."""
        pending = [value]
        while pending:
            obj = pending.pop()
            key = id(obj)
            if key in self.seen:
                continue
            self.seen.add(key)
            if not self.collect and key in self.targets.ids:
                return True
            kind = type(obj)
            if kind in _LEAVES or isinstance(obj, (types.ModuleType, types.CodeType)):
                continue
            if (kind.__module__ or "").startswith("cash."):
                continue  # cash's own state holds copies, never the caller's objects
            if kind in (list, tuple, set, frozenset):
                if not set(map(type, obj)) <= _LEAVES:
                    pending.extend(obj)
                continue
            if kind is dict:
                values = obj.values()
                if not set(map(type, values)) <= _LEAVES:
                    pending.extend(values)
                continue
            if _ndarray(obj):
                if self._memory(_bounds(obj), obj):
                    return True
                if obj.dtype.hasobject:
                    pending.extend(obj.ravel(order="K").tolist())
                continue
            if isinstance(obj, type):
                if _user_class(obj):
                    pending.extend(vars(obj).values())
                continue
            if isinstance(obj, (types.FunctionType, types.MethodType)):
                pending.extend(_held_by_function(obj))
                continue
            ranges = _arrow_ranges(obj)
            if ranges is not None:
                if any([self._memory(span, obj) for span in ranges]):
                    return True
                continue
            if self._memory(_buffer_range(obj), obj):
                return True
            if gc.is_tracked(obj):
                pending.extend(
                    held for held in gc.get_referents(obj) if not isinstance(held, type) or _user_class(held)
                )
        return False


def _held_by_function(fn: Any) -> list:
    """What a function holds that it may hand on: its defaults, its
    closure's contents, and for a bound method its ``self``."""
    if isinstance(fn, types.MethodType):
        return [fn.__self__, fn.__func__]
    if getattr(fn, "_cash_cached", False):
        # A cached function: what it hands back on a hit is a copy, and
        # what it holds is cash's; the function it wraps holds the rest.
        wrapped = getattr(fn, "__wrapped__", None)
        return [wrapped] if wrapped is not None else []
    held = list(fn.__defaults__ or ())
    held += list((fn.__kwdefaults__ or {}).values())
    for cell in fn.__closure__ or ():
        try:
            held.append(cell.cell_contents)
        except ValueError:  # an empty cell
            pass
    return held


def memory_of(value: Any, owners: list | None = None) -> list[tuple[int, int]]:
    """The address ranges of the memory *value* holds: its arrays', its
    buffers', however deep; what owns each is added to *owners*."""
    targets = Targets.__new__(Targets)
    targets.ids, targets.ranges, targets._keep = set(), [], []
    walk = _Walk(targets, collect=True)
    found = walk.memory_of(value)
    if owners is not None:
        owners.extend(walk.owners)
    return found


def shared_in(targets: Targets, value: Any) -> bool:
    """Whether *value* is, holds, or shares memory with what *targets*
    describes. Raises `Unknown` when that cannot be told."""
    return _Walk(targets).run(value)


def reads_of(unit: Any, func: Callable) -> tuple[tuple, ...]:
    """What the code a call ran (*unit*, the trace `code_trace` recorded)
    may read outside its arguments, as names a hit looks up again
    (`read_values`): the globals each function that ran reads, the
    attribute chains it reads off a module of the user's, and the
    function's own closure and defaults. Raises `Unknown` when that cannot
    be known: nothing recorded, code that looks names up by a string or ran
    on a thread the trace did not follow, or a closure made outside the
    call."""
    if unit is None:
        raise Unknown("what the call ran was not recorded")
    from .tracking.trace_items import name_table

    markers = sorted(getattr(unit, "markers", ()) or ())
    if markers:
        raise Unknown(f"what it ran could not all be followed ({', '.join(m.replace('_', ' ') for m in markers)})")
    if getattr(unit, "lookups", None):
        raise Unknown("it looks a name up by a string")
    codes = dict(unit.codes)
    codes.setdefault(func.__code__, (func.__globals__, func.__module__))
    ran = {code.co_qualname for code in codes}
    reads: set[tuple] = {("own",)}
    for code, (namespace, module) in codes.items():
        qualname = code.co_qualname
        held = sys.modules.get(module)
        if held is None or getattr(held, "__dict__", None) is not namespace:
            raise Unknown(f"it ran {qualname}, whose module cash cannot find again")
        table = name_table(code)
        by_string = sorted(BY_STRING & table.reads)
        by_string += sorted(chain for chain in table.chains if BY_STRING_ATTRS & set(chain.split(".")[1:]))
        if by_string:
            raise Unknown(f"{qualname} may look names up by a string ({', '.join(by_string)})")
        free = set(code.co_freevars) - {"__class__"}
        if free and code is not func.__code__ and qualname.rsplit(".<locals>.", 1)[0] not in ran:
            raise Unknown(f"it ran {qualname}, a closure made outside the call")
        for name in table.reads:
            reads.add(("global", module, name))
        for chain in table.chains:
            reads.add(("attr", module, chain))
    return tuple(sorted(reads))


def read_values(reads: tuple[tuple, ...], func: Callable) -> Iterator[tuple[str, Any]]:
    """``(what, value)`` for each of *reads* (`reads_of`) as it is now."""
    from .install_paths import is_user_path

    for read in reads:
        if read[0] == "own":
            for value in _held_by_function(func):
                yield f"a default or closure value of {func.__qualname__}", value
            continue
        _, module, name = read
        held = sys.modules.get(module)
        if held is None:
            raise Unknown(f"the module {module} it read from is not imported")
        namespace = vars(held)
        if read[0] == "global":
            if name in namespace:
                yield f"the global {name!r}", namespace[name]
            continue
        head, *rest = name.split(".")
        value = namespace.get(head)
        reached = [head]
        for part in rest:
            if not isinstance(value, types.ModuleType):
                break
            if not is_user_path(getattr(value, "__file__", None)):
                value = None  # a library's module: its globals are its own
                break
            value = vars(value).get(part)
            reached.append(part)
        if value is not None and not isinstance(value, types.ModuleType):
            yield ".".join(reached), value
