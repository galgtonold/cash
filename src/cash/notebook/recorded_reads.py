"""What a statement read from outside the notebook's variables, as it ran.

A statement reading the environment (``os.environ["MODE"]``) or the data of
one of the user's modules (``mylib.K``, through a function of the module) is
keyed on the values it read. The runtime reads them when it keys the
statement and records them (:class:`ReadParts`); the upstream simulation,
which runs where the environment and the modules hold what the whole notebook
left in them, takes a value from that record rather than live when the
difference is the notebook's own doing: ``os.environ["MODE"] = "b"`` in a
cell below the reader must not re-key the reader. A change made outside the
notebook's cells -- by the shell that started the kernel, a launcher, a
debugger, an edit of the module's file -- is the live value, and is seen.

The runtime tells the two apart by watching the values the records hold
around every statement it runs (:func:`snapshot`, :func:`note_writes`): the
last statement that changed one, and what it left. A live value equal to
what a statement of the notebook left is the notebook's doing; any other is
not. Every statement, not only one whose text spells a change: a function
it calls (``setmode("b")``, ``mylib.configure()``), a magic (``%env``,
``%cd``) or ``importlib.reload(mylib)`` changes them as surely.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any, NamedTuple

from .._memo import CODE_OBJECTS, LruMemo
from ..effects import environment_digests, environment_entry_digest, environment_parts_component
from .callee_reach import reached_user_code
from .lineage_formula import (
    UNHASHABLE,
    module_data_digest,
    module_data_digests,
    module_data_parts_component,
    statement_environment_reads,
)

__all__ = [
    "ReadParts",
    "ReadRecord",
    "Write",
    "choose",
    "note_writes",
    "outside_changes",
    "read_parts",
    "snapshot",
    "watch",
]

#: ``"env"`` for an environment read, ``"mod"`` for module data.
_ENV = "env"
_MOD = "mod"

class ReadParts(NamedTuple):
    """``(label, digest)`` of each environment read and each piece of module
    data a statement reads, in the order its key folds them."""

    env: tuple[tuple[str, str], ...] = ()
    mod: tuple[tuple[str, str], ...] = ()

    def component(self) -> str:
        """The key component: what ``compute_cache_key`` appends."""
        return environment_parts_component(self.env) + module_data_parts_component(self.mod)


class Write(NamedTuple):
    """The last change the runtime saw to a watched value."""

    digest: str
    code: str


@dataclass
class ReadRecord:
    """The runtime's record of what statements read and what changed it
    (``TrackingState.reads``)."""

    #: sha256 of a statement's key without these reads -> what it read when
    #: the runtime last keyed it.
    seen: dict[str, ReadParts] = field(default_factory=dict)
    #: A full key -> the component it folded, for its outputs' lineage.
    by_key: dict[str, str] = field(default_factory=dict)
    #: ``(kind, label)`` of every value a record holds: what :func:`snapshot` watches.
    watched: set[tuple[str, str]] = field(default_factory=set)
    #: ``kind:label`` -> the last change the runtime saw a statement make to it.
    writes: dict[str, Write] = field(default_factory=dict)
    #: ``(kind, label)`` -> its digest when the runtime last read it, keying a
    #: statement or around one that changed it: a different digest now was
    #: made outside the notebook's cells (:func:`outside_changes`).
    known: dict[tuple[str, str], str] = field(default_factory=dict)

    def clear(self) -> None:
        """Forget every record, in place: the components hold this object
        (``TrackingState.reset_session_state``)."""
        for f in fields(self):
            getattr(self, f.name).clear()

    def __len__(self) -> int:
        return sum(len(getattr(self, f.name)) for f in fields(self))


def read_parts(code: str, user_ns: Mapping[str, Any] | None) -> ReadParts:
    """What the environment reads and the module data in *code* hold now."""
    if not code or not user_ns:
        return ReadParts()
    env = tuple(environment_digests(statement_environment_reads(code, user_ns)))
    mod = tuple(module_data_digests(reached_user_code(code, user_ns)))
    return ReadParts(env, mod)


def watch(parts: ReadParts, record: ReadRecord) -> None:
    """Watch what *parts* read, as they read it."""
    for kind, pairs in ((_ENV, parts.env), (_MOD, parts.mod)):
        for label, digest in pairs:
            record.watched.add((kind, label))
            record.known[(kind, label)] = digest


def outside_changes(record: ReadRecord) -> bool:
    """Whether a watched value changed since the runtime last read it, so
    outside the notebook's cells; and take the new values as known."""
    changed = False
    for key in record.watched:
        digest = _current(*key)
        if record.known.get(key) != digest and not digest.startswith(UNHASHABLE):
            changed = True
        record.known[key] = digest
    return changed


def snapshot(watched: Iterable[tuple[str, str]]) -> dict[tuple[str, str], str]:
    """The current digest of each watched value, before a statement runs."""
    return {key: _current(*key) for key in watched}


def note_writes(before: Mapping[tuple[str, str], str], code: str, record: ReadRecord) -> None:
    """Record, for each value in *before* that *code* changed, what it left."""
    for (kind, label), digest in before.items():
        after = _current(kind, label)
        if after != digest and not (after.startswith(UNHASHABLE) and digest.startswith(UNHASHABLE)):
            record.writes[f"{kind}:{label}"] = Write(after, code)
            record.known[(kind, label)] = after


def choose(
    recorded: ReadParts,
    live: ReadParts,
    writes: Mapping[str, Write] | None,
    in_notebook: Callable[[str], bool] | None,
) -> ReadParts:
    """The values a key not made by the runtime takes: each one *live* as a
    rule, the *recorded* one where the difference is the notebook's own doing
    -- the live value is what a statement of the notebook left -- or cannot
    be told (a value that cannot be hashed)."""

    def pick(kind: str, rec: tuple[tuple[str, str], ...], now: tuple[tuple[str, str], ...]):
        seen = dict(rec)
        chosen = []
        for label, digest in now:
            old = seen.get(label)
            if old is not None and old != digest and _keep_recorded(kind, label, old, digest, writes, in_notebook):
                digest = old
            chosen.append((label, digest))
        return tuple(chosen)

    return ReadParts(pick(_ENV, recorded.env, live.env), pick(_MOD, recorded.mod, live.mod))


def _keep_recorded(
    kind: str,
    label: str,
    old: str,
    now: str,
    writes: Mapping[str, Write] | None,
    in_notebook: Callable[[str], bool] | None,
) -> bool:
    if old.startswith(UNHASHABLE) and now.startswith(UNHASHABLE):
        return True
    write = (writes or {}).get(f"{kind}:{label}")
    return write is not None and write.digest == now and in_notebook is not None and in_notebook(write.code)


def _current(kind: str, label: str) -> str:
    if kind == _ENV:
        return environment_entry_digest(label)
    found, value = _module_value(label)
    if not found:
        return "missing"
    if not _immutable(value):
        return module_data_digest(label, value)
    # Watched around every statement: a constant the module still holds is
    # not hashed again. The value is held, so its id is not reused.
    held = _IMMUTABLE_DIGESTS.get(id(value))
    if held is None or held[0] is not value:
        held = _IMMUTABLE_DIGESTS[id(value)] = (value, module_data_digest(label, value))
    return held[1]


#: id(value) -> (value, digest) of the immutable module data :func:`_current` read.
_IMMUTABLE_DIGESTS: LruMemo[int, tuple[Any, str]] = LruMemo(CODE_OBJECTS)


def _immutable(value: Any) -> bool:
    """A number, string or bytes, or a tuple or frozenset of them."""
    if isinstance(value, (bool, int, float, complex, str, bytes)):
        return True
    return isinstance(value, (tuple, frozenset)) and all(_immutable(item) for item in value)


def _module_of(label: str) -> str | None:
    """The loaded module the module-data *label* lives in."""
    parts = label.split(".")
    for end in range(len(parts) - 1, 0, -1):
        name = ".".join(parts[:end])
        if isinstance(sys.modules.get(name), types.ModuleType):
            return name
    return None


def _module_value(label: str) -> tuple[bool, Any]:
    """``(found, value)`` of the module data at *label* now."""
    name = _module_of(label)
    if name is None:
        return False, None
    obj: Any = sys.modules[name]
    for attr in label[len(name) + 1 :].split("."):
        namespace = vars(obj) if isinstance(obj, (types.ModuleType, type)) else None
        if namespace is None or attr not in namespace:
            return False, None
        obj = namespace[attr]
    return True, obj
