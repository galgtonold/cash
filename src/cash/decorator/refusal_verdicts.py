"""What earlier calls learned about a cached function that changes its
arguments in a way a hit cannot replay: such a call is never stored
(`ResultStore.refusal`), so building its key is work thrown away.

A call that was refused for that reason leaves a verdict, kept in the
process and beside the cache (`StoredKeyRecord.note_refused`), under a key
made of what is known before the cache key is built: the function's own
code digest, the digests of the helpers its analysis walked and of the
cached functions it calls, how it is checked (``purity=``), and the types of
the arguments it is handed, with the values of flags and strings
(``copy=True`` and ``inplace=False`` take another path through the same
code). A call with
a verdict runs uncached straight away, giving the warnings the refused call
gave (`Notices.replay_warnings`).

A verdict can only make cash store less: the call it skips would not have
been stored. It is probed again -- the call keyed, run and checked as any
other -- every `REPROBE_EVERY`-th time it is used, counted across processes,
so what its key does not name (a library the body calls, a helper reached
some way the analysis did not walk, an argument's contents) is seen within
that many calls; a probe that stores, or refuses for another reason, drops
the verdict. Any change to the function's own code or a walked helper's
names another verdict key at once.

The first call of a process uses the verdict too: a script calling such a
function once per run would otherwise pay the full key build every run, and
the analysis of a new process re-reads the code the key is made of.
"""

from __future__ import annotations

import hashlib
import threading
from typing import TYPE_CHECKING, Any

from .function_identity import hash_callable_source

if TYPE_CHECKING:
    from ..analysis.purity_report import PurityReport
    from .cached_function import CachedFunction
    from .stored_keys import StoredKeyRecord

__all__ = ["REPROBE_EVERY", "ArgumentRefusal", "RefusalVerdicts"]

#: One call in this many with a verdict is keyed and checked again.
REPROBE_EVERY = 20

#: Scalars whose value, not only their type, goes into a verdict key: the
#: flags and modes that pick a path through the code (``copy=True``,
#: ``inplace=False``, ``flavor="seurat"``). A number is data, keyed by type:
#: with its value, ``f(box, i)`` in a loop never met its own verdict.
_SCALARS = (bool, str, type(None))

#: A scalar longer than this is in a verdict key by its digest.
_SCALAR_REPR_MAX = 64

#: Absent from the process's memo after a look at the record.
_ABSENT: dict = {}


class ArgumentRefusal(str):
    """A refusal (`ResultStore.refusal`) because the call changed an
    argument in place in a way a hit cannot replay: the one a verdict is
    kept for. *unreplayable*: the change is of a kind no hit replays (an
    attribute bound to an object pickled by its own ``__reduce__``), not
    merely one nothing was fingerprinted for before this body ran."""

    unreplayable: bool = False

    def __new__(cls, text: str, *, unreplayable: bool = False) -> ArgumentRefusal:
        made = super().__new__(cls, text)
        made.unreplayable = unreplayable
        return made


def _type_name(value: Any) -> str:
    kind = type(value)
    return f"{getattr(kind, '__module__', '')}.{getattr(kind, '__qualname__', kind.__name__)}"


def _argument_part(value: Any) -> str:
    if isinstance(value, _SCALARS) and type(value) in _SCALARS:
        text = repr(value)
        if len(text) > _SCALAR_REPR_MAX:
            text = "#" + hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:16]
        return f"{type(value).__name__}={text}"
    return _type_name(value)


def _code_part(func: Any, report: PurityReport | None) -> str:
    """The function's own code digest and those of what its analysis walked."""
    parts = [f"own={hash_callable_source(func)}"]
    if report is not None:
        parts.extend(f"helper:{q}={d}" for q, d in sorted(report.helper_source_hashes.items()))
        for ref in report.cached_callees:
            callee = ref()
            parts.append(f"cached={hash_callable_source(callee) if callee is not None else 'gone'}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


class RefusalVerdicts:
    """The verdicts of one `Cash` (see the module docstring)."""

    def __init__(self, stored_keys: StoredKeyRecord) -> None:
        self._stored_keys = stored_keys
        self._lock = threading.Lock()
        #: (func_name, key) -> the verdict, or `_ABSENT` once the record had none.
        self._memo: dict[tuple[str, str], dict] = {}
        #: id(func) -> (func, report, code part).
        self._code: dict[int, tuple[Any, Any, str]] = {}
        #: func_name -> whether it may have a verdict (`may_have`).
        self._names: dict[str, bool] = {}

    def may_have(self, func_name: str) -> bool:
        """Whether *func_name* may have a verdict: one kept in this process,
        or any in its record, read once. A function that never had one pays
        nothing per call beyond this lookup."""
        got = self._names.get(func_name)
        if got is None:
            got = self._names[func_name] = self._stored_keys.any_refused(func_name)
        return got

    def key_for(
        self, spec: CachedFunction, report: PurityReport | None, mode: str, args: tuple, kwargs: dict
    ) -> str | None:
        """The verdict key of a call of *spec* with these arguments, or None
        for one no verdict is kept for: a closure, whose captures are not in
        the key, or a function checked with ``purity="silent"``, which is
        never refused for changing its arguments."""
        func = spec.func
        if mode == "silent" or getattr(func, "__closure__", None):
            return None
        try:
            held = self._code.get(id(func))
            if held is None or held[0] is not func or held[1] is not report:
                held = (func, report, _code_part(func, report))
                self._code[id(func)] = held
            from .. import __version__

            parts = [__version__, spec.name, held[2], mode]
            parts.extend(_argument_part(a) for a in args)
            parts.extend(f"{k}:{_argument_part(v)}" for k, v in sorted(kwargs.items()))
        except Exception:  # noqa: BLE001 - no key: the call is keyed as always
            return None
        return hashlib.sha256("\x00".join(parts).encode("utf-8", "surrogatepass")).hexdigest()[:32]

    def _verdict(self, func_name: str, key: str) -> dict | None:
        slot = (func_name, key)
        verdict = self._memo.get(slot)
        if verdict is None:
            verdict = self._stored_keys.refused(func_name, key) or _ABSENT
            with self._lock:
                verdict = self._memo.setdefault(slot, verdict)
        return None if verdict is _ABSENT else verdict

    def use(self, func_name: str, key: str) -> dict | None:
        """The verdict to run this call uncached by, counted as used; None
        when there is none, or this call is the one to probe again."""
        verdict = self._verdict(func_name, key)
        if verdict is None:
            return None
        with self._lock:
            uses = int(verdict.get("uses", 0))
            if uses + 1 >= REPROBE_EVERY:
                return None
            verdict = {**verdict, "uses": uses + 1}
            self._memo[(func_name, key)] = verdict
        self._stored_keys.note_refused(func_name, key, verdict, write=False)
        return verdict

    def refused(self, func_name: str, key: str, why: str, warnings: list[dict[str, Any]] | None) -> None:
        """A probed call of *key* was refused for changing its arguments."""
        verdict = {"why": str(why), "warnings": list(warnings or ()), "uses": 0}
        with self._lock:
            self._memo[(func_name, key)] = verdict
            self._names[func_name] = True
        self._stored_keys.note_refused(func_name, key, verdict, write=True)

    def drop(self, func_name: str, key: str) -> None:
        """A probed call of *key* was stored, or refused for another reason."""
        with self._lock:
            had = self._memo.get((func_name, key)) not in (None, _ABSENT)
            self._memo[(func_name, key)] = _ABSENT
        if had:  # `use` read the record for this key before the call
            self._stored_keys.note_refused(func_name, key, None, write=True)
