"""What earlier calls learned about a cached function that changes its
arguments in a way a hit cannot replay: such a call is never stored
(`ResultStore.refusal`), so building its key is work thrown away.

A call that was refused for that reason leaves a verdict, kept in the
process and beside the cache (`StoredKeyRecord.note_refused`), under a key
made of what is known before the cache key is built: the function's own
code digest, the digests of the helpers holding a finding that can change
state (`_code_part`; every walked helper's when no finding is a write in
place) and of the cached functions it calls, how it is checked
(``purity=``), and the types of
the arguments it is handed, with the values of flags and strings
(``copy=True`` and ``inplace=False`` take another path through the same
code), what a closure captures (`_captured_part`: a closure capturing
anything else keeps no verdict) and the type of the object a method handed
to ``cash.cache`` is bound to. A call with
a verdict runs uncached straight away, giving the warnings the refused call
gave (`Notices.replay_warnings`).

A verdict can only make cash store less: the call it skips would not have
been stored. It is probed again -- the call keyed, run and checked as any
other -- every `REPROBE_EVERY`-th time it is used, counted across processes,
so what its key does not name (a library the body calls, a helper reached
some way the analysis did not walk, an argument's contents) is seen within
that many calls; a probe that stores, or refuses for another reason, drops
the verdict. A change to the function's own code, to a helper holding such
a finding, or a helper that gains one, names another verdict key at once;
an edit to a helper that only computes does not (a library's ``_fit``
reaches hundreds, and each commit touching one cost a full key build).
The warnings the analysis gives are shown before a verdict is looked up,
from the code as it is now; a verdict replays only those its refused call
gave while keyed and run.

The first call of a process uses the verdict too: a script calling such a
function once per run would otherwise pay the full key build every run, and
the analysis of a new process re-reads the code the key is made of.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import types
from typing import TYPE_CHECKING, Any

from .._type_memo import class_module, class_name, class_qualname
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
    return f"{getattr(kind, '__module__', '')}.{getattr(kind, '__qualname__', class_name(kind))}"


def _argument_part(value: Any) -> str:
    if isinstance(value, _SCALARS) and type(value) in _SCALARS:
        text = repr(value)
        if len(text) > _SCALAR_REPR_MAX:
            text = "#" + hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:16]
        return f"{class_name(type(value))}={text}"
    return _type_name(value)


#: Captured values named by their value in a closure's part of a key.
_CAPTURED_SCALARS = (bool, int, float, complex, str, bytes, type(None))

#: How deep a captured function's own captures are followed.
_CLOSURE_DEPTH = 4


def _captured_part(value: Any, depth: int) -> str | None:
    """What names one captured *value* in a verdict key, or None for a
    value that cannot be named: the closure then keeps no verdict. A
    function is named by its code and by what it captures in turn, a class
    by what it holds (`class_digest`), a logger by its name, a scalar by
    its value."""
    kind = type(value)
    if kind in _CAPTURED_SCALARS:
        text = repr(value)
        if len(text) > _SCALAR_REPR_MAX:
            text = "#" + hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:16]
        return f"{class_name(kind)}={text}"
    if kind is types.FunctionType:
        if depth >= _CLOSURE_DEPTH:
            return None
        inner = _closure_part(value, depth + 1)
        defaults = _values_part(value.__defaults__ or (), depth)
        kwdefaults = sorted((value.__kwdefaults__ or {}).items())
        named = _values_part(tuple(v for _k, v in kwdefaults), depth)
        if inner is None or defaults is None or named is None:
            return None
        keywords = ",".join(k for k, _v in kwdefaults)
        return f"fn={hash_callable_source(value)}[{inner}]({defaults};{keywords}={named})"
    if isinstance(value, type):
        from ..tracking.trace_items import class_digest

        return f"class={class_module(value)}.{class_qualname(value)}={class_digest(value)}"
    if kind is logging.Logger or kind is logging.RootLogger:
        return f"logger={value.name}"
    if kind is tuple:
        inner = _values_part(value, depth)
        return None if inner is None else f"tuple=({inner})"
    return None


def _values_part(values: tuple, depth: int) -> str | None:
    parts = []
    for v in values:
        got = _captured_part(v, depth)
        if got is None:
            return None
        parts.append(got)
    return ",".join(parts)


def _closure_part(func: Any, depth: int = 0) -> str | None:
    """What *func*'s closure captures, each cell by `_captured_part`: ``""``
    for a function that captures nothing, None when a capture cannot be
    named. Read each call: a cell can be rebound."""
    cells = getattr(func, "__closure__", None)
    if not cells:
        return ""
    parts = []
    for name, cell in zip(func.__code__.co_freevars, cells):
        try:
            value = cell.cell_contents
        except ValueError:  # an empty cell: not bound yet
            return None
        got = _captured_part(value, depth)
        if got is None:
            return None
        parts.append(f"{name}={got}")
    return ";".join(parts)


def _code_part(func: Any, report: PurityReport | None) -> str:
    """The function's own code digest and those of the helpers holding a
    finding that can change state (`PurityReport.state_changing_helpers`):
    what the refusal rests on. A helper that only computes, edited, keeps
    the verdict; one that starts to change state holds a finding, and names
    another key. When no finding is a write in place (the change was made
    where the analysis saw none: compiled code, a call by a run-time name)
    or a finding's place is not a walked helper, every walked helper's
    digest is in the key."""
    parts = [f"own={hash_callable_source(func)}"]
    if report is not None:
        hashes = report.helper_source_hashes
        changing = report.state_changing_helpers
        if report.writes_in_place and all(q in hashes for q in changing):
            parts.extend(f"changes:{q}={hashes[q]}" for q in sorted(changing))
        else:
            parts.extend(f"helper:{q}={d}" for q, d in sorted(hashes.items()))
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
        for one no verdict is kept for: a closure capturing a value that
        cannot be named (`_captured_part`), or a function checked with
        ``purity="silent"``, which is never refused for changing its
        arguments. A closure's captures are in the key: a wrapper that
        closes over the function it wraps, a logger and a count (NiMARE's
        ``use_memmap``) keeps one verdict per thing it wraps."""
        func = spec.func
        if mode == "silent":
            return None
        try:
            captured = _closure_part(func)
            if captured is None:
                return None
            held = self._code.get(id(func))
            if held is None or held[0] is not func or held[1] is not report:
                held = (func, report, _code_part(func, report))
                self._code[id(func)] = held
            from .. import __version__

            parts = [__version__, spec.name, held[2], mode, captured]
            if isinstance(func, types.MethodType):
                # ``cash.cache(estimator.fit)``: the instance is in no
                # argument, and subclasses sharing the method take other
                # paths through it (their own attributes and overrides).
                owner = func.__self__
                if isinstance(owner, type):  # a classmethod: the class it is bound to
                    parts.append(f"cls={class_module(owner)}.{class_qualname(owner)}")
                else:
                    parts.append("self=" + _type_name(owner))
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
        when there is none, or this call is the one to probe again.

        The probe starts the count again: one that settles nothing (its
        body raised, or it was the first change a process saw, which
        `CallRunner._settle_verdict` keeps no verdict for) leaves the
        verdict to the calls after it, as one refused would, instead of
        each of them building its key until one is refused, in this
        process and, as the count is kept beside the cache, in the next."""
        verdict = self._verdict(func_name, key)
        if verdict is None:
            return None
        with self._lock:
            uses = int(verdict.get("uses", 0)) + 1
            probe = uses >= REPROBE_EVERY
            counted = {**verdict, "uses": 0 if probe else uses}
            self._memo[(func_name, key)] = counted
        self._stored_keys.note_refused(func_name, key, counted, write=False)
        return None if probe else counted

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
