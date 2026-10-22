"""Traces: record which code a unit ran, and verify that trace before the
unit's entry is served again.

On by default on Python 3.12 or newer (``sys.monitoring``); ``CASH_TRACE=0``
turns it off. Off, nothing here is loaded beyond this module, and the hooks
in the statement processor and the call unit cost one attribute read each.

No key changes, and an entry whose trace says something it ran changed is
not served: the statement or call runs again (``stale_suspect`` below).
``CASH_SHADOW=1`` adds the diagnosis this started as (shadow mode). Beside the key, each
statement and cached call that runs and is stored gets a *trace*: the user
code that ran (``recorder``), turned into verify items (``items``). On each
lookup the stored traces are checked again, and:

* the key hits but its trace says something it ran changed:
  ``stale_suspect`` (the entry served may be stale);
* the key misses but an earlier trace of the same statement (or of the
  same call with the same arguments) says nothing changed, and that entry is
  still stored: ``missed_reuse``.

Disagreements go to ``<cache dir>/shadow/log.jsonl`` (``CASH_SHADOW_LOG``
moves it), traces to ``<cache dir>/shadow/traces.jsonl`` so a restarted
kernel can verify the entries an earlier one stored, and a summary to
``%cash_status`` (``"shadow"``) and to stderr at exit. A trace with a
marker the recorder cannot account for (a thread still running, worker
processes, a resumed generator, a jitted function, a lookup by string, a
lost monitoring tool id) gives no verdict and is counted as unusable.
"""

from __future__ import annotations

import ast
import atexit
import builtins
import collections
import contextlib
import contextvars
import json
import logging
import os
import sys
import tempfile
import threading
import time
import types
from collections.abc import Callable, Iterable, Mapping
from typing import Any, NamedTuple

from cash._user_attrs import attribute_or, instance_of

__all__ = ["SHADOW", "Shadow", "disable", "enable", "requested", "status"]

logger = logging.getLogger(__name__)

_ENV = "CASH_SHADOW"
_LOG_ENV = "CASH_SHADOW_LOG"
#: Traces kept per statement (per call and arguments): the newest keys.
_PER_GROUP = 4

#: The group of a call whose key `note_call` was not told of.
_UNNOTED = "call:unnoted"
#: Lines past which a traces file mostly of replaced traces is rewritten.
_COMPACT_AT = 2000
_TEXT = 300
#: Markers that make a trace give no verdict.
UNUSABLE = frozenset(
    {
        "thread_alive",
        "process_pool",
        "generator_resumed",
        "tool_conflict",
        "trace_failed",
        "string_lookup",
        "jit",
        "memo",
        "handed_iterable",
        "call_hit_untraced",
    }
)

#: Where a module was loaded from, not what it computes: a file it reads
#: beside itself is the file tracker's to check.
_WHERE_INSTALLED = frozenset({"__file__", "__cached__", "__spec__", "__loader__", "__path__"})

SHADOW: Shadow | None = None
_REASON: str | None = None


def diagnosing() -> bool:
    """``CASH_SHADOW=1``: log disagreements, check misses against earlier
    traces, and report at exit."""
    return os.environ.get(_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def requested() -> bool:
    from ...decorator import code_trace

    return diagnosing() or code_trace.on()


def _interned(item: tuple, table: dict[tuple, tuple]) -> tuple:
    got = table.get(item)
    if got is None:
        table[item] = got = item
    return got


class _Snap:
    """The notebook's state as a unit began: lineages and bound names."""

    __slots__ = ("lineage", "names")

    def __init__(self, lineage: Mapping[str, str], names: Iterable[str]) -> None:
        self.lineage = dict(lineage)
        self.names = frozenset(names)


class Shadow:
    def __init__(self) -> None:
        from ...decorator import code_trace

        # The one the decorator's traced calls use: a process has one.
        self.recorder = code_trace.recorder_or_reason()
        self.ok = self.recorder.tool is not None
        self.reason = self.recorder.reason
        self.diagnose = diagnosing()
        self.dir: str | None = None
        self._backend: Any = None
        self.notebook: dict | None = None
        self.notebook_name = "__main__"
        self.lineage: Mapping[str, str] = {}
        #: key -> trace; group -> keys, newest last.
        self.traces: dict[str, dict] = {}
        self.groups: dict[str, collections.OrderedDict] = {}
        self._items: dict[tuple, tuple] = {}
        self._call_groups: collections.OrderedDict[str, tuple[str, frozenset]] = collections.OrderedDict()
        self._loaded = False
        self._log = None
        self._trace_file = None
        self._lock = threading.Lock()
        self.counts: collections.Counter[str] = collections.Counter()
        self.markers: collections.Counter[str] = collections.Counter()
        self.spent = 0.0
        self.started = time.process_time()

    # -- wiring -----------------------------------------------------------
    def attach(self, user_ns: dict, lineage: Mapping[str, str], backend: Any) -> None:
        if backend is not self._backend or user_ns is not self.notebook:
            # Another cache or another notebook: traces of this one's own.
            self._backend = backend
            self.traces.clear()
            self.groups.clear()
            self._call_groups.clear()
            self._loaded = False
            self.dir = None
            if self._trace_file is not None:
                self._trace_file.close()
                self._trace_file = None
            if self._log is not None and not os.environ.get(_LOG_ENV):
                self._log.close()
                self._log = None
        if self.recorder.notebook_ns is not user_ns:
            self.recorder.notebook_ns = user_ns
            self.recorder.reclassify()
        self.notebook = user_ns
        name = user_ns.get("__name__")
        self.notebook_name = name if isinstance(name, str) else "__main__"
        from ...tracking import trace_items as items

        items.NOTEBOOK[0] = self.notebook_name
        if self.recorder.notebook_name != self.notebook_name:
            self.recorder.notebook_name = self.notebook_name
            self.recorder.reclassify()
        self.lineage = lineage
        if self.dir is None:
            # A property on the backends, a method on the decorator's slot.
            # Called as a method, the property raised a TypeError that was
            # taken for "no local folder": every notebook's traces went to
            # one folder under the system temp dir, and the traces kept per
            # statement were counted across them.
            try:
                try:
                    local = backend.local_dir
                except AttributeError:
                    local = None
                base = local() if callable(local) else local
            except Exception as exc:  # noqa: BLE001 - said, and the temp dir used
                self.counts["errors"] += 1
                logger.warning("cash: traces kept in the temp dir: the cache's local_dir failed: %r", exc)
                base = None
            if not base:
                base = os.environ.get("CASH_CACHE_DIR") or tempfile.gettempdir()
            self.dir = os.path.join(base, "shadow")

    def _ensure_loaded(self) -> None:
        if self._loaded or self.dir is None:
            return
        contextvars.Context().run(self._load)

    def _load(self) -> None:
        self._loaded = True
        path = os.path.join(self.dir, "traces.jsonl")
        if not os.path.exists(path):
            return
        lines = 0
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    lines += 1
                    try:
                        t = json.loads(line)
                        t["items"] = tuple(_interned(tuple(i), self._items) for i in t["items"])
                        self._keep(t)
                    except (ValueError, TypeError, KeyError, AttributeError):
                        # A line cut short or of another shape: skipped.
                        continue
        except OSError:
            # Its entries have no trace now: they run again (`vouched`).
            self.counts["load_failed"] += 1
            return
        self.counts["traces_loaded"] = len(self.traces)
        if lines > _COMPACT_AT and lines > 4 * len(self.traces):
            self._compact(path)

    def _compact(self, path: str) -> None:
        """Rewrite the traces file with the traces kept: it is appended to
        on every store, and read whole by every new kernel."""
        tmp = f"{path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                for t in self.traces.values():
                    fh.write(json.dumps({**t, "items": [list(i) for i in t["items"]]}, default=str) + "\n")
            os.replace(tmp, path)
            self.counts["traces_compacted"] += 1
        except OSError:
            with contextlib.suppress(OSError):
                os.remove(tmp)

    def _keep(self, trace: dict) -> None:
        key, group = trace["k"], trace["g"]
        self.traces[key] = trace
        keys = self.groups.setdefault(group, collections.OrderedDict())
        keys.pop(key, None)
        keys[key] = None
        while len(keys) > _PER_GROUP:
            old, _ = keys.popitem(last=False)
            if self.traces.get(old, {}).get("g") == group:
                self.traces.pop(old, None)

    def _write(self, name: str, row: dict) -> None:
        # In an empty context: cash's own observers (the file tracker, the
        # write observer) look for their collectors in context variables, so
        # a statement does not see these writes as its own.
        contextvars.Context().run(self._write_now, name, row)

    def _write_now(self, name: str, row: dict) -> None:
        if self.dir is None and not (name == "log" and os.environ.get(_LOG_ENV)):
            return
        try:
            with self._lock:
                if name == "log":
                    if self._log is None:
                        path = os.environ.get(_LOG_ENV) or os.path.join(self.dir, "log.jsonl")
                        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                        self._log = open(path, "a", encoding="utf-8")
                    fh = self._log
                else:
                    if self._trace_file is None:
                        os.makedirs(self.dir, exist_ok=True)
                        self._trace_file = open(os.path.join(self.dir, "traces.jsonl"), "a", encoding="utf-8")
                    fh = self._trace_file
                if name == "log" and "PYTEST_CURRENT_TEST" in os.environ:
                    row = {**row, "test": os.environ["PYTEST_CURRENT_TEST"].split(" ")[0]}
                fh.write(json.dumps(row, default=str) + "\n")
                fh.flush()
        except OSError:
            self.counts["write_failed"] += 1

    # -- units ------------------------------------------------------------
    def begin(self, kind: str):
        """Start recording a unit; None when the tool could not be had."""
        if not self.ok:
            return None
        t = time.perf_counter()
        unit = self.recorder.begin(kind)
        if self.notebook is not None:
            unit.snap = _Snap(self.lineage, self.notebook)
        self.spent += time.perf_counter() - t
        return unit

    def end(self, unit) -> None:
        if unit is not None:
            t = time.perf_counter()
            self.recorder.end(unit)
            self.spent += time.perf_counter() - t

    def current(self):
        stack = self.recorder.stack
        return stack[-1] if stack else None

    # -- items ------------------------------------------------------------
    def _global_digest(self, module: str, name: str, ns: Mapping[str, Any] | None, snap: _Snap | None) -> str:
        from ...tracking.trace_items import value_digest

        if ns is None:
            return "missing"
        if module == self.notebook_name and ns is self.notebook:
            if instance_of(ns.get(name), (types.ModuleType, types.FunctionType, type)):
                # By what it is: a module by name (what of it ran is
                # checked by items of its own; its lineage moves with any
                # edit to it, used or not), a function by its code,
                # defaults and closure, a class by what it holds.
                return value_digest(ns[name])
            lineage = snap.lineage if snap is not None else self.lineage
            got = lineage.get(name)
            if got is not None:
                return "L:" + str(got)
            if snap is not None and name not in snap.names:
                return "absent"
        if name not in ns:
            return "absent"
        return value_digest(ns[name])

    def _holder(self, code: types.CodeType, spaces: Iterable[tuple[str, Mapping[str, Any]]], ran: Iterable[Any] = ()):
        """The global holding a function whose code is *code*. One decorator's
        wrappers share their code (``compute_a`` and ``compute_b``, both made
        by ``@passes_through``): the holder taken is the one whose functions
        include the most of what else *ran*, the function the wrapper called,
        and the first one found otherwise. Taking the first, ``compute_b``'s
        call was checked against ``compute_a`` and ran again every time."""
        from ...tracking.trace_items import candidates

        ran = set(ran)
        best, best_score = None, -1
        for module, ns in spaces:
            try:
                pairs = list(ns.items())
            except Exception:  # noqa: BLE001 - a namespace that cannot be listed is skipped
                continue
            for name, value in pairs:
                if isinstance(name, str) and not name.startswith("__") and callable(value):
                    codes = {c.__code__ for c in candidates(value)}
                    if code in codes:
                        score = len(codes & ran)
                        if score > best_score:
                            best, best_score = (module, name), score
        return best

    def build(self, unit, inputs: Iterable[str] = ()) -> tuple[tuple, set[str], set[str]]:
        """The verify items of a finished *unit*, its markers and notes."""
        from ...tracking.trace_items import call_starts

        with call_starts(unit.cells):
            return self._build(unit, inputs)

    def _build(self, unit, inputs: Iterable[str]) -> tuple[tuple, set[str], set[str]]:
        from ...install_paths import is_user_code_module
        from ...jit_functions import jit_python_function
        from ...tracking.trace_items import class_digest, name_table
        from ..string_lookups import code_looks_up_by_string

        inputs = [n for n in inputs if isinstance(n, str) and not n.startswith("__cash")]
        markers = set(unit.markers)
        notes = set(unit.notes)
        snap = unit.snap
        found_items: dict[tuple, str] = {}
        nb, nbn = self.notebook or {}, self.notebook_name

        def add(item: tuple, digest: str) -> None:
            found_items.setdefault(item, digest)

        # What the statement itself binds: its outputs, keyed by its own
        # source, never inputs to check (a class it defines, a function).
        defined: set[str] = set()
        for code, (g, _module) in unit.codes.items():
            if code.co_qualname == "<module>" and g is nb:
                defined |= name_table(code).stores
        defined -= set(inputs)

        def own(module: str, qualname: str) -> bool:
            return module == nbn and qualname.split(".", 1)[0] in defined

        def user_class(cls: Any) -> bool:
            if not isinstance(cls, type):
                return False
            if cls.__module__ == nbn:
                return True
            mod = sys.modules.get(cls.__module__ or "")
            return mod is not None and is_user_code_module(mod)

        def add_class(cls: type) -> None:
            q = cls.__qualname__
            if own(cls.__module__, q):
                return
            if "<locals>" in q:
                add(("gap", cls.__module__, q), "local class")
                return
            add(("class", cls.__module__, q), class_digest(cls))

        for code, (g, module) in list(unit.codes.items()):
            qn = code.co_qualname
            table = name_table(code)
            optimized = bool(code.co_flags & 0x1)
            try:
                if code_looks_up_by_string(code, g) and not self._followed(code, g, unit, nb):
                    markers.add("string_lookup")
            except Exception:  # noqa: BLE001 - a check that fails leaves the trace usable as is
                pass
            if qn == "<module>":
                if not (module == nbn and g is nb):
                    add(("gap", module, qn), "module body")
            elif optimized and not own(module, qn):
                self._code_items(code, g, module, add, add_class, user_class, unit.codes)
            recv = unit.receivers.get(code)
            if recv is not None and user_class(recv):
                add_class(recv)
            ns: Mapping[str, Any] | None = nb if (module == nbn and g is nb) else g
            reads = table.reads if optimized else table.reads - table.stores
            for name in reads:
                if not isinstance(name, str) or name.startswith("__cash"):
                    continue
                if (module == nbn and name in defined) or name in _WHERE_INSTALLED:
                    continue
                add(("global", module, name), self._global_digest(module, name, ns, snap))
                value = ns.get(name) if ns is not None else None
                if value is None:
                    continue
                if value is os.environ or value is os.getenv:
                    add(("environ", "", ""), _environ_digest())
                if jit_python_function(value) is not None:
                    markers.add("jit")
                if issubclass(type(value), types.ModuleType):
                    for chain in table.chains:
                        head, _, rest = chain.partition(".")
                        if head != name:
                            continue
                        if value is os and rest.split(".", 1)[0] in ("environ", "getenv"):
                            add(("environ", "", ""), _environ_digest())
                        elif is_user_code_module(value):
                            add(("attr", module, chain), self._attr_digest(module, chain))
        # What code looking names up by a string called: by the names that
        # hold it, where the string could have found it (`_followed`).
        for fn, g in attribute_or(unit, "callees", {}).values():
            for module, ns in ((nbn, nb), (g.get("__name__"), g)):
                if not isinstance(module, str) or (ns is not nb and module == nbn):
                    continue
                try:
                    names = [n for n, v in list(ns.items()) if v is fn]
                except Exception:  # noqa: BLE001 - a namespace that cannot be listed holds nothing found
                    names = []
                for name in names:
                    if not isinstance(name, str) or name.startswith("__") or (ns is nb and name in defined):
                        continue
                    add(("global", module, name), self._global_digest(module, name, ns, snap if ns is nb else None))
        for name in inputs:
            add(("global", nbn, name), self._global_digest(nbn, name, nb, snap))
        # The file of each module of the user's the trace names, as it was
        # loaded: what a restore reads when the module is not loaded yet
        # (`_restore_verdict`), wherever ``sys.path`` points then.
        for module in {item[1] for item in found_items if item[0] != "file"}:
            if module and module != nbn:
                filed = _loaded_file(module, found_items)
                if filed is not None:
                    add(("file", module, filed[0]), filed[1])
        out = [_interned((*k, v), self._items) for k, v in found_items.items()]
        out.extend(unit.extra)
        return tuple(dict.fromkeys(out)), markers, notes

    def _followed(self, code: types.CodeType, g: Mapping[str, Any], unit, nb: Mapping[str, Any]) -> bool:
        """Does the trace follow *code*'s lookups by a string: what it called
        recorded (`code_trace._Recorder.by_string`), and each lookup's value
        called at once? A statement's own code is the processor's to judge
        (``cache_key.lookups_followed``): the unit says so."""
        from ..string_lookups import code_lookups_traceable

        if code not in attribute_or(self.recorder, "by_string", ()):
            return False
        if code.co_name == "<module>":
            return g is nb and attribute_or(unit, "lookups_followed", False)
        return code_lookups_traceable(code, g)

    def _code_items(self, code, g, module, add, add_class, user_class, ran=()) -> None:
        """The items naming *code*, a function or method that ran: by its
        qualified name, or through the global holding it; and the class it
        was found on, with that class's subclasses."""
        from ...tracking.trace_items import code_digest, namespace_of, resolve, value_digest

        nb, nbn = self.notebook or {}, self.notebook_name
        qn = code.co_qualname
        rc, fn, owner = resolve(module, qn, nb, nbn)
        found = rc is not None
        held = None
        if not found or code.co_freevars:
            held = self._holder(code, [(module, g), (nbn, nb)], ran)
        if held is not None:
            hmod, hname = held
            hns = namespace_of(hmod, nb, nbn)
            add(("held", hmod, hname, qn), value_digest(hns.get(hname)) if hns is not None else "missing")
        if found:
            add(("code", module, qn), code_digest(code) + _defaults(fn, code, value_digest))
            if code.co_freevars and held is None:
                add(("gap", module, qn + "#closure"), "closure not held by a global")
        elif held is None:
            add(("gap", module, qn), "unresolved")
        if isinstance(owner, type) and user_class(owner):
            add_class(owner)
            try:
                subs = owner.__subclasses__()[:16]
            except TypeError:
                subs = []
            for sub in subs:
                if user_class(sub):
                    add_class(sub)

    def _attr_digest(self, module: str, chain: str) -> str:
        from ...tracking.trace_items import inspect_getattr, namespace_of, value_digest

        ns = namespace_of(module, self.notebook or {}, self.notebook_name)
        parts = chain.split(".")
        if ns is None or parts[0] not in ns:
            return "missing"
        obj = ns[parts[0]]
        for p in parts[1:]:
            if not issubclass(type(obj), types.ModuleType):
                break
            try:
                obj = inspect_getattr(obj, p)
            except AttributeError:
                return "absent"
        return value_digest(obj)

    # -- verification -----------------------------------------------------
    def _now(self, item: tuple, memo: dict) -> str | None:
        """What *item*'s digest is now; None for a gap."""
        got = memo.get(item[:-1], _UNSET)
        if got is not _UNSET:
            return got
        from ...tracking.trace_items import (
            class_digest,
            code_digest,
            inspect_getattr,
            namespace_of,
            resolve,
            value_digest,
        )

        kind = item[0]
        nb, nbn = self.notebook or {}, self.notebook_name
        now: str | None
        if kind == "code":
            rc, fn, _ = resolve(item[1], item[2], nb, nbn)
            now = "missing" if rc is None else code_digest(rc) + _defaults(fn, rc, value_digest)
        elif kind == "held":
            ns = namespace_of(item[1], nb, nbn)
            now = "missing" if ns is None else value_digest(ns[item[2]]) if item[2] in ns else "absent"
        elif kind == "class":
            ns = namespace_of(item[1], nb, nbn)
            obj: Any = None
            if ns is not None:
                parts = item[2].split(".")
                obj = ns.get(parts[0])
                for p in parts[1:]:
                    try:
                        obj = inspect_getattr(obj, p)
                    except AttributeError:
                        obj = None
                        break
            now = class_digest(obj) if isinstance(obj, type) else "missing"
        elif kind == "global":
            ns = namespace_of(item[1], nb, nbn)
            now = self._global_digest(item[1], item[2], ns, None)
        elif kind == "attr":
            now = self._attr_digest(item[1], item[2])
        elif kind == "environ":
            now = _environ_digest()
        elif kind == "file":
            # A loaded module is checked by its items, as it runs: an edit
            # to its file not loaded yet changes nothing a run would see.
            if item[1] in sys.modules:
                now = item[-1]
            else:
                state = _file_state(item[2])
                now = "missing" if state is None else state.digest
        else:
            now = None
        memo[item[:-1]] = now
        return now

    def verify(self, trace: dict, memo: dict) -> dict:
        changed = []
        opaque = gaps = 0
        for item in trace["items"]:
            now = self._now(item, memo)
            if now is None:
                gaps += 1
                continue
            was = item[-1]
            if was.startswith("opaque:"):
                opaque += 1
            if now != was:
                changed.append([*item[:-1], was, now])
        return {"changed": changed, "opaque": opaque, "gaps": gaps}

    def _usable(self, trace: dict) -> bool:
        return not (set(trace.get("markers", ())) & UNUSABLE)

    def check(
        self,
        kind: str,
        key: str,
        group: str,
        found: bool,
        code: str,
        peek: Callable[[str], Any] | None,
        entry: Any = None,
    ) -> dict | None:
        """Verify the traces for a lookup of *key* in *group*: the verdict to
        log once the outcome is known (`resolve`). A hit is checked by the
        trace its entry carries (*entry*, from its metadata), never by one
        kept elsewhere: a file of traces shared by caches could hold another
        cache's trace for the same key."""
        t = time.perf_counter()
        try:
            memo: dict = {}
            pending: dict = {"kind": kind, "key": key, "group": group, "code": code[:_TEXT], "found": found}
            if not found and not self.diagnose:
                # Whether an earlier trace would have served it is a
                # question for the diagnosis alone.
                pending["miss"] = "unchecked"
                return pending
            if found:
                trace = self._from_entry(entry)
                if trace is None:
                    pending["hit"] = "no_trace"
                elif not self._usable(trace):
                    pending["hit"] = "unusable"
                    pending["markers"] = sorted(trace.get("markers", ()))
                else:
                    v = self.verify(trace, memo)
                    pending["hit"] = "stale_suspect" if v["changed"] else "agree"
                    pending["verify"] = v
                    pending["trace_code"] = trace.get("code")
            else:
                self._ensure_loaded()
                keys = [k for k in self.groups.get(group, ()) if k != key]
                pending["miss"] = "no_candidates" if not keys else "agree"
                unusable = 0
                for k in reversed(keys):
                    trace = self.traces.get(k)
                    if trace is None:
                        continue
                    if not self._usable(trace):
                        unusable += 1
                        continue
                    v = self.verify(trace, memo)
                    if v["changed"]:
                        continue
                    present = None
                    if peek is not None:
                        try:
                            present = peek(k) is not None
                        except Exception:  # noqa: BLE001 - an entry that cannot be peeked counts as gone
                            present = None
                    pending["miss"] = "missed_reuse" if present is not False else "missed_reuse_evicted"
                    pending["trace_key"] = k
                    pending["verify"] = v
                    break
                if unusable and pending["miss"] == "agree" and unusable == len(keys):
                    pending["miss"] = "unusable"
            return pending
        except Exception as exc:  # noqa: BLE001 - never break the user's run
            self.counts["errors"] += 1
            self._write("log", {"event": "shadow_error", "where": "check", "error": repr(exc)})
            return None
        finally:
            self.spent += time.perf_counter() - t

    def resolve(self, pending: dict | None, served: bool) -> None:
        """Compare the verdict *pending* with today's outcome and log a
        disagreement."""
        if pending is None:
            return
        kind = pending["kind"]
        if served:
            verdict = pending.get("hit", "no_trace")
        elif pending["found"]:
            verdict = "found_not_served"
        else:
            verdict = pending.get("miss", "no_candidates")
        if not served and pending.get("hit") == "stale_suspect":
            verdict = "rerun"
        self.counts[f"{kind}:{'hit' if served else 'miss'}:{verdict}"] += 1
        if self.diagnose and verdict in ("stale_suspect", "rerun", "missed_reuse", "missed_reuse_evicted"):
            row = {
                "event": verdict,
                "unit": kind,
                "code": pending["code"],
                "key": pending["key"][:24],
                "pid": os.getpid(),
                "t": time.time(),
            }
            if "trace_key" in pending:
                row["trace_key"] = pending["trace_key"][:24]
            v = pending.get("verify") or {}
            row["changed"] = v.get("changed", [])[:12]
            row["n_changed"] = len(v.get("changed", []))
            row["opaque"] = v.get("opaque", 0)
            row["gaps"] = v.get("gaps", 0)
            self._write("log", row)

    def entry_trace(self, kind: str, key: str, group: str, code: str, unit, inputs: Iterable[str]) -> dict | None:
        """The trace of *unit*, which ran and is about to be stored under
        *key*: what the entry carries in its own metadata (`check` reads it
        from there), or None when it could not be built. Kept here too for
        the miss diagnosis."""
        if unit is None:
            return None
        t = time.perf_counter()
        try:
            items, markers, notes = self.build(unit, inputs)
            trace = {
                "k": key,
                "g": group,
                "kind": kind,
                "code": code[:_TEXT],
                "items": items,
                "markers": sorted(markers),
                "notes": sorted(notes),
                "pid": os.getpid(),
            }
            if self.diagnose:
                self._ensure_loaded()
            self._keep(trace)
            self.counts[f"{kind}:traces"] += 1
            for m in markers:
                self.markers[m] += 1
            if markers & UNUSABLE:
                self._write(
                    "log", {"event": "trace_unusable", "unit": kind, "code": code[:_TEXT], "markers": sorted(markers)}
                )
            if self.diagnose:
                self._write("traces", trace)
            # Plain lists of strings, as metadata must be.
            return {"items": [list(i) for i in items], "markers": sorted(markers)}
        except Exception as exc:  # noqa: BLE001 - never break the user's run
            self.counts["errors"] += 1
            self._write("log", {"event": "shadow_error", "where": "store", "error": repr(exc)})
            return None
        finally:
            self.spent += time.perf_counter() - t

    def _from_entry(self, entry: Any) -> dict | None:
        """The trace an entry's metadata carries, as `verify` reads it; None
        for an entry stored without one, or with one that does not read."""
        if not isinstance(entry, Mapping):
            return None
        try:
            items = tuple(_interned(tuple(str(part) for part in i), self._items) for i in entry["items"])
            markers = [str(m) for m in entry.get("markers", ())]
        except Exception:  # noqa: BLE001 - a trace that does not read vouches for nothing
            self.counts["entry_trace_unread"] += 1
            return None
        return {"items": items, "markers": markers}

    def statement_trace(self, run: Any) -> dict | None:
        """The trace a statement's entry is stored with (`entry_trace`)."""
        unit = attribute_or(run, "trace_unit", None)
        if unit is None or run.skip_cache or not run.cache_key:
            return None
        if run.memo_used:
            unit.markers.add("memo")
        if run.handed_iterable:
            unit.markers.add("handed_iterable")
        return self.entry_trace("stmt", run.cache_key, _stmt_group(run), run.code, unit, run.inputs)

    def call_trace(self, key: str, site: Any, unit) -> dict | None:
        """The trace a cached call's entry is stored with (`entry_trace`)."""
        if unit is None:
            return None
        group, by_content = self._call_groups.get(key) or (_UNNOTED, frozenset())
        inputs = [n for n in site.free_names if n not in by_content]
        return self.entry_trace("call", key, group, site.source, unit, inputs)

    # -- statements -------------------------------------------------------
    def statement_lookup(self, proc: Any, run: Any, found: bool, entry: Any = None) -> None:
        """Before a statement is served or runs: verify its traces, a hit by
        the trace its entry carries (*entry*)."""
        try:
            self.attach(proc.shell.user_ns, proc.tracking_state.variable_lineage, proc.cash_instance.backend)
        except Exception as exc:  # noqa: BLE001 - shadow never breaks a statement
            # No check made: ``run.shadow`` stays None, not vouched.
            self.counts["errors"] += 1
            self._write("log", {"event": "shadow_error", "where": "attach", "error": repr(exc)})
            logger.debug("cash: the trace check could not start: %r", exc)
            return
        if run.skip_cache or not run.cache_key:
            return
        backend = proc.cash_instance.backend
        run.shadow = self.check(
            "stmt",
            run.cache_key,
            _stmt_group(run),
            found,
            run.code,
            attribute_or(backend, "peek_metadata", None),
            entry,
        )

    def _restore_verdict(self, items: Iterable[tuple]) -> str:
        """`restore_vouched`'s verdict on the items of a trace. The
        notebook's own globals are its inputs, not bound yet (restored
        virtually, by the lineage its key names): skipped, and so is what
        the notebook does not define yet (a ``def`` above, run first when
        it is needed). A module's items are checked as at a lookup when it
        is loaded. A module of the user's not loaded yet (imported inside a
        function, after a restart) can have been edited on disk: its
        functions are compiled from its file, unrun, and compared; anything
        else of it cannot be told, and the statement runs. A library not
        loaded tells nothing: it changes with an install, not an edit.
        A trace that names the module's file, as it was loaded, is read by
        it instead (`_vouched_by_file`): wherever ``sys.path`` points now, a
        file gone or edited does not vouch, and an unchanged one vouches
        for what its text settles."""
        memo: dict = {}
        files: dict[str, Any] = {}
        nbn = self.notebook_name
        items = list(items)
        filed = {item[1]: item for item in items if item[0] == "file" and len(item) == 4}
        for item in items:
            kind, module = item[0], item[1]
            if kind == "gap":
                continue
            if module in filed and module not in sys.modules:
                verdict = _vouched_by_file(item, filed[module])
                if verdict == "agree":
                    continue
                return verdict
            if kind == "environ":
                now = self._now(item, memo)
            elif module == nbn:
                if kind == "global":
                    continue
                now = self._now(item, memo)
                bound = len(item) > 2 and str(item[2]).split(".", 1)[0] in (self.notebook or {})
                if now is None or now == "absent" or (now == "missing" and not bound):
                    continue
            elif module in sys.modules:
                now = self._now(item, memo)
                if now is None:
                    continue
            else:
                if module not in files:
                    files[module] = _unloaded_users_module(module)
                found = files[module]
                if found is None:
                    continue
                if kind != "code" or found is False:
                    return "module_not_loaded"
                compiled = _compiled_code(found, item[2])
                if compiled is None:
                    return "module_not_loaded"
                from ...tracking.trace_items import code_digest

                # Its defaults are not read without running the module.
                if code_digest(compiled) != str(item[-1]).split("/", 1)[0]:
                    return "stale_suspect"
                continue
            if now != item[-1]:
                return "stale_suspect"
        return "agree"

    def restore_vouched(self, key: str, code: str, entry: Any, proc: Any = None) -> bool:
        """May an entry an upstream restore found (``try_virtual_restore``,
        which restores a statement above the one run without running it)
        be restored, by the trace it carries (*entry*)? As at a lookup
        (`vouched`): not when its trace says what it ran changed, nor with
        no trace to check. *proc* has the notebook it is read against (a
        restore can come before any statement of the kernel looked up)."""
        if proc is not None:
            try:
                self.attach(proc.shell.user_ns, proc.tracking_state.variable_lineage, proc.probe.cash_instance.backend)
            except Exception as exc:  # noqa: BLE001 - no check made: not vouched
                self.counts["errors"] += 1
                logger.debug("cash: the trace check could not start: %r", exc)
                return not self.ok
        if not self.ok:
            return True
        t = time.perf_counter()
        try:
            trace = self._from_entry(entry)
            if trace is None:
                verdict = "no_trace"
            elif not self._usable(trace):
                # A statement that found code by a string is served on an
                # agreeing trace only, as at a lookup (``needs_trace``).
                verdict = "string_lookup" if "string_lookup" in trace.get("markers", ()) else "unusable"
            else:
                verdict = self._restore_verdict(trace["items"])
        except Exception as exc:  # noqa: BLE001 - a check that failed vouches for nothing
            self.counts["errors"] += 1
            logger.debug("cash: the trace check of a restore failed: %r", exc)
            verdict = "error"
        finally:
            self.spent += time.perf_counter() - t
        ok = verdict in ("agree", "unusable")
        self.counts[f"upstream:{'hit' if ok else 'miss'}:{verdict}"] += 1
        return ok

    @staticmethod
    def agreed(pending: dict | None) -> bool:
        """Did the check before a lookup (`check`) find the entry's trace,
        usable, with nothing it names changed?"""
        return pending is not None and pending.get("hit") == "agree"

    def vouched(self, pending: dict | None) -> bool:
        """May the entry a lookup found be served, by its trace? Only when
        the check ran and found the trace with nothing it names changed. An
        entry with no trace (stored before traces were kept where they are
        read now, or its trace lost or not written) or a check that failed is
        not vouched for: it runs again, and its run stores a trace. A trace
        that cannot give a verdict (`UNUSABLE`) leaves it to the key, as
        before: running again would only record another such trace. With no
        monitoring tool to record traces at all, the key alone decides, as
        with traces off."""
        if not self.ok:
            return True
        return pending is not None and pending.get("hit") in ("agree", "unusable")

    @staticmethod
    def stale(pending: dict | None) -> bool:
        """Does the check before a lookup (`check`) say the entry found is
        stale: something its unit ran changed since it was stored?"""
        return pending is not None and pending.get("hit") == "stale_suspect"

    def statement_served(self, run: Any, done: Mapping[str, Any]) -> None:
        from ..cache_status import CacheStatus

        served = CacheStatus.parse(done.get("status")) is CacheStatus.RESTORED
        self.resolve(run.shadow, served)
        run.shadow = None

    def statement_finished(self, run: Any, unit, metrics: Mapping[str, Any]) -> None:
        from ..cache_status import CacheStatus

        self.resolve(run.shadow, False)
        run.shadow = None
        if unit is None or run.skip_cache or not run.cache_key:
            return
        if CacheStatus.parse(metrics.get("status")) is not CacheStatus.COMPUTED:
            return
        if not metrics.get("storage"):
            # Its trace went with its entry (`statement_trace`), if it had one.
            self.counts["stmt:not_stored"] += 1

    # -- cached calls -----------------------------------------------------
    def note_call(self, key: str, site: Any, parts: Iterable[Any], by_content: Iterable[str] = ()) -> None:
        """What a call's key was built from, beside the code: its group.
        *by_content* are the names it is keyed on by value, in *parts*: no
        input of its trace by lineage."""
        import hashlib

        h = hashlib.blake2b(digest_size=12)
        h.update(repr((site.source, site.occurrence_index, attribute_or(site, "stmt_identity", None))).encode())
        for p in parts:
            h.update(b"|" + repr(sorted(p.items()) if isinstance(p, Mapping) else p).encode())
        self._call_groups[key] = ("call:" + h.hexdigest(), frozenset(by_content))
        while len(self._call_groups) > 4096:
            self._call_groups.popitem(last=False)

    def call_lookup(self, key: str, site: Any, hit: bool, peek: Callable[[str], Any] | None, entry: Any = None) -> bool:
        """Check a call's entry before it is served, by the trace it carries
        (*entry*): False when the trace says it is stale or there is none to
        check, and the call runs again."""
        # A key built where `note_call` was not told (a call written with
        # ``*``/``**`` before it was) is checked all the same: a hit by the
        # trace its entry carries.
        noted = self._call_groups.get(key) or (_UNNOTED, frozenset())
        pending = self.check("call", key, noted[0], hit, site.source, peek, entry)
        if hit and not self.vouched(pending):
            self.resolve(pending, False)
            return False
        self.resolve(pending, hit)
        if hit:
            unit = self.current()
            if unit is not None:
                trace = self._from_entry(entry)
                if trace is None:
                    unit.markers.add("call_hit_untraced")
                else:
                    unit.extra.extend(trace["items"])
                    unit.markers.update(trace.get("markers", ()))
        return hit

    # -- reporting --------------------------------------------------------
    def summary(self) -> dict:
        c = self.counts
        disagree = {
            k: v for k, v in c.items() if k.endswith((":stale_suspect", ":missed_reuse", ":missed_reuse_evicted"))
        }
        return {
            "on": self.ok,
            "reason": self.reason,
            "log": os.environ.get(_LOG_ENV) or (os.path.join(self.dir, "log.jsonl") if self.dir else None),
            "counts": dict(sorted(c.items())),
            "disagreements": disagree,
            "markers": dict(self.markers),
            "shadow_seconds": round(self.spent, 4),
            "restarts": self.recorder.restarts,
            "other_tools": self.recorder.other_tools() if self.ok else {},
        }

    def report_at_exit(self) -> None:
        try:
            s = self.summary()
            if os.environ.get(_LOG_ENV):
                self._write("log", {"event": "summary", "pid": os.getpid(), **s})
            if self.dir is not None:
                os.makedirs(self.dir, exist_ok=True)
                with open(os.path.join(self.dir, f"summary-{os.getpid()}.json"), "w", encoding="utf-8") as fh:
                    json.dump(s, fh, indent=1, default=str)
            n = sum(s["disagreements"].values())
            print(
                f"cash shadow: {n} disagreement(s) {s['disagreements']}, "
                f"{sum(v for k, v in s['counts'].items() if k.endswith(':traces'))} trace(s), "
                f"{s['shadow_seconds']}s; log {s['log']}",
                file=sys.stderr,
            )
        except Exception:  # noqa: BLE001 - the exit summary is best effort
            pass


_UNSET = object()


def _unloaded_users_module(module: str) -> Any:
    """For a module not imported: its file's compiled code when the module
    is the user's (a file outside the standard library and site-packages),
    False when its file cannot be read or compiled, None for a library or
    a module that is not found. Found without importing anything: a
    package's ``__init__`` is not run to find a module in it."""
    import importlib.machinery

    from ...install_paths import is_user_code_file

    path: list[str] | None = None
    spec = None
    parts = module.split(".")
    for i, part in enumerate(parts):
        name = ".".join(parts[: i + 1])
        loaded = sys.modules.get(name)
        if loaded is not None and i < len(parts) - 1:
            path = list(attribute_or(loaded, "__path__", None) or [])
            continue
        try:
            spec = importlib.machinery.PathFinder.find_spec(part if path is None else name, path)
        except (ImportError, ValueError):
            return None
        if spec is None:
            return None
        path = list(spec.submodule_search_locations or [])
    origin = spec.origin if spec is not None else None
    if not origin or not origin.endswith(".py") or not is_user_code_file(origin):
        return None
    try:
        with open(origin, "rb") as fh:
            return compile(fh.read(), origin, "exec", dont_inherit=True)
    except (OSError, SyntaxError, ValueError):
        return False


class _FileState(NamedTuple):
    digest: str
    #: Its compiled code, None when it does not compile.
    code: types.CodeType | None
    #: ``{name: what binds it}`` for each module global bound once, by a
    #: plain statement of the module's own (`_settled_names`).
    settled: dict[str, ast.stmt]
    #: Every name the module's text binds, at any depth outside its
    #: functions and classes, or declares ``global``; None when its
    #: globals can be bound by text it does not hold (``import *``,
    #: ``globals()``) or its body runs code (`_inert_imports`).
    bound: frozenset[str] | None
    #: The modules its body imports, which must be libraries for what its
    #: text settles to hold (`_vouched_by_file`).
    imports: frozenset[str]


#: ``path -> (stat, state)``: a file's `_FileState`, read again when its
#: stat changes: mtime, size, inode (an atomic save) and ctime, which a
#: copy keeping the mtime (``cp -p``, ``touch -r``) cannot set back. One
#: older than `_SETTLED_S` when read is kept: an edit within a coarse
#: mtime's tick could keep both.
_FILE_STATES: dict[str, tuple[tuple[int, int, int, int], _FileState]] = {}
_SETTLED_S = 2.0


def _file_state(path: str) -> _FileState | None:
    """The state of the module file *path*, or None when it cannot be read.
    Cash's own read: not a file the statement being stored read."""
    from ...tracking.tracker_context import untracked

    with untracked():
        return _read_file_state(path)


def _read_file_state(path: str) -> _FileState | None:
    try:
        st = os.stat(path)
    except (OSError, ValueError):
        return None
    stat = (st.st_mtime_ns, st.st_size, st.st_ino, st.st_ctime_ns)
    kept = _FILE_STATES.get(path)
    if kept is not None and kept[0] == stat:
        return kept[1]
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    import hashlib

    try:
        code: types.CodeType | None = compile(data, path, "exec", dont_inherit=True)
        tree: ast.Module | None = ast.parse(data)
    except (SyntaxError, ValueError):
        code = tree = None
    imports = _inert_imports(tree) if tree is not None else None
    settled, bound = _settled_names(data, tree) if imports is not None else ({}, None)
    # Its code, not its formatting: a comment reworded changes nothing a
    # run computes. A docstring can (``description=__doc__``).
    text = data if tree is None else ast.dump(tree).encode()
    state = _FileState(hashlib.blake2b(text, digest_size=16).hexdigest(), code, settled, bound, imports or frozenset())
    if time.time() - st.st_mtime > _SETTLED_S:
        _FILE_STATES[path] = (stat, state)
    return state


def _inert_imports(tree: ast.Module) -> frozenset[str] | None:
    """The modules *tree*'s body imports, when running it can bind its
    globals only as its text says: it defines functions and classes,
    assigns literals and imports modules, and runs nothing else. None when
    it runs code whose outcome its text does not settle: a call
    (``TABLE.update(settings.OVERRIDES)``, ``settings.apply(__name__)``),
    a branch (``if os.environ.get(...)``), a store to an attribute or an
    item, an augmented assignment, a walrus. The imports must then be
    libraries' (`_vouched_by_file`): a module of the user's can set this
    one's globals as it is imported."""
    imports: set[str] = set()
    imported_names: set[str] = set()

    def plain_value(node: ast.expr | None) -> bool:
        if node is None or isinstance(node, ast.Lambda):
            return True
        while isinstance(node, ast.Attribute):
            node = node.value
        return isinstance(node, ast.Name) or _literal(node)

    def library_call(node: ast.expr | None) -> bool:
        # ``logging.getLogger(__name__)``, ``re.compile("x+")``: a call of
        # an imported module's function (a library's, `_library_imports`)
        # or a builtin, handed plain values; it binds only what it is
        # assigned to, which the text does not settle.
        if not isinstance(node, ast.Call):
            return False
        root = node.func
        while isinstance(root, ast.Attribute):
            root = root.value
        if not isinstance(root, ast.Name) or root.id in _BINDING_BUILTINS:
            return False
        if root.id not in imported_names and root.id not in vars(builtins):
            return False
        return all(plain_value(a) for a in node.args) and all(plain_value(k.value) for k in node.keywords)

    def decorator(node: ast.expr) -> bool:
        if isinstance(node, ast.Call):
            if not all(_literal(a) for a in node.args) or not all(_literal(k.value) for k in node.keywords):
                return False
            node = node.func
        return isinstance(node, (ast.Name, ast.Attribute)) and plain_value(node)

    def function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        args = node.args
        defaults = [*args.defaults, *(d for d in args.kw_defaults if d is not None)]
        return all(decorator(d) for d in node.decorator_list) and all(_literal(d) for d in defaults)

    def body(stmts: list[ast.stmt], *, in_class: bool) -> bool:
        for stmt in stmts:
            if isinstance(stmt, ast.Pass) or (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)):
                continue
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not function(stmt):
                    return False
            elif isinstance(stmt, ast.ClassDef):
                if not (
                    all(decorator(d) for d in stmt.decorator_list)
                    and all(plain_value(b) and not _literal(b) for b in stmt.bases)
                    and all(plain_value(k.value) for k in stmt.keywords)
                    and body(stmt.body, in_class=True)
                ):
                    return False
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                if not all(isinstance(t, ast.Name) for t in targets) or not (
                    plain_value(stmt.value) or library_call(stmt.value)
                ):
                    return False
            elif isinstance(stmt, ast.Expr) and library_call(stmt.value) and not in_class:
                continue
            elif isinstance(stmt, ast.Import) and not in_class:
                imports.update(alias.name for alias in stmt.names)
                imported_names.update(alias.asname or alias.name.split(".", 1)[0] for alias in stmt.names)
            elif isinstance(stmt, ast.ImportFrom) and not in_class and stmt.level == 0 and stmt.module:
                imports.add(stmt.module)
                imported_names.update(alias.asname or alias.name for alias in stmt.names)
            elif not in_class and _main_guard(stmt):
                continue  # not run on import
            else:
                return False
        return True

    return frozenset(imports) if body(tree.body, in_class=False) else None


#: Builtins that bind names, or run code, by what they are handed.
_BINDING_BUILTINS = frozenset({"setattr", "delattr", "exec", "eval", "globals", "vars", "locals", "__import__"})


def _main_guard(stmt: ast.stmt) -> bool:
    """``if __name__ == "__main__":``, with no ``else``."""
    test = stmt.test if isinstance(stmt, ast.If) and not stmt.orelse else None
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


#: Text by which a module binds its globals without naming them.
_UNNAMED_BINDS = (b"globals()", b"vars(", b"sys.modules", b"__dict__", b"setattr(")


def _settled_names(data: bytes, tree: ast.Module | None) -> tuple[dict[str, ast.stmt], frozenset[str] | None]:
    """The module globals *data*'s text binds once and for good, and every
    name it binds (`_FileState`). A name is settled when one statement of
    the module body binds it -- a ``def``, a ``class``, an ``import`` or an
    assignment of a literal -- and nothing else binds it or declares it
    ``global``. Nothing is settled in a module that can bind globals
    without naming them."""
    if tree is None or any(text in data for text in _UNNAMED_BINDS):
        return {}, None
    count: collections.Counter[str] = collections.Counter()
    plain: dict[str, ast.stmt] = {}
    star = False

    def binds(node: ast.AST) -> None:
        nonlocal star
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            count[node.name] += 1
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    star = True
                else:
                    count[alias.asname or alias.name.split(".", 1)[0]] += 1
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            count[node.id] += 1
        elif isinstance(node, ast.ExceptHandler) and node.name:
            count[node.name] += 1
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            count[node.name] += 1
        elif isinstance(node, ast.MatchMapping) and node.rest:
            count[node.rest] += 1
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                continue
            binds(child)

    for stmt in tree.body:
        binds(stmt)
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import)):
            names = (
                [stmt.name]
                if not isinstance(stmt, ast.Import)
                else [a.asname or a.name.split(".", 1)[0] for a in stmt.names]
            )
            plain.update(dict.fromkeys(names, stmt))
        elif isinstance(stmt, ast.ImportFrom) and stmt.level == 0 and stmt.module:
            plain.update({a.asname or a.name: stmt for a in stmt.names if a.name != "*"})
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None:
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            if all(isinstance(t, ast.Name) for t in targets) and _literal(stmt.value):
                plain.update(dict.fromkeys((t.id for t in targets if isinstance(t, ast.Name)), stmt))
    declared = {name for node in ast.walk(tree) if isinstance(node, ast.Global) for name in node.names}
    if star:
        return {}, None
    settled = {name: stmt for name, stmt in plain.items() if count[name] == 1 and name not in declared}
    return settled, frozenset(count) | declared


def _literal(node: ast.expr) -> bool:
    try:
        ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False
    return True


def _loaded_file(module: str, items: Mapping[tuple, str]) -> tuple[str, str] | None:
    """``(path, digest)`` of the file *module*, loaded and the user's, was
    loaded from; a digest that matches no file when the file was edited
    since: a code of it the trace names (*items*) is not its file's."""
    from ...install_paths import is_user_code_file
    from ...tracking.trace_items import code_digest
    from ...tracking.tracker_context import untracked

    path = attribute_or(sys.modules.get(module), "__file__", None)
    if not isinstance(path, str) or not path.endswith(".py"):
        return None
    with untracked():
        if not is_user_code_file(path):
            return None
    state = _file_state(path)
    if state is None:
        return None
    if not _loaded_from(module, path, state):
        return path, "edited since it was loaded"
    for item, digest in items.items():
        if item[0] == "code" and item[1] == module:
            compiled = _compiled_code(state.code, item[2]) if state.code is not None else None
            if compiled is None or code_digest(compiled) != digest.split("/", 1)[0]:
                return path, "edited since it was loaded"
    return path, state.digest


#: module name -> (the module, the digest of its file when cash first
#: read it while loaded): what a later read of the file is held against.
_LOADED_DIGESTS: dict[str, tuple[Any, str]] = {}


def _loaded_from(module: str, path: str, state: _FileState) -> bool:
    """Is *state*, the file *path* as it is now, what *module* was loaded
    from? The module's compiled file keeps the source's mtime and size of
    when it was loaded (written then, or found matching); the digest cash
    first read for the module, while loaded, must match too. An edit made
    while it is loaded, and a statement stored after, then pairs no old
    value with the new file. Without a compiled file to tell (bytecode not
    written), it cannot be told: not."""
    loaded = sys.modules.get(module)
    kept = _LOADED_DIGESTS.get(module)
    if kept is not None and kept[0] is loaded:
        return kept[1] == state.digest
    if not _compiled_matches(loaded, path):
        return False
    if len(_LOADED_DIGESTS) > 4096:
        _LOADED_DIGESTS.clear()
    _LOADED_DIGESTS[module] = (loaded, state.digest)
    return True


def _compiled_matches(module: Any, path: str) -> bool:
    """Does *module*'s compiled file (``__cached__``) record the mtime and
    size *path* has now? CPython writes them there as it loads the source."""
    cached = attribute_or(module, "__cached__", None)
    if not isinstance(cached, str):
        return False
    from ...tracking.tracker_context import untracked

    try:
        with untracked():
            st = os.stat(path)
            with open(cached, "rb") as fh:
                header = fh.read(16)
    except OSError:
        return False
    if len(header) < 16 or int.from_bytes(header[4:8], "little") != 0:
        return False  # a hash-based compiled file: not by mtime
    mtime = int.from_bytes(header[8:12], "little")
    size = int.from_bytes(header[12:16], "little")
    return mtime == int(st.st_mtime) & 0xFFFFFFFF and size == st.st_size & 0xFFFFFFFF


def _vouched_by_file(item: tuple, filed: tuple) -> str:
    """`Shadow._restore_verdict` on *item*, of a module not loaded, by the
    file its trace names (*filed*, a ``file`` item): the module would be
    loaded from what that file holds now (a run imports it again). Gone
    or edited: ``stale_suspect``. Unchanged, it vouches for its code and
    for globals its text settles (`_settled_names`): a literal compared
    by value, a name bound by a ``def``, ``class`` or ``import`` by the
    text, and a name it never binds still absent. Anything else of it
    cannot be told unrun: ``module_not_loaded``."""
    from ...tracking.trace_items import value_digest

    state = _file_state(filed[2])
    if state is None or state.digest != filed[-1]:
        return "stale_suspect"
    kind = item[0]
    if kind == "file":
        return "agree"
    if state.bound is None or not _library_imports(state.imports):
        # Its body runs code its text does not settle: a function of it can
        # be replaced as it is imported (``settings.patch(__name__)``).
        return "module_not_loaded"
    if kind == "code":
        return "agree"
    if kind not in ("global", "class", "held"):
        return "module_not_loaded"
    name = str(item[2]).split(".", 1)[0]
    if kind == "global" and item[-1] == "absent":
        return "agree" if name not in state.bound else "module_not_loaded"
    stmt = state.settled.get(name)
    if stmt is None:
        return "module_not_loaded"
    if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
        if kind != "global" or stmt.value is None:
            return "module_not_loaded"
        return "agree" if value_digest(ast.literal_eval(stmt.value)) == item[-1] else "stale_suspect"
    if isinstance(stmt, ast.ImportFrom):
        # Bound to what another module holds: told only for a library's,
        # which an install changes, not an edit.
        from ...install_paths import is_user_code_module
        from ...tracking.tracker_context import untracked

        source = sys.modules.get(stmt.module or "")
        if source is None:
            return "module_not_loaded"
        with untracked():
            return "module_not_loaded" if is_user_code_module(source) else "agree"
    return "agree"


def _library_imports(names: frozenset[str]) -> bool:
    """Are the modules *names* all loaded, and none the user's? One of the
    user's can set the importing module's globals as it is imported
    (``import overrides``, a self-import)."""
    if not names:
        return True
    from ...install_paths import is_user_code_module
    from ...tracking.tracker_context import untracked

    with untracked():
        for name in names:
            module = sys.modules.get(name)
            if module is None or is_user_code_module(module):
                return False
    return True


def _compiled_code(module_code: types.CodeType, qualname: str) -> types.CodeType | None:
    """The code of the function *qualname* in a module's compiled code."""
    pending = [module_code]
    while pending:
        code = pending.pop()
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                if const.co_qualname == qualname:
                    return const
                pending.append(const)
    return None


def _defaults(fn: Any, code: Any, value_digest: Callable[[Any], str]) -> str:
    if type(fn) is types.FunctionType and fn.__code__ is code and (fn.__defaults__ or fn.__kwdefaults__):
        return "/" + value_digest((fn.__defaults__, fn.__kwdefaults__))
    return ""


def _environ_digest() -> str:
    from ...tracking.trace_items import value_digest

    return value_digest(sorted(os.environ.items()))


def _stmt_group(run: Any) -> str:
    return f"stmt:{run.source_hash}:{run.occurrence_index}"


def enable() -> Shadow | None:
    """Turn shadow mode on in this process (``CASH_SHADOW=1`` does it at import)."""
    global SHADOW, _REASON
    if SHADOW is not None:
        return SHADOW
    if sys.version_info < (3, 12):
        _REASON = "shadow mode needs Python 3.12 or newer (sys.monitoring)"
        return None
    shadow = Shadow()
    if not shadow.ok:
        _REASON = shadow.reason
        return None
    SHADOW = shadow
    if shadow.diagnose:
        atexit.register(shadow.report_at_exit)
    return shadow


def disable() -> None:
    """Turn shadow mode off. The recorder stays: traced calls use it too."""
    global SHADOW
    shadow, SHADOW = SHADOW, None
    if shadow is not None:
        shadow.recorder.notebook_ns = None
        if shadow.diagnose:
            atexit.unregister(shadow.report_at_exit)
        for fh in (shadow._log, shadow._trace_file):
            if fh is not None:
                fh.close()


def status() -> dict | None:
    """The summary for ``%cash_status``; None unless ``CASH_SHADOW=1``."""
    if not diagnosing():
        return None
    if SHADOW is not None:
        return SHADOW.summary()
    if requested():
        return {"on": False, "reason": _REASON}
    return None


# The diagnosis from import, as before; traces alone from the notebook's
# start (`CashMagics`), so a script that never runs a cell installs nothing.
if diagnosing():
    enable()
