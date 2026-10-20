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

import atexit
import collections
import contextlib
import contextvars
import json
import os
import sys
import tempfile
import threading
import time
import types
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from cash._user_attrs import attribute_or

__all__ = ["SHADOW", "Shadow", "disable", "enable", "requested", "status"]

_ENV = "CASH_SHADOW"
_LOG_ENV = "CASH_SHADOW_LOG"
#: Traces kept per statement (per call and arguments): the newest keys.
_PER_GROUP = 4
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
            base = None
            try:
                base = backend.local_dir()
            except Exception:  # noqa: BLE001 - no local folder: shadow keeps its log elsewhere
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
        me = os.getpid()
        lines = 0
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    lines += 1
                    try:
                        t = json.loads(line)
                    except ValueError:
                        continue
                    if not t.get("disk") and t.get("pid") != me:
                        continue
                    t["items"] = tuple(_interned(tuple(i), self._items) for i in t["items"])
                    self._keep(t)
        except OSError:
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
            if isinstance(ns.get(name), (types.ModuleType, types.FunctionType, type)):
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
                if isinstance(value, types.ModuleType):
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
            if not isinstance(obj, types.ModuleType):
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
    ) -> dict | None:
        """Verify the traces for a lookup of *key* in *group*: the verdict to
        log once the outcome is known (`resolve`)."""
        t = time.perf_counter()
        try:
            self._ensure_loaded()
            memo: dict = {}
            pending: dict = {"kind": kind, "key": key, "group": group, "code": code[:_TEXT], "found": found}
            if not found and not self.diagnose:
                # Whether an earlier trace would have served it is a
                # question for the diagnosis alone.
                pending["miss"] = "unchecked"
                return pending
            if found:
                trace = self.traces.get(key)
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

    def store(self, kind: str, key: str, group: str, code: str, unit, inputs: Iterable[str], disk: bool) -> None:
        """Keep the trace of *unit*, which ran and was stored under *key*."""
        if unit is None:
            return
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
                "disk": disk,
                "pid": os.getpid(),
            }
            self._ensure_loaded()
            self._keep(trace)
            self.counts[f"{kind}:traces"] += 1
            for m in markers:
                self.markers[m] += 1
            if markers & UNUSABLE:
                self._write(
                    "log", {"event": "trace_unusable", "unit": kind, "code": code[:_TEXT], "markers": sorted(markers)}
                )
            self._write("traces", trace)
        except Exception as exc:  # noqa: BLE001 - never break the user's run
            self.counts["errors"] += 1
            self._write("log", {"event": "shadow_error", "where": "store", "error": repr(exc)})
        finally:
            self.spent += time.perf_counter() - t

    # -- statements -------------------------------------------------------
    def statement_lookup(self, proc: Any, run: Any, found: bool) -> None:
        """Before a statement is served or runs: verify its traces."""
        try:
            self.attach(proc.shell.user_ns, proc.tracking_state.variable_lineage, proc.cash_instance.backend)
        except Exception:  # noqa: BLE001 - shadow never breaks a statement
            return
        if run.skip_cache or not run.cache_key:
            return
        backend = proc.cash_instance.backend
        run.shadow = self.check(
            "stmt", run.cache_key, _stmt_group(run), found, run.code, attribute_or(backend, "peek_metadata", None)
        )

    @staticmethod
    def agreed(pending: dict | None) -> bool:
        """Did the check before a lookup (`check`) find the entry's trace,
        usable, with nothing it names changed?"""
        return pending is not None and pending.get("hit") == "agree"

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
        storage = metrics.get("storage") or ()
        if not storage:
            self.counts["stmt:not_stored"] += 1
            return
        if run.memo_used:
            unit.markers.add("memo")
        if run.handed_iterable:
            unit.markers.add("handed_iterable")
        self.store(
            "stmt",
            run.cache_key,
            _stmt_group(run),
            run.code,
            unit,
            run.inputs,
            disk=any(s != "RAM" for s in storage),
        )

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

    def call_lookup(self, key: str, site: Any, hit: bool, peek: Callable[[str], Any] | None) -> bool:
        """Check a call's entry before it is served: False when its trace
        says it is stale, and the call runs again."""
        noted = self._call_groups.get(key)
        if noted is None:
            return hit
        pending = self.check("call", key, noted[0], hit, site.source, peek)
        if hit and self.stale(pending):
            self.resolve(pending, False)
            return False
        self.resolve(pending, hit)
        if hit:
            unit = self.current()
            if unit is not None:
                trace = self.traces.get(key)
                if trace is None:
                    unit.markers.add("call_hit_untraced")
                else:
                    unit.extra.extend(trace["items"])
                    unit.markers.update(trace.get("markers", ()))
        return hit

    def call_finished(self, key: str, site: Any, unit, stored: bool, disk: bool) -> None:
        noted = self._call_groups.get(key)
        if unit is None or noted is None or not stored:
            return
        group, by_content = noted
        inputs = [n for n in site.free_names if n not in by_content]
        self.store("call", key, group, site.source, unit, inputs, disk)

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


def _defaults(fn: Any, code: Any, value_digest: Callable[[Any], str]) -> str:
    if isinstance(fn, types.FunctionType) and fn.__code__ is code and (fn.__defaults__ or fn.__kwdefaults__):
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
