"""Where a ``@cash.cache`` call's time goes, phase by phase, per function.

Off unless ``CASH_PHASE_TIMES=1`` or ``CASH_PHASE_TIMES_FILE=<path>`` is set
when cash is imported: on, it costs a few microseconds a call (a dozen clock
reads and the counters they feed), which a cheap function would notice; off,
a call pays one flag test per boundary.

A call's time is cut at boundaries into consecutive phases, each the time
since the boundary before (`mark`), so the phases of a call add up to the
time the wrapper took. Inside the key build and the backend read a phase can
be split further (`sub_begin` / `sub_end`): the inner phase's time is taken
out of the one around it, so these add up too.

The phases, by the outcome they are counted under (``hit``, ``fast_hit``,
``miss``, ``uncached``: no key, ran without one; ``not_stored``: ran, then
refused storing; ``raised``):

``wrapper``      the stats wrapper's own work, before and after the call
``fast_probe``   the remembered-hit probe (`hit_memo`): under ``fast_hit``
                 the one that served, its ``read.decode`` apart
``setup``        the per-function analysis, ttl, before the key
``key.code``     the state segment: code, helpers, callees, closure, defaults
``key.globals``  the globals the function and its helpers read
``key.arg_code`` the code the arguments carry (methods, passed functions)
``key.args``     hashing the argument values
``key.frozen_audit``  of those, re-hashing a ``frozen=True`` result passed
                 on that nothing proves unchanged (`decorator.frozen`)
``key.other``    the rest of the key build
``read.fetch``   finding and reading the entry (and promoting it)
``read.decode``  unpickling or copying the value read
``checks``       whether a found entry may be served (or the miss verdict)
``replay``       on a hit: what it puts back (RNG, arguments, hand-off);
                 on a miss: the argument hash the in-place check compares;
                 on an uncached call: what precedes the body
``body_setup``   the trackers, snapshots and watches set up around the body
``body``         the function body, wall time (nested cached calls included)
``post``         after the body: purity, mutation, trace and refusal checks
``store``        serialise and write the entry (what of it the caller waits on)

Exposed per function as ``cache_info()["phase_times"]``, and written at exit,
one JSON line per process, to ``CASH_PHASE_TIMES_FILE`` (a directory gets one
file per process). ``tools/phase_report.py`` adds such files up.
"""

from __future__ import annotations

import atexit
import contextvars
import json
import math
import os
import sys
import threading
import time
from typing import Any

from ._clock import real_perf_counter
from ._user_attrs import plain_getattr

_perf_counter = real_perf_counter()


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no", "off")


#: Where the at-exit dump goes, or "".
DUMP_PATH = os.environ.get("CASH_PHASE_TIMES_FILE", "").strip()
#: Are calls timed by phase? Read on every call: a test may flip it.
ON = _flag("CASH_PHASE_TIMES") or bool(DUMP_PATH)

# Slots of a call's phase list (`new`).
_LAST, _CUR, _START = 0, 1, 2
WRAPPER = 3
FAST_PROBE = 4
SETUP = 5
KEY_CODE = 6
KEY_GLOBALS = 7
KEY_ARG_CODE = 8
KEY_ARGS = 9
KEY_OTHER = 10
READ_FETCH = 11
READ_DECODE = 12
CHECKS = 13
REPLAY = 14
BODY_SETUP = 15
BODY = 16
POST = 17
STORE = 18
KEY_FROZEN_AUDIT = 19
_SIZE = 20

NAMES = {
    WRAPPER: "wrapper",
    FAST_PROBE: "fast_probe",
    SETUP: "setup",
    KEY_CODE: "key.code",
    KEY_GLOBALS: "key.globals",
    KEY_ARG_CODE: "key.arg_code",
    KEY_ARGS: "key.args",
    KEY_FROZEN_AUDIT: "key.frozen_audit",
    KEY_OTHER: "key.other",
    READ_FETCH: "read.fetch",
    READ_DECODE: "read.decode",
    CHECKS: "checks",
    REPLAY: "replay",
    BODY_SETUP: "body_setup",
    BODY: "body",
    POST: "post",
    STORE: "store",
}

#: The phase list of the call whose key build or backend read is running,
#: for the splits deep inside them (`sub_begin`); None outside those.
SINK: contextvars.ContextVar[list | None] = contextvars.ContextVar("_cash_phase_sink", default=None)
#: The running cached call's phase list, set by its wrapper.
CURRENT: contextvars.ContextVar[list | None] = contextvars.ContextVar("_cash_phase_call", default=None)

#: Histogram slots per doubling of a duration: a median read off them is
#: within about 4% of the true one.
_STEPS = 8


def new() -> list:
    """A call's phase list, its clock started now."""
    now = _perf_counter()
    pt = [0.0] * _SIZE
    pt[_LAST] = pt[_START] = now
    pt[_CUR] = 0
    return pt


def mark(pt: list, phase: int) -> None:
    """The time since the last boundary was *phase*'s."""
    now = _perf_counter()
    pt[phase] += now - pt[_LAST]
    pt[_LAST] = now


def elapsed(pt: list) -> float:
    """Seconds since *pt* was started."""
    return _perf_counter() - pt[_START]


def open_window(pt: list, inner: int) -> contextvars.Token:
    """From now, time goes to *inner* unless a split claims it (`sub_begin`)."""
    pt[_CUR] = inner
    return SINK.set(pt)


def close_window(pt: list, token: contextvars.Token) -> None:
    """End the window `open_window` opened: its last stretch goes to the
    phase it was in."""
    SINK.reset(token)
    cur = pt[_CUR]
    if cur:
        mark(pt, cur)
    pt[_CUR] = 0


def sub_begin(phase: int) -> tuple | None:
    """Time from now goes to *phase*, inside the open window; hand the result
    to `sub_end`. None outside a window."""
    pt = SINK.get()
    if pt is None:
        return None
    cur = pt[_CUR]
    if not cur:
        return None
    mark(pt, cur)
    pt[_CUR] = phase
    return pt, cur


def sub_end(handle: tuple | None) -> None:
    """Back to the phase `sub_begin` interrupted."""
    if handle is None:
        return
    pt, previous = handle
    mark(pt, pt[_CUR])
    pt[_CUR] = previous


_frexp = math.frexp
# Histogram slots: `_STEPS` per doubling, from 2**-30 s (1 ns) up; a duration
# outside the range counts in the first or last slot.
_LOW_EXP = -30
_SLOTS = 44 * _STEPS
# Where `_keys_of` puts the two keys that are not phases.
_TOTAL, _CASH = _SIZE, _SIZE + 1
#: outcome -> its stat keys, by slot (`_keys_of`).
_KEYS: dict[str, list] = {}


def _keys_of(outcome: str) -> list:
    keys: list = [None] * (_SIZE + 2)
    for phase, name in NAMES.items():
        keys[phase] = f"{outcome}.{name}"
    keys[_TOTAL], keys[_CASH] = f"{outcome}.total", f"{outcome}.cash"
    _KEYS[outcome] = keys
    return keys


def _slot(seconds: float) -> int:
    if seconds <= 0.0:
        return 0
    m, e = _frexp(seconds)
    return min(max((e - _LOW_EXP) * _STEPS + int((m - 0.5) * (2 * _STEPS)), 0), _SLOTS - 1)


def bucket_value(bucket: int) -> float:
    """The middle of histogram slot *bucket*, in seconds."""
    e, step = divmod(int(bucket), _STEPS)
    return math.ldexp(0.5 + (step + 0.5) / (2 * _STEPS), e + _LOW_EXP)


def median_of(hist: dict) -> float:
    """The median a histogram ``{slot: count}`` stands for."""
    total = sum(hist.values())
    seen = 0
    for bucket in sorted(hist, key=int):
        seen += hist[bucket]
        if seen * 2 >= total:
            return bucket_value(int(bucket))
    return 0.0


def _stat() -> list:
    return [0, 0.0, 0.0, [0] * _SLOTS]


class PhaseRecord:
    """One cached function's phase times in this process: per outcome, per
    slot of `_keys_of`, ``[count, total seconds, max seconds, histogram]``."""

    __slots__ = ("module", "qualname", "name", "outcomes", "lock")

    def __init__(self, module: str, qualname: str, name: str) -> None:
        self.module = module
        self.qualname = qualname
        self.name = name
        #: outcome -> one stat (or None) per slot of `_keys_of`.
        self.outcomes: dict[str, list] = {}
        self.lock = threading.Lock()

    def commit(self, pt: list, outcome: str) -> None:
        """Count one finished call's phase list under *outcome*."""
        mark(pt, WRAPPER)
        total = pt[_LAST] - pt[_START]
        # Its slots 0-2 are the clock's; the totals go there instead.
        pt[0], pt[1], pt[2] = total, total - pt[BODY], 0.0
        frexp, steps, top = _frexp, _STEPS, _SLOTS - 1
        with self.lock:
            rows = self.outcomes.get(outcome)
            if rows is None:
                rows = self.outcomes[outcome] = [None] * (_SIZE + 2)
            for phase in _COMMITTED:
                seconds = pt[phase]
                if not seconds:
                    continue
                at = _TOTAL + phase if phase < 2 else phase
                stat = rows[at]
                if stat is None:
                    stat = rows[at] = _stat()
                stat[0] += 1
                stat[1] += seconds
                if seconds > stat[2]:
                    stat[2] = seconds
                if seconds > 0.0:
                    m, e = frexp(seconds)
                    b = (e - _LOW_EXP) * steps + int((m - 0.5) * (2 * steps))
                    stat[3][b if 0 <= b <= top else (0 if b < 0 else top)] += 1
                else:
                    stat[3][0] += 1

    def _items(self) -> list[tuple[str, int, float, float, dict]]:
        with self.lock:
            out = []
            for outcome, rows in self.outcomes.items():
                keys = _KEYS.get(outcome) or _keys_of(outcome)
                for at, stat in enumerate(rows):
                    if stat is not None:
                        hist = {i: n for i, n in enumerate(stat[3]) if n}
                        out.append((keys[at], stat[0], stat[1], stat[2], hist))
        return sorted(out)

    def summary(self) -> dict[str, dict[str, float]]:
        """``{"<outcome>.<phase>": {count, total, median, max}}``."""
        return {
            key: {"count": n, "total": total, "median": median_of(hist), "max": peak}
            for key, n, total, peak, hist in self._items()
        }

    def dump(self) -> dict[str, Any]:
        stats = {
            key: {"count": n, "total": total, "max": peak, "hist": hist} for key, n, total, peak, hist in self._items()
        }
        return {"module": self.module, "qualname": self.qualname, "name": self.name, "phases": stats}

    def clear(self) -> None:
        with self.lock:
            self.outcomes.clear()


#: The slots `PhaseRecord.commit` reads: the totals it put in 0 and 1, and
#: every phase.
_COMMITTED = (0, 1, *range(WRAPPER, _SIZE))


_RECORDS: list[PhaseRecord] = []
_RECORDS_LOCK = threading.Lock()


def record_for(func: Any, name: str) -> PhaseRecord:
    """A new record for the cached function *func*, kept for the at-exit dump."""
    record = PhaseRecord(
        plain_getattr(func, "__module__", None) or "?", plain_getattr(func, "__qualname__", None) or name, name
    )
    with _RECORDS_LOCK:
        _RECORDS.append(record)
    return record


def outcome_of(entry: dict | None, pt: list) -> str:
    """The outcome a call's `CallLog` entry says it had."""
    if entry is None:
        return "raised"
    if entry["cache_hit"]:
        return "hit"
    reason = entry.get("miss_reason")
    if reason is not None and getattr(reason, "kind", None) == "raised":
        return "raised"
    if not entry.get("cache_key"):
        return "uncached"
    if entry.get("not_stored"):
        return "not_stored"
    return "miss"


def snapshot() -> dict[str, Any]:
    """This process's records, as the at-exit dump writes them."""
    with _RECORDS_LOCK:
        records = list(_RECORDS)
    functions = [r.dump() for r in records]
    return {
        "pid": os.getpid(),
        "argv": list(sys.argv[:4]),
        "time": time.time(),
        "functions": [f for f in functions if f["phases"]],
    }


def dump(path: str = "") -> str | None:
    """Append this process's phase times as one JSON line to *path* (default
    ``CASH_PHASE_TIMES_FILE``; a directory gets ``phase-times-<pid>.jsonl``).
    Returns the file written, or None when there was nothing to write."""
    path = path or DUMP_PATH
    if not path:
        return None
    data = snapshot()
    if not data["functions"]:
        return None
    if os.path.isdir(path) or path.endswith(("/", "\\")):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, f"phase-times-{os.getpid()}.jsonl")
    line = json.dumps(data, separators=(",", ":")) + "\n"
    # One write of one line: appends from processes ending together do not
    # interleave mid-line on a local disk.
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
    return path


def _dump_at_exit() -> None:
    try:
        dump()
    except Exception as exc:  # noqa: BLE001 - never fail interpreter exit
        print(f"cash: could not write phase times to {DUMP_PATH!r}: {exc}", file=sys.stderr)


if DUMP_PATH:
    atexit.register(_dump_at_exit)
