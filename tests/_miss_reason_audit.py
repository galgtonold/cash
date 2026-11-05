"""The test-time check that a statement run again says why it ran.

A statement or loop iteration that ran instead of being served must say why
on its badge row: the input that changed, the refusal, the random state that
moved. A row that runs again with none of them read "computed, no reason
given", and a user had nothing to go on.

The first run of a statement carries no reason on purpose (see
``tests/test_notebook/statement/test_miss_reason.py``), so the check only
looks at a statement this session has already counted: the same text, in the
same pass of the same loops.

`audit` is the check itself, a pure function over one cell's metrics. With
``CASH_TEST_AUDIT_MISS_REASONS`` set to a file, `install` wraps
``CashSession.record_cell`` so every cell counted in this process is
checked and each row without a reason is appended to that file as one JSON
line. The root conftest installs it in pytest's own process, and the
notebook harness writes an IPython startup file that installs it in every
test kernel.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from typing import Any

ENV = "CASH_TEST_AUDIT_MISS_REASONS"

#: The fields any one of which explains why a COMPUTED row ran.
REASON_FIELDS = ("miss_reason", "uncacheable_reasons", "skipped_reason")


def _identity(m: Mapping[str, Any]) -> str:
    """Which statement *m* is: its text and the pass of the loops around it.
    Not the loop markers in its stored text: they carry the iterable's
    lineage, so an iteration whose iterable moved would never count as run
    again."""
    from cash.control_markers import strip_markers

    loop_vars = m.get("loop_vars")
    passes = repr(sorted((str(k), repr(v)[:200]) for k, v in loop_vars.items())) if isinstance(loop_vars, dict) else ""
    where = (m.get("loop_header_chain") or m.get("loop_header") or "", m.get("body_index_chain") or "")
    raw = f"{strip_markers(str(m.get('code') or ''))}\x00{passes}\x00{where}"
    return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()


def _status(m: Mapping[str, Any]) -> str:
    status = m.get("status")
    return str(getattr(status, "name", status) or "").upper()


def audit(metrics: Iterable[Mapping[str, Any]], seen: set[str]) -> list[dict[str, Any]]:
    """The COMPUTED rows of one cell's *metrics* that a statement already in
    *seen* produced without a reason; every row's statement joins *seen*."""
    found = []
    for m in metrics:
        # Only rows the statement processor made: it always starts the
        # list of uncacheable reasons. A test that hands the session stats
        # rows of its own leaves it out.
        if not isinstance(m, Mapping) or "uncacheable_reasons" not in m:
            continue
        ident = _identity(m)
        repeat = ident in seen
        seen.add(ident)
        if not repeat or _status(m) != "COMPUTED":
            continue
        if any(m.get(name) for name in REASON_FIELDS):
            continue
        found.append(
            {
                "code": str(m.get("code") or "")[:400],
                "loop_vars": repr(m.get("loop_vars"))[:200] if m.get("loop_vars") else None,
                "is_upstream": bool(m.get("is_upstream")),
                "execution_time": m.get("execution_time"),
                "cache_key": str(m.get("cache_key") or "")[:24],
                "storage": list(m.get("storage") or ()),
            }
        )
    return found


#: ``id(session) -> (session, the statements it counted)``, the newest last.
_SEEN: dict[int, tuple[Any, set[str]]] = {}


def install(path: str | None = None) -> bool:
    """Check every cell this process counts, appending each row without a
    reason to *path* (default: ``$CASH_TEST_AUDIT_MISS_REASONS``)."""
    path = path or os.environ.get(ENV)
    if not path:
        return False
    from cash.notebook.ipython.session import CashSession

    if getattr(CashSession.record_cell, "_miss_reason_audit", False):
        return True
    original = CashSession.record_cell

    def record_cell(self, all_metrics, wall_time: float = 0.0):
        # CashSession has __slots__: the statements each session counted
        # are kept here, by session.
        held = _SEEN.get(id(self))
        if held is None or held[0] is not self:
            held = _SEEN[id(self)] = (self, set())
            while len(_SEEN) > 64:
                del _SEEN[next(iter(_SEEN))]
        seen = held[1]
        try:
            found = audit(all_metrics or (), seen)
        except Exception as exc:  # the check must never break a test
            found = [{"audit_error": repr(exc)}]
        if found:
            where = {"test": os.environ.get("PYTEST_CURRENT_TEST"), "cwd": os.getcwd()}
            try:
                with open(path, "a", encoding="utf-8") as fh:
                    for row in found:
                        fh.write(json.dumps({"where": where, **row}, default=repr) + "\n")
            except OSError:
                pass
        return original(self, all_metrics, wall_time)

    record_cell._miss_reason_audit = True  # type: ignore[attr-defined]
    CashSession.record_cell = record_cell  # type: ignore[method-assign]
    return True


def startup_file(ipython_dir: str) -> None:
    """Write the IPython startup file that installs the check in a kernel
    started with *ipython_dir*, when ``$CASH_TEST_AUDIT_MISS_REASONS`` is set."""
    if not os.environ.get(ENV):
        return
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    startup = os.path.join(ipython_dir, "profile_default", "startup")
    os.makedirs(startup, exist_ok=True)
    with open(os.path.join(startup, "00-cash-miss-reason-audit.py"), "w", encoding="utf-8") as fh:
        fh.write(
            "def _cash_miss_reason_audit():\n"
            "    import sys\n"
            f"    sys.path.append({root!r})\n"
            "    try:\n"
            "        from tests._miss_reason_audit import install\n"
            "        install()\n"
            "    except Exception:\n"
            "        pass\n"
            "    finally:\n"
            f"        sys.path.remove({root!r})\n"
            "_cash_miss_reason_audit()\n"
            "del _cash_miss_reason_audit\n"
        )
