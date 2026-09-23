"""What the decorator does with the purity analyzer's findings and with the
effects it observes while a body runs."""

from __future__ import annotations

from typing import Any

from ..effect_observer import observed_label
from ..purity_analyzer import ISSUE_IMPURE_CALL, ISSUE_NETWORK_READ, PurityIssue
from ..value_types import writable_types


def is_mutable(value) -> bool:
    """Whether a caller can write through *value*, so a copy would differ.

    Only what can actually be written into: a container, an array or a frame,
    or an object whose attributes can be rebound (it has a ``__dict__``). A
    `date`, a `Path`, a `Decimal`, a string or a number cannot be changed
    through the name at all, so handing back a copy of one is the same value --
    warning about those made `return sum(rows), as_of` a finding.
    """
    if isinstance(value, writable_types()):
        return True
    return getattr(type(value), "__dictoffset__", 0) != 0 and hasattr(value, "__dict__")


def shares_memory(result, value) -> bool:
    """Whether *result* and *value* may sit on the same buffer, cheaply.

    ``may_share_memory`` is a bounds check, not the exact analysis, so it costs
    nothing and errs toward saying yes -- which for a warning is the right
    direction.
    """
    try:
        import numpy as _np
    except ImportError:
        return False
    if not isinstance(result, _np.ndarray) or not isinstance(value, _np.ndarray):
        return False
    return bool(_np.may_share_memory(result, value))


def make_opaque_issue(func_name: str, opaque_list: str) -> Any:
    """Build a synthetic `PurityIssue` for opaque callees
    encountered in ``strict`` mode. Defined at module scope so the
    ``_surface_purity`` import stays local."""

    return PurityIssue(
        kind=ISSUE_IMPURE_CALL,
        description=f"opaque callees (strict): {opaque_list}",
        where=func_name,
        line=0,
    )


def format_issues_summary(func_name: str, issues: list[Any]) -> str:
    """Pretty-print a list of `PurityIssue` records, grouped
    by their ``where`` field. Used by both the warning body and the
    strict-mode exception body so users get the same diagnostic.
    """
    by_where: dict[str, list[Any]] = {}
    for i in issues:
        by_where.setdefault(i.where, []).append(i)
    lines = []
    for where in sorted(by_where):
        # The defining file, so a finding in a helper names the helper's
        # module -- the warning's own header names the CALL site's file.
        filename = next((getattr(i, "filename", "") for i in by_where[where] if getattr(i, "filename", "")), "")
        lines.append(f"  in {where} ({filename}):" if filename else f"  in {where}:")
        for issue in by_where[where]:
            line_part = f"line {issue.line}: " if issue.line else ""
            lines.append(f"    {line_part}[{issue.kind}] {issue.description}")
    return "\n".join(lines)


def static_effect_kinds(report: Any) -> set[str]:
    """The observed-effect kinds (``EffectObserver``) a static report names.

    So an effect the static warning already listed is not repeated by
    IMPURE-OBSERVED-EFFECTS, while one of another kind still is. Errs toward
    NOT covering: a finding with no effect kind covers nothing.
    """
    kinds: set[str] = set()
    for issue in getattr(report, "issues", ()) or ():
        if "changes the argument" in getattr(issue, "description", ""):
            kinds.add("argument mutation")
        if getattr(issue, "kind", None) not in (ISSUE_IMPURE_CALL, ISSUE_NETWORK_READ):
            continue
        label = observed_label(getattr(issue, "effect_kind", None))
        if label is not None:
            kinds.add(label)
    return kinds
