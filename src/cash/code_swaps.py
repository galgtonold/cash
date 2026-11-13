"""Code a key folded that was replaced while the call ran.

Some libraries put a function's real code in place on its first call:
networkx's ``argmap`` starts every function it decorates as a stub that
only calls the decorator, and the first call sets the stub's ``__code__``
to generated code that runs the decorated body. A key built before that
call folded the stub, which leaves out what the body reads, and the body
then ran. Stored, that entry is found again by every process that starts
with the same stub, after the body or what it reads changed.

The key build notes what code it folded (`FoldedCode`): the helpers of the
purity reports it keyed (``walked_code`` names them) and the functions and
classes whose code surface it digested, each with the code it runs then.
After the body the code is compared again, by identity alone
(`FoldedCode.swapped`). A call whose code moved is returned but not stored,
and the next call keys the code that is there now.
"""

from __future__ import annotations

import contextvars
import operator
import types
from typing import Any

from .loaded_code import code_objects_of

__all__ = ["FOLDED_CODE", "FoldedCode", "note_owner", "note_report", "settable"]


class FoldedCode:
    """The code one key build folded, kept to be compared after the body."""

    __slots__ = ("owners", "reports")

    def __init__(self) -> None:
        #: id -> (function or class, the code objects it ran when folded).
        self.owners: dict[int, tuple[Any, tuple]] = {}
        #: id -> purity report whose helpers are among the owners (held, so
        #: its id stays its own while the build runs).
        self.reports: dict[int, Any] = {}

    def swapped(self) -> Any | None:
        """A function or class that runs other code than when the key was
        built, or None. Identity compares only: no hashing, no source."""
        for owner, codes in self.owners.values():
            if _moved(owner, codes):
                return owner
        return None


def settable(owner: Any) -> bool:
    """Does *owner* run code a plain ``__code__`` assignment replaces: a
    function, or a class through its methods? Not a proxy (``wrapt``) whose
    ``__code__`` is a new stand-in object on every read."""
    return type(owner) is types.FunctionType or isinstance(owner, type)


def _moved(owner: Any, codes: tuple) -> bool:
    now = code_objects_of(owner)
    return len(now) != len(codes) or not all(map(operator.is_, now, codes))


#: The `FoldedCode` of the real call's key being built: set by
#: `KeyBuilder.resolve`, None for explain() and for the state rebuilt after
#: the body, which store nothing.
FOLDED_CODE: contextvars.ContextVar[FoldedCode | None] = contextvars.ContextVar("_cash_folded_code", default=None)


def note_report(report: Any) -> None:
    """The key being built folded the helpers *report* walked, as they are
    now: the code a report recorded when it walked a helper may be older
    than the code the key reads live (`DependencyStateHasher.compute`)."""
    folded = FOLDED_CODE.get()
    if folded is None or report is None or id(report) in folded.reports:
        return
    folded.reports[id(report)] = report
    owners = folded.owners
    for ref, _codes in getattr(report, "walked_code", ()) or ():
        owner = ref()
        if owner is not None and id(owner) not in owners and settable(owner):
            codes = code_objects_of(owner)
            if codes:
                owners[id(owner)] = (owner, codes)


def note_owner(owner: Any, codes: tuple) -> None:
    """The key being built folded *owner*'s code, which was *codes*."""
    folded = FOLDED_CODE.get()
    if folded is not None and codes and settable(owner):
        folded.owners.setdefault(id(owner), (owner, codes))
