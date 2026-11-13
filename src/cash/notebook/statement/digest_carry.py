"""A holder's in-place-edit digest, carried from one statement to the next.

Each statement that may change an object in place digests the variables
sharing it (`StatementProcessor._holder_digests`) before it runs and again
after (`_digested_unchanged`). In a cell of small statements over one big
object (``voting_clf.voting = 'soft'``, ``voting_clf.set_params(...)``,
``est = voting_clf.estimators_``), every statement digested the same
holders twice, and the digest one took after it ran was the very digest the
next took before it ran: nothing ran between the two.

`DigestCarry` hands the digests taken after one statement to the next one's
digests before it, only where no code can have changed the value between:

* the next statement is the next one of the same cell, run as a statement
  of its own with nothing of the cell's between (no loop, no restore of the
  plan, no plot served; the cell executor says so), and no other statement
  ran in between (a nested one);
* no other thread could run (`other_thread_alive`), and no work a statement
  left running (a process writing shared memory) is still running, when the
  digests were taken, when they were kept and when they are taken up;
* the variable is still bound to the same object, held from when the digest
  was kept until the next statement starts, with the same lineage;
* the next statement was not served from the cache, nor tried to be: a
  restore may run code of the user's.

Anything else is digested again. A statement that awaits runs on the event
loop, where other tasks run between: it neither hands on nor takes up.

A digest carried forward can never hide a change: the holder reads as
unchanged only when its digest after the next statement is the one it had
after this one, under the same lineage, which is then true whatever ran
between. A change made between the two (one the checks above did not see
coming) reads as the next statement's own, and moves the holder on, as a
fresh digest would not. That is also why a garbage collection or a signal
handler between them does not end the carry: they run at a moment no code
chooses, and one that changes a holder there is taken as running as the
next statement starts, as it could have.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cash.tracking.trace_recorder import other_thread_alive

__all__ = ["DigestCarry"]

_MISSING = object()


class DigestCarry:
    """The holder digests one statement took after it ran, for the next.

    The statement collects them (`begin_after`, `note_after`), keeps them as
    it ends (`keep`), and the next one of the cell takes them up as it starts
    (`take_up`) and reads them where it would digest (`before`).
    """

    def __init__(self) -> None:
        #: The statement collects digests: nothing else could run as it
        #: began to (`begin_after`).
        self._collecting = False
        #: ``name -> (id, digest)`` collected after the statement ran: by
        #: id, as the share checks that follow count references.
        self._after: dict[str, tuple[int, str]] = {}
        #: Kept as the statement ended: ``name -> (value, digest, lineage)``,
        #: and how many statements had started then.
        self._kept: dict[str, tuple[Any, str, str | None]] = {}
        self._kept_at = -1
        #: Taken up by the statement running now, its values let go:
        #: ``name -> (id, digest, lineage)``.
        self._taken: dict[str, tuple[int, str, str | None]] = {}

    def held(self) -> list[dict[str, Any]]:
        """The mappings holding values of the user's, for the share check
        to count as cash's own."""
        return [self._kept]

    def drop(self) -> None:
        self._collecting = False
        self._after = {}
        self._kept = {}
        self._taken = {}

    @staticmethod
    def _quiet(namespace: Mapping[str, Any], still_running: str | None) -> bool:
        return still_running is None and not other_thread_alive(namespace)

    def begin_after(self, namespace: Mapping[str, Any], still_running: str | None) -> None:
        """The statement ran and its holders are about to be digested again:
        collect those digests when nothing else can run now."""
        self._after = {}
        self._collecting = self._quiet(namespace, still_running)

    def note_after(self, name: str, value: Any, digest: str) -> None:
        if self._collecting:
            self._after[name] = (id(value), digest)

    def keep(
        self,
        namespace: Mapping[str, Any],
        lineage: Mapping[str, str],
        started: int,
        still_running: str | None,
        last: bool = False,
    ) -> None:
        """The statement ended (*started* statements so far): keep what it
        collected for the next statement, when nothing could have run since
        it was collected, each value held until then (its id stays its own).
        Nothing is kept from a cell's *last* statement: nothing follows it
        in the cell, and the values would be held past the cell."""
        after, collected = self._after, self._collecting
        self.drop()
        if last or not after or not collected or not self._quiet(namespace, still_running):
            return
        kept = {}
        for name, (known, digest) in after.items():
            value = namespace.get(name, _MISSING)
            if id(value) == known:
                kept[name] = (value, digest, lineage.get(name))
        self._kept = kept
        self._kept_at = started

    def take_up(
        self,
        follows: bool,
        namespace: Mapping[str, Any],
        lineage: Mapping[str, str],
        started: int,
        still_running: str | None,
    ) -> None:
        """A statement starts (*started* statements so far): take up what the
        one before kept when it *follows* it with nothing between, and let
        the values go either way (the share checks count references)."""
        kept, at = self._kept, self._kept_at
        self.drop()
        if not kept or not follows or at != started - 1 or not self._quiet(namespace, still_running):
            return
        self._taken = {
            name: (id(value), digest, held_lineage)
            for name, (value, digest, held_lineage) in kept.items()
            if namespace.get(name) is value and lineage.get(name) == held_lineage
        }

    def give_up(self) -> None:
        """The statement may have run code of the user's since it started
        (a restore was tried): nothing taken up is used."""
        self._taken = {}

    def before(self, name: str, value: Any, lineage: str | None) -> str | None:
        """The digest carried for *name* bound to *value* at *lineage*, or
        None: digest it."""
        found = self._taken.get(name)
        if found is None or found[0] != id(value) or found[2] != lineage:
            return None
        return found[1]

    def end_before(self) -> None:
        """The statement's digests before it ran are taken."""
        self._taken = {}
