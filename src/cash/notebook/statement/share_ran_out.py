"""The share checks that ran out of their budget, to refuse the same check
again without walking it (`ShareRanOut`).

A statement's share check (`StatementProcessor._refuse_unrestorable_outputs`)
walks everything its outputs hold, and the notebook's variables when one of
those objects is held elsewhere, within a budget of CPU time. On a big
fitted model a tiny statement changes in place (``voting_clf.voting =
'soft'`` on a 500-tree ensemble) the walk runs out of its 0.1 s every time,
and the statement is refused as re-running every time, after the walk.

The walk's cost is set by what it walks: the outputs' objects and their
content, and the variables of the notebook it may climb through. When the
very check was measured running out of a budget at least as large as this
one, over the same objects with the same content (their content digests,
taken as the statement's outputs were recorded), while the notebook binds
the same objects to the same names with the same content (or lineage, for a
value without a content digest), the walk would do the same work again and
run out again: the same refusal, given at once, and the same search from
the outputs up for the variables to link that follows a walk run out. A
check with any of these different is walked as before.

Nothing is held of a user's object here: ids, digests and lineages only.
The ids stand for an object only beside its content digest, so an object
made at a freed one's address with other content is walked.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

__all__ = ["Bindings", "Outputs", "ShareRanOut", "other_bindings", "outputs_bindings"]

#: ``((name, id of its value, content digest), ...)``, sorted: the outputs.
Outputs = tuple[tuple[str, int, str], ...]
#: ``((name, id of its value, session hash or lineage), ...)``, sorted: the
#: notebook's other variables.
Bindings = tuple[tuple[str, int, str | None], ...]


class ShareRanOut:
    """The share checks measured running out of their budget, by the
    objects and content they walked.

    *limit* is how many are kept, the oldest let go first."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        #: ``outputs -> (budget run out of, the notebook's other bindings)``.
        self._ran_out: dict[Outputs, tuple[float, Bindings]] = {}

    def __len__(self) -> int:
        return len(self._ran_out)

    def note(
        self,
        outputs: Outputs | None,
        budget: float,
        others: Callable[[], Bindings],
    ) -> None:
        """The check over *outputs* (`outputs_bindings`) ran out of *budget*;
        *others* gives the notebook's other bindings now (`other_bindings`).
        Nothing is noted for outputs whose content has no digest (None)."""
        if outputs is None:
            return
        ran_out = self._ran_out
        ran_out.pop(outputs, None)
        if len(ran_out) >= self._limit:
            ran_out.pop(next(iter(ran_out)))
        ran_out[outputs] = (budget, others())

    def known(self, outputs: Outputs | None, budget: float, others: Callable[[], Bindings]) -> bool:
        """Whether the check over *outputs* within *budget* is one measured
        running out of a budget at least as large, over the same objects with
        the same content, while the notebook bound what *others* gives now."""
        if outputs is None:
            return False
        entry = self._ran_out.get(outputs)
        if entry is None:
            return False
        ran_out_of, bound = entry
        return budget <= ran_out_of and others() == bound


def outputs_bindings(
    names: Iterable[str],
    values: Mapping[str, Any],
    session: Mapping[str, str],
    lineage: Mapping[str, str],
) -> Outputs | None:
    """The outputs *names* a share check walks, each by its object's id and
    content digest (*session*); None when one has no content digest -- none
    taken, or its *lineage* standing in for its content (a value too costly
    to read) -- so the content cannot be told the same. Not by lineage: a
    change in place moves it on even when it puts back what was there
    (``voting = 'hard'``, then ``'soft'`` again)."""
    out = []
    for name in names:
        value = values.get(name)
        if value is None:
            continue
        digest = session.get(name)
        if digest is None or digest == lineage.get(name):
            return None
        out.append((name, id(value), digest))
    if not out:
        return None
    return tuple(sorted(out))


def other_bindings(
    user_ns: Mapping[str, Any],
    skip: Callable[[str], bool],
    session: Mapping[str, str],
    lineage: Mapping[str, str],
) -> Bindings:
    """Every name of *user_ns* but those *skip* names (IPython's own, the
    outputs), with the id of its object and its content: its session hash
    (*session*), which follows a change cash sees through another name as
    its lineage does (`replay_derivation_bumps`); its *lineage* where it has
    none. By content where there is one, as an output is: a forest a voting
    classifier holds moves on with every change of the classifier, as
    another name for what it may have changed, and is the same forest."""
    return tuple(
        sorted(
            (name, id(value), session.get(name) or lineage.get(name))
            for name, value in list(user_ns.items())
            if not skip(name)
        )
    )
