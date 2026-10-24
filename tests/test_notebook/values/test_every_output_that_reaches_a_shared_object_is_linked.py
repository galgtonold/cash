"""Every output that reaches an object a variable outside the outputs holds
is linked to that variable, not only the first one walked.

The share check named, for each object held from outside, the output its
walk reached it from first. The outputs are a set of names, so which one
that was followed the hash seed: of ``draw, rebind = make(BIG)``, two
closures over one cell, only one was linked to ``BIG``, and a change to
``BIG`` below moved that one alone. Where the walks of two outputs meet,
both are named now.
"""

from __future__ import annotations

import pytest

from cash.notebook import shared_objects


def _closures():
    def make(rows):
        def draw(k):
            return len(rows)

        def rebind(r):
            nonlocal rows
            rows = r

        return draw, rebind

    big = [{"id": i} for i in range(2000)]
    draw, rebind = make(big)
    return {"draw": draw, "rebind": rebind, "BIG": big}, ["draw", "rebind"]


def _alias():
    big = [{"id": i} for i in range(2000)]
    return {"alias": big, "BIG": big, "other": [big]}, ["alias", "BIG"]


@pytest.mark.parametrize("made", [_closures, _alias], ids=["two closures over one cell", "an alias"])
@pytest.mark.parametrize("reverse", [False, True], ids=["in order", "reversed"])
def test_each_output_is_linked_whichever_is_walked_first(made, reverse):
    ns, names = made()
    gave_up = shared_objects.GaveUp()
    outputs = {name: ns[name] for name in names}  # what the statement bound, as the processor hands it
    shared_objects.share_group(names[::-1] if reverse else names, outputs, ns, gave_up=gave_up)
    linked = set().union(*(owners for owners, _holders in gave_up.links))
    assert linked == set(names)


def test_an_output_that_meets_no_other_is_not_linked():
    big = [{"id": i} for i in range(2000)]
    ns = {"held": big, "apart": [{"id": 1}], "other": [big]}
    gave_up = shared_objects.GaveUp()
    outputs = {name: ns[name] for name in ("held", "apart")}
    shared_objects.share_group(["held", "apart"], outputs, ns, gave_up=gave_up)
    assert set().union(*(owners for owners, _holders in gave_up.links)) == {"held"}
