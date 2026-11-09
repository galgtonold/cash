"""A move no digest checked follows the lineage a digest kept; a move a
digest saw change does not.

A run whose digests saw a holder unchanged keeps its lineage and records
the move it would have taken as an alias of it, which the upstream
simulation resolves. A later run of the same statement that took no digest
(its share check out of time) moved the holder on all the same, though the
simulation kept it: the next cell re-ran everything above that read it
(E6, homl ch02: ``cv_res.sort_values(..., inplace=True)`` moved
the pipelines the search holds, 130 s of refits on a same-code Run All).
Now such a run moves it as the simulation does; one whose digests saw it
change still moves it on.
"""

from __future__ import annotations

from cash.notebook.lineage_formula import held_lineage, keep_held_lineage
from cash.notebook.statement.derivation_edges import bump_derived_lineages

EDGES = {"grid_search": {"preprocessing"}, "preprocessing": {"grid_search"}}


def _walk(**kwargs) -> dict[str, str]:
    """``grid_search`` changed in place, from the same lineages each time."""
    lineage = {"grid_search": "G", "preprocessing": "P"}
    bump_derived_lineages(
        EDGES,
        lineage,
        {"grid_search"},
        {"grid_search"},
        record=lineage.__setitem__,
        present=lambda name: True,
        **kwargs,
    )
    return lineage


def test_a_walk_with_no_digest_keeps_what_a_digested_walk_kept():
    assert _walk(kept={"preprocessing"})["preprocessing"] == "P"

    assert _walk()["preprocessing"] == "P"


def test_a_walk_whose_digest_saw_a_change_moves_on():
    _walk(kept={"preprocessing"})

    assert _walk(moved_on={"preprocessing"})["preprocessing"] != "P"


def test_with_no_digest_ever_taken_the_partner_moves_on():
    assert _walk()["preprocessing"] != "P"


def test_a_holder_moves_as_the_digest_that_kept_it_says():
    keep_held_lineage("preprocessing", "P", "key")

    assert held_lineage("preprocessing", "P", "key") == "P"
    assert held_lineage("preprocessing", "P", "key", resolve=False) != "P"


def test_a_name_kept_keeps_no_other_name_of_the_same_lineage():
    """``c`` and ``orders``, both bound by the loop that fills ``orders``,
    have one lineage. ``customers[0].rules.append(f)`` leaves ``c``, the
    last customer, as it was, and changes ``orders``, which holds
    ``customers[0]``: keeping ``c`` must not keep ``orders`` -- in the run,
    nor in the simulation, which takes no digest."""
    edges = {"customers": {"c", "orders"}, "c": {"customers"}, "orders": {"customers"}}

    def walk(**kwargs) -> dict[str, str]:
        lineage = {"customers": "C", "c": "L", "orders": "L"}
        bump_derived_lineages(
            edges,
            lineage,
            {"customers"},
            {"customers"},
            record=lineage.__setitem__,
            present=lambda name: True,
            **kwargs,
        )
        return lineage

    run = walk(kept={"c"})
    assert run["c"] == "L"
    assert run["orders"] != "L"

    simulated = walk()
    assert simulated["c"] == "L"
    assert simulated["orders"] == run["orders"]
