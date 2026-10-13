"""Why a call in a loop missed is told against the same iteration of an
earlier run, not against the iteration before it.

A grid ``for k in ...: for hfa in ...: h, _ = elo.run_elo(played, k=k,
hfa=hfa)`` re-run after a restart said ``0/9 hit - changed: hfa``: each
iteration's call was compared with the previous iteration's, whose ``hfa``
differs by design. ``hfa`` took the same three values as the run before; the
reason named the loop doing its job.
"""

import ast

from cash.notebook.cache_key import CacheKeyContext
from cash.notebook.call_interception import wrap_eligible_calls
from cash.notebook.call_key import CallKeys

GRID = "h = run_elo(played, k=k, hfa=hfa)"  # the body of `for hfa in values:`


def _keys(lineage: dict, loop: dict) -> CallKeys:
    return CallKeys(
        lambda: CacheKeyContext(variable_lineage=lineage, user_ns={}),
        loop_vars_provider=lambda: dict(loop),
    )


def _site():
    _, sites = wrap_eligible_calls(ast.parse(GRID))
    return sites[0]


def _iteration(keys: CallKeys, lineage: dict, loop: dict, hfa: int):
    # A loop variable's lineage moves with its value, as the loop records it.
    loop["0:hfa"], lineage["hfa"] = hfa, f"h{hfa}"
    keys.begin_statement()
    keys.key(_site(), (None,), {"k": 20, "hfa": hfa})
    return keys.why_missed(_site())


def test_the_next_iteration_is_not_a_change():
    lineage = {"played": "p1", "run_elo": "r", "k": "k"}
    loop: dict = {}
    keys = _keys(lineage, loop)

    assert [_iteration(keys, lineage, loop, hfa) for hfa in (40, 60, 80)] == [None, None, None]


def test_a_rerun_names_what_moved_for_the_same_iteration():
    lineage = {"played": "p1", "run_elo": "r", "k": "k"}
    loop: dict = {}
    keys = _keys(lineage, loop)
    for hfa in (40, 60, 80):
        _iteration(keys, lineage, loop, hfa)

    lineage["played"] = "p2"

    assert [_iteration(keys, lineage, loop, hfa) for hfa in (40, 60, 80)] == ["changed: played"] * 3
