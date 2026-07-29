"""A sub-unit hit must not erase the statement's file dependency (CAS-243 Task 7).

The call runs inside the statement's ambient ``FileAccessTracker`` -- the
same one ``@cash.cache`` already defends via ``core.py``'s
``_propagate_file_deps_to_active_tracker``. Exactly one case is broken: the
statement misses and re-executes, but the CALL inside it hits, so the file
read the call depends on does not re-happen this run. Without this task's
fix the statement's own tracker never sees it, so the statement's own entry
is (re)written missing a dependency it still transitively has.

**Why three runs, not two** (the two-run shape cannot fail -- see the task
brief): run 2 must force the STATEMENT to miss on something the CALL does
NOT read, so the call genuinely hits while the statement recomputes. Editing
`k` (an argument to the call) would force the CALL to miss too -- a
"both miss" run, which is already correct with no fix needed and proves
nothing. Editing the accumulator's *initial* value instead
(`total = 0` -> `total = 5`) changes what the AugAssign statement reads
(`total`'s lineage) without changing anything the call reads (`k`), which is
exactly the shape the brief specifies: "change something the statement reads
but the call does not".

    run 1: total=0, k=2 -> statement miss, call miss  -> both record data.csv
           TOTAL 20  (0 + 10*2)
    run 2: total's initial value edited to 5 (k unchanged) -> the AugAssign
           statement MISSES (total's lineage changed) and re-executes, but
           the CALL's own key (source + k's lineage) is untouched -> call
           HITS, file never re-read this run.
           TOTAL 25  (5 + 10*2, the call's still-correct cached value)
    run 3: data.csv changed on disk, cell re-run unchanged -> the AugAssign
           statement's OWN key still matches run 2's entry, so whether it
           re-executes depends ENTIRELY on whether the statement's own
           recorded file deps include data.csv -- which they only do if run
           2's call hit contributed its dependency back to the statement's
           ambient tracker. Without that contribution nothing invalidates
           and run 2's stale total (25) is served again -- the bug.
           TOTAL 205  (5 + 100*2, once the fix makes both the statement AND
           the call itself notice the file changed)

**Verified one-line mutation** (this is the one that actually distinguishes
pass/fail at the full-notebook level -- see the report for why
``CallUnit._replay_deps`` alone does not): in
``CallUnit._store``, change ``if file_deps or remote_deps:`` to
``if False and (file_deps or remote_deps):``, so a call's own
``auto_file_deps`` snapshot is never written at all. Applied and observed:
run 3 prints ``TOTAL 25`` (the stale run-2 value) instead of ``TOTAL 205``
-- verified, then reverted. Re-validating a LOCAL file's freshness itself
performs a real, tracked read (``file_dep_is_fresh`` -> ``file_content_hash``
-> ``open()``, through the same monkey-patched ``open`` the ambient tracker
observes), so at this end-to-end level the freshness re-check's own side
effect already re-registers the dependency whenever one was recorded at
store time -- ``CallUnit._replay_deps`` in isolation is verified directly,
and for the un-masked remote channel, in
``tests/test_notebook/test_call_unit_ambient_capture.py``.
"""
from __future__ import annotations

SETUP = """\
import cash
%cash_on
"""

DEFS = """\
import time
from pathlib import Path
Path('data.csv').write_text('10')

def load_and_scale(k):
    time.sleep(0.3)
    return int(Path('data.csv').read_text()) * k
"""


def _cell(total_init: int) -> str:
    return (
        f"total = {total_init}\n"
        "# @cash:cache-calls\n"
        "total += load_and_scale(k)\n"
        "print('TOTAL', total)"
    )


def test_sub_unit_hit_preserves_the_statements_file_dep(nb_runner, tmp_path):
    nb_runner.create_notebook([
        SETUP,
        DEFS,
        "k = 2",
        _cell(0),
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "TOTAL 20" in nb_runner.get_output(4)

    # Force the STATEMENT to miss (its `total` input's lineage changes)
    # while the CALL -- same source, same `k` -- still hits.
    nb_runner.set_cell_source(4, _cell(5))
    nb_runner.run_cells([4])
    assert "TOTAL 25" in nb_runner.get_output(4)

    # Now change the file. The statement must NOT restore a stale total --
    # not because it recomputes blindly, but because the dependency the call
    # observed on the (fresh) first run was correctly carried onto the
    # statement's own entry when the call hit on the second.
    (tmp_path / "data.csv").write_text("100")
    nb_runner.run_cells([4])
    assert "TOTAL 205" in nb_runner.get_output(4), (
        "the file dependency was lost when the sub-call hit"
    )
