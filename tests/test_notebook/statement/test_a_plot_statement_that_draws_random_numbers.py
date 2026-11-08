"""What a statement of a plot run did with the global random streams and the
helper memos, as ``statement_plot_facts`` reads it.

A stream it seeded with a value before drawing from it ends where its code
alone puts it, so a served plot run may put it there. One it drew from
without seeding first is a read: the plot run is served only while that
stream stands where it did. A seed from the system's entropy, a draw from a
stream with no lineage cash follows, and a memo change no entry can carry
keep the plot run from being kept.
"""

from __future__ import annotations

from types import SimpleNamespace

from cash.notebook.plot_outputs import statement_plot_facts


def _run(**fields):
    base = {
        "metrics": {"uncacheable_reasons": []},
        "outputs": set(),
        "env_reads": {},
        "state_modules": (),
        "process_state": (),
        "rebound_modules": (),
        "callee_globals": frozenset(),
    }
    base.update(fields)
    return SimpleNamespace(**base)


def _execution(**fields):
    base = {
        "written_paths": (),
        "waived_paths": (),
        "unseen_writer": False,
        "children_running": False,
        "accessed_files": (),
        "accessed_remote": (),
        "rng_moved": {},
        "rng_reseeded": frozenset(),
    }
    base.update(fields)
    return SimpleNamespace(**base)


def _facts(run=None, execution=None, *, drew=(), seeded_first=(), unplaced=False):
    return statement_plot_facts(
        run or _run(), execution or _execution(), drew=drew, seeded_first=seeded_first, unplaced=unplaced
    )


def test_a_draw_after_its_own_seed_is_no_read_and_keeps_where_it_left_the_stream():
    execution = _execution(rng_moved={"random": "after"}, rng_reseeded=frozenset({"random"}))
    facts = _facts(execution=execution, drew={"random"}, seeded_first={"random"})
    assert facts.blockers == ()
    assert facts.rng_read == frozenset()
    assert facts.rng_state == {"random": "after"}
    assert facts.rng_reseeded == {"random"}


def test_a_draw_without_a_seed_first_is_a_read():
    facts = _facts(execution=_execution(rng_moved={"random": "after"}), drew={"random"})
    assert facts.blockers == ()
    assert facts.rng_read == {"random"}


def test_a_seed_other_than_before_its_draws_is_not_kept():
    execution = _execution(rng_moved={"random": "after"}, rng_reseeded=frozenset({"random"}))
    facts = _facts(execution=execution, drew={"random"})
    assert facts.blockers


def test_a_draw_from_a_stream_with_no_lineage_is_not_kept():
    facts = _facts(execution=_execution(rng_moved={"torch": "after"}), drew={"torch"})
    assert facts.blockers


def test_a_read_of_an_unplaced_stream_is_not_kept():
    facts = _facts(execution=_execution(rng_moved={"random": "after"}), drew={"random"}, unplaced=True)
    assert facts.blockers
    placed = _facts(
        execution=_execution(rng_moved={"random": "after"}, rng_reseeded=frozenset({"random"})),
        drew={"random"},
        seeded_first={"random"},
        unplaced=True,
    )
    assert placed.blockers == (), "a seed before the draw places the stream"


def test_the_memos_it_changed_travel_with_it():
    records, changes = [{"where": ["f"], "delta": 0}], [{"emptied": False, "pairs": [(1, 2)]}]
    facts = _facts(run=_run(plot_memos=(records, changes)))
    assert facts.blockers == ()
    assert facts.memos == (records, changes)


def test_a_memo_change_no_entry_can_carry_is_not_kept():
    facts = _facts(run=_run(plot_memos="the helper memo f was filled in a way cash cannot put back"))
    assert facts.blockers == ("the helper memo f was filled in a way cash cannot put back",)
    assert facts.memos is None
