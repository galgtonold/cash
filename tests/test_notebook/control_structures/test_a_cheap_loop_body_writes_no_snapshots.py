"""A cheap loop body does not store a snapshot of what it changes, per iteration.

A 631-iteration loop setting a column slice of a 3130x800
bool frame took 0.05 s plain and 11-23 s under cash (180 s once in their
notebook). After one run the RAM tier held 1,265 entries -- one per iteration
for ``members.loc[...] = True``, each a full 2.4 MB snapshot of ``members``
(1.5 GiB in all), plus one per iteration for the cheap ``end = ...`` -- and
a re-run deep-copied every snapshot back.

The statements take ~0.1 ms, below the "too cheap to cache" floor. They got
past it through ``RebuildCostLedger.final_over_costly_inputs``, which gives a cheap FINAL value
over costly inputs an entry. Inside a loop nothing is final
-- the next iteration overwrites it -- and the inputs' unsaved cost only grows
with every iteration, so every iteration qualified.
"""

from tests._cell_driver import run_cash_cell

SETUP = (
    "import pandas as pd, numpy as np\n"
    "idx = pd.date_range('2013-01-01', periods=300)\n"
    "cols = [f's{i}' for i in range(80)]\n"
    "uni = pd.DataFrame({'s': [f's{i % 80}' for i in range(120)],\n"
    "                    'j': [idx[(i * 7) % 290] for i in range(120)],\n"
    "                    'l': [pd.NaT] * 120})"
)
LOOP = (
    "members = pd.DataFrame(False, index=idx, columns=cols)\n"
    "for s, j, l in uni.itertuples(index=False):\n"
    "    if s in members.columns:\n"
    "        end = l if pd.notna(l) else members.index[-1] + pd.Timedelta(days=1)\n"
    "        members.loc[(members.index >= j) & (members.index < end), s] = True"
)


def _entries(cash_magics):
    backend = cash_magics._cash_instance.backend
    return sum(len(getattr(t, "_store", {}) or {}) for t in getattr(backend, "backends", [backend]))


def _expected():
    ns: dict = {}
    exec(SETUP + "\n" + LOOP, ns)
    return int(ns["members"].values.sum())


def test_a_cheap_loop_body_stores_no_entry_per_iteration(cash_magics):
    run_cash_cell(cash_magics, SETUP)
    before = _entries(cash_magics)
    run_cash_cell(cash_magics, LOOP)
    written = _entries(cash_magics) - before
    assert written < 20, (
        f"{written} entries written for a 120-iteration loop whose body costs "
        "~0.1 ms a statement; each is a snapshot of the frame it changes"
    )
    assert int(cash_magics.shell.user_ns["members"].values.sum()) == _expected()


def test_the_loop_is_still_right_when_run_again(cash_magics):
    run_cash_cell(cash_magics, SETUP)
    run_cash_cell(cash_magics, LOOP)
    run_cash_cell(cash_magics, LOOP)
    assert int(cash_magics.shell.user_ns["members"].values.sum()) == _expected()


def test_a_long_cheap_itertuples_loop_runs_as_one_unit(cash_magics):
    """Even with no snapshots, that loop cost ~5 ms of per-statement
    machinery per statement against ~0.1 ms of work: 9 s for 0.07 s. The
    handler already runs a long cheap loop as ONE unit -- but it sized the loop
    with len(), which a DataFrame.itertuples() iterator does not have, and it
    refused to re-evaluate a header whose value is a one-shot iterator, which
    itertuples() makes afresh every time it is called."""
    from cash.notebook.control_structures import single_unit_policy

    calls = []
    orig = single_unit_policy.should_run_as_single_unit

    def spy(node, iterable, user_ns, **kwargs):
        result = orig(node, iterable, user_ns, **kwargs)
        calls.append(result and single_unit_policy.header_safe_to_reevaluate(node.iter, iterable, user_ns))
        return result

    single_unit_policy.should_run_as_single_unit = spy
    try:
        run_cash_cell(cash_magics, SETUP)
        run_cash_cell(cash_magics, LOOP)
    finally:
        single_unit_policy.should_run_as_single_unit = orig
    assert calls and calls[0] is True, calls
    assert int(cash_magics.shell.user_ns["members"].values.sum()) == _expected()


def test_a_stored_iterator_still_never_runs_twice(cash_magics):
    """The safety the re-evaluation check exists for: a Name bound to a
    generator yields nothing the second time."""
    run_cash_cell(cash_magics, "rows = iter(range(200))")
    run_cash_cell(cash_magics, "out = []\nfor r in rows:\n    out.append(r * 2)")
    assert cash_magics.shell.user_ns["out"] == [r * 2 for r in range(200)]
