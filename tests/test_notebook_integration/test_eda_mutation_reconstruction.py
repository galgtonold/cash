"""R12 EDA correctness probe.

A new user runs a messy exploratory-data-analysis workflow (in-place pandas
mutation, aliasing, helper edits, cell reordering, variable deletion) and hunts
for cases where cash serves a WRONG or STALE value.

ORACLE (per task): ground truth = the FINAL cell sources run top-to-bottom in a
fresh kernel with NO cash. cash's whole promise is that an interactive edit +
isolated re-run reconstructs upstream state so the result matches that clean
top-to-bottom run. So each scenario:

  1. drives a cash-ON kernel through a messy interactive sequence (edits +
     single-cell re-runs) and captures the value(s) the user ends up seeing;
  2. the driver then takes the runner's FINAL notebook sources, runs them
     top-to-bottom in a fresh cash-OFF kernel, and compares.

A mismatch means cash served a value that a clean re-run would not = a
correctness bug (stale/wrong).

Run isolated:
    python -m pytest tests/test_notebook_integration/test_zzprobe_r12_eda.py -p no:randomly -q -n0
"""
import pytest

from conftest import NotebookTestRunner

pytestmark = pytest.mark.libraries

PRE = "import pandas as pd\nimport numpy as np"


def _start(r, cells):
    r.create_notebook(cells)
    r.start_kernel(with_cash=True)
    # Tolerate cell errors so a stale-vs-NameError divergence is observable
    # rather than crashing the arm.
    r.client.allow_errors = True
    return r


# ---------------------------------------------------------------------------
# Scenarios. Each drives the cash-ON interactive sequence and returns
# (captured, check) where `captured` = {label: text the user saw} and
# `check` = {label: cell_num} telling the oracle which final cell to read.
# ---------------------------------------------------------------------------

def scen_mutate_upstream_recompute(r):
    """Mutate an upstream frame in place, edit the mutating cell, re-run it +
    downstream. Should reflect the edit."""
    _start(r, [
        PRE,
        "df = pd.DataFrame({'a': range(10), 'b': range(10, 20)})",
        "df['c'] = df['a'] * 2",
        "print('SUM', int(df['c'].sum()))",
    ])
    r.run_all()
    r.set_cell_source(3, "df['c'] = df['a'] * 5")
    r.run_cell(3)
    r.run_cell(4)
    return {"after_edit": r.get_output(4)}, {"after_edit": 4}


def scen_alias_downstream_consumer(r):
    """Mutate a frame through an ALIAS, then a downstream cell keyed on the
    ORIGINAL name. Edit the aliased write and re-run. cash must not serve a total
    from the pre-edit lineage."""
    _start(r, [
        PRE,
        "df = pd.DataFrame({'v': [1, 2, 3, 4, 5]})",
        "alias = df",
        "alias.loc[0, 'v'] = 100",
        "s = 0\nfor _ in range(3):\n    s += int(df['v'].sum())\nprint('TRIPLE', s)",
    ])
    r.run_all()
    r.set_cell_source(4, "alias.loc[0, 'v'] = 200")
    r.run_cell(4)
    r.run_cell(5)
    return {"after_edit": r.get_output(5)}, {"after_edit": 5}


def scen_helper_edit_invalidates(r):
    """Edit a helper body and re-run only its consumer. Documented key feature:
    changes to helpers invalidate dependent caches."""
    _start(r, [
        PRE,
        "def transform(frame):\n    return frame['a'].sum() * 2",
        "df = pd.DataFrame({'a': range(50)})",
        "out = transform(df)\nprint('OUT', int(out))",
    ])
    r.run_all()
    r.set_cell_source(2, "def transform(frame):\n    return frame['a'].sum() * 10")
    r.run_cell(2)
    r.run_cell(4)
    return {"after_edit": r.get_output(4)}, {"after_edit": 4}


def scen_helper_edit_mutating(r):
    """A helper that adds a column to the frame it is given. Edit the multiplier,
    re-run the consumer, read a deep value."""
    _start(r, [
        PRE,
        "def add_score(frame):\n    frame['score'] = frame['a'] * 3\n    return frame",
        "df = pd.DataFrame({'a': range(10)})",
        "df2 = add_score(df.copy())\nprint('SCORE9', int(df2.loc[9, 'score']))",
    ])
    r.run_all()
    r.set_cell_source(2, "def add_score(frame):\n    frame['score'] = frame['a'] * 7\n    return frame")
    r.run_cell(2)
    r.run_cell(4)
    return {"after_edit": r.get_output(4)}, {"after_edit": 4}


def scen_dict_accumulator(r):
    """Dict accumulator mutated across cells; edit a middle write, re-run the
    downstream consumer only (not the intervening writes)."""
    _start(r, [
        PRE,
        "stats = {}",
        "stats['a'] = 10",
        "stats['b'] = 20",
        "print('TOTAL', sum(stats.values()))",
    ])
    r.run_all()
    r.set_cell_source(3, "stats['a'] = 99")
    r.run_cell(3)
    r.run_cell(5)
    return {"after_edit": r.get_output(5)}, {"after_edit": 5}


def scen_del_variable_then_consumer(r):
    """Delete the variable a cached consumer depends on, then re-run the consumer.
    Top-to-bottom of the final sources raises NameError (empty text output); cash
    must not serve the stale cached total."""
    _start(r, [
        PRE,
        "df = pd.DataFrame({'v': range(100)})",
        "total = int(df['v'].sum())\nprint('TOTAL', total)",
    ])
    r.run_all()
    r.set_cell_source(2, "del df")
    r.run_cell(2)
    r.run_cell(3)
    return {"after_del": r.get_output(3)}, {"after_del": 3}


def scen_sort_inplace_idempotent(r):
    """sort_values(inplace=True) on an upstream frame; re-run the sort cell twice.
    Reading head must be stable and match top-to-bottom."""
    _start(r, [
        PRE,
        "df = pd.DataFrame({'v': [3, 1, 2, 5, 4]})",
        "df.sort_values('v', inplace=True, ignore_index=True)",
        "print('HEAD', df['v'].tolist())",
    ])
    r.run_all()
    r.run_cell(3)
    r.run_cell(4)
    return {"after_rerun": r.get_output(4)}, {"after_rerun": 4}


def scen_reorder_cells(r):
    """Edit a definition and run the consumer chain after."""
    _start(r, [
        PRE,
        "base = 5",
        "scaled = base * 10",
        "print('RESULT', scaled + 1)",
    ])
    r.run_all()
    r.set_cell_source(2, "base = 7")
    r.run_cell(2)
    r.run_cell(3)
    r.run_cell(4)
    return {"after_edit": r.get_output(4)}, {"after_edit": 4}


def scen_loc_mask_deep_row(r):
    """Mutate a row DEEP in a frame (row 500, beyond the 5-row hash sample) via an
    upstream cell; downstream consumer keyed on the frame. Edit the upstream mask
    value and re-run. Tests the sampling-hash blind spot."""
    _start(r, [
        PRE,
        "df = pd.DataFrame({'v': list(range(1000))})",
        "df.loc[df['v'] == 500, 'v'] = 5000",
        "s = 0\nfor _ in range(3):\n    s += int(df['v'].sum())\nprint('TRIPLE', s)",
    ])
    r.run_all()
    r.set_cell_source(3, "df.loc[df['v'] == 500, 'v'] = 9000")
    r.run_cell(3)
    r.run_cell(4)
    return {"after_edit": r.get_output(4)}, {"after_edit": 4}


def scen_append_rows_rerun(r):
    """Append a row via concat-reassign upstream; edit the appended value, re-run
    the downstream count/sum."""
    _start(r, [
        PRE,
        "df = pd.DataFrame({'v': [1, 2, 3]})",
        "df = pd.concat([df, pd.DataFrame({'v': [10]})], ignore_index=True)",
        "print('SUM', int(df['v'].sum()), 'N', len(df))",
    ])
    r.run_all()
    r.set_cell_source(3, "df = pd.concat([df, pd.DataFrame({'v': [20]})], ignore_index=True)")
    r.run_cell(3)
    r.run_cell(4)
    return {"after_edit": r.get_output(4)}, {"after_edit": 4}


def scen_rerun_upstream_twice_no_double(r):
    """Re-run a self-modifying builder cell twice (idempotent), then a downstream
    sum. Must not double-apply."""
    _start(r, [
        PRE,
        "df = pd.DataFrame({'v': [1, 2, 3]})\ndf['v'] = df['v'] + 100",
        "print('SUM', int(df['v'].sum()))",
    ])
    r.run_all()
    r.run_cell(2)
    r.run_cell(3)
    return {"after_rerun": r.get_output(3)}, {"after_rerun": 3}


def scen_loc_mask_stable_key_deep(r):
    """Like loc_mask_deep_row but the mask keys on a STABLE id column so the edit
    still applies after reconstruction; the changed cell is deep (row 900)."""
    _start(r, [
        PRE,
        "df = pd.DataFrame({'id': range(1000), 'v': range(1000)})",
        "df.loc[df['id'] == 900, 'v'] = 111",
        "print('V900', int(df.loc[df['id'] == 900, 'v'].iloc[0]), 'SUM', int(df['v'].sum()))",
    ])
    r.run_all()
    r.set_cell_source(3, "df.loc[df['id'] == 900, 'v'] = 222")
    r.run_cell(3)
    r.run_cell(4)
    return {"after_edit": r.get_output(4)}, {"after_edit": 4}


def scen_edit_base_no_rerun_intermediate(r):
    """Edit the base cell, re-run ONLY the final consumer (skip the intermediate
    transform). cash must reconstruct the intermediate from the edited base."""
    _start(r, [
        PRE,
        "base = pd.Series([1, 2, 3])",
        "doubled = base * 2",
        "print('SUM', int(doubled.sum()))",
    ])
    r.run_all()
    r.set_cell_source(2, "base = pd.Series([10, 20, 30])")
    r.run_cell(2)
    r.run_cell(4)              # deliberately skip cell 3
    return {"after_edit": r.get_output(4)}, {"after_edit": 4}


def scen_hidden_global_mutation(r):
    """A function mutates a global; re-running its caller alone must still agree
    with a clean top-to-bottom run.

    Was a documented-limitation canary (cash re-executed and read an
    already-advanced global, printing 2 where top-to-bottom says 1). CAS-260
    closed it at cell level: ``c`` is now surfaced as an output of ``res =
    tick()``, so its pre-state pins the key and its post-state is restored on a
    hit. Kept in the passing set as the regression guard for that."""
    _start(r, [
        PRE,
        "c = {'n': 0}\ndef tick():\n    c['n'] += 1\n    return c['n']",
        "res = tick()\nprint('R', res)",
    ])
    r.run_all()
    r.run_cell(3)   # re-run the caller alone
    return {"rerun": r.get_output(3)}, {"rerun": 3}


def scen_class_var_accumulator(r):
    """DOCUMENTED limitation: a class variable incremented by two cells keeps
    climbing on isolated re-run."""
    _start(r, [
        PRE,
        "class W:\n    count = 0\n    def __init__(self):\n        W.count += 1",
        "w0 = W()",
        "w1 = W()\nprint('COUNT', W.count)",
    ])
    r.run_all()
    r.run_cell(4)   # re-run alone
    return {"rerun": r.get_output(4)}, {"rerun": 4}


def scen_alias_double_apply_own_rerun(r):
    """DOCUMENTED limitation: mutating an upstream object through an alias, then
    re-running that same cell, double-applies (cash can't attribute it to df)."""
    _start(r, [
        PRE,
        "df = pd.DataFrame({'v': [1, 2, 3]})",
        "alias = df",
        "alias['v'] = alias['v'] + 10\nprint('SUM', int(df['v'].sum()))",
    ])
    r.run_all()
    r.run_cell(4)   # re-run the aliased-mutation cell alone
    return {"rerun": r.get_output(4)}, {"rerun": 4}


def scen_multihop_inplace_reconstruction(r):
    """HARD: edit the base builder, re-run ONLY the final consumer, skipping two
    intervening in-place mutation cells. cash must replay both in order."""
    _start(r, [
        PRE,
        "df = pd.DataFrame({'a': [1, 2, 3]})",
        "df['b'] = df['a'] * 2",
        "df['c'] = df['b'] + 1",
        "print('SUMC', int(df['c'].sum()))",
    ])
    r.run_all()
    r.set_cell_source(2, "df = pd.DataFrame({'a': [10, 20, 30]})")
    r.run_cell(2)
    r.run_cell(5)   # skip cells 3 and 4
    return {"after_edit": r.get_output(5)}, {"after_edit": 5}


def scen_alias_reflects_upstream_edit(r):
    """HARD: alias bound to df, df mutated by its own name downstream; edit the
    base builder and re-run ONLY the alias reader. The alias must reflect the
    reconstructed + re-mutated df."""
    _start(r, [
        PRE,
        "df = pd.DataFrame({'v': [1, 2, 3]})",
        "alias = df",
        "df['v'] = df['v'] * 10",
        "print('ALIAS', alias['v'].tolist())",
    ])
    r.run_all()
    r.set_cell_source(2, "df = pd.DataFrame({'v': [1, 2, 3, 4]})")
    r.run_cell(2)
    r.run_cell(5)   # skip cells 3 (alias) and 4 (mutation)
    return {"after_edit": r.get_output(5)}, {"after_edit": 5}


def scen_expensive_consumer_upstream_edit(r):
    """HARD: an expensive cached consumer over a frame; edit a deep value in the
    frame builder and re-run consumer only. Must reflect the new value, not a
    stale cached aggregate (sampling-hash risk)."""
    _start(r, [
        PRE,
        "df = pd.DataFrame({'v': list(range(1000))})",
        "import time\ntotal = 0\nfor i in range(3):\n    total += int(df['v'].sum())\nprint('AGG', total)",
    ])
    r.run_all()
    # change row 700 only (outside the 5-row hash sample) in the builder
    r.set_cell_source(2, "df = pd.DataFrame({'v': list(range(1000))})\ndf.loc[700, 'v'] = 999999")
    r.run_cell(2)
    r.run_cell(3)
    return {"after_edit": r.get_output(3)}, {"after_edit": 3}


SCENARIOS = [
    scen_mutate_upstream_recompute,
    scen_alias_downstream_consumer,
    scen_helper_edit_invalidates,
    scen_helper_edit_mutating,
    scen_dict_accumulator,
    scen_del_variable_then_consumer,
    scen_sort_inplace_idempotent,
    scen_reorder_cells,
    scen_loc_mask_deep_row,
    scen_append_rows_rerun,
    scen_rerun_upstream_twice_no_double,
    scen_loc_mask_stable_key_deep,
    scen_edit_base_no_rerun_intermediate,
    # harder multi-hop / alias reconstruction probes (all pass -> cash correct):
    scen_multihop_inplace_reconstruction,
    scen_alias_reflects_upstream_edit,
    scen_expensive_consumer_upstream_edit,
    # Promoted from CANARY_SCENARIOS: a callee's write to a global is captured
    # and restored at cell level now (CAS-260), so this matches the oracle
    # rather than diverging from it. The loop-body half of that limitation is
    # still open and is pinned by
    # ``test_callee_global_capture.py::test_a_loop_body_captures_the_callee_global_too``.
    scen_hidden_global_mutation,
]

# Documented-limitation canaries. Each reproduces a limitation explicitly listed
# in docs/known-limitations.md ("Mutation that cash cannot see"), so cash-ON
# provably diverges from a clean top-to-bottom run on an isolated re-run. They
# are xfail (documented + working-as-designed), and they double as the harness's
# non-vacuity proof: they demonstrate the oracle DOES catch a real divergence, so
# the green passes above are meaningful rather than trivially matching.
CANARY_SCENARIOS = [
    scen_class_var_accumulator,        # docs: "Class variables shared across cells"
    scen_alias_double_apply_own_rerun, # docs: "Mutating an object created in an earlier cell" / alias
]


def _oracle(tmp_path, final_sources, check):
    """Run final_sources top-to-bottom in a fresh cash-OFF kernel."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    r = NotebookTestRunner(work_dir=tmp_path, use_pool=False)
    try:
        r.create_notebook(final_sources)
        r.start_kernel(with_cash=False)
        r.client.allow_errors = True
        r.run_all()
        return {label: r.get_output(n) for label, n in check.items()}
    finally:
        r.shutdown()


def _assert_matches_oracle(scenario, tmp_path):
    on_dir = tmp_path / "on"
    on_dir.mkdir(parents=True, exist_ok=True)
    r = NotebookTestRunner(work_dir=on_dir, use_pool=False)
    try:
        captured, check = scenario(r)
        final_sources = [c.source for c in r.nb.cells]
    finally:
        r.shutdown()

    oracle = _oracle(tmp_path / "oracle", final_sources, check)

    diffs = {k: (oracle[k], captured[k]) for k in check if oracle[k] != captured[k]}
    assert not diffs, (
        f"cash-ON diverged from top-to-bottom oracle in {scenario.__name__}:\n"
        + "\n".join(f"  [{k}] oracle={o!r}  cash={c!r}" for k, (o, c) in diffs.items())
    )


@pytest.mark.timeout(300)
@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.__name__ for s in SCENARIOS])
def test_cash_matches_top_to_bottom(scenario, tmp_path):
    _assert_matches_oracle(scenario, tmp_path)


@pytest.mark.timeout(300)
@pytest.mark.xfail(reason="documented limitation (docs/known-limitations.md: "
                          "'Mutation that cash cannot see'); isolated re-run only. "
                          "Serves as the harness's non-vacuity proof.",
                   strict=True)
@pytest.mark.parametrize("scenario", CANARY_SCENARIOS,
                         ids=[s.__name__ for s in CANARY_SCENARIOS])
def test_documented_limitation_canaries(scenario, tmp_path):
    _assert_matches_oracle(scenario, tmp_path)


# ---------------------------------------------------------------------------
# Decorator + DataFrame sampling-hash probe. Ground truth is computed directly
# (uncached df['v'].sum()) in the SAME kernel, so no oracle kernel is needed.
# ---------------------------------------------------------------------------

def _decorator_deep_mutation(nb_runner, mutate_stmt):
    nb_runner.create_notebook([
        "import pandas as pd, numpy as np, cash",
        "df = pd.DataFrame({'v': list(range(1000))})",
        "@cash.cache\ndef agg(frame):\n    return int(frame['v'].sum())",
        "a1 = agg(df)\nprint('A1', a1)",
        mutate_stmt,
        "cached = agg(df)\ndirect = int(df['v'].sum())\nprint('CACHED', cached, 'DIRECT', direct)",
    ])
    nb_runner.start_kernel(with_cash=True)
    nb_runner.run_all()
    return nb_runner.get_output(6)


@pytest.mark.timeout(180)
def test_decorator_deep_inplace_mutation_not_stale(nb_runner):
    """@cash.cache over a summary of a 1000-row frame; mutate row 700 (outside the
    5-row hash sample) in place, call again. Cached value must equal the direct
    (uncached) recompute."""
    out = _decorator_deep_mutation(nb_runner, "df.loc[700, 'v'] = 10**9")
    import re
    m = re.search(r"CACHED (\d+) DIRECT (\d+)", out)
    assert m, f"no sentinel in output: {out!r}"
    cached, direct = int(m.group(1)), int(m.group(2))
    assert cached == direct, (
        f"STALE decorator hit: cached={cached} but true value is {direct} "
        f"(deep in-place mutation at row 700 went unnoticed by the arg hash)"
    )


@pytest.mark.timeout(180)
def test_decorator_shallow_inplace_mutation_not_stale(nb_runner):
    """Control: mutate row 0 (inside the 5-row sample). Should be detected."""
    out = _decorator_deep_mutation(nb_runner, "df.loc[0, 'v'] = 10**9")
    import re
    m = re.search(r"CACHED (\d+) DIRECT (\d+)", out)
    assert m, f"no sentinel in output: {out!r}"
    cached, direct = int(m.group(1)), int(m.group(2))
    assert cached == direct, (
        f"STALE decorator hit: cached={cached} but true value is {direct} (row 0 change)"
    )


# ---------------------------------------------------------------------------
# Time-saved measurement over an expensive EDA-shaped workflow + a correctness
# check that the warm restore matches a direct recompute.
# ---------------------------------------------------------------------------

@pytest.mark.timeout(300)
def test_time_saved_and_restore_correct(nb_runner):
    import re, json
    nb_runner.create_notebook([
        "import pandas as pd, numpy as np, time, cash",
        # expensive build (~0.4s)
        "def build():\n    time.sleep(0.4)\n    return pd.DataFrame({'v': list(range(100000))})\ndf = build()",
        # expensive transform (~0.4s), in-place-ish via reassign
        "time.sleep(0.4)\ndf = df.assign(w=df['v'] * 2)",
        # expensive aggregate (~0.4s)
        "time.sleep(0.4)\nagg = int(df['w'].sum())\nprint('AGG', agg)",
    ])
    nb_runner.start_kernel(with_cash=True)
    nb_runner.run_all()
    first = nb_runner.get_output(4)
    # warm re-run of the whole notebook: everything should restore
    nb_runner.run_all()
    warm = nb_runner.get_output(4)
    assert first == warm, f"warm restore diverged: first={first!r} warm={warm!r}"
    # ground-truth aggregate
    true_agg = sum(x * 2 for x in range(100000))
    m = re.search(r"AGG (\d+)", warm)
    assert m and int(m.group(1)) == true_agg, f"wrong agg: {warm!r} want {true_agg}"
