"""A value restored from cache still has to notice its inputs were rebuilt.

A wrong number in a support-ticket triage notebook. A cell fitted a
model per queue in a loop and derived a table from the appended rows::

    per_queue, rows = {}, []
    for q in QUEUES:
        vec, clf, yte, p = fit_queue(QUEUE_IDX[q])
        per_queue[q] = (vec, clf, yte, p)
        rows.append({"queue": q, "test_rows": len(yte), ...})
    queue_models = pd.DataFrame(rows).sort_values("auc", ascending=False)

After an upstream fix excluded auto-reply tickets, the export cell wrote the
PRE-fix table -- every queue about 28% too many rows and a flattered AUC --
while `top_terms.csv`, built from the same loop, was correct.

Why nothing caught it. `_classify_var` trusts a loop-derived variable found in
memory when no upstream edit is pending, guarded by two checks:

* `_check_loop_var_inputs_changed` skips inputs that are themselves loop-derived
  ("inherently mismatched"), so it skips `rows`;
* `_built_on_an_older_input` compares what the value WAS built from against what
  its inputs are now -- and reads that from `executed_input_lineages`, which is
  written only when a statement EXECUTES.

`queue_models` had been restored from cache, not executed, so its provenance
record was empty and the second guard compared nothing. Traced live:

    TRACE queue_models: loop_derived=True upstream_mods=False
    TRACE queue_models: TRUSTED FROM MEMORY -- never compared against its lineage
    TRACE queue_models: simulated inputs = {'rows': 'eb522b0c62'}
    TRACE queue_models: built_on vs live = {}          <- empty

In the same run `per_queue` reached the comparison, was found mismatched, and
was rebuilt. The difference was the missing provenance, not the trust rule.

The recipe needs all four of these, which is why four minimised versions by the
reporter and five by the maintainer all came out clean:

1. the derived statement must be expensive enough to be CACHED,
2. a kernel restart, so it is RESTORED rather than executed,
3. an upstream edit,
4. another consumer asked for FIRST, so the repair re-runs the loop for its sake
   and consumes the pending edit, leaving the derived value stale in memory.
"""

PIN = "cash.configure(call_cost_floor_seconds=0.0, min_execution_time_to_cache_seconds=0.0)\n"
SETUP = "import cash\n%load_ext cash\n%cash_badge print\n" + PIN + "%cash_on"
IMPORTS = "import numpy as np, pandas as pd, time, sys"

CUTOFF_BEFORE = "CUTOFF = 0.30\nprint('cutoff', CUTOFF)"
CUTOFF_AFTER = "CUTOFF = 0.70\nprint('cutoff', CUTOFF)"

HELPERS = (
    "def summarise(rs):\n"
    "    time.sleep(1.2)        # expensive enough to be worth an entry\n"
    "    return pd.DataFrame(rs).sort_values('rate', ascending=False)\n"
    "def score_one(k):\n"
    "    time.sleep(0.5)        # worth caching per iteration\n"
    "    rng = np.random.default_rng(k)\n"
    "    return float((rng.random(20_000) > CUTOFF).mean())"
)
LOOP = (
    "KEYS = [1, 2, 3, 4]\n"
    "per_key, rows = {}, []\n"
    "for k in KEYS:\n"
    "    v = score_one(k)\n"
    "    per_key[k] = v\n"
    "    rows.append({'key': k, 'rate': round(v, 4)})\n"
    "summary = summarise(rows)\n"
    "print('SUMMARY %.4f' % summary['rate'].sum())"
)
CONSUMER_A = "print('PER_KEY %.4f' % sum(per_key.values()))"
CONSUMER_B = "print('EXPORT %.4f' % summary['rate'].sum())"
# Exempt from the cache: an oracle that caches is not an oracle.
LIVE = "# @cash:no-cache\nprint('LIVE %.4f' % sum(round(score_one(k), 4) for k in KEYS))"

CELLS = [SETUP, IMPORTS, CUTOFF_BEFORE, HELPERS, LOOP, CONSUMER_A, CONSUMER_B, LIVE]


def _value(out: str, label: str) -> float:
    line = next(ln for ln in out.splitlines() if ln.startswith(label))
    return float(line.split()[1])


def _drive(nb_runner) -> tuple[str, str, str]:
    """Returns (loop cell output after the restart, consumer B's output, live)."""
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.restart()
    nb_runner.run_cells([1, 2, 3, 4, 5])  # 1-based: setup .. the loop
    after_restart = nb_runner.get_output(5)

    nb_runner.set_cell_source(3, CUTOFF_AFTER)  # the upstream edit
    nb_runner.run_cells([6])  # consumer A first: repairs the loop
    nb_runner.run_cells([7, 8])  # then the consumer of `summary`
    return after_restart, nb_runner.get_output(7), nb_runner.get_output(8)


def test_a_restored_derived_value_is_rebuilt_when_its_loop_re_runs(nb_runner):
    after_restart, export_out, live_out = _drive(nb_runner)
    export, live = _value(export_out, "EXPORT"), _value(live_out, "LIVE")
    assert abs(export - live) < 1e-9, (
        f"exported {export:.4f} against a live {live:.4f} -- the value restored "
        f"before the edit was trusted from memory after its loop re-ran:\n"
        f"{export_out}"
    )


def test_the_derived_value_really_was_restored(nb_runner):
    """The guard for the test above.

    If the derived statement re-executes after the restart, its provenance is
    recorded and the bug cannot appear -- the test would pass while proving
    nothing. That is how five earlier attempts at this repro came out clean.
    """
    after_restart, _export, _live = _drive(nb_runner)
    assert "CACHED: summary = summarise(rows)" in after_restart, (
        f"the derived statement was not restored from cache, so this notebook cannot exhibit the bug:\n{after_restart}"
    )
