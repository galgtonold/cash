"""A frame rebuilt differently is a different frame, whatever the loop over it says.

A month-end close, and a wrong number that reached a deliverable. A cell built ``status_all`` one way, then looped over it
writing columns in place::

    status_all = pd.concat([...])                     # Monday
    for col, src in ((...), (...)):
        status_all[col] = (status_all[src] * 1.2).round(2)

The next day the first line was rewritten (a matching fix; 50,000 of 300,000
rows changed) and the loop was left alone. The statement after it,
``aged = aged_debt(status_all, CLOSE_DATE)``, came back CACHED with Monday's
numbers: an aged-debt table EUR 2.94M short, exported to controlling, with a
green badge on it. Re-running kept it cached and kept it wrong.

The lineage a loop mints for a variable it mutates was built from the loop's
source, a SAMPLED hash of the value, and the lineages of the body's OTHER
reads -- the receiver's own history was excluded on purpose. With the loop
source identical and the sampled hash colliding across the two frames, nothing
was left to tell Monday's frame from Tuesday's, so the consumer's key never
moved.

The test keeps the same shape at a size that runs in seconds, and checks the
cached consumer against the same computation done live in the same kernel.
"""

PIN = "cash.configure(call_cost_floor_seconds=0.0, min_execution_time_to_cache_seconds=0.0)\n"
SETUP = "import cash\n%load_ext cash\n%cash_badge print\n" + PIN + "%cash_on"

DEFS = (
    "import numpy as np, pandas as pd\n"
    "def make_base():\n"
    "    rng = np.random.default_rng(7)\n"
    "    return pd.DataFrame({'id': np.arange(60_000),\n"
    "                         'total': rng.uniform(50, 5000, 60_000).round(2)})\n"
    "def totals(frame):\n"
    "    np.sort(np.random.default_rng(3).random(3_000_000))   # worth caching\n"
    "    return float(frame.loc[frame['outstanding'] > 0.01, 'outstanding_eur'].sum())"
)

# Monday: everyone has paid nothing, so every row is outstanding.
BUILD_A = (
    "base = make_base()\n"
    "status_all = base.assign(paid=0.0)\n"
    "status_all['outstanding'] = (status_all['total'] - status_all['paid']).round(2)\n"
    "for col, src in (('total_eur', 'total'), ('outstanding_eur', 'outstanding')):\n"
    "    status_all[col] = (status_all[src] * 1.2).round(2)\n"
    "print('rows', len(status_all))"
)

# Tuesday: the fix -- everything past the first handful of rows is settled, so
# those rows drop out of the open items. The LOOP IS BYTE-IDENTICAL; only the
# lines above it changed.
#
# The first five rows are deliberately left alone. `compute_hash` samples a
# DataFrame's first five rows, so this is what makes the sampled hash collide
# across the two builds -- which is the condition the bug needs, and why a
# version of this test that changed row 1 passed on the broken build. Four
# minimised versions of the original repro missed the same way.
BUILD_B = (
    "base = make_base()\n"
    "settled = base['id'] >= 5\n"
    "status_all = base.assign(paid=np.where(settled, base['total'], 0.0))\n"
    "status_all['outstanding'] = (status_all['total'] - status_all['paid']).round(2)\n"
    "for col, src in (('total_eur', 'total'), ('outstanding_eur', 'outstanding')):\n"
    "    status_all[col] = (status_all[src] * 1.2).round(2)\n"
    "print('rows', len(status_all))"
)

CONSUMER = "aged = totals(status_all)\nprint('AGED %.2f' % aged)"
# `# @cash:no-cache` is load-bearing. Without it this cell caches like any
# other, keyed on the same lineage as the consumer -- so on the broken build it
# went stale the same way, agreed with the stale `aged`, and the test passed
# while every number in it was wrong. An oracle inside the cached notebook is
# not an oracle unless it is exempt from the cache.
LIVE = (
    "# @cash:no-cache\n"
    "live = float(status_all.loc[status_all['outstanding'] > 0.01, 'outstanding_eur'].sum())\n"
    "print('LIVE %.2f' % live)\n"
    "print('VERDICT', 'MATCH' if abs(live - aged) < 0.01 else 'STALE')"
)


def _value(out: str, label: str) -> float:
    line = next(ln for ln in out.splitlines() if ln.startswith(label))
    return float(line.split()[1])


def test_the_consumer_sees_the_rebuilt_frame(nb_runner):
    nb_runner.create_notebook([SETUP, DEFS, BUILD_A, CONSUMER, LIVE])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "MATCH" in nb_runner.get_output(5), nb_runner.get_output(5)

    # The fix, upstream of an untouched loop.
    nb_runner.set_cell_source(3, BUILD_B)
    nb_runner.run_cells([3, 4, 5])

    consumer = nb_runner.get_output(4)
    check = nb_runner.get_output(5)
    aged, live = _value(consumer, "AGED"), _value(check, "LIVE")
    assert "VERDICT MATCH" in check, (
        f"cached {aged:.2f} against a live {live:.2f} -- the consumer kept the "
        f"value from before the upstream fix:\n{consumer}\n{check}"
    )


def test_the_two_builds_really_do_differ(nb_runner):
    """The guard for the test above: if both builds produced the same total,
    the test would pass while proving nothing."""
    nb_runner.create_notebook([SETUP, DEFS, BUILD_A, CONSUMER, LIVE])
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = _value(nb_runner.get_output(5), "LIVE")

    nb_runner.set_cell_source(3, BUILD_B)
    nb_runner.run_cells([3, 4, 5])
    second = _value(nb_runner.get_output(5), "LIVE")
    assert abs(first - second) > 1.0, (first, second)
