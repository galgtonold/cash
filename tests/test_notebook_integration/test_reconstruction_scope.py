"""CAS-193/196/200: upstream reconstruction is scoped to the current cell.

One root cause: cash re-fired an upstream file-writing statement during another
cell's reconstruction even though the current cell reads neither the writer's
output file nor any variable it produces. Manifestations:

* CAS-196 (WRONG): a downstream cell re-fired ``df.to_csv('audit.log', mode='a')``
  -> a duplicated non-idempotent append -> corrupted file.
* CAS-200/193 (BLOCKING): a cell BELOW a plot cell re-fired the plot's
  ``fig.savefig(...)`` (and, with an evicted RAM-only intermediate, raised
  ``UpstreamStateError``), wedging the notebook tail.

The scope gate: a file-writer whose output no consumer relevant to the current
cell reads is an unrelated / terminal side-effect and must never be re-fired
during reconstruction. It still re-runs when a reader that actually reads the
file (CAS-81/82) or the writer's own cell runs.
"""
import pytest

pytest.importorskip("pandas")

pytestmark = [pytest.mark.files, pytest.mark.timeout(180)]


def _rows(path):
    try:
        with open(path) as f:
            return sum(1 for ln in f if ln.strip())
    except FileNotFoundError:
        return 0


def _p(path) -> str:
    return str(path).replace("\\", "/")


def _restart(nb_runner):
    import asyncio
    nb_runner.restart()
    nb_runner._inject_notebook_path()


def test_runall_does_not_double_a_loop_fed_audit_append(nb_runner, tmp_path):
    """CAS-196 within-session: a run-all must append the audit log exactly once.

    The audit writer's payload (``audit_row``) derives from a LOOP-computed
    ``slopes``/``worst``. The loop makes the simulated and runtime lineages of
    ``audit_row`` diverge structurally, so the writer looked "input-changed" and
    was re-fired when a LATER cell (that writes summary.csv and reads only
    ``agg``) reconstructed upstream -- doubling the append. No cell reads
    audit.log, so the writer must not be re-fired at all.
    """
    audit = _p(tmp_path / "audit.log")
    summary = _p(tmp_path / "summary.csv")
    nb_runner.create_notebook([
        "import pandas as pd, numpy as np\nimport cash\n%cash_on",
        "agg = pd.DataFrame({'category': ['A','A','B','B'], 'month': [1,2,1,2],\n"
        "                    'margin': [0.5, 0.4, 0.3, 0.35]})\nprint('agg', agg.shape)",
        # loop-computed slopes/worst -> the divergence source
        "slopes = {}\n"
        "for cat, g in agg.groupby('category'):\n"
        "    g = g.sort_values('month'); x = np.arange(len(g)); yv = g['margin'].to_numpy()\n"
        "    slopes[cat] = float(np.polyfit(x, yv, 1)[0])\n"
        "worst = min(slopes, key=slopes.get)\nprint('WORST', worst)",
        # WRITER: non-idempotent append whose payload derives from the loop
        f"audit_row = pd.DataFrame([{{'answer': worst, 'slope': round(slopes[worst], 6)}}])\n"
        f"audit_row.to_csv('{audit}', mode='a', header=False, index=False)\n"
        "receipt = 'audit-appended'\nprint('AUDIT_APPENDED', worst)",
        # a LATER writer cell that reads only `agg` -- reconstructing it used to
        # drag in the audit writer and re-fire the append
        f"agg.to_csv('{summary}', index=False)\nprint('SUMMARY_WRITTEN')",
        "consumer = int(agg.shape[0])\nprint('CONSUMER', consumer)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _rows(audit) == 1, (
        f"run-all re-fired the loop-fed audit append: audit.log has {_rows(audit)} "
        f"lines (expected 1)"
    )


def test_restart_reader_does_not_refire_co_produced_writer(nb_runner, tmp_path):
    """CAS-196 post-restart: a reader depending on a var co-produced with the
    writer must reconstruct that var without re-firing the write.

    ``receipt`` is produced in the same cell as the ``to_csv`` append. After a
    restart, running only the reader (which needs ``receipt``) must rebuild
    ``receipt`` without re-appending to the audit log (nothing reads it).
    """
    audit = _p(tmp_path / "audit.log")
    nb_runner.create_notebook([
        "import pandas as pd\nimport cash\n%cash_on",
        "vals = pd.DataFrame({'v': [1, 2, 3, 4]})\nprint('vals', vals.shape)",
        "worst = int(vals['v'].max())\nprint('WORST', worst)",
        f"audit_row = pd.DataFrame([{{'answer': worst}}])\n"
        f"audit_row.to_csv('{audit}', mode='a', header=False, index=False)\n"
        "receipt = 'audit-appended'\nprint('AUDIT_APPENDED')",
        "print('READER', worst, receipt)\nreader_out = f'{worst}:{receipt}'\nprint('READER_OUT', reader_out)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _rows(audit) == 1

    _restart(nb_runner)
    nb_runner.run_cell(1)   # imports + %cash_on
    nb_runner.run_cell(5)   # reader (needs receipt, co-produced with the writer)
    out = nb_runner.get_output(5)
    assert "UpstreamStateError" not in out, f"reader wedged reconstructing writer: {out}"
    assert _rows(audit) == 1, (
        f"post-restart reader re-fired the audit append: {_rows(audit)} lines (expected 1)"
    )
    assert "READER_OUT 4:audit-appended" in out, f"reader served wrong data: {out}"


def test_plot_writer_not_refired_by_unrelated_cell_after_restart(nb_runner, tmp_path):
    """CAS-200/193: after a restart, a cell below a plot cell that shares no
    variable with it must not re-fire the plot's savefig nor UpstreamStateError.
    """
    pytest.importorskip("matplotlib")
    chart = tmp_path / "chart.png"
    chart_p = _p(chart)
    nb_runner.create_notebook([
        "import pandas as pd\nimport matplotlib\nmatplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\nimport cash\n%cash_on",
        "records = pd.DataFrame({'cat': list('abcde'), 'v': [10,20,30,40,50]})\nprint('records', records.shape)",
        "summary = records.groupby('cat', as_index=False).agg(total=('v','sum'))\nprint('summary', summary.shape)",
        # PLOT: RAM-only `itm` -> non-cacheable pie draw -> savefig writer
        "itm = summary['total'].tolist()\n"
        "fig, ax = plt.subplots()\n"
        "ax.pie(itm, labels=summary['cat'].tolist())\n"
        f"fig.savefig('{chart_p}')\nprint('PLOTTED', len(itm))",
        # independent downstream cell: reads only `records`
        "grand_total = int(records['v'].sum())\nprint('GRAND_TOTAL', grand_total)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert chart.exists(), "plot cell never wrote chart.png"

    _restart(nb_runner)
    nb_runner.run_cell(1)   # imports + %cash_on
    nb_runner.run_cell(2)   # records
    nb_runner.run_cell(3)   # summary  (NOT the plot cell 4)
    import os
    os.remove(chart)        # delete the plot artifact
    nb_runner.run_cell(5)   # independent tail cell
    out = nb_runner.get_output(5)
    assert "UpstreamStateError" not in out, f"unrelated cell wedged on the plot: {out}"
    assert "GRAND_TOTAL 150" in out, f"tail cell produced wrong value: {out}"
    assert not chart.exists(), (
        "reconstructing an unrelated cell re-fired the plot writer (fig.savefig): "
        "the deleted chart.png was re-created (CAS-200/193)"
    )
