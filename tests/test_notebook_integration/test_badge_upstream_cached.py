"""Regression test: multi-import upstream cell must be FULLY re-imported on
kernel restart before a downstream cell runs.

Bug scenario (financial demo):

    # Cell 1
    import pandas as pd
    import numpy as np

    # Cell 2
    df = pd.read_csv(...)
    df['Date'] = pd.to_datetime(df['Date'])

    # Cell 3
    df = df.sort_values(by=['Ticker', 'Date'])

    # Cell 4  (the cell the user re-runs alone after kernel restart)
    df['VolAdj_20'] = df.groupby('Ticker')['Close'].transform(
        lambda x: x.rolling(window=20).apply(
            lambda y: np.mean(y) / (np.std(y) + 1e-6), raw=True))
    ...

After a real kernel restart, clicking only Cell 4 triggered the upstream
re-execute pipeline. It auto-executed ``import pandas as pd`` (pulled in as
a transitive dep of ``df``), but skipped ``import numpy as np`` even though
``np`` was a direct required input of Cell 4. The cell then died with
``NameError: name 'np' is not defined``.

Root cause: simulating an ``import`` statement writes the module's lineage
into ``variable_lineage`` via ``_propagate_import_lineage``. The
``_check_missing_required_inputs`` short-circuit ``var_name in
self.variable_lineage`` then masked the fact that the module was missing
from ``shell.user_ns``, so it was never added to ``broken_vars`` and never
scheduled for re-execute. The module-in-memory check existed but ran after
the short-circuit. Fix: move the module check before the
``variable_lineage`` short-circuit. (src/cash/notebook/mismatch_classifier.py)
"""
import pytest


def _badge_html(cell) -> str:
    """Return concatenated text/html from a cell's display_data outputs."""
    parts = []
    for output in cell.get('outputs', []):
        if output.output_type in ('display_data', 'execute_result'):
            data = output.get('data', {})
            html = data.get('text/html', '')
            if isinstance(html, list):
                html = ''.join(html)
            if html:
                parts.append(html)
    return '\n'.join(parts)


@pytest.mark.badges
def test_multi_import_cell_fully_restored_after_kernel_restart(nb_runner, tmp_path):
    """``import pandas as pd; import numpy as np`` in one cell — both imports
    must be auto-executed when a downstream cell that uses both is run
    alone after a real kernel restart."""
    csv_path = tmp_path / "data.csv"
    csv_path_str = str(csv_path).replace('\\', '/')
    csv_path.write_text(
        "Ticker,Date,Close\n"
        + "\n".join(
            f"{t},2024-01-{d:02d},{100 + d}"
            for t in ("AAPL", "GOOGL", "MSFT")
            for d in range(1, 21)
        )
        + "\n"
    )

    nb_runner.create_notebook([
        # Cell 1 — two imports in the same cell. This is the bug trigger:
        # without the fix, only the first import gets scheduled on rerun.
        "import pandas as pd\nimport numpy as np",
        # Cell 2 — read csv (file dep + pd dep)
        f"df = pd.read_csv('{csv_path_str}')\ndf['Date'] = pd.to_datetime(df['Date'])",
        # Cell 3 — self-assignment sort
        "df = df.sort_values(by=['Ticker', 'Date'])",
        # Cell 4 — uses BOTH df (transitively needs pd) AND np (direct)
        (
            "import time\n"
            "t0 = time.time()\n"
            "df['VolAdj_5'] = df.groupby('Ticker')['Close'].transform(\n"
            "    lambda x: x.rolling(window=5).apply(\n"
            "        lambda y: np.mean(y) / (np.std(y) + 1e-6), raw=True)\n"
            ")\n"
            "def custom_weighted_mean(x):\n"
            "    weights = np.arange(1, len(x) + 1)\n"
            "    return np.sum(x * weights) / np.sum(weights)\n"
            "df['SMA_5'] = df.groupby('Ticker')['Close'].transform(\n"
            "    lambda x: x.rolling(window=5).apply(custom_weighted_mean, raw=True)\n"
            ")\n"
            "print(f'rows={len(df)}; elapsed={time.time()-t0:.3f}s')"
        ),
    ])
    nb_runner.start_kernel()

    # Populate the disk cache.
    nb_runner.run_all()
    assert 'rows=' in nb_runner.get_output(4)

    # Real kernel restart — wipes user_ns AND cash tracking state. The bug
    # only surfaces here; reset_cash_state alone leaves df/np in memory.
    nb_runner.shutdown()
    nb_runner.start_kernel()

    nb_runner.run_cell(4)

    raw = nb_runner.get_raw_output(4)
    assert 'NameError' not in raw, (
        "Upstream re-execution skipped 'import numpy as np' — cell 4 hit "
        f"NameError. Raw output:\n{raw[:2000]}"
    )
    assert 'rows=' in nb_runner.get_output(4)

    # And the badge should still surface upstream activity.
    html = _badge_html(nb_runner.get_cell(4))
    assert html, "Cell 4 produced no badge HTML on downstream-only kernel-restart run"


@pytest.mark.badges
def test_badge_shows_upstream_chain_after_downstream_only_run(nb_runner):
    """Sanity: simple a→b→c→d chain still surfaces upstream rows on the badge
    after a tracking-state reset (the non-regression case)."""
    nb_runner.create_notebook([
        "a = 1",
        "b = a + 1",
        "c = b + 1",
        "d = c + 1\nprint(f'd={d}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert 'd=4' in nb_runner.get_output(4)

    nb_runner.reset_cash_state()
    nb_runner.run_cell(4)
    assert 'd=4' in nb_runner.get_output(4)

    html = _badge_html(nb_runner.get_cell(4))
    assert html, "Cell 4 produced no badge HTML on downstream-only run"
    assert (
        'intermediate dependency step' in html
        or 'Upstream' in html
        or 'Restored' in html
    ), (
        "Badge for downstream-only run does not show upstream chain.\n"
        f"HTML:\n{html[:2000]}"
    )
