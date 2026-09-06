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
import re

import pytest


def _strip_style(html: str) -> str:
    """Drop the inlined ``<style>`` block.

    So a match below can only land on the badge's actual markup -- never on
    a CSS rule or a comment inside one. Before the stylesheet was minified,
    'Upstream' (title case) appeared only inside CSS *comments*, nowhere in
    the rendered markup -- so a match against the raw HTML blob was matching
    stylesheet prose, not the badge. Minification dropped the comments and
    turned that into an outright failure. Stripping style first means a
    future stylesheet change can never again make -- or break -- an
    assertion here.
    """
    return re.sub(r'<style>.*?</style>', '', html, flags=re.DOTALL)


def _full_restart_code(vars_to_clear):
    """Return code that simulates a kernel restart by clearing BOTH cash's
    statement-processor lineage state AND the kernel-side variables.

    Plain ``reset_cash_state()`` only clears the magics-level tracking
    dicts; it is NOT enough to make cash treat a variable as missing --
    ``mismatch_classifier._check_missing_required_inputs`` only schedules a
    required input for re-derivation when it is actually absent from
    ``shell.user_ns`` (``if var_name in self.shell.user_ns: continue``), and
    plain ``reset_cash_state()`` deliberately leaves every variable sitting
    in ``user_ns`` -- see its own docstring. So deleting the names from
    ``globals()`` here is load-bearing, not decorative. Pattern copied from
    ``test_cost_model_skip_downstream._full_restart_code`` /
    ``test_skipped_timing_after_restart._full_restart_code``.
    """
    var_list = ', '.join(f"'{v}'" for v in vars_to_clear)
    return f"""
try:
    _cash_magics = get_ipython().magics_manager.registry.get('CashMagics')
    if _cash_magics:
        _cash_magics._tracking_state.variable_sources.clear()
        _cash_magics._tracking_state.variable_hashes.clear()
        if hasattr(_cash_magics, '_statement_processor'):
            _cash_magics._statement_processor.variable_lineage.clear()
            _cash_magics._statement_processor.executed_cell_codes.clear()
            _cash_magics._statement_processor.executed_cell_hashes.clear()
except Exception:
    pass

for _v in [{var_list}]:
    try:
        del globals()[_v]
    except KeyError:
        pass
"""


def _simulate_restart(nb_runner, vars_to_clear):
    """Apply the magics-level reset AND delete the named variables from the
    live kernel namespace, so the next run is forced to actually re-derive
    or restore them -- the way a real kernel restart would."""
    nb_runner.reset_cash_state()
    nb_runner._run_async(
        nb_runner.client.kc._async_execute_interactive(
            _full_restart_code(vars_to_clear),
            store_history=False,
        )
    )


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
    after a real restart (the non-regression case).

    NOTE ON THE RESET MECHANISM: this test used to reset via
    ``nb_runner.reset_cash_state()`` alone. That clears cash's own
    bookkeeping but -- by that method's own docstring -- leaves a/b/c/d
    sitting in ``user_ns``. Cash's upstream checker only schedules a
    variable for re-derivation when it is actually MISSING from memory
    (``mismatch_classifier._check_missing_required_inputs``'s
    ``if var_name in self.shell.user_ns: continue`` gate), so that setup
    could never produce an upstream section at all: confirmed directly with
    ``%cash_debug on``, which reported "Simulation result: 0 stmts to
    re-execute, 0 stmts restored from cache" for it every time. The
    assertion below only ever passed because -- pre-minification -- it was
    matching 'Upstream'/'Restored' inside a CSS *comment* in the inlined
    stylesheet, never real upstream content. Actually deleting the
    variables (``_simulate_restart``, the same pattern already used by the
    sibling ``test_cost_model_skip_downstream.py`` and
    ``test_skipped_timing_after_restart.py`` in this directory) is what
    exercises the feature this test is named for.
    """
    nb_runner.create_notebook([
        "a = 1",
        "b = a + 1",
        "c = b + 1",
        "d = c + 1\nprint(f'd={d}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert 'd=4' in nb_runner.get_output(4)

    _simulate_restart(nb_runner, ['a', 'b', 'c', 'd'])
    nb_runner.run_cell(4)
    assert 'd=4' in nb_runner.get_output(4)

    html = _badge_html(nb_runner.get_cell(4))
    assert html, "Cell 4 produced no badge HTML on downstream-only run"
    body = _strip_style(html)
    # 'upstream context' is the section's real (lowercase) markup text --
    # CSS gives it `text-transform: uppercase` so it READS as "UPSTREAM
    # CONTEXT" on screen, but the HTML itself is lowercase. It is emitted
    # only when the upstream section actually has re-derived items (see
    # html.py's `_upstream_section_html`: `if upstream is not None and
    # upstream.items:`), so its presence is real, structural evidence that
    # a, b, c were re-derived and surfaced to the user -- not prose that
    # merely sounds relevant.
    assert 'c3-upstream' in body and 'upstream context' in body, (
        "Badge for downstream-only run does not show the upstream chain "
        "(no upstream section in the badge body) -- cash may have "
        "re-derived a/b/c silently, without surfacing that work.\n"
        f"HTML body:\n{body[:2000]}"
    )
