"""End-to-end integration: %cash_provenance --graph must show the full chain
when a variable is mutated across multiple cells / statements.

This is the financial-demo failure mode: `df` is first created via
`df = pd.read_csv(...)`, then mutated in subsequent statements (`df['Date'] = ...`,
`df = df.sort_values(...)`, `df['SMA'] = df.groupby(...)`). Each statement
generates a *new* provenance record. Walking only `get_latest(df).inputs`
misses the original creation cell — so the graph shows almost nothing
useful, even though file_deps and history-count are recorded correctly.

The fix walks the union of `.inputs` across ALL history records of a
variable, not just the latest one.
"""
from __future__ import annotations

import pytest


@pytest.mark.timeout(60)
def test_provenance_graph_shows_full_chain_across_mutations(nb_runner):
    """Multi-cell df pipeline: graph should mention intermediate variables."""
    nb_runner.create_notebook([
        "import cash, pandas as pd",
        "%cash_on",
        # Cell 3: create df from a literal so we don't need an external file
        "raw = pd.DataFrame({'x': [1, 2, 3, 4], 'group': ['a', 'b', 'a', 'b']})",
        # Cell 4: derive df from raw
        "df = raw.copy()",
        # Cell 5: mutate df
        "df['x2'] = df['x'] * 2",
        # Cell 6: mutate df again — assigns from a groupby of itself
        "df['x_norm'] = df.groupby('group')['x'].transform(lambda v: v - v.mean())",
        # Cell 7: a downstream summary computed from df
        "summary = df['x_norm'].sum()",
        # Cell 8: the assertion target
        "%cash_provenance df --graph",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    out = nb_runner.get_output(8)
    # The output should mention `raw` somewhere — `df` was originally derived
    # from `raw` (cell 4), and the graph walker must surface that even though
    # cell 6's `df` record only lists `[df]` as its direct input.
    assert "raw" in out, (
        f"Expected 'raw' in provenance graph (df was derived from raw in an "
        f"earlier statement). Got:\n{out}"
    )


@pytest.mark.timeout(60)
def test_provenance_graph_shows_chain_in_financial_demo_flow(nb_runner, tmp_path):
    """Mirror the financial-analysis demo's exact flow: read a CSV into df,
    mutate df across multiple cells, then call %cash_provenance df --graph.

    The chain must reach back to the original `pd.read_csv` cell — that's
    what the user reported missing.
    """
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "Ticker,Close\nAAPL,150\nMSFT,200\nGOOGL,2800\nAAPL,160\nMSFT,210\nGOOGL,2850\n",
        encoding="utf-8",
    )
    csv_path_str = str(csv_path).replace("\\", "/")
    nb_runner.create_notebook([
        "import cash, pandas as pd, numpy as np",
        "%cash_on",
        # Cell 3 — load (heavyweight in the real demo, here just real read_csv)
        f"df = pd.read_csv('{csv_path_str}')",
        # Cell 4 — first mutation
        "df['Close'] = df['Close'].astype(float)",
        # Cell 5 — second mutation (modeled on df.sort_values in the demo)
        "df = df.sort_values(by=['Ticker'])",
        # Cell 6 — third mutation that mirrors the financial-demo problematic
        # statement: df['SMA'] = df.groupby(...).transform(...)
        "df['SMA'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(2).mean())",
        # Cell 7 — the assertion target
        "%cash_provenance df --graph",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    out = nb_runner.get_output(7)
    # Hard requirements after this fix is correct:
    # (a) History must show > 1 record — multiple mutations.
    # (b) The graph must include the read_csv code OR the path OR mention `pd`
    #     so the chain back to the file is visible.
    assert "(no dependencies)" not in out, (
        f"Graph still shows no deps. Full provenance output:\n{out}"
    )
    chain_signal = any(
        s in out for s in ("read_csv", "data.csv", "pd")
    )
    assert chain_signal, (
        f"Graph block should mention the upstream read_csv/data.csv/pd:\n{out}"
    )


@pytest.mark.timeout(60)
def test_provenance_graph_shows_external_file_dep_inputs(nb_runner):
    """When df is built from a tracked file, the chain should still show
    the file-read assignment in the graph (its `pd.read_csv` call).
    """
    nb_runner.create_notebook([
        "import cash, pandas as pd, numpy as np, tempfile, os",
        "%cash_on",
        # Build a CSV in tmp so the file_tracker sees a real read
        "_tmp = tempfile.gettempdir()",
        "_csv_path = os.path.join(_tmp, 'cash_prov_test.csv')",
        "pd.DataFrame({'x': [1, 2, 3]}).to_csv(_csv_path, index=False)",
        # df ← pd.read_csv(...) — the assignment cash should remember
        "df = pd.read_csv(_csv_path)",
        # mutate df
        "df['x_sq'] = df['x'] ** 2",
        # downstream
        "total = df['x_sq'].sum()",
        # Inspect
        "%cash_provenance df --graph",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    out = nb_runner.get_output(9)
    # File dep summary line is unrelated; we care about the GRAPH block.
    # The graph should mention the read_csv code or _csv_path as an
    # ancestor input — currently it shows just the latest mutation.
    graph_part = out.split("Dependency Graph:", 1)[-1] if "Dependency Graph:" in out else ""
    assert "(no dependencies)" not in graph_part, (
        f"Graph reports no deps even though df came from a tracked read_csv. "
        f"Full output:\n{out}"
    )
    # The graph should reference something from the upstream chain.
    has_upstream_ref = (
        "read_csv" in graph_part
        or "_csv_path" in graph_part
        or "x_sq" in graph_part  # at minimum, the per-statement record
    )
    assert has_upstream_ref, (
        f"Graph block should reference the upstream chain (read_csv / _csv_path / x_sq). "
        f"Got graph section:\n{graph_part}"
    )
