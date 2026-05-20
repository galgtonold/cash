"""Cost-model-skip / floor-skip downstream consistency.

The notebook policy can decline to cache a statement for two reasons:

1. **10 ms execution-time floor** — statement compute is below
   ``min_execution_time_to_cache_seconds`` so no metadata entry is
   written at all (a "clean miss" on the next lookup).
2. **Cost-model ratio gate** — predicted restore time exceeds
   ``(1 - min_cache_savings_pct) × execution_time``, so a metadata-only
   entry is written and the value itself isn't persisted.

In both cases an expensive downstream statement may depend on the
declined-upstream's output. The downstream's cache key is derived from
its inputs' lineage hashes, and those lineage hashes are
``hash(code + sorted(input_lineages) + file_deps)`` — **value-independent**.
So as long as the upstream code + input chain is stable, the downstream
key is stable across runs, even when the upstream value has to be
recomputed each time because it was never cached.

These tests verify that invariant: the downstream cache survives the
upstream being declined, including across simulated kernel restarts.
This is the kind of cross-statement interaction that's easy to break
when the policy is tuned (the policy decides whether to *store*; the
lineage hash decides whether to *match*).
"""
import re

import pytest


pytestmark = [pytest.mark.core, pytest.mark.upstream]


def _get_badge_html(cell) -> str:
    """Return the badge HTML rendered by cash for a cell, or ''."""
    for output in cell.get('outputs', []):
        if output.output_type in ('execute_result', 'display_data'):
            html = output.get('data', {}).get('text/html', '')
            if html and ('Restored' in html or 'Skipped' in html
                         or 'Computed' in html or 'cash' in html.lower()):
                return html
    return ''


def _cell_status_words(cell) -> set[str]:
    """Distill cash's badge into a set of status words ({"Computed",
    "Restored", "Skipped"}) for a single cell.

    The badge HTML renders sections per status; the words appear inside
    `<th>` headers and `class="status_*"` attributes. We match conservatively
    on the readable text so the test isn't tied to badge styling.
    """
    html = _get_badge_html(cell)
    words = set()
    for w in ('Computed', 'Restored', 'Skipped'):
        if re.search(rf'\b{w}\b', html):
            words.add(w)
    return words


def _full_restart_code(vars_to_clear):
    """Return code that simulates a kernel restart by clearing BOTH cash's
    statement-processor lineage state AND the kernel-side variables. The
    plain ``reset_cash_state`` only clears the magics-level tracking, which
    is not enough to force a disk cache lookup — cash's
    ``executed_cell_codes`` on the statement processor would still
    short-circuit the cell. Pattern copied from
    ``test_skipped_timing_after_restart._full_restart_code``."""
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
    """Apply both the magics-level reset and the deeper statement-processor
    clear so the next run is forced to consult the persistent cache."""
    nb_runner.reset_cash_state()
    nb_runner._run_async(
        nb_runner.client.kc._async_execute_interactive(
            _full_restart_code(vars_to_clear),
            store_history=False,
        )
    )


def test_floor_skipped_upstream_downstream_hits_after_reset(nb_runner):
    """Trivial upstream (compute < 10 ms floor → no metadata written) feeding
    an expensive downstream. After ``reset_cash_state`` (simulating a fresh
    session that keeps the on-disk cache) the upstream is re-computed but
    the downstream must restore from disk."""
    nb_runner.create_notebook([
        # Cell 1: imports
        "import numpy as np\nimport pandas as pd",
        # Cell 2: cheap statement — well under the 10ms floor, so no
        # cache entry is written at all.
        "scale = 17",
        # Cell 3: expensive downstream that consumes `scale`. Engineered
        # to comfortably exceed the floor (~30ms) and produce a multi-MB
        # output so it definitely gets cached under the default policy.
        "big = pd.DataFrame({'x': np.arange(1_000_000, dtype=np.int64) * scale})",
        # Cell 4: stable sentinel so we have something to read back.
        "print(f'sum={int(big[\"x\"].sum())}')",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()

    # --- Cold run: cell 3 is COMPUTED + cached. ---
    nb_runner.run_all()
    out_cold = nb_runner.get_output(4)
    assert 'sum=' in out_cold, f"Cold run missing expected output: {out_cold!r}"
    expected_sum_marker = out_cold.strip()

    # --- Simulate fresh session: clear in-memory state, keep disk cache. ---
    # Clear the value-typed vars but keep np/pd in user_ns — cash's import
    # caching would skip the re-imports on the warm run, leaving them
    # undefined.
    _simulate_restart(nb_runner, ['scale', 'a', 'b', 'big'])

    # --- Warm run: scale is recomputed (no metadata for it on disk), but
    #     cell 3 must restore from the persistent cache. ---
    nb_runner.run_all()
    out_warm = nb_runner.get_output(4)

    assert expected_sum_marker == out_warm.strip(), (
        f"Output differs between cold and warm — caching produced a different "
        f"value than recomputation.\n  cold: {out_cold!r}\n  warm: {out_warm!r}"
    )

    # Cell 3's badge should report the statement as Restored or Skipped
    # (cash has multiple equivalent fast paths — both indicate "did not
    # recompute"). What we MUST NOT see is "Computed" — that would mean
    # cash failed to find the cache entry and re-derived big from scratch.
    cell3 = nb_runner.nb.cells[2]  # 0-indexed
    statuses = _cell_status_words(cell3)
    assert 'Computed' not in statuses, (
        "Cell 3 recomputed on warm run instead of restoring from cache. "
        "This means the lineage chain didn't survive the floor-skipped "
        f"upstream scale=17 — downstream cache key was unstable.\n"
        f"  badge statuses found: {statuses}\n"
        f"  badge html:\n{_get_badge_html(cell3)[:500]}"
    )
    assert statuses & {'Restored', 'Skipped'}, (
        f"Cell 3's badge has no Restored/Skipped indicator — could not "
        f"verify cache reuse. Statuses found: {statuses}"
    )


def test_two_floor_skipped_layers_downstream_still_hits(nb_runner):
    """A chain of two floor-skipped statements (a → b → big). After reset
    both a and b are recomputed; big must still restore. Tests that the
    lineage chain composes correctly across multiple non-persisted hops."""
    nb_runner.create_notebook([
        "import numpy as np\nimport pandas as pd",
        # Two cheap statements in sequence. Neither writes a cache entry;
        # both must be deterministically re-derived to give the same
        # downstream cache key.
        "a = 7",
        "b = a + 5",
        # Expensive downstream — same shape as the prior test.
        "big = pd.DataFrame({'x': np.arange(1_000_000, dtype=np.int64) * b})",
        "print(f'sum={int(big[\"x\"].sum())}')",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()

    nb_runner.run_all()
    out_cold = nb_runner.get_output(5).strip()
    assert 'sum=' in out_cold

    # Clear the value-typed vars but keep np/pd in user_ns — cash's import
    # caching would skip the re-imports on the warm run, leaving them
    # undefined.
    _simulate_restart(nb_runner, ['scale', 'a', 'b', 'big'])
    nb_runner.run_all()
    out_warm = nb_runner.get_output(5).strip()

    assert out_warm == out_cold, (
        f"Downstream value diverged across reset.\n  cold: {out_cold!r}\n"
        f"  warm: {out_warm!r}"
    )
    cell4 = nb_runner.nb.cells[3]  # 0-indexed; the `big = ...` cell
    statuses = _cell_status_words(cell4)
    assert 'Computed' not in statuses, (
        "Cell 4 (big) recomputed on warm run instead of restoring from "
        "cache — the lineage chain didn't survive two floor-skipped "
        f"upstream layers (a, b).\n  badge statuses: {statuses}\n"
        f"  badge html:\n{_get_badge_html(cell4)[:500]}"
    )
    assert statuses & {'Restored', 'Skipped'}, (
        f"Cell 4's badge has no Restored/Skipped indicator. "
        f"Statuses: {statuses}"
    )


def test_changing_floor_skipped_upstream_invalidates_downstream(nb_runner):
    """Regression sanity check: if the floor-skipped upstream's code changes,
    the downstream's cache key MUST change too — otherwise the downstream
    would be silently serving stale data."""
    nb_runner.create_notebook([
        "import numpy as np\nimport pandas as pd",
        "scale = 17",
        "big = pd.DataFrame({'x': np.arange(1_000_000, dtype=np.int64) * scale})",
        "print(f'sum={int(big[\"x\"].sum())}')",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()

    nb_runner.run_all()
    out_v1 = nb_runner.get_output(4).strip()
    assert 'sum=' in out_v1

    # Change the upstream's value (still under the 10 ms floor).
    # Clear the value-typed vars but keep np/pd in user_ns — cash's import
    # caching would skip the re-imports on the warm run, leaving them
    # undefined.
    _simulate_restart(nb_runner, ['scale', 'a', 'b', 'big'])
    nb_runner.set_cell_source(2, "scale = 19")  # 1-based index
    nb_runner.run_all()
    out_v2 = nb_runner.get_output(4).strip()

    assert out_v1 != out_v2, (
        f"Downstream output unchanged after upstream change — cache key "
        f"didn't depend on the floor-skipped upstream's code.\n"
        f"  v1: {out_v1!r}\n  v2: {out_v2!r}"
    )
    # The value-diverged assertion above is load-bearing: if cell 3 had
    # served a stale entry built when `scale=17`, out_v2 would equal
    # out_v1. The fact that it doesn't proves the floor-skipped upstream
    # code change DID flow through to the downstream cache key — the
    # invariant we care about.
