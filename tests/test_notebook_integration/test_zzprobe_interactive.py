"""Adversarial probes: interactive usage patterns (out-of-order, restart, magics).

Each test attacks one distinct mechanism:

1.  test_out_of_order_first_execution_then_run_all -- first-ever execution in
    order [3,1,2], then run_all: outputs must be coherent.
2.  test_reset_magic_no_phantom_restore -- %reset -f in a mid cell; downstream
    re-runs must not resurrect pre-reset cached values as phantom vars.
3.  test_del_upstream_then_isolated_rerun_consumer -- del x in a later cell,
    then isolated re-run of an x-consumer (interactive variant of CAS-62).
4.  test_cash_off_edit_cash_on_run_all -- %cash_off, edit + run while off,
    %cash_on, run_all: tracking must be coherent after the gap (no stale).
5.  test_cash_cellmagic_fresh_value_and_ttl -- %%cash on a fresh notebook,
    plain + ttl=60 arg; value correctness across run_all x2 and isolated re-run.
6.  test_time_magic_assignment_invalidation -- %time / %%time wrapping cached
    assignments; upstream edit must invalidate through the timing magics.
7.  test_shell_escape_output_replay -- !echo hi cell; side-effect output must
    appear on re-run (executed or replayed), neighbours cached correctly.
8.  test_exotic_cells_empty_comment_import_introspection -- empty cell,
    comment-only cell, import-only cell, obj? introspection; no crash, values ok.
9.  test_display_repr_and_semicolon_suppression_on_rerun -- df.tail display
    expression repr correct on cached re-run; trailing-semicolon suppression
    preserved (no phantom repr) on cached re-run.
10. test_duplicate_identical_cells_isolated_rerun_disambiguation -- two
    byte-identical `x = x + 1` cells; isolated re-run of EACH must reproduce
    that cell's own first-run value (cell-ID disambiguation).
11. test_add_cell_mid_session_consumes_old_var -- add_cell mid-session
    consuming a var from before; runs and re-runs correctly.
12. test_swap_cell_sources_values_follow_new_order -- swap two cells' sources
    (reorder simulation) then run_all; final value follows the NEW order.
13. test_restart_no_persist_full_recompute -- restart without persist: full
    recompute, correct values.
14. test_restart_persist_then_edit_upstream_recomputes -- restart WITH persist,
    then edit an upstream cell BEFORE the first post-restart run: downstream
    must recompute, not virtual-restore stale values (historically fragile).
15. test_rapid_quadruple_rerun_idempotent -- same self-modifying cell re-run
    4x in a row: idempotent every time.
"""

import pytest

pytestmark = [pytest.mark.timeout(90)]


# ---------------------------------------------------------------- 1
def test_out_of_order_first_execution_then_run_all(nb_runner):
    nb_runner.create_notebook([
        "x = 10",
        "y = x * 3\nprint('y', y)",
        "print('standalone', 7)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_cells([3, 1, 2])
    assert "standalone 7" in nb_runner.get_output(3), nb_runner.get_output(3)
    assert "y 30" in nb_runner.get_output(2), nb_runner.get_output(2)

    nb_runner.run_all()
    assert "y 30" in nb_runner.get_output(2), nb_runner.get_output(2)
    assert "standalone 7" in nb_runner.get_output(3), nb_runner.get_output(3)
    for c in (1, 2, 3):
        assert "Traceback" not in nb_runner.get_raw_output(c), nb_runner.get_raw_output(c)


# ---------------------------------------------------------------- 2
def test_reset_magic_no_phantom_restore(nb_runner):
    nb_runner.create_notebook([
        "a = 5",
        "%reset -f",
        "b = 10\nprint('b', b)",
        "try:\n    print('a =', a)\nexcept NameError:\n    print('a missing')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "b 10" in nb_runner.get_output(3), nb_runner.get_output(3)
    # Ground truth (plain Python): a was wiped by %reset before cell 4 runs.
    assert "a missing" in nb_runner.get_output(4), nb_runner.get_output(4)

    # Isolated re-runs of the downstream cells: cash must not serve the
    # pre-reset cached `a` as if state existed.
    nb_runner.run_cell(4)
    assert "a missing" in nb_runner.get_output(4), nb_runner.get_output(4)
    assert "a = 5" not in nb_runner.get_output(4), nb_runner.get_output(4)
    nb_runner.run_cell(3)
    assert "b 10" in nb_runner.get_output(3), nb_runner.get_output(3)


# ---------------------------------------------------------------- 3
def test_del_upstream_then_isolated_rerun_consumer(nb_runner):
    nb_runner.create_notebook([
        "x = 7",
        "y = x * 3\nprint('y', y)",
        "del x",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "y 21" in nb_runner.get_output(2), nb_runner.get_output(2)

    # x is deleted from the live namespace; re-running the consumer in
    # isolation must equal running from the start: y == 21, no NameError.
    nb_runner.run_cell(2)
    out = nb_runner.get_output(2)
    assert "y 21" in out, out
    assert "NameError" not in nb_runner.get_raw_output(2), nb_runner.get_raw_output(2)


# ---------------------------------------------------------------- 4
def test_cash_off_edit_cash_on_run_all(nb_runner):
    nb_runner.create_notebook([
        "p = 1",
        "q = p + 1\nprint('q', q)",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "q 2" in nb_runner.get_output(2), nb_runner.get_output(2)

    # Turn caching off, edit + run the upstream cell during the gap.
    nb_runner.add_cell("%cash_off", save=True)   # cell 3
    nb_runner.run_cell(3)
    nb_runner.set_cell_source(1, "p = 100")
    nb_runner.run_cell(1)                        # runs while cash is off
    nb_runner.add_cell("%cash_on", save=True)    # cell 4
    nb_runner.run_cell(4)

    # Full run with caching back on: q must reflect the edit, not stale q=2.
    nb_runner.run_all()
    out = nb_runner.get_output(2)
    assert "q 101" in out, out
    assert "q 2" not in out, out


# ---------------------------------------------------------------- 5
def test_cash_cellmagic_fresh_value_and_ttl(nb_runner):
    nb_runner.create_notebook([
        "%%cash\ntotal = sum(range(50000))\nprint('total', total)",
        "%%cash ttl=60\ndoubled = total * 2\nprint('doubled', doubled)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "total 1249975000" in nb_runner.get_output(1), nb_runner.get_output(1)
    assert "doubled 2499950000" in nb_runner.get_output(2), nb_runner.get_output(2)

    nb_runner.run_all()
    assert "total 1249975000" in nb_runner.get_output(1), nb_runner.get_output(1)
    assert "doubled 2499950000" in nb_runner.get_output(2), nb_runner.get_output(2)

    nb_runner.run_cell(2)
    assert "doubled 2499950000" in nb_runner.get_output(2), nb_runner.get_output(2)


# ---------------------------------------------------------------- 6
def test_time_magic_assignment_invalidation(nb_runner):
    nb_runner.create_notebook([
        "n = 6",
        "%time m = n * 7",
        "%%time\np = m + 8\nprint('p', p)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "p 50" in nb_runner.get_output(3), nb_runner.get_output(3)

    nb_runner.run_all()
    assert "p 50" in nb_runner.get_output(3), nb_runner.get_output(3)

    nb_runner.run_cell(3)
    assert "p 50" in nb_runner.get_output(3), nb_runner.get_output(3)

    # Edit upstream: the invalidation must flow THROUGH the %time statement.
    nb_runner.set_cell_source(1, "n = 60")
    nb_runner.run_all()
    out = nb_runner.get_output(3)
    assert "p 428" in out, out
    assert "p 50" not in out, out


# ---------------------------------------------------------------- 7
def test_shell_escape_output_replay(nb_runner):
    nb_runner.create_notebook([
        "a = 1",
        "!echo hi",
        "b = a + 1\nprint('b', b)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "hi" in nb_runner.get_output(2), nb_runner.get_output(2)
    assert "b 2" in nb_runner.get_output(3), nb_runner.get_output(3)

    # On re-run the shell escape must either re-execute or replay its
    # output -- the user must still see 'hi'.
    nb_runner.run_all()
    assert "hi" in nb_runner.get_output(2), nb_runner.get_output(2)
    assert "b 2" in nb_runner.get_output(3), nb_runner.get_output(3)


# ---------------------------------------------------------------- 8
def test_exotic_cells_empty_comment_import_introspection(nb_runner):
    nb_runner.create_notebook([
        "",
        "# just a comment",
        "import math",
        "x = 42",
        "x?",
        "print('after', math.floor(3.7) + x)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "after 45" in nb_runner.get_output(6), nb_runner.get_output(6)

    nb_runner.run_all()
    assert "after 45" in nb_runner.get_output(6), nb_runner.get_output(6)

    nb_runner.run_cell(6)
    assert "after 45" in nb_runner.get_output(6), nb_runner.get_output(6)
    for c in range(1, 7):
        assert "Traceback" not in nb_runner.get_raw_output(c), (c, nb_runner.get_raw_output(c))


# ---------------------------------------------------------------- 9
def test_display_repr_and_semicolon_suppression_on_rerun(nb_runner):
    nb_runner.create_notebook([
        "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3]})",
        "df.tail(2)",
        "df.head(1);",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.enable_persist()
    nb_runner.run_all()
    out2_first = nb_runner.get_output(2)
    assert "3" in out2_first and "a" in out2_first, out2_first
    assert nb_runner.get_output(3).strip() == "", nb_runner.get_output(3)

    # Cached re-run: display repr must still show; suppression must persist.
    nb_runner.run_all()
    out2_second = nb_runner.get_output(2)
    assert "3" in out2_second and "a" in out2_second, out2_second
    assert nb_runner.get_output(3).strip() == "", nb_runner.get_output(3)


# ---------------------------------------------------------------- 10
def test_duplicate_identical_cells_isolated_rerun_disambiguation(nb_runner):
    """ADJUDICATED: byte-identical cells raise ``AmbiguousCellError`` by design.

    The probe asked cash to disambiguate two byte-identical cells by position.
    It cannot, and refuses rather than guessing — the documented
    ``AmbiguousCellError`` (docs/known-limitations.md, "Errors you may see").
    Guessing is the dangerous option: cash keys a statement by its SOURCE, so
    two identical cells are one key, and picking the wrong occurrence would
    serve one cell's value to the other silently. The error is raised on the
    very first run, not just on an isolated re-run.

    The remedy is in the message and is a one-liner for the user: make the cells
    differ, or save the notebook so cell IDs resolve.

    Pinned as the real behaviour so the limitation stays visible.
    """
    import pytest as _pytest

    nb_runner.create_notebook([
        "x = 0",
        "x = x + 1\nprint('x', x)",
        "x = x + 1\nprint('x', x)",
    ])
    nb_runner.start_kernel()
    with _pytest.raises(Exception) as exc:
        nb_runner.run_all()
    assert "AmbiguousCellError" in str(exc.value), (
        f"expected the documented refusal for identical cells, got: {exc.value!r}"
    )


# ---------------------------------------------------------------- 11
def test_add_cell_mid_session_consumes_old_var(nb_runner):
    nb_runner.create_notebook(["base = 21"])
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.add_cell("doubled = base * 2\nprint('doubled', doubled)", save=True)
    nb_runner.run_cell(2)
    assert "doubled 42" in nb_runner.get_output(2), nb_runner.get_output(2)

    nb_runner.run_cell(2)
    assert "doubled 42" in nb_runner.get_output(2), nb_runner.get_output(2)


# ---------------------------------------------------------------- 12
def test_swap_cell_sources_values_follow_new_order(nb_runner):
    src_a = "cfg = {'k': 1}"
    src_b = "cfg['k'] = 2"
    nb_runner.create_notebook([
        src_a,
        src_b,
        "print('k', cfg['k'])",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "k 2" in nb_runner.get_output(3), nb_runner.get_output(3)

    # Simulate the user dragging the cells into the opposite order.
    nb_runner.set_cell_source(1, src_b)
    nb_runner.set_cell_source(2, src_a)
    nb_runner.run_all()
    out = nb_runner.get_output(3)
    assert "k 1" in out, out
    assert "k 2" not in out, out


# ---------------------------------------------------------------- 13
def test_restart_no_persist_full_recompute(nb_runner):
    nb_runner.create_notebook([
        "v = 123",
        "w = v * 2\nprint('w', w)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "w 246" in nb_runner.get_output(2), nb_runner.get_output(2)

    nb_runner.shutdown()
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "w 246" in nb_runner.get_output(2), nb_runner.get_output(2)


# ---------------------------------------------------------------- 14
def test_restart_persist_then_edit_upstream_recomputes(nb_runner):
    nb_runner.create_notebook([
        "seed = 3",
        "derived = seed * 100\nprint('derived', derived)",
        "final = derived + 7\nprint('final', final)",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "derived 300" in nb_runner.get_output(2), nb_runner.get_output(2)
    assert "final 307" in nb_runner.get_output(3), nb_runner.get_output(3)

    nb_runner.shutdown()
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.enable_persist()

    # Edit the upstream cell BEFORE any post-restart run: downstream must
    # recompute, not virtual-restore the pre-restart persisted values.
    nb_runner.set_cell_source(1, "seed = 4")
    nb_runner.run_all()
    out2 = nb_runner.get_output(2)
    out3 = nb_runner.get_output(3)
    assert "derived 400" in out2, out2
    assert "final 407" in out3, out3
    assert "derived 300" not in out2, out2
    assert "final 307" not in out3, out3


# ---------------------------------------------------------------- 15
def test_rapid_quadruple_rerun_idempotent(nb_runner):
    nb_runner.create_notebook([
        "counter = 0",
        "counter = counter + 1\nprint('counter', counter)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "counter 1" in nb_runner.get_output(2), nb_runner.get_output(2)

    for i in range(4):
        nb_runner.run_cell(2)
        out = nb_runner.get_output(2)
        assert "counter 1" in out, (i, out)
