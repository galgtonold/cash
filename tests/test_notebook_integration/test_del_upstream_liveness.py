"""CAS-94: position-aware liveness for ``del`` in upstream reconstruction.

``del x`` is invisible to cash's AST-output analysis (it produces no Store
target) and removes ``x`` from the live namespace while leaving a stale
``variable_lineage['x']`` entry behind. The upstream simulator now (a) gates the
missing-input check on the LIVE namespace and (b) models a bare-name ``del`` as
a position-scoped virtual-lineage removal, so an isolated re-run of a consumer
that sits ABOVE the ``del`` reconstructs the deleted upstream from its producer
rather than raising ``NameError``.

The four controls pin the exact boundary of the fix:
  main : consumer ABOVE the del -> reconstruct upstream, no NameError.
  (a)  : del then REDEFINE above the consumer -> restore the NEW value.
  (b)  : consumer BELOW the del -> genuinely-killed var stays dead (NameError).
  (c)  : del in the SAME cell as the consumer -> unchanged.
"""

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.integration, pytest.mark.timeout(90)]


def test_del_upstream_then_isolated_rerun_consumer(nb_runner):
    """Main case: del BELOW an x-consumer; isolated re-run rebuilds x=7."""
    nb_runner.create_notebook([
        "x = 7",
        "y = x * 3\nprint('y', y)",
        "del x",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "y 21" in nb_runner.get_output(2), nb_runner.get_output(2)

    # x is gone from the live namespace; re-running the consumer in isolation
    # must equal running from the start: y == 21, no NameError.
    nb_runner.run_cell(2)
    out = nb_runner.get_output(2)
    assert "y 21" in out, out
    assert "NameError" not in nb_runner.get_raw_output(2), nb_runner.get_raw_output(2)


def test_del_then_redefine_above_consumer_restores_new_value(nb_runner):
    """(a) del then a NEW binding above the consumer -> reconstruct the NEW value."""
    nb_runner.create_notebook([
        "x = 7",
        "del x",
        "x = 99",
        "y = x * 3\nprint('y', y)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "y 297" in nb_runner.get_output(4), nb_runner.get_output(4)

    # Isolated re-run of the consumer must use the post-redefine value (99*3),
    # NOT the original x=7 (21) and NOT a NameError.
    nb_runner.run_cell(4)
    out = nb_runner.get_output(4)
    assert "y 297" in out, out
    assert "y 21" not in out, out
    assert "NameError" not in nb_runner.get_raw_output(4), nb_runner.get_raw_output(4)


def test_consumer_below_del_stays_dead(nb_runner):
    """(b) consumer BELOW the del -> genuinely-killed var must NOT be resurrected.

    Covers both run_all and an isolated re-run of the below-del cell. The del
    sits ABOVE the consumer, so cash must leave the name dead in both — never
    over-resurrect. The NameError surfaces in the raised CellExecutionError
    (matching the CAS-62 sibling's assertion style, which does not rely on
    error text being captured into the cell's structured outputs).
    """
    nb_runner.create_notebook([
        "x = 7",
        "del x",
        "print(x)",
    ])
    nb_runner.start_kernel()

    # run_all: cell 3 reads a genuinely-deleted x -> NameError, no phantom value.
    with pytest.raises(CellExecutionError) as ei_all:
        nb_runner.run_all()
    assert ei_all.value.ename == "NameError", ei_all.value.ename

    # Isolated re-run of the below-del cell: still dead, still a NameError.
    with pytest.raises(CellExecutionError) as ei_iso:
        nb_runner.run_cell(3)
    assert ei_iso.value.ename == "NameError", ei_iso.value.ename


def test_del_in_same_cell_as_consumer_unchanged(nb_runner):
    """(c) del in the SAME cell as the consumer -> behaviour unchanged."""
    nb_runner.create_notebook([
        "x = 7",
        "y = x * 3\nprint('y', y)\ndel x",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "y 21" in nb_runner.get_output(2), nb_runner.get_output(2)

    # Isolated re-run: the consumer reads x (still built above) before the
    # in-cell del removes it -> y == 21, no NameError.
    nb_runner.run_cell(2)
    out = nb_runner.get_output(2)
    assert "y 21" in out, out
    assert "NameError" not in nb_runner.get_raw_output(2), nb_runner.get_raw_output(2)
