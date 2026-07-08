"""Derivation / alias edge invalidation (CAS-115 + CAS-89).

Some objects hold a *live* reference to another object that the lineage
system never models, so a later in-place mutation of one side is invisible
to a cached consumer of the other:

* **CAS-115 (base -> derived):** ``g = df.groupby('k')`` holds a live
  reference to ``df``. A later ``df.iloc[...] = v`` must bump ``g``'s lineage
  so a consumer of ``g['v'].sum()`` recomputes.
* **CAS-89 (view -> base):** ``v = a[100:200]`` is a numpy view
  (``v.base is a``). A later ``v[:] = k`` mutates ``a`` in place and must bump
  ``a``'s lineage so a consumer of ``a.sum()`` recomputes.

The fix is a *derivation edge store* on ``TrackingState`` plus the rule that
live-alias objects (views, groupby/rolling ref-holders) are never cache-
restored (restoring decouples them from the live base). Ground truth for each
scenario comes from a plain (cash-off) kernel run of the same notebook.

Controls guard the two failure modes of such a fix:

* CONTROL (a): ``v = a.copy()`` is NOT an alias -> mutating ``v`` must NOT
  invalidate ``a``'s consumer.
* CONTROL (c): an unrelated variable's consumer must stay cached (the
  edge-walk must not leak into vars with no edge).
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]


def _plain_then_cash(nb_runner, cells, edit_idx, edit_src, out_idx):
    """Return (plain_edit_output, cash_edit_output) for the same edited notebook.

    Runs the notebook in a plain (cash-off) kernel to establish ground truth for
    the edited cell, then in a cash kernel; the cash output must match plain.
    """
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()
    nb_runner.set_cell_source(edit_idx, edit_src)
    nb_runner.run_all()
    plain_edit = nb_runner.get_output(out_idx).strip()
    nb_runner.shutdown()

    # Restore the pre-edit source for the cash run so it caches the original,
    # then apply the same edit.
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel(with_cash=True)
    nb_runner.enable_persist()
    nb_runner.run_all()
    nb_runner.set_cell_source(edit_idx, edit_src)
    nb_runner.run_all()
    cash_edit = nb_runner.get_output(out_idx).strip()
    return plain_edit, cash_edit


def test_cas115_groupby_reflects_post_creation_frame_mutation(nb_runner):
    """CAS-115: a groupby created before an edit to its source frame must
    aggregate the edited frame."""
    cells = [
        "import pandas as pd\ndf = pd.DataFrame({'k': [0, 1] * 500, 'v': list(range(1000))})",
        "g = df.groupby('k')",
        "df.iloc[999, 1] = 0",
        "print('sums=', g['v'].sum().to_dict())",
    ]
    plain, cash = _plain_then_cash(
        nb_runner, cells, edit_idx=3, edit_src="df.iloc[999, 1] = 5", out_idx=4
    )
    assert "sums=" in plain
    assert plain in cash, f"groupby aggregate stale after frame edit: plain={plain!r} cash={cash!r}"


def test_cas89_view_mutation_invalidates_base_consumer(nb_runner):
    """CAS-89: mutating a numpy view must invalidate a cached consumer of the
    base array."""
    cells = [
        "import numpy as np\na = np.arange(300)",
        "b = a[200:250]",
        "b[:] = 9",
        "print('asum=', int(a.sum()))",
    ]
    plain, cash = _plain_then_cash(
        nb_runner, cells, edit_idx=3, edit_src="b[:] = 7", out_idx=4
    )
    assert "asum=" in plain
    assert plain in cash, f"base consumer stale after view edit: plain={plain!r} cash={cash!r}"


def test_control_a_copy_does_not_invalidate_base(nb_runner):
    """CONTROL (a): a .copy() is independent -> mutating it must NOT invalidate
    the base's consumer (no false edge)."""
    cells = [
        "import numpy as np\na = np.arange(100)",
        "v = a.copy()",
        "v += 3",
        "print('asum=', int(a.sum()))",
    ]
    plain, cash = _plain_then_cash(
        nb_runner, cells, edit_idx=3, edit_src="v += 4", out_idx=4
    )
    # arange(100).sum() == 4950 regardless of the edit to the copy.
    assert plain == "asum= 4950", plain
    assert plain in cash, f"copy edit wrongly changed base consumer: plain={plain!r} cash={cash!r}"


def test_control_b_view_of_view_chains_to_root_base(nb_runner):
    """CONTROL (b): a view-of-view must chain to the ultimate base so mutating
    the inner view invalidates a consumer of the root."""
    cells = [
        "import numpy as np\na = np.arange(300)",
        "v = a[100:200]",
        "w = v[10:20]",
        "w[:] = 0",
        "print('asum=', int(a.sum()))",
    ]
    plain, cash = _plain_then_cash(
        nb_runner, cells, edit_idx=4, edit_src="w[:] = 1", out_idx=5
    )
    assert "asum=" in plain
    assert plain in cash, f"root base stale after view-of-view edit: plain={plain!r} cash={cash!r}"


def test_control_c_unrelated_var_stays_cached(nb_runner):
    """CONTROL (c): the edge-walk must not leak — an unrelated variable's
    consumer stays valid (matches plain kernel) when a view mutation is edited."""
    cells = [
        "import numpy as np\na = np.arange(300)\nother = np.arange(50)",
        "b = a[200:250]",
        "b[:] = 9",
        "print('other_sum=', int(other.sum()), 'asum=', int(a.sum()))",
    ]
    plain, cash = _plain_then_cash(
        nb_runner, cells, edit_idx=3, edit_src="b[:] = 7", out_idx=4
    )
    # other is arange(50).sum() == 1225 in every run; a changes with the edit.
    assert "other_sum= 1225" in plain, plain
    assert plain in cash, f"unrelated-var consumer diverged from plain: plain={plain!r} cash={cash!r}"
