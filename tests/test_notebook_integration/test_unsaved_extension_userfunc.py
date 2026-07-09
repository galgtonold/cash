"""CAS-88: an unsaved cell edit whose new code routes through a user function
must be accepted as the current truth, not discarded in favour of a stale cache.

The bug had two layers: (1) ``_is_valid_extension`` hand-rolled the lineage
projection and omitted the function-source component, so a function-routed edit
always projected != recorded and was rejected; (2) even once kept, the
stale-value guard (``_mark_stale_value_inputs_broken``) re-marked the valid
extension broken, and the forward-probe then restored a stale cache entry keyed
on the outdated saved-notebook lineage. Both are fixed.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(90)]


def test_unsaved_function_routed_extension_is_kept(nb_runner):
    """The classic repro: unsaved edit of a ``df = clean(df)`` cell routing
    through a user function, consumed by a downstream cell."""
    nb_runner.create_notebook([
        "def clean(d):\n    return d.dropna()",
        "import pandas as pd\ndf = pd.DataFrame({'a': [1.0, None, 3.0]})",
        "df = clean(df)",
        "df = df.assign(flag=1)\nprint('idx=' + str(df.index.tolist()))",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "idx=[0, 2]" in nb_runner.get_output(4)

    # UNSAVED edit of cell 3 (do NOT save the notebook file), then execute it so
    # the kernel holds the extended df with a reset index.
    nb_runner.nb.cells[2].source = "df = clean(df).reset_index(drop=True)"
    nb_runner.run_cell(3)

    # Re-run cell 4: it must compute on the live unsaved-extension df (index
    # [0, 1]), not restore the stale cached [0, 2].
    nb_runner.run_cell(4)
    out = nb_runner.get_output(4)
    assert "idx=[0, 1]" in out, (
        f"unsaved function-routed extension discarded; cell4 printed {out!r}"
    )


def test_selfreferential_rerun_still_idempotent(nb_runner):
    """CONTROL (a): a genuine self-referential re-run (``df = df.iloc[1:]``) must
    STILL reset to its cell-entry base on an isolated re-run. The layer-2 skip
    (which spares valid upstream extensions) must NOT suppress the stale-value
    guard for real self-modification — else the re-run would double-truncate."""
    nb_runner.create_notebook([
        "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3, 4, 5]})",
        "df = df.iloc[1:]",
        "print('n=' + str(len(df)))",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "n=4" in nb_runner.get_output(3)

    # Isolated re-run of the self-referential cell must be idempotent (n stays 4,
    # not 3): the guard resets df to its cell-entry base before re-running.
    nb_runner.run_cell(2)
    nb_runner.run_cell(3)
    assert "n=4" in nb_runner.get_output(3), (
        f"self-referential re-run was not idempotent (layer-2 skip over-suppressed the guard?): "
        f"{nb_runner.get_output(3)!r}"
    )


def test_conflicting_upstream_redefinition_still_invalidates(nb_runner):
    """CONTROL (b): a genuinely conflicting upstream redefinition (saved edit)
    must still invalidate the consumer — the fix must not over-trust."""
    nb_runner.create_notebook([
        "import pandas as pd\ndf = pd.DataFrame({'a': [1.0, None, 3.0]})",
        "def clean(d):\n    return d.dropna()",
        "df2 = clean(df)\nprint('n=' + str(len(df2)))",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "n=2" in nb_runner.get_output(3)

    # Saved edit of the upstream data cell → the consumer must recompute.
    nb_runner.set_cell_source(1, "import pandas as pd\ndf = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0]})")
    nb_runner.run_cell(3)
    assert "n=4" in nb_runner.get_output(3), (
        f"conflicting upstream redefinition did not invalidate consumer: {nb_runner.get_output(3)!r}"
    )
