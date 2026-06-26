"""Catalogue: isolated-cell re-run gaps for self-modifying cells.

Each test mirrors a realistic notebook pattern and re-runs ONE cell in isolation
(``run_cell``, NOT ``run_all``) under default persistence. That is the regime
where sub-ms statements fall below the cost floor and stay uncached, so
correctness depends entirely on the upstream machinery upholding the
"executing a cell == running everything from the start" guarantee.

ROOT CAUSE: a self-modifying cell only has its input variable restored to the
cell-entry (base) value when the upstream simulation flags that variable as
stale. The 4fb5249 stale-value guard detected this only when the recorded
``variable_lineage`` had been reset to the pre-cell base (creating a mismatch
with the live value's ``_cash_lineage_hash``) -- which the downstream-advancement
reset only does for *multi-statement* chains (test_134). A single-statement
``df = df.iloc[1:]`` produces no forward-sim mismatch at all
(recorded == live == virtual == the advanced value), so the reset never fires
and the cell re-executes on its own previous output.

FIXED for lineage-carrying pure-reassignment (the first section below): the
stale-value guard now also consults ``executed_input_lineages[var][var]`` -- the
version the cell *consumed* the last time it ran. When the live value's lineage
differs from that recorded base, the namespace holds the cell's own prior output
rather than the base, so the var is marked broken and the same restore machinery
re-derives the base. Confirmed via the guard probe: the single-statement
``df = df.iloc[1:]`` gets recorded == live == advanced but base_input == base
!= live, which is the new signal.

STILL xfail (the remaining sections, documented known limitations -- they fail
identically on the pre-fix baseline, not regressions):
  * primitives / builtin containers (int swap, list/dict reassign) carry no
    ``_cash_lineage_hash``, so neither guard branch can see them;
  * in-place mutation accumulators (``lst.append``, ``arr += 1``) replay through
    a separate mutation-lineage path and are excluded from the reassign gate;
  * mutate+reassign in one cell (``df['c']=...; df=df.rename(...)``) is excluded
    because the var lands in ``modified_objects``.
"""

import pytest


def _two_cell(nb_runner, setup, cell, expect):
    nb_runner.create_notebook([setup, cell])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert expect in nb_runner.get_output(2), f"first run: {nb_runner.get_output(2)!r}"
    nb_runner.run_cell(2)
    assert expect in nb_runner.get_output(2), f"re-run: {nb_runner.get_output(2)!r}"


_GAP3 = "re-executes on its own previous output. See module docstring."


class TestIsolatedRerunGaps:

    # ---- object pure-reassignment that is NOT accidentally idempotent ----
    # FIXED: the stale-value guard's executed_input_lineages branch restores the
    # cell-entry base before these self-modifying single statements re-run.

    def test_object_pure_reassign_slice(self, nb_runner):
        _two_cell(
            nb_runner,
            "import pandas as pd\ndf = pd.DataFrame({'a': [1,2,3,4]})",
            "df = df.iloc[1:]\nprint(len(df))",
            "3",
        )

    def test_object_pure_reassign_concat_doubles(self, nb_runner):
        _two_cell(
            nb_runner,
            "import pandas as pd\ndf = pd.DataFrame({'a': [1,2]})",
            "df = pd.concat([df, df])\nprint(len(df))",
            "4",
        )

    # ---- object mutate+reassign (Bug A: gate excludes via modified_objects) ----

    @pytest.mark.xfail(reason="Bug A: cell both mutates (df['c']=) and reassigns "
                              "(df=df.rename); the var is in modified_objects so the "
                              "reassigned-names gate excludes it. Re-run reads the renamed "
                              "frame -> KeyError. Same idempotency tension as the "
                              "downstream-advancement case.", strict=False)
    def test_object_mutate_then_reassign_nonidempotent(self, nb_runner):
        _two_cell(
            nb_runner,
            "import pandas as pd\ndf = pd.DataFrame({'a': [3,1,2], 'b': [6,4,5]})",
            "df['c'] = df['a'] * 2\ndf = df.rename(columns={'a': 'x'})\nprint(df.columns.tolist())",
            "['x', 'b', 'c']",
        )

    # ---- primitives / builtin containers (no _cash_lineage_hash attribute) ----

    @pytest.mark.xfail(reason="Bug B: int swap; primitives carry no _cash_lineage_hash so "
                              "the guard skips them. " + _GAP3, strict=False)
    def test_primitive_tuple_swap(self, nb_runner):
        _two_cell(nb_runner, "a = 1\nb = 2", "a, b = b, a\nprint(f'{a},{b}')", "2,1")

    @pytest.mark.xfail(reason="Bug B family: list reassigned from itself; builtin list has "
                              "no lineage attr. " + _GAP3, strict=False)
    def test_builtin_list_reassign_grow(self, nb_runner):
        _two_cell(nb_runner, "lst = [0]", "lst = lst + [len(lst)]\nprint(lst)", "[0, 1]")

    @pytest.mark.xfail(reason="Bug B family: dict reassigned from itself. " + _GAP3,
                       strict=False)
    def test_builtin_dict_reassign_grow(self, nb_runner):
        _two_cell(nb_runner, "d = {'a': 1}", "d = {**d, str(len(d)): len(d)}\nprint(sorted(d))", "['1', 'a']")

    # ---- in-place mutation accumulators (separate mutation-lineage path) ----

    @pytest.mark.xfail(reason="In-place accumulator: lst.append re-run doubles "
                              "([1,2,99]->[1,2,99,99]). Mutation-lineage path does not "
                              "restore the base on isolated re-run. " + _GAP3, strict=False)
    def test_list_append_inplace_rerun(self, nb_runner):
        _two_cell(nb_runner, "lst = [1, 2]", "lst.append(99)\nprint(lst)", "[1, 2, 99]")

    @pytest.mark.xfail(reason="In-place accumulator: arr += 1 (numpy in-place) adds twice "
                              "on re-run. " + _GAP3, strict=False)
    def test_numpy_inplace_augmented(self, nb_runner):
        _two_cell(
            nb_runner,
            "import numpy as np\narr = np.array([1, 2, 3])",
            "arr += 1\nprint(arr.tolist())",
            "[2, 3, 4]",
        )

    # ---- control structure self-reference ----

    @pytest.mark.xfail(reason="Loop accumulator: 'for k in ...: total = total + k' doubles "
                              "(6->12) on isolated re-run. " + _GAP3, strict=False)
    def test_for_loop_accumulate_rerun(self, nb_runner):
        _two_cell(
            nb_runner,
            "total = 0",
            "for k in [1, 2, 3]:\n    total = total + k\nprint(total)",
            "6",
        )

    # FIXED (same mechanism): df is purely reassigned inside the if-branch.
    def test_conditional_self_reassign(self, nb_runner):
        _two_cell(
            nb_runner,
            "import pandas as pd\ndf = pd.DataFrame({'a': [1,2,3]})\nflag = True",
            "if flag:\n    df = df.iloc[1:]\nprint(len(df))",
            "2",
        )

    # ---- positive coverage: passes today (note: some only by accidental idempotence) ----

    def test_object_reassign_via_user_func_idempotent(self, nb_runner):
        """df = transform(df) where transform renames 'a'->'x'. Passes, but NOTE: only
        because pandas rename silently ignores the already-missing 'a' on re-run
        (accidentally idempotent), not because the base is restored."""
        _two_cell(
            nb_runner,
            "import pandas as pd\n"
            "def transform(d):\n    return d.rename(columns={'a': 'x'})\n"
            "df = pd.DataFrame({'a': [1,2], 'b': [3,4]})",
            "df = transform(df)\nprint(df.columns.tolist())",
            "['x', 'b']",
        )
