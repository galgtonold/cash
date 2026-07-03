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

FIXED for no-lineage self-modifying inputs (the second section below): primitives
/ builtin containers / ndarray carry no ``_cash_lineage_hash``, and their
self-modifying statements skip the per-statement cache (missing input lineage), so
an isolated re-run computes on the cell's own prior output. The stale-value
guard's no-lineage branch detects this two ways -- a self-modifying *output*
whose consumed cell-entry base lineage (``executed_input_lineages[var][var]``) now
differs from its advanced recorded lineage (``total = total + k``, ``arr += 1``,
``lst = lst + [..]``), and a pure in-place *mutation* whose
``current_session_hashes[var]`` (never advanced by a no-output mutation, so still
the producer's content hash) differs from the live content hash (``lst.append``).
Both self-disable on ``run_all`` (the producer restores the base first).

STILL xfail (documented known limitations -- they fail identically on the pre-fix
baseline, not regressions):
  * interdependent multi-target swap (``a, b = b, a``): both outputs share one
    statement lineage, so the per-variable base cannot distinguish ``a`` from
    ``b``;
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

    # FIXED (CAS-43, first channel) as fallout of the 2026-07-03 fix batch:
    # the multi-target swap now restores the pair correctly on isolated
    # re-run (verified deterministic across repeated --runxfail runs).
    def test_primitive_tuple_swap(self, nb_runner):
        _two_cell(nb_runner, "a = 1\nb = 2", "a, b = b, a\nprint(f'{a},{b}')", "2,1")

    # FIXED: no-lineage self-reassignment. The stale-value guard's no-lineage
    # branch compares the consumed cell-entry base (executed_input_lineages /
    # current_session_hashes) against the live value and restores the base.

    def test_builtin_list_reassign_grow(self, nb_runner):
        _two_cell(nb_runner, "lst = [0]", "lst = lst + [len(lst)]\nprint(lst)", "[0, 1]")

    def test_builtin_dict_reassign_grow(self, nb_runner):
        _two_cell(nb_runner, "d = {'a': 1}", "d = {**d, str(len(d)): len(d)}\nprint(sorted(d))", "['1', 'a']")

    # ---- in-place mutation accumulators ----
    # FIXED: a no-lineage var mutated in place produces no output, so
    # current_session_hashes still holds the upstream producer's content hash
    # (the base); a live content hash that differs flags the stale value.

    def test_list_append_inplace_rerun(self, nb_runner):
        _two_cell(nb_runner, "lst = [1, 2]", "lst.append(99)\nprint(lst)", "[1, 2, 99]")

    def test_numpy_inplace_augmented(self, nb_runner):
        _two_cell(
            nb_runner,
            "import numpy as np\narr = np.array([1, 2, 3])",
            "arr += 1\nprint(arr.tolist())",
            "[2, 3, 4]",
        )

    # ---- control structure self-reference ----
    # FIXED (same no-lineage branch): 'total' is a self-modifying output whose
    # consumed base lineage differs from the advanced recorded lineage on re-run.

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
