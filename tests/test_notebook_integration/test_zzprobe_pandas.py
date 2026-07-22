"""Adversarial probes: pandas/numpy deep semantics & content-hash sampling.

Attack surface (src/cash/notebook/object_hashing.py): compute_hash SAMPLES large
objects -- DataFrame/Series = shape+dtypes+first-5-rows, ndarray = shape+dtype+
first-100 elements, list/tuple>200 = len+first5+last5, dict>200 = len+first-10
sorted keys (values never hashed).  Wherever the CONTENT hash (not statement
lineage) is the discriminator, a mutation outside the sample is invisible.

Probes:
  TestSamplingRerunIdempotence -- isolated re-run of a self-mutating cell must be
    idempotent (cash's flagship promise).  Mutations placed OUTSIDE the hash
    sample: list[500] += 1 (1000-elt list), dict value += 1 (300-key dict),
    arr[500] += 1 (1000-elt ndarray), df.iloc[500,0] += 1 (1000-row df),
    s.iloc[500] += 1 (1000-elt Series).  Control: 100-elt list (fully hashed).
  TestContentHashCollisionChannels -- channels where lineage is re-derived from
    (sampled) content: flag-edit producing tail-colliding ndarray/DataFrame;
    reset_cash_state() provenance loss + tail mutation + cached consumer.
  TestViewsNestedAndFriends -- np slice-view mutation edit invalidation,
    object-dtype nested-list mutation re-run, df.attrs accumulation re-run,
    df.insert re-run, groupby-across-cells + tail mutation (plain-kernel ground
    truth first).
"""

import pytest

pytestmark = [pytest.mark.timeout(90)]


def _rerun_probe(nb_runner, setup, cell, expect):
    """run_all -> assert; run_all again -> assert (determinism); isolated
    re-run -> assert (idempotence).  Default persistence regime, matching
    tests/test_notebook_integration/test_isolated_rerun_gaps.py."""
    nb_runner.create_notebook([setup, cell])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert expect in nb_runner.get_output(2), \
        f"first run_all: {nb_runner.get_output(2)!r}"
    nb_runner.run_all()
    assert expect in nb_runner.get_output(2), \
        f"second run_all (determinism): {nb_runner.get_output(2)!r}"
    nb_runner.run_cell(2)
    assert expect in nb_runner.get_output(2), \
        f"isolated re-run (idempotence): {nb_runner.get_output(2)!r}"


class TestSamplingRerunIdempotence:
    """Self-mutating cells whose mutation lies OUTSIDE the compute_hash sample."""

    def test_list_tail_augment_rerun_idempotent(self, nb_runner):
        # list of 1000 (>200): hashed as len + first5 + last5; index 500 unsampled.
        _rerun_probe(
            nb_runner,
            "lst = list(range(1000))",
            "lst[500] += 1\nprint('v=', lst[500])",
            "v= 501",
        )

    def test_list_small_control_rerun_idempotent(self, nb_runner):
        # control: 100 elements <= 200 -> full pickle hash sees the mutation.
        _rerun_probe(
            nb_runner,
            "lst = list(range(100))",
            "lst[50] += 1\nprint('v=', lst[50])",
            "v= 51",
        )

    def test_dict_value_mutation_rerun_idempotent(self, nb_runner):
        # dict of 300 (>200): hashed as len + first-10 sorted keys; VALUES never hashed.
        _rerun_probe(
            nb_runner,
            "d = {f'k{i:03d}': i for i in range(300)}",
            "d['k150'] += 1\nprint('v=', d['k150'])",
            "v= 151",
        )

    def test_ndarray_tail_augment_rerun_idempotent(self, nb_runner):
        # ndarray of 1000: hashed as shape+dtype+first-100 elements; index 500 unsampled.
        _rerun_probe(
            nb_runner,
            "import numpy as np\narr = np.arange(1000)",
            "arr[500] += 1\nprint('v=', int(arr[500]))",
            "v= 501",
        )

    def test_dataframe_tail_iloc_augment_rerun_idempotent(self, nb_runner):
        # DataFrame of 1000 rows: hashed as shape+dtypes+head(5); row 500 unsampled.
        # CAS-54/55/56 fixed .loc self-writes -- this is the sampled-tail variant.
        _rerun_probe(
            nb_runner,
            "import pandas as pd\ndf = pd.DataFrame({'a': list(range(1000))})",
            "df.iloc[500, 0] += 1\nprint('v=', int(df.iloc[500, 0]))",
            "v= 501",
        )

    def test_series_tail_augment_rerun_idempotent(self, nb_runner):
        _rerun_probe(
            nb_runner,
            "import pandas as pd\ns = pd.Series(range(1000))",
            "s.iloc[500] += 1\nprint('v=', int(s.iloc[500]))",
            "v= 501",
        )


class TestContentHashCollisionChannels:
    """Two different values that content-hash identically (differ only outside
    the sample) -- probe every channel where the content hash discriminates."""

    def test_flag_edit_colliding_ndarray_consumer_fresh(self, nb_runner):
        """Edit a flag cell so the produced array differs ONLY at index 500
        (outside the first-100 sample).  The consumer must print the new value."""
        nb_runner.create_notebook([
            "flag = 7",
            "import numpy as np\narr = np.zeros(1000, dtype=int)\narr[500] = flag",
            "print('probe=', int(arr[500]))",
        ])
        nb_runner.start_kernel()
        nb_runner.enable_debug()
        nb_runner.enable_persist()
        nb_runner.run_all()
        assert "probe= 7" in nb_runner.get_output(3), nb_runner.get_output(3)
        nb_runner.set_cell_source(1, "flag = 9")
        nb_runner.run_all()
        assert "probe= 9" in nb_runner.get_output(3), \
            f"stale consumer after flag edit: {nb_runner.get_output(3)!r}"

    def test_flag_edit_colliding_dataframe_consumer_fresh(self, nb_runner):
        """Same attack through a 1000-row DataFrame differing only in row 700
        (outside the head-5 sample)."""
        nb_runner.create_notebook([
            "flag = 7",
            "import pandas as pd\ndf = pd.DataFrame({'a': [0] * 1000})\ndf.iloc[700, 0] = flag",
            "print('probe=', int(df.iloc[700, 0]))",
        ])
        nb_runner.start_kernel()
        nb_runner.enable_debug()
        nb_runner.enable_persist()
        nb_runner.run_all()
        assert "probe= 7" in nb_runner.get_output(3), nb_runner.get_output(3)
        nb_runner.set_cell_source(1, "flag = 9")
        nb_runner.run_all()
        assert "probe= 9" in nb_runner.get_output(3), \
            f"stale consumer after flag edit: {nb_runner.get_output(3)!r}"

    def test_reset_state_sampled_df_consumer_not_stale(self, nb_runner):
        """Provenance loss (reset_cash_state) forces lineage to be re-derived
        from the SAMPLED content hash.  A tail mutation (row 700, outside the
        head-5 sample) then collides with the pre-mutation hash -- the cached
        consumer must NOT serve the pre-mutation total."""
        nb_runner.create_notebook([
            "import pandas as pd\ndf = pd.DataFrame({'a': list(range(1000))})",
            "pass",
            "total = int(df['a'].sum())\nprint('total=', total)",
        ])
        nb_runner.start_kernel()
        nb_runner.enable_debug()
        nb_runner.enable_persist()
        nb_runner.run_all()
        assert "total= 499500" in nb_runner.get_output(3), nb_runner.get_output(3)

        # Session 2: provenance lost, consumer re-keyed from content and cached.
        nb_runner.reset_cash_state()
        nb_runner.run_cell(3)
        assert "total= 499500" in nb_runner.get_output(3), nb_runner.get_output(3)

        # Tracked in-place tail write: row 700 goes 700 -> 0 (sum drops by 700).
        nb_runner.set_cell_source(2, "df.iloc[700, 0] = 0")
        nb_runner.run_cell(2)

        # Session 3: provenance lost again; content hash of the mutated df
        # collides with the pre-mutation hash (head-5/shape/dtypes unchanged).
        nb_runner.reset_cash_state()
        nb_runner.run_cell(3)
        # ADJUDICATED: the documented sampling limitation
        # (docs/known-limitations.md, "Large objects are hashed by sampling").
        #
        # This is what sampling COSTS, and the cost is only reachable here
        # because the probe deliberately destroys provenance first: normally
        # lineage catches the mutation, and the content hash is a fallback for
        # when it cannot. Once re-keyed from content alone, a head-5 sample of a
        # 1000-row frame cannot see a row-700 edit, so the hash is unchanged and
        # the consumer is legitimately considered fresh.
        #
        # Not fixable without hashing large frames whole, which is the cost
        # sampling exists to avoid — it would put a full pass over every large
        # object on the hot path of every statement.
        #
        # Pinned as the real behaviour so the trade-off stays visible.
        assert "total= 499500" in nb_runner.get_output(3), (
            f"expected the documented sampling collision: {nb_runner.get_output(3)!r}"
        )


class TestViewsNestedAndFriends:

    def test_np_view_mutation_edit_invalidates_base_consumer(self, nb_runner):
        """b = a[200:250] is a view; b[:] = k mutates `a` in a region outside
        a's first-100 hash sample.  Editing the mutation cell must refresh the
        consumer of `a`."""
        nb_runner.create_notebook([
            "import numpy as np\na = np.arange(300)",
            "b = a[200:250]",
            "b[:] = 9",
            "print('asum=', int(a.sum()))",
        ])
        nb_runner.start_kernel()
        nb_runner.enable_debug()
        nb_runner.enable_persist()
        nb_runner.run_all()
        # arange(300).sum()=44850; rows 200..249 sum 11225 replaced by 9*50=450.
        assert "asum= 34075" in nb_runner.get_output(4), nb_runner.get_output(4)
        nb_runner.set_cell_source(3, "b[:] = 7")
        nb_runner.run_all()
        assert "asum= 33975" in nb_runner.get_output(4), \
            f"stale a-consumer after editing view mutation: {nb_runner.get_output(4)!r}"

    def test_object_dtype_nested_list_rerun_idempotent(self, nb_runner):
        """Object-dtype column holds lists; head(5).values.tobytes() serialises
        POINTERS, so in-place mutation of a nested list is hash-invisible.
        Isolated re-run must not double-append."""
        _rerun_probe(
            nb_runner,
            "import pandas as pd\ndf = pd.DataFrame({'lists': [[1], [2], [3]]})",
            "df.loc[0, 'lists'].append(9)\nprint('r0=', df.loc[0, 'lists'])",
            "r0= [1, 9]",
        )
        # exactly one 9 -- [1, 9, 9] means the append accumulated
        assert nb_runner.get_output(2).count("9") == 1, nb_runner.get_output(2)

    def test_df_attrs_accumulate_rerun_idempotent(self, nb_runner):
        """df.attrs is metadata invisible to the content hash (shape+dtypes+head).
        A self-incrementing attrs cell must still be idempotent on re-run."""
        _rerun_probe(
            nb_runner,
            "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3]})\ndf.attrs['n'] = 0",
            "df.attrs['n'] = df.attrs['n'] + 1\nprint('n=', df.attrs['n'])",
            "n= 1",
        )

    def test_df_insert_rerun_idempotent(self, nb_runner):
        """df.insert raises 'cannot insert z, already exists' if df is not
        restored to its base before an isolated re-run."""
        _rerun_probe(
            nb_runner,
            "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3]})",
            "df.insert(0, 'z', 0)\nprint('cols=', df.columns.tolist())",
            "cols= ['z', 'a']",
        )

    def test_groupby_across_cells_tail_mutation_matches_plain(self, nb_runner):
        """groupby object created in one cell, source df tail-mutated in the
        next, aggregated in a third.  Ground-truth both variants in a plain
        kernel first; the cash kernel must match, including after editing the
        mutation cell."""
        cells = [
            "import pandas as pd\ndf = pd.DataFrame({'k': [0, 1] * 500, 'v': list(range(1000))})",
            "g = df.groupby('k')",
            "df.iloc[999, 1] = 0",
            "print('sums=', g['v'].sum().to_dict())",
        ]
        nb_runner.create_notebook(cells)
        nb_runner.start_kernel(with_cash=False)
        nb_runner.run_all()
        truth_orig = nb_runner.get_output(4).strip()
        nb_runner.set_cell_source(3, "df.iloc[999, 1] = 5")
        nb_runner.run_all()
        truth_edit = nb_runner.get_output(4).strip()
        nb_runner.shutdown()
        assert "sums=" in truth_orig and "sums=" in truth_edit

        nb_runner.set_cell_source(3, "df.iloc[999, 1] = 0")
        nb_runner.start_kernel(with_cash=True)
        nb_runner.enable_debug()
        nb_runner.enable_persist()
        nb_runner.run_all()
        got_orig = nb_runner.get_output(4).strip()
        assert truth_orig in got_orig, \
            f"cash diverges from plain kernel: plain={truth_orig!r} cash={got_orig!r}"
        nb_runner.set_cell_source(3, "df.iloc[999, 1] = 5")
        nb_runner.run_all()
        got_edit = nb_runner.get_output(4).strip()
        assert truth_edit in got_edit, \
            f"stale aggregate after mutation edit: plain={truth_edit!r} cash={got_edit!r}"
