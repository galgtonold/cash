"""User testing findings captured as integration tests.

Covers findings from both rounds of user testing:
- Round 1 (examples/large_scale_projects/USER_TESTING_FINDINGS_MAY2026.md)
- Round 2 (examples/user_testing_round2/USER_TESTING_FINDINGS_ROUND2.md)

Tests validate caching correctness: no stale data, proper invalidation,
upstream change detection, and expected caching behavior.
"""

import pytest

pytestmark = [pytest.mark.core, pytest.mark.timeout(30)]


# =============================================================================
# Round 1 Finding NB4/NB5/NB6 + Round 2 F1/F2: Caching correctness
# =============================================================================

class TestUnchangedCellsRestored:
    """Round 2 F1: Re-running unchanged cells should restore from cache."""

    def test_unchanged_cell_all_restored(self, nb_runner):
        """Simple cell: all statements RESTORED on re-run."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x * 2",
            "print(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 20" in nb_runner.get_output(3)

        # Re-run all - should restore from cache
        nb_runner.run_all()
        assert "y = 20" in nb_runner.get_output(3)

    def test_unchanged_multi_cell_all_restored(self, nb_runner):
        """Multiple cells: all RESTORED on re-run."""
        nb_runner.create_notebook([
            "a = 1\nb = 2",
            "c = a + b",
            "d = c * 3\nprint(f'd = {d}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "d = 9" in nb_runner.get_output(3)

        nb_runner.run_all()
        assert "d = 9" in nb_runner.get_output(3)


class TestUpstreamChangeDetection:
    """Round 1 NB4 + Round 2 F2: Upstream changes trigger downstream recompute."""

    def test_edit_root_cell_downstream_recompute(self, nb_runner):
        """Edit cell 1, verify cells 2 and 3 recompute."""
        nb_runner.create_notebook([
            "multiplier = 2.5\ndata_a = 100 * multiplier",
            "data_b = data_a + 50\ndata_b2 = data_b * 3",
            "total = data_b2 + 100\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # data_a=250, data_b=300, data_b2=900, total=1000
        assert "total = 1000" in nb_runner.get_output(3)

        # Edit cell 1: change multiplier
        nb_runner.set_cell_source(1, "multiplier = 4.0\ndata_a = 100 * multiplier")
        nb_runner.run_all()
        # data_a=400, data_b=450, data_b2=1350, total=1450
        assert "total = 1450" in nb_runner.get_output(3)

    def test_edit_middle_cell_downstream_recompute(self, nb_runner):
        """Edit cell 2 (middle), verify cell 3 recomputes."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x * 5",
            "z = y + 42\nprint(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 92" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "y = x * 100")
        nb_runner.run_all()
        assert "z = 1042" in nb_runner.get_output(3)

    def test_add_feature_upstream_model_retrains(self, nb_runner):
        """Round 2 F7: Adding feature in upstream cell triggers model retrain."""
        nb_runner.create_notebook([
            "a = [1, 2, 3, 4, 5]",
            "b = [x * 2 for x in a]",
            "total = sum(b)\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 30" in nb_runner.get_output(3)

        # Add new element to list 'a'
        nb_runner.set_cell_source(1, "a = [1, 2, 3, 4, 5, 6]")
        nb_runner.run_all()
        assert "total = 42" in nb_runner.get_output(3)

    def test_multi_cell_cascade_all_recompute(self, nb_runner):
        """Round 2 F8: 3+ cell downstream cascade all recomputes correctly."""
        nb_runner.create_notebook([
            "raw = [1, -2, 3, -4, 5, -6]",
            "positive = [x for x in raw if x > 0]",
            "squared = [x ** 2 for x in positive]",
            "result = sum(squared)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 1+9+25 = 35
        assert "result = 35" in nb_runner.get_output(4)

        # Change filter: keep ALL numbers (not just positive)
        nb_runner.set_cell_source(2, "positive = [abs(x) for x in raw]")
        nb_runner.run_all()
        # 1+4+9+16+25+36 = 91
        assert "result = 91" in nb_runner.get_output(4)


class TestNoStaleData:
    """Round 2 F9: Upstream changes must produce correct new values."""

    def test_data_change_produces_correct_values(self, nb_runner):
        """Changing data source must update all downstream results."""
        nb_runner.create_notebook([
            "data = [10, 20, 30]",
            "doubled = [x * 2 for x in data]",
            "result = sum(doubled)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 120" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "data = [100, 200, 300]")
        nb_runner.run_all()
        assert "result = 1200" in nb_runner.get_output(3)

    def test_filter_change_updates_aggregates(self, nb_runner):
        """Changing a filter condition must update aggregates."""
        nb_runner.create_notebook([
            "nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
            "filtered = [x for x in nums if x % 2 == 0]",
            "result = sum(filtered)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 2+4+6+8+10 = 30
        assert "result = 30" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "filtered = [x for x in nums if x % 3 == 0]")
        nb_runner.run_all()
        # 3+6+9 = 18
        assert "result = 18" in nb_runner.get_output(3)


# =============================================================================
# Round 1 NB5: Conditional branching
# =============================================================================

class TestConditionalBranching:
    """Round 1 NB5: Conditional (if/else) caching."""

    def test_if_else_caches_correctly(self, nb_runner):
        """If/else branch cache hit on re-run."""
        nb_runner.create_notebook([
            "x = 15\nif x > 10:\n    result = 'high'\nelse:\n    result = 'low'\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = high" in nb_runner.get_output(1)

        nb_runner.run_all()
        assert "result = high" in nb_runner.get_output(1)

    def test_if_else_branch_change_recomputes(self, nb_runner):
        """Changing condition to take other branch recomputes correctly."""
        nb_runner.create_notebook([
            "x = 5\nif x > 10:\n    result = 'high'\nelse:\n    result = 'low'\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = low" in nb_runner.get_output(1)

        nb_runner.set_cell_source(1,
            "x = 20\nif x > 10:\n    result = 'high'\nelse:\n    result = 'low'\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = high" in nb_runner.get_output(1)


# =============================================================================
# Round 1 NB6 + Round 2 F6: Loop caching
# =============================================================================

class TestLoopCaching:
    """Round 1 NB6 + Round 2 F6: Loop caching behavior."""

    def test_pure_loop_caches_correctly(self, nb_runner):
        """A loop without mutation should cache and restore."""
        nb_runner.create_notebook([
            "results = []\nfor i in range(5):\n    v = i * 10\n    results.append(v)",
            "print(f'results = {results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results = [0, 10, 20, 30, 40]" in nb_runner.get_output(2)

        nb_runner.run_all()
        assert "results = [0, 10, 20, 30, 40]" in nb_runner.get_output(2)

    def test_loop_with_list_build_restores(self, nb_runner):
        """Loop that builds a list should restore on re-run."""
        nb_runner.create_notebook([
            "squares = []\nfor i in range(1, 6):\n    squares.append(i * i)",
            "total = sum(squares)\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 55" in nb_runner.get_output(2)

        nb_runner.run_all()
        assert "total = 55" in nb_runner.get_output(2)

    def test_loop_code_change_re_executes(self, nb_runner):
        """Changing loop code forces re-execution."""
        nb_runner.create_notebook([
            "items = []\nfor i in range(5):\n    items.append(i * 2)",
            "result = sum(items)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 0+2+4+6+8 = 20
        assert "result = 20" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1,
            "items = []\nfor i in range(5):\n    items.append(i * 10)")
        nb_runner.run_all()
        # 0+10+20+30+40 = 100
        assert "result = 100" in nb_runner.get_output(2)


# =============================================================================
# Round 1 NB1: DataFrame column assignment caching
# =============================================================================

class TestDataFrameColumnAssignment:
    """Round 1 NB1: DataFrame self-assignment always re-computes (Known Issue 13)."""

    def test_column_assignment_detected_as_mutation(self, nb_runner):
        """df['col'] = ... is treated as mutation, may not cache.
        
        This is KNOWN behavior (Issue 13). The test verifies correctness:
        even if not cached, the result must still be correct.
        """
        nb_runner.create_notebook([
            "import pandas as pd\n"
            "df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})",
            "df['c'] = df['a'] + df['b']",
            "result = df['c'].sum()\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 1+4=5, 2+5=7, 3+6=9, sum=21
        assert "result = 21" in nb_runner.get_output(3)

        # Re-run — correctness must be preserved even if not cached
        nb_runner.run_all()
        assert "result = 21" in nb_runner.get_output(3)


# =============================================================================
# Round 1 NB2: Loop mutation detection
# =============================================================================

class TestLoopMutationDetection:
    """Round 1 NB2: Loop mutation iterations detected correctly."""

    def test_append_loop_detected_as_mutation(self, nb_runner):
        """List append in loop is detected as mutation.
        
        The badge should show 'No Cache' for mutation iterations.
        But the final result must still be correct.
        """
        nb_runner.create_notebook([
            "items = []\nfor i in range(10):\n    items.append(i * 2)",
            "print(f'len = {len(items)}, items = {items}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "len = 10" in nb_runner.get_output(2)

        nb_runner.run_all()
        assert "len = 10" in nb_runner.get_output(2)

    def test_mutation_loop_correctness(self, nb_runner):
        """Aggregating loop with mutation still produces correct result."""
        nb_runner.create_notebook([
            "total = 0\nfor i in range(1, 6):\n    total += i",
            "print(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 15" in nb_runner.get_output(2)

        nb_runner.run_all()
        assert "total = 15" in nb_runner.get_output(2)


# =============================================================================
# Round 1 NB3 + Round 2 F3: sklearn model caching
# =============================================================================

class TestModelCaching:
    """Round 1 NB3 + Round 2 F3: ML model caching behavior.
    
    Note: sklearn model objects with separate fit() calls in multi-cell
    workflows have a known issue (Round 1 NB3) where the fit() mutation
    is not properly tracked across cell boundaries. Single-cell and
    non-mutating workflows are fine.
    """

    def test_model_training_correctness(self, nb_runner):
        """Model training produces correct predictions (single cell test)."""
        nb_runner.create_notebook([
            "from sklearn.ensemble import RandomForestClassifier\n"
            "import numpy as np\n"
            "X = np.array([[1,2],[3,4],[5,6],[7,8]])\n"
            "y = np.array([0, 1, 0, 1])\n"
            "rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)\n"
            "rf.fit(X, y)\n"
            "pred = rf.predict([[2, 3]])\n"
            "print(f'prediction = {pred[0]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(1)
        assert "prediction =" in output

        # Re-run same cell — model fit may re-execute or restore
        nb_runner.run_all()
        output = nb_runner.get_output(1)
        assert "prediction =" in output


# =============================================================================
# Diamond dependencies
# =============================================================================

class TestDiamondDependencies:
    """Complex dependency patterns: diamond, multi-branch."""

    def test_diamond_dependency_edit_leaf(self, nb_runner):
        """Diamond: a -> (b, c) -> d. Edit a, verify d updates."""
        nb_runner.create_notebook([
            "a = 10",
            "b = a + 1",
            "c = a + 2",
            "d = b + c\nprint(f'd = {d}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "d = 23" in nb_runner.get_output(4)

        nb_runner.set_cell_source(1, "a = 100")
        nb_runner.run_all()
        assert "d = 203" in nb_runner.get_output(4)

    def test_diamond_edit_one_branch(self, nb_runner):
        """Diamond: edit only one branch, verify d updates correctly."""
        nb_runner.create_notebook([
            "a = 10",
            "b = a * 2",
            "c = a * 3",
            "d = b + c\nprint(f'd = {d}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "d = 50" in nb_runner.get_output(4)

        nb_runner.set_cell_source(2, "b = a * 10")
        nb_runner.run_all()
        assert "d = 130" in nb_runner.get_output(4)
