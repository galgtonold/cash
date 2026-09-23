"""Loops that accumulate into a list or dict, re-run and edited.

Test for accumulator pattern when adding a new item to loop list.

This tests the scenario where:
1. A loop builds up an accumulator dict over several iterations
2. All iterations are cached
3. A new item is added to the loop list
4. Running a downstream cell should compute the new iteration AND keep the old values

Append to a single-statement accumulator loop; count real executions
against a cash-off oracle.

Regression guard for the removed ``cacheable_accumulator_loop`` fast path
(``control_structures/processor.py``, since deleted). That mechanism
matched the NARROW shape ``out = []`` immediately followed, in the SAME
cell with no intervening statement, by ``for t in xs: out.append(compute(t))``
-- and routed the WHOLE loop through the statement cache as ONE unit. The
unit's cache key includes the iterable's lineage, so appending a single item
to ``xs`` invalidated the whole unit and re-ran EVERY ``compute()`` call --
exactly as expensive as no caching at all (measured).

No existing test caught this. Every prior test of this shape either:

- used a two-statement body (``v = compute(t)`` / ``out.append(v)``), which
  never matched the shape gate at all (``test_call_unit_acceptance.py``
  deliberately avoids the one-statement body for this exact reason), or
- put the ``out = []`` seed in a DIFFERENT cell from the loop
  (``test_cache_calls_directive.py::test_append_loop_caches_the_call_with_no_directive``),
  which also never matched -- the gate only fired when the seed was the
  loop's IMMEDIATELY PRECEDING sibling in the SAME cell, or
- only asserted an UNCHANGED rerun costs zero calls, which both the old
  whole-unit path and the new per-iteration path satisfy identically, so it
  cannot distinguish between them.

This test deliberately uses the exact shape the removed gate matched --
same cell, no intervening statement, single-statement ``Expr(Call)`` body --
and appends one item. Real executions are counted from an external file (not
the kernel's own printed bookkeeping), so cash's own instrumentation cannot
make the assertion vacuous, and compared against a cash-off oracle run with
the IDENTICAL cell shape so a smaller count is provably cash's incremental
reuse, not an artifact of the shape itself.
"""

import pytest


@pytest.mark.mutations
@pytest.mark.loops
def test_accumulator_add_item_fresh_session(nb_runner):
    """
    Test that adding a new item to a loop list correctly shows ALL items,
    not just the new one.

    Scenario:
    1. Fresh session (no kernel state)
    2. Loop with 4 items is cached
    3. Add 5th item to the list
    4. Run cell that uses the accumulator
    5. Should see all 5 items
    """
    # First run: cache 4 iterations
    nb_runner.create_notebook(
        [
            # Cell 1: Simple accumulator loop with 4 items
            """results = {}
for x in ["A", "B", "C", "D"]:
    results[x] = x * 2
print(f"Run 1: {list(results.keys())}")""",
            # Cell 2: Use the accumulator
            """print(f"Keys: {list(results.keys())}")""",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()

    output1 = nb_runner.get_output(1)
    assert "['A', 'B', 'C', 'D']" in output1 or "dict_keys(['A', 'B', 'C', 'D'])" in output1

    output2 = nb_runner.get_output(2)
    assert "'A'" in output2 and "'B'" in output2 and "'C'" in output2 and "'D'" in output2

    # Modify loop to have 5 items
    nb_runner.set_cell_source(
        1,
        """results = {}
for x in ["A", "B", "C", "D", "E"]:
    results[x] = x * 2
print(f"Run 2: {list(results.keys())}")""",
    )

    nb_runner.run_all()

    output3 = nb_runner.get_output(1)
    print(f"After modification, Cell 1 output: {output3}")

    output4 = nb_runner.get_output(2)
    print(f"After modification, Cell 2 output: {output4}")

    # Critical assertion: ALL 5 items should be present
    assert "'A'" in output4, f"Missing 'A' in output: {output4}"
    assert "'B'" in output4, f"Missing 'B' in output: {output4}"
    assert "'C'" in output4, f"Missing 'C' in output: {output4}"
    assert "'D'" in output4, f"Missing 'D' in output: {output4}"
    assert "'E'" in output4, f"Missing 'E' in output: {output4}"


@pytest.mark.mutations
@pytest.mark.loops
def test_accumulator_add_item_skip_loop_cell(nb_runner):
    """
    Test the exact user scenario:
    1. Run loop cell (caches 4 iterations)
    2. Run keys cell
    3. Modify loop to add 5th item
    4. Run ONLY the keys cell (not the loop cell)
    5. The upstream checker should handle restoration + new computation
    """
    nb_runner.create_notebook(
        [
            # Cell 1: Loop with 4 items
            """results = {}
for x in ["A", "B", "C", "D"]:
    results[x] = x * 2
print(f"Loop done: {list(results.keys())}")""",
            # Cell 2: Print keys
            """print(f"Keys: {list(results.keys())}")""",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()

    output1 = nb_runner.get_output(2)
    print(f"Run 1 output: {output1}")
    assert "'D'" in output1

    # Modify loop to add 5th item
    nb_runner.set_cell_source(
        1,
        """results = {}
for x in ["A", "B", "C", "D", "E"]:
    results[x] = x * 2
print(f"Loop done: {list(results.keys())}")""",
    )

    # Run ONLY cell 2 - upstream checker should detect the change
    nb_runner.run_cell(2)

    output2 = nb_runner.get_output(2)
    print(f"After modification, Cell 2 output: {output2}")

    # All 5 should be present
    for item in ["A", "B", "C", "D", "E"]:
        assert f"'{item}'" in output2, f"Missing '{item}' in: {output2}"


@pytest.mark.mutations
@pytest.mark.loops
def test_accumulator_only_run_downstream_cell(nb_runner):
    """
    Test running only the downstream cell after adding a new item.
    The upstream checker should detect the change and re-execute.
    """
    nb_runner.create_notebook(
        [
            # Cell 1: Loop with 3 items
            """data = {}
for x in [1, 2, 3]:
    data[x] = x * 10
print(f"After loop: {data}")""",
            # Cell 2: Use data
            """print(f"Downstream: {data}")""",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()

    out1 = nb_runner.get_output(2)
    assert "1: 10" in out1 and "2: 20" in out1 and "3: 30" in out1

    # Add 4th item
    nb_runner.set_cell_source(
        1,
        """data = {}
for x in [1, 2, 3, 4]:
    data[x] = x * 10
print(f"After loop: {data}")""",
    )

    # Run only downstream cell
    nb_runner.run_cell(2)

    out2 = nb_runner.get_output(2)
    print(f"After modification: {out2}")

    for x in [1, 2, 3, 4]:
        assert f"{x}: {x * 10}" in out2, f"Missing {x} in: {out2}"


@pytest.mark.mutations
@pytest.mark.loops
def test_accumulator_fresh_session_only_downstream(nb_runner):
    """
    Test with fresh session, run only downstream cell.
    """
    nb_runner.create_notebook(
        [
            # Cell 1: Define data
            """data = {'a': 1, 'b': 2}""",
            # Cell 2: Use data
            """print(f"Keys: {list(data.keys())}")""",
        ]
    )
    nb_runner.start_kernel()

    # Run only cell 2 - should trigger upstream execution of cell 1
    nb_runner.run_cell(2)

    out = nb_runner.get_output(2)
    assert "'a'" in out and "'b'" in out


@pytest.mark.mutations
@pytest.mark.loops
def test_accumulator_upstream_triggered(nb_runner):
    """
    Test that running downstream cell triggers upstream execution.
    """
    nb_runner.create_notebook(
        [
            # Cell 1: Define x
            """x = 10
print(f"x = {x}")""",
            # Cell 2: Use x
            """y = x * 2
print(f"y = {y}")""",
            # Cell 3: Use y
            """z = y + 5
print(f"z = {z}")""",
        ]
    )
    nb_runner.start_kernel()

    # Run only cell 3 - should trigger cells 1 and 2
    nb_runner.run_cell(3)

    out = nb_runner.get_output(3)
    assert "z = 25" in out  # 10 * 2 + 5 = 25


# Tests for accumulator initialization skip logic with mutation-only updates.
#
# Bug: When a loop mutates a variable via .append() (pure mutation, no assignment output),
# the accumulator-init-skip logic in upstream.py doesn't recognize the variable as
# loop-updated. This causes `a = []` to be re-executed when the upstream cell is modified,
# resetting the accumulated value.
#
# The fix extends the accumulator-init-skip to also check vars_mutated_by_loops,
# not just scheduled_iteration_outputs.
#
# Root cause: CodeAnalyzer.analyze_code_block('a.append(x)') returns outputs=set(),
# so 'a' never appears in scheduled_iteration_outputs. But MutationDetector correctly
# detects 'a' as mutated, and _find_loop_mutated_vars adds it to vars_mutated_by_loops.
#
# NOTE: Mutation-only accumulator inits are now preserved across upstream
# modifications that change loop code (the fix extends the skip logic to
# vars_mutated_by_loops, while re-scheduling the init alongside any FULLY
# re-run loop so in-place accumulation does not double).
@pytest.mark.mutations
@pytest.mark.upstream
class TestMutationAccumulatorInit:
    """Test that accumulator init is skipped for mutation-only loop updates."""

    def test_append_accumulator_preserved_after_upstream_change(self, nb_runner):
        """
        Reproduces the core bug: a = [] followed by for-loop with a.append(x)
        should preserve 'a' when downstream cell re-runs after upstream modification.
        """
        nb_runner.create_notebook(
            [
                # Cell 1: Setup data
                "data = {'x': [1, 2, 3], 'y': [4, 5, 6]}",
                # Cell 2: Loop with both assignment and append mutation
                (
                    "results = {}\n"
                    "collected = []\n"
                    "for key in ['x', 'y']:\n"
                    "    vals = data[key]\n"
                    "    results[key] = sum(vals)\n"
                    "    collected.append(key)\n"
                    "print(f'results={results}')\n"
                    "print(f'collected={collected}')"
                ),
                # Cell 3: Use both variables
                "print(f'results_keys={sorted(results.keys())}')\nprint(f'collected_list={collected}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        # Verify initial run
        output3 = nb_runner.get_output(3)
        assert "results_keys=['x', 'y']" in output3
        assert "collected_list=['x', 'y']" in output3

        # Modify cell 2: change iteration order
        nb_runner.set_cell_source(
            2,
            (
                "results = {}\n"
                "collected = []\n"
                "for key in ['y', 'x']:\n"  # Changed order
                "    vals = data[key]\n"
                "    results[key] = sum(vals)\n"
                "    collected.append(key)\n"
                "print(f'results={results}')\n"
                "print(f'collected={collected}')"
            ),
        )

        # Re-run cell 3 (without re-running cell 2)
        nb_runner.run_cell(3)
        output3 = nb_runner.get_output(3)

        # Both variables should be preserved from the original run
        # (not reset to empty by re-executing the initialization)
        # 'collected' should NOT be [] (the bug would cause this)
        assert "collected_list=[]" not in output3, f"Bug: collected was reset to empty list. Output: {output3}"
        # 'collected' should still have items (either original or re-computed)
        assert "collected_list=[" in output3

    def test_append_only_accumulator_no_subscript_assignment(self, nb_runner):
        """
        Test where the ONLY loop mutation is .append() — no subscript assignment.
        This is the purest test of the fix since there's no scheduled_iteration_outputs
        at all (unlike ticker_stats[k] = v which does produce outputs).
        """
        nb_runner.create_notebook(
            [
                # Cell 1: Simple data
                "items = [10, 20, 30]",
                # Cell 2: Loop with ONLY append mutation
                (
                    "accumulated = []\n"
                    "for item in items:\n"
                    "    accumulated.append(item * 2)\n"
                    "print(f'accumulated={accumulated}')"
                ),
                # Cell 3: Use accumulated
                "print(f'total={sum(accumulated)}')\nprint(f'count={len(accumulated)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output3 = nb_runner.get_output(3)
        assert "total=120" in output3  # (10+20+30)*2 = 120
        assert "count=3" in output3

        # Modify cell 2: change items
        nb_runner.set_cell_source(
            2,
            (
                "accumulated = []\n"
                "for item in items:\n"
                "    accumulated.append(item * 3)\n"  # Changed multiplier
                "print(f'accumulated={accumulated}')"
            ),
        )

        # Re-run cell 3
        nb_runner.run_cell(3)
        output3 = nb_runner.get_output(3)

        # accumulated should NOT be [] (the bug)
        assert "count=0" not in output3, f"Bug: accumulated was reset to empty list. Output: {output3}"
        # Should still have 3 items
        assert "count=3" in output3

    def test_mixed_append_and_subscript_assignment(self, nb_runner):
        """
        Test with both .append() and subscript assignment in same loop.
        Ensures the fix works alongside the existing logic.
        """
        nb_runner.create_notebook(
            [
                # Cell 1: Data
                "tickers = ['AAPL', 'MSFT', 'TSLA']",
                # Cell 2: Loop with both patterns
                (
                    "stats = {}\n"
                    "names = []\n"
                    "for t in tickers:\n"
                    "    stats[t] = len(t)\n"
                    "    names.append(t)\n"
                    "print(f'stats={stats}')\n"
                    "print(f'names={names}')"
                ),
                # Cell 3: Use both
                "print(f'stats_keys={sorted(stats.keys())}')\nprint(f'names_list={names}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output3 = nb_runner.get_output(3)
        assert "stats_keys=['AAPL', 'MSFT', 'TSLA']" in output3
        assert "names_list=['AAPL', 'MSFT', 'TSLA']" in output3

        # Modify cell 2: change ticker order
        nb_runner.set_cell_source(
            2,
            (
                "stats = {}\n"
                "names = []\n"
                "for t in ['TSLA', 'AAPL', 'MSFT']:\n"  # Changed order
                "    stats[t] = len(t)\n"
                "    names.append(t)\n"
                "print(f'stats={stats}')\n"
                "print(f'names={names}')"
            ),
        )

        # Re-run cell 3
        nb_runner.run_cell(3)
        output3 = nb_runner.get_output(3)

        # Neither should be empty
        assert "stats_keys=[]" not in output3
        assert "names_list=[]" not in output3
        # Both should have 3 items
        assert "AAPL" in output3
        assert "MSFT" in output3
        assert "TSLA" in output3

    def test_set_add_mutation(self, nb_runner):
        """Test that set.add() mutations are also handled."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3, 2, 1]",
                ("unique = set()\nfor x in data:\n    unique.add(x)\nprint(f'unique={sorted(unique)}')"),
                "print(f'count={len(unique)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output3 = nb_runner.get_output(3)
        assert "count=3" in output3

        # Modify cell 2
        nb_runner.set_cell_source(
            2,
            (
                "unique = set()\n"
                "for x in data:\n"
                "    unique.add(x * 10)\n"  # Changed operation
                "print(f'unique={sorted(unique)}')"
            ),
        )

        nb_runner.run_cell(3)
        output3 = nb_runner.get_output(3)
        # Should NOT be count=0
        assert "count=0" not in output3, f"Bug: unique was reset to empty set. Output: {output3}"
        assert "count=3" in output3

    def test_dict_update_mutation(self, nb_runner):
        """Test that dict.update() mutations are handled."""
        nb_runner.create_notebook(
            [
                "pairs = [('a', 1), ('b', 2)]",
                ("merged = {}\nfor k, v in pairs:\n    merged.update({k: v})\nprint(f'merged={merged}')"),
                "print(f'keys={sorted(merged.keys())}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output3 = nb_runner.get_output(3)
        assert "keys=['a', 'b']" in output3

        # Modify cell 2
        nb_runner.set_cell_source(
            2,
            (
                "merged = {}\n"
                "for k, v in pairs:\n"
                "    merged.update({k: v * 10})\n"  # Changed value
                "print(f'merged={merged}')"
            ),
        )

        nb_runner.run_cell(3)
        output3 = nb_runner.get_output(3)
        assert "keys=[]" not in output3
        assert "keys=['a', 'b']" in output3


SETUP_ON = "import cash\n%cash_on\nimport time"


SETUP_OFF = "import cash\nimport time"


def _n(path):
    return len(path.read_bytes()) if path.exists() else 0


def _compute_def(counter):
    return f"def compute(v):\n    open(r'{counter}', 'a').write('X')\n    time.sleep(0.03)\n    return v * 10"


def _fast_path_cell(items):
    # `out = []` is the IMMEDIATELY preceding sibling in the SAME cell, and
    # the body is exactly one bare accumulator-method call -- the narrow
    # shape the removed `cacheable_accumulator_loop` gate matched. Do NOT
    # hoist `out = []` into an earlier cell (that would dodge the shape
    # entirely, as the pre-existing tests above did) and do NOT put anything
    # between `out = []` and the `for` (that would also dodge the gate).
    return f"out = []\nfor t in {items}:\n    out.append(compute(t))\nprint('OUT', out)"


@pytest.mark.integration
@pytest.mark.loops
def test_single_statement_accumulator_append_is_incremental(nb_runner, tmp_path):
    """Cash on: appending one item to the narrow accumulator-loop shape must
    cost exactly ONE real call, matching per-iteration reuse -- not
    ``len(items)`` calls, which is what the whole-loop unit produced."""
    counter = tmp_path / "calls_on.log"
    nb_runner.create_notebook([SETUP_ON, _compute_def(counter), _fast_path_cell([1, 2])])
    nb_runner.start_kernel()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 2, f"baseline did not run both iterations: {cold} calls"
    assert "OUT [10, 20]" in nb_runner.get_output(3)

    nb_runner.set_cell_source(3, _fast_path_cell([1, 2, 3]))
    nb_runner.run_cell(3)
    after_append = _n(counter) - cold
    assert after_append == 1, (
        f"appending one item re-ran {after_append} calls, expected 1 (only "
        "the new item) -- the whole loop was re-computed as a single unit"
    )
    assert "OUT [10, 20, 30]" in nb_runner.get_output(3)


@pytest.mark.integration
@pytest.mark.loops
def test_single_statement_accumulator_append_oracle_reruns_everything(nb_runner, tmp_path):
    """Cash off, IDENTICAL shape: proves the "1 call" result above is cash's
    incremental reuse and not some property of the shape that would hold
    even with no caching at all."""
    counter = tmp_path / "calls_off.log"
    nb_runner.create_notebook([SETUP_OFF, _compute_def(counter), _fast_path_cell([1, 2])])
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 2, f"oracle baseline did not run both iterations: {cold} calls"

    nb_runner.set_cell_source(3, _fast_path_cell([1, 2, 3]))
    nb_runner.run_cell(3)
    after_append = _n(counter) - cold
    assert after_append == 3, (
        f"oracle only re-ran {after_append} calls after appending one item; "
        "expected all 3 (no caching at all) -- oracle setup is broken"
    )
