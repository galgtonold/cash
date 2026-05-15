"""
Comprehensive tests for loop caching correctness.

Tests cover:
- Dict, list, set mutations in loops
- Various iterator patterns
- While loops
- Upstream trust logic (loop-mutated vars with/without code changes)
- Side effects in loops
- Edge cases (empty loops, break/continue, nested loops)

These tests validate that:
1. Loop-mutated variables get correct lineage updates
2. Cached per-iteration results restore correctly on re-execution
3. Upstream checker correctly handles loop-mutated variables when upstream code changes
4. MutationDetector accurately identifies in-place mutations
"""

import json
import pytest
from unittest.mock import patch

from cash.notebook.cacheability import analyze_statement
from cash.notebook.analysis import CodeAnalyzer


# ============================================================================
# Group 1: Dict Mutation in Loops
# ============================================================================

class TestLoopDictMutation:
    """Test dict mutations inside for loops."""

    def test_loop_dict_subscript_assignment(self, cash_magics, mock_shell):
        """d[k] = v in loop should produce correct results and cache per-iteration."""
        magics = cash_magics
        shell = mock_shell

        magics.cash_on("")

        code = """results = {}
for x in ["A", "B", "C"]:
    results[x] = x.lower()
"""
        magics.cash("", code)
        assert shell.user_ns['results'] == {'A': 'a', 'B': 'b', 'C': 'c'}

    def test_loop_dict_subscript_second_run_from_cache(self, cash_magics, mock_shell):
        """Second execution of same loop code should restore from cache."""
        magics = cash_magics
        shell = mock_shell

        magics.cash_on("")

        code = """results = {}
for x in ["A", "B", "C"]:
    results[x] = x.lower()
"""
        # First run
        magics.cash("", code)
        assert shell.user_ns['results'] == {'A': 'a', 'B': 'b', 'C': 'c'}

        # Second run — should use cache
        magics.cash("", code)
        assert shell.user_ns['results'] == {'A': 'a', 'B': 'b', 'C': 'c'}

    def test_loop_dict_update_method(self, cash_magics, mock_shell):
        """d.update({k: v}) in loop should be detected as mutation."""
        magics = cash_magics
        shell = mock_shell

        magics.cash_on("")

        code = """results = {}
for x in ["A", "B", "C"]:
    results.update({x: x * 2})
"""
        magics.cash("", code)
        assert shell.user_ns['results'] == {'A': 'AA', 'B': 'BB', 'C': 'CC'}

    def test_loop_dict_setdefault(self, cash_magics, mock_shell):
        """d.setdefault(k, v) in loop."""
        magics = cash_magics
        shell = mock_shell

        magics.cash_on("")

        code = """results = {}
for x in ["A", "B", "C"]:
    results.setdefault(x, x * 3)
"""
        magics.cash("", code)
        assert shell.user_ns['results'] == {'A': 'AAA', 'B': 'BBB', 'C': 'CCC'}

    def test_loop_dict_multiple_mutations(self, cash_magics, mock_shell):
        """Multiple dicts mutated in same loop."""
        magics = cash_magics
        shell = mock_shell

        magics.cash_on("")

        code = """names = {}
counts = {}
for x in ["A", "B", "C"]:
    names[x] = x.lower()
    counts[x] = len(x)
"""
        magics.cash("", code)
        assert shell.user_ns['names'] == {'A': 'a', 'B': 'b', 'C': 'c'}
        assert shell.user_ns['counts'] == {'A': 1, 'B': 1, 'C': 1}


# ============================================================================
# Group 2: List Mutation in Loops
# ============================================================================

class TestLoopListMutation:
    """Test list mutations inside for loops."""

    def test_loop_list_append(self, cash_magics, mock_shell):
        """lst.append(x) in loop."""
        magics = cash_magics
        shell = mock_shell

        magics.cash_on("")

        code = """results = []
for x in [1, 2, 3]:
    results.append(x * 10)
"""
        magics.cash("", code)
        assert shell.user_ns['results'] == [10, 20, 30]

    def test_loop_list_extend(self, cash_magics, mock_shell):
        """lst.extend([x]) in loop."""
        magics = cash_magics
        shell = mock_shell

        magics.cash_on("")

        code = """results = []
for x in [1, 2, 3]:
    results.extend([x, x * 2])
"""
        magics.cash("", code)
        assert shell.user_ns['results'] == [1, 2, 2, 4, 3, 6]

    def test_loop_list_index_assignment(self, cash_magics, mock_shell):
        """lst[i] = x in loop."""
        magics = cash_magics
        shell = mock_shell

        magics.cash_on("")

        code = """results = [0, 0, 0]
for i in range(3):
    results[i] = i * 5
"""
        magics.cash("", code)
        assert shell.user_ns['results'] == [0, 5, 10]

    def test_loop_augmented_assign_int(self, cash_magics, mock_shell):
        """total += x in loop (int augmented assign)."""
        magics = cash_magics
        shell = mock_shell

        magics.cash_on("")

        code = """total = 0
for x in [10, 20, 30]:
    total += x
"""
        magics.cash("", code)
        assert shell.user_ns['total'] == 60


# ============================================================================
# Group 3: Set Mutation in Loops
# ============================================================================

class TestLoopSetMutation:
    """Test set mutations inside for loops."""

    def test_loop_set_add(self, cash_magics, mock_shell):
        """s.add(x) in loop."""
        magics = cash_magics
        shell = mock_shell

        magics.cash_on("")

        code = """seen = set()
for x in [1, 2, 3, 2, 1]:
    seen.add(x)
"""
        magics.cash("", code)
        assert shell.user_ns['seen'] == {1, 2, 3}

    def test_loop_set_discard(self, cash_magics, mock_shell):
        """s.discard(x) in loop."""
        magics = cash_magics
        shell = mock_shell

        magics.cash_on("")

        code = """items = {1, 2, 3, 4, 5}
for x in [2, 4]:
    items.discard(x)
"""
        magics.cash("", code)
        assert shell.user_ns['items'] == {1, 3, 5}


# ============================================================================
# Group 4: Iterator Variations
# ============================================================================

class TestLoopIteratorVariations:
    """Test various iterator patterns in for loops."""

    def test_loop_over_range(self, cash_magics, mock_shell):
        """for i in range(n) pattern."""
        magics = cash_magics
        shell = mock_shell

        magics.cash_on("")

        code = """results = []
for i in range(4):
    results.append(i ** 2)
"""
        magics.cash("", code)
        assert shell.user_ns['results'] == [0, 1, 4, 9]

    def test_loop_over_list_variable(self, cash_magics, mock_shell):
        """for x in items where items is a variable."""
        magics = cash_magics
        shell = mock_shell

        magics.cash_on("")

        shell.user_ns['items'] = ['cat', 'dog', 'bird']

        code = """results = []
for x in items:
    results.append(x.upper())
"""
        magics.cash("", code)
        assert shell.user_ns['results'] == ['CAT', 'DOG', 'BIRD']

    def test_loop_over_dict_items(self, cash_magics, mock_shell):
        """for k, v in d.items() pattern."""
        magics = cash_magics
        shell = mock_shell

        magics.cash_on("")

        shell.user_ns['data'] = {'a': 1, 'b': 2}

        code = """results = {}
for k, v in data.items():
    results[k] = v * 10
"""
        magics.cash("", code)
        assert shell.user_ns['results'] == {'a': 10, 'b': 20}

    def test_loop_over_enumerate(self, cash_magics, mock_shell):
        """for i, x in enumerate(lst) pattern."""
        magics = cash_magics
        shell = mock_shell

        magics.cash_on("")

        shell.user_ns['items'] = ['a', 'b', 'c']

        code = """results = {}
for i, x in enumerate(items):
    results[i] = x
"""
        magics.cash("", code)
        assert shell.user_ns['results'] == {0: 'a', 1: 'b', 2: 'c'}

    def test_loop_over_zip(self, cash_magics, mock_shell):
        """for a, b in zip(l1, l2) pattern."""
        magics = cash_magics
        shell = mock_shell

        magics.cash_on("")

        shell.user_ns['keys'] = ['x', 'y', 'z']
        shell.user_ns['vals'] = [10, 20, 30]

        code = """results = {}
for k, v in zip(keys, vals):
    results[k] = v
"""
        magics.cash("", code)
        assert shell.user_ns['results'] == {'x': 10, 'y': 20, 'z': 30}


# ============================================================================
# Group 5: While Loops
# ============================================================================

class TestWhileLoopMutation:
    """Test mutations inside while loops."""

    def test_while_loop_counter_mutation(self, cash_magics, mock_shell):
        """count += 1 in while loop."""
        magics = cash_magics
        shell = mock_shell

        magics.cash_on("")

        code = """count = 0
while count < 5:
    count += 1
"""
        magics.cash("", code)
        assert shell.user_ns['count'] == 5

    def test_while_loop_list_append(self, cash_magics, mock_shell):
        """results.append(x) in while loop."""
        magics = cash_magics
        shell = mock_shell

        magics.cash_on("")

        code = """results = []
i = 0
while i < 3:
    results.append(i * 10)
    i += 1
"""
        magics.cash("", code)
        assert shell.user_ns['results'] == [0, 10, 20]
        assert shell.user_ns['i'] == 3

    def test_while_loop_condition_depends_on_mutation(self, cash_magics, mock_shell):
        """While loop where condition depends on mutated variable."""
        magics = cash_magics
        shell = mock_shell

        magics.cash_on("")

        code = """queue = [1, 2, 3]
processed = []
while len(queue) > 0:
    item = queue.pop(0)
    processed.append(item * 2)
"""
        magics.cash("", code)
        assert shell.user_ns['processed'] == [2, 4, 6]
        assert shell.user_ns['queue'] == []


# ============================================================================
# Group 6: Upstream Trust Logic
# ============================================================================

class TestUpstreamLoopTrust:
    """Test that the upstream checker correctly handles loop-mutated variables."""

    def _make_notebook(self, notebook_path, cells):
        """Helper to create a notebook JSON file."""
        notebook = {
            "cells": [
                {"cell_type": "code", "execution_count": None,
                 "metadata": {}, "outputs": [], "source": cell}
                for cell in cells
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4
        }
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f)

    def _run_with_notebook(self, magics, code, notebook_path):
        """Helper to run code with notebook cells patched."""
        def get_cells(_path=None):
            with open(notebook_path, encoding='utf-8') as nf:
                data = json.load(nf)
                return [cell['source'] for cell in data['cells'] if cell['cell_type'] == 'code']

        with patch('cash.notebook.upstream.get_notebook_cells') as mock_get_cells, \
             patch('cash.notebook.upstream.get_notebook_cells_with_ids') as mock_get_ids:
                mock_get_cells.side_effect = get_cells
                mock_get_ids.return_value = []
                magics.cash("", code)

    def test_upstream_trusts_loop_vars_when_unchanged(self, cash_magics, mock_shell, tmp_path):
        """When upstream code hasn't changed, loop-mutated vars should be trusted in-memory."""
        magics = cash_magics
        shell = mock_shell

        loop_code = """results = {}
for x in ["A", "B"]:
    results[x] = x * 2
"""
        downstream_code = "output = list(results.keys())"

        notebook_path = str(tmp_path / 'test.ipynb')
        self._make_notebook(notebook_path, [loop_code, downstream_code])

        # First run
        self._run_with_notebook(magics, loop_code, notebook_path)
        assert shell.user_ns['results'] == {'A': 'AA', 'B': 'BB'}

        self._run_with_notebook(magics, downstream_code, notebook_path)
        assert set(shell.user_ns['output']) == {'A', 'B'}

        # Second run of downstream — upstream unchanged, should trust in-memory
        self._run_with_notebook(magics, downstream_code, notebook_path)
        assert set(shell.user_ns['output']) == {'A', 'B'}

    @pytest.mark.xfail(reason="Known failure: upstream loop variable re-execution")
    def test_upstream_reexecutes_loop_vars_when_changed(self, cash_magics, mock_shell, tmp_path):
        """When upstream loop code changes, loop-mutated vars must be re-executed."""
        magics = cash_magics
        shell = mock_shell

        loop_code_v1 = """results = {}
for x in ["A", "B", "C", "D"]:
    results[x] = x * 2
"""
        loop_code_v2 = """results = {}
for x in ["A", "B", "C", "D", "E"]:
    results[x] = x * 2
"""
        downstream_code = "output = list(results.keys())"
        notebook_path = str(tmp_path / 'test.ipynb')

        # Run with v1
        self._make_notebook(notebook_path, [loop_code_v1, downstream_code])
        self._run_with_notebook(magics, loop_code_v1, notebook_path)
        assert 'results' in shell.user_ns
        assert len(shell.user_ns['results']) == 4

        self._run_with_notebook(magics, downstream_code, notebook_path)
        assert set(shell.user_ns['output']) == {'A', 'B', 'C', 'D'}

        # Update notebook to v2 (add "E"), run ONLY downstream
        self._make_notebook(notebook_path, [loop_code_v2, downstream_code])
        self._run_with_notebook(magics, downstream_code, notebook_path)

        # Should have all 5 items
        results = shell.user_ns['results']
        assert 'A' in results, f"Missing 'A': {results}"
        assert 'E' in results, f"Missing 'E': {results}"
        assert len(results) == 5, f"Expected 5 items, got {len(results)}: {results}"

    def test_upstream_transitive_trust_unchanged(self, cash_magics, mock_shell, tmp_path):
        """Variables derived from loop-mutated vars should be trusted when upstream unchanged."""
        magics = cash_magics
        shell = mock_shell

        loop_code = """data = {}
for x in ["A", "B"]:
    data[x] = len(x)
"""
        derived_code = "summary = list(data.values())"
        downstream_code = "total = sum(summary)"

        notebook_path = str(tmp_path / 'test.ipynb')
        self._make_notebook(notebook_path, [loop_code, derived_code, downstream_code])

        # Execute all cells
        self._run_with_notebook(magics, loop_code, notebook_path)
        self._run_with_notebook(magics, derived_code, notebook_path)
        self._run_with_notebook(magics, downstream_code, notebook_path)

        assert shell.user_ns['total'] == 2  # len('A') + len('B') = 1 + 1

        # Re-run downstream — upstream unchanged, should trust
        self._run_with_notebook(magics, downstream_code, notebook_path)
        assert shell.user_ns['total'] == 2


# ============================================================================
# Group 7: MutationDetector Accuracy
# ============================================================================

class TestMutationDetectorAccuracy:
    """Test that analyze_statement accurately identifies loop mutations."""

    def test_subscript_assignment_detected(self):
        """d[k] = v should be detected as mutating d."""
        mutated = analyze_statement("results[x] = x * 2", None).all_mutated_vars
        assert 'results' in mutated

    def test_method_call_append_detected(self):
        """lst.append(x) should be detected as mutating lst."""
        mutated = analyze_statement("results.append(x)", None).all_mutated_vars
        assert 'results' in mutated

    def test_method_call_update_detected(self):
        """d.update({...}) should be detected as mutating d."""
        mutated = analyze_statement("d.update({x: y})", None).all_mutated_vars
        assert 'd' in mutated

    def test_augmented_assign_detected(self):
        """total += x should be detected as mutating total."""
        mutated = analyze_statement("total += x", None).all_mutated_vars
        assert 'total' in mutated

    def test_attribute_assign_detected(self):
        """obj.attr = val should be detected as mutating obj."""
        mutated = analyze_statement("obj.attr = val", None).all_mutated_vars
        assert 'obj' in mutated

    def test_read_only_not_detected(self):
        """Reading a variable should NOT be detected as mutation."""
        mutated = analyze_statement("y = df[df['col'] > 0]", None).all_mutated_vars
        assert 'df' not in mutated

    def test_simple_assignment_not_detected(self):
        """Simple assignment is NOT a mutation (it's a rebinding)."""
        mutated = analyze_statement("x = 42", None).all_mutated_vars
        assert 'x' not in mutated

    def test_function_call_not_mutation(self):
        """Calling a function on a variable isn't necessarily a mutation."""
        mutated = analyze_statement("y = len(results)", None).all_mutated_vars
        assert 'results' not in mutated

    def test_del_subscript_detected(self):
        """del d[k] should be detected as mutating d."""
        mutated = analyze_statement("del d[k]", None).all_mutated_vars
        assert 'd' in mutated

    def test_loop_body_mutation_detection(self):
        """Full loop body: only the actually mutated variable should be detected."""
        # In this loop body:
        # - ticker_data is a NEW variable (not mutation)
        # - stats is a NEW variable
        # - ticker_stats is MUTATED via subscript assignment
        code = """ticker_data = df[df['Ticker'] == ticker]
stats = {'mean': ticker_data['Price'].mean()}
ticker_stats[ticker] = stats"""

        mutated = analyze_statement(code, None).all_mutated_vars
        assert 'ticker_stats' in mutated
        assert 'df' not in mutated  # df is only read, not mutated
        assert 'ticker_data' not in mutated  # new assignment, not mutation
        assert 'stats' not in mutated  # new assignment, not mutation


# ============================================================================
# Group 8: Edge Cases
# ============================================================================

class TestLoopEdgeCases:
    """Test edge cases in loop caching."""

    def test_empty_loop(self, cash_magics, mock_shell):
        """for x in []: ... should not break caching."""
        magics = cash_magics
        shell = mock_shell

        magics.cash_on("")

        code = """results = []
for x in []:
    results.append(x)
"""
        magics.cash("", code)
        assert shell.user_ns['results'] == []

    def test_single_iteration_loop(self, cash_magics, mock_shell):
        """Loop with single iteration."""
        magics = cash_magics
        shell = mock_shell

        magics.cash_on("")

        code = """results = {}
for x in ["only"]:
    results[x] = 42
"""
        magics.cash("", code)
        assert shell.user_ns['results'] == {'only': 42}

    def test_nested_loop_dict_mutation(self, cash_magics, mock_shell):
        """Nested loops both mutating same dict."""
        magics = cash_magics
        shell = mock_shell

        magics.cash_on("")

        code = """results = {}
for i in range(2):
    for j in range(2):
        results[(i, j)] = i * 10 + j
"""
        magics.cash("", code)
        assert shell.user_ns['results'] == {(0, 0): 0, (0, 1): 1, (1, 0): 10, (1, 1): 11}

    def test_loop_with_if_inside(self, cash_magics, mock_shell):
        """Loop with conditional inside."""
        magics = cash_magics
        shell = mock_shell

        magics.cash_on("")

        code = """evens = []
odds = []
for x in range(6):
    if x % 2 == 0:
        evens.append(x)
    else:
        odds.append(x)
"""
        magics.cash("", code)
        assert shell.user_ns['evens'] == [0, 2, 4]
        assert shell.user_ns['odds'] == [1, 3, 5]

    def test_loop_init_before_loop(self, cash_magics, mock_shell):
        """Accumulator init + loop in same cell."""
        magics = cash_magics
        shell = mock_shell

        magics.cash_on("")

        code = """total = 0
items = []
for x in [1, 2, 3]:
    total += x
    items.append(x * 2)
"""
        magics.cash("", code)
        assert shell.user_ns['total'] == 6
        assert shell.user_ns['items'] == [2, 4, 6]

    def test_loop_modifies_and_reads_same_var(self, cash_magics, mock_shell):
        """Loop body reads and writes same variable (accumulation pattern)."""
        magics = cash_magics
        shell = mock_shell

        magics.cash_on("")

        code = """s = ""
for c in ["hello", " ", "world"]:
    s += c
"""
        magics.cash("", code)
        assert shell.user_ns['s'] == "hello world"


# ============================================================================
# Group 9: CodeAnalyzer + MutationDetector Integration
# ============================================================================

class TestCodeAnalyzerMutationIntegration:
    """Test that CodeAnalyzer correctly classifies mutation patterns."""

    def test_subscript_assign_has_var_in_outputs(self):
        """results[x] = v should have 'results' in outputs."""
        inputs, outputs = CodeAnalyzer.analyze_code_block("results[x] = x * 2")
        assert 'results' in outputs

    def test_method_mutation_has_var_in_inputs(self):
        """results.append(x) — 'results' is in inputs (it's being called on)."""
        inputs, outputs = CodeAnalyzer.analyze_code_block("results.append(x)")
        assert 'results' in inputs

    def test_augmented_assign_has_var_in_outputs(self):
        """total += x should have 'total' in outputs."""
        inputs, outputs = CodeAnalyzer.analyze_code_block("total += x")
        assert 'total' in outputs

    def test_iteration_context_stripped_for_analysis(self):
        """CodeAnalyzer should handle iteration context prefix."""
        code = "# __iteration_context__: abc123\nresults[x] = x * 2"
        inputs, outputs = CodeAnalyzer.analyze_code_block(code)
        assert 'results' in outputs
        assert 'x' in inputs
