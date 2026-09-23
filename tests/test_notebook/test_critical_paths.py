"""
Phase 1.1: Additional test coverage for critical paths.

Tests covering:
- Loop caching edge cases (nested loops, early break, continue)
- Cascading invalidation across multiple notebook cells
- Unicode/special characters in variable names
- File dependency tracking for various readers
- Error recovery and edge cases
"""

from tests._cell_driver import run_cash_cell

# ============================================================================
# Loop Caching Edge Cases
# ============================================================================


class TestLoopCachingEdgeCases:
    """Test loop caching for edge cases: break, continue, nested, empty."""

    def test_loop_with_append(self, cash_magics, mock_shell):
        """Loop that appends to a list should execute correctly on first run."""
        code = "results = []\nfor i in range(5):\n    results.append(i)\n"
        run_cash_cell(cash_magics, code)
        assert mock_shell.user_ns["results"] == [0, 1, 2, 3, 4]

    def test_loop_with_continue(self, cash_magics, mock_shell):
        """Loop with continue should skip iterations correctly."""
        code = "evens = []\nfor i in range(6):\n    if i % 2 != 0:\n        continue\n    evens.append(i)\n"
        run_cash_cell(cash_magics, code)
        assert mock_shell.user_ns["evens"] == [0, 2, 4]

    def test_nested_loops(self, cash_magics, mock_shell):
        """Nested loops should be handled correctly."""
        code = "pairs = []\nfor i in range(3):\n    for j in range(2):\n        pairs.append((i, j))\n"
        run_cash_cell(cash_magics, code)
        expected = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
        assert mock_shell.user_ns["pairs"] == expected

    def test_empty_loop(self, cash_magics, mock_shell):
        """Loop over empty iterable should produce no output."""
        code = "items = []\nfor x in []:\n    items.append(x)\n"
        run_cash_cell(cash_magics, code)
        assert mock_shell.user_ns["items"] == []

    def test_while_loop(self, cash_magics, mock_shell):
        """While loop with mutation should execute correctly."""
        code = "count = 0\nwhile count < 5:\n    count += 1\n"
        run_cash_cell(cash_magics, code)
        assert mock_shell.user_ns["count"] == 5

    def test_loop_with_dict_accumulation(self, cash_magics, mock_shell):
        """Loop that builds a dictionary should work across runs."""
        code = "mapping = {}\nfor k in ['a', 'b', 'c']:\n    mapping[k] = len(k)\n"
        run_cash_cell(cash_magics, code)
        assert mock_shell.user_ns["mapping"] == {"a": 1, "b": 1, "c": 1}

    def test_enumerate_loop(self, cash_magics, mock_shell):
        """Loop with enumerate should be handled."""
        code = "indexed = {}\nfor idx, val in enumerate(['x', 'y', 'z']):\n    indexed[idx] = val\n"
        run_cash_cell(cash_magics, code)
        assert mock_shell.user_ns["indexed"] == {0: "x", 1: "y", 2: "z"}


# ============================================================================
# Cascading Invalidation
# ============================================================================


class TestCascadingInvalidation:
    """Test that changes to upstream values correctly invalidate downstream."""

    def test_simple_chain_invalidation(self, cash_magics, mock_shell):
        """Changing x should invalidate y that depends on x."""
        # Cell 1: x = 10
        run_cash_cell(cash_magics, "x = 10")
        assert mock_shell.user_ns["x"] == 10

        # Cell 2: y = x * 2
        run_cash_cell(cash_magics, "y = x * 2")
        assert mock_shell.user_ns["y"] == 20

        # Cell 3: z = y + 1
        run_cash_cell(cash_magics, "z = y + 1")
        assert mock_shell.user_ns["z"] == 21

        # Change x
        run_cash_cell(cash_magics, "x = 100")
        assert mock_shell.user_ns["x"] == 100

        # y and z should be invalidated and recomputed if we re-run
        run_cash_cell(cash_magics, "y = x * 2")
        assert mock_shell.user_ns["y"] == 200

        run_cash_cell(cash_magics, "z = y + 1")
        assert mock_shell.user_ns["z"] == 201

    def test_independent_chains_no_cross_invalidation(self, cash_magics, mock_shell):
        """Independent chains should not affect each other."""
        # Chain A
        run_cash_cell(cash_magics, "a = 1")
        run_cash_cell(cash_magics, "b = a + 1")
        assert mock_shell.user_ns["b"] == 2

        # Chain B (independent)
        run_cash_cell(cash_magics, "p = 10")
        run_cash_cell(cash_magics, "q = p * 3")
        assert mock_shell.user_ns["q"] == 30

        # Change chain A
        run_cash_cell(cash_magics, "a = 999")
        # Chain B should be unaffected
        run_cash_cell(cash_magics, "q = p * 3")
        assert mock_shell.user_ns["q"] == 30

    def test_diamond_dependency(self, cash_magics, mock_shell):
        """Diamond dependency: A -> B, A -> C, B+C -> D."""
        run_cash_cell(cash_magics, "a = 5")
        run_cash_cell(cash_magics, "b = a + 1")
        run_cash_cell(cash_magics, "c = a * 2")
        run_cash_cell(cash_magics, "d = b + c")
        assert mock_shell.user_ns["d"] == 16  # 6 + 10

        # Change a
        run_cash_cell(cash_magics, "a = 10")
        run_cash_cell(cash_magics, "b = a + 1")
        run_cash_cell(cash_magics, "c = a * 2")
        run_cash_cell(cash_magics, "d = b + c")
        assert mock_shell.user_ns["d"] == 31  # 11 + 20


# ============================================================================
# Unicode and Special Characters
# ============================================================================


class TestUnicodeVariables:
    """Test handling of unicode and special patterns in variable names."""

    def test_underscore_variable(self, cash_magics, mock_shell):
        """Variables starting with underscore should work."""
        run_cash_cell(cash_magics, "_private = 42")
        assert mock_shell.user_ns["_private"] == 42

    def test_numeric_suffix_variable(self, cash_magics, mock_shell):
        """Variables with numbers should work."""
        run_cash_cell(cash_magics, "var123 = 'hello'")
        assert mock_shell.user_ns["var123"] == "hello"

    def test_long_variable_name(self, cash_magics, mock_shell):
        """Very long variable names should work."""
        long_name = "very_long_variable_name_" * 5
        run_cash_cell(cash_magics, f"{long_name} = 99")
        assert mock_shell.user_ns[long_name] == 99

    def test_multiple_assignments(self, cash_magics, mock_shell):
        """Multiple assignment targets should all be cached."""
        run_cash_cell(cash_magics, "x, y, z = 1, 2, 3")
        assert mock_shell.user_ns["x"] == 1
        assert mock_shell.user_ns["y"] == 2
        assert mock_shell.user_ns["z"] == 3


# ============================================================================
# File Dependency Tracking
# ============================================================================


class TestFileDependencyTracking:
    """Test file tracking for various read operations."""

    def test_csv_file_tracking(self, cash_magics, mock_shell, tmp_path):
        """CSV file read should be tracked as dependency."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        path_str = str(csv_path).replace("\\", "/")

        code = f"import pandas as pd\ndf = pd.read_csv('{path_str}')"
        run_cash_cell(cash_magics, code)
        assert "df" in mock_shell.user_ns
        assert len(mock_shell.user_ns["df"]) == 2

    def test_json_file_tracking(self, cash_magics, mock_shell, tmp_path):
        """JSON file read should be tracked."""
        import json

        json_path = tmp_path / "test.json"
        json_path.write_text(json.dumps({"key": "value", "num": 42}), encoding="utf-8")
        path_str = str(json_path).replace("\\", "/")

        code = f"import json\nwith open('{path_str}') as f:\n    data = json.load(f)"
        run_cash_cell(cash_magics, code)
        assert mock_shell.user_ns["data"] == {"key": "value", "num": 42}

    def test_text_file_tracking(self, cash_magics, mock_shell, tmp_path):
        """Plain text file read should be tracked."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("hello world", encoding="utf-8")
        path_str = str(txt_path).replace("\\", "/")

        code = f"with open('{path_str}') as f:\n    content = f.read()"
        run_cash_cell(cash_magics, code)
        assert mock_shell.user_ns["content"] == "hello world"

    def test_file_change_invalidates_cache(self, cash_magics, mock_shell, tmp_path):
        """Changing a file should invalidate cached results."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
        path_str = str(csv_path).replace("\\", "/")

        code = f"import pandas as pd\ndf = pd.read_csv('{path_str}')\nrow_count = len(df)"
        run_cash_cell(cash_magics, code)
        assert mock_shell.user_ns["row_count"] == 1

        # Modify the file
        csv_path.write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")

        # Re-run should detect file change
        run_cash_cell(cash_magics, code)
        assert mock_shell.user_ns["row_count"] == 3

    def test_size_change_invalidates_cache_with_same_mtime(self, cash_magics, mock_shell, tmp_path):
        """Same mtime + different size must still invalidate.

        Regression test for the macOS / coarse-mtime-filesystem bug where
        two back-to-back rewrites of the same CSV produced identical mtimes
        and the cache stayed valid even though the content changed.  We
        force the post-rewrite mtime to match the original via os.utime to
        simulate that filesystem behaviour deterministically — the
        invalidator must still notice via the size delta.
        """
        import os as _os

        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
        original_mtime = _os.stat(csv_path).st_mtime
        path_str = str(csv_path).replace("\\", "/")

        code = f"import pandas as pd\ndf = pd.read_csv('{path_str}')\nrow_count = len(df)"
        run_cash_cell(cash_magics, code)
        assert mock_shell.user_ns["row_count"] == 1

        # Rewrite with different content but pin the mtime to its original
        # value, simulating a filesystem with coarse mtime granularity.
        csv_path.write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")
        _os.utime(csv_path, (original_mtime, original_mtime))
        assert _os.stat(csv_path).st_mtime == original_mtime  # sanity

        run_cash_cell(cash_magics, code)
        # If the invalidator only checked mtime, this would still be 1.
        assert mock_shell.user_ns["row_count"] == 3


# ============================================================================
# Error Recovery Edge Cases
# ============================================================================


class TestErrorRecovery:
    """Test graceful handling of errors during caching."""

    def test_syntax_error_in_code(self, cash_magics, mock_shell):
        """Syntax errors should be handled gracefully."""
        # SyntaxError should propagate to user, not crash cash
        import contextlib

        with contextlib.suppress(SyntaxError):
            run_cash_cell(cash_magics, "x = ")  # SyntaxError
        # Cash should still be functional
        run_cash_cell(cash_magics, "y = 42")
        assert mock_shell.user_ns["y"] == 42

    def test_runtime_error_recovery(self, cash_magics, mock_shell):
        """Runtime errors should not corrupt cache state."""
        run_cash_cell(cash_magics, "x = 10")
        import contextlib

        with contextlib.suppress(ZeroDivisionError, Exception):
            run_cash_cell(cash_magics, "y = 1 / 0")  # ZeroDivisionError
        # Should still work
        run_cash_cell(cash_magics, "z = x + 1")
        assert mock_shell.user_ns["z"] == 11

    def test_large_data_caching(self, cash_magics, mock_shell):
        """Large data structures should cache without issues."""
        code = "big_list = list(range(100000))"
        run_cash_cell(cash_magics, code)
        assert len(mock_shell.user_ns["big_list"]) == 100000

    def test_none_value_caching(self, cash_magics, mock_shell):
        """None values should be cacheable."""
        run_cash_cell(cash_magics, "result = None")
        assert mock_shell.user_ns["result"] is None

    def test_empty_string_caching(self, cash_magics, mock_shell):
        """Empty strings should be cacheable."""
        run_cash_cell(cash_magics, "text = ''")
        assert mock_shell.user_ns["text"] == ""

    def test_boolean_caching(self, cash_magics, mock_shell):
        """Boolean values should be cacheable."""
        run_cash_cell(cash_magics, "flag = True")
        assert mock_shell.user_ns["flag"] is True

    def test_tuple_caching(self, cash_magics, mock_shell):
        """Tuples should be cacheable."""
        run_cash_cell(cash_magics, "coords = (1.5, 2.5, 3.5)")
        assert mock_shell.user_ns["coords"] == (1.5, 2.5, 3.5)

    def test_set_caching(self, cash_magics, mock_shell):
        """Sets should be cacheable."""
        run_cash_cell(cash_magics, "unique = {1, 2, 3}")
        assert mock_shell.user_ns["unique"] == {1, 2, 3}


# ============================================================================
# Statement Ordering and Dependencies
# ============================================================================


class TestStatementOrdering:
    """Test that statement ordering is respected during caching."""

    def test_sequential_statements(self, cash_magics, mock_shell):
        """Statements in a cell should execute in order."""
        code = "a = 1\nb = a + 1\nc = b + 1"
        run_cash_cell(cash_magics, code)
        assert mock_shell.user_ns["a"] == 1
        assert mock_shell.user_ns["b"] == 2
        assert mock_shell.user_ns["c"] == 3

    def test_overwrite_variable(self, cash_magics, mock_shell):
        """Overwriting a variable should use the latest value."""
        code = "x = 1\nx = x + 10\nx = x * 2"
        run_cash_cell(cash_magics, code)
        assert mock_shell.user_ns["x"] == 22

    def test_conditional_assignment(self, cash_magics, mock_shell):
        """Conditional assignments should produce correct values."""
        code = "x = 10\ny = 'big' if x > 5 else 'small'"
        run_cash_cell(cash_magics, code)
        assert mock_shell.user_ns["y"] == "big"

    def test_list_comprehension(self, cash_magics, mock_shell):
        """List comprehensions should be cacheable."""
        code = "squares = [i**2 for i in range(10)]"
        run_cash_cell(cash_magics, code)
        assert mock_shell.user_ns["squares"] == [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

    def test_dict_comprehension(self, cash_magics, mock_shell):
        """Dict comprehensions should be cacheable."""
        code = "mapping = {str(i): i**2 for i in range(5)}"
        run_cash_cell(cash_magics, code)
        expected = {"0": 0, "1": 1, "2": 4, "3": 9, "4": 16}
        assert mock_shell.user_ns["mapping"] == expected


# ============================================================================
# Multi-cell Interactions
# ============================================================================


class TestMultiCellInteractions:
    """Test interactions between multiple cells."""

    def test_variable_sharing_between_cells(self, cash_magics, mock_shell):
        """Variables from one cell should be accessible in next."""
        run_cash_cell(cash_magics, "base = 100")
        run_cash_cell(cash_magics, "derived = base * 2")
        assert mock_shell.user_ns["derived"] == 200

    def test_function_definition_and_use(self, cash_magics, mock_shell):
        """Function defined in one cell should work in another."""
        run_cash_cell(cash_magics, "def double(x):\n    return x * 2")
        run_cash_cell(cash_magics, "result = double(21)")
        assert mock_shell.user_ns["result"] == 42

    def test_import_and_use(self, cash_magics, mock_shell):
        """Import in one cell should be available in next."""
        run_cash_cell(cash_magics, "import math")
        run_cash_cell(cash_magics, "pi_val = round(math.pi, 2)")
        assert mock_shell.user_ns["pi_val"] == 3.14

    def test_class_definition_and_instantiation(self, cash_magics, mock_shell):
        """Class defined in one cell should be instantiable in another."""
        run_cash_cell(
            cash_magics, "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y"
        )
        run_cash_cell(cash_magics, "p = Point(3, 4)")
        assert mock_shell.user_ns["p"].x == 3
        assert mock_shell.user_ns["p"].y == 4
