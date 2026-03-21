"""
Integration tests for duplicate expression statement caching.

BUG (Finding F2): When the same expression statement appears multiple times 
in a cell (e.g., `c.increment()` called 3 times), the 2nd and subsequent calls 
are RESTORED from cache instead of COMPUTED. This is because all identical 
statements produce the same cache key - there is no occurrence counter.

For pure functions (like print), this is cosmetically wrong but functionally 
correct. For stateful method calls, this causes DATA CORRUPTION because the 
object's state is only modified once.

Root cause: cache_key.py:compute_cache_key() doesn't include a statement 
occurrence index, so identical statements get the same cache key.
"""
import pytest

pytestmark = pytest.mark.core


class TestDuplicateExpressionStatements:
    """Tests for duplicate expression statement deduplication bug."""

    def test_duplicate_method_calls_with_mutation(self, nb_runner):
        """
        Core bug: calling obj.method() multiple times without assignment
        should execute each call, not cache-deduplicate them.
        """
        nb_runner.create_notebook([
            (
                "class Counter:\n"
                "    def __init__(self, start=0):\n"
                "        self.value = start\n"
                "    def increment(self):\n"
                "        self.value += 1\n"
                "        return self.value\n"
                "\n"
                "c = Counter(10)\n"
                "c.increment()\n"
                "c.increment()\n"
                "c.increment()\n"
                "print(f'c.value = {c.value}')"
            ),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        
        out = nb_runner.get_output(1)
        assert "c.value = 13" in out, (
            f"Counter should be 13 (10 + 3 increments). "
            f"If it shows 11, duplicate calls were cached. Got: {out}"
        )

    def test_duplicate_method_calls_return_values_differ(self, nb_runner):
        """
        When return values are assigned to variables, each call should
        produce different results. This tests the workaround of using
        assignment to differentiate calls.
        """
        nb_runner.create_notebook([
            (
                "class Counter:\n"
                "    def __init__(self, start=0):\n"
                "        self.value = start\n"
                "    def increment(self):\n"
                "        self.value += 1\n"
                "        return self.value\n"
                "\n"
                "c = Counter(10)\n"
                "r1 = c.increment()\n"
                "r2 = c.increment()\n"
                "r3 = c.increment()\n"
                "print(f'results: {r1}, {r2}, {r3}')"
            ),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        
        out = nb_runner.get_output(1)
        assert "results: 11, 12, 13" in out, (
            f"With variable assignment, each call should produce unique results. "
            f"Got: {out}"
        )

    def test_duplicate_list_append_calls(self, nb_runner):
        """
        Multiple list.append() calls should all execute.
        This is important for accumulator patterns.
        """
        nb_runner.create_notebook([
            (
                "items = []\n"
                "items.append('a')\n"
                "items.append('b')\n"
                "items.append('c')\n"
                "print(f'items = {items}')"
            ),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        
        out = nb_runner.get_output(1)
        assert "items = ['a', 'b', 'c']" in out, (
            f"All three append calls should execute. Got: {out}"
        )

    def test_duplicate_print_calls_output_correct(self, nb_runner):
        """
        Even though duplicate print calls share cache keys,
        the output should be reproduced correctly on cache restore.
        """
        nb_runner.create_notebook([
            (
                "print('hello')\n"
                "print('hello')\n"
                "print('hello')"
            ),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        
        out = nb_runner.get_output(1)
        # Count occurrences of 'hello' in the non-badge output
        hello_count = out.count('hello')
        assert hello_count >= 3, (
            f"Three print('hello') calls should produce at least 3 'hello' outputs. "
            f"Got {hello_count} in: {out}"
        )

    def test_duplicate_calls_across_re_execution(self, nb_runner):
        """
        After initial run, re-executing should still produce correct results
        for duplicate method calls.
        """
        nb_runner.create_notebook([
            (
                "class Accumulator:\n"
                "    def __init__(self):\n"
                "        self.total = 0\n"
                "    def add(self, val):\n"
                "        self.total += val\n"
                "        return self.total\n"
                "\n"
                "acc = Accumulator()\n"
                "acc.add(10)\n"
                "acc.add(20)\n"
                "acc.add(30)\n"
                "print(f'total = {acc.total}')"
            ),
        ])
        nb_runner.start_kernel()
        
        # First run
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "total = 60" in out1, f"First run: expected total=60. Got: {out1}"
        
        # Second run (should use cache but still be correct)
        nb_runner.run_all()
        out2 = nb_runner.get_output(1)
        assert "total = 60" in out2, (
            f"Second run: expected total=60 (cached). Got: {out2}"
        )

    def test_duplicate_global_function_calls(self, nb_runner):
        """
        Duplicate calls to a global function with side effects should
        each be executed.
        """
        nb_runner.create_notebook([
            (
                "call_log = []\n"
                "def track():\n"
                "    call_log.append(len(call_log))\n"
                "\n"
                "track()\n"
                "track()\n"
                "track()\n"
                "print(f'call_log = {call_log}')"
            ),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        
        out = nb_runner.get_output(1)
        assert "call_log = [0, 1, 2]" in out, (
            f"Three track() calls should produce [0, 1, 2]. Got: {out}"
        )
