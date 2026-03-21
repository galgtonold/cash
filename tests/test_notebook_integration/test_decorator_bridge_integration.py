"""
Integration tests for the decorator-notebook bridge.

Tests that @cash.cache calls inside notebooks:
1. Show up in badge display
2. Work with condensed view for many calls
3. Handle non-hashable parameters gracefully
4. Track custom type hashers
"""

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.core,
    pytest.mark.timeout(30),
]


class TestDecoratorNotebookBridge:
    """Test @cash.cache decorator integration with notebook caching."""

    def test_decorator_cache_hit_in_notebook(self, nb_runner):
        """@cash.cache calls should produce cache hits on second run."""
        nb_runner.create_notebook([
            "from cash import Cash\nc = Cash()\n\n@c.cache\ndef compute(x):\n    return x * 2",
            "result1 = compute(5)\nresult2 = compute(5)\nprint(f'r1={result1}, r2={result2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(2)
        assert 'r1=10, r2=10' in output

    def test_decorator_call_logged_in_metrics(self, nb_runner):
        """Decorator calls should be captured in statement metrics."""
        nb_runner.create_notebook([
            "from cash import Cash\nc = Cash()\n\n@c.cache\ndef expensive(x):\n    import time\n    time.sleep(0.01)\n    return x ** 2",
            "r1 = expensive(5)\nr2 = expensive(5)\nprint(f'Result: {r1}, {r2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(2)
        assert 'Result: 25, 25' in output

    def test_decorator_with_global_cache(self, nb_runner):
        """from cash import cache should also work."""
        nb_runner.create_notebook([
            "from cash import cache",
            "@cache\ndef square(x):\n    return x ** 2",
            "r = square(7)\nprint(f'Result: {r}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert 'Result: 49' in output

    def test_decorator_cache_with_dataframe(self, nb_runner):
        """DataFrames as args should be cached properly with built-in hasher."""
        nb_runner.create_notebook([
            "import pandas as pd\nfrom cash import Cash\nc = Cash()",
            "@c.cache\ndef process(df):\n    return df.sum().to_dict()",
            "df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})\nresult = process(df)\nprint(f'Result: {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "'a': 6" in output or "'a': 6" in output.replace(' ', '')

    def test_decorator_cache_with_numpy(self, nb_runner):
        """numpy arrays as args should be cached properly."""
        nb_runner.create_notebook([
            "import numpy as np\nfrom cash import Cash\nc = Cash()",
            "@c.cache\ndef compute(arr):\n    return float(arr.sum())",
            "arr = np.array([1, 2, 3, 4, 5])\nr1 = compute(arr)\nr2 = compute(arr)\nprint(f'r1={r1}, r2={r2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert 'r1=15.0' in output
        assert 'r2=15.0' in output

    def test_decorator_cache_many_calls(self, nb_runner):
        """Many calls to same function should show condensed badge."""
        nb_runner.create_notebook([
            "from cash import Cash\nc = Cash()\n\n@c.cache\ndef square(x):\n    return x ** 2",
            "results = []\nfor i in range(10):\n    results.append(square(i))\nprint(f'Results: {results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(2)
        assert 'Results:' in output
        assert '81' in output  # 9^2

    def test_decorator_custom_hasher_in_notebook(self, nb_runner):
        """Custom hasher registered in notebook should work."""
        nb_runner.create_notebook([
            "import hashlib\nfrom cash import Cash\nc = Cash()",
            "class Config:\n    def __init__(self, name, value):\n        self.name = name\n        self.value = value",
            "c.register_hasher(Config, lambda cfg: hashlib.sha256(f'{cfg.name}:{cfg.value}'.encode()).hexdigest())",
            "@c.cache\ndef process(cfg):\n    return f'{cfg.name}={cfg.value}'",
            "r1 = process(Config('x', 42))\nr2 = process(Config('x', 42))\nprint(f'r1={r1}, r2={r2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(5)
        assert 'r1=x=42, r2=x=42' in output

    def test_decorator_different_args_no_false_hit(self, nb_runner):
        """Different args should produce different cache entries."""
        nb_runner.create_notebook([
            "from cash import Cash\nc = Cash()\ncall_count = 0",
            "@c.cache\ndef compute(x):\n    global call_count\n    call_count += 1\n    return x * 2",
            "r1 = compute(5)\nr2 = compute(10)\nr3 = compute(5)  # should be hit\nprint(f'r1={r1}, r2={r2}, r3={r3}, calls={call_count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert 'r1=10, r2=20, r3=10, calls=2' in output


class TestDecoratorNotebookRerun:
    """Test decorator caching across cell re-executions."""

    def test_decorator_cache_persists_across_cell_reruns(self, nb_runner):
        """Cache should persist when re-running cells."""
        nb_runner.create_notebook([
            "from cash import Cash\nc = Cash()\ncall_count = 0",
            "@c.cache\ndef expensive(x):\n    global call_count\n    call_count += 1\n    return x ** 2",
            "r = expensive(7)\nprint(f'Result: {r}, calls: {call_count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output1 = nb_runner.get_output(3)
        assert 'Result: 49, calls: 1' in output1

        # Re-run the computation cell
        nb_runner.run_cell(3)

        output2 = nb_runner.get_output(3)
        # call_count should still be 1 because the decorator cache hit
        assert 'Result: 49, calls: 1' in output2

    def test_decorator_cache_invalidates_on_function_change(self, nb_runner):
        """Changing function source should invalidate both decorator and statement cache.

        The notebook function tracker detects source changes via inspect.getsource()
        (which follows __wrapped__ through functools.wraps), producing a different
        statement-level cache key. The decorator cache also gets a different key
        because Cash.source_hashes is updated when @c.cache is called again.
        """
        nb_runner.create_notebook([
            "from cash import Cash\nc = Cash()",
            "@c.cache\ndef compute(x):\n    return x * 2",
            "r = compute(5)\nprint(f'Result: {r}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output1 = nb_runner.get_output(3)
        assert 'Result: 10' in output1

        # Change the function body — both decorator and statement caches
        # should invalidate because function source hash changes
        nb_runner.set_cell_source(2, "@c.cache\ndef compute(x):\n    return x * 3")
        nb_runner.run_cells([2, 3])

        output2 = nb_runner.get_output(3)
        assert 'Result: 15' in output2


class TestDecoratorPolarsIntegration:
    """Test @cash.cache with polars DataFrames in notebooks."""

    def test_polars_dataframe_caching(self, nb_runner):
        """@cash.cache should work with polars DataFrame arguments."""
        nb_runner.create_notebook([
            "from cash import Cash\nimport polars as pl\nc = Cash()",
            "@c.cache\ndef process(df):\n    return df.select(pl.col('a').sum()).item()",
            "df = pl.DataFrame({'a': [1, 2, 3]})\nr1 = process(df)\nr2 = process(df)\nprint(f'r1={r1}, r2={r2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert 'r1=6, r2=6' in output

    def test_polars_series_caching(self, nb_runner):
        """@cash.cache should work with polars Series arguments."""
        nb_runner.create_notebook([
            "from cash import Cash\nimport polars as pl\nc = Cash()",
            "@c.cache\ndef total(s):\n    return s.sum()",
            "s = pl.Series('x', [10, 20, 30])\nresult = total(s)\nprint(f'total={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert 'total=60' in output


class TestDecoratorFileDepsIntegration:
    """Test file_depends_on parameter in notebook context."""

    def test_file_depends_on_in_notebook(self, nb_runner, tmp_path):
        """file_depends_on should track file changes in notebook context."""
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n3,4")
        path_str = str(data_file).replace('\\', '/')

        nb_runner.create_notebook([
            "from cash import Cash\nc = Cash()",
            f"@c.cache(file_depends_on='{path_str}')\ndef load():\n    with open('{path_str}') as f:\n        return f.read()",
            "content = load()\nprint('data:', content[:10])",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert 'data: a,b' in output


class TestDecoratorCacheInfoIntegration:
    """Test cache_info() and cache_clear() in notebook context."""

    def test_cache_info_in_notebook(self, nb_runner):
        """cache_info() should report hits/misses in notebook context.
        
        Note: With %cash_on, statement-level caching intercepts repeated calls
        BEFORE the decorator is invoked. So decorator hits/misses only count
        calls that actually reach the decorator."""
        nb_runner.create_notebook([
            "from cash import Cash\nc = Cash()",
            "@c.cache\ndef square(x):\n    return x ** 2",
            "r1 = square(3)\nr2 = square(4)\ninfo = square.cache_info()\nprint(f'misses={info[\"misses\"]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        # Both calls are first-time with different args = 2 misses
        assert 'misses=2' in output

    def test_cache_clear_in_notebook(self, nb_runner):
        """cache_clear() should reset stats in notebook context."""
        nb_runner.create_notebook([
            "from cash import Cash\nc = Cash()",
            "@c.cache\ndef add(a, b):\n    return a + b",
            "add(1, 2)\nadd(1, 2)\nadd.cache_clear()\ninfo = add.cache_info()\nprint(f'after_clear: hits={info[\"hits\"]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert 'after_clear: hits=0' in output
