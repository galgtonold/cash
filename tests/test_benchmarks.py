"""
Benchmark suite for cash performance (Phase 3.1).

Measures:
- Cache hit/miss overhead
- AST parsing overhead
- Lineage hash computation
- Serialization speed for common types
- Backend comparison (InMemory vs File)
- Statement processor overhead
- Upstream simulation overhead

Run with: pytest tests/test_benchmarks.py -v
Each test asserts that the measured operation stays within acceptable bounds.
"""

import ast
import hashlib
import pickle
import time
import sys
import statistics

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _timeit(fn, iterations=100, warmup=5):
    """Run *fn* and return (median_ms, p95_ms, all_times_ms)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)  # ms
    times.sort()
    median = statistics.median(times)
    p95 = times[int(len(times) * 0.95)]
    return median, p95, times


# ---------------------------------------------------------------------------
# 1. Lineage hash computation
# ---------------------------------------------------------------------------

class TestLineageHashPerformance:
    """Target: <1ms per hash computation."""

    def test_simple_hash(self):
        """Hash a single statement code string."""
        code = "result = df.groupby('category').agg({'sales': 'sum', 'quantity': 'mean'}).reset_index()"
        def fn():
            hashlib.sha256(code.encode('utf-8')).hexdigest()
        median, p95, _ = _timeit(fn, iterations=1000)
        assert median < 1.0, f"Simple hash too slow: {median:.3f}ms (target <1ms)"

    def test_lineage_hash_with_inputs(self):
        """Hash code + sorted input lineages (typical lineage computation)."""
        code = "result = process(df, config)"
        input_lineages = {
            "df": "a" * 64,
            "config": "b" * 64,
            "process": "c" * 64,
        }
        def fn():
            sorted_inputs = ":".join(
                f"{k}={v}" for k, v in sorted(input_lineages.items())
            )
            lineage_str = f"{code}:{sorted_inputs}"
            hashlib.sha256(lineage_str.encode('utf-8')).hexdigest()
        median, p95, _ = _timeit(fn, iterations=1000)
        assert median < 1.0, f"Lineage hash too slow: {median:.3f}ms (target <1ms)"

    def test_lineage_hash_many_inputs(self):
        """Hash with 20 input variables (worst-case realistic scenario)."""
        code = "result = complex_function(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t)"
        input_lineages = {f"var_{i}": hashlib.sha256(f"val_{i}".encode()).hexdigest() for i in range(20)}
        def fn():
            sorted_inputs = ":".join(
                f"{k}={v}" for k, v in sorted(input_lineages.items())
            )
            lineage_str = f"{code}:{sorted_inputs}"
            hashlib.sha256(lineage_str.encode('utf-8')).hexdigest()
        median, p95, _ = _timeit(fn, iterations=1000)
        assert median < 1.0, f"Many-input lineage hash too slow: {median:.3f}ms (target <1ms)"


# ---------------------------------------------------------------------------
# 2. AST parsing
# ---------------------------------------------------------------------------

class TestASTParsing:
    """Target: <5ms for typical statements."""

    def test_simple_statement(self):
        code = "x = 42"
        def fn():
            ast.parse(code)
        median, p95, _ = _timeit(fn, iterations=500)
        assert median < 5.0, f"Simple AST parse too slow: {median:.3f}ms (target <5ms)"

    def test_complex_statement(self):
        code = """
result = (
    df.groupby(['category', 'region'])
    .agg({
        'sales': ['sum', 'mean', 'std'],
        'quantity': ['count', 'median'],
        'profit': lambda x: x.sum() / len(x)
    })
    .reset_index()
    .sort_values('sales_sum', ascending=False)
    .head(20)
)
"""
        def fn():
            ast.parse(code)
        median, p95, _ = _timeit(fn, iterations=500)
        assert median < 5.0, f"Complex AST parse too slow: {median:.3f}ms (target <5ms)"

    def test_multiline_cell(self):
        """Parse a typical 20-line notebook cell."""
        code = "\n".join([
            "import pandas as pd",
            "import numpy as np",
            "",
            "# Load data",
            "df = pd.read_csv('data.csv')",
            "df['date'] = pd.to_datetime(df['date'])",
            "df = df.dropna(subset=['value'])",
            "",
            "# Process",
            "grouped = df.groupby('category')",
            "stats = grouped.agg({'value': ['mean', 'std', 'count']})",
            "stats.columns = ['_'.join(col) for col in stats.columns]",
            "stats = stats.reset_index()",
            "",
            "# Filter",
            "mask = stats['value_count'] > 10",
            "result = stats[mask].sort_values('value_mean', ascending=False)",
            "",
            "print(f'Found {len(result)} categories')",
            "result.head()",
        ])
        def fn():
            ast.parse(code)
        median, p95, _ = _timeit(fn, iterations=500)
        assert median < 5.0, f"Multi-line cell AST parse too slow: {median:.3f}ms (target <5ms)"

    def test_large_cell_50_lines(self):
        """Parse a large 50-line cell (edge case)."""
        lines = ["import pandas as pd"]
        for i in range(49):
            lines.append(f"x_{i} = x_{max(0, i-1)} + {i}")
        code = "\n".join(lines)
        def fn():
            ast.parse(code)
        median, p95, _ = _timeit(fn, iterations=200)
        assert median < 10.0, f"Large cell AST parse too slow: {median:.3f}ms (target <10ms)"


# ---------------------------------------------------------------------------
# 3. CodeAnalyzer overhead
# ---------------------------------------------------------------------------

class TestCodeAnalyzerPerformance:
    """Test the analyze_code_block function performance."""

    def test_analyze_simple(self):
        from cash.notebook.analysis import CodeAnalyzer
        code = "y = x * 2 + z"
        def fn():
            CodeAnalyzer.analyze_code_block(code)
        median, p95, _ = _timeit(fn, iterations=500)
        assert median < 5.0, f"Simple analysis too slow: {median:.3f}ms (target <5ms)"

    def test_analyze_complex(self):
        from cash.notebook.analysis import CodeAnalyzer
        code = """
result = (
    df.groupby('category')
    .agg({'sales': 'sum', 'quantity': 'mean'})
    .merge(other_df, on='category')
    .assign(ratio=lambda x: x['sales'] / x['quantity'])
)
"""
        def fn():
            CodeAnalyzer.analyze_code_block(code)
        median, p95, _ = _timeit(fn, iterations=500)
        assert median < 5.0, f"Complex analysis too slow: {median:.3f}ms (target <5ms)"

    def test_analyze_with_function_def(self):
        from cash.notebook.analysis import CodeAnalyzer
        code = """
def process_data(df, threshold=0.5):
    filtered = df[df['score'] > threshold]
    grouped = filtered.groupby('category')
    result = grouped.agg({'value': 'mean'})
    return result
"""
        def fn():
            CodeAnalyzer.analyze_code_block(code)
        median, p95, _ = _timeit(fn, iterations=500)
        assert median < 5.0, f"Function def analysis too slow: {median:.3f}ms (target <5ms)"

    def test_strip_magics(self):
        from cash.notebook.analysis import CodeAnalyzer
        code = """%cash_on
%load_ext cash
!pip install pandas
import pandas as pd
%matplotlib inline
df = pd.read_csv('data.csv')
"""
        def fn():
            CodeAnalyzer.strip_magics(code)
        median, p95, _ = _timeit(fn, iterations=1000)
        assert median < 1.0, f"Strip magics too slow: {median:.3f}ms (target <1ms)"


# ---------------------------------------------------------------------------
# 4. Serialization speed
# ---------------------------------------------------------------------------

class TestSerializationPerformance:
    """Test pickle serialization for common types."""

    def test_serialize_dict(self):
        data = {f"key_{i}": list(range(100)) for i in range(50)}
        def fn():
            pickle.dumps(data)
        median, p95, _ = _timeit(fn, iterations=200)
        # Dict of 50 keys × 100 ints ≈ 5000 items, should be fast
        assert median < 10.0, f"Dict serialization too slow: {median:.3f}ms (target <10ms)"

    def test_serialize_list(self):
        data = list(range(10000))
        def fn():
            pickle.dumps(data)
        median, p95, _ = _timeit(fn, iterations=200)
        assert median < 10.0, f"List serialization too slow: {median:.3f}ms (target <10ms)"

    def test_serialize_string(self):
        data = "x" * 100000  # 100KB string
        def fn():
            pickle.dumps(data)
        median, p95, _ = _timeit(fn, iterations=200)
        assert median < 5.0, f"String serialization too slow: {median:.3f}ms (target <5ms)"

    def test_deserialize_dict(self):
        data = {f"key_{i}": list(range(100)) for i in range(50)}
        serialized = pickle.dumps(data)
        def fn():
            pickle.loads(serialized)
        median, p95, _ = _timeit(fn, iterations=200)
        assert median < 10.0, f"Dict deserialization too slow: {median:.3f}ms (target <10ms)"

    @pytest.mark.skipif(
        not any(m in sys.modules for m in ('pandas', 'numpy')),
        reason="pandas/numpy not imported"
    )
    def test_serialize_dataframe(self):
        """Benchmark DataFrame serialization if pandas is available."""
        try:
            import pandas as pd
            import numpy as np
        except ImportError:
            pytest.skip("pandas/numpy not available")
        df = pd.DataFrame(np.random.randn(1000, 10), columns=[f"col_{i}" for i in range(10)])
        def fn():
            pickle.dumps(df)
        median, p95, _ = _timeit(fn, iterations=100)
        assert median < 50.0, f"DataFrame serialization too slow: {median:.3f}ms (target <50ms)"


# ---------------------------------------------------------------------------
# 5. Backend comparison
# ---------------------------------------------------------------------------

class TestBackendPerformance:
    """Compare InMemoryBackend vs FileBackend performance."""

    def test_inmemory_set_get(self):
        from cash.backends.backend import InMemoryBackend
        backend = InMemoryBackend()
        key = "test_key"
        value = {"data": list(range(1000))}
        metadata = {"created": time.time()}

        def fn_set():
            backend.set(key, value, metadata)
        def fn_get():
            backend.get(key)

        set_median, _, _ = _timeit(fn_set, iterations=500)
        get_median, _, _ = _timeit(fn_get, iterations=500)
        assert set_median < 10.0, f"InMemory set too slow: {set_median:.3f}ms (target <10ms)"
        assert get_median < 10.0, f"InMemory get too slow: {get_median:.3f}ms (target <10ms)"

    def test_file_backend_set_get(self, tmp_path):
        from cash.backends.backend import FileBackend
        backend = FileBackend(str(tmp_path / "cache"))
        value = {"data": list(range(1000))}
        metadata = {"created": time.time()}

        # Use different keys to avoid overwrite contention
        keys = [f"test_key_{i}" for i in range(100)]

        def fn_set():
            for k in keys:
                backend.set(k, value, metadata)
        def fn_get():
            for k in keys:
                backend.get(k)

        # File I/O is slower, allow more time per operation
        set_median, _, _ = _timeit(fn_set, iterations=10)
        get_median, _, _ = _timeit(fn_get, iterations=10)
        per_op_set = set_median / len(keys)
        per_op_get = get_median / len(keys)
        assert per_op_set < 20.0, f"File set too slow: {per_op_set:.3f}ms/op (target <20ms)"
        assert per_op_get < 20.0, f"File get too slow: {per_op_get:.3f}ms/op (target <20ms)"

    def test_inmemory_cache_hit_overhead(self):
        """Measure the overhead of a cache hit vs direct variable access.
        
        This is the core user-facing metric: how much slower is
        'get from cache' compared to 'already have the variable'.
        Target: <10ms overhead per cache hit.
        """
        from cash.backends.backend import InMemoryBackend
        backend = InMemoryBackend()
        # Store a moderate-sized value
        value = {f"col_{i}": list(range(500)) for i in range(10)}
        backend.set("stmt:abc123", value, {"created": time.time()})

        def fn():
            backend.get("stmt:abc123")
        median, p95, _ = _timeit(fn, iterations=500)
        assert median < 10.0, f"Cache hit overhead too slow: {median:.3f}ms (target <10ms)"


# ---------------------------------------------------------------------------
# 6. End-to-end statement processing overhead
# ---------------------------------------------------------------------------

class TestStatementProcessorOverhead:
    """Measure the overhead of statement processing components."""

    def test_ast_unparse_roundtrip(self):
        """Measure AST parse + unparse (used in simulation)."""
        code = "result = df.groupby('category').agg({'sales': 'sum'}).reset_index()"
        def fn():
            tree = ast.parse(code)
            for node in tree.body:
                ast.unparse(node)
        median, p95, _ = _timeit(fn, iterations=500)
        assert median < 5.0, f"AST roundtrip too slow: {median:.3f}ms (target <5ms)"

    def test_simulation_per_cell_overhead(self):
        """Simulate processing cost for one upstream cell (parse + analyze + hash).
        
        This measures the per-cell overhead in _simulate_and_find_changes.
        For a 100-cell notebook, total overhead = this * 100.
        Target: <2ms per cell → <200ms for 100 cells.
        """
        from cash.notebook.analysis import CodeAnalyzer
        
        cell_code = """
df = pd.read_csv('data.csv')
df['processed'] = df['raw'].apply(lambda x: x * 2)
result = df.groupby('category').mean()
"""
        virtual_lineage = {f"var_{i}": f"hash_{i}" for i in range(20)}

        def fn():
            clean = CodeAnalyzer.strip_magics(cell_code)
            tree = ast.parse(clean)
            for node in tree.body:
                stmt = ast.unparse(node)
                inputs, outputs = CodeAnalyzer.analyze_code_block(stmt)
                # Simulate lineage update
                input_hashes = {inp: virtual_lineage.get(inp, "") for inp in inputs}
                sorted_inputs = ":".join(f"{k}={v}" for k, v in sorted(input_hashes.items()))
                source_hash = hashlib.sha256(stmt.encode('utf-8')).hexdigest()
                lineage_str = f"{source_hash}:{sorted_inputs}"
                lineage = hashlib.sha256(lineage_str.encode('utf-8')).hexdigest()
                for out in outputs:
                    virtual_lineage[out] = lineage

        median, p95, _ = _timeit(fn, iterations=200)
        assert median < 5.0, f"Per-cell simulation too slow: {median:.3f}ms (target <5ms per cell)"

    def test_simulation_100_cells(self):
        """Simulate 100-cell notebook overhead (realistic worst case).
        
        Target: <500ms for 100 cells total simulation.
        """
        from cash.notebook.analysis import CodeAnalyzer
        
        # Generate 100 cells with varying complexity
        cells = []
        for i in range(100):
            cells.append(f"x_{i} = x_{max(0, i-1)} + {i}\ny_{i} = x_{i} * 2")


        def fn():
            vl = {}
            for cell_code in cells:
                clean = CodeAnalyzer.strip_magics(cell_code)
                try:
                    tree = ast.parse(clean)
                except SyntaxError:
                    continue
                for node in tree.body:
                    stmt = ast.unparse(node)
                    inputs, outputs = CodeAnalyzer.analyze_code_block(stmt)
                    input_hashes = {inp: vl.get(inp, "") for inp in inputs}
                    sorted_inputs = ":".join(f"{k}={v}" for k, v in sorted(input_hashes.items()))
                    source_hash = hashlib.sha256(stmt.encode('utf-8')).hexdigest()
                    lineage_str = f"{source_hash}:{sorted_inputs}"
                    lineage = hashlib.sha256(lineage_str.encode('utf-8')).hexdigest()
                    for out in outputs:
                        vl[out] = lineage

        median, p95, _ = _timeit(fn, iterations=20)
        assert median < 500.0, f"100-cell simulation too slow: {median:.3f}ms (target <500ms)"


# ---------------------------------------------------------------------------
# 7. Memory footprint
# ---------------------------------------------------------------------------

class TestMemoryFootprint:
    """Ensure lineage tracking doesn't use excessive memory."""

    def test_lineage_dict_size(self):
        """1000 variables with 64-char hashes should use reasonable memory."""
        lineage = {}
        for i in range(1000):
            name = f"variable_{i}"
            h = hashlib.sha256(f"code_{i}".encode()).hexdigest()
            lineage[name] = h
        
        size = sys.getsizeof(lineage)
        # Dict overhead + 1000 entries with ~20-char keys + 64-char values
        # Should be well under 1MB
        assert size < 1_000_000, f"Lineage dict too large: {size} bytes for 1000 vars"

    def test_executed_cell_codes_size(self):
        """1000 code entries (avg 200 chars each) should be manageable."""
        codes = {}
        for i in range(1000):
            name = f"variable_{i}"
            codes[name] = f"variable_{i} = some_function(input_{i}) + another_function(arg_{i})" * 3
        
        size = sys.getsizeof(codes)
        # Should be well under 5MB
        assert size < 5_000_000, f"Executed cell codes dict too large: {size} bytes for 1000 vars"
