"""
Performance Benchmarks for Cash

Run with: python -m pytest benchmarks/bench_core.py -v

Measures key performance characteristics:
- Cache key computation speed
- Lineage hashing speed
- AST parsing speed
- Backend get/set speed
- Statement processing overhead
"""

import time
import hashlib
import pickle
import ast
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def benchmark(func, iterations=1000, label=None):
    """Run a benchmark and print results."""
    # Warmup
    for _ in range(min(100, iterations)):
        func()

    # Measure
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    elapsed = time.perf_counter() - start

    per_call_us = (elapsed / iterations) * 1_000_000
    name = label or func.__name__
    print(f"  {name}: {per_call_us:.1f} µs/call ({iterations} iterations, {elapsed:.3f}s total)")
    return per_call_us


def bench_hash_computation():
    """Benchmark SHA256 hash computation for cache keys."""
    print("\n=== Hash Computation ===")

    # Small string (variable name)
    small = "x = 42"
    benchmark(lambda: hashlib.sha256(small.encode()).hexdigest(), label="small string (6 chars)")

    # Medium string (typical statement)
    medium = "result = pd.read_csv('data.csv').groupby('category').agg({'value': 'mean'})"
    benchmark(lambda: hashlib.sha256(medium.encode()).hexdigest(), label="medium string (75 chars)")

    # Large string (multi-line code block)
    large = "\n".join([f"x_{i} = compute(data_{i}, params_{i})" for i in range(50)])
    benchmark(lambda: hashlib.sha256(large.encode()).hexdigest(), label=f"large string ({len(large)} chars)")


def bench_pickle_hashing():
    """Benchmark pickle-based object hashing."""
    print("\n=== Pickle-based Object Hashing ===")

    # Integer
    benchmark(lambda: hashlib.sha256(pickle.dumps(42)).hexdigest(), label="integer")

    # String
    benchmark(lambda: hashlib.sha256(pickle.dumps("hello world")).hexdigest(), label="string")

    # List of 100 items
    lst = list(range(100))
    benchmark(lambda: hashlib.sha256(pickle.dumps(lst)).hexdigest(), label="list (100 ints)")

    # Dict with 50 keys
    d = {f"key_{i}": i * 3.14 for i in range(50)}
    benchmark(lambda: hashlib.sha256(pickle.dumps(d)).hexdigest(), label="dict (50 keys)")

    # Nested structure
    nested = {"data": [{"x": i, "y": i * 2} for i in range(20)], "meta": {"version": 1}}
    benchmark(lambda: hashlib.sha256(pickle.dumps(nested)).hexdigest(), label="nested dict/list")


def bench_ast_parsing():
    """Benchmark AST parsing performance."""
    print("\n=== AST Parsing ===")

    # Simple statement
    simple = "x = 42"
    benchmark(lambda: ast.parse(simple), label="simple assignment")

    # Function call
    call = "result = compute(x, y, z=10)"
    benchmark(lambda: ast.parse(call), label="function call")

    # Multi-line
    multi = """
for i in range(10):
    result = compute(data[i])
    output.append(result)
"""
    benchmark(lambda: ast.parse(multi), label="multi-line (4 lines)")

    # Complex
    complex_code = """
import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')
df['new_col'] = df['a'] * df['b'] + np.sqrt(df['c'])
grouped = df.groupby('category').agg({'new_col': ['mean', 'std', 'count']})
result = grouped.reset_index()
result.to_csv('output.csv', index=False)
"""
    benchmark(lambda: ast.parse(complex_code), label="complex (8 lines)", iterations=500)


def bench_backend_operations():
    """Benchmark backend get/set operations."""
    print("\n=== Backend Operations ===")

    from cash.backends.backend import InMemoryBackend, FileBackend
    import tempfile

    # InMemoryBackend
    mem = InMemoryBackend()
    benchmark(lambda: mem.set("key_bench", 42, {"code": "x=42"}), label="InMemory set")
    mem.set("key_bench", 42, {"code": "x=42"})
    benchmark(lambda: mem.get("key_bench"), label="InMemory get (hit)")
    benchmark(lambda: mem.get("nonexistent_key"), label="InMemory get (miss)")

    # FileBackend
    with tempfile.TemporaryDirectory() as tmpdir:
        fb = FileBackend(tmpdir)
        data = list(range(100))
        benchmark(lambda: fb.set("key_bench", data, {"code": "x=range(100)"}), label="File set (100 ints)", iterations=200)
        fb.set("key_bench", data, {"code": "x=range(100)"})
        benchmark(lambda: fb.get("key_bench"), label="File get (hit)", iterations=200)
        benchmark(lambda: fb.get("nonexistent_key"), label="File get (miss)", iterations=200)


def bench_code_analysis():
    """Benchmark code analysis (input/output detection)."""
    print("\n=== Code Analysis ===")

    from cash.notebook.analysis import CodeAnalyzer

    # Simple
    benchmark(lambda: CodeAnalyzer.analyze_code_block("x = 42"), label="simple assignment")

    # With inputs
    benchmark(lambda: CodeAnalyzer.analyze_code_block("result = func(a, b, c)"), label="function call")

    # Complex
    code = "df = pd.read_csv(file_path)\nresult = df.groupby(col).agg(agg_func)"
    benchmark(lambda: CodeAnalyzer.analyze_code_block(code), label="multi-statement", iterations=500)


def bench_lineage_key_computation():
    """Benchmark the full cache key computation pipeline."""
    print("\n=== Cache Key Pipeline ===")

    code = "result = compute(data, params)"
    input_lineages = {
        "compute": "abc123def456",
        "data": "789ghi012jkl",
        "params": "345mno678pqr"
    }

    def compute_cache_key():
        sorted_lineages = ':'.join(
            f"{k}={v}" for k, v in sorted(input_lineages.items())
        )
        key_material = f"{code}|{sorted_lineages}"
        return hashlib.sha256(key_material.encode()).hexdigest()

    benchmark(compute_cache_key, label="full cache key computation")


def bench_purity_check():
    """Benchmark purity checking overhead."""
    print("\n=== Purity Check ===")

    from cash.notebook.purity import is_pure, is_stateful, is_known_pure, pure, stateful

    @pure
    def my_pure_func(x):
        return x * 2

    @stateful
    def my_stateful_func(x):
        return x

    def unmarked(x):
        return x

    benchmark(lambda: is_pure(my_pure_func), label="is_pure (marked)")
    benchmark(lambda: is_stateful(my_stateful_func), label="is_stateful (marked)")
    benchmark(lambda: is_pure(unmarked), label="is_pure (unmarked)")
    benchmark(lambda: is_known_pure("sorted"), label="is_known_pure (hit)")
    benchmark(lambda: is_known_pure("my_func"), label="is_known_pure (miss)")

    # Full purity check (parse + check)
    code = "result = sorted(data)"
    def full_purity_check():
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                is_known_pure(node.func.id)

    benchmark(full_purity_check, label="full purity check (parse + walk + check)")


def main():
    print("Cash Performance Benchmarks")
    print("=" * 60)
    print(f"Python {sys.version}")
    print(f"Platform: {sys.platform}")

    bench_hash_computation()
    bench_pickle_hashing()
    bench_ast_parsing()
    bench_code_analysis()
    bench_lineage_key_computation()
    bench_purity_check()
    bench_backend_operations()

    print("\n" + "=" * 60)
    print("Benchmark complete.")
    print("\nTarget overhead per statement: < 5ms")
    print("Target 100-cell simulation: < 500ms")


if __name__ == "__main__":
    main()
