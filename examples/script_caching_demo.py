#!/usr/bin/env python
"""
Script Caching Demo - Using @cash.cache in Python scripts

Cash isn't just for notebooks! The @cash.cache decorator lets you cache
expensive function results in regular Python scripts too.

Usage:
    python examples/script_caching_demo.py

What this demonstrates:
    1. Basic function caching with @cash.cache
    2. TTL (time-to-live) for cache expiration
    3. File dependency tracking (re-compute when data files change)
    4. Cache persistence across script runs
    5. Dependency chain invalidation
"""

import time
import os
import sys
import tempfile

# Ensure cash is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cash import Cash, FileDataSource

# Create a Cash instance with file-based persistence
# This means cached results survive across script runs!
cache_dir = os.path.join(tempfile.gettempdir(), "cash_script_demo")
cash = Cash(cache_dir=cache_dir, register_magic=False)


# =============================================================================
# 1. Basic Function Caching
# =============================================================================

@cash.cache
def expensive_computation(n):
    """Simulate an expensive computation."""
    print(f"  [Computing fibonacci({n})...]")
    time.sleep(0.5)  # Simulate work
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


print("=" * 60)
print("1. Basic Function Caching")
print("=" * 60)

# First call: actually computes
start = time.time()
result1 = expensive_computation(30)
time1 = time.time() - start
print(f"  Result: {result1}, Time: {time1:.3f}s")

# Second call: cached!
start = time.time()
result2 = expensive_computation(30)
time2 = time.time() - start
print(f"  Result: {result2}, Time: {time2:.3f}s (cached!)")
print(f"  Speedup: {time1/max(time2, 0.001):.0f}x\n")


# =============================================================================
# 2. TTL (Time-to-Live)
# =============================================================================

@cash.cache(ttl=2)  # Cache expires after 2 seconds
def get_timestamp():
    """Returns current timestamp. Cached for 2 seconds."""
    return time.time()


print("=" * 60)
print("2. TTL (Time-to-Live)")
print("=" * 60)

ts1 = get_timestamp()
print(f"  First call:  {ts1:.3f}")

ts2 = get_timestamp()
print(f"  Second call: {ts2:.3f} (same - cached)")
assert ts1 == ts2

print("  Waiting 3 seconds for cache to expire...")
time.sleep(3)

ts3 = get_timestamp()
print(f"  Third call:  {ts3:.3f} (different - expired)")
assert ts3 > ts1
print()


# =============================================================================
# 3. File Dependency Tracking
# =============================================================================

# Create a temporary data file
data_file = os.path.join(tempfile.gettempdir(), "cash_demo_data.txt")
with open(data_file, 'w') as f:
    f.write("10\n20\n30\n")

data_source = FileDataSource(data_file)


@cash.cache(depends_on=[data_source])
def process_data():
    """Process data from file. Re-computes when file changes."""
    print("  [Reading and processing data file...]")
    with open(data_file) as f:
        numbers = [int(line.strip()) for line in f if line.strip()]
    return sum(numbers)


print("=" * 60)
print("3. File Dependency Tracking")
print("=" * 60)

result1 = process_data()
print(f"  Sum: {result1}")

result2 = process_data()
print(f"  Sum: {result2} (cached - file unchanged)")

# Modify the file
with open(data_file, 'w') as f:
    f.write("100\n200\n300\n")

# Update the data source state
data_source.update_state()

result3 = process_data()
print(f"  Sum: {result3} (re-computed - file changed!)")
print()


# =============================================================================
# 4. Multiple Arguments & Keyword Arguments
# =============================================================================

@cash.cache
def transform(data, multiplier=1, offset=0):
    """Transform data with caching. Different args = different cache entries."""
    print(f"  [Computing transform({data}, mult={multiplier}, off={offset})...]")
    time.sleep(0.2)
    return [x * multiplier + offset for x in data]


print("=" * 60)
print("4. Multiple Arguments & Keyword Arguments")
print("=" * 60)

r1 = transform([1, 2, 3], multiplier=2)
print(f"  transform([1,2,3], mult=2) = {r1}")

r2 = transform([1, 2, 3], multiplier=2)  # Same args = cache hit
print(f"  transform([1,2,3], mult=2) = {r2} (cached)")

r3 = transform([1, 2, 3], multiplier=3)  # Different args = cache miss
print(f"  transform([1,2,3], mult=3) = {r3} (different args)")
print()


# =============================================================================
# 5. Cache Statistics
# =============================================================================

print("=" * 60)
print("5. Cache Info")
print("=" * 60)
print(f"  Cache directory: {cache_dir}")
print(f"  Backend type: {type(cash.backend).__name__}")
print()

# Cleanup
print("Cleaning up temporary files...")
os.remove(data_file)
cash.shutdown()
print("Done!")
