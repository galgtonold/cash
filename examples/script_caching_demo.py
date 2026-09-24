#!/usr/bin/env python
"""Caching functions in a plain Python script with ``@cash.cache``.

Run it twice:

    python examples/script_caching_demo.py

The first run computes everything. On the second run, sections 1 and 2 come
entirely from the cache on disk.

Each section prints the function's ``cache_info()``: "ran" counts calls that
executed the function body, "from cache" counts calls answered from the cache.
"""

import tempfile
import time
from pathlib import Path

import cash

# A cache folder of its own, so the demo leaves your project's cache alone.
DEMO_DIR = Path(tempfile.gettempdir()) / "cash_script_demo"
cash.configure(cache_dir=str(DEMO_DIR / "cache"))


def show(fn):
    info = fn.cache_info()
    print(f"  {fn.__name__}: {info['misses']} ran, {info['hits']} from cache\n")


# 1. The same call twice -------------------------------------------------------
@cash.cache
def fibonacci(n):
    time.sleep(0.5)  # stands in for real work
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


print("1. The same call twice")
print(f"  fibonacci(30) = {fibonacci(30)}")
print(f"  fibonacci(30) = {fibonacci(30)}")
show(fibonacci)


# 2. Different arguments, different entries ------------------------------------
@cash.cache
def scale(values, factor=1, offset=0):
    time.sleep(0.2)  # stands in for real work
    return [v * factor + offset for v in values]


print("2. Different arguments, different entries")
print(f"  {scale([1, 2, 3], factor=2)}")
print(f"  {scale([1, 2, 3], factor=2)}  (same arguments)")
print(f"  {scale([1, 2, 3], factor=3)}  (new arguments)")
show(scale)


# 3. A file the function reads -------------------------------------------------
data_file = DEMO_DIR / "numbers.txt"
data_file.parent.mkdir(parents=True, exist_ok=True)
data_file.write_text("10\n20\n30\n", encoding="utf-8")


@cash.cache
def total():
    # cash sees this read: when the file's content changes, the call runs again.
    with open(data_file, encoding="utf-8") as f:
        return sum(int(line) for line in f if line.strip())


print("3. A file the function reads")
print(f"  total() = {total()}")
print(f"  total() = {total()}  (file unchanged)")
data_file.write_text("100\n200\n300\n", encoding="utf-8")
print(f"  total() = {total()}  (file changed)")
show(total)


# 4. A result that expires -----------------------------------------------------
@cash.cache(ttl=2)
def exchange_rate(currency):
    time.sleep(0.2)  # stands in for a slow lookup
    return {"EUR": 1.08, "GBP": 1.27}[currency]


print("4. A result that expires (ttl=2)")
print(f"  exchange_rate('EUR') = {exchange_rate('EUR')}")
print(f"  exchange_rate('EUR') = {exchange_rate('EUR')}")
print("  waiting 3 seconds...")
time.sleep(3)
print(f"  exchange_rate('EUR') = {exchange_rate('EUR')}  (expired, so it ran again)")
show(exchange_rate)

print(f"Run the script again: sections 1 and 2 now come from the cache in {DEMO_DIR / 'cache'}.")
