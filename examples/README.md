# Cash Examples

This directory contains polished examples demonstrating cash's caching features for both **Jupyter notebooks** and **Python scripts**.

## Quick Start

1. Install cash: `pip install -e .` (from project root)
2. Open any `.ipynb` file in Jupyter or VS Code, or run a `.py` script
3. Run the cells/script sequentially

## Example Notebooks

### [`demo_cell_caching.ipynb`](demo_cell_caching.ipynb) - Statement-Level Caching
**What it demonstrates:** The `%%cash` cell magic for explicit caching of individual cells.

**Key features shown:**
- Basic cache hit/miss with `%%cash`
- Dependency tracking (changing `multiplier` invalidates downstream cache)
- TTL (Time To Live) for cache expiration
- Cache explorer widget

**What to try:**
1. Run all cells once — see "COMPUTED" for first run
2. Run again — see "CACHED" (instant results)
3. Change `multiplier = 10` to `multiplier = 20` and re-run dependent cell — auto-invalidation!

---

### [`demo_notebook_caching.ipynb`](demo_notebook_caching.ipynb) - Automatic Notebook Caching
**What it demonstrates:** The `%cash_on` magic for automatic transparent caching of all cells.

**Key features shown:**
- Auto-caching mode (`%cash_on` / `%cash_off`)
- Execution timing comparison (cached vs uncached)
- Cache invalidation when upstream code changes
- `%cash_stats` for session statistics
- Explicit `%%cash` alongside auto mode

**What to try:**
1. Run notebook once — observe execution times
2. Restart kernel and run again — see instant restoration from cache
3. Modify a data processing step — watch downstream invalidation cascade

---

### [`file_caching_demo.ipynb`](file_caching_demo.ipynb) - File Dependency Tracking
**What it demonstrates:** How cash tracks file reads and invalidates caches when files change.

**Key features shown:**
- Pandas `read_csv` tracking — cache invalidates when CSV changes
- Variable lineage across cells
- Downstream cascade — changing raw data invalidates all derived variables

**What to try:**
1. Run all cells — data is loaded and processed
2. Modify `sales_data.csv` externally
3. Re-run the data loading cell — cash detects the file change and re-executes

---

### [`file_tracking_demo.ipynb`](file_tracking_demo.ipynb) - Comprehensive File Tracking
**What it demonstrates:** All supported file tracking methods (open, pandas, numpy, pathlib).

**Key features shown:**
- `open()` file reads tracked automatically
- `pd.read_csv()` tracked
- `Path.read_text()` tracked
- `np.loadtxt()` tracked
- Debug mode (`%cash_debug on`) showing internal cache decisions

**What to try:**
1. Enable debug mode to see file hashing in action
2. Modify a tracked file between runs
3. Watch cash detect the change and invalidate the correct cache entry

---

### [`financial_analysis_demo.ipynb`](financial_analysis_demo.ipynb) - Real-World Data Pipeline
**What it demonstrates:** A realistic financial analysis workflow with multiple caching features.

**Key features shown:**
- Data loading with file tracking
- Multi-step preprocessing pipeline
- Heavy computation caching (volatility metrics, RSI)
- Loop caching (per-ticker iteration caching)
- Conditional caching (branching logic)
- Nested loop caching

**Prerequisites:**
- Run `python generate_financial_data.py` first to create `large_financial_data.csv`

**What to try:**
1. Run the full pipeline — first run computes everything
2. Re-run — cached results load instantly
3. Add a new ticker or change a computation — only affected steps re-execute

---

### [`script_caching_demo.py`](script_caching_demo.py) - Python Script Caching
**What it demonstrates:** Using `@cash.cache` decorator in regular Python scripts (not notebooks).

**Key features shown:**
- `@cash.cache` decorator for function result caching
- TTL (time-to-live) for automatic cache expiration
- `FileDataSource` for file dependency tracking
- Multiple arguments and keyword arguments
- Cache persistence across script runs

**Usage:**
```bash
python examples/script_caching_demo.py
```

**What to try:**
1. Run the script once — see computations happen
2. Run again immediately — see cached results load instantly
3. Modify the temporary data file between runs — see file-dependent cache invalidate

---

## Common Patterns

### Enable auto-caching
```python
%load_ext cash
%cash_on
```

### Check cache status
```python
%cash_stats        # Session statistics
%cash_stats json   # Machine-readable format
```

### Debug cache decisions
```python
%cash_debug on     # Verbose logging
%cash_debug json   # JSON-formatted logs
%cash_debug off    # Disable
```

### Track external modules
```python
%cash_track my_utils        # Track module file changes
%cash_track my_utils --reload  # Force reload
```

### View variable provenance
```python
%cash_provenance df         # Show history of 'df'
%cash_provenance df --graph # Show dependency tree
%cash_provenance --time     # Timeline of all computations
```

### Verify cache integrity
```python
%cash_verify       # Check for corruption
%cash_repair       # Fix corrupted entries
```
