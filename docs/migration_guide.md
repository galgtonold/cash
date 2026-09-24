# Coming from other caches

!!! info "Applies to: both paths"
    If you already cache with another tool: the same code before and after.
    How the tools compare is on [Why cash?](why-cash.md#compared-with-other-tools).

=== "Decorator"

    **`functools.lru_cache`**

    <!-- test:skip reason="illustrative: before/after code; the other tools and data are not set up" -->
    ```python
    from functools import lru_cache

    @lru_cache(maxsize=128)
    def load_prices(ticker):
        ...
    ```

    <!-- test:skip reason="illustrative: before/after code; the other tools and data are not set up" -->
    ```python
    import cash

    @cash.cache                  # stored on disk; arguments need not be hashable
    def load_prices(ticker):
        ...
    ```

    `load_prices.cache_info()` and `load_prices.cache_clear()` work as before.

    **`joblib.Memory`**

    <!-- test:skip reason="illustrative: before/after code; the other tools and data are not set up" -->
    ```python
    from joblib import Memory

    memory = Memory("/tmp/joblib-cache", verbose=0)

    @memory.cache
    def transform(data):
        return data.apply(complex_transform)
    ```

    <!-- test:skip reason="illustrative: before/after code; the other tools and data are not set up" -->
    ```python
    import cash

    @cash.cache                  # cache folder: .cash/ in your project, or CASH_CACHE_DIR
    def transform(data):
        return data.apply(complex_transform)
    ```

    **`diskcache`**

    <!-- test:skip reason="illustrative: before/after code; the other tools and data are not set up" -->
    ```python
    from diskcache import Cache

    cache = Cache("/tmp/diskcache")

    @cache.memoize()
    def transform(data):
        return data.apply(complex_transform)
    ```

    <!-- test:skip reason="illustrative: before/after code; the other tools and data are not set up" -->
    ```python
    import cash

    @cash.cache
    def transform(data):
        return data.apply(complex_transform)
    ```

    **Hand-written pickle files**

    <!-- test:skip reason="illustrative: before/after code; the other tools and data are not set up" -->
    ```python
    import os
    import pickle

    if os.path.exists("result.pkl"):
        with open("result.pkl", "rb") as f:
            result = pickle.load(f)
    else:
        result = expensive_computation()
        with open("result.pkl", "wb") as f:
            pickle.dump(result, f)
    ```

    <!-- test:skip reason="illustrative: before/after code; the other tools and data are not set up" -->
    ```python
    import cash

    @cash.cache
    def expensive_computation():
        ...

    result = expensive_computation()
    ```

=== "Notebook"

    **`%store`**

    <!-- test:skip reason="illustrative: before/after code; the other tools and data are not set up" -->
    ```python { .nb-cell }
    %store -r df
    if "df" not in dir():
        df = pd.read_csv("large_file.csv")
        %store df
    ```

    <!-- test:skip reason="illustrative: before/after code; the other tools and data are not set up" -->
    ```python { .nb-cell }
    # first cell
    import cash
    %cash_on
    ```

    <!-- test:skip reason="illustrative: before/after code; the other tools and data are not set up" -->
    ```python { .nb-cell }
    # any cell below it
    df = pd.read_csv("large_file.csv")
    ```

    **Hand-written pickle files**

    <!-- test:skip reason="illustrative: before/after code; the other tools and data are not set up" -->
    ```python { .nb-cell }
    import os
    import pickle

    if os.path.exists("result.pkl"):
        with open("result.pkl", "rb") as f:
            result = pickle.load(f)
    else:
        result = expensive_computation()
        with open("result.pkl", "wb") as f:
            pickle.dump(result, f)
    ```

    <!-- test:skip reason="illustrative: before/after code; the other tools and data are not set up" -->
    ```python { .nb-cell }
    # below the import cash / %cash_on cell
    result = expensive_computation()
    ```

    **`jupyter-cache`**

    Nothing to replace: `jupyter-cache` skips re-executing unchanged notebooks
    in a book build, and cash caches statements while you work. Use both if
    you need both.
