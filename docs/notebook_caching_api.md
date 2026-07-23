# Notebook Caching API Reference

This document provides detailed API reference for the notebook caching components.

---

## Cell badges

Every cell shows a Cash badge above its output. The badge is the canonical UI for "what did cash do, and why?" — for example, a typical cell where the upstream input restored from cache and the current computation ran fresh:

<iframe class="cash-badge" src="/_badges/status_mixed.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

See [Reading the Cash badge](badges.md) for the full anatomy, status reference, and a walkthrough of the most common cache-miss and not-cached situations.

---

## Magic Commands

### `%cash_on`

Enable automatic caching for all subsequent cells.

```python
%cash_on           # Enable caching
%cash_on ttl=3600  # Enable with 1-hour TTL for all statements
```

**Options:**
- `ttl=<seconds>`: Set global time-to-live for cached results

---

### `%cash_off`

Disable automatic caching.

```python
%cash_off
```

---

### `%%cash`

Cache a specific cell (when auto-caching is off).

<!-- test:skip reason="illustrative — references undefined pd and missing data.csv" -->
```python
%%cash
df = pd.read_csv('data.csv')
df = df.sort_values('date')
```

**Cell-level options:**
```python
%%cash ttl=7200
# This cell's results expire after 2 hours
```

---

### Top-level `await` cells

Cells using Jupyter's top-level `await` are cached like any other cell — no
opt-in, no separate magic. ipykernel dispatches them through
`shell.run_cell_async` rather than the `pre_run_cell` hook, so cash intercepts
that entry point too and routes the cell into the same pipeline
(`CellExecutor.execute_cell_async` → `StatementProcessor.process_statement_async`).

<!-- test:skip reason="illustrative — top-level await requires a live IPython kernel" -->
```python
%%cash
data = await fetch_from_api(url)   # cached: lineage, reset, and the result
```

The async path is the line-for-line twin of the sync one, so awaited cells get
lineage tracking, upstream reset, **and** result caching. On a cache hit the
restore returns before the coroutine is ever built, so an unchanged re-run
skips the `await` entirely rather than re-issuing the request.

---

### `%cash_debug`

Toggle debug output.

```python
%cash_debug on   # Enable detailed logging
%cash_debug off  # Disable logging
```

---

## Statement-Level Annotations

Fine-tune caching behavior for individual statements using comments.
All annotation directives use the `@cash:` prefix.

### `# @cash:no-cache`

Skip caching for a statement. Also accepts `@cash:nocache`.

```python
# @cash:no-cache
result = api.fetch_data()  # Always executed, never cached
```

---

### `# @cash:ttl=<seconds>`

Set TTL for a specific statement.

```python
# @cash:ttl=300
prices = get_stock_prices()  # Cached for 5 minutes
```

---

### `# @cash:persist`

Force persistence to disk (override smart policy).

```python
# @cash:persist
model = train_model(data)  # Always saved to disk
```

---

### `# @cash:allow-random`

Suppress randomness warnings for a statement. Also accepts `@cash:allowrandom`.

```python
# @cash:allow-random
result = np.random.randn(100)  # No warning about unseeded randomness
```

---

## Consumable inputs and isolated re-runs

Some objects are **drained in place** by reading them: a generator, a
`queue.Queue`, an open file handle. Re-running *only* the cell that drains one
would otherwise read the leftovers of its own previous run — a drained queue
gives `got=[]`, an exhausted generator totals `0` — where `run_all` re-runs the
producer first and gives the real answer.

Cash detects this and re-executes the producer, so an isolated re-run matches
`run_all`:

<!-- test:skip reason="illustrative — spans multiple notebook cells with a live kernel" -->
```python
# Cell 1
q = Queue()
for i in range(3):
    q.put(i)

# Cell 2 — re-running this alone still prints got=[0, 1, 2]
got = []
while not q.empty():
    got.append(q.get())
print(f"got={got}")
```

The check compares the object's drain position (generator state, queue
`qsize()`, file offset) against a baseline recorded at the cell's *entry*, so it
self-disables on `run_all` — where the producer has already handed the cell the
same state — and on a cell's first run, which has no baseline. It is also scoped
to inputs the cell actually consumes: a reporting read like `n = q.qsize()`
leaves the producer alone.

Deep-copyable iterators (`map`, `zip`, `enumerate`, `io.StringIO`, `iter(range(6))`)
are snapshotted fresh by the cache and restore correctly, so they are **not**
flagged and their producers are never re-run. Opaque `itertools` cursors
(`cycle`, `chain`, `tee`) expose no observable position and are deliberately
left alone.

---

## Output replay and display suppression

A cache hit replays the statement's captured `stdout`, `stderr`, and rich
outputs, so a restored cell looks the same as one that just ran.

A trailing `;` still suppresses output on a **cached** re-run. `ast.unparse`
drops the semicolon, so cash recovers it from the raw cell source and re-attaches
it to the statement code — the suppression therefore rides through both the cache
key and the execution path, and nothing is displayed *or* captured:

<!-- test:skip reason="illustrative — display suppression requires a live IPython kernel" -->
```python
df.head();   # no repr on the first run, and none on a cached re-run either
```

Because the `;` is part of the statement code, `df.head()` and `df.head();` are
distinct cache entries.

---

## Core Classes

### CashMagics

Main entry point for notebook integration.

```python
class CashMagics(Magics):
    """IPython magic commands for transparent caching."""
    
    def __init__(self, shell, cash_instance: Cash):
        """
        Initialize CashMagics.
        
        Args:
            shell: IPython InteractiveShell instance
            cash_instance: Cash backend for storage
        """
```

**Key Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `_auto_cache_enabled` | `bool` | Whether auto-caching is active |
| `_global_ttl` | `int` | Default TTL for all statements |
| `_debug` | `bool` | Debug mode flag |
| `_variable_lineage` | `Dict[str, str]` | Variable name → lineage hash |
| `_executed_cell_codes` | `Dict[str, str]` | Variable name → defining code |

---

### CodeAnalyzer

Static code analysis utilities.

```python
class CodeAnalyzer:
    """Analyzes Python code to determine dependencies."""
    
    @staticmethod
    def analyze_code_block(code: str) -> tuple[Set[str], Set[str]]:
        """
        Analyze code to find input and output variables.
        
        Args:
            code: Python source code
            
        Returns:
            Tuple of (input_vars, output_vars)
            
        Raises:
            SyntaxError: If code cannot be parsed
        """
    
    @staticmethod
    def strip_magics(code: str) -> str:
        """
        Remove Jupyter magic commands from code.
        
        Args:
            code: Code possibly containing magics
            
        Returns:
            Code with magics removed
        """
```

**Example:**

<!-- test:skip reason="CodeAnalyzer is an internal class with no public analyze_code_block method" -->
```python
inputs, outputs = CodeAnalyzer.analyze_code_block("""
df = pd.read_csv('data.csv')
result = df.groupby('x').sum()
""")

print(inputs)   # {'pd'}
print(outputs)  # {'df', 'result'}
```

---

### StatementProcessor

Handles individual statement execution and caching.

```python
class StatementProcessor:
    """Processes and caches individual Python statements."""
    
    def __init__(
        self,
        shell,
        cash_instance: Cash,
        debug: bool = False,
        compute_hash_fn=None,
        calculate_memory_fn=None
    ):
        """
        Initialize StatementProcessor.
        
        Args:
            shell: IPython shell instance
            cash_instance: Cache backend
            debug: Enable debug output
            compute_hash_fn: Function to hash variable values
            calculate_memory_fn: Function to estimate memory usage
        """
    
    def process_statement(
        self,
        code: str,
        ttl: Optional[int] = None,
        silent: bool = False,
        render_badge: bool = True,
        annotation: Optional[CacheAnnotation] = None,
        iteration_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a single statement with caching.
        
        Args:
            code: Python statement to execute
            ttl: Time-to-live in seconds
            silent: Suppress output
            render_badge: Show execution badge
            annotation: Cache annotation from comments
            iteration_context: Loop iteration context
            
        Returns:
            Dict with execution metrics:
            {
                'status': 'COMPUTED' | 'RESTORED' | 'SKIPPED' | 'ERROR',
                'execution_time': float,
                'total_time': float,
                'saved_time': float,
                'restored_vars': List[str],
                'code': str,
                'error': Optional[str],
                'uncacheable_reasons': List[str]
            }
        """
```

---

### UpstreamChecker

Manages upstream dependency detection and re-execution.

```python
class UpstreamChecker:
    """Detects and handles changed upstream dependencies."""
    
    def __init__(
        self,
        shell,
        cash_instance=None,
        debug: bool = False,
        compute_hash_fn=None
    ):
        """
        Initialize UpstreamChecker.
        
        Args:
            shell: IPython shell instance
            cash_instance: Cache backend for lookups
            debug: Enable debug output
            compute_hash_fn: Function to hash values
        """
    
    def check_and_reexecute(
        self,
        cell_code: str,
        required_inputs: Set[str],
        process_statement_callback,
        global_ttl: Optional[int] = None,
        cell_id: Optional[str] = None
    ) -> UpstreamResult:
        """
        Check upstream dependencies and re-execute if needed.
        
        Args:
            cell_code: Current cell's code
            required_inputs: Variables needed by current cell
            process_statement_callback: Function to execute statements
            global_ttl: Default TTL
            cell_id: Jupyter cell ID (for disambiguation)
            
        Returns:
            Tuple of:
            - List of execution metrics for re-executed statements
            - Total restore time
            - Total execution time
        """
```

---

### ControlStructureProcessor

Handles loops and conditionals with per-iteration caching.

```python
class ControlStructureProcessor:
    """Processes control structures with fine-grained caching."""
    
    def __init__(
        self,
        shell,
        statement_processor,
        debug: bool = False
    ):
        """
        Initialize ControlStructureProcessor.
        
        Args:
            shell: IPython shell instance
            statement_processor: StatementProcessor for body statements
            debug: Enable debug output
        """
    
    def process(
        self,
        node: ast.AST,
        ttl: Optional[int] = None,
        silent: bool = False,
        parent_context: Optional[Dict[str, Any]] = None
    ) -> ControlStructureResult:
        """
        Process a control structure node.
        
        Args:
            node: AST node (For, While, If, With, Try)
            ttl: Time-to-live
            silent: Suppress output
            parent_context: Context from parent structure (for nesting)
            
        Returns:
            ControlStructureResult with:
            - success: bool
            - metrics: List of statement metrics
            - total_iterations: int
            - cached_iterations: int
            - computed_iterations: int
        """
```

---

### ControlStructureResult

Result from processing a control structure.

<!-- test:skip reason="@dataclass requires the class's __module__ to be in sys.modules" -->
```python
@dataclass
class ControlStructureResult:
    """Result from executing a control structure."""
    success: bool
    metrics: List[Dict[str, Any]]
    error: Optional[Exception] = None
    total_iterations: int = 0
    cached_iterations: int = 0
    computed_iterations: int = 0
```

---

## Cache Backend Interface

The caching system uses a pluggable backend architecture.

### CacheBackend (Abstract)

```python
class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Tuple[Optional[Dict], Optional[Any]]:
        """
        Retrieve item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Tuple of (metadata, data) or (None, None) if not found
        """
    
    @abstractmethod
    def set(
        self,
        key: str,
        data: Any,
        metadata: Dict = None,
        ttl: int = None
    ) -> bool:
        """
        Store item in cache.
        
        Args:
            key: Cache key
            data: Data to cache
            metadata: Optional metadata dict
            ttl: Time-to-live in seconds
            
        Returns:
            True if stored successfully
        """
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete item from cache."""
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all items from cache."""
```

### Built-in Backends

| Backend | Description | Persistence |
|---------|-------------|-------------|
| `InMemoryBackend` | Fast dictionary-based storage | No |
| `FileBackend` | Disk-based with optional compression | Yes |
| `TieredBackend` | Combines multiple backends (L1+L2) | Configurable |
| `CascadingBackend` | Chain of backends with fallthrough | Configurable |

---

## Configuration

### Cash Initialization

```python
from cash import Cash

# Default configuration (auto-tiered)
cash = Cash()

# Custom configuration — any CashConfig field name works as a kwarg.
cash = Cash(
    cache_dir=".my_cache",     # Custom cache directory
    compress=True,              # Enable compression
    debug=True,                 # Enable debug output
    use_locking=True,           # Thread-safe operations
    # Any other CashConfig field: backend, redis_host, redis_port,
    # s3_bucket, cache_dir, persist_all, ...
)
```

Background writes are on by default for every backend except RAM —
serialisation happens on the calling thread, the actual storage write
runs in a per-backend background worker, so `set()` returns once the
bytes are captured. There's no opt-in flag.

### Environment variables

Every `CashConfig` field has a `CASH_*` env-var binding; see the
[Configuration reference](getting-started/configuration.md#all-cashconfig-fields)
for the complete table.

---

## Execution Metrics

Each statement returns metrics in this format:

<!-- test:skip reason="illustrative dict literal at top level (not a runnable script)" -->
```python
{
    'status': str,           # 'COMPUTED', 'RESTORED', 'SKIPPED', 'ERROR', or 'UNKNOWN'
    'execution_time': float, # Actual execution time (seconds)
    'total_time': float,     # Total processing time (seconds)
    'saved_time': float,     # Time saved by caching (seconds)
    'restored_vars': List[str],  # Variables restored from cache
    'code': str,             # The statement code (trimmed)
    'error': Optional[str],  # Error message if status='ERROR'
    'uncacheable_reasons': List[str],  # Reasons caching was skipped
    'decorator_calls': List[dict],    # @cash.cache call events (if any)
}
```

### Decorator Call Metrics

When a statement invokes `@cash.cache` decorated functions, the `decorator_calls` list contains entries like:

<!-- test:skip reason="illustrative dict literal at top level (not a runnable script)" -->
```python
{
    'func_name': 'my_module.process',  # Module-qualified function name
    'cache_hit': True,                  # Whether the decorator cache was hit
    'execution_time': 0.001,            # Time for this decorator call
    'args_hash': 'abc123...',           # Hash of the arguments
    'cache_key': 'my_module.process:...', # Full decorator cache key
    'timestamp': 1718000000.0,          # When the call occurred
}
```

These metrics are displayed in the notebook badge alongside statement-level metrics.

---

## Error Handling

### Common Exceptions

| Exception | When Raised | Recovery |
|-----------|-------------|----------|
| `SyntaxError` | Invalid Python in cell | Fix syntax |
| `RuntimeError` | Ambiguous cell content | Save notebook |
| `PicklingError` | Unpicklable result | Use `@cash:no-cache` |
| `ValueError` | Invalid annotation | Fix comment syntax |

### Graceful Degradation

When caching fails, the system falls back to normal execution:

```python
try:
    result = process_with_caching(code)
except Exception as e:
    logger.warning(f"Caching failed: {e}. Falling back to normal execution.")
    exec(code, namespace)
```

---

## Best Practices

### Do's

✅ Use for expensive computations (>1s execution time)
✅ Enable `%cash_debug on` when troubleshooting
✅ Use TTL for data that changes frequently
✅ Use `@cash:no-cache` for side effects (API calls, file writes)
✅ Use `@cash:allow-random` to suppress warnings for intentionally random code

### Don'ts

❌ Don't rely on cached values from unseeded random code being reproducible — seed with `random.seed()` first, or acknowledge it with `@cash:allow-random` (which silences the warning; it does *not* stop the caching — use `@cash:no-cache` for that)
❌ Don't cache code that reads from stdin
❌ Don't expect caching of closures over mutable state
❌ Don't use identical cell content in multiple cells (causes ambiguity)

---

## See also

- [Purity decorators](tutorials/feature-guides/purity-decorators.md) —
  `@pure` / `@stateful` and `mark_pure` / `mark_stateful` for declaring
  function-level purity so the notebook tracker treats helpers correctly.
- [Controlling cache behavior](tutorials/feature-guides/controlling-cache-behavior.md) —
  end-to-end tour of the annotation directives (`# @cash:persist`,
  `no-cache`, `ttl=`) and how they interact with the cost model.
