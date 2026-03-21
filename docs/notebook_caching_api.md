# Notebook Caching API Reference

This document provides detailed API reference for the notebook caching components.

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

# Custom configuration
cash = Cash(
    cache_dir=".my_cache",     # Custom cache directory
    compress=True,              # Enable compression
    debug=True,                 # Enable debug output
    background_io=True,         # Async disk writes
    use_locking=True            # Thread-safe operations
)
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CASH_DEBUG` | Enable debug mode | `false` |
| `CASH_CACHE_DIR` | Default cache directory | `.cash` |
| `CASH_COMPRESS` | Enable compression | `false` |

---

## Execution Metrics

Each statement returns metrics in this format:

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

❌ Don't cache code with random outputs (use `random.seed()` first, or `@cash:allow-random`)
❌ Don't cache code that reads from stdin
❌ Don't expect caching of closures over mutable state
❌ Don't use identical cell content in multiple cells (causes ambiguity)
