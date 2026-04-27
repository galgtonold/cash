"""
Notebook integration test fixtures.

This module provides fixtures for running notebook integration tests with:
1. Real notebook files that cash can read naturally (no mocking)
2. Selective cell execution with kernel state persistence
3. Kernel pool for parallel test execution
4. Cell modification support via file updates
"""

import pytest
import nbformat
from nbclient import NotebookClient
import shutil
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import hashlib
import asyncio
import concurrent.futures
import threading
from queue import Queue, Empty
import atexit

# ---------------------------------------------------------------------------
# Python 3.14+ compatibility: nest_asyncio + asyncio.timeout() is broken.
# Use a dedicated background thread with its own event loop instead.
# ---------------------------------------------------------------------------


def _make_async_runner():
    """Create a new background event loop + thread pair.

    Returns ``(loop, run_async_fn)`` where *run_async_fn* submits coroutines
    to the loop running in a daemon thread.
    """
    loop = asyncio.new_event_loop()

    def _run(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=_run, args=(loop,), daemon=True)
    thread.start()

    def run_async(coro):
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=120)

    return loop, run_async


# Module-level runner used only by KernelPool (which is rarely used).
_pool_loop, _pool_run_async = _make_async_runner()


def _run_async(coro):
    """Module-level helper used by KernelPool methods only."""
    return _pool_run_async(coro)


# =============================================================================
# KERNEL POOL - Pre-create kernels for faster test execution
# =============================================================================

class KernelPool:
    """
    A pool of pre-warmed Jupyter kernels for faster test execution.
    
    Creates kernels asynchronously in the background so tests don't wait
    for kernel startup.
    """
    
    def __init__(self, pool_size: int = 4, kernel_name: str = 'python3'):
        self.pool_size = pool_size
        self.kernel_name = kernel_name
        self._pool: Queue = Queue()
        self._created_kernels: List[Any] = []
        self._lock = threading.Lock()
        self._shutdown = False
        self._fill_thread: Optional[threading.Thread] = None
        
    def start(self):
        """Start filling the pool with kernels in background."""
        self._fill_thread = threading.Thread(target=self._fill_pool, daemon=True)
        self._fill_thread.start()
    
    def _fill_pool(self):
        """Background thread that keeps the pool full."""
        while not self._shutdown:
            try:
                with self._lock:
                    current_size = self._pool.qsize()
                
                if current_size < self.pool_size:
                    kernel = self._create_kernel()
                    if kernel and not self._shutdown:
                        self._pool.put(kernel)
                        with self._lock:
                            self._created_kernels.append(kernel)
                else:
                    # Pool is full, wait a bit
                    import time
                    time.sleep(0.1)
            except Exception:
                pass
    
    def _create_kernel(self):
        """Create a new kernel manager and client with cash pre-initialized."""
        try:
            from jupyter_client import KernelManager
            km = KernelManager(kernel_name=self.kernel_name)
            km.start_kernel()
            kc = km.client()
            kc.start_channels()
            
            # Wait for ready
            async def wait_ready():
                await kc._async_wait_for_ready(timeout=30)
            
            _run_async(wait_ready())
            
            # Pre-initialize cash so it's ready to use
            self._init_cash_in_kernel(kc)
            
            return {'km': km, 'kc': kc, 'cash_initialized': True}
        except Exception as e:
            print(f"Error creating kernel: {e}")
            return None
    
    def get_kernel(self, timeout: float = 30.0):
        """
        Get a kernel from the pool, or create one if pool is empty.
        
        Returns:
            Dict with 'km' (KernelManager) and 'kc' (KernelClient)
        """
        try:
            return self._pool.get(timeout=timeout)
        except Empty:
            # Pool empty, create on demand
            return self._create_kernel()
    
    def return_kernel(self, kernel: Dict):
        """Return a kernel to the pool for reuse after full restart."""
        if not self._shutdown and kernel:
            # Restart kernel completely to get fresh state
            # This ensures no stale cash state between tests
            try:
                kernel['km'].restart_kernel(now=True)
                kernel['kc'].start_channels()
                
                async def wait_ready():
                    await kernel['kc']._async_wait_for_ready(timeout=10)
                
                _run_async(wait_ready())
                
                # Pre-initialize cash in the fresh kernel
                self._init_cash_in_kernel(kernel['kc'])
                kernel['cash_initialized'] = True
                
                self._pool.put(kernel)
            except Exception:
                # Kernel is broken, shut it down
                self._shutdown_kernel(kernel)
    
    def _init_cash_in_kernel(self, kc):
        """Initialize cash in a kernel."""
        cash_setup = """
%load_ext cash
from cash import Cash
%cash_on
"""
        async def run_setup():
            return await kc._async_execute_interactive(cash_setup, store_history=False)
        
        reply = _run_async(run_setup())
        if reply['content']['status'] != 'ok':
            raise RuntimeError(f"Failed to init cash: {reply['content'].get('evalue', 'unknown error')}")
    
    def _shutdown_kernel(self, kernel: Dict):
        """Shutdown a single kernel."""
        try:
            if kernel.get('kc'):
                kernel['kc'].stop_channels()
            if kernel.get('km') and kernel['km'].has_kernel:
                kernel['km'].shutdown_kernel(now=True)
        except Exception:
            pass
    
    def shutdown(self):
        """Shutdown the pool and all kernels."""
        self._shutdown = True
        
        # Drain the pool
        while True:
            try:
                kernel = self._pool.get_nowait()
                self._shutdown_kernel(kernel)
            except Empty:
                break
        
        # Shutdown any tracked kernels
        with self._lock:
            for kernel in self._created_kernels:
                self._shutdown_kernel(kernel)
            self._created_kernels.clear()


# Global kernel pool instance
_kernel_pool: Optional[KernelPool] = None


def get_kernel_pool() -> KernelPool:
    """Get or create the global kernel pool."""
    global _kernel_pool
    if _kernel_pool is None:
        _kernel_pool = KernelPool(pool_size=4)
        _kernel_pool.start()
        atexit.register(_kernel_pool.shutdown)
    return _kernel_pool


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_text_output(cell, filter_debug: bool = True) -> str:
    """
    Extracts text output from a cell, ignoring HTML/Badge outputs and debug logs.
    
    Args:
        cell: The notebook cell to extract output from
        filter_debug: If True, filter out debug/timing lines (default: True)
    
    Returns:
        str: The extracted text output
    """
    text_outputs = []
    for output in cell.get('outputs', []):
        if output.output_type == 'stream':
            text_outputs.append(output.text)
        elif output.output_type in ('execute_result', 'display_data'):
            data = output.get('data', {})
            if 'text/plain' in data:
                val = data['text/plain']
                if '<IPython.core.display.HTML object>' not in val:
                    text_outputs.append(val)
    
    raw_text = "\n".join(text_outputs)
    
    if filter_debug:
        debug_markers = ['[TIMING', '[UPSTREAM', '[LINEAGE', '[ALREADY', '[CACHE', '[CONTROL', 
                         '_DEBUG]', '[PROXY_CELL_ID]', '[CELL_CHANGED]', '[CELL_UNCHANGED]',
                         '[CELL_ID]', '[STATE]', '[ENSURE_STATE', '[SKIP_', 
                         'Cash:', 'cache_key: stmt:', '| source_hash:']
        filtered_lines = [
            line for line in raw_text.split('\n')
            if not any(marker in line for marker in debug_markers)
        ]
        return "\n".join(filtered_lines).strip()
    
    return raw_text.strip()


# =============================================================================
# NOTEBOOK TEST RUNNER
# =============================================================================

REFERENCE_NOTEBOOKS_DIR = Path(__file__).parent / "reference_notebooks"


class NotebookTestRunner:
    """
    A test runner for notebooks that supports selective cell execution.
    
    Key design principles:
    1. Uses REAL notebook files - cash reads the file naturally, no mocking
    2. Copies notebooks to work_dir so modifications don't affect originals
    3. Modifies cells by rewriting the file - cash sees the changes
    4. Optionally uses kernel pool for faster execution
    
    Example:
        runner = NotebookTestRunner(work_dir=tmp_path)
        runner.create_notebook([
            "x = 10",
            "y = x * 2", 
            "print(f'Result: {y}')"
        ])
        runner.start_kernel()
        runner.run_all()
        assert "Result: 20" in runner.get_output(3)
        
        # Modify a cell and re-run
        runner.set_cell_source(1, "x = 100")
        runner.run_cells([1, 2, 3])
        assert "Result: 200" in runner.get_output(3)
    """
    
    def __init__(
        self,
        work_dir: Path,
        kernel_name: str = 'python3',
        timeout: int = 120,
        use_pool: bool = True,
    ):
        self.work_dir = Path(work_dir)
        self.kernel_name = kernel_name
        self.timeout = timeout
        self.use_pool = use_pool
        
        self.nb: Optional[nbformat.NotebookNode] = None
        self.nb_path: Optional[Path] = None
        self.client: Optional[NotebookClient] = None
        self._kernel_started = False
        self._cash_initialized = False
        self._pooled_kernel: Optional[Dict] = None
        # Each runner gets its own event loop to avoid cross-test contamination
        self._loop, self._run_async = _make_async_runner()
    
    def load(self, notebook_path: Union[str, Path]) -> 'NotebookTestRunner':
        """
        Load a notebook by copying it to the work directory.
        """
        src_path = Path(notebook_path)
        if not src_path.exists():
            raise FileNotFoundError(f"Notebook not found: {src_path}")
        
        name_hash = hashlib.md5(str(src_path).encode()).hexdigest()[:8]
        self.nb_path = self.work_dir / f"test_{src_path.stem}_{name_hash}.ipynb"
        shutil.copy2(src_path, self.nb_path)
        
        with open(self.nb_path, 'r', encoding='utf-8') as f:
            self.nb = nbformat.read(f, as_version=4)
        
        return self
    
    def create_notebook(self, cells: List[str]) -> 'NotebookTestRunner':
        """
        Create a new notebook with the given cells.
        """
        self.nb = nbformat.v4.new_notebook()
        for i, source in enumerate(cells):
            cell = nbformat.v4.new_code_cell(source)
            cell.id = f"cell_{i}"
            self.nb.cells.append(cell)
        
        self.nb_path = self.work_dir / "test_notebook.ipynb"
        self._save_notebook()
        
        return self
    
    def _save_notebook(self) -> None:
        """Save the notebook to disk."""
        with open(self.nb_path, 'w', encoding='utf-8') as f:
            nbformat.write(self.nb, f)
    
    def start_kernel(self, with_cash: bool = True) -> 'NotebookTestRunner':
        """
        Start the kernel and optionally initialize cash.
        
        If using the kernel pool, cash is already pre-initialized in pooled kernels.
        """
        if self.nb is None:
            raise ValueError("No notebook loaded. Call load() or create_notebook() first.")
        
        self.client = NotebookClient(
            self.nb,
            timeout=self.timeout,
            kernel_name=self.kernel_name,
            resources={'metadata': {'path': str(self.work_dir)}}
        )
        
        cash_already_initialized = False
        
        if self.use_pool:
            # Try to get a pre-warmed kernel from the pool
            pool = get_kernel_pool()
            self._pooled_kernel = pool.get_kernel(timeout=5.0)
            
            if self._pooled_kernel:
                self.client.km = self._pooled_kernel['km']
                self.client.kc = self._pooled_kernel['kc']
                cash_already_initialized = self._pooled_kernel.get('cash_initialized', False)
            else:
                # Fall back to creating new kernel
                self._start_new_kernel()
        else:
            self._start_new_kernel()
        
        self._kernel_started = True
        
        # Inject notebook path so cash can find the notebook file
        # This is required for upstream detection to work
        self._inject_notebook_path()
        
        # Only init cash if requested AND not already initialized in pooled kernel
        if with_cash and not cash_already_initialized:
            self._init_cash()
        elif cash_already_initialized:
            self._cash_initialized = True
        
        return self
    
    def _start_new_kernel(self) -> None:
        """Start a new kernel (not from pool)."""
        self.client.km = self.client.create_kernel_manager()
        self.client.start_new_kernel(cwd=str(self.work_dir))
        self.client.kc = self.client.km.client()
        self.client.kc.start_channels()
        
        async def _wait_ready():
            await self.client.kc._async_wait_for_ready(timeout=30)
        
        self._run_async(_wait_ready())
    
    def _inject_notebook_path(self) -> None:
        """Inject the notebook path into the kernel namespace.
        
        This allows cash's get_notebook_path() to find the notebook file,
        which is required for upstream detection to work correctly.
        """
        if self.nb_path is None:
            return
        
        # Use the same variable name that VS Code uses
        # This is checked first in get_notebook_path()
        path_str = str(self.nb_path).replace('\\', '\\\\')
        inject_code = f"__vsc_ipynb_file__ = r'{path_str}'"
        
        reply = self._run_async(
            self.client.kc._async_execute_interactive(inject_code, store_history=False)
        )
        if reply['content']['status'] != 'ok':
            # Non-fatal - just log and continue
            pass
    
    def _init_cash(self) -> None:
        """Initialize cash in the kernel by running setup code directly."""
        cash_setup = """
%load_ext cash
from cash import Cash
%cash_on
"""
        # Run setup code directly via kernel client, not via execute_cell
        # This avoids overwriting notebook cells
        # Suppress output to avoid UnicodeEncodeError on Windows (cp1252 can't encode emojis)
        reply = self._run_async(
            self.client.kc._async_execute_interactive(cash_setup, store_history=False, output_hook=lambda msg: None)
        )
        if reply['content']['status'] != 'ok':
            error_name = reply['content'].get('ename', 'Unknown')
            error_value = reply['content'].get('evalue', '')
            raise RuntimeError(f"Failed to initialize cash: {error_name}: {error_value}")
        self._cash_initialized = True
    
    def set_cell_source(self, cell_num: int, source: str) -> 'NotebookTestRunner':
        """
        Modify a cell's source code and save the notebook.
        
        This rewrites the notebook file so cash can see the changes.
        """
        idx = cell_num - 1
        if idx < 0 or idx >= len(self.nb.cells):
            raise IndexError(f"Cell {cell_num} out of range (1-{len(self.nb.cells)})")
        
        self.nb.cells[idx].source = source
        self.nb.cells[idx].outputs = []
        self._save_notebook()
        
        return self
    
    def get_cell_source(self, cell_num: int) -> str:
        """Get a cell's source code."""
        idx = cell_num - 1
        return self.nb.cells[idx].source

    def add_cell(self, source: str, save: bool = False) -> 'NotebookTestRunner':
        """
        Add a new code cell to the end of the notebook (in memory).
        
        By default, the cell is NOT saved to disk, simulating an unsaved cell
        in VS Code. This is useful for testing that cash handles cells that
        exist in the kernel but not yet in the .ipynb file.
        
        Args:
            source: The cell source code
            save: If True, also save to disk. If False (default), only in memory.
        """
        cell = nbformat.v4.new_code_cell(source)
        cell.id = f"cell_{len(self.nb.cells)}"
        self.nb.cells.append(cell)
        if save:
            self._save_notebook()
        return self

    def run_cell(self, cell_num: int) -> 'NotebookTestRunner':
        """Execute a single cell (1-based indexing)."""
        if not self._kernel_started:
            raise RuntimeError("Kernel not started. Call start_kernel() first.")
        
        idx = cell_num - 1
        cell = self.nb.cells[idx]
        cell.outputs = []
        
        self._run_async(self.client.async_execute_cell(cell, idx))
        
        return self
    
    def run_cells(self, cell_nums: List[int]) -> 'NotebookTestRunner':
        """Execute multiple cells in order."""
        for num in cell_nums:
            self.run_cell(num)
        return self
    
    def run_all(self) -> 'NotebookTestRunner':
        """Execute all cells in the notebook."""
        return self.run_cells(list(range(1, len(self.nb.cells) + 1)))
    
    def get_output(self, cell_num: int, filter_debug: bool = True) -> str:
        """Get the text output from a cell."""
        idx = cell_num - 1
        return get_text_output(self.nb.cells[idx], filter_debug=filter_debug)
    
    def get_raw_output(self, cell_num: int) -> str:
        """Get the raw output from a cell (no filtering)."""
        return self.get_output(cell_num, filter_debug=False)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get machine-readable cash status from the last cell execution.
        
        Returns a dict with:
            - last_cell: Metrics from the last cell execution
            - lineage: Current variable lineage state  
            - executed_codes: Variable to code mapping
            - auto_cache_enabled: Whether auto-caching is on
            - cache_stats: Backend statistics
        """
        import json
        
        status_code = "_cash_status_result = get_ipython().run_line_magic('cash_status', 'dict')"
        self._run_async(
            self.client.kc._async_execute_interactive(status_code, store_history=False)
        )
        
        # Get the result from the kernel
        get_result_code = """
import json as _json
print(_json.dumps(_cash_status_result, default=str))
"""
        self._run_async(
            self.client.kc._async_execute_interactive(get_result_code, store_history=False, output_hook=lambda msg: None)
        )
        
        # Extract result from iopub messages
        try:
            # Find the stream output
            for msg in self.client.kc.iopub_channel.get_msgs():
                if msg['msg_type'] == 'stream' and msg['content'].get('name') == 'stdout':
                    return json.loads(msg['content']['text'].strip())
        except Exception:
            pass
        
        return {}
    
    def get_cell(self, cell_num: int):
        """Get the cell object."""
        return self.nb.cells[cell_num - 1]
    
    def cell_count(self) -> int:
        """Return the number of cells."""
        return len(self.nb.cells)
    
    def reset_cash_state(self) -> 'NotebookTestRunner':
        """
        Reset cash's internal state to simulate a fresh session.
        
        Clears all lineage tracking, executed code records, and file tracking
        state. Variables remain in user_ns but their provenance is lost.
        """
        reset_code = """
try:
    _cash_magics = get_ipython().magics_manager.registry.get('CashMagics')
    if _cash_magics:
        # Clear shared tracking dicts (underscore-prefixed private attributes)
        _cash_magics._tracking_state.variable_lineage.clear()
        _cash_magics._tracking_state.executed_cell_codes.clear()
        _cash_magics._tracking_state.executed_cell_hashes.clear()
        _cash_magics._tracking_state.current_session_hashes.clear()
        _cash_magics._tracking_state.executed_file_deps.clear()
        _cash_magics._tracking_state.vars_with_mutation_lineage.clear()
        # Clear cell code change tracking
        if hasattr(_cash_magics, '_executed_cell_raw_codes'):
            _cash_magics._executed_cell_raw_codes.clear()
        # Clear statement processor's input lineages
        if hasattr(_cash_magics, '_statement_processor'):
            _cash_magics._statement_processor.executed_input_lineages.clear()
        # Clear upstream checker's simulation cache
        if hasattr(_cash_magics, '_upstream_checker'):
            _cash_magics._upstream_checker._simulation_cache = []
        # Clear file tracker state
        if hasattr(_cash_magics, '_file_tracker') and _cash_magics._file_tracker:
            _cash_magics._file_tracker.clear()
except Exception:
    pass
"""
        # Run directly via kernel client to avoid overwriting notebook cells
        self._run_async(
            self.client.kc._async_execute_interactive(reset_code, store_history=False)
        )
        return self
    
    def enable_debug(self) -> 'NotebookTestRunner':
        """Enable cash debug output."""
        # Run directly via kernel client to avoid overwriting notebook cells
        self._run_async(
            self.client.kc._async_execute_interactive("%cash_debug on", store_history=False)
        )
        return self
    
    def shutdown(self) -> None:
        """Shutdown the kernel."""
        if self._pooled_kernel and self.use_pool:
            # Return kernel to pool for reuse
            try:
                pool = get_kernel_pool()
                pool.return_kernel(self._pooled_kernel)
            except Exception:
                pass
            self._pooled_kernel = None
        elif self.client:
            async def _shutdown():
                try:
                    if self.client.kc:
                        self.client.kc.stop_channels()
                    if self.client.km and self.client.km.has_kernel:
                        await self.client.km._async_shutdown_kernel(now=True)
                except Exception:
                    pass
            
            try:
                self._run_async(_shutdown())
            except Exception:
                pass
        
        self._kernel_started = False
        # Replace the event loop so a subsequent start_kernel() works.
        # The old loop's thread is a daemon and will be cleaned up at exit.
        self._loop, self._run_async = _make_async_runner()
    
    def __enter__(self) -> 'NotebookTestRunner':
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


# =============================================================================
# PYTEST FIXTURES
# =============================================================================

@pytest.fixture
def nb_runner(tmp_path):
    """
    Primary fixture for notebook integration tests.
    
    Provides a NotebookTestRunner instance that:
    - Uses real notebook files (no mocking)
    - Copies notebooks to tmp_path for isolation
    - Supports cell modification via file rewrites
    - Uses kernel pool for faster execution (kernels have cash pre-initialized)
    
    Example:
        def test_example(nb_runner):
            nb_runner.create_notebook([
                "x = 10",
                "y = x * 2",
                "print(f'Result: {y}')"
            ])
            nb_runner.start_kernel()  # with_cash=True by default (already done by pool)
            nb_runner.run_all()
            assert "Result: 20" in nb_runner.get_output(3)
            
            # Modify a cell and re-run
            nb_runner.set_cell_source(1, "x = 100")
            nb_runner.run_cells([1, 2, 3])
            assert "Result: 200" in nb_runner.get_output(3)
    """
    # Disable pool for stability - kernel pooling causes hanging issues
    runner = NotebookTestRunner(work_dir=tmp_path, use_pool=False)
    yield runner
    runner.shutdown()


@pytest.fixture
def nb_runner_no_pool(tmp_path):
    """
    Fixture that doesn't use kernel pool (for debugging).
    """
    runner = NotebookTestRunner(work_dir=tmp_path, use_pool=False)
    yield runner
    runner.shutdown()


# Shutdown kernel pool at end of test session
def pytest_sessionfinish(session, exitstatus):
    """Cleanup kernel pool at end of test session."""
    global _kernel_pool
    if _kernel_pool is not None:
        _kernel_pool.shutdown()
        _kernel_pool = None
