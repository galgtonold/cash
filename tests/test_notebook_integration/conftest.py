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
import os
import signal
import sys

# Crash visibility (faulthandler) is installed in the ROOT conftest
# (tests/conftest.py) so it covers every worker, not just notebook workers.

# Opt-in warm-kernel reuse (off by default so the standard run boots a fresh
# kernel per test). When CASH_TEST_REUSE_KERNEL=1, each xdist worker keeps a
# single long-lived kernel alive and resets its state between tests instead of
# paying the ~1.8s boot cost every time. See _WarmKernel below.
_REUSE_KERNEL = os.environ.get("CASH_TEST_REUSE_KERNEL") == "1"

DEFAULT_KERNEL_NAME = 'python3'


@pytest.fixture(scope="session", autouse=True)
def assert_kernelspec_is_this_interpreter():
    """Fail loudly if the kernelspec is not the interpreter under test (CAS-147).

    Every runner here boots ``kernel_name='python3'``, which
    :class:`KernelSpecManager` resolves from the user/system Jupyter search
    paths -- NOT from ``sys.executable``. So a stray ``ipykernel install --user``
    run from any other environment silently repoints the whole integration
    suite at a different interpreter. The suite then grades code that is not
    the editable install under test and still reports green, which makes every
    result quietly worthless. That failure is invisible by construction: there
    is no assertion anywhere that the kernel is us.

    Session-scoped and static (it reads the spec, boots nothing), so it costs
    one file read per worker and reports the problem before any test runs.

    Conservative: only a PROVABLE mismatch fails. A missing spec, a bare
    ``python``/``python3`` argv[0] with no path, or any resolution error leaves
    the suite alone -- this guard exists to catch a specific silent
    misconfiguration, not to become a new source of flakiness.
    """
    try:
        from jupyter_client.kernelspec import KernelSpecManager
        argv = KernelSpecManager().get_kernel_spec(DEFAULT_KERNEL_NAME).argv
    except Exception:  # noqa: BLE001 - no spec / unreadable -> stay out of the way
        return
    problem = kernelspec_mismatch(argv, sys.executable)
    if problem:
        raise RuntimeError(problem)


def kernelspec_mismatch(argv, executable) -> str | None:
    """The CAS-147 comparison, split out so it is directly testable.

    Returns an explanatory message when *argv* provably launches an interpreter
    other than *executable*, else ``None``. Split from the fixture because a
    guard whose negative branch never runs is not a guard -- see
    ``test_kernelspec_guard.py``.
    """
    if not argv or not os.path.isabs(argv[0]):
        return None  # bare `python` -> resolved via PATH at boot; nothing to compare

    def _canon(p):
        return os.path.normcase(os.path.realpath(p))

    if _canon(argv[0]) == _canon(executable):
        return None
    return (
        f"The '{DEFAULT_KERNEL_NAME}' kernelspec points at a DIFFERENT "
        f"interpreter than the one running pytest, so this suite would grade "
        f"code that is not the package under test.\n"
        f"  kernelspec argv[0]: {argv[0]}\n"
        f"  sys.executable:     {executable}\n"
        f"This usually means `ipykernel install --user` was run from another "
        f"environment and overwrote the shared 'python3' spec. Re-run it from "
        f"THIS environment to repoint it:\n"
        f"  {executable} -m ipykernel install --user"
    )


# ---------------------------------------------------------------------------
# Cross-process kernel-boot throttle.
#
# At -n auto (32 workers here) the full suite crashes: workers spawn Jupyter
# kernel subprocesses faster than the OS can absorb (each kernel = several zmq
# sockets + an asyncio loop), workers die ("node down: Not properly terminated")
# and xdist's loadscope scheduler then aborts the whole session with an
# INTERNALERROR KeyError. The deaths cluster around kernel *creation* - a boot
# storm when warm kernels all spin up at once, continuous create/destroy churn
# without reuse - not around test execution.
#
# This caps how many kernels may be *booting* simultaneously across ALL xdist
# workers, so creation pressure stays bounded while test execution keeps full
# parallelism. Shared state lives in a temp dir; a mkdir spin-lock guards the
# read-modify-write, and stale entries (from a crashed worker) expire by TTL so
# a slot is never lost permanently. Cap = CASH_TEST_BOOT_THROTTLE (default 8);
# set 0 to disable.
# ---------------------------------------------------------------------------
import time
import json
import tempfile
from contextlib import contextmanager

_BOOT_CAP = int(os.environ.get("CASH_TEST_BOOT_THROTTLE", "8"))
_BOOT_DIR = os.path.join(tempfile.gettempdir(), "cash_kernel_boot_throttle")
_BOOT_STATE = os.path.join(_BOOT_DIR, "active.json")
_BOOT_LOCK = os.path.join(_BOOT_DIR, "lock.d")
_BOOT_ENTRY_TTL = 90.0  # a boot never takes this long; older entry = dead worker
_BOOT_LOCK_TTL = 15.0   # lock is held only for a quick RMW; older = dead holder


def _boot_lock_acquire():
    while True:
        try:
            os.mkdir(_BOOT_LOCK)
            return
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(_BOOT_LOCK) > _BOOT_LOCK_TTL:
                    os.rmdir(_BOOT_LOCK)
                    continue
            except OSError:
                pass
            time.sleep(0.02)
        except PermissionError:
            # Windows throws WinError 5 (Access is denied) on mkdir when another
            # worker is mid-rmdir of the lock dir, or an AV/indexer briefly holds
            # a handle on it. Transient, not a real failure -- just retry. (Only
            # FileExistsError was caught before; the denser worksteal scheduling
            # surfaced this race as spurious PermissionError test failures.)
            time.sleep(0.02)


def _boot_lock_release():
    try:
        os.rmdir(_BOOT_LOCK)
    except OSError:
        pass


def _boot_state_read():
    try:
        with open(_BOOT_STATE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _boot_state_write(state):
    try:
        with open(_BOOT_STATE, "w") as f:
            json.dump(state, f)
    except OSError:
        pass


@contextmanager
def _boot_throttle():
    """Block until fewer than _BOOT_CAP kernels are booting, then hold a slot."""
    if _BOOT_CAP <= 0:
        yield
        return
    os.makedirs(_BOOT_DIR, exist_ok=True)
    token = f"{os.getpid()}-{time.time_ns()}"
    while True:
        _boot_lock_acquire()
        try:
            now = time.time()
            active = {k: v for k, v in _boot_state_read().items()
                      if now - v < _BOOT_ENTRY_TTL}
            if len(active) < _BOOT_CAP:
                active[token] = now
                _boot_state_write(active)
                break
            _boot_state_write(active)  # persist the TTL pruning
        finally:
            _boot_lock_release()
        time.sleep(0.2)
    try:
        yield
    finally:
        _boot_lock_acquire()
        try:
            active = _boot_state_read()
            active.pop(token, None)
            _boot_state_write(active)
        finally:
            _boot_lock_release()


# ---------------------------------------------------------------------------
# Python 3.14+ compatibility: nest_asyncio + asyncio.timeout() is broken.
# Use a dedicated background thread with its own event loop instead.
# ---------------------------------------------------------------------------


def _make_async_runner():
    """Create a new background event loop + thread pair.

    Returns ``(loop, run_async_fn)`` where *run_async_fn* submits coroutines
    to the loop running in a daemon thread.
    """
    # On Windows the default loop is the Proactor loop, which can't do the
    # add_reader() that zmq needs - so pyzmq/tornado silently spins up an EXTRA
    # "selector" thread per loop. That doubled the per-test thread leak AND made
    # loop.close() race that orphan thread (WinError 10038 "not a socket"). A
    # SelectorEventLoop drives zmq directly: no helper thread, clean close.
    if sys.platform == "win32":
        loop = asyncio.SelectorEventLoop()
    else:
        loop = asyncio.new_event_loop()

    def _run(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=_run, args=(loop,), daemon=True)
    thread.start()
    # Stash the thread on the loop so _close_async_runner() can join it on
    # teardown instead of leaking the thread + loop (and their OS handles).
    loop._cash_thread = thread  # type: ignore[attr-defined]

    def run_async(coro):
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=120)

    return loop, run_async


def _close_async_runner(loop) -> None:
    """Stop, join, and close a runner created by :func:`_make_async_runner`.

    Each NotebookTestRunner spins up its own loop + daemon thread. Previously
    shutdown() abandoned them ("cleaned up at exit"), leaking one thread/loop
    (plus self-pipe handles) per test - thousands by the end of a full run on a
    worker that handles many notebook modules. Closing them here keeps the
    worker's handle/thread count flat.
    """
    if loop is None:
        return
    try:
        loop.call_soon_threadsafe(loop.stop)
    except Exception:
        pass
    thread = getattr(loop, "_cash_thread", None)
    if thread is not None:
        try:
            thread.join(timeout=5)
        except Exception:
            pass
    try:
        loop.close()
    except Exception:
        pass


def _force_kill_kernel(km) -> None:
    """Terminate a kernel's OS process directly from the calling thread.

    The fresh-boot path uses an AsyncKernelManager whose
    ``_async_shutdown_kernel()`` awaits the provisioner on a background loop, and
    that await can deadlock on a Future that never resolves. A wedged teardown
    stalls the xdist worker long enough for the master to declare it "node down",
    and under ``--dist loadscope`` the master then hangs the WHOLE session
    (observed: 15h) instead of recovering. Killing the PID needs no event loop and
    can't deadlock, so a single kernel can never take the run down with it.
    """
    if km is None:
        return
    pid = None
    prov = getattr(km, "provisioner", None)
    if prov is not None:
        pid = getattr(prov, "pid", None)
    if pid is None:
        kernel = getattr(km, "kernel", None)  # legacy jupyter_client: a Popen
        if kernel is not None:
            pid = getattr(kernel, "pid", None)
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)  # Windows: maps to TerminateProcess
        except (OSError, ProcessLookupError):
            pass
    # Drop the connection file etc. so force-killing thousands of kernels over a
    # full run doesn't litter the Jupyter runtime dir. cleanup_resources() is
    # sync on a sync KernelManager (pool/warm) but a coroutine on an
    # AsyncKernelManager (fresh path); the process is already dead and there's no
    # live loop to await on here, so just close the coroutine to avoid a
    # "never awaited" warning.
    try:
        res = km.cleanup_resources()
        if asyncio.iscoroutine(res):
            res.close()
    except Exception:
        pass


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
        # '[cash.' catches every line emitted by the %cash_debug console
        # handler — it formats records as "[<logger name>] <message>" and all
        # cash loggers are named "cash.*", so a single prefix match strips the
        # whole debug stream (e.g. "[cash.notebook.statement.processor] [DEBUG]
        # Processing statement: print(f"...")" which would otherwise leak the
        # statement source into the parsed output).
        # Do NOT add a bare '[DEBUG]'/'[SIZE_AWARE]' marker: cash's own debug
        # lines are formatted "[cash.<logger>] [DEBUG]/[SIZE_AWARE] ..." and the
        # '[cash.' prefix already strips them. User output legitimately contains
        # things like "[DEBUG] Hello" (e.g. a logging decorator), which a bare
        # match would wrongly drop.
        debug_markers = ['[cash.', '[TIMING', '[UPSTREAM', '[LINEAGE', '[ALREADY', '[CACHE',
                         '[CONTROL',
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
# WARM KERNEL (opt-in reuse) - one persistent kernel per xdist worker
# =============================================================================

class _WarmKernel:
    """A single long-lived kernel reused across tests within one worker process.

    Boots the kernel + initialises cash exactly once (so CashMagics' pre_run_cell
    hook and run_cell monkey-patch are installed exactly once - re-running
    %load_ext is a no-op via IPython's extension registry, which avoids the
    hook-stacking trap). Between tests, prepare_for_test() makes the namespace
    and cash state look fresh without re-registering anything.
    """

    def __init__(self, kernel_name: str = 'python3'):
        self.kernel_name = kernel_name
        self.km = None
        self.kc = None
        # All kernel I/O for this warm kernel runs on its own dedicated loop so
        # the async kernel client is never driven across event loops (the
        # likely cause of the old pool's hangs).
        self.loop, self.run_async = _make_async_runner()
        self._initialized = False

    def boot(self) -> None:
        from jupyter_client import KernelManager
        km = KernelManager(kernel_name=self.kernel_name)
        # Throttle holds a slot through wait-for-ready (the whole startup window
        # is where the resource pressure lives), so only _BOOT_CAP kernels boot
        # at once across all workers.
        with _boot_throttle():
            km.start_kernel()
            kc = km.client()
            kc.start_channels()

            async def _wait_ready():
                await kc._async_wait_for_ready(timeout=30)

            self.run_async(_wait_ready())
        self.km = km
        self.kc = kc

    def _exec(self, code: str) -> None:
        self.run_async(
            self.kc._async_execute_interactive(
                code, store_history=False, output_hook=lambda msg: None
            )
        )

    def prepare_for_test(self, work_dir: Path, nb_path: Path) -> None:
        """Reset the kernel so it behaves like a freshly booted one.

        Clears the user namespace, repoints cwd + notebook path at this test's
        tmp dir, rebuilds a fresh default cache backend at this cwd (so no prior
        test's cached entries produce spurious RESTORED hits), clears CashMagics'
        tracking dicts, and purges test-authored modules from sys.modules (so a
        prior test's same-named module can't shadow this test's). Then ensures
        cash is loaded + enabled (idempotent).
        """
        path_str = str(nb_path).replace('\\', '\\\\')
        dir_str = str(work_dir).replace('\\', '\\\\')
        # Clear the namespace in its OWN execution. reset() rebinds user_ns to a
        # fresh dict; if we assigned __vsc_ipynb_file__ in the same cell the
        # assignment would land in the old (captured) globals dict and be
        # invisible afterwards, breaking cash's notebook-path discovery.
        self._exec("get_ipython().reset(new_session=False)")
        # Repoint cwd + notebook path at THIS test's tmp dir (separate exec, so
        # it runs against the new user_ns).
        self._exec(f"import os as _os\n_os.chdir(r'{dir_str}')\n__vsc_ipynb_file__ = r'{path_str}'")
        # Drop & rebuild the cash singleton so NO in-memory tracking state leaks
        # from the previous test. reset_session() gives a fresh Cash (its backend
        # is rebuilt lazily against THIS test's cwd, so no prior test's cached
        # entries produce spurious RESTORED hits) and cleanly re-registers the
        # magics (run_cell hook + pre_run_cell handler) without nesting wrappers.
        # This replaces hand-clearing ~20 tracking dicts, which kept missing
        # attributes (lineage, variable hashes, simulator caches) and leaked
        # state across reused tests. Then re-enable auto-caching on the fresh
        # instance (a fresh Cash starts with caching off).
        self._exec("import cash as _cash\nfrom cash import Cash\n_cash.reset_session()")
        self._exec("%cash_on")
        # Purge test-authored modules from sys.modules so a stale same-named
        # module from a prior test can't shadow this test's import. Runs last,
        # after cash state is rebuilt, so its baseline snapshot includes cash.
        self._exec(_REUSE_PURGE_TEST_MODULES)

    def shutdown(self) -> None:
        if self.km is None:
            return
        # Force-kill rather than await _async_shutdown_kernel: that await can
        # deadlock on this warm kernel's loop and, at session end, hang the whole
        # worker (and the loadscope session). See _force_kill_kernel.
        try:
            if self.kc:
                self.kc.stop_channels()
        except Exception:
            pass
        _force_kill_kernel(self.km)
        self.km = None
        self.kc = None
        # The warm kernel owns its loop + thread for the worker's lifetime; close
        # them now instead of leaving a dangling daemon thread at process exit.
        _close_async_runner(self.loop)
        self.loop = None
        self.run_async = None


# Reset snippet run in-kernel between reused tests. A freshly booted kernel
# starts with a clean sys.modules AND a clean sys.path; reset(new_session=False)
# touches neither, so we emulate a fresh boot here:
#
#   1. sys.path: restore to the first-test baseline. Tests that import a local
#      module do `sys.path.insert(0, str(tmp_path))` in a cell; in a warm kernel
#      that absolute tmp_path would otherwise linger and shadow a later test's
#      same-named module (e.g. one test's helpers.py defining greet() resolving
#      ahead of another's helpers.py defining double(), which relies on cwd).
#      The baseline still contains '' (cwd-relative), so each test re-adds its
#      own path in its own cell and cwd-based imports keep working.
#   2. sys.modules: purge test-authored modules so a stale same-named module
#      from a prior test can't be returned by import. Heavy library modules
#      (pandas/numpy/etc., under stdlib or site-packages) stay cached for speed -
#      re-importing them would defeat warm-kernel reuse and they don't change.
_REUSE_PURGE_TEST_MODULES = """
try:
    import sys as _sys, os as _os, importlib as _il
    _base = getattr(_sys, '_cash_test_baseline_modules', None)
    if _base is None:
        # First reused test: snapshot the post-%cash_on module set and sys.path
        # as the protected baseline (stdlib + cash + cash deps, clean import
        # roots). Only state introduced by tests AFTER this point is reset.
        _sys._cash_test_baseline_modules = frozenset(_sys.modules)
        _sys._cash_test_baseline_syspath = list(_sys.path)
    else:
        _baseline_path = getattr(_sys, '_cash_test_baseline_syspath', None)
        if _baseline_path is not None:
            _sys.path[:] = list(_baseline_path)

        _roots = set()
        for _p in (_sys.base_prefix, _sys.prefix,
                   _sys.base_exec_prefix, _sys.exec_prefix):
            if _p:
                _roots.add(_os.path.normcase(_os.path.abspath(_p)))
        try:
            import site as _site
            for _sp in list(_site.getsitepackages()) + [_site.getusersitepackages()]:
                if _sp:
                    _roots.add(_os.path.normcase(_os.path.abspath(_sp)))
        except Exception:
            pass

        def _cash_is_library(_name, _mod):
            # Protect the package under test by name (its submodules may be
            # imported lazily and aren't always in the baseline, and re-importing
            # them could break isinstance identity for live cash objects).
            if _name == 'cash' or _name.startswith('cash.'):
                return True
            _f = getattr(_mod, '__file__', None)
            if not _f:
                _pth = getattr(_mod, '__path__', None)
                if _pth:
                    try:
                        _f = list(_pth)[0]
                    except Exception:
                        _f = None
            if not _f:
                # No file (builtin/namespace/dynamic) and not in baseline:
                # purge to match a fresh kernel.
                return False
            _fn = _os.path.normcase(_os.path.abspath(_f))
            return any(_fn.startswith(_r) for _r in _roots)

        for _name in [n for n in _sys.modules if n not in _base]:
            _mod = _sys.modules.get(_name)
            if _mod is None:
                continue
            if _cash_is_library(_name, _mod):
                continue
            try:
                del _sys.modules[_name]
            except Exception:
                pass
        try:
            _il.invalidate_caches()
        except Exception:
            pass
except Exception:
    pass
"""


# One warm kernel per worker process (module-level singleton).
_warm_kernel: Optional['_WarmKernel'] = None


def _get_warm_kernel(kernel_name: str = 'python3') -> '_WarmKernel':
    global _warm_kernel
    if _warm_kernel is None:
        wk = _WarmKernel(kernel_name)
        wk.boot()
        atexit.register(wk.shutdown)
        _warm_kernel = wk
    return _warm_kernel


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
        self._warm: Optional['_WarmKernel'] = None
        # Each runner gets its own event loop to avoid cross-test contamination.
        # Created lazily in start_kernel() (and torn down in shutdown()) so a
        # runner that is never started - or one using the shared warm loop -
        # doesn't create and leak a loop + thread.
        self._loop = None
        self._run_async = None
    
    # Whether start_kernel() injected __vsc_ipynb_file__; restart() mirrors it so
    # a no-path run stays a no-path run across a restart.
    _inject_path: bool = True

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
    
    def start_kernel(
        self,
        with_cash: bool = True,
        inject_notebook_path: bool = True,
    ) -> 'NotebookTestRunner':
        """
        Start the kernel and optionally initialize cash.

        If using the kernel pool, cash is already pre-initialized in pooled kernels.

        Args:
            with_cash: install cash + ``%cash_on``. NOTE this records INTENT, not
                outcome — warm-kernel reuse and the kernel pool can both hand back
                a kernel that already has cash installed. If your test's
                conclusion depends on an arm really being cash-off, call
                :meth:`assert_cash_active` rather than trusting this flag: a
                cash-ON "control" reports the same warm-count as a genuine hit.
            inject_notebook_path: define ``__vsc_ipynb_file__`` so cash can
                resolve the notebook. Defaults True because upstream tracking
                needs it — but that default is also a BLIND SPOT: real
                papermill / nbconvert runs have no such variable, and because the
                suite always injected it, cash's no-path branch was never
                exercised and shipped an uncaught IndexError that disabled
                caching and printed an internal error on every cell (CAS-205).
                Pass False to test that environment.
        """
        if self.nb is None:
            raise ValueError("No notebook loaded. Call load() or create_notebook() first.")

        # Remembered so restart() re-injects (or keeps NOT injecting) to match.
        self._inject_path = inject_notebook_path
        
        self.client = NotebookClient(
            self.nb,
            timeout=self.timeout,
            kernel_name=self.kernel_name,
            resources={'metadata': {'path': str(self.work_dir)}}
        )

        cash_already_initialized = False

        # Warm-kernel reuse (opt-in). Only for the common with_cash=True path;
        # with_cash=False tests want a bare kernel with no cash hooks installed,
        # which a persistent warm kernel can't provide, so they fall through to a
        # fresh boot below. Likewise skipped when the caller asked for NO path
        # injection: prepare_for_test() always defines __vsc_ipynb_file__, which
        # would silently defeat the no-path environment the test is trying to
        # reproduce (CAS-205).
        if _REUSE_KERNEL and with_cash and inject_notebook_path:
            wk = _get_warm_kernel(self.kernel_name)
            self._warm = wk
            # Drive ALL kernel I/O on the warm kernel's own loop so the async
            # client is never used across event loops.
            self._loop = wk.loop
            self._run_async = wk.run_async
            self.client.km = wk.km
            self.client.kc = wk.kc
            self._kernel_started = True
            wk.prepare_for_test(self.work_dir, self.nb_path)
            self._cash_initialized = True
            return self

        # Fresh-boot path (no reuse): this runner drives its own kernel I/O, so
        # lazily create its dedicated loop now (closed again in shutdown()).
        if self._loop is None:
            self._loop, self._run_async = _make_async_runner()

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
        if inject_notebook_path:
            self._inject_notebook_path()
        
        # Only init cash if requested AND not already initialized in pooled kernel
        if with_cash and not cash_already_initialized:
            self._init_cash()
        elif cash_already_initialized:
            self._cash_initialized = True
        
        return self
    
    def _start_new_kernel(self) -> None:
        """Start a new kernel (not from pool), retrying a flaky boot.

        Kernel startup binds several ZMQ ports; under parallel boots two
        kernels can race for the same ephemeral port, leaving one unable to
        bind so wait-for-ready times out (``ZMQError: Address already in
        use`` → ``Kernel didn't respond in 30 seconds``). A fresh
        KernelManager re-allocates ports, so retry a couple of times. The
        first attempt is unchanged, so healthy boots behave exactly as
        before; only an already-failing boot pays the retry.
        """
        last_exc: Optional[BaseException] = None
        for _attempt in range(3):
            # Same throttle as warm-kernel boot: without reuse this path runs
            # once per test, so the cap also smooths the create/destroy churn
            # that kills workers mid-run at -n auto.
            with _boot_throttle():
                km = self.client.create_kernel_manager()
                self.client.km = km
                try:
                    self.client.start_new_kernel(cwd=str(self.work_dir))
                    self.client.kc = km.client()
                    self.client.kc.start_channels()

                    async def _wait_ready():
                        await self.client.kc._async_wait_for_ready(timeout=30)

                    self._run_async(_wait_ready())
                    return  # booted cleanly
                except Exception as exc:  # noqa: BLE001 — retry ANY boot failure
                    last_exc = exc
                    # Free the half-booted kernel's ports/PID before retrying.
                    # _force_kill_kernel needs no event loop and can't deadlock.
                    try:
                        if self.client.kc is not None:
                            self.client.kc.stop_channels()
                    except Exception:
                        pass
                    _force_kill_kernel(km)
                    self.client.kc = None
                    self.client.km = None
        raise RuntimeError(
            f"kernel failed to boot after 3 attempts: {last_exc!r}"
        ) from last_exc
    
    def probe_cash_active(self) -> bool:
        """Whether cash auto-caching is ACTUALLY live in the kernel right now.

        OBSERVED by asking the kernel, never inferred from the ``with_cash``
        argument. That distinction is the whole point: ``start_kernel`` records
        intent, but warm-kernel reuse and the kernel pool can both hand back a
        kernel with cash already installed, so a test that *asked* for
        ``with_cash=False`` may still be running cash-ON. When that happens the
        control arm reports ``warm == 0`` -- byte-identical to a genuine cache
        hit -- so the comparison silently proves nothing. Assert on this in any
        test whose conclusion depends on an arm really being cash-off.
        """
        probe = (
            "try:\n"
            "    _ip = get_ipython()\n"
            "    _lm = _ip.magics_manager.magics.get('line', {})\n"
            "    _loaded = 'cash_on' in _lm\n"
            "    _mag = getattr(_ip, '_cash_magics_instance', None)\n"
            "    _auto = bool(getattr(_mag, '_auto_cache_enabled', False)) if _mag else None\n"
            "    _on = bool(_loaded) if _auto is None else bool(_auto)\n"
            "except Exception:\n"
            "    _on = False\n"
            "print('CASH_ACTIVE=' + ('1' if _on else '0'))"
        )
        seen = []

        def _hook(msg):
            if msg['msg_type'] == 'stream':
                seen.append(msg['content'].get('text', ''))

        self._run_async(
            self.client.kc._async_execute_interactive(
                probe, store_history=False, output_hook=_hook,
            )
        )
        return 'CASH_ACTIVE=1' in ''.join(seen)

    def assert_cash_active(self, expected: bool) -> 'NotebookTestRunner':
        """Fail loudly if the kernel's real cash state isn't *expected*.

        Use this to validate a control arm BEFORE trusting its numbers.
        """
        actual = self.probe_cash_active()
        if actual is not expected:
            raise AssertionError(
                f"cash is {'ON' if actual else 'OFF'} in the kernel but the test "
                f"expected {'ON' if expected else 'OFF'}. A cash-ON 'control' "
                f"reports the same warm-count as a genuine cache hit, so any "
                f"comparison built on it is meaningless. If you wanted a bare "
                f"kernel, pass start_kernel(with_cash=False)."
            )
        return self

    def restart(self) -> 'NotebookTestRunner':
        """Restart the kernel in place, preserving the runner's wiring.

        Restart behaviour is a whole class of bug the suite was blind to
        (CAS-190): nb_runner shipped no restart, so every test that needed one
        hand-rolled ``km._async_restart_kernel`` -- 9 copies across the suite,
        each free to get the re-injection wrong. Re-injects the notebook path
        afterwards ONLY if this runner was started with injection, so a
        no-path run stays a no-path run across the restart.
        """
        self._run_async(self.client.km._async_restart_kernel(now=True))
        self._run_async(self.client.kc._async_wait_for_ready(timeout=30))
        if self._inject_path:
            self._inject_notebook_path()
        return self

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

    def enable_persist(self) -> 'NotebookTestRunner':
        """Enable cash 'persist everything' mode.

        Bypasses the cost-aware floors (the 10 ms 'too cheap to cache' floor
        and the size-aware skip) so that even sub-millisecond statements are
        written to the cache and can produce a cache HIT on a subsequent
        identical run.  Use this when a test needs to assert a cache hit on a
        statement that is otherwise too cheap to cache by default.
        """
        # Run directly via kernel client to avoid overwriting notebook cells
        self._run_async(
            self.client.kc._async_execute_interactive("%cash_persist on", store_history=False)
        )
        return self

    def shutdown(self) -> None:
        """Shutdown the kernel."""
        if self._warm is not None:
            # Reuse mode: leave the warm kernel alive for the next test. Drop
            # this test's big variables now to free memory; full state reset
            # happens in prepare_for_test() at the next start_kernel().
            try:
                self._warm._exec("get_ipython().reset(new_session=False)")
            except Exception:
                pass
            self._warm = None
            self._kernel_started = False
            self.client = None
            return
        if self._pooled_kernel and self.use_pool:
            # Return kernel to pool for reuse
            try:
                pool = get_kernel_pool()
                pool.return_kernel(self._pooled_kernel)
            except Exception:
                pass
            self._pooled_kernel = None
        elif self.client:
            # COVERAGE EXPERIMENT (env-gated, no-op in normal runs):
            # ipykernel exits via os._exit(), which skips the atexit hook that
            # coverage.process_startup() relies on to flush. So we explicitly
            # save the kernel's coverage data while the channels are still
            # alive, before the kill.
            if os.environ.get("COVERAGE_PROCESS_START") and self.client.kc:
                async def _save_cov():
                    save_code = (
                        "import coverage as _cov\n"
                        "_c = _cov.Coverage.current()\n"
                        "if _c is not None:\n"
                        "    _c.save()\n"
                    )
                    await self.client.kc._async_execute_interactive(
                        save_code, store_history=False,
                        output_hook=lambda msg: None,
                    )
                try:
                    self._run_async(_save_cov())
                except Exception:
                    pass
            # Tear the kernel down without the async shutdown coroutine, which
            # can deadlock on the background loop (see _force_kill_kernel).
            # stop_channels() is sync + bounded; the PID kill is loop-independent.
            try:
                if self.client.kc:
                    self.client.kc.stop_channels()
            except Exception:
                pass
            _force_kill_kernel(self.client.km)

        self._kernel_started = False
        # Close this runner's loop + background thread instead of leaking them.
        # A subsequent start_kernel() lazily creates a fresh loop, so restart
        # still works without abandoning the old one.
        _close_async_runner(self._loop)
        self._loop = None
        self._run_async = None
    
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


# =============================================================================
# UPSTREAM DECISION TRACE (debugging the simulation/re-execution engine)
# =============================================================================
#
# The simulation engine runs inside the kernel subprocess, and not every module
# logger's output is surfaced through IOPub into get_raw_output (notably the
# `simulator` logger never appears). `cash.notebook._trace.trace_event` writes
# structured JSONL to the file named by CASH_TRACE_FILE, which the cold kernel
# inherits from this process's environment — bulletproof, capture-quirk-free.
#
# Events currently emitted (see _trace.py call sites):
#   simulate_enter      {cell_idx, reassigned, mutated, method_receivers}
#   broken_after_pass2  {broken, tainted}     # lineage/mismatch staleness
#   broken_after_guard  {broken}              # + stale-value guard
#   schedule_reexec     {stmt}                # producer scheduled to re-run
#
# NOTE: requires a cold kernel (the default). The opt-in warm kernel
# (CASH_TEST_REUSE_KERNEL=1) boots once per worker before the env is set, so it
# will not pick up CASH_TRACE_FILE.

class TraceResult:
    """Queryable view over the captured trace records.

    ``run_all`` and ``rerun`` split the records at the ``__run_all_done__``
    boundary so you can inspect only what the re-run decided.
    """

    def __init__(self, records: list[dict]):
        self.records = records
        split = next((i for i, r in enumerate(records)
                      if r.get("event") == "__run_all_done__"), len(records))
        self.run_all = records[:split]
        self.rerun = records[split + 1:]

    def events(self, name: str, *, phase: str = "rerun") -> list[dict]:
        recs = {"rerun": self.rerun, "run_all": self.run_all, "all": self.records}[phase]
        return [r for r in recs if r.get("event") == name]

    def scheduled(self, *, phase: str = "rerun") -> list[str]:
        return [r["stmt"] for r in self.events("schedule_reexec", phase=phase)]

    def __repr__(self) -> str:
        return f"TraceResult(run_all={len(self.run_all)}, rerun={len(self.rerun)})"


@pytest.fixture
def upstream_trace(nb_runner):
    """Capture the upstream-checker decision trace for a scenario.

    Returns a callable ``capture(cells, actions, *, with_cash=True) -> TraceResult``.
    ``actions(nb_runner)`` performs the re-run (e.g. ``lambda r: r.run_cell(2)``)
    after an initial ``run_all``. Example::

        def test_x(upstream_trace):
            t = upstream_trace(
                ["log = []", "# @cash: no-cache\nlog.append(len(log))\nprint(len(log))"],
                lambda r: r.run_cell(2),
            )
            assert "log = []" not in t.scheduled()   # producer must not re-run
    """
    import json
    import os
    import tempfile

    created: list[str] = []

    def _capture(cells, actions, *, with_cash: bool = True) -> 'TraceResult':
        fd, path = tempfile.mkstemp(suffix=".cashtrace.jsonl")
        os.close(fd)
        created.append(path)
        prev = os.environ.get("CASH_TRACE_FILE")
        os.environ["CASH_TRACE_FILE"] = path
        try:
            nb_runner.create_notebook(cells)
            nb_runner.start_kernel(with_cash=with_cash)
            nb_runner.run_all()
            with open(path, "a", encoding="utf-8") as fh:
                fh.write('{"event": "__run_all_done__"}\n')
            actions(nb_runner)
            with open(path, encoding="utf-8") as fh:
                records = [json.loads(line) for line in fh if line.strip()]
            return TraceResult(records)
        finally:
            if prev is None:
                os.environ.pop("CASH_TRACE_FILE", None)
            else:
                os.environ["CASH_TRACE_FILE"] = prev

    yield _capture

    for p in created:
        try:
            os.unlink(p)
        except OSError:
            pass
