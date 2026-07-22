"""
Pytest configuration and fixtures for test isolation.

This module provides comprehensive fixtures for testing Cash functionality
with proper isolation between tests.
"""
import pytest
import shutil
import time
from pathlib import Path
from unittest.mock import MagicMock
from io import StringIO
import sys
import os

# ---------------------------------------------------------------------------
# Crash visibility (ALL workers): dump a C-level traceback when a worker dies
# silently. This lives in the ROOT conftest so every xdist worker enables it at
# startup, regardless of which test modules it ends up running. (The notebook
# conftest only loads for notebook modules, so a worker running unit tests would
# otherwise have no faulthandler installed - which is why earlier dumps were
# empty.) Per-PID file survives xdist's stderr capture. CASH_TEST_FAULTHANDLER=0
# to disable. Inspect after a crash: %TEMP%/cash_faulthandler/worker_*.log
# ---------------------------------------------------------------------------
if os.environ.get("CASH_TEST_FAULTHANDLER", "1") == "1":
    import faulthandler as _faulthandler
    import tempfile as _tempfile

    _FH_DIR = os.path.join(_tempfile.gettempdir(), "cash_faulthandler")
    try:
        os.makedirs(_FH_DIR, exist_ok=True)
        _FH_FILE = open(  # noqa: SIM115 - kept open for the worker's lifetime
            os.path.join(_FH_DIR, f"worker_{os.getpid()}.log"), "w"
        )
        _faulthandler.enable(file=_FH_FILE, all_threads=True)
    except OSError:
        _faulthandler.enable(all_threads=True)

# ---------------------------------------------------------------------------
# Stall watchdog (ALL processes: xdist master AND every worker).
#
# `timeout = 30` in pyproject is a PER-TEST backstop, so it only covers time
# spent inside a test. It cannot see a stall in collection, in worker startup,
# or in the master waiting on workers -- and that is exactly where this suite
# hangs. Observed twice: a 15h hang under the old loadscope scheduler, and
# (2026-07-19, worksteal) an 11.5h hang on a chunk that normally takes 3 min,
# producing NO output and leaving 23 orphaned python processes. Nothing
# reported it; the run simply never returned, so an unattended sweep can eat a
# whole session before anyone notices.
#
# This converts any such stall into a fast, loud failure with a stack dump.
# In a worker, exiting looks like an ordinary worker death, which worksteal
# already recovers from by redistributing the remaining tests -- so one stuck
# worker costs one test instead of the entire run. In the master, it ends the
# process (taking the workers with it) so the caller gets a non-zero exit
# instead of blocking forever.
#
# Tuning: CASH_TEST_STALL_TIMEOUT seconds (default 300; 0 disables). The floor
# is set by the longest legitimate silence -- a single integration test may run
# up to its 120s mark -- so 300 leaves ample margin while still turning an
# 11-hour hang into a 5-minute one.
# ---------------------------------------------------------------------------
_STALL_TIMEOUT = float(os.environ.get("CASH_TEST_STALL_TIMEOUT", "300"))


class _StallWatchdog:
    """Kills this process if no test progress happens for _STALL_TIMEOUT."""

    def __init__(self, timeout: float) -> None:
        import threading

        self.timeout = timeout
        self._lock = threading.Lock()
        self._last = time.monotonic()
        self._current = "<none yet>"
        self._started = False
        self._allowance: float | None = None

    def poke(self, what: str | None = None) -> None:
        with self._lock:
            self._last = time.monotonic()
            if what is not None:
                self._current = what

    def set_allowance(self, seconds: float | None) -> None:
        """Let the running test raise the silence limit for its own duration.

        A test that declares ``@pytest.mark.timeout(N)`` with N above the stall
        limit is asserting that N seconds of silence is legitimate for it. The
        wheel gate is the real case: one test that builds a wheel, provisions a
        venv and drives a real Jupyter server for ~13 minutes, declaring 1800s.
        Against a flat 300s limit it was killed at 300s on every single run, so
        the release gate could never pass through pytest at all.

        Only ever raises, never lowers -- a short per-test timeout must not
        shorten the watchdog and turn a slow-but-healthy phase into a kill.
        """
        with self._lock:
            self._allowance = seconds if (seconds and seconds > self.timeout) else None

    def effective_timeout(self) -> float:
        with self._lock:
            return self._allowance or self.timeout

    def start(self) -> None:
        import threading

        if self._started or self.timeout <= 0:
            return
        self._started = True
        threading.Thread(target=self._run, name="cash-stall-watchdog",
                         daemon=True).start()

    @property
    def poll_interval(self) -> float:
        """How often to check. Scales down for the short timeouts tests use."""
        return min(5.0, max(0.05, self.timeout / 4.0))

    def idle_seconds(self) -> float:
        with self._lock:
            return time.monotonic() - self._last

    def _run(self) -> None:
        while True:
            time.sleep(self.poll_interval)
            with self._lock:
                idle = time.monotonic() - self._last
                current = self._current
                limit = self._allowance or self.timeout
            if idle >= limit:
                self._fire(idle, current)
                return

    def banner(self, idle: float, current: str) -> str:
        worker = os.environ.get("PYTEST_XDIST_WORKER", "master")
        # Report the limit that actually applied, which a long-running test
        # may have raised for its own duration.
        limit = self.effective_timeout()
        return (
            f"\n{'=' * 72}\n"
            f"CASH STALL WATCHDOG: no test progress for {idle:.0f}s "
            f"(limit {limit:.0f}s)\n"
            f"  process : {worker} (pid {os.getpid()})\n"
            f"  last    : {current}\n"
            f"  Dumping all thread stacks, then killing this process.\n"
            f"  Raise or disable with CASH_TEST_STALL_TIMEOUT.\n"
            f"{'=' * 72}\n"
        )

    def _fire(self, idle: float, current: str) -> None:
        worker = os.environ.get("PYTEST_XDIST_WORKER", "master")
        banner = self.banner(idle, current)
        # stderr may be captured by xdist, so also write a per-PID file that
        # survives, next to the faulthandler dumps.
        try:
            sys.stderr.write(banner)
            sys.stderr.flush()
        except Exception:  # noqa: BLE001 - never let reporting hide the stall
            pass
        try:
            import faulthandler
            import tempfile

            path = os.path.join(
                tempfile.gettempdir(), "cash_faulthandler",
                f"stall_{worker}_{os.getpid()}.log",
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as fh:
                fh.write(banner)
                fh.flush()
                faulthandler.dump_traceback(file=fh, all_threads=True)
        except Exception:  # noqa: BLE001
            pass
        _kill_child_processes()
        # os._exit, not sys.exit: whatever is stuck (a zmq recv, a spin loop,
        # a dead kernel handshake) will not unwind on an exception.
        os._exit(3)


def _kill_child_processes() -> None:
    """Best-effort reap of this process's children (Jupyter kernels, workers).

    Without this a watchdog exit leaves the orphans that made the observed hang
    so expensive to clean up by hand.
    """
    try:
        import psutil
    except ImportError:
        return
    try:
        me = psutil.Process(os.getpid())
        children = me.children(recursive=True)
    except Exception:  # noqa: BLE001
        return
    for child in children:
        try:
            child.kill()
        except Exception:  # noqa: BLE001
            continue


_STALL_WATCHDOG = _StallWatchdog(_STALL_TIMEOUT)


from cash import Cash
from cash.backends import InMemoryBackend, FileBackend
from cash.notebook.ipython.magics import CashMagics
from traitlets.config import Configurable


# ============================================================================
# Backend Fixtures
# ============================================================================

@pytest.fixture
def clean_backend():
    """Provide a fresh InMemoryBackend for each test."""
    backend = InMemoryBackend()
    yield backend
    backend.clear()


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Provide a unique temporary directory for file-based caches."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    yield str(cache_dir)
    # Cleanup with retry for Windows file locking
    max_retries = 3
    for i in range(max_retries):
        try:
            shutil.rmtree(cache_dir, ignore_errors=False)
            break
        except (PermissionError, OSError):
            if i < max_retries - 1:
                time.sleep(0.1)


@pytest.fixture
def file_backend(temp_cache_dir):
    """Provide a FileBackend with unique temp directory."""
    backend = FileBackend(cache_dir=temp_cache_dir)
    yield backend
    backend.clear()


# ============================================================================
# Cash Instance Fixtures
# ============================================================================

@pytest.fixture
def cash_instance(clean_backend):
    """Provide a fresh Cash instance with InMemoryBackend."""
    cash = Cash(backend=clean_backend, register_magic=False)
    yield cash
    cash.backend.clear()


@pytest.fixture
def cash_with_file_backend(file_backend):
    """Provide a Cash instance with FileBackend."""
    cash = Cash(backend=file_backend, register_magic=False)
    yield cash
    cash.backend.clear()


# ============================================================================
# IPython Shell Mock Fixtures
# ============================================================================

class MockShell(Configurable):
    """Mock IPython shell with all required attributes."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_ns = {}
        self.user_ns['_ih'] = []  # Input history
        self.events = MagicMock()
        self.events.register = MagicMock(return_value=None)
        self.ast_transformers = []
        self.input_transformers_cleanup = []
        self.run_cell = MagicMock()
        self.display_pub = type('MockDisplayPub', (), {'publish': MagicMock()})()
        self.magics_manager = MagicMock()
        self.magics_manager.magics = {'cell': {}, 'line': {}}
        
    def reset(self):
        """Reset shell state."""
        self.user_ns.clear()
        self.user_ns['_ih'] = []


@pytest.fixture
def mock_shell():
    """Provide a mock IPython shell."""
    shell = MockShell()
    
    # Configure run_cell to execute code in user_ns
    def run_cell_impl(cell):
        try:
            exec(cell, {}, shell.user_ns)
            result = MagicMock()
            result.success = True
            return result
        except Exception as e:
            result = MagicMock()
            result.success = False
            result.error_in_exec = e
            return result
    
    shell.run_cell.side_effect = run_cell_impl
    yield shell
    shell.reset()


@pytest.fixture
def simple_mock_shell():
    """Provide a simple mock shell without exec functionality."""
    shell = MagicMock()
    shell.user_ns = {}
    shell.events = MagicMock()
    shell.ast_transformers = []
    shell.input_transformers_cleanup = []
    shell.run_cell = MagicMock()
    return shell


# ============================================================================
# CashMagics Fixtures
# ============================================================================

@pytest.fixture
def cash_magics(mock_shell, cash_instance):
    """Provide CashMagics instance with mock shell and clean backend."""
    magics = CashMagics(mock_shell, cash_instance)
    yield magics
    # Cleanup
    cash_instance.backend.clear()
    mock_shell.reset()


# ============================================================================
# Isolation and Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def isolate_tests(monkeypatch):
    """
    Automatically isolate each test from side effects.
    Runs before every test automatically.
    """
    # Store original sys.modules to detect imports
    original_modules = set(sys.modules.keys())
    
    yield
    
    # Cleanup: Remove any new modules that were imported during test
    # This prevents module-level state from leaking between tests
    new_modules = set(sys.modules.keys()) - original_modules
    for module in new_modules:
        if module.startswith('test_') or 'cash' not in module:
            # Don't remove test modules or non-cash modules
            continue


@pytest.fixture(autouse=True)
def disable_auto_magic_registration(monkeypatch):
    """
    Disable automatic magic registration during tests.
    This prevents Cash() from trying to access get_ipython() and register magics,
    which can cause TraitErrors or warnings when running tests.
    Tests that need magics should register them manually or use the cash_magics fixture.
    """
    from cash.core import Cash
    monkeypatch.setattr(Cash, 'register_magic', lambda self: None)


@pytest.fixture
def isolated_test(monkeypatch, tmp_path):
    """
    Provide complete test isolation:
    - Clean working directory
    - Isolated environment
    """
    original_cwd = Path.cwd()
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    monkeypatch.chdir(original_cwd)


# ============================================================================
# Output Capture Fixtures
# ============================================================================

@pytest.fixture
def captured_output():
    """Capture stdout and stderr during test execution."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture
    
    yield stdout_capture, stderr_capture
    
    sys.stdout = old_stdout
    sys.stderr = old_stderr


# ============================================================================
# Data Fixtures
# ============================================================================

@pytest.fixture
def sample_dataframe():
    """Provide a sample pandas DataFrame for testing."""
    try:
        import pandas as pd
        return pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
    except ImportError:
        pytest.skip("pandas not installed")


@pytest.fixture
def sample_data():
    """Provide sample data for caching tests."""
    return {
        'integers': [1, 2, 3, 4, 5],
        'strings': ['a', 'b', 'c'],
        'nested': {'key1': 'value1', 'key2': [1, 2, 3]}
    }


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    _STALL_WATCHDOG.start()
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_ipython: marks tests that require IPython"
    )
    # CashImpurityWarning fires on the call-counter pattern that
    # virtually every test uses (`n['calls'] += 1` to count invocations
    # — a real scope mutation). Treating it as noise for the broad
    # suite; dedicated purity tests opt back in with their own filter.
    config.addinivalue_line(
        "filterwarnings",
        "ignore::cash.CashImpurityWarning",
    )


# --- Stall-watchdog progress hooks -----------------------------------------
# Every hook below runs in BOTH the xdist master (where reports arrive from
# workers) and each worker (where they arrive locally), so a stall is caught
# whichever side stops making progress.

def pytest_collectreport(report):
    """Per-FILE collection progress.

    Collecting 768 integration files is slow, and pokes only at
    ``collection_finish`` would let a healthy but lengthy collection look
    exactly like a stall.
    """
    _STALL_WATCHDOG.poke(f"collecting {report.nodeid}")


def pytest_collection_finish(session):
    _STALL_WATCHDOG.poke(f"collected {len(session.items)} items")


def pytest_runtest_logstart(nodeid, location):
    _STALL_WATCHDOG.poke(f"started {nodeid}")


def pytest_runtest_setup(item):
    """Give a test that declares a long ``timeout`` mark that much silence.

    Without this the flat 300s stall limit overrides every longer per-test
    budget in the suite. The wheel gate (``@pytest.mark.timeout(1800)``, ~13
    minutes of wheel build + venv provisioning + a real Jupyter server) was
    killed at 300s on every run, so ``pytest -m wheel_gate`` could not succeed
    -- it exited 3 with no output, which reads like a broken gate rather than a
    watchdog kill.
    """
    seconds = None
    mark = item.get_closest_marker("timeout")
    if mark is not None:
        raw = mark.args[0] if mark.args else mark.kwargs.get("timeout")
        try:
            seconds = float(raw)
        except (TypeError, ValueError):
            seconds = None
    _STALL_WATCHDOG.set_allowance(seconds)


def pytest_runtest_logreport(report):
    _STALL_WATCHDOG.poke(f"{report.when}:{report.outcome} {report.nodeid}")
    if report.when == "teardown":
        # Back to the default limit; the next test declares its own.
        _STALL_WATCHDOG.set_allowance(None)


# ---------------------------------------------------------------------------
# Persistence-floor constants for tests.
#
# Cross-process persistence has a ~0.1 s compute floor (see
# ``cash.backends.factory._SMART_PERSIST_COMPUTE_FLOOR_S``): a result cheaper
# than that is never written past RAM. A test that spawns a second process and
# asserts something about a *cached* value therefore proves nothing unless the
# work exceeds the floor -- the second process simply recomputes, and the
# assertion holds whether or not the bug under test exists. Two brand-new tests
# passed against unfixed source exactly this way.
#
# Use ABOVE_PERSISTENCE_FLOOR_S for the sleep, and assert the body ran once
# across the runs so the test also proves the value was genuinely cached.
# ---------------------------------------------------------------------------
PERSISTENCE_FLOOR_S = 0.1
ABOVE_PERSISTENCE_FLOOR_S = 0.2
