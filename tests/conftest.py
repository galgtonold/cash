"""
Pytest configuration and fixtures for test isolation.

This module provides comprehensive fixtures for testing Cash functionality
with proper isolation between tests.
"""

import itertools
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

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
        _FH_FILE = open(  # kept open for the worker's lifetime
            os.path.join(_FH_DIR, f"worker_{os.getpid()}.log"), "w", encoding="utf-8"
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
        # Waited on, never set: the poll's timer. Not `time.sleep`, which a
        # test may patch -- tests/docs patched it to a no-op for every docs
        # test, and this thread then spun holding the GIL, slowing whatever
        # test was running 10-100x until pytest-timeout killed the worker
        # ("node down: Not properly terminated", a different test each run).
        self._tick = threading.Event()

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
        threading.Thread(target=self._run, name="cash-stall-watchdog", daemon=True).start()

    @property
    def poll_interval(self) -> float:
        """How often to check. Scales down for the short timeouts tests use."""
        return min(5.0, max(0.05, self.timeout / 4.0))

    def _run(self) -> None:
        while True:
            self._tick.wait(self.poll_interval)
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
        except Exception:  # never let reporting hide the stall
            pass
        try:
            import faulthandler
            import tempfile

            path = os.path.join(
                tempfile.gettempdir(),
                "cash_faulthandler",
                f"stall_{worker}_{os.getpid()}.log",
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(banner)
                fh.flush()
                faulthandler.dump_traceback(file=fh, all_threads=True)
        except Exception:
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
    except Exception:
        return
    for child in children:
        try:
            child.kill()
        except Exception:
            continue


_STALL_WATCHDOG = _StallWatchdog(_STALL_TIMEOUT)


from traitlets.config import Configurable

from cash import Cash
from cash.backends import FileBackend, InMemoryBackend
from cash.notebook.ipython.magics import CashMagics

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
    """The IPython shell every unit test of the notebook code runs against.

    Just the attributes cash reads from a real ``InteractiveShell``: the
    namespace (``user_global_ns`` is the same dict, as in IPython), the input
    history, the event registry, the transformer lists, ``run_cell`` and the
    display publisher. Use it through the ``mock_shell`` and ``cash_magics``
    fixtures below rather than copying it.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_ns = {}
        self.user_global_ns = self.user_ns
        self.user_ns["_ih"] = []  # Input history
        self.events = MagicMock()
        self.events.register = MagicMock(return_value=None)
        self.ast_transformers = []
        self.input_transformers_cleanup = []
        self.run_cell = MagicMock()
        self.display_pub = type("MockDisplayPub", (), {"publish": MagicMock()})()
        self.magics_manager = MagicMock()
        self.magics_manager.magics = {"cell": {}, "line": {}}

    def reset(self):
        """Reset shell state."""
        self.user_ns.clear()
        self.user_ns["_ih"] = []


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


# ============================================================================
# CashMagics Fixtures
# ============================================================================


@pytest.fixture
def cash_magics(mock_shell, cash_instance):
    """The CashMagics every notebook unit test uses: ``mock_shell`` in front,
    ``cash_instance`` (and so ``clean_backend``) behind.

    As ``%load_ext cash`` leaves it, before ``%cash_on``: run a cell through
    it with ``tests._cell_driver.run_cash_cell``. A test that needs other
    settings builds on this fixture and says what it changes; it does not
    construct its own shell or magics.
    """
    magics = CashMagics(mock_shell, cash_instance)
    yield magics
    # Cleanup
    cash_instance.backend.clear()
    mock_shell.reset()


@pytest.fixture
def statement_processor(cash_magics):
    """The StatementProcessor inside ``cash_magics``, wired to the same shell,
    backend and tracking state, for tests that drive one statement at a time.
    """
    return cash_magics._statement_processor


# ============================================================================
# Isolation and Cleanup Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def disable_auto_magic_registration(monkeypatch):
    """
    Disable automatic magic registration during tests.
    This prevents Cash() from trying to access get_ipython() and register magics,
    which can cause TraitErrors or warnings when running tests.
    Tests that need magics should register them manually or use the cash_magics fixture.
    """
    from cash.core import Cash

    monkeypatch.setattr(Cash, "register_magic", lambda self: None)


#: The process's own ``__main__``, as pytest started it.
_MAIN = sys.modules["__main__"]


@pytest.fixture(autouse=True)
def _main_module_put_back():
    """Put the process's ``__main__`` back after every test.

    ``InteractiveShell.instance()`` installs the shell's user module as
    ``sys.modules["__main__"]`` and ``clear_instance()`` leaves it there, and
    ``shell.reset()`` deletes ``__spec__`` from it. A spawned process later in
    the same worker then failed to start (``get_preparation_data`` reads
    ``__main__.__spec__``): ``test_a_worker_process_is_one`` raised
    ``AttributeError: module '__main__' has no attribute '__spec__'`` in a
    full run, after the edit-effectiveness test's shell, and passed alone.
    """
    yield
    if sys.modules.get("__main__") is not _MAIN:
        sys.modules["__main__"] = _MAIN


@pytest.fixture(autouse=True)
def _cash_logger_put_back():
    """Leave the ``cash`` logger, and cash's record of its own handlers, as
    the test found them.

    ``Cash(debug=True)``, ``cash.configure(debug=True)`` or ``verbose=True``
    give the ``cash`` logger a level and a ``StreamHandler`` on the
    ``sys.stderr`` of that moment, which under pytest is the test's capture
    file. Both outlived the test: cash's debug lines went to a dead capture
    for the rest of the worker, and a later test that passed a logger to a
    cached function reached that handler. Handlers the test added are closed.
    """
    cash_logger = logging.getLogger("cash")
    handlers, level, propagate = cash_logger.handlers[:], cash_logger.level, cash_logger.propagate
    log_module = sys.modules.get("cash._log")
    own = list(log_module._OWN_HANDLERS) if log_module else []
    level_set = log_module._LEVEL_SET if log_module else None
    yield
    for handler in cash_logger.handlers:
        if handler not in handlers:
            handler.close()
    cash_logger.handlers[:] = handlers
    cash_logger.setLevel(level)
    cash_logger.propagate = propagate
    log_module = sys.modules.get("cash._log")
    if log_module is not None:
        log_module._OWN_HANDLERS[:] = own
        log_module._LEVEL_SET = level_set


# ---------------------------------------------------------------------------
# Unit tests never write into the checkout.
#
# Under pytest, cash anchors its default ``.cash`` to the project pytest was
# started in, which is this checkout. Every test that built ``Cash()`` or used
# the module-level ``@cash.cache`` without a ``cache_dir`` wrote there, so a
# unit run left a ``.cash/`` in the working tree and tests could serve each
# other's entries. ``CASH_CACHE_DIR`` is the documented way to move the default,
# so the unit suite sets it: to a per-test folder while a test runs, and to a
# per-process folder while a test module is imported (a module-level
# ``@cash.cache`` or ``Cash()`` builds its instance then, and keeps it).
#
# Integration and wheel-gate tests are left alone. Their kernels and venvs are
# other processes that choose their own directories, and would inherit the
# variable: the per-worker kernel would keep the first test's folder.
# ---------------------------------------------------------------------------
_OWN_CACHE_DIRS = ("test_notebook_integration", "test_wheel_gate")
_DEFAULT_CACHE_DIRS = itertools.count()
_IMPORT_TIME_CACHE_DIR: str | None = None


def _runs_in_process(path: Path) -> bool:
    return not any(part in _OWN_CACHE_DIRS for part in path.parts)


def _import_time_cache_dir() -> str:
    global _IMPORT_TIME_CACHE_DIR
    if _IMPORT_TIME_CACHE_DIR is None:
        import tempfile

        _IMPORT_TIME_CACHE_DIR = tempfile.mkdtemp(prefix="cash-tests-import-")
    return _IMPORT_TIME_CACHE_DIR


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector):
    """Set the import-time folder around a module's collection, which imports it.

    Set and put back in one wrapper, not in ``pytest_collectstart`` and
    ``pytest_collectreport``: pytest skips the report for a module it only
    passes through on the way to a test named by node id, which is how the
    core set and CI select tests, and the folder then stayed set for the rest
    of the session and reached every notebook kernel booted after it.
    """
    if not (isinstance(collector, pytest.Module) and _runs_in_process(collector.path)):
        return (yield)
    previous = os.environ.get("CASH_CACHE_DIR")
    os.environ["CASH_CACHE_DIR"] = _import_time_cache_dir()
    try:
        return (yield)
    finally:
        if previous is None:
            os.environ.pop("CASH_CACHE_DIR", None)
        else:
            os.environ["CASH_CACHE_DIR"] = previous


def pytest_unconfigure(config):
    if _IMPORT_TIME_CACHE_DIR is not None:
        shutil.rmtree(_IMPORT_TIME_CACHE_DIR, ignore_errors=True)


@pytest.fixture(autouse=True)
def _default_cache_dir_outside_the_checkout(request, monkeypatch, tmp_path_factory):
    """Point ``CASH_CACHE_DIR`` at a folder of this test's own, and fail the
    test if a ``.cash/`` still appears in the repository root.

    A test that needs the real default still gets it with
    ``monkeypatch.delenv("CASH_CACHE_DIR")``.
    """
    if not _runs_in_process(request.node.path):
        yield
        return
    checkout_cache = request.config.rootpath / ".cash"
    existed = checkout_cache.exists()
    # Not created here: cash makes it on first write, and most tests never write.
    per_test = tmp_path_factory.getbasetemp() / "default_cache" / str(next(_DEFAULT_CACHE_DIRS))
    monkeypatch.setenv("CASH_CACHE_DIR", str(per_test))
    yield
    if not existed and checkout_cache.exists():
        pytest.fail(
            f"{checkout_cache} appeared while this test ran. A unit test must not "
            f"write into the checkout: pass cache_dir=tmp_path, or keep the "
            f"CASH_CACHE_DIR this conftest sets. (Under xdist, a test running at "
            f"the same time in another worker may be the one that wrote it.)",
            pytrace=False,
        )


# ============================================================================
# Data Fixtures
# ============================================================================


@pytest.fixture
def sample_dataframe():
    """Provide a sample pandas DataFrame for testing."""
    try:
        import pandas as pd

        return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    except ImportError:
        pytest.skip("pandas not installed")


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    _STALL_WATCHDOG.start()
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "requires_ipython: marks tests that require IPython")
    # CashImpurityWarning fires on the call-counter pattern that
    # virtually every test uses (`n['calls'] += 1` to count invocations
    # — a real scope mutation). Treating it as noise for the broad
    # suite; dedicated purity tests opt back in with their own filter.
    config.addinivalue_line(
        "filterwarnings",
        "ignore::cash.CashImpurityWarning",
    )


def pytest_collection_modifyitems(config, items):
    """Every test under ``tests/test_notebook_integration`` IS an integration
    test, whether or not its file remembered to say so.

    The marker was declared but applied by hand, so 249 of 911 files in that
    directory carried it and the other 662 did not. `-m "not integration"`
    therefore still spun up kernels for most of the suite -- it looks like it
    selected a unit run, takes minutes, and brings the load-dependent notebook
    flakes with it. CI never hit this because it excludes the directory by
    path (`--ignore=tests/test_notebook_integration`); only a human or an
    agent typing `-m` did.

    Marking by location makes `-m "not integration"` mean what CI's --ignore
    means. Files that already declare the marker are unaffected: applying it
    twice is a no-op.
    """
    integration_dir = str(Path(__file__).parent / "test_notebook_integration")
    for item in items:
        if str(getattr(item, "fspath", "")).startswith(integration_dir):
            item.add_marker(pytest.mark.integration)


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


# Failures that were retried away, kept so the run can still report them.
#
# A rerun that passes is reported as a bare `R` and a "1 rerun" tally -- the
# failure text is dropped. For the infra signatures the global --only-rerun
# covers (a kernel that could not get a socket) that is the right amount of
# noise. For a test that opts in with `@pytest.mark.flaky` because it is
# load-sensitive, it is not: those failures are rare, not summonable, and
# carry the on-disk evidence that says which mechanism fired. Retrying them
# without this hook would spend the occurrence and keep nothing.
_RERUN_FAILURES: list[tuple[str, str]] = []


def pytest_runtest_logreport(report):
    _STALL_WATCHDOG.poke(f"{report.when}:{report.outcome} {report.nodeid}")
    if report.outcome == "rerun" and report.longrepr is not None:
        _RERUN_FAILURES.append((report.nodeid, str(report.longrepr)))
    if report.when == "teardown":
        # Back to the default limit; the next test declares its own.
        _STALL_WATCHDOG.set_allowance(None)


def pytest_terminal_summary(terminalreporter):
    """Print what each retried-away failure actually said."""
    if not _RERUN_FAILURES:
        return
    tr = terminalreporter
    tr.write_sep("=", "failures that passed on a retry", yellow=True)
    tr.write_line(
        "These did not fail the run. They are shown because a green retry "
        "otherwise discards the only evidence a rare failure produces."
    )
    for nodeid, longrepr in _RERUN_FAILURES:
        tr.write_line("")
        tr.write_line(f"--- {nodeid}", bold=True)
        for ln in _retry_evidence(longrepr):
            tr.write_line(f"    {ln.strip()}")


#: A traceback entry's location line, ``path/to/file.py:123: in func``.
_FRAME_LOCATION = re.compile(r"^\S.*?\.py:\d+: ")


def _retry_evidence(longrepr: str) -> list[str]:
    """The lines of a retried-away failure worth printing.

    The assertion line carries the diagnostic of a failed assertion; the
    frames above it are the same every time and would bury the summary. A
    timeout's message says only how long pytest waited, so for one the
    frames it was stuck in come first, each with its source line under
    ``--tb=short``: where it hung is the whole finding.
    """
    lines = longrepr.splitlines()
    errors = [ln for ln in lines if ln.startswith("E ")]
    if not any("Timeout" in ln for ln in errors):
        return (errors or lines)[-6:]
    frames = []
    for i, ln in enumerate(lines):
        if _FRAME_LOCATION.match(ln):
            code = lines[i + 1].strip() if i + 1 < len(lines) and lines[i + 1].startswith("    ") else ""
            frames.append(f"{ln.strip()}  {code}".rstrip())
    return frames[-8:] + errors[-2:]


# ---------------------------------------------------------------------------
# Persistence-floor constants for tests.
#
# Cross-process persistence has a ~0.1 s compute floor (see
# ``cash.backends.persistence_policy.COMPUTE_FLOOR_S``): a result cheaper
# than that is never written past RAM. A test that spawns a second process and
# asserts something about a *cached* value therefore proves nothing unless the
# work exceeds the floor -- the second process simply recomputes, and the
# assertion holds whether or not the bug under test exists. Two brand-new tests
# passed against unfixed source exactly this way.
#
# Use ABOVE_PERSISTENCE_FLOOR_S for the sleep, and assert the body ran once
# across the runs so the test also proves the value was genuinely cached.
# ---------------------------------------------------------------------------
ABOVE_PERSISTENCE_FLOOR_S = 0.2


# ---------------------------------------------------------------------------
# Silent-degradation guard: a discarded cache write must fail the suite.
#
# A failed cache write does not raise. The entry is simply absent, the work is
# recomputed, and every test still passes -- so a cache that has quietly
# stopped caching looks exactly like one that works. Cash reports it once, from
# ``_report_failed_writes`` at shutdown, which in a notebook means at kernel
# death: in practice, never.
#
# That is not hypothetical. Windows denied ``os.replace`` whenever a reader
# held the destination open, discarding writes on effectively every Windows CI
# job and 10 of 12 consecutive local runs of one test -- green the whole time,
# and found only by grepping the full logs of PASSING jobs. This guard is the
# detector that would have caught it on the first run.
#
# Delta-snapshotted rather than absolute: a queue outlives the test that made
# it, so an absolute check would blame whichever test ran next. And drained
# before the check: a write still in flight when its test ends would otherwise
# fail, or log, during the NEXT test and be charged to it. The first disk store
# logs the cache's size cap on cash.storage from the write worker, so without
# the drain that line landed in the next test's caplog.
#
# Opt out with ``@pytest.mark.expects_failed_writes`` for tests that induce a
# failure deliberately.
# ---------------------------------------------------------------------------
def _discarded_write_count():
    try:
        from cash.backends._writes import discarded_writes
    except Exception:  # import cycles during collection
        return 0
    return len(discarded_writes())


def _discarded_since(n):
    from cash.backends._writes import discarded_writes

    return discarded_writes()[n:]


def _drain_pending_writes():
    """Let every background write submitted so far finish, with whatever it
    logs or fails with, before the next test starts."""
    try:
        from cash.backends._writes import all_pending_writes
    except Exception:  # import cycles during collection
        return
    for queue in all_pending_writes():
        try:
            queue.wait_all()
        except Exception:  # a dying queue is not a finding
            continue


@pytest.fixture(autouse=True)
def _no_silently_discarded_cache_writes(request):
    before = _discarded_write_count()
    yield
    # This test's writes land in this test: a failure is charged to it, and a
    # record a write logs (the size cap on cash.storage) cannot reach the next
    # test's caplog or handlers. test_label_consistency and a verbose-logging
    # test each once caught the previous test's tail.
    _drain_pending_writes()
    if request.node.get_closest_marker("expects_failed_writes"):
        from cash.backends._writes import reset_discarded_writes

        reset_discarded_writes(keep=before)
        return
    new = _discarded_since(before)
    if not new:
        return
    detail = "\n".join(f"    {key}\n        {exc}" for key, exc in new)
    pytest.fail(
        f"{len(new)} cache write(s) failed and were silently discarded.\n"
        f"The entries are absent, so the work will be recomputed -- a cache "
        f"that has stopped caching without anything going red.\n"
        f"(Each test's writes are drained when it ends, so these are its own.)\n{detail}",
        pytrace=False,
    )


#: Discarded writes reported back by xdist workers, collected on the controller.
_WORKER_DISCARDED: list[tuple[str, str]] = []


def pytest_testnodedown(node, error):
    """Collect each xdist worker's discarded writes as it finishes.

    Required because the backstop below runs *per process*. Under ``-n auto``
    that is a worker, whose ``session.exitstatus`` the controller never reads --
    so a worker-side failure vanished entirely. Verified: with the bug present,
    ``-n0`` exited 1 and ``-n4`` exited 0. CI runs ``-n auto``, so without this
    handoff the guard would have been decorative exactly where it matters.
    """
    payload = (getattr(node, "workeroutput", None) or {}).get("cash_discarded_writes")
    if payload:
        _WORKER_DISCARDED.extend(tuple(item) for item in payload)


def pytest_sessionfinish(session, exitstatus):
    """Backstop for writes that failed after their own test had finished.

    The per-test fixture cannot see a write still in flight when the test ends.
    Draining every live queue here catches that tail, at the cost of no
    attribution -- the right trade for a detector whose job is to ensure the
    class cannot pass silently.
    """
    try:
        from cash.backends._writes import all_pending_writes, discarded_writes
    except Exception:
        return
    for queue in all_pending_writes():
        try:
            queue.wait_all()
        except Exception:  # a dying queue is not a finding
            continue

    leaked = list(discarded_writes())

    # In an xdist worker: hand the findings to the controller and stop. Failing
    # here would be invisible.
    workeroutput = getattr(session.config, "workeroutput", None)
    if workeroutput is not None:
        workeroutput["cash_discarded_writes"] = [list(item) for item in leaked]
        return

    leaked = _WORKER_DISCARDED + leaked  # controller, or plain -n0 run
    if not leaked or exitstatus != 0:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line("")
        reporter.write_line(
            f"DISCARDED CACHE WRITES: {len(leaked)} write(s) failed and were thrown away during this session.",
            red=True,
        )
        reporter.write_line(
            "The work was recomputed. Nothing raised at the call site, so this "
            "is a cache doing less than it appears to.",
            red=True,
        )
        for key, exc in leaked[:10]:
            reporter.write_line(f"    {key}: {exc}")
        if len(leaked) > 10:
            reporter.write_line(f"    ... and {len(leaked) - 10} more")
    session.exitstatus = 1
