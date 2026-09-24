"""Booting, reusing and killing the Jupyter kernels the test runner drives."""

import asyncio
import atexit
import json
import os
import shutil
import signal
import sys
import tempfile
import threading
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

# Warm-kernel reuse: each xdist worker keeps ONE long-lived kernel alive and
# resets its state between tests instead of paying the ~2.3s boot every time.
# The suite booted 4008 kernels for 4102 tests while a trivial test's own work
# took 0.02s, so the boot WAS the runtime. Measured over all 838 files, both
# arms green: 33.4 min fresh-per-test vs 14-16 min reused.
#
# ON by default. Set CASH_TEST_REUSE_KERNEL=0 to force a fresh kernel per test
# -- worth doing when a failure looks like cross-test contamination, because
# that comparison is exactly what attributes it. Six classes of state a fresh
# kernel resets by dying and a warm one does not each cost this suite real
# failures; see _WarmKernel.prepare_for_test, which undoes all six.
#
# Individual tests opt out with `@pytest.mark.fresh_kernel`; start_kernel()
# opts out automatically for with_cash=False and no-path runs, and shutdown()
# opts out for the remainder of a test that shut its kernel down mid-way.
_REUSE_KERNEL = os.environ.get("CASH_TEST_REUSE_KERNEL", "1") != "0"

DEFAULT_KERNEL_NAME = "python3"

# Test kernels get an empty IPython directory. Kernels inherit this process's
# environment, so without this a developer's own profile startup files run in
# every test kernel -- `cash autoload` writes one that imports cash and runs
# %cash_on -- and a test that forgot `import cash` passed on that machine and
# failed everywhere else. Set at import, before any kernel is started.
_IPYTHON_DIR = tempfile.mkdtemp(prefix="cash-test-ipython-")
os.environ["IPYTHONDIR"] = _IPYTHON_DIR
atexit.register(shutil.rmtree, _IPYTHON_DIR, ignore_errors=True)

# A CASH_CACHE_DIR in the developer's shell would reach every test kernel, and
# all tests on a worker would then share one cache instead of each notebook's
# own `.cash/`, so a test could be served another test's results.
os.environ.pop("CASH_CACHE_DIR", None)


def kernelspec_mismatch(argv, executable) -> str | None:
    """The kernelspec comparison, split out so it is directly testable.

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

_BOOT_CAP = int(os.environ.get("CASH_TEST_BOOT_THROTTLE", "8"))


# When to throw a warm kernel away and boot a fresh one. See
# _WarmKernel._recycle_if_bloated: a reused kernel's RSS only climbs, and 16 of
# them growing across a 4000-test unchunked run filled a 64 GB machine and
# stalled every worker at once.
_WARM_MAX_RSS_BYTES = int(os.environ.get("CASH_TEST_WARM_MAX_RSS_MB", "1200")) * 2**20
_WARM_MAX_TESTS = int(os.environ.get("CASH_TEST_WARM_MAX_TESTS", "250"))
_BOOT_DIR = os.path.join(tempfile.gettempdir(), "cash_kernel_boot_throttle")
_BOOT_STATE = os.path.join(_BOOT_DIR, "active.json")
_BOOT_LOCK = os.path.join(_BOOT_DIR, "lock.d")
_BOOT_ENTRY_TTL = 90.0  # a boot never takes this long; older entry = dead worker
_BOOT_LOCK_TTL = 15.0  # lock is held only for a quick RMW; older = dead holder
# Hard deadlines so neither wait here can spin forever. Both are far above any
# legitimate wait (the lock is held for milliseconds; a slot frees within a
# kernel boot), so hitting one means something is wrong -- and proceeding
# degrades throttling, whereas spinning hangs the suite with no output.
_BOOT_LOCK_WAIT_MAX = 60.0
_BOOT_SLOT_WAIT_MAX = 180.0


def _boot_lock_acquire() -> bool:
    """Take the boot-throttle lock, giving up rather than spinning forever.

    Returns True when the lock is HELD and must be released, False when the
    deadline passed and the caller is proceeding without it. Callers must pass
    that result to :func:`_boot_lock_release` -- releasing a lock we do not own
    would delete the holder's lock and hand it to two processes at once.

    Both loops in this module were unbounded ``while True``. The lock guards a
    few-millisecond read-modify-write on a small JSON file, and the state lives
    in the SYSTEM temp dir, so it is shared across runs and survives a killed
    session. Every failure path here (an undeletable lock dir, a handle held by
    an AV scanner, a crashed holder) therefore spun silently and forever, with
    no timeout and no diagnostic -- one of the few places in the harness that
    could hang with no output at all, which is the observed symptom.

    On deadline we proceed WITHOUT the lock. The worst case is a lost update to
    the slot table -- one kernel too many or too few boots concurrently, a
    throttling hiccup. That is strictly better than hanging the suite, because
    the throttle is a performance guard, not a correctness device.
    """
    deadline = time.time() + _BOOT_LOCK_WAIT_MAX
    while True:
        try:
            os.mkdir(_BOOT_LOCK)
            return True
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
        if time.time() > deadline:
            warnings.warn(
                f"cash test harness: boot-throttle lock not acquired in "
                f"{_BOOT_LOCK_WAIT_MAX:.0f}s ({_BOOT_LOCK}); proceeding without "
                f"it. Throttling may be briefly inaccurate. This is a guard "
                f"against an unbounded spin, not a test failure.",
                RuntimeWarning,
                stacklevel=2,
            )
            return False


def _boot_lock_release(held: bool = True):
    """Release the lock, but ONLY if this process actually acquired it."""
    if not held:
        return
    try:
        os.rmdir(_BOOT_LOCK)
    except OSError:
        pass


def _boot_state_read():
    try:
        with open(_BOOT_STATE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _boot_state_write(state):
    try:
        with open(_BOOT_STATE, "w", encoding="utf-8") as f:
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
    deadline = time.time() + _BOOT_SLOT_WAIT_MAX
    while True:
        held = _boot_lock_acquire()
        try:
            now = time.time()
            active = {k: v for k, v in _boot_state_read().items() if now - v < _BOOT_ENTRY_TTL}
            if len(active) < _BOOT_CAP:
                active[token] = now
                _boot_state_write(active)
                break
            _boot_state_write(active)  # persist the TTL pruning
        finally:
            _boot_lock_release(held)
        if time.time() > deadline:
            # Every slot has looked busy for the whole deadline even though
            # entries expire by TTL. Boot anyway: an over-subscribed boot is a
            # slow test, a permanent wait is a dead suite.
            warnings.warn(
                f"cash test harness: no kernel-boot slot for "
                f"{_BOOT_SLOT_WAIT_MAX:.0f}s (cap {_BOOT_CAP}); booting anyway. "
                f"Guard against an unbounded wait, not a test failure.",
                RuntimeWarning,
                stacklevel=2,
            )
            break
        time.sleep(0.2)
    try:
        yield
    finally:
        held = _boot_lock_acquire()
        try:
            active = _boot_state_read()
            active.pop(token, None)
            _boot_state_write(active)
        finally:
            _boot_lock_release(held)


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
    # sync on a sync KernelManager (warm) but a coroutine on an
    # AsyncKernelManager (fresh path); the process is already dead and there's no
    # live loop to await on here, so just close the coroutine to avoid a
    # "never awaited" warning.
    try:
        res = km.cleanup_resources()
        if asyncio.iscoroutine(res):
            res.close()
    except Exception:
        pass


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

    def __init__(self, kernel_name: str = "python3"):
        self.kernel_name = kernel_name
        self.km = None
        self.kc = None
        # All kernel I/O for this warm kernel runs on its own dedicated loop so
        # the async kernel client is never driven across event loops, which
        # can hang.
        self.loop, self.run_async = _make_async_runner()
        self._tests_since_boot = 0
        # Whether user_ns may still hold a previous test's names. Cleared by
        # whoever resets, so prepare_for_test can skip a reset that the last
        # teardown already did -- ~47ms per test, and it was every test.
        # Starts True so an unknown state is always cleared.
        self._ns_dirty = True

    def boot(self) -> None:
        """Start the kernel, on fresh ports each attempt.

        A kernel picks its ports before it binds them, so a kernel booting in
        another worker can take one in between; this one then dies with
        ``ZMQError: Address in use``. A new KernelManager draws new ports,
        which is what a retry needs -- the same manager would draw the same
        ones again.
        """
        from jupyter_client import KernelManager

        last_exc: Optional[BaseException] = None
        for _attempt in range(3):
            km = KernelManager(kernel_name=self.kernel_name)
            kc = None
            # Throttle holds a slot through wait-for-ready (the whole startup
            # window is where the resource pressure lives), so only _BOOT_CAP
            # kernels boot at once across all workers.
            with _boot_throttle():
                try:
                    km.start_kernel()
                    kc = km.client()
                    kc.start_channels()

                    async def _wait_ready(kc=kc):
                        await kc._async_wait_for_ready(timeout=30)

                    self.run_async(_wait_ready())
                except Exception as exc:  # retry ANY boot failure
                    last_exc = exc
                    try:
                        if kc is not None:
                            kc.stop_channels()
                    except Exception:
                        pass
                    _force_kill_kernel(km)
                    continue
            self.km = km
            self.kc = kc
            # A just-booted kernel has nothing to clear.
            self._ns_dirty = False
            return
        raise RuntimeError(f"kernel failed to boot after 3 attempts: {last_exc!r}") from last_exc

    def reboot(self) -> None:
        """Replace this kernel with a new one on fresh ports."""
        try:
            if self.kc:
                self.kc.stop_channels()
        except Exception:
            pass
        _force_kill_kernel(self.km)
        self.km = self.kc = None
        self.boot()
        self._tests_since_boot = 0

    def _is_alive(self) -> bool:
        try:
            return self.km is not None and self.kc is not None and bool(self.km.is_alive())
        except Exception:
            return False

    def _recycle_if_bloated(self) -> None:
        """Re-boot this kernel once it has grown too large to be safe to keep.

        A warm kernel never forgets. `get_ipython().reset()` clears the user
        namespace but not the allocator's arenas, not pandas/sklearn/matplotlib
        module state, and not whatever the C extensions hold -- so RSS only
        climbs. Chunking used to hide this by starting a fresh pytest (and so a
        fresh kernel) every 60 files; running all 838 in one invocation does
        not, and 16 kernels each growing for ~260 tests is what fills 64 GB.

        The failure mode is not a slow test, it is a CLIFF: measured on one
        unchunked run, ALL 16 workers hit the 300s stall watchdog within four
        seconds of each other, every one blocked in `select` waiting on a
        kernel that had stopped answering, and the run took 22:47 instead of
        6:00 while the watchdog killed and restarted them.

        Recycling on RSS rather than on a test count targets the actual
        resource: a worker running cheap tests keeps its kernel indefinitely,
        while one that just loaded sklearn twice gets a fresh one. The count is
        a backstop for when RSS cannot be read (psutil missing, permissions).
        A re-boot costs ~2.3s and, at these thresholds, happens a handful of
        times per worker per full run.
        """
        self._tests_since_boot += 1
        over = self._tests_since_boot >= _WARM_MAX_TESTS
        if not over:
            try:
                import psutil

                prov = getattr(self.km, "provisioner", None)
                pid = getattr(prov, "pid", None) or getattr(getattr(self.km, "kernel", None), "pid", None)
                if pid:
                    rss = psutil.Process(pid).memory_info().rss
                    over = rss >= _WARM_MAX_RSS_BYTES
            except Exception:
                pass  # fall back to the count backstop
        if not over:
            return
        self.reboot()
        # cash is not initialised here: the rest of prepare_for_test, which
        # called us, does exactly that (reset_session + %cash_on) and works the
        # same on a freshly booted kernel as on a reused one.

    def _exec(self, code: str) -> None:
        self.run_async(self.kc._async_execute_interactive(code, store_history=False, output_hook=lambda msg: None))

    def prepare_for_test(self, work_dir: Path, nb_path: Path) -> None:
        """Reset the kernel so it behaves like a freshly booted one.

        Clears the user namespace, repoints cwd + notebook path at this test's
        tmp dir, rebuilds a fresh default cache backend at this cwd (so no prior
        test's cached entries produce spurious RESTORED hits), clears CashMagics'
        tracking dicts, and purges test-authored modules from sys.modules (so a
        prior test's same-named module can't shadow this test's). Then ensures
        cash is loaded + enabled (idempotent).
        """
        path_str = str(nb_path).replace("\\", "\\\\")
        dir_str = str(work_dir).replace("\\", "\\\\")
        # A kernel that died since the last test -- a restart whose relaunch
        # lost its ports to another worker's kernel -- would answer nothing,
        # and every exec below would wait out its 120 s timeout, in this test
        # and in each of its reruns. Replace it instead.
        if not self._is_alive():
            self.reboot()
        # Clear the namespace in its OWN execution. reset() rebinds user_ns to a
        # fresh dict; if we assigned __vsc_ipynb_file__ in the same cell the
        # assignment would land in the old (captured) globals dict and be
        # invisible afterwards, breaking cash's notebook-path discovery.
        # `@cash:no-cache` because cash processes these setup execs like any
        # cell, and without it the harness's own bookkeeping is written into
        # the cache under test -- observed as `get_ipython().reset(...)` and
        # `_c.reset_session()` entries in the REPO-ROOT .cash after a full run
        # (the warm kernel's boot cwd, which is where the backend still points
        # when these run).
        # Clear the namespace -- unless shutdown() already did it for this
        # kernel. The two resets were redundant and each costs ~47ms, but only
        # ONE of them is disposable, and it is this one. Dropping the teardown
        # reset instead looked identical on paper and broke three module-reload
        # tests (in modules/test_helper_module_edits.py):
        # they pass alone and fail behind any other test, so something the
        # previous test leaves in the warm kernel has to be cleared at teardown
        # and not merely before the next test runs. Turning cash off first did
        # not help, so it is not about cash processing the reset. Do not
        # re-attempt without a reproducer -- two files, `-n0`, ~20s.
        if self._ns_dirty:
            self._exec("# @cash:no-cache\nget_ipython().reset(new_session=False)")
            self._ns_dirty = False
        # Recycle AFTER the reset, not before, so the RSS it reads is what this
        # kernel has PERMANENTLY accumulated rather than that plus whatever the
        # last test happened to leave live. Reading it first would inflate every
        # sample and buy spurious re-boots at ~2.3s each.
        #
        # If this DOES recycle, the reset above (when it ran at all) is wasted
        # on a kernel about to die -- one exec, only on the handful of tests
        # that trip the threshold.
        self._recycle_if_bloated()
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
        # Put cash's logger back as found. `%cash_debug on` (41 callers via
        # `enable_debug()`) sets the level on the KERNEL's logger, and a fresh
        # kernel forgets it when the process dies -- a warm one does not. The
        # leak is loud but easy to misread: every later test's cells fill with
        # [cash.notebook...] DEBUG lines, which breaks any assertion about cell
        # output and any out-of-band read that scans stdout for a marker.
        #
        # The next five steps are ONE exec, not five. They are independent of
        # each other -- no ordering constraint, unlike the reset / chdir /
        # reset_session / %cash_on sequence around them -- and each kernel
        # round-trip is ~8ms of real work in two processes, so five of them is
        # ~32ms on EVERY test. Per-step failure isolation is kept by wrapping
        # each step in its own try/except (the three code blocks already carry
        # one, and the magics are called through run_line_magic so they can be
        # wrapped the same way); merging them must not turn one broken step
        # into four skipped ones.
        self._exec(
            "try:\n"
            "    get_ipython().run_line_magic('cash_debug', 'off')\n"
            "except Exception:\n"
            "    pass\n"
            "try:\n"
            "    get_ipython().run_line_magic('cash_persist', 'off')\n"
            "except Exception:\n"
            "    pass\n" + _REUSE_RESET_WARNING_REGISTRIES + _REUSE_CLOSE_FIGURES + _REUSE_PURGE_TEST_MODULES
        )
        # Same shape, different flag: `%cash_persist on` sets `persist_all` on
        # the Cash instance's config, and `reset_session()` does NOT rebuild
        # the CashMagics holding that instance -- IPython keeps the object
        # already in its magics registry, so register_magic() rebinds the
        # functions but the flag rides along.
        #
        # Measured, peeking right after prepare_for_test::
        #
        #     baseline        magics=False processor=False
        #     after a test    magics=True  processor=True
        #     that enabled it
        #
        # persist bypasses the cost floors, so the leak makes later tests cache
        # statements that are far too cheap to cache -- which is exactly what
        # loops/test_persist_on_a_growing_loop.py measures. One
        # persist-enabling test ahead of it turned its 3 passes into 3
        # failures, reproduced in 9s.
        #
        # Forget which warnings have already been shown.
        #
        # `warnings.warn` under the default filter fires once per unique
        # (text, category, module, lineno) and records that in the calling
        # module's `__warningregistry__` -- process state a fresh kernel starts
        # empty and a warm one does not. cash warns the user exactly once about
        # several conditions, so on a warm kernel only the FIRST test to
        # provoke a given warning sees it and every later one reads silence.
        #
        # Measured: loops/test_persist_on_a_growing_loop.py's three
        # tests each provoke the same persist-amplification warning. The first
        # passed and the other two failed on "cache was bounded but the user
        # was never told why" -- the guard had worked, only the warning was
        # missing.
        #
        # And purge test-authored modules from sys.modules so a stale
        # same-named module from a prior test can't shadow this test's import.
        #
        # Re-enable auto-caching LAST (a fresh Cash starts with caching off).
        #
        # The order is load-bearing, not cosmetic. Every `self._exec` here is a
        # real kernel execution, so once cash is ON it processes these setup
        # cells like any other cell -- and `__vsc_ipynb_file__` already points
        # at THIS test's notebook. cash then treats the setup cell as one that
        # sits below the notebook's cells, reconstructs the upstream state it
        # thinks is missing, and RUNS those cells' statements for their side
        # effects, before the test has executed a single cell.
        #
        # Measured with an `os.write` tick stamped with the execution counter:
        # a three-cell notebook read "1,3," under reuse against "3," on a fresh
        # kernel, and the tick file was already non-empty when start_kernel()
        # returned. Every "cell executed N+1 times" failure in the reuse arm
        # traced back to this one extra execution. Keeping `%cash_on` last
        # makes prepare_for_test end exactly where _init_cash does on the
        # fresh-boot path: cash enabled, nothing executed after it.
        self._exec("%cash_on")
        # From here the test owns the namespace; assume it dirties it.
        self._ns_dirty = True

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
#   3. warning registries: see _REUSE_RESET_WARNING_REGISTRIES below.

# Clear every module's warning registry, plus cash's own once-flags, so each
# reused test starts as ignorant of past warnings as a freshly booted kernel.
# `warnings.warn` under the default filter fires once per unique (text,
# category, module, lineno) and records that in the calling module's
# `__warningregistry__` -- so on a warm kernel only the first test to provoke a
# given cash warning ever sees it. `_filters_mutated()` bumps the version stamp
# those registries are validated against, which is what makes already-cached
# "already warned" entries stale; clearing the dicts alone is not always enough.
_REUSE_RESET_WARNING_REGISTRIES = """
try:
    import sys as _sys, warnings as _warnings
    for _mod in list(_sys.modules.values()):
        _reg = getattr(_mod, '__warningregistry__', None)
        if _reg:
            try:
                _reg.clear()
            except Exception:
                pass
    try:
        _warnings._filters_mutated()
    except Exception:
        pass
    # cash keeps its own warn-once flags outside the warnings machinery.
    try:
        from cash.notebook import server_discovery as _sd
        _sd._warned_notebook_not_found = False
    except Exception:
        pass
except Exception:
    pass
"""

# Close every open matplotlib figure. pyplot keeps its figures in a
# PROCESS-GLOBAL registry (`_pylab_helpers.Gcf`), which `get_ipython().reset()`
# knows nothing about, so on a warm kernel each test inherits every figure the
# previous ones left open and the inline backend flushes them ALL into the next
# plotting cell's output.
#
# Measured: test_upstream_plot_not_leaked_into_downstream asserted its plot cell
# had 1 image and found 4. It only surfaced once the whole suite ran in ONE
# pytest invocation -- chunking had been spreading matplotlib tests across
# different workers, so few enough shared a kernel to stay under the threshold.
#
# Looked up in sys.modules rather than imported: importing pyplot into every
# reused kernel would cost far more than it saves, and a kernel that never
# imported it has no figures to close.
_REUSE_CLOSE_FIGURES = """
try:
    import sys as _sys
    _plt = _sys.modules.get('matplotlib.pyplot')
    if _plt is not None:
        _plt.close('all')
except Exception:
    pass
"""

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
_warm_kernel: Optional["_WarmKernel"] = None


def _get_warm_kernel(kernel_name: str = "python3") -> "_WarmKernel":
    global _warm_kernel
    if _warm_kernel is None:
        wk = _WarmKernel(kernel_name)
        wk.boot()
        atexit.register(wk.shutdown)
        _warm_kernel = wk
    return _warm_kernel
