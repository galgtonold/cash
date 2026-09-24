"""NotebookTestRunner: runs a real notebook file on a real kernel, cell by cell."""

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import nbformat
from nbclient import NotebookClient

from tests._nbharness.kernels import (
    _REUSE_KERNEL,
    _boot_throttle,
    _close_async_runner,
    _force_kill_kernel,
    _get_warm_kernel,
    _make_async_runner,
    _WarmKernel,
)

# Pin cash's cost thresholds so a caching DECISION stops depending on how busy
# the machine is. Paste into a test's SETUP CELL -- a normal cell, not an
# out-of-band exec, because cash reconstructs state around an execution it
# cannot find among the notebook's cells and that perturbs the very decisions
# this is meant to stabilise (measured: seeding a loop-split verdict that way
# left the correct verdict in the store and the loop still re-ran 124/124).
#
# WHY the threshold and not the workload: these decisions are wall-clock
# measurements, and the interesting cases sit just BELOW a ceiling (a "cheap"
# body, a call under the cost floor) while descheduling only pushes a
# measurement UP. A test can therefore never buy more than ~2x headroom by
# changing what it runs -- this body is ~3.2ms against a 6ms ceiling. Moving
# the threshold instead buys as much as you like: at a 1s ceiling the same
# body has 300x margin and no realistic stall crosses it.
#
# Use ONLY where the decision is a precondition of what the test asserts. A
# test whose SUBJECT is the threshold (test_an_expensive_body_is_never_split)
# must keep the real one.
# Badge vocabulary, as asserted by the integration tests.
CASH_TEST_PIN_THRESHOLDS = (
    "cash.configure(call_cost_floor_seconds=0.0, "
    "min_execution_time_to_cache_seconds=0.0, "
    "loop_split_max_iter_seconds=1.0, "
    "loop_split_min_remaining_seconds=0.0)\n"
)


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
    # (kind, text) so adjacent stream chunks can be rejoined without inserting a
    # separator that was never in the output.
    parts: list[tuple[str, str]] = []
    for output in cell.get("outputs", []):
        if output.output_type == "stream":
            parts.append(("stream", output.text))
        elif output.output_type in ("execute_result", "display_data"):
            data = output.get("data", {})
            if "text/plain" in data:
                val = data["text/plain"]
                if "<IPython.core.display.HTML object>" not in val:
                    parts.append(("value", val))

    # A stream output is a slice of a BYTE STREAM, not a line: its own text
    # already carries whatever newlines the code printed. Two consecutive
    # stream outputs are two halves of one stream and must be concatenated,
    # while a separate execute_result / display_data is its own block and gets
    # a newline before it.
    #
    # Joining everything with "\n" was a latent flake. The kernel usually
    # coalesces a cell's stdout into ONE stream output, so it read correctly --
    # but under load it can flush mid-print, and then `print('p', p)` arrived
    # as 'p' + ' 50\n' and was reassembled as "p\n 50". The value was right and
    # the assertion still failed. Measured on one 58-file chunk run repeatedly:
    # 4 failures in 16 randomized runs across three unrelated tests, all of
    # which assert on printed text, against 0 in 6 runs with a fresh kernel per
    # test -- fresh kernels hid it by being slow enough never to split.
    buf: list[str] = []
    for i, (kind, text) in enumerate(parts):
        if i and not (kind == "stream" and parts[i - 1][0] == "stream"):
            buf.append("\n")
        buf.append(text)
    raw_text = "".join(buf)

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
        debug_markers = [
            "[cash.",
            "[TIMING",
            "[UPSTREAM",
            "[LINEAGE",
            "[ALREADY",
            "[CACHE",
            "[CONTROL",
            "_DEBUG]",
            "[PROXY_CELL_ID]",
            "[CELL_CHANGED]",
            "[CELL_UNCHANGED]",
            "[CELL_ID]",
            "[STATE]",
            "[ENSURE_STATE",
            "[SKIP_",
            "Cash:",
            "cache_key: stmt:",
            "| source_hash:",
        ]

        # cash's own WARNINGS, which reach stderr and therefore land in the
        # cell's outputs. They have to go for the same reason the debug lines
        # do, and more sharply for the oracle harnesses: those compare a
        # cash-ON kernel's output against a cash-OFF one, and a cash warning
        # is by construction present in the first and impossible in the
        # second -- so any warning turns a correct run into a failure. It has
        # to be load-dependent to show up as a flake, and the write-failure
        # warning is exactly that.
        #
        # Matched on the CATEGORY name rather than the message. The message
        # texts are inconsistent about case ('Cash: the disk cache evicted'
        # against 'cash could not store the result'), so the existing 'Cash:'
        # marker catches some and misses others, while a bare 'cash' would
        # drop legitimate user output the way a bare '[DEBUG]' would.
        cash_warning_markers = [
            "CashWarning:",
            "CashCacheIneffectiveWarning:",
            "CashUpstreamSyntaxWarning:",
            "CashCacheStoreFailedWarning:",
            "CashImpurityWarning:",
        ]

        filtered_lines = []
        drop_continuation = False
        for line in raw_text.split("\n"):
            # Python prints a warning as two lines: the location + category +
            # message, then the offending source line, indented. Dropping only
            # the first leaves a stray '  warnings.warn(' behind, which
            # diverges from the oracle just as loudly.
            if drop_continuation:
                drop_continuation = False
                if line[:1] in (" ", "\t") and line.strip():
                    continue
            if any(marker in line for marker in debug_markers):
                continue
            if any(marker in line for marker in cash_warning_markers):
                drop_continuation = True
                continue
            filtered_lines.append(line)
        return "\n".join(filtered_lines).strip()

    return raw_text.strip()


# =============================================================================
# NOTEBOOK TEST RUNNER
# =============================================================================

REFERENCE_NOTEBOOKS_DIR = Path(__file__).parent.parent / "test_notebook_integration" / "reference_notebooks"


class NotebookTestRunner:
    """
    A test runner for notebooks that supports selective cell execution.

    Key design principles:
    1. Uses REAL notebook files - cash reads the file naturally, no mocking
    2. Copies notebooks to work_dir so modifications don't affect originals
    3. Modifies cells by rewriting the file - cash sees the changes

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
        kernel_name: str = "python3",
        timeout: int = 120,
    ):
        self.work_dir = Path(work_dir)
        self.kernel_name = kernel_name
        self.timeout = timeout

        self.nb: Optional[nbformat.NotebookNode] = None
        self.nb_path: Optional[Path] = None
        self.client: Optional[NotebookClient] = None
        self._kernel_started = False
        self._cash_initialized = False
        self._force_fresh_kernel = False
        self._warm: Optional["_WarmKernel"] = None
        # Each runner gets its own event loop to avoid cross-test contamination.
        # Created lazily in start_kernel() (and torn down in shutdown()) so a
        # runner that is never started - or one using the shared warm loop -
        # doesn't create and leak a loop + thread.
        self._loop = None
        self._run_async = None

    # Whether start_kernel() injected __vsc_ipynb_file__; restart() mirrors it so
    # a no-path run stays a no-path run across a restart.
    _inject_path: bool = True

    def load(self, notebook_path: Union[str, Path]) -> "NotebookTestRunner":
        """
        Load a notebook by copying it to the work directory.
        """
        src_path = Path(notebook_path)
        if not src_path.exists():
            raise FileNotFoundError(f"Notebook not found: {src_path}")

        name_hash = hashlib.md5(str(src_path).encode()).hexdigest()[:8]
        self.nb_path = self.work_dir / f"test_{src_path.stem}_{name_hash}.ipynb"
        shutil.copy2(src_path, self.nb_path)

        with open(self.nb_path, "r", encoding="utf-8") as f:
            self.nb = nbformat.read(f, as_version=4)

        return self

    def create_notebook(self, cells: List[str]) -> "NotebookTestRunner":
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
        with open(self.nb_path, "w", encoding="utf-8") as f:
            nbformat.write(self.nb, f)

    def start_kernel(
        self,
        with_cash: bool = True,
        inject_notebook_path: bool = True,
    ) -> "NotebookTestRunner":
        """
        Start the kernel and optionally initialize cash.

        Args:
            with_cash: install cash + ``%cash_on``. NOTE this records INTENT, not
                outcome — warm-kernel reuse can hand back a kernel that already
                has cash installed. If your test's
                conclusion depends on an arm really being cash-off, call
                :meth:`assert_cash_active` rather than trusting this flag: a
                cash-ON "control" reports the same warm-count as a genuine hit.
            inject_notebook_path: define ``__vsc_ipynb_file__`` so cash can
                resolve the notebook. Defaults True because upstream tracking
                needs it — but that default is also a BLIND SPOT: real
                papermill / nbconvert runs have no such variable, and because the
                suite always injected it, cash's no-path branch was never
                exercised and shipped an uncaught IndexError that disabled
                caching and printed an internal error on every cell.
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
            resources={"metadata": {"path": str(self.work_dir)}},
        )

        # Warm-kernel reuse (opt-in). Only for the common with_cash=True path;
        # with_cash=False tests want a bare kernel with no cash hooks installed,
        # which a persistent warm kernel can't provide, so they fall through to a
        # fresh boot below. Likewise skipped when the caller asked for NO path
        # injection: prepare_for_test() always defines __vsc_ipynb_file__, which
        # would silently defeat the no-path environment the test is trying to
        # reproduce.
        if _REUSE_KERNEL and with_cash and inject_notebook_path and not self._force_fresh_kernel:
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
            # RE-READ km/kc: prepare_for_test may have recycled a bloated
            # kernel, in which case the handles captured above are dead and
            # every call this test makes would hit a killed process.
            self.client.km = wk.km
            self.client.kc = wk.kc
            self._cash_initialized = True
            return self

        # Fresh-boot path (no reuse): this runner drives its own kernel I/O, so
        # lazily create its dedicated loop now (closed again in shutdown()).
        if self._loop is None:
            self._loop, self._run_async = _make_async_runner()

        self._start_new_kernel()

        self._kernel_started = True

        # Inject notebook path so cash can find the notebook file
        # This is required for upstream detection to work
        if inject_notebook_path:
            self._inject_notebook_path()

        if with_cash:
            self._init_cash()
        else:
            self._force_cash_off()

        return self

    def _force_cash_off(self) -> None:
        """Guarantee a ``with_cash=False`` kernel really has caching disabled.

        A fresh kernel is not necessarily a bare one. ``cash autoload`` installs
        an IPython **startup file** (``~/.ipython/profile_default/startup/
        00-cash.py``) that imports cash and runs ``%cash_on`` in every session
        on that machine — so on a developer box with autoload enabled, every
        "cash off" control arm in this suite is actually cash ON, reporting the
        same warm-count as a genuine cache hit and silently proving nothing.

        The suite must not depend on the developer having left a product feature
        switched off, so the control is asserted rather than assumed. Best
        effort: if the magics were never registered there is nothing to turn
        off, which is the state we wanted anyway.
        """
        snippet = "try:\n    get_ipython().run_line_magic('cash_off', '')\nexcept Exception:\n    pass\n"
        try:
            self._run_async(
                self.client.kc._async_execute_interactive(
                    snippet,
                    store_history=False,
                    output_hook=lambda _m: None,
                )
            )
        except Exception:  # noqa: BLE001 - a control that can't be forced off
            pass  # is caught by assert_cash_active, not hidden here

    def _start_new_kernel(self) -> None:
        """Start a new kernel, retrying a flaky boot.

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
        raise RuntimeError(f"kernel failed to boot after 3 attempts: {last_exc!r}") from last_exc

    def probe_cash_active(self) -> bool:
        """Whether cash auto-caching is ACTUALLY live in the kernel right now.

        OBSERVED by asking the kernel, never inferred from the ``with_cash``
        argument. That distinction is the whole point: ``start_kernel`` records
        intent, but warm-kernel reuse can hand back a
        kernel with cash already installed, so a test that *asked* for
        ``with_cash=False`` may still be running cash-ON. When that happens the
        control arm reports ``warm == 0`` -- byte-identical to a genuine cache
        hit -- so the comparison silently proves nothing. Assert on this in any
        test whose conclusion depends on an arm really being cash-off.
        """
        # Read ``_auto_cache_enabled`` off the registered CashMagics instance,
        # which ``ip.register_magics()`` files under its class name.
        #
        # This used to look for ``ip._cash_magics_instance`` -- an attribute
        # cash never sets -- so the precise limb was dead code and the probe
        # ALWAYS fell through to "are the cash magics registered?". That is a
        # different question with a different answer: registration happens when
        # a ``Cash()`` is constructed (``register_magic`` defaults True), so
        # merely importing cash makes a genuinely cash-OFF kernel report ON.
        # The hook is installed but inert while ``_auto_cache_enabled`` is
        # False, so nothing is being cached and the arm really is a control.
        #
        # There is deliberately no fallback to the registration signal. Falling
        # back is what produced the false positive, and a guard that answers a
        # question it wasn't asked is worse than one that says "off".
        probe = (
            "try:\n"
            "    _ip = get_ipython()\n"
            "    _reg = getattr(_ip.magics_manager, 'registry', {})\n"
            "    _mag = _reg.get('CashMagics')\n"
            "    _on = bool(getattr(_mag, '_auto_cache_enabled', False))\n"
            "except Exception:\n"
            "    _on = False\n"
            "print('CASH_ACTIVE=' + ('1' if _on else '0'))"
        )
        seen = []

        def _hook(msg):
            if msg["msg_type"] == "stream":
                seen.append(msg["content"].get("text", ""))

        self._run_async(
            self.client.kc._async_execute_interactive(
                probe,
                store_history=False,
                output_hook=_hook,
            )
        )
        return "CASH_ACTIVE=1" in "".join(seen)

    def assert_cash_active(self, expected: bool) -> "NotebookTestRunner":
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

    def restart(self) -> "NotebookTestRunner":
        """Restart the kernel in place, preserving the runner's wiring.

        Restart behaviour is a whole class of bug the suite was blind to:
        nb_runner shipped no restart, so every test that needed one
        hand-rolled ``km._async_restart_kernel`` -- 9 copies across the suite,
        each free to get the re-injection wrong. Re-injects the notebook path
        afterwards ONLY if this runner was started with injection, so a
        no-path run stays a no-path run across the restart.
        """
        try:
            self._run_async(self.client.km._async_restart_kernel(now=True))
            self._run_async(self.client.kc._async_wait_for_ready(timeout=30))
        except Exception:  # noqa: BLE001 - replaced below, whatever the failure
            self._replace_kernel()
        self._restore_working_directory()
        if self._inject_path:
            self._inject_notebook_path()
        return self

    def _replace_kernel(self) -> None:
        """Stand in a new kernel, on fresh ports, for one that did not come back.

        A restart relaunches the kernel on the ports it had. Another worker's
        kernel can take one of them while it is down, and the relaunched
        kernel then dies with ``ZMQError: Address in use``. A new kernel is
        what a restart gives anyway, so one started on new ports is an equal
        stand-in.
        """
        if self._warm is not None:
            self._warm.reboot()
            self.client.km = self._warm.km
            self.client.kc = self._warm.kc
            return
        try:
            if self.client.kc is not None:
                self.client.kc.stop_channels()
        except Exception:
            pass
        _force_kill_kernel(self.client.km)
        self._start_new_kernel()

    def _restore_working_directory(self) -> None:
        """Re-enter ``work_dir`` after a restart.

        A freshly-booted runner's kernel PROCESS is launched with
        ``cwd=work_dir``, so restarting it lands back there for free. A WARM
        kernel (``CASH_TEST_REUSE_KERNEL=1``) is launched once at the repo root
        and moved into each test's directory IN-PROCESS by
        ``_WarmKernel.prepare_for_test``. A restart discards that, and the
        kernel silently resumes at the repo root -- so the test's ``.cash``
        directory, and every relative path it touches, land outside its tmp
        directory and are shared with every other test on that worker.

        Measured before this was added, same probe, both modes::

            default   before: .../pytest-.../test_x0   after: .../pytest-.../test_x0
            reuse     before: .../pytest-.../test_x0   after: <repo root>

        The symptom is not a crash. Restart-dependent tests fail their own
        non-vacuity guards ("nothing was served from cache after the restart")
        because they are no longer reading the cache they wrote, which reads as
        a cash bug rather than a harness one.

        Unconditional rather than gated on the warm path: it is a no-op for a
        fresh kernel (already there) and keeps the two modes' post-restart
        state identical, which is the property the suite actually depends on.
        """
        dir_str = str(self.work_dir).replace("\\", "\\\\")
        self._run_async(
            self.client.kc._async_execute_interactive(
                f"import os as _os; _os.chdir(r'{dir_str}')",
                store_history=False,
                output_hook=lambda msg: None,
            )
        )
        if not _REUSE_KERNEL:
            return
        # Re-point the cache directory, which the chdir above does NOT move.
        #
        # `config.cache_dir` defaults to the RELATIVE `".cash"`, and
        # `FileBackend.__init__` resolves it with `os.path.abspath` -- so an
        # entry's location is fixed by the cwd at BACKEND CONSTRUCTION, and a
        # later chdir cannot move it. Cash's auto-load rebuilds that backend
        # while the kernel is still starting, before any code here can run.
        #
        # A fresh kernel's PROCESS is launched with `cwd=work_dir`, so that
        # startup construction already lands in the right place. The warm
        # kernel's process is launched once at the repo root, so post-restart
        # it pins to `<repo>/.cash` -- shared with every other test on this
        # worker, and gitignored, so the leak is invisible. Traced, one test,
        # both modes::
        #
        #     default  cwd=<tmp> -> <tmp>/.cash     (x2, both correct)
        #     reuse    cwd=<repo> -> <repo>/.cash   boot
        #              cwd=<tmp>  -> <tmp>/.cash    prepare_for_test, correct
        #              cwd=<repo> -> <repo>/.cash   AFTER RESTART, wrong
        #
        # `reset_session()` drops the singleton so the next access rebuilds
        # against the cwd we just set -- the same fix `prepare_for_test`
        # already applies, which is why its construction is the correct one.
        # Gated on the warm path: a fresh kernel is already right, and dropping
        # its singleton would discard state a test may be mid-way through.
        self._run_async(
            self.client.kc._async_execute_interactive(
                "# @cash:no-cache\nimport cash as _c; _c.reset_session()",
                store_history=False,
                output_hook=lambda msg: None,
            )
        )

    def _inject_notebook_path(self) -> None:
        """Inject the notebook path into the kernel namespace.

        This allows cash's get_notebook_path() to find the notebook file,
        which is required for upstream detection to work correctly.
        """
        if self.nb_path is None:
            return

        # Use the same variable name that VS Code uses
        # This is checked first in get_notebook_path()
        path_str = str(self.nb_path).replace("\\", "\\\\")
        inject_code = f"__vsc_ipynb_file__ = r'{path_str}'"

        reply = self._run_async(self.client.kc._async_execute_interactive(inject_code, store_history=False))
        if reply["content"]["status"] != "ok":
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
        if reply["content"]["status"] != "ok":
            error_name = reply["content"].get("ename", "Unknown")
            error_value = reply["content"].get("evalue", "")
            raise RuntimeError(f"Failed to initialize cash: {error_name}: {error_value}")
        self._cash_initialized = True

    def set_cell_source(self, cell_num: int, source: str) -> "NotebookTestRunner":
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

    def add_cell(self, source: str, save: bool = False) -> "NotebookTestRunner":
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

    def run_cell(self, cell_num: int) -> "NotebookTestRunner":
        """Execute a single cell (1-based indexing)."""
        if not self._kernel_started:
            raise RuntimeError("Kernel not started. Call start_kernel() first.")

        idx = cell_num - 1
        cell = self.nb.cells[idx]
        cell.outputs = []

        self._run_async(self.client.async_execute_cell(cell, idx))

        return self

    def run_cells(self, cell_nums: List[int]) -> "NotebookTestRunner":
        """Execute multiple cells in order."""
        for num in cell_nums:
            self.run_cell(num)
        return self

    def run_all(self) -> "NotebookTestRunner":
        """Execute all cells in the notebook."""
        return self.run_cells(list(range(1, len(self.nb.cells) + 1)))

    def get_output(self, cell_num: int, filter_debug: bool = True) -> str:
        """Get the text output from a cell."""
        idx = cell_num - 1
        return get_text_output(self.nb.cells[idx], filter_debug=filter_debug)

    def get_raw_output(self, cell_num: int) -> str:
        """Get the raw output from a cell (no filtering)."""
        return self.get_output(cell_num, filter_debug=False)

    def peek(self, expr: str) -> str:
        """Evaluate *expr* in the live kernel and return its ``repr``.

        Runs outside the notebook's cells with ``store_history=False``, so
        nothing about it is cached, replayed, or added to the notebook.

        **Use this when the claim is about kernel state; use ``get_output``
        when the claim is about what the user sees.** Both are legitimate;
        conflating them is the bug. A cached statement's stdout is REPLAYED on
        a hit, so reading state through a printed cell reports what was on
        screen when the entry was written, not what the variable holds now.
        Measured once: ``print('C', compute_c(1), CALLS_C)`` reported
        ``[1]`` after a restart while the live value was ``[]``. The printed
        reading made a broken arm look correct and sent one round of that
        investigation down a false trail.

        A bare name is wrapped as ``globals().get(name)`` automatically. That
        is not politeness: a bare *undefined* name raises, which produces no
        stdout, and "no output" would then read as a value rather than as a
        lookup failure. Pass any non-identifier expression (``len(CALLS)``,
        ``obj.attr``) and it is evaluated as written -- there the caller has
        chosen to risk the raise.

        Every output line is scanned for the marker rather than matching the
        joined text, because a badge or a debug line shares the stdout channel
        and would otherwise silently read as "no value".

        Returns the marker-less ``repr`` string, or ``"?"`` when no marker was
        found at all (a kernel error, or output that never arrived).

        Note: depends on ``kc._async_execute_interactive``, private nbclient
        API. An nbclient bump is what would break this.
        """
        if expr.isidentifier():
            expr = f"globals().get({expr!r})"

        seen: List[str] = []

        def _hook(msg):
            if msg["msg_type"] == "stream" and msg["content"].get("name") == "stdout":
                seen.append(msg["content"]["text"])

        self._run_async(
            self.client.kc._async_execute_interactive(
                f"print('__CASH_PEEK__', repr({expr}))",
                store_history=False,
                output_hook=_hook,
            )
        )
        for line in "".join(seen).splitlines():
            if "__CASH_PEEK__" in line:
                return line.split("__CASH_PEEK__", 1)[1].strip()
        return "?"

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
        self._run_async(self.client.kc._async_execute_interactive(status_code, store_history=False))

        # Get the result from the kernel
        get_result_code = """
import json as _json
print(_json.dumps(_cash_status_result, default=str))
"""
        self._run_async(
            self.client.kc._async_execute_interactive(
                get_result_code, store_history=False, output_hook=lambda msg: None
            )
        )

        # Extract result from iopub messages
        try:
            # Find the stream output
            for msg in self.client.kc.iopub_channel.get_msgs():
                if msg["msg_type"] == "stream" and msg["content"].get("name") == "stdout":
                    return json.loads(msg["content"]["text"].strip())
        except Exception:
            pass

        return {}

    def get_cell(self, cell_num: int):
        """Get the cell object."""
        return self.nb.cells[cell_num - 1]

    def cell_count(self) -> int:
        """Return the number of cells."""
        return len(self.nb.cells)

    def reset_cash_state(self) -> "NotebookTestRunner":
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
        _cash_magics.tracking_state.lineage.clear()
        _cash_magics.tracking_state.executed_cell_codes.clear()
        _cash_magics.tracking_state.executed_cell_hashes.clear()
        _cash_magics.tracking_state.current_session_hashes.clear()
        _cash_magics.tracking_state.executed_file_deps.clear()
        # Clear statement processor's input lineages
        if hasattr(_cash_magics, '_statement_processor'):
            _cash_magics._statement_processor.tracking_state.executed_input_lineages.clear()
        # Clear upstream checker's simulation cache
        if hasattr(_cash_magics, '_upstream_checker'):
            _cash_magics._upstream_checker.simulator.cache.reset()
        # Clear file tracker state
        if hasattr(_cash_magics, '_file_tracker') and _cash_magics._file_tracker:
            _cash_magics._file_tracker.clear()
except Exception:
    pass
"""
        # Run directly via kernel client to avoid overwriting notebook cells
        self._run_async(self.client.kc._async_execute_interactive(reset_code, store_history=False))
        return self

    def enable_debug(self) -> "NotebookTestRunner":
        """Enable cash debug output."""
        # Run directly via kernel client to avoid overwriting notebook cells
        self._run_async(self.client.kc._async_execute_interactive("%cash_debug on", store_history=False))
        return self

    def enable_persist(self) -> "NotebookTestRunner":
        """Enable cash 'persist everything' mode.

        Bypasses the cost-aware floors (the 10 ms 'too cheap to cache' floor
        and the size-aware skip) so that even sub-millisecond statements are
        written to the cache and can produce a cache HIT on a subsequent
        identical run.  Use this when a test needs to assert a cache hit on a
        statement that is otherwise too cheap to cache by default.
        """
        # Run directly via kernel client to avoid overwriting notebook cells
        self._run_async(self.client.kc._async_execute_interactive("%cash_persist on", store_history=False))
        return self

    def shutdown(self) -> None:
        """Shutdown the kernel."""
        if self._warm is not None:
            # Reuse mode: leave the warm kernel alive for the next test, but
            # clear its namespace HERE. This reset is load-bearing and its
            # timing is part of that -- see prepare_for_test, which used to run
            # a second, identical one and now skips it while `_ns_dirty` says
            # this one already happened.
            try:
                self._warm._exec("get_ipython().reset(new_session=False)")
                self._warm._ns_dirty = False
            except Exception:
                pass
            self._warm = None
            self._kernel_started = False
            self.client = None
            # Disown the warm kernel's loop. It belongs to _WarmKernel, not to
            # this runner, and leaving it here is a use-after-free waiting to
            # happen: a later start_kernel() on this same runner that does NOT
            # take the warm path (with_cash=False, or no path injection) skips
            # creating its own loop because `self._loop is None` is False, and
            # runs its fresh kernel on the WARM kernel's loop. Its shutdown()
            # then takes the kill branch below and calls _close_async_runner()
            # on it -- closing the shared loop, so every later test on this
            # worker dies with "RuntimeError: Event loop is closed".
            #
            # `calls/test_call_caching_overhead_on_a_numeric_loop.py` does exactly this: a cash-on arm followed by a
            # cash-off arm in one test, and it was the first casualty of a
            # 138-failure cascade under CASH_TEST_REUSE_KERNEL=1.
            #
            # Cleared rather than closed -- the warm kernel is still using it.
            self._loop = None
            self._run_async = None
            # A test that shuts down and starts again MID-TEST is asserting
            # something about a kernel that really died. Handing it the same
            # warm kernel back makes "shutdown" a lie: prepare_for_test resets
            # the namespace and the cash singleton, but the process -- and
            # whatever cash state is not reachable from that singleton --
            # carries straight through, and on into every LATER test.
            #
            # Measured: test_restart_persist_then_edit_upstream_recomputes does
            # exactly this, and behind it the last two tests of
            # loops/test_persist_on_a_growing_loop.py failed with every
            # statement RESTORED from a cache they never wrote -- so the
            # amplification guard never engaged and its warning never fired.
            # The sibling test that restarts WITHOUT persist did not poison
            # anything, and forcing a fresh kernel here makes both pass.
            #
            # Costs one boot for the handful of tests with this shape, and only
            # after they have already had a warm one.
            self._force_fresh_kernel = True
            return
        if self.client:
            # COVERAGE EXPERIMENT (env-gated, no-op in normal runs):
            # ipykernel exits via os._exit(), which skips the atexit hook that
            # coverage.process_startup() relies on to flush. So we explicitly
            # save the kernel's coverage data while the channels are still
            # alive, before the kill.
            if os.environ.get("COVERAGE_PROCESS_START") and self.client.kc:

                async def _save_cov():
                    save_code = (
                        "import coverage as _cov\n_c = _cov.Coverage.current()\nif _c is not None:\n    _c.save()\n"
                    )
                    await self.client.kc._async_execute_interactive(
                        save_code,
                        store_history=False,
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

    def __enter__(self) -> "NotebookTestRunner":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
