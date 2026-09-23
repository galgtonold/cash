"""IPython magic commands for transparent cell caching in Jupyter notebooks."""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import time
import weakref

# Any is used at IPython API boundaries where types come from the shell's dynamic
# namespace (user_ns, execution info objects).  These cannot be typed more precisely
# without declaring a hard IPython dependency in production code.
from collections.abc import Iterator
from typing import Any

from IPython.core.magic import Magics, line_magic, magics_class

from ... import _log
from ..._console import safe_text
from ...backends._writes import all_pending_writes
from ...core import Cash
from ...object_hashing import compute_hash
from ...tracking import io_watch
from ...tracking.function_tracker import FunctionTracker
from .. import compute_baselines
from .._protocols import ShellProtocol, TrackingState
from ..cache_status import CacheStatus
from ..control_structures import ControlStructureProcessor
from ..live_cells import install_expiry_hook, register_target
from ..live_cells import reset as _reset_live_cells
from ..module_invalidator import ModuleInvalidator
from ..provenance import ProvenanceTracker
from ..restore import Restorer
from ..server_discovery import (
    extract_notebook_path_from_vscode_cell_id,
    in_colab,
    invalidate_notebook_path_cache,
    labextension_installed,
    set_notebook_path,
)
from ..statement import ProcessResult, StatementProcessor

# The SAME floor reader the cache-write decision uses. The cacheable/trivial
# split in %cash_stats is only honest if "worth caching" means exactly what the
# cache meant by it, so this deliberately shares the reader rather than
# re-deriving the threshold here.
from ..statement.capture import replay_outputs
from ..statement.store import config_float
from ..upstream import UpstreamChecker
from ._args import strip_inline_comment
from ._help import help_text
from ._types import CellMetrics
from .admin import CashAdminMagicsMixin
from .badges import BadgePresenter
from .cell_executor import (
    CellExecutor,
    EarlyReturn,
    PipelineCompleted,
    PipelineSyntaxError,
    discarded_writes_notification,
)

__all__ = ["CashMagics"]

logger = logging.getLogger(__name__)


def new_session_stats() -> dict[str, Any]:
    """A zeroed session-stats dict.

    The single definition of what the stats ARE, so that creating a session and
    resetting one cannot disagree. ``%cash_stats reset`` used to re-list the
    keys by hand, which silently left any later-added counter carrying over the
    reset -- the reset would report success while the next session inherited
    the last one's numbers. Same rule as ``measured_compute``: a
    reset must forget everything the stats claim to summarise.
    """
    return {
        "cells_executed": 0,
        "statements_computed": 0,
        "statements_restored": 0,
        "statements_skipped": 0,
        "total_compute_time": 0.0,
        "total_restored_time": 0.0,
        # GROSS avoided recompute. Every contribution is a ``saved_time``
        # copied off cache metadata — i.e. how long the statement took when
        # it was FIRST computed, on a possibly colder machine. It is an
        # estimate of a counterfactual, never a measurement of this
        # session, and it may overstate without bound.
        "total_time_saved": 0.0,
        # The subset of ``total_time_saved`` whose baseline this session
        # measured itself: the statement was COMPUTED here before it was
        # RESTORED here, so the recompute cost is known under today's
        # conditions rather than assumed from the cache.
        "total_verified_saved": 0.0,
        # The subset whose baseline was measured on this machine in an
        # EARLIER kernel (``compute_baselines``, the least cost ever
        # measured). A Restart & Run All recomputes nothing, so without
        # this the headline net after a restart was "at least -overhead,
        # at best <gross>" -- a range straddling zero in the one reading
        # every user takes.
        "total_measured_saved": 0.0,
        # Cash's OWN added wall-time this session (restore + simulation +
        # hashing + badge machinery), accumulated per cell. Subtracted from
        # the gross ``total_time_saved`` to report an honest NET saving so a
        # session whose overhead outweighs its cache hits reads as a cost,
        # not a phantom win.
        "total_overhead": 0.0,
        # The hit rate over ALL statements is dominated by print/import
        # trivia that cash deliberately never tried to cache, so it made a
        # session where every expensive statement hit read as 14.9% —
        # arithmetically true, practically meaningless. These two
        # count only statements whose compute cost cleared cash's OWN
        # caching floor (``min_execution_time_to_cache_seconds``), i.e. the
        # statements caching was ever on the table for.
        "statements_cacheable_hit": 0,
        "statements_cacheable_miss": 0,
    }


class CashSession:
    """Groups session-level concerns owned by a single CashMagics instance.

    Separating these from execution-level state (backend, shell, tracking
    dictionaries) makes the sub-boundary explicit and each component
    independently addressable.
    """

    __slots__ = ("stats", "provenance", "measured_compute", "measured_decorator_compute", "baselines")

    def __init__(self) -> None:
        self.stats: dict[str, Any] = new_session_stats()
        self.provenance: ProvenanceTracker = ProvenanceTracker()
        # source -> execution_time measured in THIS session. Bounded by the
        # notebook's statement count; pure in-memory floats, no I/O.
        self.measured_compute: dict[str, float] = {}
        # decorator cache_key -> compute time measured in THIS session (a miss).
        # The @cash.cache sibling of measured_compute: it lets a later decorator
        # HIT be credited as VERIFIED under the same rule.
        self.measured_decorator_compute: dict[str, float] = {}
        # The same two measurements, kept on disk beside the cache so the next
        # kernel can still point at one. Bound to a real directory on first
        # use (``CashMagics._baselines``), not here: the backend is not
        # settled while the magics are being constructed.
        self.baselines: Any = compute_baselines.get_store(None)


def _is_silent(args: tuple, kwargs: dict) -> bool:
    """Whether ``run_cell(raw_cell, store_history, silent, ...)`` was asked to be silent.

    A frontend sends ``silent=True`` for code of its own -- a variable
    explorer's query, a coverage or profiling hook -- never for a cell the
    user runs, and Jupyter's protocol has it run "as quietly as possible".
    Put through cash, such code was treated as a cell below the notebook:
    its names were given lineage (a test tool's ``_cov`` and ``_cov_mod``,
    deleted by the same code, stayed in it) and its statements were cached
    and planned against the notebook's cells. It runs as plain IPython.
    """
    if "silent" in kwargs:
        return bool(kwargs["silent"])
    return len(args) >= 2 and bool(args[1])


@magics_class
class CashMagics(CashAdminMagicsMixin, Magics):
    def __init__(self, shell: ShellProtocol, cash_instance: Cash) -> None:
        """Initialise CashMagics in three phases (ordering matters):

        1. **State setup** — mode flags, tracking dicts, convenience aliases.
        2. **Processing components** — upstream checker, statement processor,
           control structure processor, module invalidator (requires state
           from phase 1).
        3. **Session state** — badge throttle, cell tracking, event hooks
           (requires components from phase 2).
        """
        super().__init__(shell)
        self._cash_instance = cash_instance
        # Re-entrancy guard for the run_cell_async wrapper. IPython's *sync*
        # ``run_cell`` delegates to ``run_cell_async`` internally (via the
        # pseudo-sync runner), so once we patch ``run_cell_async`` it would
        # otherwise fire a SECOND time for every sync cell — double-running
        # cash's pre/post work. This flag is raised for the dynamic extent of
        # the sync ``_execute_cell`` so the nested async call knows to just
        # delegate without repeating cash's pipeline.
        self._in_sync_cell = False
        self._debug = cash_instance.debug

        # Auto-caching mode state
        self._auto_cache_enabled = False
        # Releases the I/O observation scope held while %cash_on is on (see
        # `cash_on`); also run if this instance goes away still holding it.
        self._io_release: weakref.finalize | None = None
        self.global_ttl = None

        # The cell's badge; ``%cash_badge`` sets its mode.
        self.badges = BadgePresenter(shell, cash_instance)

        # Shared tracking state — single owner of all lineage/dependency dicts
        self.tracking_state = TrackingState()

        self._init_processing_components(shell, cash_instance)
        self._init_session_state(shell)

    def _init_processing_components(self, shell: ShellProtocol, cash_instance: Cash) -> None:
        """Create upstream checker, statement processor, control structure processor,
        and module invalidator.

        Wires shared tracking state and function tracker so all components
        use the same lineage dictionaries and source hashes.
        """
        # One function tracker for both, so the upstream simulation computes
        # cache keys with the same func_source_hashes as the statement processor.
        function_tracker = FunctionTracker()
        self._statement_processor = StatementProcessor(
            shell,
            cash_instance,
            compute_hash_fn=compute_hash,
            tracking_state=self.tracking_state,
            function_tracker=function_tracker,
        )

        self._upstream_checker = UpstreamChecker(
            shell,
            cash_instance=cash_instance,
            compute_hash_fn=compute_hash,
            tracking_state=self.tracking_state,
            function_tracker=function_tracker,
        )

        self._control_structure_processor = ControlStructureProcessor(
            shell,
            self._statement_processor,
        )

        self._module_invalidator = ModuleInvalidator(shell)

        self._restorer = Restorer(
            shell,
            backend=cash_instance.backend,
            tracking_state=self.tracking_state,
        )

        self._cell_executor = CellExecutor(
            shell,
            cash_instance=cash_instance,
            badges=self.badges,
            tracking_state=self.tracking_state,
            statement_processor=self._statement_processor,
            upstream_checker=self._upstream_checker,
            restorer=self._restorer,
            module_invalidator=self._module_invalidator,
            control_structure_processor=self._control_structure_processor,
        )

    def _init_session_state(self, shell: ShellProtocol) -> None:
        """Initialise cell ID tracking, session stats, and event hooks."""
        # Whether %cash_on has shown its save-the-notebook tip this session.
        self._save_hint_shown = False
        # How many discarded cache writes the badge has already reported.
        self._discarded_writes_seen = 0

        # Cell ID tracking (available since IPython 8.3)
        self.current_cell_id = None

        # Last cell execution metrics (for %cash_status)
        self._last_cell_metrics: CellMetrics = {
            "statements": [],
            "total_time": 0.0,
            "total_restored_time": 0.0,
            "total_computed_time": 0.0,
            "upstream_metrics": [],
            "status": None,
        }

        # Session-level concerns (statistics, provenance) grouped in one object
        self._session = CashSession()

        # When CashMagics is re-instantiated in a still-running kernel
        # (cash.reset_session(), a second Cash(), or %load_ext after a reset),
        # un-patch the previous instance's hooks FIRST. Otherwise we'd capture an
        # already-wrapped run_cell as our "original" (nesting wrappers on every
        # reset) and stack duplicate pre_run_cell handlers. The true-original
        # run_cell and the prior handler are stashed on the shell for this.
        prior = getattr(shell, "_cash_hooks", None)
        if isinstance(prior, dict):
            try:
                shell.events.unregister("pre_run_cell", prior["capture_cell_id"])
            except (ValueError, KeyError, AttributeError, TypeError):
                pass
            try:
                shell.run_cell = prior["original_run_cell"]
            except (KeyError, AttributeError):
                pass
            # Restore the async entry point too, so a reset_session /
            # second-Cash re-patch captures the true-original run_cell_async
            # rather than nesting our wrapper on every reset.
            if "original_run_cell_async" in prior:
                try:
                    shell.run_cell_async = prior["original_run_cell_async"]
                except (KeyError, AttributeError):
                    pass

        # Durability checkpoint. Registered on IPython's own event
        # rather than inside CellExecutor: the pipeline has several exit paths
        # and a drain placed after its last phase turned out to run for only
        # one cell in three, missing precisely the cells that do the caching.
        # post_run_cell fires for every cell however it finished.
        if isinstance(prior, dict) and prior.get("flush_pending_writes") is not None:
            try:
                shell.events.unregister("post_run_cell", prior["flush_pending_writes"])
            except (ValueError, KeyError, AttributeError, TypeError):
                pass
        try:
            shell.events.register("post_run_cell", self._flush_pending_writes)
        except (AttributeError, TypeError) as e:
            logger.warning(
                "Could not register post_run_cell handler: %s. Cached results "
                "will still be written, but a kernel killed (rather than shut "
                "down) may lose writes that were still queued.",
                e,
            )

        # Register event handler to capture cell_id before execution
        try:
            shell.events.register("pre_run_cell", self._capture_cell_id)
        except (AttributeError, TypeError) as e:
            logger.warning(
                "Could not register pre_run_cell event handler: %s. "
                "Cell ID tracking will be disabled — upstream change detection "
                "and VS Code cell-level caching may not work correctly.",
                e,
            )

        # JupyterLab live-cell push: register the comm target
        # that receives cell sources pushed by cash's frontend extension, when
        # one is present. A silent no-op everywhere else — register_target()
        # returns False rather than raising when there is no kernel / comm
        # manager to attach to (bare IPython, older ipykernel, MockShell, ...).
        register_target(shell)
        # ...and retire each pushed snapshot when the execution it arrived for
        # ends. The store outlives the frontend that fills it, so without this a
        # frontend that stops pushing without re-opening (a reload onto the same
        # kernel with the extension disabled, a second client attached without
        # it, the extension erroring after having worked once) leaves cash
        # serving one frozen snapshot for the rest of the kernel's life -- and
        # suppressing the notice that would have said so. See ``expire``.
        #
        # Registered here rather than beside _flush_pending_writes above because
        # it belongs to the comm, not to the cache: both halves of the live-cell
        # contract are then visible in one place, and neither is conditional on
        # %cash_on having run.
        install_expiry_hook(shell)

        # Monkey-patch run_cell to intercept execution.
        #
        # Both hooks are installed as ``functools.wraps``-ed proxies rather than
        # as the bound methods directly. That is load-bearing, not cosmetic:
        # ipykernel introspects these signatures to decide what to pass us
        # (ipkernel.py: ``_accepts_parameters(run_cell, ["cell_id"])``), and its
        # helper treats a ``**kwargs`` signature as "accepts every parameter".
        # Our proxies are ``(*args, **kwargs)``, so bare they would claim to
        # accept ``cell_id`` even against an IPython too old to have it (<8.3,
        # which ``[notebook]``'s ``ipython>=8.0`` floor still allows) — ipykernel
        # would then pass ``cell_id=...``, our forward would raise TypeError
        # before ``execute_reply`` was sent, and the cell would hang at ``[*]``.
        # ``functools.wraps`` sets ``__wrapped__``, which ``inspect.signature``
        # follows, so introspection sees the *original's* signature and every
        # verdict about us is identical to the verdict about the shell we
        # replaced.
        self._original_run_cell = shell.run_cell
        shell.run_cell = self._signature_preserving_proxy(
            self._original_run_cell,
            "_execute_cell",
        )

        # Also intercept run_cell_async: ipykernel dispatches top-level-await
        # cells (``x = await f()``) through ``shell.run_cell_async``, NOT the
        # sync ``run_cell`` we patch above, so without this they would bypass
        # cash's pipeline entirely (no upstream reconstruction, no self-mod
        # reset). Guarded because older IPython lacks run_cell_async.
        self._original_run_cell_async = None
        if hasattr(shell, "run_cell_async"):
            self._original_run_cell_async = shell.run_cell_async
            shell.run_cell_async = self._signature_preserving_proxy(
                self._original_run_cell_async,
                "_execute_cell_async",
                is_async=True,
            )

        try:
            shell._cash_hooks = {
                "original_run_cell": self._original_run_cell,
                "capture_cell_id": self._capture_cell_id,
                "flush_pending_writes": self._flush_pending_writes,
            }
            if self._original_run_cell_async is not None:
                shell._cash_hooks["original_run_cell_async"] = self._original_run_cell_async
        except (AttributeError, TypeError):
            pass

    def _signature_preserving_proxy(
        self,
        original: Any,
        handler_name: str,
        is_async: bool = False,
    ) -> Any:
        """Wrap *original* with a proxy that dispatches to ``self.<handler_name>``.

        The proxy forwards ``*args, **kwargs`` verbatim but, thanks to
        ``functools.wraps``, presents *original*'s signature to
        ``inspect.signature`` (via ``__wrapped__``). Callers that introspect
        these hooks to decide what to pass — ipykernel does exactly this for
        ``cell_id`` — therefore get the same answer they would have got from the
        unpatched shell, so we can never be handed an argument the real callee
        rejects.

        The handler is resolved by name **at call time** rather than captured, so
        tests can swap ``self._execute_cell`` out and still be routed through.
        """
        if is_async:

            @functools.wraps(original)
            async def proxy(*args: Any, **kwargs: Any) -> Any:
                return await getattr(self, handler_name)(*args, **kwargs)
        else:

            @functools.wraps(original)
            def proxy(*args: Any, **kwargs: Any) -> Any:
                return getattr(self, handler_name)(*args, **kwargs)

        return proxy

    @line_magic
    def cash_on(self, line: str) -> None:
        """Enable automatic caching for all subsequent cells.

        Usage:
            %cash_on              - enable, entries never expire
            %cash_on ttl=3600     - enable, entries expire after an hour
        """
        # Parse optional TTL. Comment-stripped first, or `%cash_on ttl=3600  #
        # one hour` would parse "3600  # one hour" as an int, fail, and silently
        # leave caching OFF on the `return` below.
        ttl = None
        arg = strip_inline_comment(line)
        if arg:
            parts = arg.split("=")
            if len(parts) == 2 and parts[0].strip() == "ttl":
                try:
                    ttl = int(parts[1].strip())
                except ValueError:
                    print(f"[Error] %cash_on: invalid TTL value: {parts[1].strip()!r}. Caching NOT enabled.")
                    return
            else:
                print(f"[Error] %cash_on: unrecognised argument: {arg!r}. Caching NOT enabled.")
                print("   Valid forms: %cash_on | %cash_on ttl=<seconds>")
                return

        # CASH_DISABLE / disable=True switches off BOTH paths: a CI job that
        # executes notebooks with it set means "run everything, cache nothing".
        # An autoload hook calls this in every kernel, so it must say why it
        # did nothing rather than fail quietly.
        if getattr(getattr(self._cash_instance, "config", None), "disable", False):
            print(
                "[cash] caching is disabled (disable=True / CASH_DISABLE), so %cash_on did nothing: cells run uncached."
            )
            return

        # Invalidate notebook path cache so we re-discover the current notebook
        # (fixes Issue 23: switching notebooks within the same kernel session)
        invalidate_notebook_path_cache()

        # Clear upstream checker's simulation and AST caches to prevent stale
        # data from a previous notebook from interfering (Issue 23 part 2)
        self._upstream_checker.reset_caches()

        # Drop any cell snapshot cash's JupyterLab extension pushed for the
        # PREVIOUS notebook -- otherwise the new notebook's first upstream
        # check would read stale cells from a document this session no longer
        # even has open, exactly the per-notebook staleness reset above.
        _reset_live_cells()

        self._auto_cache_enabled = True
        # Keep cash's I/O watch installed between statements rather than
        # installing and removing it around each one. A `from json import load`
        # run in a cell binds whatever is installed at that moment.
        if self._io_release is None or not self._io_release.alive:
            io_watch.hold()
            self._io_release = weakref.finalize(self, io_watch.release)
        self.global_ttl = ttl
        ttl_msg = f" (TTL: {ttl}s)" if ttl is not None else ""
        # ASCII, like the text badge: this line lands in the .ipynb and is read
        # back by nbconvert / a headless agent, whose console may be cp1252.
        print(safe_text(f"Cash enabled.{ttl_msg} Your computations will be cached automatically."))
        print("   Run %cash_help for available commands.")
        # Report existing cache state if available. Counted, not listed: a
        # listing reads every entry's metadata, and every %cash_on paid that --
        # re-running a notebook's first cell took 22.7 s.
        try:
            count = self._cash_instance.backend.entry_count()
            if count:
                print(f"   Found existing cache with {count} entries.")
        except (OSError, AttributeError, TypeError):
            pass
        # One-time hint: with no live reader, cash reads upstream cells from the
        # .ipynb file on disk, so an upstream edit that has not been saved is
        # invisible until it is — hence the save advice. Skipped wherever a LIVE
        # reader exists, because there the advice is not merely redundant, it is
        # FALSE: in Colab the cells come from the frontend (``get_ipynb``), and
        # on JupyterLab cash's own extension pushes the live sources over a comm
        # before every execution (see ``cash.notebook.live_cells``). Telling a
        # user of the feature they just installed that their unsaved edit is
        # invisible contradicts the thing they can watch working.
        #
        # The extension gate is a filesystem probe, not a look at the pushed
        # store: at %cash_on time no comm has opened yet for anyone, so the
        # store cannot distinguish absent from not-yet. See
        # ``labextension_installed`` for the one topology it answers wrongly
        # (a split install) and why suppressing-by-omission is the safe error.
        if not self._save_hint_shown:
            self._save_hint_shown = True

            if not in_colab() and not labextension_installed():
                print("[Tip] Cash reads upstream cells from the saved notebook file.")
                print("   Save (Ctrl+S) after editing a cell you are not about to run:")
                print("   an unsaved edit is invisible, so the upstream check skips it")
                print("   and you get the previous answer for the new code.")
                print('   JupyterLab autosaves on a timer; VS Code: "files.autoSave".')
        logger.debug("TTL: %s", ttl)

    @line_magic
    def cash_off(self, line: str) -> None:
        """Disable automatic caching.

        Usage:
            %cash_off
        """
        self._auto_cache_enabled = False
        if self._io_release is not None:
            self._io_release()  # releases once, however often %cash_off runs
        self.global_ttl = None
        print("[OK] Auto-caching disabled")

    @line_magic
    def cash_debug(self, line: str) -> None:
        """Toggle debug output on/off.

        Usage:
            %cash_debug on          - Enable debug logging
            %cash_debug off         - Disable debug logging
            %cash_debug json        - Enable JSON-formatted debug output
            %cash_debug file path   - Also log to file in JSON format
            %cash_debug             - Toggle between on and off

        Every mode goes through ``cash._log``, which records the handlers it
        adds: switching mode replaces them instead of stacking a second one,
        and ``off`` removes them.
        """
        mode, _, rest = strip_inline_comment(line).partition(" ")
        mode = mode.lower()
        # Only the mode is case-insensitive: a path keeps its case, and one
        # quoted to protect a space or a "#" loses the quotes.
        path = rest.strip()
        if len(path) >= 2 and path[0] == path[-1] and path[0] in "'\"":
            path = path[1:-1]
        if not mode:
            mode = "off" if self._debug else "on"

        if mode in ("on", "true", "1", "enable") and not path:
            _log.enable_console(logging.DEBUG)
            self._debug = True
            print("Cache debug output enabled.")
        elif mode in ("off", "false", "0", "disable") and not path:
            _log.disable()
            if self._cash_instance.verbose:
                _log.enable(logging.INFO)  # verbose=True's one line per call stays
            self._debug = False
            print("Cache debug output disabled.")
        elif mode == "json" and not path:
            _log.setup_logging(level=logging.DEBUG, json_output=True)
            self._debug = True
            print("Cache debug output enabled (JSON format).")
        elif mode == "file" and path:
            _log.setup_logging(level=logging.DEBUG, log_file=path)
            self._debug = True
            print(f"Cache debug output enabled (logging to {path}).")
        else:
            print(f"[Error] %cash_debug: unrecognised argument: {strip_inline_comment(line)!r}")
            print("   Valid forms: %cash_debug on | off | json | file <path> | (no argument to toggle)")
            return
        self._cash_instance.debug = self._debug

    @line_magic
    def cash_persist(self, line: str) -> None:
        """Toggle 'persist everything' mode on/off.

        When on, every statement is cached regardless of how cheap it was to
        compute - equivalent to putting ``# @cash:persist`` on every statement.
        Bypasses the cost-aware floors (the 10 ms 'too cheap to cache' floor and
        the size-aware skip). Useful for reproducibility, benchmarks, and
        debugging cache behavior; wasteful for trivial statements in normal use.

        Usage:
            %cash_persist on       - cache every statement
            %cash_persist off      - restore the default cost-aware policy
            %cash_persist          - toggle
        """
        # A bare %cash_persist toggles, so an unparsed argument doesn't merely
        # get ignored here — it inverts the request. ``%cash_persist on  #
        # comment`` used to turn persistence OFF if it was already on.
        mode = strip_inline_comment(line).lower()
        if mode in ("on", "true", "1", "enable"):
            persist_all = True
        elif mode in ("off", "false", "0", "disable"):
            persist_all = False
        elif mode:
            print(f"[Error] %cash_persist: unrecognised argument: {mode!r}")
            print("   Valid forms: %cash_persist on | off | (no argument to toggle)")
            return
        else:
            persist_all = not self._statement_processor.persist_all
        # Config is the one place the flag lives; the statement pipeline reads
        # it at each statement, as it does after cash.configure(persist_all=...).
        self._cash_instance.reconfigure(persist_all=persist_all)
        print(f"Cash persist-everything mode: {'enabled' if persist_all else 'disabled'}.")

    @line_magic
    def cash_help(self, line: str) -> None:
        """Print the list of Cash magics, or one magic's full usage.

        Usage:
            %cash_help              - every magic with its one-line summary
            %cash_help badge        - full usage of %cash_badge (the cash_ prefix is optional)
        """
        print(help_text(self, strip_inline_comment(line)))

    @line_magic
    def cash_badge(self, line: str) -> None:
        """Set the badge display mode.

        Usage:
            %cash_badge html   - Interactive HTML badges with live updates (default)
            %cash_badge print  - Text summary printed once after cell completes
            %cash_badge off    - No badge output at all

        Badge status icons:
            [C] COMPUTED - the statement ran (cache miss)
            [R] RESTORED - the result was loaded from the cache (cache hit)
            [S] SKIPPED  - unchanged since the last run, no work needed
        """
        mode = strip_inline_comment(line).lower()
        if mode in ("html", "print", "off"):
            self.badges.mode = mode
            print(f"Badge mode set to: {mode}")
        else:
            if mode:
                print(f"[Error] %cash_badge: unrecognised argument: {mode!r} (mode unchanged)")
            print(f"Current badge mode: {self.badges.mode}")
            print("Usage: %cash_badge html|print|off")

    @line_magic
    def cash_status(self, line: str) -> dict[str, Any] | None:
        """Get machine-readable status of the last cell execution.

        Usage:
            %cash_status          # Print JSON status
            %cash_status json     # Return JSON string
            %cash_status dict     # Return Python dict

        Returns a dict/JSON with:
            - statements: List of statement metrics (status, code, outputs, times)
            - total_time: Total execution time
            - total_restored_time: Time saved by cache hits
            - total_computed_time: Time spent computing
            - upstream_metrics: Metrics from upstream re-executions
            - status: Overall status (COMPUTED, RESTORED, SKIPPED, MIXED)
            - lineage: Current variable lineage state
            - cache_stats: {"keys": number of entries in the backend}
        """

        mode = strip_inline_comment(line).lower() or "print"

        # Build comprehensive status
        status = {
            "last_cell": self._last_cell_metrics.copy(),
            "lineage": dict(self.tracking_state.variable_lineage),
            "executed_codes": {
                k: v[:50] + "..." if len(v) > 50 else v for k, v in self.tracking_state.executed_cell_codes.items()
            },
            "auto_cache_enabled": self._auto_cache_enabled,
            "debug_enabled": self._debug,
        }

        # Counted, not listed: listing reads every entry's metadata, which on a
        # file cache of a few thousand entries takes seconds (see entry_count).
        try:
            status["cache_stats"] = {"keys": self._cash_instance.backend.entry_count()}
        except (AttributeError, TypeError, OSError) as exc:
            logger.debug("Failed to retrieve cache stats: %s", exc)
            status["cache_stats"] = {}

        if mode == "dict":
            return status
        if mode == "json":
            return json.dumps(status, default=str, indent=2)
        # Print formatted output
        print(json.dumps(status, default=str, indent=2))
        return status

    @staticmethod
    def cell_id_from_parent_metadata(shell: Any) -> str | None:
        """Return cell_id from IPython parent-header metadata, or None.

        Checks the two locations VS Code and other frontends use.
        """
        if not hasattr(shell, "get_parent"):
            return None
        parent = shell.get_parent()
        if not parent:
            return None
        metadata = parent.get("metadata", {})
        if "cellId" in metadata:
            return metadata["cellId"]
        if "vscode" in metadata and "cellId" in metadata["vscode"]:
            return metadata["vscode"]["cellId"]
        return None

    @staticmethod
    def maybe_seed_notebook_path(cell_id: str | None) -> None:
        """If cell_id is a VS Code URI, seed the notebook-path cache from it."""
        if not cell_id:
            return

        nb_path = extract_notebook_path_from_vscode_cell_id(cell_id)
        if nb_path:
            set_notebook_path(nb_path)

    def resolve_cell_id(self) -> str | None:
        """The running cell's id, from the kernel's parent-header metadata.

        Resolved before a cell runs, so the upstream check knows which cell it
        is looking at. As a side effect a VS Code cell id seeds the
        notebook-path cache, which the upstream check reads the notebook
        through. Remembered as :attr:`current_cell_id`, which is left as it
        was if the metadata cannot be read.
        """
        try:
            cell_id = self.cell_id_from_parent_metadata(self.shell)
            self.current_cell_id = cell_id
            self.maybe_seed_notebook_path(cell_id)
            if cell_id:
                logger.debug("[PROXY_CELL_ID] Captured cell_id early: %s", cell_id)
            else:
                logger.debug("[PROXY_CELL_ID] No cell_id in parent metadata")
        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            logger.debug("[PROXY_CELL_ID] Could not capture cell_id early: %s", e)
        return self.current_cell_id

    def _flush_pending_writes(self, result: Any = None) -> None:
        """Make this cell's cached results durable, on ``post_run_cell``.

        Cache writes are asynchronous. Nothing drains the queue when the kernel
        is *killed* rather than shut down — a crash, an OOM, a force-quit, or a
        tool that terminates the process instead of asking it to exit. Anything
        still queued at that moment is lost: the badge reported the result as
        cached, and after the restart it is not there. A graceful shutdown does
        drain (measured), so this closes the violent paths only.

        Drains every live queue in the process, not just this instance's
        backend: a notebook routinely has more than one Cash — the ``%cash_on``
        instance plus any ``Cash(...)`` built in a cell — and decorator writes
        go to the latter.

        Hooked to IPython's event rather than to the end of ``CellExecutor``'s
        pipeline. The pipeline has several exit paths, and a drain placed after
        its final phase fired for only one cell in three, missing exactly the
        cells that do the caching.

        Never raises: a failed write is already reported by the backend, and a
        durability best-effort must not turn a working cell into an error.
        """
        try:
            for queue in all_pending_writes():
                queue.wait_all()
        except Exception:  # noqa: BLE001 — best-effort, must not break the cell
            logger.debug("Flushing pending cache writes failed", exc_info=True)

    def _capture_cell_id(self, info: Any) -> None:
        """Capture cell_id from IPython's pre_run_cell event.

        This is called by IPython before each cell execution.
        The cell_id is available since IPython 8.3.
        In VS Code, it might be in metadata.

        As a side-effect, if the cell_id is a VS Code URI we extract the
        notebook file path from it and seed the notebook-path cache so that
        upstream checking can find the notebook even when
        ``__vsc_ipynb_file__`` is not injected.
        """
        try:
            self.current_cell_id = None

            # 1. Try standard info.cell_id (JupyterLab / IPython 8.3+)
            if hasattr(info, "cell_id") and info.cell_id:
                self.current_cell_id = info.cell_id

            # 2. Try to get it from parent header metadata (VS Code / others)
            if not self.current_cell_id:
                self.current_cell_id = self.cell_id_from_parent_metadata(self.shell)

            # 3. Seed notebook-path cache from VS Code cell_id URI
            self.maybe_seed_notebook_path(self.current_cell_id)

            # Debug-level logging (not a raw print): when %cash_debug is on the
            # ``cash`` logger is at DEBUG with a console handler attached, so
            # these surface; otherwise they stay silent instead of printing on
            # every cell (the "No cell_id found" case fires constantly in
            # environments that don't supply a cell_id).
            if self.current_cell_id:
                logger.debug("[CELL_ID] Captured cell_id: %s", self.current_cell_id)
            else:
                logger.debug("[CELL_ID] No cell_id found in info or metadata")

        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            logger.debug("[CELL_ID] Could not capture cell_id: %s", e)
            self.current_cell_id = None

    def _execute_cell(self, raw_cell: str, *args: Any, **kwargs: Any) -> Any:
        """Proxy for ``interactiveshell.run_cell`` to implement caching when
        ``%cash_on`` is active.  Delegates to :meth:`CellExecutor.execute_cell`."""
        # Raise the re-entrancy guard for the whole sync execution: IPython's
        # ``run_cell`` delegates to ``run_cell_async`` internally, so our
        # patched ``run_cell_async`` would otherwise re-run cash's pipeline on
        # every sync cell (and on the ``"pass"`` delegation the finaliser
        # issues). The guard makes that nested async call a no-op passthrough.
        prev_in_sync = self._in_sync_cell
        self._in_sync_cell = True
        try:
            return self._execute_cell_inner(raw_cell, *args, **kwargs)
        finally:
            self._in_sync_cell = prev_in_sync

    def _execute_cell_inner(self, raw_cell: str, *args: Any, **kwargs: Any) -> Any:
        if not self._auto_cache_enabled or _is_silent(args, kwargs):
            return self._original_run_cell(raw_cell, *args, **kwargs)

        try:
            result = self._cell_executor.execute_cell(
                raw_cell,
                args,
                kwargs,
                original_run_cell=self._original_run_cell,
                ttl=self.global_ttl,
                cell_id=self.resolve_cell_id(),
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001 - intentionally broad: surfaces user code exceptions to IPython
            return self._synthesize_run_cell_raise(e, args, kwargs)

        if isinstance(result, EarlyReturn):
            return result.value
        if isinstance(result, PipelineSyntaxError):
            return self._original_run_cell(raw_cell, *args, **kwargs)

        return self._finalize_cell_execution(raw_cell, result, args, kwargs)

    async def _execute_cell_async(self, raw_cell: str, *args: Any, **kwargs: Any) -> Any:
        """Proxy for ``interactiveshell.run_cell_async`` (stage 2).

        ipykernel routes cells that contain top-level ``await`` (IPython
        autoawait) through ``run_cell_async``, not the sync ``run_cell`` that
        :meth:`_execute_cell` intercepts.  Without this wrapper those cells
        would skip cash's pipeline entirely — no upstream reconstruction, no
        self-modifying-reassignment reset, no lineage capture, **no caching**.

        **Execute-exactly-once contract.** Cash now owns per-statement
        execution for these cells: it runs each statement through
        :meth:`StatementProcessor.process_statement_async`, which compiles the
        unit under ``ast.PyCF_ALLOW_TOP_LEVEL_AWAIT`` and awaits the coroutine
        on IPython's live loop when the statement contains a top-level
        ``await``.  A cache *hit* returns before any coroutine is built, so an
        identical second run skips the await entirely.

        Because cash runs the statements itself, it must NOT also delegate the
        whole cell to ``_original_run_cell_async(raw_cell)`` — that would
        double-run every side effect.  Instead the finaliser delegates a no-op
        ``"pass"`` cell through the original ``run_cell_async`` so IPython fires
        ``pre_run_cell`` / ``post_run_cell`` exactly once, advances the
        execution count + history, and returns a real ``ExecutionResult``.
        """
        if self._original_run_cell_async is None:
            # Defensive: should not happen (we only patch when it exists).
            raise RuntimeError("run_cell_async wrapper invoked without an original")

        # Passthrough when there is nothing to do, or when this call is the
        # RE-ENTRANT one that IPython's sync ``run_cell`` makes internally
        # (guarded by ``_in_sync_cell``). Doing cash work here would
        # double-run the pipeline / double-fire events for the sync cell.
        if not self._auto_cache_enabled or self._in_sync_cell or _is_silent(args, kwargs):
            return await self._original_run_cell_async(raw_cell, *args, **kwargs)

        # Raise the re-entrancy guard for the whole async execution: the
        # finaliser's ``"pass"`` delegation and any upstream reconstruction must
        # not re-enter this wrapper.
        prev_in_sync = self._in_sync_cell
        self._in_sync_cell = True
        try:
            return await self._execute_cell_async_inner(raw_cell, *args, **kwargs)
        finally:
            self._in_sync_cell = prev_in_sync

    async def _execute_cell_async_inner(self, raw_cell: str, *args: Any, **kwargs: Any) -> Any:
        try:
            result = await self._cell_executor.execute_cell_async(
                raw_cell,
                args,
                kwargs,
                original_run_cell=None,
                ttl=self.global_ttl,
                cell_id=self.resolve_cell_id(),
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001 - surfaces user code exceptions to IPython
            return await self._synthesize_run_cell_raise_async(e, args, kwargs)

        if isinstance(result, EarlyReturn):
            return result.value
        if isinstance(result, PipelineSyntaxError):
            # The cell's own AST failed to parse — let IPython handle it (it
            # will render the SyntaxError) exactly once on its live loop.
            return await self._original_run_cell_async(raw_cell, *args, **kwargs)

        return await self._finalize_cell_execution_async(raw_cell, result, args, kwargs)

    def _substitute_cell_kwargs(self, source: str, kwargs: dict) -> dict:
        """Kwargs for delegating the stand-in cell *source* to ``run_cell_async``.

        ipykernel calls ``run_cell_async(code, transformed_cell=…,
        preprocessing_exc_tuple=…)``.  IPython runs ``transformed_cell`` and
        ignores ``raw_cell`` -- so delegating our bookkeeping cell (``"pass"`` /
        ``"raise __cash_exception__"``) with the caller's kwargs would re-run the
        WHOLE user cell a second time (double side effects).  We replace both
        with the transform of the stand-in cell itself.  Passing them rather
        than dropping them is required: from IPython 9.16 ``run_cell_async``
        raises ``TypeError`` without ``transformed_cell``, and every IPython
        cash supports (>= 8.0) accepts it.
        """
        return {
            **kwargs,
            "transformed_cell": self.shell.transform_cell(source),
            "preprocessing_exc_tuple": None,
        }

    async def _synthesize_run_cell_raise_async(
        self,
        e: BaseException,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        """:meth:`_synthesize_run_cell_raise` through the original ``run_cell_async``."""
        with self._raising_quietly(e):
            return await self._original_run_cell_async(
                "raise __cash_exception__",
                *args,
                **self._substitute_cell_kwargs("raise __cash_exception__", kwargs),
            )

    def _synthesize_run_cell_raise(
        self,
        e: BaseException,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        """Re-raise *e* through IPython's run_cell so the kernel reply status
        is "error" while suppressing IPython's duplicate traceback (the clean
        error display was already rendered by the executor's
        ``_finalize_error_badge``).

        Hook-path only: makes sense when ``_execute_cell`` is itself standing
        in for ``run_cell``.
        """
        with self._raising_quietly(e):
            return self._original_run_cell("raise __cash_exception__", *args, **kwargs)

    @contextlib.contextmanager
    def _raising_quietly(self, e: BaseException) -> Iterator[None]:
        """Bind *e* as ``__cash_exception__`` for a ``raise __cash_exception__``
        cell, with IPython's traceback display switched off for its duration."""
        self.shell.user_ns["__cash_exception__"] = e
        orig_showtb = getattr(self.shell, "showtraceback", None)
        try:
            self.shell.showtraceback = lambda *a, **kw: None
        except (AttributeError, TypeError):
            logger.debug("Could not suppress IPython showtraceback")
        try:
            yield
        finally:
            try:
                if orig_showtb is not None:
                    self.shell.showtraceback = orig_showtb
                else:
                    with contextlib.suppress(AttributeError, TypeError):
                        del self.shell.showtraceback
            except (AttributeError, TypeError):
                logger.debug("Could not restore showtraceback")

    def _finalize_cell_execution(
        self,
        raw_cell: str,
        done: PipelineCompleted,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        """Post-process a cell execution: flush analytics, record metrics, render final badge.

        Tail phase of ``_execute_cell`` (the hook-driven `%cash_on`). Handles
        session statistics updates, provenance recording, debug output, and
        the final badge render, then ends by calling
        ``self._original_run_cell("pass", *args, **kwargs)`` so IPython's
        internal bookkeeping (execution count, history) stays in sync.
        """
        self._finalize_cell_body(raw_cell, done)

        # Delegate to original run_cell with "pass" so IPython keeps its
        # execution count + history consistent.
        return self._original_run_cell("pass", *args, **kwargs)

    def _finalize_cell_body(self, raw_cell: str, done: PipelineCompleted) -> None:
        """Finaliser body shared by the sync and async tails.

        Everything the finaliser does *except* the ``"pass"`` delegation to
        IPython (which differs: sync ``_original_run_cell`` vs
        ``await _original_run_cell_async``).  Extracting it keeps the async
        top-level-await path from drifting on analytics, session stats,
        observability, buffered-output replay, and the final badge render.
        """
        all_metrics, timing_breakdown = done.all_metrics, done.timing_breakdown
        hook_total = time.time() - done.hook_start

        # Analytics events are intentionally NOT flushed here, per cell.
        # The AnalyticsManager buffers events and flushes on its own policy —
        # every ``_flush_threshold`` events, on any stats query, and via an
        # atexit hook on a clean shutdown.  Forcing a SQLite connect+commit on
        # *every* cell fsync'd the DB per cell and dominated per-cell wall time
        # (~12 ms/cell, measured), defeating the very batch buffer it was
        # draining. Trade-off: on a hard kernel kill the buffered analytics
        # events may be lost — acceptable because analytics is best-effort
        # observability, not correctness.

        # Post-execution: auto-track any newly imported local modules.
        # On the FIRST run of `import trackmod`, auto_track_local_imports couldn't
        # track it pre-execution because it wasn't in sys.modules yet.  Now that
        # statements have executed, the module IS in sys.modules and we can track
        # its file mtime so future runs detect source changes.
        ft = self._statement_processor.function_tracker
        try:
            ft.auto_track_local_imports(raw_cell)
        except (ImportError, AttributeError, OSError, TypeError) as exc:
            logger.debug("Post-execution auto-track failed: %s", exc)

        # Update last cell metrics for %cash_status
        self._update_last_cell_metrics(all_metrics, hook_total)

        # Update session-wide statistics
        self._update_session_stats(all_metrics, hook_total)

        self._record_provenance(all_metrics)

        logger.debug(
            "[TIMING_PROXY] Total %.1fms (badge init %.1fms, upstream check %.1fms, badge progress %.1fms)",
            hook_total * 1000,
            timing_breakdown.get("badge_init", 0) * 1000,
            timing_breakdown.get("upstream_check", 0) * 1000,
            done.badge_render_time * 1000,
        )

        # Show the buffered result (if any)
        replay_outputs(rich=done.buffered_outputs)

        # A write that failed and was thrown away is the one loss the rows
        # cannot express -- see discarded_writes_notification. Appended here,
        # after execution, because that is when writes fail; both the sync and
        # async finalizers funnel through this method, so one call covers both.
        #
        # Writes are asynchronous, so a failure can surface on the cell AFTER
        # the one that caused it. Reporting it late is strictly better than the
        # alternative, which was reporting it at kernel shutdown.
        row, self._discarded_writes_seen = discarded_writes_notification(self._discarded_writes_seen)
        if row is not None:
            all_metrics = list(all_metrics) + [row]

        # The cell's final badge on the success path, for the sync and the
        # async hook alike.
        self.badges.finish(all_metrics, done.badge_display_id, hook_total, timing_breakdown)

    async def _finalize_cell_execution_async(
        self,
        raw_cell: str,
        done: PipelineCompleted,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        """Async twin of :meth:`_finalize_cell_execution` for top-level-await cells.

        Runs the shared finaliser body, then delegates a no-op ``"pass"`` cell
        through the ORIGINAL ``run_cell_async`` — not the sync ``run_cell`` —
        so IPython fires ``pre_run_cell`` / ``post_run_cell`` exactly once,
        advances the execution count + history, and returns a real
        ``ExecutionResult`` on its own live loop.  The user's statements were
        already executed per-statement by the async pipeline, so this ``"pass"``
        adds no side effects (execute-exactly-once).
        """
        self._finalize_cell_body(raw_cell, done)
        # Replace ``transformed_cell`` so IPython runs our ``"pass"`` and NOT
        # the original user cell again (see _substitute_cell_kwargs).
        return await self._original_run_cell_async(
            "pass",
            *args,
            **self._substitute_cell_kwargs("pass", kwargs),
        )

    def _update_last_cell_metrics(self, all_metrics: list[ProcessResult], hook_total: float) -> None:
        """Compute and store ``_last_cell_metrics`` for ``%cash_status``."""
        statuses = [m.get("status") for m in all_metrics if m.get("status")]
        if all(s == CacheStatus.RESTORED for s in statuses) and statuses:
            overall_status = "RESTORED"
        elif all(s == CacheStatus.COMPUTED for s in statuses) and statuses:
            overall_status = "COMPUTED"
        elif all(s == CacheStatus.SKIPPED for s in statuses) and statuses:
            overall_status = "SKIPPED"
        elif statuses:
            overall_status = "MIXED"
        else:
            overall_status = None

        self._last_cell_metrics = {
            "statements": [
                {
                    "code": m.get("code", "")[:100],
                    "status": m.get("status"),
                    "execution_time": m.get("execution_time", 0.0),
                    "saved_time": m.get("saved_time", 0.0),
                    "outputs": m.get("restored_vars", []),
                    "is_upstream": m.get("is_upstream", False),
                }
                for m in all_metrics
            ],
            "total_time": hook_total,
            "total_restored_time": sum(m.get("saved_time", 0.0) for m in all_metrics),
            "total_computed_time": sum(
                m.get("execution_time", 0.0) for m in all_metrics if m.get("status") == CacheStatus.COMPUTED
            ),
            "upstream_metrics": [m for m in all_metrics if m.get("is_upstream", False)],
            "status": overall_status,
        }

    def _baselines(self):
        """Measurements from earlier kernels against this cache directory.

        Resolved on use, not in ``__init__``: a notebook's backend is not
        settled when the magics are constructed (``%cash_on`` may still
        replace it), and a store bound to "nowhere to persist" then would
        stay that way for the session -- silently reporting no measured
        saving, which is the bug this store exists to fix.
        """
        store = self._session.baselines
        if getattr(store, "_path", None) is None:
            resolved = compute_baselines.store_for_backend(getattr(self._cash_instance, "backend", None))
            if resolved is not None and getattr(resolved, "_path", None) is not None:
                self._session.baselines = resolved
                return resolved
        return store

    def _update_session_stats(self, all_metrics: list[ProcessResult], cell_total_time: float = 0.0) -> None:
        """Increment session-wide caching statistics from *all_metrics*.

        ``cell_total_time`` is this cell's full cash-mediated wall time
        (``hook_total``). Cash's own overhead for the cell is that wall time
        minus the user compute that would have run anyway (the COMPUTED
        statements). What remains — cache restores, upstream simulation,
        hashing, badge machinery — is time cash *added*, so it is accumulated
        into ``total_overhead`` and later subtracted from the gross
        ``total_time_saved`` to report an honest NET figure. This is
        a single float subtraction per cell, no I/O — it must never
        reintroduce the per-cell fsync that was removed earlier.

        Upstream COMPUTED statements count as user compute, NOT as overhead:
        they are the user's own notebook code, and the state they rebuild is
        state the user would have had to rebuild by hand (that is the restart
        pain cash exists to absorb). Booking them as cash's overhead would make
        cash understate itself by the size of the user's own ETL on exactly the
        sessions where it helps most — a mirror-image lie.

        The gross saving is credited from ``saved_time``, which is a *stale*
        baseline (see ``total_time_saved``). Restores whose baseline this
        session re-measured are additionally credited to
        ``total_verified_saved``, which is what ``%cash_stats`` reports as the
        headline NET — so an unverifiable claim can never print as a win.
        """
        stats = self._session.stats
        measured = self._session.measured_compute
        baselines = self._baselines()
        stats["cells_executed"] += 1
        cell_compute_time = 0.0
        # Cash's own "too cheap to cache" floor, so the cacheable/trivial split
        # below matches the decision the cache actually made rather than a
        # second opinion invented here.
        floor = config_float(
            getattr(self._cash_instance, "config", None),
            "min_execution_time_to_cache_seconds",
            0.01,
        )
        for m in all_metrics:
            status = m.get("status")
            if status == CacheStatus.COMPUTED:
                stats["statements_computed"] += 1
                # What the USER's code cost: the statement's wall time less
                # cash's own time inside it -- recording file reads, keying and
                # hashing the arguments of the calls it routes, storing them
                # (``cash_tax``, the same measurement ``CallRouting.statement_cost``
                # uses). Counting that as the user's compute cancelled it out
                # of the overhead below, so a paired run measured 370 s of
                # slowdown where %cash_stats reported 210 s.
                exec_time = max(0.0, m.get("execution_time", 0.0) - m.get("cash_tax", 0.0))
                stats["total_compute_time"] += exec_time
                cell_compute_time += exec_time
                code = m.get("code")
                if code:
                    measured[code] = exec_time
                    # Kept on disk too, so tomorrow's kernel can still point at
                    # a measurement of what this costs.
                    baselines.record(code, exec_time)
                # Measured today: a real miss on a statement worth caching.
                if exec_time >= floor:
                    stats["statements_cacheable_miss"] += 1
            elif status == CacheStatus.RESTORED:
                stats["statements_restored"] += 1
                saved = m.get("saved_time", 0.0)
                stats["total_restored_time"] += saved
                stats["total_time_saved"] += saved
                # ``saved`` is the cache's stale baseline, so it is NOT trusted
                # for time — it is used only to answer "was this the
                # kind of statement caching was for?". A hit is a fact either
                # way; only the denominator's membership rests on the baseline.
                if saved >= floor:
                    stats["statements_cacheable_hit"] += 1
                # Credit a VERIFIED saving only where this session computed the
                # same statement itself and so knows today's cost. Take the
                # min: if the cache's baseline is the smaller of the two it is
                # the one we can defend, and if today's measurement is smaller
                # the cache's baseline was stale-high and must not be credited.
                today = measured.get(m.get("code"))
                if today is not None:
                    stats["total_verified_saved"] += min(saved, today)
                else:
                    # Nothing recomputed it here -- the usual case right after
                    # a restart. An earlier run on this machine measured it,
                    # and the least it ever cost is what it is credited.
                    before = baselines.get(m.get("code") or "")
                    if before is not None:
                        stats["total_measured_saved"] += min(saved, before)
            elif status == CacheStatus.SKIPPED:
                stats["statements_skipped"] += 1
            # A ``@cash.cache`` HIT inside this statement saved real compute that
            # is invisible to the counting above: the value came from the
            # decorator, so the statement itself only did a fast lookup and reads
            # as cheap COMPUTED work. Credit it here, from the same
            # drained call log the badge uses. No double-count risk: a decorator
            # is only invoked when the statement EXECUTES, so a RESTORED statement
            # (whose ``saved_time`` already covers the whole compute) carries no
            # decorator_calls to add.
            self._credit_decorator_calls(m.get("decorator_calls"), stats, floor)
        # Overhead = cell wall time minus the user compute that ran this cell.
        # Floor at 0: the wall time always covers the compute it contains, but
        # clamp defensively against clock skew / partial timing.
        stats["total_overhead"] += max(0.0, cell_total_time - cell_compute_time)
        # One small write per cell that measured something new, and none at
        # all for a cell that restored everything. A Restart & Run All kills
        # the kernel, so nothing may be left for an exit hook to write.
        baselines.flush()

    def _credit_decorator_calls(
        self,
        decorator_calls: "list[dict[str, Any]] | None",
        stats: dict[str, Any],
        floor: float,
    ) -> None:
        """Fold ``@cash.cache`` call metrics into the session totals.

        Without this, a session whose expensive work sits behind the decorator —
        the docs' own recommendation for training — reported the exact inverse of
        cash's value: ``%cash_stats`` counted only statement restores, so a warm
        pass that avoided a 30s fit via a decorator hit read as a net *cost*.

        A **miss** records this session's measured compute for that key, so a
        later hit can be credited as VERIFIED rather than merely gross. It is NOT
        added to compute totals: the enclosing statement's ``execution_time``
        already contains it, and adding it here would double-count.

        A **hit** credits ``time_saved`` to gross, mirrors CacheStatus.RESTORED:
        gross always, verified only under the min-rule when this session
        measured the same key's compute, and the cacheable-hit denominator when
        the saving cleared the floor.
        """
        if not decorator_calls:
            return
        measured = self._session.measured_decorator_compute
        baselines = self._baselines()
        for call in decorator_calls:
            if call.get("ran_plain"):
                continue  # run without the cache: neither a hit nor a measured miss
            key = call.get("cache_key")
            if call.get("cache_hit"):
                saved = call.get("time_saved", 0.0) or 0.0
                stats["total_time_saved"] += saved
                if saved >= floor:
                    stats["statements_cacheable_hit"] += 1
                today = measured.get(key)
                if today is not None:
                    stats["total_verified_saved"] += min(saved, today)
                else:
                    before = baselines.get(f"call:{key}") if key is not None else None
                    if before is not None:
                        stats["total_measured_saved"] += min(saved, before)
            else:
                # A miss's execution_time IS the measured compute for this key.
                if key is not None:
                    measured[key] = call.get("execution_time", 0.0) or 0.0
                    baselines.record(f"call:{key}", call.get("execution_time", 0.0) or 0.0)
                if (call.get("execution_time", 0.0) or 0.0) >= floor:
                    stats["statements_cacheable_miss"] += 1

    def _record_provenance(self, all_metrics: list[ProcessResult]) -> None:
        """Record a provenance entry per output variable of each statement in *all_metrics*."""
        for m in all_metrics:
            code = m.get("code", "")
            status = m.get("status", "computed")
            duration_ms = m.get("execution_time", 0.0) * 1000
            # ``rich_outputs`` holds IPython rich-display objects, NOT variable
            # names — never source variable names from it.
            outputs = m.get("restored_vars", []) or m.get("evaluated_vars", [])
            inputs_list = list(m.get("inputs", []))
            # Outputs may contain rich-display dicts; provenance only cares
            # about string variable names.
            var_names = [o for o in (outputs or []) if isinstance(o, str)]

            provenance_status = str(status).lower() if status else "computed"
            for out_var in var_names:
                self._session.provenance.record(
                    variable=out_var,
                    code=code,
                    inputs=inputs_list,
                    status=provenance_status,
                    duration_ms=duration_ms,
                    lineage_hash=self.tracking_state.variable_lineage.get(out_var, ""),
                    file_deps=list(self.tracking_state.executed_file_deps.get(out_var, [])),
                )
