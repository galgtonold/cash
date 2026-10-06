"""One simulated statement: its key, and the lineage of each name it writes.

:class:`StatementLineage` reproduces, for a statement the simulation reaches,
the key and output lineages the runtime would give it. It reads what the
statement reads and writes as the runtime does, builds the key with the same
``compute_cache_key``, forward-propagates the lineages a cache entry recorded
when the entry is still valid, and otherwise computes them with the runtime's
own formula (``lineage_formula``).
"""

from __future__ import annotations

import ast
import logging
import types
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, NamedTuple

from ..._clock import perf_counter as _perf_counter
from ...analysis.ast_util import parse_cached
from ...analysis.code_analyzer import parse_cell_source
from ...analysis.mutation_effects import (
    StatementEffects,
    captured_call_receiver_names,
    classify_receivers,
    live_function_source,
    statement_effects,
)
from ...analysis.namespace_effects import bare_call_argument_names, bare_call_arguments
from ..consumables import watched_call_receivers
from ...tracking.randomness import (
    advanced_carrier_lineage,
    hidden_lineage_writes,
    hidden_write_lineage,
)
from ...value_types import BUILTIN_NAMES
from .._protocols import ShellProtocol
from ..cache_key import CacheKeyContext, compute_cache_key, statement_source_hash
from ..lineage_formula import (
    callable_source_component,
    input_lineage,
    key_hidden_reads,
    lineage_hidden_reads,
    module_source_component,
    no_cache_value_component,
    output_lineage,
    recorded_reads_lineage_component,
    statement_environment_component,
)
from ..run_memo import stats_this_run
from ..stateful_carriers import carrier_kind_from_producer
from ..statement import is_control_body
from ..statement.carrier_advances import reachable_generators
from ..statement.derivation_edges import bump_derived_lineages
from ..statement.file_deps import compute_file_hash_component
from ..tracking_state import TrackingState
from .cache_probe import CacheProbe
from .simulated_callables import SimulatedCallables

if TYPE_CHECKING:
    from ...tracking.function_tracker import FunctionTracker

__all__ = ["StatementLineage", "StatementOutcome", "unbound_builtin"]

logger = logging.getLogger(__name__)


def unbound_builtin(name: str, variable_lineage: Mapping[str, str], bound: Mapping[str, str] | None = None) -> bool:
    """Is *name* a builtin here: one of `BUILTIN_NAMES` that neither the
    kernel (*variable_lineage*) nor the simulation so far (*bound*) has
    bound? A user's ``max = ...`` or ``id = ...`` is an input like any other,
    as the runtime treats it."""
    return name in BUILTIN_NAMES and name not in variable_lineage and (bound is None or name not in bound)


class StatementOutcome(NamedTuple):
    """What simulating one statement gave."""

    #: The names it writes, including the aliases its write bumps.
    outputs: set[str]
    #: Seconds spent reading its cache entry's metadata.
    lookup_time: float
    #: Whether a file its entry recorded changed since.
    files_stale: bool
    #: ``{path: mtime}`` of the files it depends on.
    file_deps: dict[str, float]


class _CacheLookup(NamedTuple):
    """What a statement's cache entry said about its outputs' lineages."""

    #: The entry was valid and its lineages were taken.
    hit: bool
    lookup_time: float
    files_stale: bool
    file_deps: dict[str, float]
    #: On a miss, the files the entry recorded, for the file component.
    files_to_check: set[str]
    #: On a hit, the aliases the write bumped.
    bumped: set[str]
    #: The generators the entry says the statement drew from
    #: (``StatementCacheMetadata.carriers_advanced``); None when it does not say.
    carriers: list[str] | None = None


class StatementLineage:
    """The key and output lineages of one simulated statement, as the runtime has them.

    What it propagates for an import (the lineage of each bound name) it
    writes to ``TrackingState`` at once, because the next statement's cache
    key reads it.
    """

    def __init__(
        self,
        shell: ShellProtocol,
        tracking_state: TrackingState,
        probe: CacheProbe,
        callables: SimulatedCallables,
        compute_hash_fn: Callable[[Any], str] | None = None,
        function_tracker: FunctionTracker | None = None,
    ) -> None:
        self.shell = shell
        #: The checker's. ``mutation_verdicts`` and
        #: ``observed_rng_statement_draws`` are read from it because the
        #: simulation must reproduce the runtime's key inputs exactly.
        self.tracking_state = tracking_state
        self.probe = probe
        self.callables = callables
        self.compute_hash_fn = compute_hash_fn
        #: The runtime's tracker, so the simulation hashes called functions
        #: and modules exactly as the statement processor does.
        self.function_tracker = function_tracker
        #: The lineage ``_propagate_import_lineage`` last gave each name, so a
        #: later import of that name can replace it -- but not one the runtime set.
        self.propagated_imports: dict[str, str] = {}
        #: ``{name: source}`` of every top-level def in the notebook's cells
        #: (see ``set_notebook_functions``).
        self._notebook_functions: dict[str, str] = {}
        #: Lineages the simulation gave a variable that holds a random
        #: generator: one bound by a call that makes one, and each lineage a
        #: draw moved it on to. After a restart the namespace has no
        #: generator to look at, and a draw no record describes is assumed
        #: from these (see ``_advanced_carriers``).
        self._generator_lineages: set[str] = set()

    # -- What a statement reads and writes ------------------------------------

    def set_notebook_functions(self, notebook_cells: list[str]) -> None:
        """Take the source of every top-level ``def`` across *notebook_cells*.

        Resolves from cell SOURCE (not ``inspect.getsource``, which has no
        linecache entry under nbclient) so ``function_arg_mutations`` can analyse
        a called function's body during the headless simulation. Later same-name
        defs win (last definition), matching the runtime namespace. A cell's
        magics are stripped first, as the simulation reads every cell.
        """
        sources: dict[str, str] = {}
        for code in notebook_cells:
            tree = parse_cell_source(code)
            if tree is None:
                continue
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    try:
                        sources[node.name] = ast.unparse(node)
                    except (ValueError, AttributeError):
                        continue
        self._notebook_functions = sources

    def _function_source(self, name: str) -> str | None:
        """Source of function *name* for the headless mutation analysis.

        Cell-defined functions have no ``linecache`` entry under nbclient, so
        they come from the notebook's cell text. A function imported from a
        ``.py`` file is not in the cell text; it resolves as the runtime
        resolves it, so an imported helper that mutates its argument is seen by
        both engines.
        """
        source = self._notebook_functions.get(name)
        if source is not None:
            return source
        return live_function_source(name, self.shell.user_ns)

    def _mutation_verdict(self, source_hash: str) -> set[str] | None:
        """The runtime's verdict on a bare method call: this session's, else
        an earlier kernel's, kept in ``mutation_verdicts`` once read (the
        runtime overwrites it when the statement runs again)."""
        verdict = self.tracking_state.mutation_verdicts.get(source_hash)
        if verdict is not None:
            return verdict
        verdict = self.probe.mutation_verdict(source_hash)
        if verdict is not None:
            self.tracking_state.mutation_verdicts.setdefault(source_hash, verdict)
        return verdict

    def _mutation_receivers(
        self,
        stmt_code: str,
        tree: ast.Module,
        virtual_modules: set[str] | None = None,
    ) -> set[str]:
        """Names *tree*'s calls change in place, decided as the runtime decides
        them (``classify_receivers``).

        The runtime's verdict for the statement is read from this session, else
        from the backend (an earlier kernel). *virtual_modules* names modules
        the simulation bound but the kernel does not hold yet: after a restart
        ``pd.set_option(...)`` must still read as a module call, not as an
        unknown method that bumps ``pd`` for every reader.
        """
        user_ns = self.shell.user_ns

        def load_verdict() -> set[str] | None:
            return self._mutation_verdict(statement_source_hash(stmt_code))

        # Bare-call arguments: the live ones the runtime watches, and, after a
        # restart, the ones not live yet, whose recorded verdict is all there
        # is to go on (`heapq.heapify(xs)` must replay before `xs[0]`).
        # Likewise the receivers of a method call whose result is bound
        # (`history = net.fit(X)`), which the runtime fingerprints too.
        arguments = (
            bare_call_arguments(tree, user_ns)
            | watched_call_receivers(tree, user_ns)
            | {n for n in bare_call_argument_names(tree) | captured_call_receiver_names(tree) if n not in user_ns}
        )
        classes = classify_receivers(
            tree, user_ns, load_verdict, arguments=arguments, virtual_modules=virtual_modules or ()
        )
        # The simulation cannot watch the statement run: an undecided
        # receiver is assumed to change, an undecided argument not to.
        return set(classes.mutated | classes.unknown_receivers)

    def reads_writes(
        self,
        stmt_code: str,
        tree: ast.Module | None,
        virtual_modules: set[str],
    ) -> tuple[StatementEffects, set[str], set[str]]:
        """*stmt_code*'s effects, and what it reads and writes, as its key sees them.

        The same analysis the runtime's ``_analyze_and_hash`` runs, with the
        notebook's cell text as the source of called functions: the two
        engines must agree on what a statement reads and writes, or they mint
        different keys. A bare method call (``lst.append(x)``) or a bare call
        to a helper that mutates its argument has no Store target, so the
        receivers the runtime treats as mutated are writes too. The globals a
        callee writes (``effects.callee_globals``) are left to the caller.
        """
        effects = statement_effects(
            stmt_code,
            tree,
            namespace=self.shell.user_ns,
            resolve_source=self._function_source,
            control_body=is_control_body(stmt_code),
            virtual_modules=virtual_modules,
        )
        inputs, outputs = set(effects.inputs), set(effects.outputs)
        if tree is not None:
            outputs |= self._mutation_receivers(stmt_code, tree, virtual_modules)
            outputs |= effects.arg_mutations
        return effects, inputs, outputs

    def is_unbound_builtin(self, name: str, bound: Mapping[str, str] | None = None) -> bool:
        """``unbound_builtin`` against the kernel's lineages."""
        return unbound_builtin(name, self.tracking_state.variable_lineage, bound)

    # -- Keys and lineages -----------------------------------------------------

    def key_context(self, virtual_lineage: Mapping[str, str], virtual_modules: set[str]) -> CacheKeyContext:
        """The ``compute_cache_key`` context for a statement simulated with
        *virtual_lineage* and *virtual_modules*."""
        return CacheKeyContext(
            variable_lineage=self.tracking_state.variable_lineage,
            user_ns=self.shell.user_ns,
            function_tracker=self.function_tracker,
            virtual_lineage=virtual_lineage,
            virtual_modules=virtual_modules,
            compute_hash_fn=self.compute_hash_fn,
            virtual_callables=self.callables.by_lineage,
            recorded_reads=self.tracking_state.recorded_reads,
            recorded_reads_by_key=self.tracking_state.recorded_reads_by_key,
        )

    def _input_lineages(
        self, stmt_code: str, inputs: set[str], virtual_lineage: dict[str, str], virtual_modules: set[str]
    ) -> list[str]:
        """Each input's lineage (``lineage_formula.input_lineage``), with the
        simulation's own lineages in front of the recorded ones."""
        input_lineages_all = []
        for inp in sorted(inputs):
            if inp in {"get_ipython", "__builtins__"}:
                continue
            lineage = input_lineage(
                inp,
                self.shell.user_ns,
                (virtual_lineage, self.tracking_state.variable_lineage),
                compute_hash=self.compute_hash_fn,
                function_tracker=self.function_tracker,
                code=stmt_code,
                virtual_modules=virtual_modules,
            )
            if lineage:
                input_lineages_all.append(lineage)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[LINEAGE_DEBUG] %s... inputs %s -> %s",
                stmt_code[:50],
                sorted(inputs),
                [ln[:12] + "..." for ln in input_lineages_all],
            )
        return input_lineages_all

    def output_lineages(
        self,
        source_hash: str,
        input_lineages_all: list[str],
        file_hash_component: str,
        inputs: set[str],
        outputs: set[str],
        stmt_code: str,
        tree: ast.Module | None = None,
        virtual_lineage: dict[str, str] | None = None,
        no_cache_values: dict[str, str] | None = None,
        recorded_reads: str | None = None,
    ) -> dict[str, str]:
        """The lineage of each output of a simulated statement.

        *no_cache_values* are the value digests a ``no-cache`` statement
        recorded when it last ran, read back so the simulation reaches the
        lineage the runtime recorded; *recorded_reads* the same for the
        environment and the data of the user's modules it read
        (``recorded_reads_lineage_component``).

        Built by the runtime's own formula (``lineage_formula``), one output at
        a time as the runtime does: a module-source component belongs to the
        name that came from the module.

        A callee that is only a simulated def contributes its digest as the
        live function would (``SimulatedCallables.source_hashes``).
        """
        function_tracker = self.function_tracker
        user_ns = self.shell.user_ns
        try:
            func_component = callable_source_component(function_tracker, inputs, user_ns)
            virtual = self.callables.source_hashes(inputs, virtual_lineage or {})
            if virtual and function_tracker is not None:
                hashes = function_tracker.get_callable_source_hashes(inputs, user_ns)
                hashes.update(virtual)
                func_component = ":" + ":".join(f"{k}:{v}" for k, v in sorted(hashes.items()))
        except (TypeError, ValueError, AttributeError):
            logger.debug("[UPSTREAM] Failed to compute function source hashes for capture")
            func_component = ""
        environment = statement_environment_component(stmt_code, user_ns) if recorded_reads is None else recorded_reads
        return {
            out: output_lineage(
                source_hash,
                input_lineages_all,
                file_hash_component,
                func_component,
                module_source_component(function_tracker, user_ns.get(out), out, stmt_code, tree),
                environment,
                no_cache_value_component(no_cache_values, out),
            )
            for out in outputs
        }

    # -- Files -----------------------------------------------------------------

    def _session_file_deps(self, outputs: set[str]) -> set[str]:
        """Return file dependencies from the current session for *outputs*."""
        file_deps: set[str] = set()
        executed_file_deps = self.tracking_state.executed_file_deps
        for out in outputs:
            file_deps.update(executed_file_deps.get(out, ()))
        return file_deps

    @staticmethod
    def _file_hash_component(file_deps_to_check: set[str], stmt_file_deps: dict[str, float]) -> str:
        """The file component of a statement's lineage when the runtime's own
        record of what it read is not available: the files its outputs depend
        on, valued by the runtime's formula (``compute_file_hash_component``).

        Also updates stmt_file_deps with current mtimes for tracked files.
        """
        if not file_deps_to_check:
            return ""

        present: set[str] = set()
        current = stats_this_run(file_deps_to_check)
        for file_path in file_deps_to_check:
            resolved, stat = current[file_path]
            # Only a file that is where it was recorded, as before: the
            # relocation fallbacks would put a different path's state in a key.
            if stat is not None and resolved == file_path:
                present.add(file_path)
                stmt_file_deps[file_path] = stat.st_mtime
        return compute_file_hash_component(present) if present else ""

    # -- Imports ---------------------------------------------------------------

    def _bound_modules(self, outputs: set[str], tree: ast.Module | None, stmt_code: str = "") -> set[str]:
        """The names an import statement binds to a MODULE.

        ``import x`` always binds one. ``from m import name`` usually binds a
        function or a constant, and the runtime's key builder decides by the
        value (``is_module_like``): a function goes in as an input with its
        source hash. Counting every imported name as a module would give
        ``df = clean(raw)`` a different key in the simulation, so the
        simulation would never find that statement's entry.

        A name not bound yet is answered by what the statement bound when it
        last ran (``CacheProbe.import_bindings``), and without that record
        counts as a module.
        """
        if tree is None:
            return set(outputs)
        from_bound = {
            alias.asname or alias.name for node in tree.body if isinstance(node, ast.ImportFrom) for alias in node.names
        }
        user_ns = self.shell.user_ns
        recorded = self.probe.import_bindings(stmt_code) if stmt_code and (from_bound - set(user_ns)) else {}

        def is_module(out: str) -> bool:
            if out in user_ns:
                return isinstance(user_ns[out], types.ModuleType)
            if out in recorded:
                return bool(recorded[out].get("module"))
            return True

        return {out for out in outputs if out not in from_bound or is_module(out)}

    def _propagate_import_lineage(self, outputs: set[str], lineage_by_out: dict[str, str]) -> None:
        """Record the lineage of each name an import binds in ``variable_lineage``.

        Done at once so that ``compute_cache_key`` finds the module in
        ``variable_lineage`` for the next statement and includes it in the
        module component, as the runtime does.
        """
        # Every name the import binds, not only modules: an import the runtime
        # SKIPPED leaves its names without a lineage otherwise, and a statement
        # reading one is then not cached ("input variable missing lineage").
        # A lineage put there for an EARLIER import of the name is replaced:
        # `import os, sys` in the cell that turns cash on (it runs uncached) and
        # `import sys` in the next. After a restart the second never runs
        # (`sys` is bound), and `sys` must still carry the second's lineage,
        # which is the one the session before keyed everything with.
        for out in outputs:
            if out not in lineage_by_out:
                continue
            held = self.tracking_state.variable_lineage.get(out)
            if held is None or held == self.propagated_imports.get(out):
                self.tracking_state.lineage.record(out, lineage_by_out[out])
                self.propagated_imports[out] = lineage_by_out[out]
                logger.debug(
                    "[LINEAGE_DEBUG] Propagated module '%s' lineage to variable_lineage: %s...",
                    out,
                    lineage_by_out[out][:12],
                )

    # -- Forward propagation from the cache ------------------------------------

    def _take_cached_lineages(
        self,
        stmt_code: str,
        outputs: set[str],
        inputs: set[str],
        virtual_lineage: dict[str, str],
        is_import: bool,
        output_lineages: dict[str, str],
    ) -> set[str]:
        """Take a valid cache entry's *output_lineages* into *virtual_lineage*,
        and return the aliases the write bumps."""
        logger.debug("[UPSTREAM] Forward propagating cached lineages for %s...", stmt_code[:30])
        for var, h in output_lineages.items():
            virtual_lineage[var] = h
        # Even on a cache hit, replay the derivation-alias bump so a mutation of
        # a base/frame (its own lineage restored from cache here) still bumps its
        # live-alias derivatives. Same skip-inputs rule and
        # deterministic formula as the runtime and the miss path. Bumped vars are
        # threaded back so the caller can union them into ``outputs``.
        bumped = bump_derived_lineages(
            self.tracking_state.derivation_edges,
            virtual_lineage,
            outputs,
            inputs,
            record=lambda t, h: virtual_lineage.__setitem__(t, h),
            present=lambda t: True,
        )
        if is_import:
            for out in outputs:
                if out not in self.tracking_state.variable_lineage:
                    lineage_val = output_lineages.get(out)
                    if lineage_val:
                        # Recorded now: later statements' cache keys read it.
                        self.tracking_state.lineage.record(out, lineage_val)
                        logger.debug(
                            "[LINEAGE_DEBUG] Propagated module '%s' lineage (from cache): %s...",
                            out,
                            lineage_val[:12],
                        )
        return bumped

    def _lookup_cached_lineages(
        self,
        stmt_code: str,
        cache_key: str,
        outputs: set[str],
        inputs: set[str],
        virtual_lineage: dict[str, str],
        is_import: bool,
    ) -> _CacheLookup:
        """Forward-propagate lineages from *cache_key*'s entry when it is valid.

        A hit takes the entry's output lineages. A miss -- no entry, no
        lineages, or a file the entry recorded has changed -- returns the
        files the entry recorded, for the caller's file component.
        """
        lookup_time = 0.0
        files_stale = False
        carriers = None
        stmt_file_deps: dict[str, float] = {}
        files_to_check: set[str] = set()

        if not self.probe.cash_instance:
            return _CacheLookup(False, lookup_time, files_stale, stmt_file_deps, files_to_check, set())

        try:
            logger.debug("[UPSTREAM] Virtual lookup Key: %s", cache_key)

            t_lookup = _perf_counter()
            metadata = self.probe.metadata(cache_key)
            lookup_time = _perf_counter() - t_lookup

            if metadata:
                carriers = metadata.get("carriers_advanced")
                hist_files = metadata.get("file_dependencies", {})
                output_lineages = metadata.get("output_lineages", {})
                files_valid = not hist_files or self.probe.files_fresh(hist_files, memo_key=cache_key)

                if files_valid and output_lineages:
                    bumped = self._take_cached_lineages(
                        stmt_code, outputs, inputs, virtual_lineage, is_import, output_lineages
                    )
                    hit_file_deps = CacheProbe.stat_file_deps(hist_files)
                    return _CacheLookup(True, lookup_time, False, hit_file_deps, set(), bumped, carriers)

                if not files_valid:
                    files_stale = True

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "[UPSTREAM] Forward prop aborted. files_valid=%s, output_lineages keys=%s",
                        files_valid,
                        list(output_lineages.keys()) if output_lineages else "None/Empty",
                    )

                if hist_files:
                    files_to_check.update(hist_files.keys())
                    stmt_file_deps.update(CacheProbe.stat_file_deps(hist_files))
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "[UPSTREAM] Found historical file deps (validation failed/skipped): %s",
                            list(hist_files.keys()),
                        )
        except (KeyError, TypeError, OSError, ValueError) as e:
            logger.debug("[UPSTREAM] Virtual lookup failed: %s", e)

        return _CacheLookup(False, lookup_time, files_stale, stmt_file_deps, files_to_check, set(), carriers)

    # -- One statement ---------------------------------------------------------

    def apply(
        self,
        stmt_code: str,
        virtual_lineage: dict[str, str],
        virtual_modules: set[str] | None = None,
        occurrence_index: int = 0,
    ) -> StatementOutcome:
        """Simulate *stmt_code*: update *virtual_lineage* and *virtual_modules*
        with what it binds, and return what it wrote.

        *occurrence_index* is the zero-based occurrence of the same statement
        text within its cell, which the runtime's key counts.
        """
        try:
            if virtual_modules is None:
                virtual_modules = set()

            tree = parse_cached(stmt_code)
            effects, inputs, outputs = self.reads_writes(stmt_code, tree, virtual_modules)
            _apply_deletes(tree, virtual_lineage, virtual_modules)

            is_import = stmt_code.strip().startswith(("import ", "from "))
            if is_import:
                virtual_modules.update(self._bound_modules(outputs, tree, stmt_code))

            # RNG state is a hidden lineage variable: a draw reads it,
            # a seed produces it. Kept out of the plain ``inputs``.
            hidden_reads = key_hidden_reads(stmt_code, self.tracking_state)
            hidden_writes = hidden_lineage_writes(stmt_code)

            # A bare ``seed()`` carries no output, so it would return below before
            # recording its hidden variable. Compute its key (a seed is not a
            # draw, so no hidden read) and write the variable first.
            if hidden_writes and not outputs:
                seed_key = self._key(stmt_code, inputs, outputs, virtual_lineage, virtual_modules, occurrence_index)
                for var in hidden_writes:
                    virtual_lineage[var] = hidden_write_lineage(seed_key)

            # The globals a CALLEE writes join ``outputs`` so the
            # simulated lineage is bumped with the same source-based formula
            # the runtime uses. The runtime ALSO skip-caches such a statement;
            # that half is runtime-only.
            outputs = outputs | effects.callee_globals

            if not outputs:
                # A bare call can still draw from a generator it is handed
                # (`print(draw(0, rng))`), which moves the generator's lineage.
                advanced = self._advance_carriers_without_outputs(
                    stmt_code, inputs, hidden_reads, virtual_lineage, virtual_modules, occurrence_index
                )
                return StatementOutcome(advanced, 0.0, False, {})

            return self._apply_writes(
                stmt_code,
                tree,
                inputs,
                outputs,
                hidden_reads,
                hidden_writes,
                is_import,
                virtual_lineage,
                virtual_modules,
                occurrence_index,
            )
        except (KeyError, TypeError, ValueError, OSError) as e:
            logger.error("[UPSTREAM] Error simulating statement '%s...': %s", stmt_code[:20], e)
            raise

    def _key(
        self,
        stmt_code: str,
        inputs: set[str],
        outputs: set[str],
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        occurrence_index: int,
    ) -> str:
        """*stmt_code*'s cache key, built by the runtime's ``compute_cache_key``."""
        cache_key, _, _, _, _ = compute_cache_key(
            stmt_code,
            inputs,
            ctx=self.key_context(virtual_lineage, virtual_modules),
            outputs=outputs,
            occurrence_index=occurrence_index,
        )
        return cache_key

    def _apply_writes(
        self,
        stmt_code: str,
        tree: ast.Module | None,
        inputs: set[str],
        outputs: set[str],
        hidden_reads: set[str],
        hidden_writes: set[str],
        is_import: bool,
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        occurrence_index: int,
    ) -> StatementOutcome:
        """Give each of *outputs* its lineage: the cache entry's when it is
        valid, else the runtime's formula's."""
        input_lineages_all = self._input_lineages(
            stmt_code, inputs | lineage_hidden_reads(stmt_code), virtual_lineage, virtual_modules
        )
        cache_key = self._key(
            stmt_code, inputs | hidden_reads, outputs, virtual_lineage, virtual_modules, occurrence_index
        )

        # Combined seed+draw statement (has an output): record its hidden
        # write AFTER its key, matching the runtime's ordering.
        for var in hidden_writes:
            virtual_lineage[var] = hidden_write_lineage(cache_key)

        file_deps_to_check = self._session_file_deps(outputs)
        lookup = self._lookup_cached_lineages(stmt_code, cache_key, outputs, inputs, virtual_lineage, is_import)

        # The generators it draws from move on, as the runtime moves them on
        # after the run or the hit (`carrier_advances`).
        advanced = self._advanced_carriers(stmt_code, lookup.carriers, inputs, outputs, virtual_lineage)
        for name in advanced:
            virtual_lineage[name] = advanced_carrier_lineage(cache_key, name)
            self._generator_lineages.add(virtual_lineage[name])

        if lookup.hit:
            # Union derivation-bumped vars so this cached mutation statement
            # is still recorded as a producer of the aliased base.
            self._note_made_generators(stmt_code, outputs, virtual_lineage)
            outputs = outputs | lookup.bumped | advanced
            self._register_callables(stmt_code, tree, virtual_lineage, is_import)
            return StatementOutcome(outputs, lookup.lookup_time, False, lookup.file_deps)

        stmt_file_deps = lookup.file_deps
        file_deps_to_check.update(lookup.files_to_check)
        file_hash_component = self._file_hash_component(file_deps_to_check, stmt_file_deps)
        own_reads = self.tracking_state.statement_file_reads.get(cache_key)
        if own_reads is not None:
            # The runtime hashed the files THIS statement read -- not the
            # ones its outputs inherited -- with compute_file_hash_component.
            # Same files, same function: an unchanged file gives the
            # runtime's lineage, a changed one a different lineage.
            file_hash_component = compute_file_hash_component(*own_reads)

        source_hash = statement_source_hash(stmt_code)
        lineage_by_out = self.output_lineages(
            source_hash,
            input_lineages_all,
            file_hash_component,
            inputs,
            outputs,
            stmt_code,
            tree,
            virtual_lineage,
            no_cache_values=self.tracking_state.no_cache_values.get(cache_key),
            recorded_reads=recorded_reads_lineage_component(
                self.tracking_state, cache_key, stmt_code, self.shell.user_ns
            ),
        )
        _log_lineage_calc(stmt_code, source_hash, input_lineages_all, file_hash_component, lineage_by_out)

        virtual_lineage.update(lineage_by_out)
        self._note_made_generators(stmt_code, outputs, virtual_lineage)
        outputs = outputs | advanced
        self._register_callables(stmt_code, tree, virtual_lineage, is_import)

        # Mirror the runtime derivation-alias bump: when
        # a base/frame is mutated in place, bump its live-alias derivatives.
        # The simulator cannot observe ``.base`` / ``.obj`` identity, so it
        # only REPLAYS the runtime-recorded edge map with the SAME
        # skip-inputs rule and the SAME deterministic derived-hash formula,
        # keeping runtime and simulation byte-identical. No live namespace,
        # so every edge target counts as present. Union bumped vars into
        # ``outputs`` so the reexecution planner records THIS statement as a
        # producer of the aliased base and reschedules it on an isolated
        # re-run (the base's own cache is the stale pre-mutation value).
        bumped = bump_derived_lineages(
            self.tracking_state.derivation_edges,
            virtual_lineage,
            outputs,
            inputs,
            record=lambda t, h: virtual_lineage.__setitem__(t, h),
            present=lambda t: True,
        )
        outputs = outputs | bumped

        # The next statement's key must find an imported module's lineage in
        # ``variable_lineage``, or it leaves the module out of its module
        # component and disagrees with the runtime's key.
        if is_import:
            self._propagate_import_lineage(outputs, lineage_by_out)

        return StatementOutcome(outputs, lookup.lookup_time, lookup.files_stale, stmt_file_deps)

    def _note_made_generators(self, stmt_code: str, outputs: set[str], virtual_lineage: Mapping[str, str]) -> None:
        """Remember the lineages *stmt_code* gave its *outputs* when its code
        makes a random generator (``rng = np.random.default_rng(42)``)."""
        if carrier_kind_from_producer(stmt_code) not in _GENERATOR_KINDS:
            return
        for name in outputs:
            if name in virtual_lineage:
                self._generator_lineages.add(virtual_lineage[name])

    def _recorded_draws(self, stmt_code: str) -> frozenset[str] | None:
        """The generators *stmt_code* drew from when it last ran: this
        session's record, else an earlier kernel's, kept in
        ``carrier_advances`` once read (the runtime overwrites it when the
        statement runs again). None when neither says."""
        source_hash = statement_source_hash(stmt_code)
        recorded = self.tracking_state.carrier_advances.get(source_hash)
        if recorded is not None:
            return recorded
        recorded = self.probe.carrier_advances(source_hash) if self.probe.cash_instance else None
        if recorded is not None:
            self.tracking_state.carrier_advances.setdefault(source_hash, recorded)
        return recorded

    def _reachable_generators(self, inputs: set[str], virtual_lineage: Mapping[str, str]) -> set[str]:
        """The generators a statement reading *inputs* can draw from: the live
        ones, and those not live yet whose simulated lineage is a generator's."""
        user_ns = self.shell.user_ns
        dead = {
            name for name in inputs if name not in user_ns and virtual_lineage.get(name) in self._generator_lineages
        }
        return reachable_generators(inputs, user_ns) | dead

    def _advanced_carriers(
        self,
        stmt_code: str,
        in_entry: list[str] | None,
        inputs: set[str],
        outputs: set[str],
        virtual_lineage: Mapping[str, str],
    ) -> set[str]:
        """The variables holding a generator that *stmt_code* draws from.

        As its cache entry recorded them (*in_entry*: the runtime moves those
        on after a hit), else as its last run saw them (this session's, or an
        earlier kernel's record: a cheap draw has no entry). A statement never
        seen run is assumed to draw from every generator it can reach: when it
        does not, the simulation only re-runs something it could have kept.
        """
        if is_control_body(stmt_code):
            return set()
        recorded = in_entry
        if recorded is None:
            recorded = self._recorded_draws(stmt_code)
        if recorded is None:
            recorded = self._reachable_generators(inputs, virtual_lineage)
        return set(recorded) - outputs

    def _advance_carriers_without_outputs(
        self,
        stmt_code: str,
        inputs: set[str],
        hidden_reads: set[str],
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        occurrence_index: int,
    ) -> set[str]:
        """:meth:`_advanced_carriers` for a statement that binds nothing, with
        their new lineages written into *virtual_lineage*. Keyed only when it
        reads a generator, or its last run drew from one."""
        if is_control_body(stmt_code) or not inputs:
            return set()
        recorded = self._recorded_draws(stmt_code)
        if recorded is not None and not recorded:
            return set()
        if recorded is None and not self._reachable_generators(inputs, virtual_lineage):
            return set()
        cache_key = self._key(
            stmt_code, inputs | hidden_reads, set(), virtual_lineage, virtual_modules, occurrence_index
        )
        try:
            metadata = self.probe.metadata(cache_key) if self.probe.cash_instance else None
        except (KeyError, TypeError, ValueError, OSError, AttributeError):
            metadata = None
        in_entry = metadata.get("carriers_advanced") if metadata else None
        advanced = self._advanced_carriers(stmt_code, in_entry, inputs, set(), virtual_lineage)
        for name in advanced:
            virtual_lineage[name] = advanced_carrier_lineage(cache_key, name)
            self._generator_lineages.add(virtual_lineage[name])
        return advanced

    def _register_callables(
        self, stmt_code: str, tree: ast.Module | None, virtual_lineage: dict[str, str], is_import: bool
    ) -> None:
        """Remember a def, or what an import bound, for later statements' keys."""
        self.callables.register_def(stmt_code, tree, virtual_lineage)
        if is_import:
            self.callables.register_imports(tree, virtual_lineage, stmt_code)


#: The kinds ``carrier_kind_from_producer`` gives a random generator.
_GENERATOR_KINDS = frozenset({"numpy Generator", "random.Random"})


def _apply_deletes(tree: ast.Module | None, virtual_lineage: dict[str, str], virtual_modules: set[str]) -> None:
    """Drop the names a bare-name ``del x`` in *tree* removes.

    Modelled as a namespace removal so the position-scoped liveness check
    downstream reconstructs an above-the-del consumer's inputs. Only
    ``ast.Name`` targets remove a lineage entry; ``del d[k]`` / ``del
    obj.attr`` are container mutations (``MutationVisitor.visit_Delete``), so
    they must NOT pop the base's lineage.
    """
    if tree is None:
        return
    for node in tree.body:
        if not isinstance(node, ast.Delete):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                virtual_lineage.pop(tgt.id, None)
                virtual_modules.discard(tgt.id)


def _log_lineage_calc(
    stmt_code: str,
    source_hash: str,
    input_lineages_all: list[str],
    file_hash_component: str,
    lineage_by_out: dict[str, str],
) -> None:
    """Debug-log how a statement's output lineages were computed."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug("[LINEAGE_CALC] Statement: %s...", stmt_code[:40])
    logger.debug("[LINEAGE_CALC]   source_hash: %s...", source_hash[:16])
    logger.debug(
        "[LINEAGE_CALC]   sorted(input_lineages_all): %s",
        [h[:12] + "..." for h in sorted(input_lineages_all)],
    )
    logger.debug(
        "[LINEAGE_CALC]   file_hash_component: %s...",
        file_hash_component[:20] if file_hash_component else "(empty)",
    )
    logger.debug("[LINEAGE_CALC]   => lineages: %s", {v: h[:16] for v, h in lineage_by_out.items()})
