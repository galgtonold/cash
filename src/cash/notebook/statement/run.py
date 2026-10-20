"""One statement's trip through the processor, and the code that runs it.

:class:`StatementRun` carries a statement from its request through analysis,
the cache lookup and execution to storage, in place of the long argument
lists the pipeline's steps used to pass each other. :class:`CodeRunner` runs
the statement's code; the sync and async paths differ only in which of its
two methods they call.
"""

from __future__ import annotations
import __future__

import ast
import functools
import inspect
import operator
import sys
import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cash._user_attrs import attribute_or
from cash.notebook.cache_status import ExecutionResult
from cash.notebook.compiled_source import is_cash_filename, line_offset, register_cell_source, shifted
from cash.notebook.statement.capture import NoCapture

if TYPE_CHECKING:
    import types

    from cash.analysis.analysis_budget import AnalysisBudget
    from cash.analysis.annotations import CacheAnnotation
    from cash.analysis.cacheability import StatementAnalysis
    from cash.notebook.shared_objects import GaveUp
    from cash.notebook.statement.results import ProcessResult

__all__ = [
    "ECHO_FIELD",
    "LOOP_ITERABLE_NAME",
    "NO_LOOP_ITERABLE",
    "CodeRunner",
    "StatementExecution",
    "StatementRun",
    "echoes",
    "error_result",
    "over_the_handed_iterable",
]

#: The key that carries the value a cell's last expression echoed: in a
#: statement's metrics, and in the cache entry that replays it on a hit.
ECHO_FIELD = "echo_value"

#: The name a loop run as one unit iterates when its header was evaluated
#: before the unit ran (``StatementRun.loop_iterable``). Bound only while the
#: unit runs; never an input, an output or part of a key.
LOOP_ITERABLE_NAME = "__cash_loop_iter__"


class _NoLoopIterable:
    def __repr__(self) -> str:
        return "NO_LOOP_ITERABLE"


#: ``StatementRun.loop_iterable`` when the statement evaluates its own header.
NO_LOOP_ITERABLE: Any = _NoLoopIterable()


def over_the_handed_iterable(source: str) -> tuple[str, ast.Module]:
    """*source*, a ``for`` loop, iterating :data:`LOOP_ITERABLE_NAME` in place
    of its header, and the tree of that text.

    Only the header expression changes. It sits on the ``for`` line of the
    unparsed text, so every line keeps its number and a traceback still
    points at the body line that raised.
    """
    tree = ast.parse(source)
    loop = tree.body[0] if len(tree.body) == 1 else None
    if not isinstance(loop, ast.For):
        raise ValueError("a handed-over loop iterable needs a single for statement")
    loop.iter = ast.copy_location(ast.Name(id=LOOP_ITERABLE_NAME, ctx=ast.Load()), loop.iter)
    ast.fix_missing_locations(tree)
    text = ast.unparse(tree)
    return text, ast.parse(text)


@dataclass
class StatementRun:
    """A statement being processed: the request, then what analysis decided.

    The request fields are set by the caller; the rest are filled in by the
    processor as the statement goes through analysis and the cache lookup.
    """

    code: str
    ttl: int | None = None
    silent: bool = False
    annotation: CacheAnnotation | None = None
    #: The user's own text, shown on the badge; never part of the key.
    display_code: str | None = None
    #: The statement's original text (comments intact), compiled in place of
    #: ``code`` so a function defined here keeps its comments. Never keyed.
    exec_source: str | None = None
    occurrence_index: int = 0
    stream_output: bool = False
    #: Streaming, still capture the displays, which a hit shows again.
    keep_displays: bool = False
    #: Extra outputs to capture and restore (the accumulator-loop fast path).
    force_outputs: set[str] | None = None
    is_last: bool = True
    #: For a loop run as one unit whose header cash already evaluated: the
    #: value that evaluation gave, which the loop iterates in place of
    #: evaluating its header a second time. Never keyed: the key is ``code``.
    loop_iterable: Any = NO_LOOP_ITERABLE
    #: The files and remote URLs evaluating that header read. It ran before
    #: the unit's own tracker started, so they are the unit's reads too.
    header_files: frozenset[str] = frozenset()
    header_remote: frozenset[str] = frozenset()

    metrics: ProcessResult = field(default_factory=dict)  # type: ignore[assignment]
    process_start: float = 0.0
    tree: ast.Module | None = None
    effective_ttl: int | None = None
    force_persist: bool = False
    skip_cache: bool = False
    #: The statement reads from an open file, whose position it moves.
    drew_from_stream: bool = False
    #: Refused because what it binds can differ between two runs with the
    #: same key (``CacheabilityVerdict.varies``): the clock, a fresh id.
    value_varies: bool = False
    allow_random: bool = False
    unseeded_calls: list = field(default_factory=list)
    inputs: set[str] = field(default_factory=set)
    outputs: set[str] = field(default_factory=set)
    #: The lineage of each input as the statement started to run, recorded
    #: for a statement that turns out to write a file
    #: (``TrackingState.executed_write_input_lineages``).
    entry_lineages: dict[str, str] = field(default_factory=dict)
    #: The entry each input that is also an output came from as the
    #: statement started to run (``TrackingState.variable_sources``): by the
    #: store it names this statement's own entry, not yet written.
    entry_sources: dict[str, str] = field(default_factory=dict)
    source_hash: str = ""
    cache_key: str = ""
    #: The key the RNG streams it drew from move on from, when not
    #: ``cache_key``: a draw first seen as it ran (``_key_a_newly_seen_draw``).
    rng_advance_key: str = ""
    analysis: StatementAnalysis | None = None
    #: Receivers to content-observe after execution, receivers assumed
    #: mutated, and whether this run learns the statement's mutation verdict.
    mut_observe: set[str] = field(default_factory=set)
    mut_assumed: set[str] = field(default_factory=set)
    mut_record: bool = False
    #: ``id(value) -> (value, digest)``: the content digests the check after
    #: execution read (`DigestHandoff.take_settled`), for the outputs'
    #: lineage and session hash, with no user code run since.
    settled_digests: dict[int, tuple[Any, str]] = field(default_factory=dict)
    #: Modules a ``set_*`` call is assumed to change until a name it is
    #: handed is seen changing (``ReceiverClasses.provisional``), and the
    #: reason they were skip-cached with.
    mut_provisional: dict[str, frozenset[str]] = field(default_factory=dict)
    mut_provisional_reason: str = ""
    #: A statement whose verdict is known, watched as it runs on new inputs
    #: for what the verdict does not hold (``MutationClassifier.start_rewatch``).
    rewatching: bool = False
    #: What the mutation verdict of the statement is recorded under:
    #: ``source_hash``, but the text without its iteration marker for a
    #: statement of a loop or branch body, so each pass shares it.
    verdict_hash: str = ""
    #: For a statement of a loop or branch body, or a structure run as one
    #: unit: the names it changes in place that are not its outputs. It
    #: runs every time for them, and the loop it is in owns their lineage.
    in_place: set[str] = field(default_factory=set)
    #: It decides a verdict as it runs, so it must run rather than be
    #: restored (an entry would leave what it changes unseen).
    must_run: bool = False
    #: For a loop run as one unit whose watch is learning (``loop_watch``):
    #: the loop's code, the watch's name.
    loop_watch: str | None = None
    #: Receivers of a bare estimator fit, cached with the statement.
    est_fit: set[str] = field(default_factory=set)
    #: The file whose change threw the statement's cache entry away, when
    #: that is why it runs (``CacheFreshnessChecker.last_stale_file``).
    stale_file: str | None = None
    #: What actually executes: ``code`` with eligible calls routed through
    #: the call cache. When that rewrite fires, ``exec_source`` is cleared so
    #: the rewritten code is what compiles.
    exec_code: str = ""
    exec_tree: ast.Module | None = None
    #: The filename the statement's code was compiled under when it offered
    #: its call a hand-off (`cash.handed_digests`); None when it did not.
    offered_as: str | None = None
    #: The input variables holding a random generator that the statement drew
    #: from (``carrier_advances``); None when it read no generator.
    carriers_advanced: set[str] | None = None
    #: The inputs holding a sparse matrix the run put in canonical form
    #: (`sparse_order`); None when it put none.
    sparse_ordered: list[str] | None = None
    #: name -> digest of the environment variables installed code read while
    #: it ran (`env_reads.environment_digests`).
    env_reads: dict[str, str] | None = None
    #: For a statement that runs a magic or a shell command: the identity of
    #: every binding in the namespace before it ran (``calls_ipython``).
    ipython_bindings: dict[str, int] | None = None
    #: For a magic statement (``is_magic_statement``): the lineage of each
    #: name it reads, before it ran (``magic_effects``).
    magic_reads: dict[str, str | None] | None = None
    #: For a ``%time``/``%timeit``/``%prun`` statement: the fingerprint of
    #: each name its Python hands to a call, before it ran
    #: (``MutationClassifier.magic_snapshots``); None when nothing is watched.
    magic_snapshots: dict[str, str | None] | None = None
    #: The variables that hold an output's object too, stored and restored
    #: with the outputs as one graph, each with its lineage before the
    #: statement (``StatementCacheMetadata.holders``).
    holders: dict[str, str] = field(default_factory=dict)
    #: Whether the run changed an object that existed before it in place, so
    #: the variables in ``holders`` hold a changed object and their lineages
    #: move on (``StatementCacheMetadata.holders_moved``).
    moves_holders: bool = False
    #: The ``holders`` the run left as they were, by their content digest:
    #: they keep their lineage (``lineage_formula.keep_held_lineage``).
    holders_unchanged: set[str] = field(default_factory=set)
    #: The variables whose objects the outputs hold that are stored as the
    #: variable, not copied, each with its content digest: a restore binds
    #: the live object again, when its digest is still this one
    #: (``StatementCacheMetadata.anchors``, `share_group`).
    anchors: dict[str, str] = field(default_factory=dict)
    #: The ``anchors`` and the variables found holding an object of the
    #: outputs beside them: linked to the outputs as holders are.
    anchor_links: set[str] = field(default_factory=set)
    #: For each of the ``anchor_links`` that is not an anchor, the anchors
    #: whose object it held: a restore misses unless it still holds those
    #: live objects (``StatementCacheMetadata.anchor_holds``).
    anchor_holds: dict[str, list[str]] = field(default_factory=dict)
    #: ``id()`` of the object each output and observed receiver was bound to
    #: before the run: an id, not the object, so it adds no reference the
    #: shared-object check would count.
    bound_before: dict[str, int] = field(default_factory=dict)
    #: Whether the entry the lookup found under ``cache_key`` stores holders.
    entry_holders: bool = False
    #: The local modules the text says the statement sets state on
    #: (``module_state_writes``), and the names holding them it has for
    #: outputs (``module_state_names``).
    state_modules: frozenset[str] = frozenset()
    state_names: frozenset[str] = frozenset()
    #: The names bound to pyplot's current figure and axes, when the
    #: statement drew through pyplot (``pyplot_draws``): outputs and inputs
    #: both, so their lineage folds in where they were.
    pyplot_drawn: frozenset[str] = frozenset()
    #: The share check of its outputs ran out of time
    #: (``StatementProcessor._refuse_unrestorable_outputs``), so which other
    #: variables hold them is not known from it.
    share_unchecked: bool = False
    #: The share check of its outputs found a holder that is not a variable
    #: (a library's registry, a closure, a function's defaults): the
    #: variables it found holding them all the same (``shared_objects.GaveUp``),
    #: None when it did not give up.
    share_gave_up: GaveUp | None = None
    #: The local modules running it was seen rebinding a global of
    #: (``callee_reach.rebound_modules``).
    rebound_modules: frozenset[str] = frozenset()
    #: What of the process it changes (``callee_reach.process_state_writes``:
    #: the environment, the working directory).
    process_state: frozenset[str] = frozenset()
    #: Its analysis budget (`analysis_budget`), from what it last took.
    analysis_budget: AnalysisBudget | None = None
    #: Seconds its analyses took so far, before and after it ran.
    analysis_spent: float = 0.0
    #: Its analysis ran past the budget: it runs uncached, and what it binds
    #: is keyed by value, as a ``# @cash:no-cache`` statement's is.
    past_budget: bool = False
    #: The working directory and the environment before it ran, kept for a
    #: statement past its budget, whose writes to them are seen, not read.
    process_before: tuple[str, dict[str, str]] | None = None
    #: Shadow mode (``cash.notebook.shadow``): the verdict of its traces,
    #: waiting for the lookup's outcome. Never read by anything else.
    shadow: Any = None
    #: It looks names up by a string, which its trace follows
    #: (``cache_key.lookups_followed``): served only on a checked trace.
    needs_trace: bool = False
    #: Seconds recording what it ran took (``Shadow.begin`` / ``end``), inside
    #: its timed run but cash's cost, not the statement's.
    record_time: float = 0.0
    #: The edges its run linked (``StatementCacheMetadata.share_edges``),
    #: or None when not recorded.
    linked_edges: frozenset[tuple[str, str]] | None = None
    #: Cached only if pyplot's code does not run in it: why it runs every
    #: time when it does (``CacheabilityVerdict.unless_it_draws``).
    unless_it_draws: str | None = None
    #: What shadow mode recorded of its run (``Shadow.begin``), or None.
    trace_unit: Any = None
    #: It ran with a kept value standing in for part of it (``ExpressionMemo``).
    memo_used: bool = False
    #: It ran over a loop header cash evaluated before it (``loop_iterable``).
    handed_iterable: bool = False


@dataclass
class StatementExecution:
    """What running a statement's code produced."""

    result: ExecutionResult | None = None
    captured: Any = field(default_factory=NoCapture)
    #: Wall time of the run, as the badge shows it.
    wall_time: float = 0.0
    #: What the statement's code cost: the wall time less cash's own time
    #: inside it, plus what the calls it served from the cache would have cost.
    #: What a hit is credited with, and the entry's ``execution_time``.
    cost: float = 0.0
    #: What storing its value saves: ``cost`` less the calls the cache holds,
    #: plus restoring their results (``CallRouting.store_cost``). What decides
    #: whether the value is stored.
    store_cost: float = 0.0
    #: Cash's own seconds inside the run.
    tax: float = 0.0
    accessed_files: set[str] = field(default_factory=set)
    accessed_remote: set[str] = field(default_factory=set)
    #: The paths it checked for without reading them, with their answers
    #: (``probe_answers``): ``absent``, or what the found path was asked to be.
    probes: dict[str, str] = field(default_factory=dict)
    #: Files the statement was seen writing (``write_observer``).
    written_paths: frozenset[str] = frozenset()
    #: Those of ``written_paths`` it only appended to.
    appended_paths: frozenset[str] = frozenset()
    #: Those of ``written_paths`` written only from a line the user waived.
    waived_paths: frozenset[str] = frozenset()
    #: The `FileAccessTracker` of the run, for the store: which files moved
    #: since the statement read them, and their digests as it read them.
    file_tracker: Any = None
    #: Files and URLs read only inside calls the cache holds.
    cached_call_reads: frozenset[str] = frozenset()
    #: The value the statement echoed as its cell's result, in a one-item
    #: tuple; empty when it echoed nothing.
    echo: tuple[Any, ...] = ()
    #: Whether it printed, warned or displayed anything before its value
    #: was shown: a value reused elsewhere would lose that.
    printed: bool = False
    #: Where it left the global RNG streams it moved or seeded, and only
    #: those (`StatementRandomness.moved_streams`): what a hit replays.
    rng_moved: dict[str, Any] = field(default_factory=dict)
    #: The global streams it seeded as it ran (`reseeded_streams`).
    rng_reseeded: frozenset[str] = frozenset()


def echoes(code: str, tree: ast.Module | None, is_last: bool) -> bool:
    """Whether the statement *code* (parsed: *tree*) echoes its value as the
    cell's result, as :class:`CodeRunner` decides it."""
    return (
        is_last
        and tree is not None
        and bool(tree.body)
        and isinstance(tree.body[-1], ast.Expr)
        and not code.rstrip().endswith(";")
    )


def _future_flags_of_the_cells() -> int:
    """The ``from __future__ import ...`` features the notebook's cells asked for.

    Compiling with ``dont_inherit`` keeps this module's own
    ``from __future__ import annotations`` out of the user's code (which made
    ``x: Undefined = 1`` run without a ``NameError``); the user's own are
    what IPython's compiler remembers.
    """
    from IPython.core.interactiveshell import InteractiveShell

    if not InteractiveShell.initialized():
        return 0
    return attribute_or(InteractiveShell.instance().compile, "flags", 0) & _FUTURE_MASK


def _remember_future_imports(tree: ast.Module | None) -> None:
    """Hand a ``from __future__ import x`` this statement ran to IPython's
    compiler, which is what makes it apply to the cells after it."""
    if tree is None:
        return
    flags = 0
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            flags |= sum(attribute_or(attribute_or(__future__, a.name, None), "compiler_flag", 0) for a in node.names)
    if flags:
        from IPython.core.interactiveshell import InteractiveShell

        if InteractiveShell.initialized():
            InteractiveShell.instance().compile.flags |= flags


_FUTURE_MASK = functools.reduce(
    operator.or_, (getattr(__future__, name).compiler_flag for name in __future__.all_feature_names), 0
)


class CodeRunner:
    """Compiles a statement and runs it in the user namespace.

    A statement ending in a bare expression runs as two units, the body and
    the expression, so the expression's value can be echoed the way IPython
    echoes a cell's last line -- but only for the statement that ends its cell
    (*is_last*), and never after a trailing ``;``.
    """

    def __init__(self, code: str, source: str, tree: ast.Module | None, is_last: bool, namespace: dict) -> None:
        self.code = code
        self.source = source
        self.tree = tree
        self.is_last = is_last
        self.namespace = namespace
        self.execution = StatementExecution()
        self._echo = False

    def _units(self, flags: int) -> list[types.CodeType]:
        flags |= _future_flags_of_the_cells()
        tree = self.tree
        if tree is None:
            try:
                tree = ast.parse(self.source)
            except SyntaxError:
                tree = None
            self.tree = tree
        # One linecache-registered filename per statement, so a traceback
        # inside a function defined here shows its source.
        filename = register_cell_source(self.source)
        # At its line in the cell, as IPython compiles the whole cell.
        lines = line_offset()
        if not (tree and tree.body and isinstance(tree.body[-1], ast.Expr)):
            return [shifted(compile(self.source, filename, "exec", flags=flags, dont_inherit=True), lines)]
        units = []
        if tree.body[:-1]:
            units.append(
                compile(
                    ast.Module(body=tree.body[:-1], type_ignores=[]), filename, "exec", flags=flags, dont_inherit=True
                )
            )
        expression = ast.Expression(body=tree.body[-1].value)
        ast.fix_missing_locations(expression)
        units.append(compile(expression, filename, "eval", flags=flags, dont_inherit=True))
        units = [shifted(unit, lines) for unit in units]
        self._echo = echoes(self.code, tree, self.is_last)
        return units

    def run(self) -> None:
        """Run the statement."""
        value = None
        for unit in self._units(0):
            value = eval(unit, self.namespace, self.namespace)
        _remember_future_imports(self.tree)
        self._show(value)

    async def run_async(self) -> None:
        """Run the statement, awaiting a top-level ``await`` on the running loop.

        Compiled under ``PyCF_ALLOW_TOP_LEVEL_AWAIT``, as IPython's own
        ``run_code`` does. A unit without an ``await`` does not get
        ``CO_COROUTINE`` and runs exactly as :meth:`run` runs it.
        """
        value = None
        for unit in self._units(ast.PyCF_ALLOW_TOP_LEVEL_AWAIT):
            value = eval(unit, self.namespace, self.namespace)
            if unit.co_flags & inspect.CO_COROUTINE:
                value = await value
        _remember_future_imports(self.tree)
        self._show(value)

    def _show(self, value: Any) -> None:
        if not self._echo:
            return
        # Shown once the cell is done, by IPython's display hook, which also
        # fills its output history (``_``, ``Out``): ``CashMagics._showing_result``.
        # A hit needs it as much as the display, so a None is recorded too.
        self.execution.echo = (value,)
        captured = self.execution.captured
        self.execution.printed = bool(
            attribute_or(captured, "stdout", "")
            or attribute_or(captured, "stderr", "")
            or attribute_or(captured, "outputs", None)
        )


def error_result(exception: Exception) -> ExecutionResult:
    """The failed result for *exception*, raised by the statement's code, with
    its traceback cleaned.

    Called from the ``except`` block that caught *exception*.

    Filters out cash framework frames, keeping only user code frames.
    The traceback starts from the first ``<cash>`` frame (where user
    code is compiled and executed) and includes all subsequent frames
    (e.g., user-defined function calls).
    """

    exc_type, exc_value, exc_tb = sys.exc_info()

    # This preserves the user's call chain (e.g., user code calling
    # a user-defined function) while dropping cash internals above.
    clean_tb = None
    tb = exc_tb
    while tb is not None:
        frame = tb.tb_frame
        filename = frame.f_code.co_filename
        if is_cash_filename(filename):
            clean_tb = tb
            break
        tb = tb.tb_next

    if clean_tb is None:
        clean_tb = exc_tb

    e_with_clean_tb = exc_value.with_traceback(clean_tb)
    formatted_tb = "".join(traceback.format_exception(exc_type, exc_value, clean_tb))

    return ExecutionResult(
        success=False,
        error=e_with_clean_tb,
        tb_string=formatted_tb,
    )
