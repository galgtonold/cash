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

from cash.notebook.cache_status import ExecutionResult
from cash.notebook.compiled_source import is_cash_filename, register_cell_source
from cash.notebook.statement.capture import NoCapture

if TYPE_CHECKING:
    import types

    from cash.analysis.annotations import CacheAnnotation
    from cash.analysis.cacheability import StatementAnalysis
    from cash.notebook.statement.results import ProcessResult

__all__ = ["ECHO_FIELD", "CodeRunner", "StatementExecution", "StatementRun", "echoes", "error_result"]

#: The key that carries the value a cell's last expression echoed: in a
#: statement's metrics, and in the cache entry that replays it on a hit.
ECHO_FIELD = "echo_value"


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
    #: Extra outputs to capture and restore (the accumulator-loop fast path).
    force_outputs: set[str] | None = None
    is_last: bool = True

    metrics: ProcessResult = field(default_factory=dict)  # type: ignore[assignment]
    process_start: float = 0.0
    tree: ast.Module | None = None
    effective_ttl: int | None = None
    force_persist: bool = False
    skip_cache: bool = False
    allow_random: bool = False
    #: ``# @cash:cache-fit``: cache a bare estimator fit rather than re-run it.
    cache_fit: bool = False
    unseeded_calls: list = field(default_factory=list)
    inputs: set[str] = field(default_factory=set)
    outputs: set[str] = field(default_factory=set)
    source_hash: str = ""
    cache_key: str = ""
    analysis: StatementAnalysis | None = None
    #: Receivers to content-observe after execution, receivers assumed
    #: mutated, and whether this run learns the statement's mutation verdict.
    mut_observe: set[str] = field(default_factory=set)
    mut_assumed: set[str] = field(default_factory=set)
    mut_record: bool = False
    #: Estimator-fit receivers cached under ``# @cash:cache-fit``.
    est_fit: set[str] = field(default_factory=set)
    #: What actually executes: ``code`` with eligible calls routed through
    #: the call cache. When that rewrite fires, ``exec_source`` is cleared so
    #: the rewritten code is what compiles.
    exec_code: str = ""
    exec_tree: ast.Module | None = None
    #: The input variables holding a random generator that the statement drew
    #: from (``carrier_advances``); None when it read no generator.
    carriers_advanced: set[str] | None = None
    #: For a statement that runs a magic or a shell command: the identity of
    #: every binding in the namespace before it ran (``calls_ipython``).
    ipython_bindings: dict[str, int] | None = None
    #: For a magic statement (``is_magic_statement``): the lineage of each
    #: name it reads, before it ran (``magic_effects``).
    magic_reads: dict[str, str | None] | None = None
    #: The variables that hold an output's object too, stored and restored
    #: with the outputs as one graph, each with its lineage before the
    #: statement (``StatementCacheMetadata.holders``).
    holders: dict[str, str] = field(default_factory=dict)
    #: Whether the run changed an object that existed before it in place, so
    #: the variables in ``holders`` hold a changed object and their lineages
    #: move on (``StatementCacheMetadata.holders_moved``).
    moves_holders: bool = False
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
    #: The local modules running it was seen rebinding a global of
    #: (``callee_reach.rebound_modules``).
    rebound_modules: frozenset[str] = frozenset()
    #: What of the process it changes (``callee_reach.process_state_writes``:
    #: the environment, the working directory).
    process_state: frozenset[str] = frozenset()


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
    #: Files the statement was seen writing (``write_observer``).
    written_paths: frozenset[str] = frozenset()
    #: Files and URLs read only inside calls the cache holds.
    cached_call_reads: frozenset[str] = frozenset()
    #: The value the statement echoed as its cell's result, in a one-item
    #: tuple; empty when it echoed nothing.
    echo: tuple[Any, ...] = ()


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
    return getattr(InteractiveShell.instance().compile, "flags", 0) & _FUTURE_MASK


def _remember_future_imports(tree: ast.Module | None) -> None:
    """Hand a ``from __future__ import x`` this statement ran to IPython's
    compiler, which is what makes it apply to the cells after it."""
    if tree is None:
        return
    flags = 0
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            flags |= sum(getattr(getattr(__future__, a.name, None), "compiler_flag", 0) for a in node.names)
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
        if not (tree and tree.body and isinstance(tree.body[-1], ast.Expr)):
            return [compile(self.source, filename, "exec", flags=flags, dont_inherit=True)]
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
        # IPython's output history (``_``, ``Out``) is filled once the cell is
        # done, from this value (``CashMagics._record_output_history``); a hit
        # needs it as much as the display, so a None is recorded too.
        self.execution.echo = (value,)
        if value is not None:
            from IPython.display import display

            display(value)


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
