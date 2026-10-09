"""When a ``for`` loop runs as ONE cache entry instead of per iteration.

Per-iteration decomposition adds a fixed cost per body statement (AST
analysis, cache key, mutation detection, capture, cache I/O). For a long loop
of cheap statements that cost dwarfs the work, so :func:`should_run_as_single_unit`
sends such a loop to the orchestrator's single-unit path instead.

The single-unit path runs the loop FROM SOURCE, which evaluates its header a
second time; :func:`header_safe_to_reevaluate` is the guard that makes that
correct. The split policy (:mod:`.split_policy`) reuses the same guard, so
there is one rule rather than two that can drift.

Pure functions of the loop's AST, its evaluated iterable and the user
namespace: nothing here runs a statement or touches a cache.
"""

from __future__ import annotations

import ast
import builtins as _builtins
import io
import logging
import os
from typing import Any

from ...analysis.cacheability import statement_writes_files

logger = logging.getLogger(__name__)

# Approximate per-statement overhead in seconds (analysis + cache + capture)
# of one process() call.
PER_STMT_OVERHEAD_SEC = 0.008

# Minimum number of iterations before single-unit mode is even considered.
# Loops with few iterations benefit greatly from per-iteration caching
# (granular invalidation, partial re-computation on changes) and the
# absolute overhead is small regardless.
MIN_ITERATIONS_FOR_SINGLE_UNIT = 50

# How many times a loop inside the body is assumed to run per iteration of the
# one around it, when the policy has to guess: it is typically sized by a
# variable the outer iteration binds, so it cannot be read before it runs.
ASSUMED_INNER_ITERATIONS = 10

# Minimum estimated overhead (in seconds) to trigger single-unit mode.
MIN_OVERHEAD_SEC = 1.0

# Builtin callables that PRODUCE an iterable without side effects, so the
# single-unit fast path may re-evaluate the loop header a second time
# (once in the handler, once inside ``execute_as_single_unit``) and get the
# same iteration.  Any *other* call in the loop iterable (a bare user function
# like ``drain()``, or an unknown name) may be a one-shot consumable whose
# second evaluation drains an already-exhausted source — those are routed
# to the per-iteration path, which iterates the single, already-evaluated
# iterator.  Method calls (``df['c'].unique()``, ``d.items()``) are assumed
# to be pure accessors and stay on the fast path so re-iterable containers
# (ndarray/Series/DataFrame/dict views) keep the current behaviour.
#
# The builtins that compute a BOUND are here too, not only the ones that
# produce the iterable. `for t in range(0, len(frame), STEP):` is about
# the commonest loop header there is, and without `len` on this list it
# would be refused the fast path and decomposed per iteration.
#
# A name on this list is only trusted while it still IS the builtin -- see
# `header_safe_to_reevaluate`. And none of these can drain a one-shot
# iterator unseen, because every name the header reads is checked
# for being one; that check, not this list, is what stops `sorted(g)`
# re-draining `g`.
PURE_ITER_PRODUCERS = frozenset(
    {
        "range",
        "sorted",
        "reversed",
        "list",
        "tuple",
        "set",
        "frozenset",
        "dict",
        "enumerate",
        "zip",
        "map",
        "filter",
        "iter",
        "bytes",
        "bytearray",
        "str",
        "len",
        "min",
        "max",
        "abs",
        "round",
        "int",
        "float",
    }
)

#: Methods that build a FRESH iterator over their object on every call.
#: A header calling one may be evaluated twice: the second call iterates
#: the same data again, where a stored iterator would be exhausted.
FRESH_ITERATOR_METHODS = frozenset(
    {
        "itertuples",
        "iterrows",
        "items",
        "iteritems",
        "keys",
        "values",
    }
)

# Call names that read or write a file. A loop calling one is never split
# (see :mod:`.split_policy`).
_FILE_IO_CALLS = frozenset(
    {
        "open",
        "read",
        "write",
        "read_csv",
        "to_csv",
        "read_excel",
        "to_excel",
        "save",
        "load",
        "savez",
        "savetxt",
        "loadtxt",
        "read_parquet",
        "to_parquet",
    }
)

# The names on that list that WRITE. The write analyzer
# (`statement_writes_files`) is the main check for the single-unit path;
# these catch the numpy writers it does not know (`np.savez`, `np.savetxt`).
_FILE_WRITE_CALLS = frozenset({"write", "to_csv", "to_excel", "save", "savez", "savetxt", "to_parquet"})


def _is_the_builtin_open(func: Any) -> bool:
    """The builtin ``open``, or IPython's copy of it.

    IPython puts its own wrapper (``_modified_open``, which only refuses file
    descriptors 0 to 2) under the name ``open`` in every notebook namespace,
    and the file tracker swaps ``builtins.open`` while a statement runs, so
    identity with ``builtins.open`` is never true there. A notebook's own
    ``def open`` lives in ``__main__``; the wrapper carries the builtin's
    module and name.
    """
    return func is _builtins.open or (
        getattr(func, "__name__", None) == "open" and getattr(func, "__module__", None) in ("_io", "io", "builtins")
    )


def _is_progress_bar(name: str, user_ns: dict[str, Any]) -> bool:
    """``tqdm(...)`` and its notebook and auto variants, wrapping the iterable.

    A progress bar iterates what it wraps and draws; evaluating it twice
    draws a second bar and reads the wrapped iterable twice, which is the
    same as evaluating the wrapped iterable twice. Judged by where the name
    comes from, so a notebook's own ``tqdm`` gets no benefit of the doubt.
    """
    func = user_ns.get(name)
    return callable(func) and str(getattr(func, "__module__", "")).partition(".")[0] == "tqdm"


def _opens_for_reading(call: ast.Call, user_ns: dict[str, Any]) -> bool:
    """Is *call* the builtin ``open`` in a read-only mode?

    Each such call opens a NEW handle at the start of the file, so evaluating
    it twice reads the file twice and drains nothing. The mode must be a
    literal: ``open(p, mode)`` could be a write.
    """
    func = call.func
    if not (isinstance(func, ast.Name) and func.id == "open"):
        return False
    if "open" in user_ns and not _is_the_builtin_open(user_ns["open"]):
        return False
    mode: ast.expr | None = call.args[1] if len(call.args) > 1 else None
    for keyword in call.keywords:
        if keyword.arg == "mode":
            mode = keyword.value
    if mode is None:
        return True
    return isinstance(mode, ast.Constant) and isinstance(mode.value, str) and set(mode.value) <= set("rtb")


#: How much of a file `_lines_in_file` reads to learn how long its lines are.
_LINE_SAMPLE_BYTES = 64 * 1024


def _lines_in_file(handle: Any) -> int | None:
    """How many lines iterating *handle* will yield, estimated; None if unknown.

    The file's size over the length of the lines at its start: exact for a
    file that fits the sample, a good guess for a log or a csv, and only ever
    used to decide how the loop is run, never what it computes.
    """
    name = getattr(handle, "name", None)
    if not isinstance(name, (str, bytes, os.PathLike)):
        return None
    try:
        size = os.path.getsize(name)
        with open(name, "rb") as head:
            sample = head.read(_LINE_SAMPLE_BYTES)
    except OSError:
        return None
    if not sample:
        return 0
    lines = sample.count(b"\n")
    if size <= len(sample):
        return lines + (0 if sample.endswith(b"\n") else 1)
    return max(1, size * lines // len(sample))


def should_run_as_single_unit(node: ast.For, iterable: Any, user_ns: dict[str, Any]) -> bool:
    """Run this ``for`` loop as one cacheable unit rather than per iteration?

    Decomposition costs ~8 ms per body statement per iteration (key, mutation
    and side-effect analysis, cache I/O), which can be 100-300x the work of a
    tight numeric loop. True when ALL hold:

    1. More than ``MIN_ITERATIONS_FOR_SINGLE_UNIT`` iterations -- a small
       loop keeps per-iteration granularity, whose overhead is small.
    2. Estimated overhead (iterations x body statements x per-statement
       cost) above ``MIN_OVERHEAD_SEC``.
    3. The body writes, moves or removes no file (:func:`writes_files`).
       Reading is fine: the unit runs under one file tracker, like any
       statement, so its entry depends on every file and folder the loop
       read, and an edited, added or deleted file is a miss.

    Nested loops qualify too: an outer loop's iteration key still moves when
    the inner body changes, because the inner unit's key is part of it.

    The price: a single unit is all-or-nothing, so extending ``range(100)``
    re-runs every iteration. Mutated variables still get a correct lineage
    (``update_lineage_after_execution`` covers the whole loop).
    """
    n_iterations = estimated_iterations(node.iter, iterable, user_ns)
    if n_iterations is None:
        # Generators, iterators without __len__ — can't estimate
        return False
    return worth_one_unit(node, n_iterations)


def worth_one_unit(node: ast.For, n_iterations: int) -> bool:
    """Whether *node* run *n_iterations* times goes as one unit
    (:func:`should_run_as_single_unit`'s rule, for a known count)."""
    # What one pass of the body costs counts a loop inside it ten times over: a
    # loop of 40 over a body that loops over each log line's actions ran
    # 1,100 statements, 18 s of per-statement machinery around 5 ms of work.
    n_body_stmts = count_body_statements(node.body, nested_loop_factor=ASSUMED_INNER_ITERATIONS)
    reach = n_iterations * (ASSUMED_INNER_ITERATIONS if _holds_a_loop(node.body) else 1)

    # Small loops always benefit from per-iteration caching — the
    # absolute overhead is small and granular invalidation is valuable.
    if reach <= MIN_ITERATIONS_FOR_SINGLE_UNIT:
        return False

    estimated_overhead = n_iterations * n_body_stmts * PER_STMT_OVERHEAD_SEC
    if estimated_overhead < MIN_OVERHEAD_SEC:
        return False

    # A loop that changes files keeps per-iteration mode (see
    # `writes_files`). A loop that only reads does not need it: the unit's
    # own file tracker records every file and folder the loop read, as it
    # does for a comprehension. Decomposed, `for f in files: d =
    # pd.read_csv(f)` over 1000 files cost 3.5 ms per statement per
    # iteration: 11 s cold and 4.6 s warm against 1.8 s with cash off.
    if writes_files(node.body):
        return False

    logger.debug(
        "[FAST_LOOP] Estimated overhead: %.1fs (%s iters × %s stmts × %.0fms/stmt)",
        estimated_overhead,
        n_iterations,
        n_body_stmts,
        PER_STMT_OVERHEAD_SEC * 1000,
    )
    return True


def iterations_for_one_unit(node: ast.For) -> int | None:
    """From how many iterations :func:`worth_one_unit` runs *node* as one
    unit; None when it never does (its body writes files).

    For a loop over an iterator of unknown length: once it has run that
    many passes, it is at least that long, and the rest of it goes as one
    unit.
    """
    if writes_files(node.body):
        return None
    n_body_stmts = max(1, count_body_statements(node.body, nested_loop_factor=ASSUMED_INNER_ITERATIONS))
    factor = ASSUMED_INNER_ITERATIONS if _holds_a_loop(node.body) else 1
    n = max(
        1, MIN_ITERATIONS_FOR_SINGLE_UNIT // factor, int(MIN_OVERHEAD_SEC / (n_body_stmts * PER_STMT_OVERHEAD_SEC)) - 1
    )
    while n * factor <= MIN_ITERATIONS_FOR_SINGLE_UNIT or n * n_body_stmts * PER_STMT_OVERHEAD_SEC < MIN_OVERHEAD_SEC:
        n += 1
    return n


def header_names_the_iterator(iter_node: ast.AST, iterable: Any, user_ns: dict[str, Any]) -> bool:
    """Whether the header is a bare name holding *iterable*, an iterator.

    ``it = filter(...)`` then ``for x in it:``. Evaluating the header again
    is a lookup that hands back the same iterator, still where the loop
    left it, so the loop may continue as one unit from source: the iterator
    is drawn from once, item by item, as plain Python draws from it. The
    unit reads an iterator, so it is never stored or served from the cache
    (``drawn_stream_inputs``).
    """
    if not isinstance(iter_node, ast.Name) or user_ns.get(iter_node.id) is not iterable:
        return False
    if not hasattr(type(iterable), "__next__"):
        return False  # not an iterator; asked without running its __iter__
    try:
        return iter(iterable) is iterable
    except Exception:  # a user __iter__ can raise anything; the loop re-raises it
        logger.debug("[FAST_LOOP] iter() on the loop's iterable raised", exc_info=True)
        return False


def _numpy_array_of_a_sequence(call: ast.Call, user_ns: dict[str, Any]) -> bool:
    """``np.array(x)`` / ``np.asarray(x)`` with *x* a name bound to a list,
    tuple, range or array: an array as long as *x* along its first axis.

    ``for n, line in tqdm(enumerate(np.array(lines))):`` over 87k lines could
    not be sized, so it ran pass by pass, about 0.3 s a line against 6-9 s
    for the whole loop plain. Only a sequence counts: ``np.array("abc")`` or
    of a set is a 0-d array, which has no length.
    """
    func = call.func
    if not (
        isinstance(func, ast.Attribute)
        and func.attr in ("array", "asarray", "asanyarray")
        and isinstance(func.value, ast.Name)
        and len(call.args) == 1
        and isinstance(call.args[0], ast.Name)
        and not any(kw.arg in (None, "ndmin") for kw in call.keywords)  # `ndmin=2` adds a leading axis
    ):
        return False
    module = user_ns.get(func.value.id)
    if getattr(module, "__name__", None) != "numpy" or not isinstance(module, type(_builtins)):
        return False
    value = user_ns.get(call.args[0].id)
    return isinstance(value, (list, tuple, range, module.ndarray)) and (
        not isinstance(value, module.ndarray) or value.ndim > 0
    )


def _is_own_iterator(value: Any) -> bool:
    """``iter(value) is value``, asked without running a user ``__iter__``
    when it cannot be so: ``iter`` refuses a result with no ``__next__``,
    so a value whose type has none is never its own iterator."""
    return hasattr(type(value), "__next__") and iter(value) is value


#: Stands for the loop's iterable when the header has not been evaluated:
#: it has no length and cannot be iterated, so only the header's text and
#: the names it reads decide.
_UNEVALUATED = object()


def runs_whole_unevaluated(node: ast.For, user_ns: dict[str, Any]) -> bool:
    """Whether *node* goes as one unit, decided without evaluating its header.

    The unit runs the loop from source, which evaluates the header. Deciding
    first on a header evaluated for the purpose built it twice:
    ``for a, b in tqdm(list(zip(actions, next_actions))):`` over 2M pairs
    built the list twice and drew a second bar, 3.4 s plain against 8.2 s.
    A header that is a call to a pure builtin producer or a progress bar is
    sized from what it wraps (:func:`estimated_iterations` reads the names'
    lengths) and judged safe from its text and the names it reads
    (:func:`header_safe_to_reevaluate`), so it is evaluated once, by the
    unit, as plain Python evaluates it. Anything else is evaluated first, as
    before.
    """
    call = node.iter
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        return False
    name = call.func.id
    builtin = getattr(_builtins, name, None)
    pure = name in PURE_ITER_PRODUCERS and user_ns.get(name, builtin) is builtin
    if not (pure or _is_progress_bar(name, user_ns)):
        return False
    return header_safe_to_reevaluate(call, _UNEVALUATED, user_ns) and should_run_as_single_unit(
        node, _UNEVALUATED, user_ns
    )


def header_safe_to_reevaluate(iter_node: ast.AST, iterable: Any, user_ns: dict[str, Any]) -> bool:
    """Whether the loop header may be safely evaluated a second time.

    The single-unit fast path re-executes the loop from source, evaluating
    ``iter_node`` again after the handler already evaluated it once.
    That is only correct when the second evaluation reproduces the same
    iteration — i.e. the header is a re-iterable container built by a
    side-effect-free expression.

    Returns ``False`` (route to the per-iteration path, which consumes the
    single already-evaluated iterator) when EITHER:

    * the evaluated value is a *self-iterator* — ``iter(x) is x`` — a
      generator, ``map``/``zip``/``filter``/``enumerate``, an open file,
      a csv reader, or ``iter(...)``: iterating it a second time yields
      nothing because the first pass exhausted it; OR
    * the header contains a *call to a bare name that is not a known-pure
      iterable producer* (``drain()``, ``next_batch()``): such a call may
      mutate/consume external state, so a second evaluation returns a
      different (often empty) result.

    Method calls (``df['c'].unique()``, ``d.items()``) are treated as pure
    accessors and kept on the fast path, so re-iterable containers keep
    the current byte-identical behaviour.
    """
    # One-shot self-iterators: re-iterating drains an exhausted source.
    # (Most lack ``__len__`` and never reach the single-unit heuristic, but
    # a custom self-iterator that defines ``__len__`` would — guard it.)
    # A header that is itself a call to a pure producer, or to a method
    # that builds a fresh iterator (`df.itertuples()`, `enumerate(rows)`),
    # makes a NEW one-shot iterator each time it is evaluated, so its value
    # being a self-iterator says nothing about the second evaluation. Only
    # a header that merely names a stored iterator is exhausted by the
    # first -- the walk below still refuses those, and any unknown call.
    fresh = isinstance(iter_node, ast.Call) and (
        (isinstance(iter_node.func, ast.Name) and iter_node.func.id in PURE_ITER_PRODUCERS)
        or (isinstance(iter_node.func, ast.Attribute) and iter_node.func.attr in FRESH_ITERATOR_METHODS)
        or _opens_for_reading(iter_node, user_ns)
    )
    try:
        if not fresh and _is_own_iterator(iterable):
            return False
    except TypeError:
        pass  # not iterable: the loop itself will say so
    except Exception:  # a user __iter__ can raise anything; the loop re-raises it
        logger.debug("[FAST_LOOP] iter() on the loop's iterable raised", exc_info=True)

    for sub in ast.walk(iter_node):
        # A one-shot iterator ANYWHERE in the header, not only as the
        # header. The check above only sees the RESULT, and
        # `sorted(g)` returns a list: the first evaluation drains `g`,
        # the second gets nothing, and the loop runs zero times, on the
        # first run and with a clean EXECUTED badge. A dict lookup, so
        # this costs nothing and runs no user code.
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
            value = user_ns.get(sub.id)
            if value is not None:
                try:
                    if _is_own_iterator(value):
                        return False
                except TypeError:
                    pass  # not iterable: cannot be drained
                except Exception:  # a user __iter__ can raise anything
                    logger.debug("[FAST_LOOP] iter(%s) raised", sub.id, exc_info=True)

        # A bare-name call to anything other than a known side-effect-free
        # builtin may consume/mutate state on re-evaluation.
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
            name = sub.func.id
            if _opens_for_reading(sub, user_ns):
                continue  # a new handle each time (`_opens_for_reading`)
            if _is_progress_bar(name, user_ns):
                continue  # judged by what it wraps, which this walk reaches next
            if name not in PURE_ITER_PRODUCERS:
                return False
            # ...and only while the name still IS that builtin. A notebook
            # that defines its own `len` or `sorted` gets no benefit of the
            # doubt from sharing the name.
            if name in user_ns and user_ns[name] is not getattr(_builtins, name, None):
                return False
    return True


def estimated_iterations(iter_node: ast.AST, iterable: Any, user_ns: dict[str, Any]) -> int | None:
    """How many times the loop will run, or ``None`` if it cannot be told.

    ``len(iterable)`` alone missed ``df.itertuples()`` / ``iterrows()``,
    whose value is an iterator with no length, so a long cheap loop over a
    frame never ran as one unit: a 631-iteration loop spent ~9 s in
    per-statement machinery around 0.07 s of work. The length is
    read from what the header iterates instead -- the frame's rows, its
    columns for ``items()``, through ``enumerate``/``zip``/``reversed``/
    ``sorted``/``list``/``tuple``.
    """
    try:
        return len(iterable)
    except TypeError:
        pass
    if isinstance(iterable, io.IOBase):
        return _lines_in_file(iterable)

    def length_of(node: ast.AST) -> int | None:
        if isinstance(node, ast.Name):
            value = user_ns.get(node.id)
            try:
                return len(value) if value is not None else None
            except TypeError:
                return None
        if _is_pure_access(node):
            # `a.var["symbol"]` in `for gid, s in a.var["symbol"].items()`:
            # attribute reads and constant subscripts on a name, read here
            # to size the loop. Reading only a plain name left a
            # 200,000-iteration inner loop counted as unknown, so it went
            # through the per-statement machinery: 243 s against 3.8 s.
            try:
                value = eval(compile(ast.Expression(node), "<loop-size>", "eval"), {"__builtins__": {}}, dict(user_ns))
                return len(value)
            except Exception:  # user expression; sizing is advisory, unknown is safe
                logger.debug("[FAST_LOOP] could not size %s", ast.unparse(node), exc_info=True)
                return None
        if not isinstance(node, ast.Call):
            return None
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in FRESH_ITERATOR_METHODS:
            base = length_of(func.value)
            if base is None:
                return None
            owner = user_ns.get(func.value.id) if isinstance(func.value, ast.Name) else None
            if func.attr in ("items", "iteritems") and hasattr(owner, "columns"):
                return len(owner.columns)
            return base
        if _numpy_array_of_a_sequence(node, user_ns):
            # `np.array(lines)`: as long as the list along its first axis.
            return len(user_ns[node.args[0].id])
        if isinstance(func, ast.Name) and node.args:
            if func.id in ("enumerate", "reversed", "sorted", "list", "tuple", "iter") or _is_progress_bar(
                func.id, user_ns
            ):
                return length_of(node.args[0])
            if func.id in ("zip", "map"):
                # `map(f, a, b)` stops at the shortest, as `zip` does.
                lengths = [length_of(a) for a in (node.args if func.id == "zip" else node.args[1:])]
                return None if not lengths or any(n is None for n in lengths) else min(lengths)
            if func.id == "filter" and len(node.args) == 2:
                # At most this many: enough to decide how the loop runs.
                return length_of(node.args[1])
        return None

    try:
        return length_of(iter_node)
    except Exception:  # user __len__; no estimate means per-iteration, the safe default
        logger.debug("[FAST_LOOP] could not estimate the iteration count", exc_info=True)
        return None


def count_body_statements(body: list[ast.AST], nested_loop_factor: int = 1) -> int:
    """Count the total number of executable statements in a loop body,
    including statements inside nested control structures.

    A statement in a loop inside the body counts *nested_loop_factor* times:
    it runs that often per pass of the body."""
    count = 0
    for node in body:
        if isinstance(node, ast.If):
            count += count_body_statements(node.body, nested_loop_factor)
            count += count_body_statements(node.orelse, nested_loop_factor)
        elif isinstance(node, (ast.For, ast.While)):
            count += nested_loop_factor * count_body_statements(node.body, nested_loop_factor)
        elif isinstance(node, ast.Try):
            count += count_body_statements(node.body, nested_loop_factor)
            for handler in node.handlers:
                count += count_body_statements(handler.body, nested_loop_factor)
        elif isinstance(node, ast.With):
            count += count_body_statements(node.body, nested_loop_factor)
        else:
            count += 1
    return count


def _holds_a_loop(body: list[ast.AST]) -> bool:
    """Whether a ``for`` or ``while`` statement is anywhere in *body*."""
    return any(isinstance(sub, (ast.For, ast.While)) for node in body for sub in ast.walk(node))


def writes_files(body: list[ast.AST]) -> bool:
    """Whether any statement in the body writes, moves or removes a file.

    Such a loop stays per-iteration. Run as one unit it would be one
    statement that both reads and writes files, and it may read what it
    wrote itself (`open(out).read()` after `df.to_csv(out)`), so the files
    recorded as its inputs would include its own outputs.
    """
    module = ast.Module(body=body, type_ignores=[])
    if any(isinstance(node, ast.Call) and _call_name(node) in _FILE_WRITE_CALLS for node in ast.walk(module)):
        return True
    return statement_writes_files(ast.unparse(module), module)


def has_file_io_calls(body: list[ast.AST]) -> bool:
    """Whether any statement in the body performs file I/O.

    The split policy (:mod:`.split_policy`) never splits such a loop. The
    single-unit policy asks the narrower :func:`writes_files`.
    """
    return any(
        isinstance(node, ast.Call) and _call_name(node) in _FILE_IO_CALLS
        for node in ast.walk(ast.Module(body=body, type_ignores=[]))
    )


def _call_name(node: ast.Call) -> str | None:
    """The called function's name: ``f`` for ``f()`` and ``x.f()``."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _is_constant_index(node: ast.AST) -> bool:
    """A constant, or a slice whose bounds are constants (``[:40]``, ``[5:]``)."""
    if isinstance(node, ast.Slice):
        return all(part is None or isinstance(part, ast.Constant) for part in (node.lower, node.upper, node.step))
    return isinstance(node, ast.Constant)


def _is_pure_access(node: ast.AST) -> bool:
    """A name followed only by attribute reads and constant subscripts
    (``a.var["symbol"]``, ``df.x``): no calls, nothing that could run code but
    a property or ``__getitem__``."""
    if isinstance(node, ast.Attribute):
        return _is_pure_access(node.value)
    if isinstance(node, ast.Subscript):
        return _is_constant_index(node.slice) and _is_pure_access(node.value)
    return isinstance(node, ast.Name)
