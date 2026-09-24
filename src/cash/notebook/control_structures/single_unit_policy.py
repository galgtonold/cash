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
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Approximate per-statement overhead in seconds (analysis + cache + capture)
# of one process() call.
PER_STMT_OVERHEAD_SEC = 0.008

# Minimum number of iterations before single-unit mode is even considered.
# Loops with few iterations benefit greatly from per-iteration caching
# (granular invalidation, partial re-computation on changes) and the
# absolute overhead is small regardless.
MIN_ITERATIONS_FOR_SINGLE_UNIT = 50

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

# Call names that read or write a file. A loop calling one needs
# per-iteration file-dependency tracking.
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


def should_run_as_single_unit(node: ast.For, iterable: Any, user_ns: dict[str, Any]) -> bool:
    """Run this ``for`` loop as one cacheable unit rather than per iteration?

    Decomposition costs ~8 ms per body statement per iteration (key, mutation
    and side-effect analysis, cache I/O), which can be 100-300x the work of a
    tight numeric loop. True when ALL hold:

    1. More than ``MIN_ITERATIONS_FOR_SINGLE_UNIT`` iterations -- a small
       loop keeps per-iteration granularity, whose overhead is small.
    2. Estimated overhead (iterations x body statements x per-statement
       cost) above ``MIN_OVERHEAD_SEC``.
    3. No file I/O in the body -- file dependencies need per-iteration
       tracking.

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

    # Small loops always benefit from per-iteration caching — the
    # absolute overhead is small and granular invalidation is valuable.
    if n_iterations <= MIN_ITERATIONS_FOR_SINGLE_UNIT:
        return False

    # Count body statements (including nested control structure bodies)
    n_body_stmts = count_body_statements(node.body)
    estimated_overhead = n_iterations * n_body_stmts * PER_STMT_OVERHEAD_SEC
    if estimated_overhead < MIN_OVERHEAD_SEC:
        return False

    # File I/O needs per-iteration tracking
    if has_file_io_calls(node.body):
        return False

    logger.debug(
        "[FAST_LOOP] Estimated overhead: %.1fs (%s iters × %s stmts × %.0fms/stmt)",
        estimated_overhead,
        n_iterations,
        n_body_stmts,
        PER_STMT_OVERHEAD_SEC * 1000,
    )
    return True


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
    )
    try:
        if not fresh and iter(iterable) is iterable:
            return False
    except TypeError:
        pass  # not iterable: the loop itself will say so
    except Exception:  # noqa: BLE001 - a user __iter__ can raise anything; the loop re-raises it
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
                    if iter(value) is value:
                        return False
                except TypeError:
                    pass  # not iterable: cannot be drained
                except Exception:  # noqa: BLE001 - a user __iter__ can raise anything
                    logger.debug("[FAST_LOOP] iter(%s) raised", sub.id, exc_info=True)

        # A bare-name call to anything other than a known side-effect-free
        # builtin may consume/mutate state on re-evaluation.
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
            name = sub.func.id
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
            except Exception:  # noqa: BLE001 - user expression; sizing is advisory, unknown is safe
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
        if isinstance(func, ast.Name) and node.args:
            if func.id in ("enumerate", "reversed", "sorted", "list", "tuple"):
                return length_of(node.args[0])
            if func.id == "zip":
                lengths = [length_of(a) for a in node.args]
                return None if any(n is None for n in lengths) else min(lengths)
        return None

    try:
        return length_of(iter_node)
    except Exception:  # noqa: BLE001 - user __len__; no estimate means per-iteration, the safe default
        logger.debug("[FAST_LOOP] could not estimate the iteration count", exc_info=True)
        return None


def count_body_statements(body: list[ast.AST]) -> int:
    """Count the total number of executable statements in a loop body,
    including statements inside nested control structures."""
    count = 0
    for node in body:
        if isinstance(node, ast.If):
            count += count_body_statements(node.body)
            count += count_body_statements(node.orelse)
        elif isinstance(node, (ast.For, ast.While)):
            count += count_body_statements(node.body)
        elif isinstance(node, ast.Try):
            count += count_body_statements(node.body)
            for handler in node.handlers:
                count += count_body_statements(handler.body)
        elif isinstance(node, ast.With):
            count += count_body_statements(node.body)
        else:
            count += 1
    return count


def has_file_io_calls(body: list[ast.AST]) -> bool:
    """Whether any statement in the body performs file I/O.

    File I/O needs per-iteration tracking for proper dependency
    invalidation, so loops with file operations take neither the
    single-unit fast path nor a split.
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


def _is_pure_access(node: ast.AST) -> bool:
    """A name followed only by attribute reads and constant subscripts
    (``a.var["symbol"]``, ``df.x``): no calls, nothing that could run code but
    a property or ``__getitem__``."""
    if isinstance(node, ast.Attribute):
        return _is_pure_access(node.value)
    if isinstance(node, ast.Subscript):
        return isinstance(node.slice, ast.Constant) and _is_pure_access(node.value)
    return isinstance(node, ast.Name)
