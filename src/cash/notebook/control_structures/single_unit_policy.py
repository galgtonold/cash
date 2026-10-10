"""When a ``for`` loop runs as ONE cache entry instead of per iteration.

Per-iteration decomposition adds a fixed cost per body statement (AST
analysis, cache key, mutation detection, capture, cache I/O). For a long loop
of cheap statements that cost dwarfs the work, so :func:`should_run_as_single_unit`
sends such a loop to the orchestrator's single-unit path instead.

A header is evaluated once, as plain Python evaluates it. Usually the handler
evaluates it to size the loop, and the unit iterates that value. A header
:func:`runs_whole_unevaluated` can size from its text is left to the unit
instead, which then evaluates it -- only when the unit runs, so a cache hit
skips it; :func:`header_may_be_left_to_the_unit` is the guard for that.

Pure functions of the loop's AST, its evaluated iterable and the user
namespace: nothing here runs a statement, a property or a cache.
"""

from __future__ import annotations

import ast
import builtins as _builtins
import functools
import inspect
import io
import logging
import os
import types
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

# Builtin callables that PRODUCE an iterable without side effects, so a
# header built only from them may be left to the unit to evaluate
# (`runs_whole_unevaluated`): skipping it on a cache hit loses nothing. Any
# other call -- a notebook function like ``drain()``, a method like
# ``inbox.drain()`` or ``np.random.permutation(n)`` -- may draw, consume or
# change state, so such a header is evaluated first and its value handed to
# the unit.
#
# The builtins that compute a BOUND are here too, not only the ones that
# produce the iterable. `for t in range(0, len(frame), STEP):` is about
# the commonest loop header there is, and without `len` on this list it
# would be evaluated first, for nothing.
#
# A name on this list is only trusted while it still IS the builtin -- see
# `header_may_be_left_to_the_unit`. And none of these can drain a one-shot
# iterator unseen, because every value the header reads is checked
# for being one; that check, not this list, is what stops `sorted(g)`
# being skipped on a hit.
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

#: Methods that build a FRESH iterator over their object on every call: a
#: header calling one is sized from the object.
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


def file_in_progress_bar(iterable: Any) -> io.IOBase | None:
    """The open file a progress bar wraps (``tqdm(open(path))``), else None.

    The bar has no length of its own and the header's ``open(...)`` call
    cannot be evaluated to size it, so the file it was handed is read for it.
    """
    if str(type(iterable).__module__).partition(".")[0] != "tqdm":
        return None
    inner = getattr(iterable, "iterable", None)
    return inner if isinstance(inner, io.IOBase) else None


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


def should_run_as_single_unit(
    node: ast.For, iterable: Any, user_ns: dict[str, Any], sizes: dict[str, int | None] | None = None
) -> bool:
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
    n_iterations = estimated_iterations(node.iter, iterable, user_ns, sizes)
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
#: The same sentinel, for callers that size a loop whose header is still unevaluated.
UNEVALUATED = _UNEVALUATED


def runs_whole_unevaluated(node: ast.For, user_ns: dict[str, Any]) -> bool:
    """Whether *node* goes as one unit, decided without evaluating its header.

    The unit then evaluates the header, the only evaluation: building it
    first to size the loop (``for a, b in tqdm(list(zip(actions,
    next_actions))):`` over 2M pairs) costs what plain Python pays once more.
    A header that is a call to a pure builtin producer or a progress bar is
    sized from what it wraps (:func:`estimated_iterations` reads the names'
    lengths, never running a property) and judged from its text and the
    values it reads (:func:`header_may_be_left_to_the_unit`). Anything else
    is evaluated first and its value handed to the unit.
    """
    call = node.iter
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        return False
    name = call.func.id
    builtin = getattr(_builtins, name, None)
    pure = name in PURE_ITER_PRODUCERS and user_ns.get(name, builtin) is builtin
    if not (pure or _is_progress_bar(name, user_ns)):
        return False
    return header_may_be_left_to_the_unit(call, user_ns) and should_run_as_single_unit(node, _UNEVALUATED, user_ns)


def _is_builtin_callable(value: Any) -> bool:
    """A builtin function or type (``str``, ``len``, ``int``): calling it runs
    no notebook code."""
    if isinstance(value, types.BuiltinFunctionType):
        return getattr(value, "__module__", None) == "builtins"
    return isinstance(value, type) and value.__module__ == "builtins"


def header_may_be_left_to_the_unit(iter_node: ast.AST, user_ns: dict[str, Any]) -> bool:
    """Whether the header may be evaluated by the unit alone.

    Such a header is evaluated only when the unit runs: a cache hit skips it.
    That is only right for a header with no effect of its own, so it is
    refused when it:

    * calls anything but a pure builtin producer (:data:`PURE_ITER_PRODUCERS`,
      while the name still is that builtin), a progress bar, or ``open`` for
      reading -- ``drain()``, ``inbox.drain()``, ``np.random.permutation(n)``
      and ``df.sample(n=300)`` may draw or change state;
    * hands a notebook function on to one (``map(f, xs)``,
      ``sorted(xs, key=f)``): its effects are the header's;
    * reads a stored iterator, by name or through an attribute or a constant
      subscript (``sorted(g)``, ``list(b.gen)``, ``list(gens['a'])``): the
      evaluation drains it; or
    * reads an attribute or subscript that cannot be read without running
      code (a property, ``__getattr__``, a ``__getitem__``).

    Such a header is evaluated first and the unit iterates its value, which
    is as correct and costs the same.
    """
    called: set[int] = set()
    for sub in ast.walk(iter_node):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        if not isinstance(func, ast.Name):
            return False  # a method or a call of a call: may draw or change state
        called.add(id(func))
        name = func.id
        if _opens_for_reading(sub, user_ns) or _is_progress_bar(name, user_ns):
            continue  # judged by what it wraps, which the walk below reaches
        if name not in PURE_ITER_PRODUCERS:
            return False
        # ...and only while the name still IS that builtin. A notebook
        # that defines its own `len` or `sorted` gets no benefit of the
        # doubt from sharing the name.
        if name in user_ns and user_ns[name] is not getattr(_builtins, name, None):
            return False
    for sub in ast.walk(iter_node):
        if id(sub) in called or not isinstance(sub, (ast.Name, ast.Attribute, ast.Subscript)):
            continue
        if not isinstance(sub.ctx, ast.Load) or (isinstance(sub, ast.Name) and sub.id not in user_ns):
            continue  # a builtin, a lambda's parameter, or a name the evaluation reports missing
        value = _static_value(sub, user_ns)
        if value is _UNKNOWN:
            return False
        if callable(value) and not _is_builtin_callable(value) and not isinstance(value, types.ModuleType):
            return False  # a notebook function handed on, or a class built per item
        try:
            if _is_own_iterator(value):
                return False
        except TypeError:
            pass  # not iterable: cannot be drained
        except Exception:  # a user __iter__ can raise anything
            logger.debug("[FAST_LOOP] iter(%s) raised", ast.unparse(sub), exc_info=True)
            return False
    return True


#: An attribute or subscript :func:`_static_value` cannot read without
#: running code.
_UNKNOWN = object()


def _plain_attribute(base: Any, attr: str) -> Any:
    """``base.attr`` when it is plain data -- in the instance's ``__dict__``,
    a module global, or a class attribute that is no descriptor -- read
    without running a property, ``__getattr__`` or ``__getattribute__``;
    else :data:`_UNKNOWN`."""
    if isinstance(base, types.ModuleType):
        return vars(base).get(attr, _UNKNOWN)
    getter = inspect.getattr_static(type(base), "__getattribute__", None)
    if getter is not object.__getattribute__ and getter is not type.__getattribute__:
        return _UNKNOWN
    try:
        value = inspect.getattr_static(base, attr)
    except AttributeError:
        return _UNKNOWN  # a `__getattr__` answer, or no such attribute
    try:
        own = object.__getattribute__(base, "__dict__")
    except AttributeError:
        own = None
    if isinstance(own, dict) and attr in own and own[attr] is value:
        return value  # the instance's own data, which Python returns as it is
    if hasattr(type(value), "__get__"):
        return _UNKNOWN  # a property, a method, any descriptor: reading it runs code
    return value


def _static_value(node: ast.AST, user_ns: dict[str, Any]) -> Any:
    """The value of a name followed by attribute reads and constant
    subscripts, read without running any code of the notebook's; else
    :data:`_UNKNOWN`. Subscripts only of exact ``dict``, ``list`` and
    ``tuple``, whose ``__getitem__`` is the builtin's."""
    if isinstance(node, ast.Name):
        return user_ns.get(node.id, _UNKNOWN)
    if isinstance(node, ast.Attribute):
        base = _static_value(node.value, user_ns)
        return _UNKNOWN if base is _UNKNOWN else _plain_attribute(base, node.attr)
    if isinstance(node, ast.Subscript) and _is_constant_index(node.slice):
        base = _static_value(node.value, user_ns)
        if type(base) not in (dict, list, tuple):
            return _UNKNOWN
        index = node.slice
        try:
            if isinstance(index, ast.Slice):
                if type(base) is dict:
                    return _UNKNOWN
                parts = (index.lower, index.upper, index.step)
                return base[slice(*(None if p is None else p.value for p in parts))]
            key = index.value
            if type(base) is dict and type(key) not in (str, int, float, bool, bytes, type(None)):
                return _UNKNOWN
            return base[key]
        except (LookupError, TypeError, ValueError):
            return _UNKNOWN
    return _UNKNOWN


#: The name the header's sizing probe is bound to while the header is evaluated.
SIZE_PROBE_NAME = "__cash_size_probe__"

_SINGLE_ARG_PRODUCERS = ("enumerate", "reversed", "sorted", "list", "tuple", "iter")


def _wrap_sized(node: ast.expr, wrap) -> ast.expr:
    """*node* with each expression :func:`estimated_iterations` sizes by
    reading it (an attribute or subscript chain) passed through *wrap*."""
    if _is_pure_access(node) and not isinstance(node, ast.Name):
        return wrap(node)
    if not isinstance(node, ast.Call):
        return node
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr in FRESH_ITERATOR_METHODS:
        func.value = _wrap_sized(func.value, wrap)
    elif isinstance(func, ast.Name) and node.args:
        if func.id == "zip":
            node.args = [_wrap_sized(a, wrap) for a in node.args]
        elif func.id == "map":
            node.args = [node.args[0], *(_wrap_sized(a, wrap) for a in node.args[1:])]
        elif func.id == "filter":
            if len(node.args) == 2:
                node.args = [node.args[0], _wrap_sized(node.args[1], wrap)]
        else:  # the single-argument producers, and a progress bar of any name
            node.args = [_wrap_sized(node.args[0], wrap), *node.args[1:]]
    return node


@functools.lru_cache(maxsize=512)
def sizing_header(iter_code: str) -> tuple[str, tuple[str, ...]] | None:
    """*iter_code* with each attribute or subscript chain that sizes the loop
    wrapped in a call of :data:`SIZE_PROBE_NAME`, and the text of each; None
    when it has none.

    Evaluating that text instead of the header gives the same value, and
    the probe records the length of each chain AS the header reads it. Sizing
    the loop by reading ``loader.batch`` again ran its property a second
    time, and the loop then iterated a second batch.
    """
    try:
        tree = ast.parse(iter_code, mode="eval")
    except SyntaxError:
        return None
    texts: list[str] = []

    def wrap(sub: ast.expr) -> ast.expr:
        texts.append(ast.unparse(sub))
        probe = ast.Call(
            func=ast.Name(id=SIZE_PROBE_NAME, ctx=ast.Load()),
            args=[sub, ast.Constant(value=len(texts) - 1)],
            keywords=[],
        )
        return ast.copy_location(probe, sub)

    if any(isinstance(n, ast.NamedExpr) for n in ast.walk(tree)):
        return None  # binds a name: evaluated as written
    tree.body = _wrap_sized(tree.body, wrap)
    if not texts:
        return None
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), tuple(texts)


def size_probe(texts: tuple[str, ...], sizes: dict[str, int | None]):
    """The function :data:`SIZE_PROBE_NAME` is bound to: records the length of
    the value it is handed under its text in *sizes*, and returns the value."""

    def probe(value: Any, index: int) -> Any:
        try:
            sizes[texts[index]] = len(value)
        except Exception:  # user __len__; unknown is safe
            sizes[texts[index]] = None
        return value

    return probe


def estimated_iterations(
    iter_node: ast.AST, iterable: Any, user_ns: dict[str, Any], sizes: dict[str, int | None] | None = None
) -> int | None:
    """How many times the loop will run, or ``None`` if it cannot be told.

    ``len(iterable)`` alone missed ``df.itertuples()`` / ``iterrows()``,
    whose value is an iterator with no length, so a long cheap loop over a
    frame never ran as one unit: a 631-iteration loop spent ~9 s in
    per-statement machinery around 0.07 s of work. The length is
    read from what the header iterates instead -- the frame's rows, its
    columns for ``items()``, through ``enumerate``/``zip``/``reversed``/
    ``sorted``/``list``/``tuple``, and ``range`` of constants and plain ints.

    *sizes* holds the lengths the evaluation of the header recorded
    (:func:`sizing_header`). An attribute or subscript it does not hold is
    read only when that runs no code (:func:`_static_value`): never a
    property, ``__getattr__`` or ``__getitem__``, whose second run could
    differ from the one the loop iterates.
    """
    try:
        return len(iterable)
    except TypeError:
        pass
    if isinstance(iterable, io.IOBase):
        return _lines_in_file(iterable)
    wrapped = file_in_progress_bar(iterable)
    if wrapped is not None:
        return _lines_in_file(wrapped)

    def length_of(node: ast.AST) -> int | None:
        if isinstance(node, ast.Name):
            value = user_ns.get(node.id)
            try:
                return len(value) if value is not None else None
            except TypeError:
                return None
        if _is_pure_access(node):
            # `a.var["symbol"]` in `for gid, s in a.var["symbol"].items()`:
            # the length the header's own evaluation recorded. Reading only a
            # plain name left a 200,000-iteration inner loop counted as
            # unknown, so it went through the per-statement machinery: 243 s
            # against 3.8 s.
            text = ast.unparse(node)
            if sizes is not None and text in sizes:
                return sizes[text]
            value = _static_value(node, user_ns)
            if value is _UNKNOWN:
                return None
            try:
                return len(value)
            except TypeError:
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
            if func.id == "range":
                return _length_of_range(node, user_ns)
            if func.id in _SINGLE_ARG_PRODUCERS or _is_progress_bar(func.id, user_ns):
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


def _length_of_range(call: ast.Call, user_ns: dict[str, Any]) -> int | None:
    """``len(range(...))`` for the builtin ``range`` of int constants and names
    bound to plain ints (``range(300)``, ``range(0, n, step)``); else None."""
    if "range" in user_ns and user_ns["range"] is not range:
        return None
    if call.keywords or not 1 <= len(call.args) <= 3:
        return None
    bounds = []
    for arg in call.args:
        if isinstance(arg, ast.Constant):
            value = arg.value
        elif isinstance(arg, ast.Name):
            value = user_ns.get(arg.id)
        elif isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub) and isinstance(arg.operand, ast.Constant):
            value = arg.operand.value
            value = -value if type(value) is int else None
        else:
            return None
        if type(value) is not int:
            return None
        bounds.append(value)
    try:
        return len(range(*bounds))
    except (ValueError, OverflowError):
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
