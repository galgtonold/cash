from __future__ import annotations

"""@pure and @stateful decorator system for caching decisions."""

import functools
import threading
from collections.abc import Callable
from typing import Any, TypeVar

from ..exceptions import SOURCE_RETRIEVAL_ERRORS

__all__ = ["pure", "stateful", "is_pure", "is_stateful", "is_known_pure", "KNOWN_PURE_BUILTINS", "analyze_function_purity", "clear_purity_cache"]

F = TypeVar('F', bound=Callable[..., Any])

# Attribute names used to mark functions
_PURE_ATTR = '_cash_pure'
_STATEFUL_ATTR = '_cash_stateful'

def pure(func: F) -> F:
    """Mark a function as pure (no side effects) for the purity analyzer.

    This is a promise to the analyzer, not a caching switch. Its load-bearing
    effect is on the ``@cash.cache`` decorator: a callee marked pure is trusted,
    so the ``CashImpurityWarning`` that would otherwise fire for it is
    suppressed (see :mod:`cash.purity_analyzer`). In the notebook *statement*
    path it changes no verdict — an unmarked helper's statements already cache,
    and ``@pure`` merely short-circuits the AST heuristic to the same answer.
    Use :func:`stateful` when you need to stop a statement from caching.

    Args:
        func: The function to mark as pure.

    Returns:
        The same function with a ``_cash_pure`` attribute set to True.

    Example::

        @pure
        def compute(x, y):
            return x + y
    """
    setattr(func, _PURE_ATTR, True)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    setattr(wrapper, _PURE_ATTR, True)
    return wrapper  # type: ignore[return-value]  # wrapper preserves F's signature via @wraps

def stateful(func: F) -> F:
    """Mark a function as stateful (has side effects).

    Stateful functions should never be cached because their return value
    alone does not capture their full effect. The notebook caching system
    will skip caching for statements that call stateful functions.

    Args:
        func: The function to mark as stateful.

    Returns:
        The same function with a ``_cash_stateful`` attribute set to True.

    Example::

        @stateful
        def train_model(data):
            model.fit(data)
            return model.score(data)
    """
    setattr(func, _STATEFUL_ATTR, True)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    setattr(wrapper, _STATEFUL_ATTR, True)
    return wrapper  # type: ignore[return-value]  # wrapper preserves F's signature via @wraps

def is_pure(func: Any) -> bool:
    """Check if a function is marked as pure.

    Args:
        func: The function or callable to check.

    Returns:
        True if the function has the ``_cash_pure`` attribute set to True.
    """
    return getattr(func, _PURE_ATTR, False) is True

def is_stateful(func: Any) -> bool:
    """Check if a function is marked as stateful.

    Args:
        func: The function or callable to check.

    Returns:
        True if the function has the ``_cash_stateful`` attribute set to True.
    """
    return getattr(func, _STATEFUL_ATTR, False) is True

# ============================================================================
# Known-pure built-in and stdlib functions
# ============================================================================

# These built-in functions have no side effects and always return
# the same output for the same inputs.
KNOWN_PURE_BUILTINS: frozenset[str] = frozenset({
    # Type constructors / conversions
    'int', 'float', 'str', 'bool', 'bytes', 'complex',
    'list', 'tuple', 'set', 'frozenset', 'dict',
    # Numeric / math
    'abs', 'round', 'pow', 'divmod', 'min', 'max', 'sum',
    # Sequence / iteration
    'len', 'sorted', 'reversed', 'enumerate', 'zip', 'range',
    'map', 'filter', 'all', 'any',
    # Object introspection
    'type', 'isinstance', 'issubclass', 'id', 'hash',
    'callable', 'hasattr', 'getattr',
    'repr', 'ascii', 'format', 'chr', 'ord',
    'hex', 'oct', 'bin',
    # Containers
    'iter', 'next', 'slice',
})

def is_known_pure(name: str) -> bool:
    """Check if a function name is a known-pure built-in.

    This allows the statement processor to skip mutation detection
    for statements that only call known-pure built-in functions,
    even without explicit ``@pure`` annotations.

    Args:
        name: The function name to check.

    Returns:
        True if the name is in the known-pure builtins list.
    """
    return name in KNOWN_PURE_BUILTINS

# ============================================================================
# Automatic purity analysis for user-defined functions
# ============================================================================

import ast
import hashlib
import inspect
import textwrap

# Cache for purity analysis results: func_source_hash -> is_pure
_purity_analysis_cache: dict[str, bool] = {}
_purity_cache_lock = threading.Lock()
_PURITY_CACHE_MAX_SIZE = 200

# Operations that indicate impurity (side effects or global state access)
_IMPURE_FUNCTION_CALLS = frozenset({
    'print', 'input', 'open', 'exec', 'eval', 'compile',
    'exit', 'quit', 'breakpoint',
})

_IMPURE_MODULE_CALLS = frozenset({
    'os.system', 'os.remove', 'os.unlink', 'os.mkdir', 'os.makedirs',
    'os.rename', 'os.replace', 'os.rmdir',
    'subprocess.run', 'subprocess.call', 'subprocess.Popen',
    'subprocess.check_call', 'subprocess.check_output',
    'shutil.copy', 'shutil.copy2', 'shutil.move', 'shutil.rmtree',
    'logging.info', 'logging.debug', 'logging.warning', 'logging.error',
    'logging.critical',
    'requests.get', 'requests.post', 'requests.put', 'requests.delete',
    'json.dump', 'pickle.dump',
})

_WRITE_METHODS = frozenset({
    'write', 'writelines', 'send', 'sendall', 'sendto',
    'append', 'extend', 'insert', 'pop', 'remove', 'sort', 'reverse', 'clear',
    'update', 'add', 'discard',
    'to_csv', 'to_excel', 'to_parquet', 'to_json', 'to_pickle',
    'savefig', 'save',
    'write_text', 'write_bytes',  # pathlib.Path
})

def analyze_function_purity(func: Any, user_ns: dict[str, Any] | None = None) -> bool:
    """Analyze a user-defined function to determine if it's pure.

    A function is considered pure if:
    1. It has no I/O operations (file, network, print)
    2. It doesn't modify global/nonlocal state
    3. It doesn't call known-impure functions
    4. It doesn't use mutable operations on parameters

    This is a conservative analysis — if in doubt, returns False (not pure).
    Cached results are stored by source hash to avoid re-analyzing.

    Args:
        func: The function to analyze.
        user_ns: Optional namespace for resolving called functions.

    Returns:
        True if the function is provably pure, False otherwise.
    """
    # Check explicit decorators first
    if is_pure(func):
        return True
    if is_stateful(func):
        return False

    # Only analyze user-defined functions (not builtins, C extensions, etc.)
    try:
        source = inspect.getsource(func)
    except SOURCE_RETRIEVAL_ERRORS:
        return False  # Can't get source → conservatively impure

    # Check cache
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    with _purity_cache_lock:
        if source_hash in _purity_analysis_cache:
            return _purity_analysis_cache[source_hash]

    # Parse the function source
    try:
        # Dedent to handle methods or nested functions
        source = textwrap.dedent(source)
        tree = ast.parse(source)
    except SyntaxError:
        _cache_purity_result(source_hash, False)
        return False

    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_def = node
            break

    if func_def is None:
        _cache_purity_result(source_hash, False)
        return False

    # Analyze the function body for impurity indicators
    is_func_pure = _check_body_purity(func_def, user_ns)
    _cache_purity_result(source_hash, is_func_pure)
    return is_func_pure

def _cache_purity_result(source_hash: str, is_pure_result: bool) -> None:
    """Cache a purity analysis result, evicting oldest if full."""
    global _purity_analysis_cache
    with _purity_cache_lock:
        if len(_purity_analysis_cache) >= _PURITY_CACHE_MAX_SIZE:
            # Evict first entry (oldest insertion)
            oldest = next(iter(_purity_analysis_cache))
            del _purity_analysis_cache[oldest]
        _purity_analysis_cache[source_hash] = is_pure_result

class _ImpurityVisitor(ast.NodeVisitor):
    """AST visitor that sets ``self.impure = True`` on the first impurity sign."""

    def __init__(self) -> None:
        self.impure = False

    # Early-exit helper — generic_visit is skipped once we're done.
    def _flag(self) -> None:
        self.impure = True

    def visit_Global(self, node: ast.Global) -> None:  # noqa: N802
        self._flag()

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:  # noqa: N802
        self._flag()

    def visit_Yield(self, node: ast.Yield) -> None:  # noqa: N802
        self._flag()

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:  # noqa: N802
        self._flag()

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if isinstance(node.func, ast.Name) and node.func.id in _IMPURE_FUNCTION_CALLS or isinstance(node.func, ast.Attribute) and (
            _get_dotted_name(node.func) in _IMPURE_MODULE_CALLS or node.func.attr in _WRITE_METHODS
        ):
            self._flag()
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        for target in node.targets:
            if isinstance(target, (ast.Attribute, ast.Subscript)):
                self._flag()
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:  # noqa: N802
        if isinstance(node.target, (ast.Attribute, ast.Subscript)):
            self._flag()
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:  # noqa: N802
        for target in node.targets:
            if isinstance(target, (ast.Attribute, ast.Subscript)):
                self._flag()
        self.generic_visit(node)


def _check_body_purity(func_def: ast.AST, user_ns: dict[str, Any] | None = None) -> bool:
    """Check if a function body is pure by analyzing its AST.

    Returns True if no impurity indicators are found.
    """
    visitor = _ImpurityVisitor()
    visitor.visit(func_def)
    return not visitor.impure

def _get_dotted_name(node: ast.Attribute) -> str:
    """Get a dotted name like 'os.path.join' from an Attribute node."""
    parts = [node.attr]
    current = node.value
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return '.'.join(reversed(parts))

def clear_purity_cache() -> None:
    """Clear the purity analysis cache. Useful for testing."""
    global _purity_analysis_cache
    with _purity_cache_lock:
        _purity_analysis_cache.clear()

