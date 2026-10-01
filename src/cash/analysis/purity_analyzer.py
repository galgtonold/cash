"""Decorator-side purity analyzer.

`PurityAnalyzer.analyze` returns the `PurityReport` for a ``@cash.cache``
function: what its body and the user-code helpers it reaches do (the body
rules are in `purity_visitor`), and the source digest and binding of each of
those helpers, which the decorator folds into the cache key (the walk is in
`helper_walk`). Reports are memoised per function, and one analyzer is shared
by every :class:`Cash` instance (`get_analyzer`).

The analysis has no side effects of its own; the one warning it raises says
a function cannot be keyed and runs uncached. The decorator layer turns the
report into warnings, exceptions and cache-key components.
"""

from __future__ import annotations

import dataclasses
import hashlib
import threading
import weakref
from collections.abc import Callable
from typing import Any

from .._memo import PURITY_REPORTS, LruMemo
from ..diagnostics import warn_diagnostic
from ..exceptions import SOURCE_RETRIEVAL_ERRORS, CashCacheIneffectiveWarning
from ..purity import is_stateful
from ..source_reading import getsource
from .helper_bindings import bindings_changed
from .helper_code import qualname_of
from .helper_walk import HelperWalk
from .purity_report import ISSUE_IMPURE_CALL, PurityIssue, PurityReport

__all__ = ["PurityAnalyzer"]


class PurityAnalyzer:
    """Walks a callable's body + its module-bounded helpers and
    returns a :class:`PurityReport`.

    Results are cached by the analyzed callable's source hash.
    Multiple :class:`Cash` instances share a single process-wide
    analyzer via :func:`get_analyzer`.
    """

    def __init__(self) -> None:
        # memo key -> (report, the function it was built from); see `analyze`
        self._cache: LruMemo[str, tuple[PurityReport, weakref.ref | None]] = LruMemo(PURITY_REPORTS)
        self._cache_lock = threading.Lock()

    def analyze(self, func: Callable[..., Any]) -> PurityReport:
        """Return a :class:`PurityReport` for *func*.

        Idempotent and cached by source hash. A function marked ``@pure`` or
        ``@stateful`` is walked for the cache key like any other, but its body
        is not audited: ``@pure`` reports nothing, ``@stateful`` reports the
        function itself.
        """
        memo_key, owner = _memo_key(func)
        if memo_key is not None:
            cached = self._cached_report(memo_key, func)
            if cached is not None:
                return cached

        report = HelperWalk(func).run()
        if report.unwalkable:
            warn_diagnostic(
                CashCacheIneffectiveWarning,
                "KEY-HELPERS-UNWALKABLE",
                f"cash cannot key {qualname_of(func)}: {report.unwalkable}. It runs uncached.",
                "Name the helpers it reaches with depends_on=[...] instead of creating them on every read.",
            )
        if is_stateful(func):
            # The user has spoken: one finding for the function itself.
            report = dataclasses.replace(
                report,
                issues=(
                    PurityIssue(
                        kind=ISSUE_IMPURE_CALL,
                        description="explicitly marked @stateful",
                        where=qualname_of(func),
                        line=0,
                    ),
                ),
            )

        if memo_key is not None:
            with self._cache_lock:
                self._cache[memo_key] = (report, owner)
        return report

    def _cached_report(self, memo_key: str, func: Callable[..., Any]) -> PurityReport | None:
        """The memoised report for *func*, if it still describes it."""
        with self._cache_lock:
            entry = self._cache.get(memo_key)
        if entry is None:
            return None
        cached, cached_owner = entry
        target = getattr(func, "__func__", func)
        built_from = cached_owner() if cached_owner is not None else None
        if built_from is None:
            return None
        if getattr(func, "__closure__", None) and built_from is not target:
            return None
        if getattr(built_from, "__globals__", None) is not getattr(func, "__globals__", None):
            return None
        # The source is the same, but a name it calls through may hold a
        # different object now (a patched helper, or a real one restored):
        # the tree below that binding is not the one this report walked.
        if bindings_changed(cached):
            return None
        return cached


_global_analyzer: PurityAnalyzer | None = None
_global_analyzer_lock = threading.Lock()


def get_analyzer() -> PurityAnalyzer:
    """Return the process-wide :class:`PurityAnalyzer` singleton.

    Multiple :class:`Cash` instances share one analyzer cache so
    redundant AST walks are avoided across instances.
    """
    global _global_analyzer
    if _global_analyzer is not None:
        return _global_analyzer
    with _global_analyzer_lock:
        if _global_analyzer is None:
            _global_analyzer = PurityAnalyzer()
        return _global_analyzer


def _memo_key(func: Callable[..., Any]) -> tuple[str | None, weakref.ref | None]:
    """The key *func*'s report is memoised under, and a reference to the
    function it is built from; ``(None, None)`` when it cannot be memoised."""
    target = getattr(func, "__func__", func)  # a bound method is made anew per access
    source_hash = _try_source_hash(func)
    if source_hash is None:
        return None, None
    # Keyed by the namespace the names resolve in as well as the text:
    # `def run(): return step()` written identically in two modules
    # calls two different `step`s, and sharing one report handed the
    # second module the first one's helpers -- editing its own `step`
    # then changed nothing its key could see.
    source_hash = f"{source_hash}:{id(getattr(func, '__globals__', None))}"
    # An id outlives nothing: a module dropped from `sys.modules` frees
    # its namespace, and a new module with the same text can be given
    # the same address. Its function was then handed the dead one's
    # report, bindings and all -- and a binding into a module that has
    # gone proves nothing (`bindings_changed`), so a helper patched
    # with a mock was never seen and the call was served from the
    # cache. So an entry holds the function it was built from, and
    # serves only while that function is alive in the same namespace.
    try:
        owner = weakref.ref(target)
    except TypeError:
        return None, None
    # A closure's names also resolve in its cells: two closures
    # with the same text in one module (one factory called twice,
    # or two factories) can capture different helpers, and sharing
    # a report keyed the second by the first one's helpers. A
    # report of a closure belongs to that function object alone.
    if getattr(func, "__closure__", None):
        source_hash = f"{source_hash}:{id(func)}"
    return source_hash, owner


def _try_source_hash(func: Callable[..., Any]) -> str | None:
    """Memo key for the analyzer's own report cache -- NOT a cache key.

    Deliberately the raw text, unlike every channel that goes through
    ``source_identity_digest``. Nothing downstream keys on this, so a comment
    or a reformat only means a function is analyzed again. The normalized
    form drops what the key ignores and the report does not: a waiver, the
    ``# @cash:assume-safe`` comment or a ``with cash.assume_safe():`` block.
    Two methods named alike in one module, one of them waived, shared one
    report, and the second got the first one's findings.
    """
    try:
        src = getsource(func)
    except SOURCE_RETRIEVAL_ERRORS:
        return None
    return hashlib.sha256(src.encode("utf-8")).hexdigest()
