"""The environment variables a cached function reads, folded into its key
at their current values."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..dependency_state import ledger_note
from ..effects import environment_component

if TYPE_CHECKING:
    from ..analysis.purity_analyzer import PurityReport
    from .registry import FunctionRegistry


class EnvironmentFold:
    """Folds the environment reads of a cached function, and of the cached
    functions it calls, into the state segment."""

    def __init__(self, registry: FunctionRegistry) -> None:
        self._registry = registry

    def fold_environment(self, func: Callable, func_name: str, state_hash: str) -> str:
        """Fold the current value of every environment read into the key.

        ``os.environ["TENANT"]`` in a cached body served the first tenant's
        answer to every other tenant: the value is an input that never reached
        the key. The analyzer lists the reads whose name is written out
        (`PurityReport.environment_reads`), in the function, its helpers and
        the cached functions it calls -- a dependency's own key moves with
        the variable, but this function's stored result would not. Each is
        read again on every call, and a new value is a new entry.

        Nothing is added when there are none, so such a key is unchanged.
        """
        entries = self._environment_reads(func_name, set(), self._registry.report_for(func, func_name))
        if not entries:
            return state_hash
        component = environment_component(entries, note=lambda label, digest: ledger_note(("env", label), digest))
        return hashlib.sha256(f"{state_hash}{component}".encode("utf-8")).hexdigest()

    def _environment_reads(
        self, func_name: str, visited: set[str], report: PurityReport | None = None
    ) -> set[tuple[str, str]]:
        """The environment reads of *func_name* and every cached function it
        (transitively) depends on (cycle-guarded). *report* is *func_name*'s
        own, when the caller has it (`FunctionRegistry.report_for`)."""
        if func_name in visited:
            return set()
        visited.add(func_name)
        if report is None:
            report = self._registry.purity_reports.get(func_name)
        found = set(getattr(report, "environment_reads", ()) or ())
        for dep in self._registry.graph.get_dependencies(func_name):
            found |= self._environment_reads(dep, visited)
        return found
