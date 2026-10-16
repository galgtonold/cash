"""The environment variables a cached function reads, folded into its key
at their current values, and the installed versions of the libraries it
calls into (`library_versions`)."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..dependency_state import ledger_note
from ..effects import environment_component
from .library_versions import library_versions
from .user_code import own_package

if TYPE_CHECKING:
    from ..analysis.purity_report import PurityReport
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
        if entries:
            component = environment_component(entries, note=lambda label, digest: ledger_note(("env", label), digest))
            state_hash = hashlib.sha256(f"{state_hash}{component}".encode("utf-8")).hexdigest()
        return self._fold_library_versions(func, func_name, state_hash)

    def _fold_library_versions(self, func: Callable, func_name: str, state_hash: str) -> str:
        """Fold the installed version of each library the call reaches.

        A library is keyed by name, its code not read, so ``pip install -U``
        served the old version's results (`library_versions`). The function,
        its helpers and the cached functions it calls are asked
        (`FunctionRegistry.code_functions`), as their stored results would not
        move with a dependency's own key. Nothing is added when they reach no
        library, so such a key is unchanged.
        """
        versions = library_versions(self._registry.code_functions(func, func_name), own_package(func))
        if not versions:
            return state_hash
        for dist, version in versions:
            ledger_note(("installed library", dist), version)
        payload = ",".join(f"{dist}=={version}" for dist, version in versions)
        return hashlib.sha256(f"{state_hash}:libraries:{payload}".encode("utf-8")).hexdigest()

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
