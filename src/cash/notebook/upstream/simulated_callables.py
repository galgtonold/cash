"""Functions and classes the simulation has bound but the kernel does not hold.

After a restart the simulation reaches a ``def`` or a ``from X import f``
before the kernel has run it again. The runtime folds a callee's source digest
into keys and lineages, so the simulation must too: :class:`SimulatedCallables`
remembers each simulated def and imported callable by its lineage and answers
the key builder's questions about them.
"""

from __future__ import annotations

import ast
import base64
import marshal
import sys
import types
from collections.abc import Mapping

from ...source_norm import source_identity_digest
from .._protocols import ShellProtocol
from ..cache_key import (
    VirtualCallable,
    called_function_dependencies,
    called_function_globals,
    virtual_callable_key,
    virtual_namespace,
)
from ..tracking_state import TrackingState
from .cache_probe import CacheProbe

__all__ = ["SimulatedCallables"]


class SimulatedCallables:
    """Simulated defs and imported callables, by lineage."""

    #: See ``by_lineage``.
    MAX_CALLABLES = 4096

    def __init__(
        self,
        shell: ShellProtocol,
        tracking_state: TrackingState,
        probe: CacheProbe,
        function_tracker=None,
    ) -> None:
        self.shell = shell
        self.tracking_state = tracking_state
        self.probe = probe
        self.function_tracker = function_tracker
        #: Simulated ``def``s by lineage (``VirtualCallable``). Content-
        #: addressed, so an entry never goes stale; the cap bounds memory.
        #: The key builder reads it (``CacheKeyContext.virtual_callables``).
        self.by_lineage: dict[str, VirtualCallable] = {}
        #: Classes a simulated ``from X import C`` bound, by lineage -> the
        #: source digest the runtime folds into a lineage (see
        #: ``register_imports``). Not in the key: it skips classes.
        self._imported_classes: dict[str, str] = {}
        #: The text of each simulated ``def``, keyed as ``by_lineage``: what
        #: a call of it seeds (``helper_seeded_modules``).
        self._def_sources: dict[str, str] = {}

    def register_def(self, stmt_code: str, tree: ast.Module | None, virtual_lineage: Mapping[str, str]) -> None:
        """Remember a simulated ``def`` under its lineage (see ``VirtualCallable``).

        *stmt_code* is ``ast.unparse`` of the def, which is also the text the
        runtime compiles it from and ``inspect.getsource`` returns for it, so
        its digest and code are the live function's. A decorated def is left
        out: the name is bound to whatever the decorator returns, whose source
        and code are not the def's.
        """
        if tree is None or len(tree.body) != 1:
            return
        node = tree.body[0]
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.decorator_list:
            return
        lineage = virtual_lineage.get(node.name)
        if not lineage or virtual_callable_key(lineage, node.name) in self.by_lineage:
            return
        try:
            module = compile(stmt_code, "<cash-simulated>", "exec", dont_inherit=True)
        except (SyntaxError, ValueError):
            return
        code = next((c for c in module.co_consts if isinstance(c, types.CodeType) and c.co_name == node.name), None)
        if code is None:
            return
        if len(self.by_lineage) >= self.MAX_CALLABLES:
            self.by_lineage.clear()
            self._def_sources.clear()
        self.by_lineage[virtual_callable_key(lineage, node.name)] = VirtualCallable(
            source_identity_digest(stmt_code), code
        )
        self._def_sources[virtual_callable_key(lineage, node.name)] = stmt_code

    def def_source(self, name: str, virtual_lineage: Mapping[str, str]) -> str | None:
        """The text of the simulated ``def`` *name* is bound to here, or None."""
        lineage = virtual_lineage.get(name)
        return self._def_sources.get(virtual_callable_key(lineage, name)) if lineage else None

    def register_imports(
        self, tree: ast.Module | None, virtual_lineage: Mapping[str, str], stmt_code: str = ""
    ) -> None:
        """Remember what a simulated ``from X import Y`` binds, as the live object would count.

        The runtime folds a source digest into the lineage of every statement
        that reads a callable -- a class included -- and into the key for a
        function. Without it, a statement simulated after a restart, before
        the import has run again (``EXPORTS = Path(...)``), gets a lineage
        the runtime never gave it. The digest comes from the module object
        when it is already imported (``pathlib`` always is), else from what the
        statement bound when it last ran (``CacheProbe.import_bindings``): the
        simulation never imports anything itself.
        """
        if tree is None:
            return
        tracker = self.function_tracker
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
                continue
            module = sys.modules.get(node.module)
            for alias in node.names:
                name = alias.asname or alias.name
                lineage = virtual_lineage.get(name)
                if not lineage or name in self.shell.user_ns:
                    continue
                obj = getattr(module, alias.name, None) if module is not None else None
                if obj is not None and tracker is not None:
                    if not callable(obj):
                        continue
                    try:
                        digest = tracker.get_function_source_hash(obj)
                    except Exception:  # noqa: BLE001 - a digest we cannot take is one we do not claim
                        continue
                    code = getattr(obj, "__code__", None)
                    is_class = isinstance(obj, type)
                else:
                    entry = self.probe.import_bindings(stmt_code).get(name) if stmt_code else None
                    if not entry or entry.get("module"):
                        continue
                    digest, is_class, code = entry.get("digest"), bool(entry.get("is_class")), None
                    if entry.get("code"):
                        try:
                            code = marshal.loads(base64.b64decode(entry["code"]))
                        except (ValueError, EOFError, TypeError):
                            code = None
                if digest is None:
                    continue
                if is_class:
                    self._imported_classes[virtual_callable_key(lineage, name)] = digest
                elif isinstance(code, types.CodeType):
                    self.by_lineage.setdefault(virtual_callable_key(lineage, name), VirtualCallable(digest, code))

    def callee_lineages(
        self,
        inputs: set[str],
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
    ) -> dict[str, str] | None:
        """Lineages, here, of the globals read by callees that exist only as simulated defs."""
        user_ns = self.shell.user_ns
        if not self.by_lineage or all(name in user_ns for name in inputs):
            return None
        variable_lineage = self.tracking_state.variable_lineage
        virtual = virtual_namespace(self.by_lineage, virtual_lineage, variable_lineage, virtual_modules)
        deps = called_function_dependencies(sorted(inputs), user_ns, variable_lineage, virtual)
        found = dict(dep.split(":", 1) for dep in deps)
        return {name: lin for name, lin in found.items() if lin != "ABSENT"} or None

    def absent_callee_globals(
        self,
        inputs: set[str],
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
    ) -> set[str]:
        """Names the callees in *inputs* read that the kernel does not hold.

        A statement re-run to rebuild a value needs them bound, and its own
        inputs do not name them: after a restart ``summary = score(raw)``
        must not re-run without ``score``'s ``OFFSET``, or it raises a
        NameError where the cell ran fine. Modules included: the def's cell
        imported them.
        """
        user_ns = self.shell.user_ns
        virtual = (
            virtual_namespace(self.by_lineage, virtual_lineage, self.tracking_state.variable_lineage, virtual_modules)
            if self.by_lineage
            else None
        )
        names = called_function_globals(inputs, user_ns, virtual, keep_modules=True)
        return {name for name in names if name not in user_ns}

    def source_hashes(self, inputs: set[str], virtual_lineage: Mapping[str, str]) -> dict[str, str]:
        """``name -> source digest`` for callable inputs the kernel does not hold yet:
        simulated defs, and what a simulated ``from X import Y`` bound."""
        if not self.by_lineage and not self._imported_classes:
            return {}
        user_ns = self.shell.user_ns
        found: dict[str, str] = {}
        for name in inputs:
            if name in user_ns:
                continue
            lineage = virtual_lineage.get(name) or self.tracking_state.variable_lineage.get(name) or ""
            key = virtual_callable_key(lineage, name)
            virtual = self.by_lineage.get(key)
            if virtual is not None:
                found[name] = virtual.source_hash
            elif key in self._imported_classes:
                found[name] = self._imported_classes[key]
        return found
