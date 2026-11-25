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

from cash._user_attrs import attribute_or, plain_getattr

from ...source_norm import source_identity_digest
from .._protocols import ShellProtocol
from ..cache_key import (
    VirtualCallable,
    called_function_dependencies,
    called_function_globals,
    virtual_callable_key,
    virtual_namespace,
)
from ..lineage_formula import forward_helper
from ...string_lookups import code_looks_up_by_string, statement_looks_up_by_string
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
                obj = attribute_or(module, alias.name, None) if module is not None else None
                if obj is not None and tracker is not None:
                    if not callable(obj):
                        continue
                    try:
                        digest = tracker.get_function_source_hash(obj)
                    except Exception:  # noqa: BLE001 - a digest we cannot take is one we do not claim
                        continue
                    code = attribute_or(obj, "__code__", None)
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
        deps = called_function_dependencies(sorted(inputs), user_ns, variable_lineage, virtual, keyed=False)
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
        return {name for name in self.callee_globals(inputs, virtual_lineage, virtual_modules) if name not in user_ns}

    def callee_globals(
        self,
        inputs: set[str],
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
    ) -> set[str]:
        """Names the callees in *inputs* read, modules included."""
        virtual = (
            virtual_namespace(self.by_lineage, virtual_lineage, self.tracking_state.variable_lineage, virtual_modules)
            if self.by_lineage
            else None
        )
        return called_function_globals(inputs, self.shell.user_ns, virtual, keep_modules=True)

    #: How many functions `reads_names_unseen` follows from one statement.
    _CALLEES_FOLLOWED = 64

    def reads_names_unseen(self, stmt_code: str, inputs: set[str], virtual_lineage: Mapping[str, str]) -> bool:
        """Whether *stmt_code*, or a function of the notebook's it reaches
        through *inputs* (held by the kernel, or a simulated ``def``), looks
        a name up by a string (`string_lookups`): ``eval('K * 10')``,
        ``globals()['K']``. Which variable it reads is then in none of its
        inputs. Past `_CALLEES_FOLLOWED` functions it is taken to."""
        user_ns = self.shell.user_ns
        if statement_looks_up_by_string(stmt_code, user_ns):
            return True
        live = self.tracking_state.variable_lineage
        todo = sorted(inputs)
        seen: set[str] = set()
        while todo:
            name = todo.pop()
            if name in seen:
                continue
            if len(seen) >= self._CALLEES_FOLLOWED:
                return True
            seen.add(name)
            code = None
            if name in user_ns:
                value = user_ns[name]
                if plain_getattr(value, "__globals__", None) is user_ns:
                    code = plain_getattr(value, "__code__", None)
            else:
                lineage = virtual_lineage.get(name) or live.get(name) or ""
                found = self.by_lineage.get(virtual_callable_key(lineage, name))
                code = found.code if found is not None else None
            if not isinstance(code, types.CodeType):
                continue
            if code_looks_up_by_string(code, user_ns):
                return True
            nested = [code]
            while nested:
                inner = nested.pop()
                todo.extend(inner.co_names)
                nested.extend(const for const in inner.co_consts if isinstance(const, types.CodeType))
        return False

    def stale_callee_globals(
        self,
        inputs: set[str],
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
    ) -> set[str]:
        """Helpers defined below a function (`forward_helper`) that the
        callees in *inputs* read and the kernel holds with another lineage
        than the simulation gave them: one edited above and not run yet. The
        function's lineage does not carry them
        (`lineage_formula.def_time_names`), so a statement re-run to rebuild
        a value needs them rebuilt first, as it needs its own inputs."""
        live = self.tracking_state.variable_lineage
        user_ns = self.shell.user_ns
        return {
            name
            for name in self.callee_globals(inputs, virtual_lineage, virtual_modules)
            if name in virtual_lineage
            and name in live
            and live[name] != virtual_lineage[name]
            and forward_helper(name, user_ns)
        }

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
