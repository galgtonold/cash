"""What must hold of the notebook before the upstream check simulates it.

:class:`NotebookVetter` warns about an upstream cell that does not parse,
evicts a variable whose definition is gone, and refuses a cell that reads a
name only a later cell binds.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import logging
import types

from ...analysis.ast_util import called_names
from ...analysis.code_analyzer import CodeAnalyzer, clean_cell_source, parse_cell_source
from ...diagnostics import log_diagnostic, warn_diagnostic
from ...exceptions import CashUpstreamSyntaxWarning, ForwardReferenceError
from .._protocols import ShellProtocol, TrackingState

__all__ = ["NotebookVetter"]

logger = logging.getLogger(__name__)


def _cell_writes(cell_code: str) -> set[str]:
    """The names a cell binds or changes (by its source); raises SyntaxError
    when it does not parse."""
    tree = parse_cell_source(cell_code)
    if tree is None:
        # ``await`` at the top of a cell parses here and not in the simulation.
        return CodeAnalyzer.analyze_code_block(cell_code)[1]
    return CodeAnalyzer.analyze_code_block(cell_code, tree=tree)[1]


@functools.lru_cache(maxsize=1024)
def _cell_reads(cell_code: str) -> frozenset[str]:
    """The names a cell reads that it does not bind first (by its source)."""
    try:
        clean = CodeAnalyzer.strip_magics(cell_code.replace("\r\n", "\n"))
        inputs, _ = CodeAnalyzer.analyze_code_block(clean)
    except (SyntaxError, ValueError, TypeError):
        return frozenset()
    return frozenset(inputs)


def _bound_by(fn: "ast.AST") -> set[str]:
    """Names a function or lambda binds itself: parameters and local targets.

    Only these can be subtracted safely. A name assigned in the body is bound
    there and never comes from an enclosing cell, so counting it would refuse
    on something no cell above could possibly provide.
    """
    bound: set[str] = set()
    args = getattr(fn, "args", None)
    if args is not None:
        for a in (*getattr(args, "posonlyargs", []), *args.args, *args.kwonlyargs):
            bound.add(a.arg)
        for extra in (args.vararg, args.kwarg):
            if extra is not None:
                bound.add(extra.arg)
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
    return bound


class NotebookVetter:
    """Settles, before a simulation, what the notebook text alone decides."""

    def __init__(self, shell: ShellProtocol, tracking_state: TrackingState) -> None:
        self.shell = shell
        self.tracking_state = tracking_state
        # per-session ledger of already-warned broken upstream cells,
        # keyed by cell index -> cell source hash. Keeps the "cell N has a
        # syntax error" warning to once per distinct break (not once per
        # downstream cell run) while still re-warning when the break changes or
        # a fixed cell is broken again.
        self._warned_broken_cells: dict[int, str] = {}

    def forget_warnings(self) -> None:
        """Re-arm the broken-upstream-cell warning (a notebook switch)."""
        self._warned_broken_cells.clear()

    def vet(self, notebook_cells: list[str], cell_code: str, current_cell_idx: int, required_inputs: set[str]) -> None:
        """What must be settled about the notebook before simulating it.

        Raises ForwardReferenceError for a name only a cell below binds.
        """
        self.tracking_state.read_by_later_cells = frozenset().union(
            *(_cell_reads(code) for code in notebook_cells[current_cell_idx + 1 :])
        )
        # Disclose any unparseable UPSTREAM cell. The simulator skips such a
        # cell so unrelated downstream cells keep caching, but the user must be
        # told which cell is broken — otherwise caching degrades silently
        # mid-edit while the badge and auto_cache_enabled still say it is on.
        self._warn_broken_upstream_cells(notebook_cells, current_cell_idx)
        # A variable whose definition was removed/renamed across an edit is
        # orphaned — no cell produces it anymore. Evict it (and its transitive
        # consumers) so they re-run from the start and raise NameError like a
        # fresh kernel, instead of serving a stale value.
        self._evict_orphaned_definitions(notebook_cells, cell_code)
        # ...and the other half: a name only a cell BELOW binds. Same
        # invisible-while-it-works shape, but it raises.
        self._refuse_forward_references(notebook_cells, cell_code, current_cell_idx, required_inputs)
        logger.debug("[UPSTREAM_DEBUG] Current cell found at index %s", current_cell_idx)

    def _evict_orphaned_definitions(self, notebook_cells: list[str], cell_code: str) -> None:
        """Evict variables whose producing definition no longer exists.

        A variable cash previously produced but that NO current cell statically
        produces is orphaned — its definition was removed, commented out, or
        renamed. It survives in ``user_ns`` with a still-matching lineage, so a
        cached consumer keeps serving a stale value instead of the ``NameError``
        a from-start run would raise. Evict the orphan and its transitive
        consumers from the namespace and all tracking dicts so the consumers
        re-execute their producers (and fail or recompute) on this run.

        Conservative by construction: a candidate must be a real, plainly-named
        user variable currently bound in ``user_ns`` — modules (a ``from X
        import Y`` keeps the tracked source module X in lineage though no cell
        outputs it), cash-internal / dunder names (``_cash_magics``, ``_v``), and
        tracking-only entries with no live binding (an exception ``as e`` that
        Python already unbound) are excluded so they are never wrongly evicted.
        The produced-set spans ALL cells PLUS the current cell (a var produced or
        mutated anywhere is safe — guards an unsaved current cell the on-disk
        notebook view may omit), and if any cell fails to parse the pass is
        skipped rather than risk a wrong eviction.
        """
        produced: set[str] = set()
        for cell in (*notebook_cells, cell_code):
            try:
                produced |= _cell_writes(cell)
            except (SyntaxError, ValueError):
                return  # can't be sure what is produced — do nothing

        user_ns = self.shell.user_ns
        orphaned = {
            v
            for v in set(self.tracking_state.variable_lineage) - produced
            if v in user_ns and not v.startswith("_") and not isinstance(user_ns[v], types.ModuleType)
        }
        if not orphaned:
            return

        # Cascade to transitive consumers via the recorded input lineages.
        to_evict = set(orphaned)
        changed = True
        while changed:
            changed = False
            for var, inputs in self.tracking_state.executed_input_lineages.items():
                if var not in to_evict and not inputs.keys().isdisjoint(to_evict):
                    to_evict.add(var)
                    changed = True

        state = self.tracking_state
        dict_attrs = (
            "executed_input_lineages",
            "current_session_hashes",
            "variable_hashes",
            "variable_sources",
            "executed_cell_codes",
            "executed_cell_hashes",
            "executed_file_deps",
            "granular_preserved_vars",
            "module_attribute_deps",
            "from_import_sources",
            "from_import_components",
        )
        for var in to_evict:
            self.shell.user_ns.pop(var, None)
            state.lineage.discard(var)
            for attr in dict_attrs:
                getattr(state, attr).pop(var, None)
            logger.debug("[UPSTREAM] evicted orphaned variable '%s'", var)

    @staticmethod
    def _module_level_reads(cell_code: str) -> set[str] | None:
        """Names *cell_code* reads when it RUNS, ignoring deferred lookups.

        A name inside a function or class body is resolved when that function
        is called, not when the cell executes -- but that is a premise, not
        the condition. It only says the cell can run in order if the call
        happens AFTER the binding. Two shapes call it inside this very cell
        and so really do read the name now:

        * a lambda handed to a call -- ``s.map(lambda v: f(v))`` invokes it
          inside that statement;
        * a function defined and CALLED in the same cell --
          ``def use_it(): return f(21)`` followed by ``use_it()``.

        A helper that is merely defined here and called from a later cell is
        still deferred, and must stay allowed: judging on the statement's
        inputs, which include those names, refused
        ``test_downward_function_dependency``'s notebook, which runs fine from
        the top. So is a lambda that is stored rather than invoked
        (``handlers = {'x': lambda: f()}``).

        What executes at definition time -- decorators, default arguments,
        base classes -- is collected as before.

        ``None`` when the cell cannot be parsed, which the caller treats as
        "do not refuse anything".
        """
        tree = parse_cell_source(cell_code)
        if tree is None:
            return None

        names: set[str] = set()
        called_here = set(called_names(tree, "eager"))

        def visit(node: ast.AST, into: set[str]) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Evaluated now; the body is not -- unless this cell also
                    # calls it, in which case the body runs before the cell is
                    # over and its free names are read now.
                    for sub in (*child.decorator_list, *child.args.defaults, *(d for d in child.args.kw_defaults if d)):
                        visit_expr(sub, into)
                    if child.name in called_here:
                        _absorb_body(child, into)
                    continue
                if isinstance(child, ast.ClassDef):
                    for sub in (*child.decorator_list, *child.bases):
                        visit_expr(sub, into)
                    continue
                if isinstance(child, ast.Lambda):
                    # Reached other than as a call argument (stored, bound to
                    # a name): only its defaults run now.
                    for sub in (*child.args.defaults, *(d for d in child.args.kw_defaults if d)):
                        visit_expr(sub, into)
                    continue
                if isinstance(child, ast.Call):
                    # A lambda passed to a call is invoked by that call.
                    for arg in (*child.args, *(k.value for k in child.keywords)):
                        if isinstance(arg, ast.Lambda):
                            _absorb_body(arg, into)
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                    into.add(child.id)
                visit(child, into)

        def visit_expr(node: ast.AST, into: set[str]) -> None:
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                into.add(node.id)
            visit(node, into)

        def _absorb_body(fn: ast.AST, into: set[str]) -> None:
            """Free names of *fn*'s body, minus what *fn* itself binds.

            Collected into a scratch set so the parameters can be removed:
            ``lambda v: f(v)`` reads ``f`` from outside and binds ``v``
            itself, and reporting ``v`` would make the guard refuse on a name
            no cell can bind.
            """
            inner: set[str] = set()
            body = fn.body if isinstance(fn.body, list) else [fn.body]
            for stmt in body:
                visit_expr(stmt, inner)
            into |= inner - _bound_by(fn)

        visit(tree, names)
        return names

    def _refuse_forward_references(
        self,
        notebook_cells: list[str],
        cell_code: str,
        current_cell_idx: int,
        required_inputs: set[str],
    ) -> None:
        """Raise when this cell reads a name only a LATER cell binds.

        The sibling of ``_evict_orphaned_definitions``: that one catches a name
        NO cell produces any more, this one a name only a cell BELOW produces.
        Both describe a notebook that cannot reproduce itself, and both are
        invisible while the value happens to be sitting in ``user_ns``.

        A cell read a variable bound in a cell below it. Under
        cash the notebook worked -- the later cell had been run at some point,
        so the name was there -- and a clean in-order run died with
        ``NameError``. Only the uncached oracle caught it; cash reported
        success on a notebook that was already broken.

        It fails rather than warns. A warning would leave cash caching against
        a namespace its own in-order run could not produce, and everything
        keyed on that state would be built on an ordering the notebook does not
        have. ``NameError`` is what the user is going to get anyway; the only
        question is whether they get it now, with the cell number, or on the
        morning they restart.

        Deliberately conservative, because a false positive here stops a cell
        that works:

        * the current cell's own bindings count as above (``x = x + 1``, or a
          name bound by an earlier statement of the same cell);
        * a name bound anywhere above is fine, whatever else rebinds it below
          -- the common shape of a variable set early and reassigned later;
        * if any cell fails to parse, the whole check is skipped rather than
          risk refusing on a half-read notebook.
        """
        if not required_inputs or current_cell_idx is None:
            return
        # Only names this cell reads at MODULE level can break an in-order run.
        # A name referenced inside a `def` is resolved when the function is
        # CALLED, so `def a(n): return b(n) * 2` above `def b` is ordinary
        # Python -- the call site further down runs after both. Judging on
        # `required_inputs`, which includes those deferred free names, refused
        # `test_downward_function_dependency`'s notebook, which runs fine.
        reads = self._module_level_reads(cell_code)
        if reads is None:
            return
        required_inputs = required_inputs & reads
        if not required_inputs:
            return
        above: set[str] = set()
        below: dict[str, int] = {}
        for idx, cell in enumerate((*notebook_cells, cell_code)):
            try:
                outs = _cell_writes(cell)
            except (SyntaxError, ValueError):
                return  # can't be sure what binds what — never refuse on a guess
            if idx <= current_cell_idx or idx >= len(notebook_cells):
                above |= outs  # the current cell's own bindings included
            else:
                for name in outs:
                    below.setdefault(name, idx)

        user_ns = self.shell.user_ns
        forward = sorted(
            (name, below[name]) for name in required_inputs if name in below and name not in above and name in user_ns
        )
        if not forward:
            return
        names = ", ".join(f"`{n}` (cell {i + 1})" for n, i in forward)
        raise ForwardReferenceError(
            f"this cell reads {names}, which nothing above it binds. It works "
            f"right now only because that cell has already run and the name is "
            f"still in memory -- a run from the top, or tomorrow's kernel, "
            f"raises NameError here. cash refuses rather than cache against a "
            f"namespace your own notebook cannot rebuild in order. Move the "
            f"binding above this cell, or move this cell below it."
        )

    def _warn_broken_upstream_cells(
        self,
        notebook_cells: list[str],
        current_cell_idx: int,
    ) -> set[int]:
        """Emit a visible warning for any UPSTREAM cell that cannot be parsed.

        A half-written cell the user has SAVED but not run makes the upstream
        simulator SKIP that cell (see ``VirtualLineage.simulate_one_cell``)
        so unrelated downstream cells keep caching. But the user must
        still be told: the broken cell will not run, and any cell that depends
        on it can no longer have its dependency tracked. Without this, caching
        degrades silently mid-edit while every signal the user has (the badge,
        ``auto_cache_enabled``) still says it is on — the exact trap that cost
        users a long debugging detour.

        Deduped per ``(cell index, cell hash)`` on this checker so a persistent
        break warns once — not on every downstream cell run — but a NEW or
        CHANGED break re-warns, and a fixed cell that is later re-broken warns
        again. Parsing mirrors the simulator exactly (``strip_magics`` then
        ``ast.parse`` on ``\\r\\n``-normalised source) so a VALID cell — the
        multi-line ``%``-format print — is never falsely
        flagged. Returns the set of broken cell indices (0-based).
        """
        broken: dict[int, str] = {}
        for idx in range(min(current_cell_idx, len(notebook_cells))):
            raw = notebook_cells[idx]
            try:
                if not clean_cell_source(raw).strip():
                    continue
            except (ValueError, TypeError):
                continue
            if parse_cell_source(raw) is None:
                broken[idx] = hashlib.sha256(raw.encode("utf-8")).hexdigest()

        for idx, cell_hash in broken.items():
            if self._warned_broken_cells.get(idx) == cell_hash:
                continue  # already warned about this exact break — stay quiet
            raw = notebook_cells[idx]
            snippet = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
            if len(snippet) > 60:
                snippet = snippet[:57] + "..."
            what = (
                f"cell {idx + 1} has a syntax error and could not be parsed "
                f"({snippet!r}), so it is skipped and any cell depending on it "
                f"can no longer be dependency-tracked."
            )
            fix = (
                "fix the syntax error, then re-run that cell and the cells "
                "below that use its output; if it is not really code, delete it "
                "or make it a markdown cell."
            )
            log_diagnostic(logger, "NOTEBOOK-CELL-SYNTAX", what, fix)
            # warn_explicit with registry=None bypasses the "once per location"
            # __warningregistry__ dedupe (every break is raised from this one
            # line); our own per-(idx, hash) ledger supplies the dedupe we
            # actually want, and this still consults the user's filters. Mirrors
            # randomness.py's established pattern.
            warn_diagnostic(
                CashUpstreamSyntaxWarning,
                code="NOTEBOOK-CELL-SYNTAX",
                what=what,
                fix=fix,
                location=("<cash>", idx + 1),
            )

        # Replace the ledger with exactly the current break set: a fixed cell
        # drops out (so a later re-break warns again); a changed break re-warned
        # above and its new hash is recorded here.
        self._warned_broken_cells = broken
        return set(broken)
