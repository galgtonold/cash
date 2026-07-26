"""Harness for running documentation code fences as tests.

Supports two execution paths:

- The plain ``exec()`` path concatenates non-skipped fences into a single
  script and runs them together. Cache claims are inferred by AST analysis.
- The IPython ``InteractiveShell`` path runs each fence as a separate cell
  so magic commands (``%cash_on``, ``%%cash``) and ``{ .nb-cell }`` fences
  can execute as they would in a notebook.

``run_page`` chooses the path automatically based on fence content
(``use_ipy='auto'``).
"""

from __future__ import annotations

import asyncio
import re
import warnings
from ast import PyCF_ALLOW_TOP_LEVEL_AWAIT
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cash.exceptions import CashWarning
from tests.docs._annotations import (
    find_expect_raises_for_fence,
    find_expect_warning_for_fence,
    find_skip_for_fence,
)

# Warning classes that reflect the TEST ENVIRONMENT, not doc quality, so they
# must never fail a page. ``CashNotebookDiscoveryWarning`` fires on every
# ``%cash_on`` page because CI has no live Jupyter Server — its own message says
# it's "expected under papermill / nbconvert / CI". Matched by class name so the
# harness needn't import an internal notebook module. Excluding it keeps the
# check focused on the doc-quality family (ineffective-cache, impurity, randomness).
_ENV_ARTIFACT_WARNING_NAMES: frozenset[str] = frozenset(
    {"CashNotebookDiscoveryWarning"}
)


_FENCE_RE = re.compile(
    r"^```python(?P<attrs>(?:\s+\{[^}]*\})?)\s*$"
    r"(?P<body>.*?)^```\s*$",
    re.MULTILINE | re.DOTALL,
)


@dataclass
class Fence:
    """One python code fence extracted from a markdown file."""

    code: str
    line_start: int  # 1-based, the line containing ```python
    line_end: int    # 1-based, the line containing the closing ```
    attrs: str = ""  # raw attrs string, e.g. "{ .nb-cell }"
    skip: bool = False
    skip_reason: str | None = None
    expect_raises: bool = False
    expect_warning: bool = False

    @property
    def is_nb_cell(self) -> bool:
        return ".nb-cell" in self.attrs


def extract_fences(md_path: Path) -> list[Fence]:
    """Extract every ```python ... ``` fence from a markdown file in source order."""
    text = md_path.read_text(encoding="utf-8")
    fences: list[Fence] = []
    # Walk line-by-line so we get accurate line numbers.
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^```python(?P<attrs>(?:\s+\{[^}]*\})?)\s*$", line)
        if m:
            attrs = m.group("attrs").strip()
            start_line = i + 1  # 1-based
            body_lines: list[str] = []
            j = i + 1
            while j < len(lines) and lines[j].rstrip() != "```":
                body_lines.append(lines[j])
                j += 1
            end_line = j + 1  # 1-based line of closing ```
            skip_ann = find_skip_for_fence(lines, start_line)
            expect_raises = find_expect_raises_for_fence(lines, start_line)
            expect_warning = find_expect_warning_for_fence(lines, start_line)
            fences.append(
                Fence(
                    code="\n".join(body_lines),
                    line_start=start_line,
                    line_end=end_line,
                    attrs=attrs,
                    skip=skip_ann is not None,
                    skip_reason=skip_ann.reason if skip_ann else None,
                    expect_raises=expect_raises,
                    expect_warning=expect_warning,
                )
            )
            i = j + 1
        else:
            i += 1
    return fences


class PageExecutionError(RuntimeError):
    """Raised when a docs-parity page fails to exec."""


class PageUnexercisedError(RuntimeError):
    """Raised when a page defines a ``@cash.cache`` function it never calls.

    Executing a fence proves it doesn't raise; it does not prove the behaviour
    the page teaches actually happens. A definition-only example is how a
    genuinely broken ``DataSource`` once shipped: the fence ran, but nothing
    called the function, so the token was never read and the
    ``CashCacheIneffectiveWarning`` that would have exposed it never fired.
    """


def unexercised_cached_functions(namespace: dict[str, Any]) -> list[str]:
    """Names of ``@cash.cache``-wrapped functions in *namespace* never called.

    A wrapper reports ``hits == misses == 0`` only if it never served a call.
    Note this is per-wrapper, so a notebook page that rebuilds its defining cell
    can legitimately read 0/0 — hence the opt-out marker rather than a hard rule.
    """
    out: list[str] = []
    for name, obj in namespace.items():
        info = getattr(obj, "cache_info", None)
        if not callable(info):
            continue
        try:
            stats = info()
        except Exception:  # noqa: BLE001 - introspection must never break a page
            continue
        if not isinstance(stats, dict):
            continue
        if stats.get("hits", 0) == 0 and stats.get("misses", 0) == 0:
            out.append(name)
    return out


class PageWarningError(RuntimeError):
    """Raised when a docs-parity fence emits a ``CashWarning`` at runtime
    without opting in via ``<!-- test:expect-warning -->``.

    A cash warning (``CashCacheIneffectiveWarning``, ``CashImpurityWarning``,
    ``CashRandomnessWarning``, …) means the documented example is misconfigured
    — e.g. a cache that will never invalidate — so an unannotated one should
    fail the page rather than pass silently.
    """


def _cash_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    """Return ``"ClassName: message"`` for each captured doc-quality
    ``CashWarning`` — environment artifacts (see ``_ENV_ARTIFACT_WARNINGS``)
    are filtered out."""
    out: list[str] = []
    for w in caught:
        cat = w.category if isinstance(w.category, type) else type(w.message)
        if not (isinstance(cat, type) and issubclass(cat, CashWarning)):
            continue
        if any(k.__name__ in _ENV_ARTIFACT_WARNING_NAMES for k in cat.__mro__):
            continue
        out.append(f"{cat.__name__}: {w.message}")
    return out


@dataclass
class ClaimResult:
    claim: "CacheClaim"
    actual_hits: int
    actual_misses: int
    matched: bool


class ClaimMismatchError(AssertionError):
    """Raised when a documented hit/miss claim does not match actual cache state."""


@dataclass
class PageResult:
    page: Path
    total_fences: int
    tested_fences: int
    skipped_fences: list[tuple[int, str]] = field(default_factory=list)
    namespace: dict[str, Any] = field(default_factory=dict)
    claim_results: list[ClaimResult] = field(default_factory=list)


def _apply_inject_comments(script: str) -> str:
    """Replace ``# test:inject: <code>`` comment lines with the code.

    Lines matching ``^(\\s*)# test:inject:\\s*(.+)$`` are replaced with
    ``<indent><code>``, preserving indentation so injections inside function
    bodies remain syntactically valid.  All other lines pass through unchanged.
    """
    out: list[str] = []
    for raw_line in script.splitlines(keepends=True):
        m = re.match(r"^(\s*)# test:inject:\s*(.+)$", raw_line.rstrip("\n"))
        if m:
            indent, code = m.group(1), m.group(2)
            out.append(indent + code + "\n")
        else:
            out.append(raw_line)
    return "".join(out)


def _run_page_ipy(
    md_path: Path,
    fences: list[Fence],
    namespace_overrides: dict[str, Any] | None,
    strict_claims: bool,
) -> PageResult:
    """Run page fences through an in-process IPython ``InteractiveShell``.

    Each fence becomes one ``shell.run_cell()`` call. State persists across
    cells within the page, matching how a notebook would behave. After the
    page completes (success or failure), the shell singleton is cleared so
    the next page starts clean.
    """
    from IPython.core.interactiveshell import InteractiveShell

    shell = InteractiveShell.instance()

    # Register Cash's magics on the shell so %cash_on / %%cash / etc. work.
    # The plain-exec conftest fixture patches Cash.register_magic to a no-op,
    # so we bypass that by importing and registering directly here.
    try:
        from cash.notebook.ipython.magics import CashMagics
        import cash as _cash_mod

        _cash_instance = _cash_mod.Cash()
        magics = CashMagics(shell, _cash_instance)
        shell.register_magics(magics)
    except Exception:
        # If magic registration fails, fences using `%foo` will surface a
        # clear UsageError below — we don't want a registration failure to
        # silently swallow other diagnostic value.
        pass

    if namespace_overrides:
        shell.user_ns.update(namespace_overrides)

    result = PageResult(
        page=md_path,
        total_fences=len(fences),
        tested_fences=0,
    )

    try:
        tested_code_pieces: list[str] = []
        # Track inclusive (start, end) line ranges within the concatenated
        # ``script_for_claims`` that correspond to expect-raises fences. Calls
        # inside those ranges are excluded from claim inference because the
        # try/except may have prevented them from running — and even when they
        # did run, the wrapper may have hit a stale cache rather than the fresh
        # decoration the fence appears to make.
        expect_raises_ranges: list[tuple[int, int]] = []
        _running_line = 1  # 1-based; tracks current line in concatenated script

        for f in fences:
            if f.skip:
                result.skipped_fences.append(
                    (f.line_start, f.skip_reason or "<no reason>")
                )
                continue
            # Note: nb-cell fences are NOT skipped here — they run through IPython.

            code = _apply_inject_comments(f.code)

            if f.expect_raises:
                # Wrap the whole cell in try/except so a raise doesn't abort the page.
                indented = "\n".join("    " + line for line in code.splitlines())
                code = f"try:\n{indented}\nexcept Exception:\n    pass\n"

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", CashWarning)
                cell_result = shell.run_cell(code, store_history=False)

            if (
                cell_result.error_before_exec is not None
                or cell_result.error_in_exec is not None
            ):
                if not f.expect_raises:
                    err = cell_result.error_before_exec or cell_result.error_in_exec
                    raise PageExecutionError(
                        f"{md_path}: cell at line {f.line_start} failed: "
                        f"{type(err).__name__}: {err}"
                    )

            cash_warns = _cash_warnings(caught)
            if cash_warns and not f.expect_warning:
                raise PageWarningError(
                    f"{md_path}: cell at line {f.line_start} emitted cash "
                    f"warning(s) with no test:expect-warning annotation:\n  "
                    + "\n  ".join(cash_warns)
                )

            result.tested_fences += 1
            # Record the (start, end) range of this fence's lines in the
            # concatenated script (used below for expect-raises exclusion).
            fence_lines = f.code.count("\n") + 1
            piece_start = _running_line
            piece_end = _running_line + fence_lines - 1
            if f.expect_raises:
                expect_raises_ranges.append((piece_start, piece_end))
            # +2 for the "\n\n" separator added by "\n\n".join below.
            _running_line = piece_end + 2
            tested_code_pieces.append(f.code)  # raw (pre-inject) for AST claim inference

        result.namespace = dict(shell.user_ns)

        # Claim inference: parse concatenated raw code, check cache_info in shell.user_ns.
        script = "\n\n".join(tested_code_pieces)
        script_with_injects = _apply_inject_comments(script)
        # Strip magic-command lines so ast.parse() doesn't choke on `%foo` / `%%foo`.
        script_for_claims = _strip_magic_lines(script_with_injects)
        try:
            claims = infer_claims(
                script_for_claims, exclude_line_ranges=expect_raises_ranges
            )
        except SyntaxError:
            # If we still can't parse (e.g. some odd magic-laden cell), skip
            # claim inference rather than abort the whole page.
            claims = []

        for claim in claims:
            fn = shell.user_ns.get(claim.function)
            if fn is None or not hasattr(fn, "cache_info"):
                actual_hits, actual_misses = 0, 0
                matched = claim.expected_hits == 0 and claim.expected_misses == 0
            else:
                info = fn.cache_info()
                actual_hits = info.get("hits", 0)
                actual_misses = info.get("misses", 0)
                matched = (
                    actual_hits == claim.expected_hits
                    and actual_misses == claim.expected_misses
                )
            result.claim_results.append(
                ClaimResult(
                    claim=claim,
                    actual_hits=actual_hits,
                    actual_misses=actual_misses,
                    matched=matched,
                )
            )

        if strict_claims:
            mismatches = [r for r in result.claim_results if not r.matched]
            if mismatches:
                lines = [f"{md_path}: cache claim mismatch:"]
                for r in mismatches:
                    lines.append(
                        f"  {r.claim.function}: "
                        f"expected hits={r.claim.expected_hits} "
                        f"misses={r.claim.expected_misses}, "
                        f"got hits={r.actual_hits} misses={r.actual_misses}"
                    )
                raise ClaimMismatchError("\n".join(lines))

        return result
    finally:
        InteractiveShell.clear_instance()


def _strip_magic_lines(source: str) -> str:
    """Comment out IPython magic-command lines so ``ast.parse`` succeeds.

    - Line magics (``%foo ...``): commented out.
    - Cell magic headers (``%%foo ...``): commented out; their body lines are
      preserved verbatim because they are data, not Python syntax.
    - Lines inside a cell-magic body are passed through unchanged.
    """
    out: list[str] = []
    in_cell_magic_body = False
    for raw in source.splitlines(keepends=True):
        stripped = raw.lstrip()
        if in_cell_magic_body:
            # Blank line ends the cell-magic body.
            if stripped == "" or stripped == "\n":
                in_cell_magic_body = False
            out.append(raw)
            continue
        if stripped.startswith("%%"):
            indent = raw[: len(raw) - len(stripped)]
            out.append(f"{indent}# {stripped}")
            in_cell_magic_body = True
            continue
        if stripped.startswith("%"):
            indent = raw[: len(raw) - len(stripped)]
            out.append(f"{indent}# {stripped}")
            continue
        out.append(raw)
    return "".join(out)


def run_page(
    md_path: Path,
    namespace_overrides: dict[str, Any] | None = None,
    strict_claims: bool = True,
    use_ipy: "str | bool" = "auto",
) -> PageResult:
    """Concatenate non-skipped fences, exec, then verify cache claims.

    When ``use_ipy='auto'`` (default), automatically routes through an
    IPython ``InteractiveShell`` when the page contains ``{ .nb-cell }``
    fences or fences whose first non-empty, non-comment line is a magic
    command (``%foo`` or ``%%foo``). Set ``use_ipy=False`` to force the
    plain ``exec()`` path; ``use_ipy=True`` forces the IPython path.
    """
    fences = extract_fences(md_path)

    if use_ipy == "auto":
        try:
            import IPython  # noqa: F401

            _ipy_available = True
        except ImportError:
            _ipy_available = False

        if _ipy_available:
            def _fence_needs_ipy(f: Fence) -> bool:
                if f.skip:
                    return False
                if f.is_nb_cell:
                    return True
                # If ANY non-empty non-comment line is a magic command, route
                # the whole page through IPython. Magic commands embedded in
                # otherwise-plain Python blocks (e.g. ``import cash`` followed
                # by ``%cash_on``) require the IPython execution path because
                # ``exec()`` rejects ``%foo`` as a SyntaxError.
                for line in f.code.splitlines():
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    if stripped.startswith("%"):
                        return True
                return False

            use_ipy_actual = any(_fence_needs_ipy(f) for f in fences)
        else:
            use_ipy_actual = False
    else:
        use_ipy_actual = bool(use_ipy)

    if use_ipy_actual:
        return _run_page_ipy(md_path, fences, namespace_overrides, strict_claims)

    result = PageResult(
        page=md_path,
        total_fences=len(fences),
        tested_fences=0,
    )

    pieces: list[str] = []
    expect_raises_ranges: list[tuple[int, int]] = []
    for f in fences:
        if f.skip:
            result.skipped_fences.append((f.line_start, f.skip_reason or "<no reason>"))
            continue
        if f.is_nb_cell:
            result.skipped_fences.append(
                (f.line_start, "nb-cell: requires IPython kernel (PR2+ scope)")
            )
            continue
        pad = max(0, f.line_start - sum(p.count("\n") + 2 for p in pieces) - 1)
        # Compute the start line of this fence's body in the assembled script.
        # The script line numbers match md line numbers because of padding.
        body_start = f.line_start + 1  # first body line (after the ```python line)
        body_end = f.line_end - 1  # last body line (before the closing ```)
        if f.expect_raises:
            indented = "\n".join("    " + line for line in f.code.splitlines())
            wrapped = "try:\n" + indented + "\nexcept Exception:\n    pass"
            pieces.append("\n" * pad + wrapped)
            # Track the original body line range so infer_claims can skip
            # calls inside it (they may not execute due to the raise).
            expect_raises_ranges.append((body_start, body_end))
        else:
            pieces.append("\n" * pad + f.code)
        result.tested_fences += 1

    if not pieces:
        return result

    script = "\n\n".join(pieces)
    script = _apply_inject_comments(script)
    namespace: dict[str, Any] = {"__name__": "__cash_docs_test__"}
    if namespace_overrides:
        namespace.update(namespace_overrides)

    # The plain path concatenates every fence into one script, so warnings
    # can't be attributed to a single fence — allow them page-wide if ANY
    # non-skipped fence opts in via test:expect-warning.
    page_allows_warning = any(f.expect_warning for f in fences if not f.skip)
    try:
        code_obj = compile(
            script,
            str(md_path),
            "exec",
            flags=PyCF_ALLOW_TOP_LEVEL_AWAIT,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", CashWarning)
            # If the compiled code is a coroutine (top-level await), run it.
            maybe_coro = eval(code_obj, namespace)
            if asyncio.iscoroutine(maybe_coro):
                asyncio.run(maybe_coro)
    except Exception as e:
        raise PageExecutionError(
            f"{md_path}: exec failed with {type(e).__name__}: {e}"
        ) from e

    cash_warns = _cash_warnings(caught)
    if cash_warns and not page_allows_warning:
        raise PageWarningError(
            f"{md_path}: fence emitted cash warning(s) with no "
            f"test:expect-warning annotation:\n  " + "\n  ".join(cash_warns)
        )

    result.namespace = namespace

    # Now verify cache claims by introspecting the decorated functions left
    # in `namespace`. For each `@cash.cache` function the harness inferred,
    # call .cache_info() and compare with the claim.
    claims = infer_claims(script, exclude_line_ranges=expect_raises_ranges)
    for claim in claims:
        fn = namespace.get(claim.function)
        if fn is None or not hasattr(fn, "cache_info"):
            # Function not in the post-exec namespace (e.g. stateful funcs
            # whose decorator returns the original). Skip the assertion;
            # mark as matched only if claim expected 0 hits/0 misses.
            actual_hits, actual_misses = 0, 0
            matched = claim.expected_hits == 0 and claim.expected_misses == 0
        else:
            info = fn.cache_info()
            actual_hits = info.get("hits", 0)
            actual_misses = info.get("misses", 0)
            matched = (
                actual_hits == claim.expected_hits
                and actual_misses == claim.expected_misses
            )
        result.claim_results.append(
            ClaimResult(
                claim=claim,
                actual_hits=actual_hits,
                actual_misses=actual_misses,
                matched=matched,
            )
        )

    if strict_claims:
        mismatches = [r for r in result.claim_results if not r.matched]
        if mismatches:
            lines = [f"{md_path}: cache claim mismatch:"]
            for r in mismatches:
                lines.append(
                    f"  {r.claim.function}: "
                    f"expected hits={r.claim.expected_hits} misses={r.claim.expected_misses}, "
                    f"got hits={r.actual_hits} misses={r.actual_misses}"
                )
            raise ClaimMismatchError("\n".join(lines))

    return result


import ast


@dataclass
class CacheClaim:
    function: str
    expected_hits: int
    expected_misses: int


# Patterns that downgrade a call to "expected to miss" via comment proximity.
_MISS_HINT = re.compile(
    r"#.*?(cache\s*miss|first\s+call|fresh|computed|EXECUTED|recompute|API\s+call|invalidat)",
    re.IGNORECASE,
)
# Patterns that mark a call as "expected to hit" via comment proximity.
_HIT_HINT = re.compile(
    r"#.*?(cache\s*hit|second\s+call|instant|RESTORED|cached)",
    re.IGNORECASE,
)


def _is_cache_decorator(deco: ast.expr) -> bool:
    """Match @cash.cache, @cache, @c.cache, @app.cache (any 'cache' attribute name)."""
    if isinstance(deco, ast.Attribute) and deco.attr == "cache":
        return True
    if isinstance(deco, ast.Name) and deco.id == "cache":
        return True
    if isinstance(deco, ast.Call):
        return _is_cache_decorator(deco.func)
    return False


def _is_stateful_decorator(deco: ast.expr) -> bool:
    if isinstance(deco, ast.Attribute) and deco.attr == "stateful":
        return True
    if isinstance(deco, ast.Name) and deco.id == "stateful":
        return True
    return False


def _get_ancestor_types(tree: ast.AST) -> dict[int, set[type]]:
    """Build a map from node id -> set of ancestor node types."""
    ancestors: dict[int, set[type]] = {}

    def _walk(node, parent_types):
        ancestors[id(node)] = set(parent_types)
        new_types = parent_types | {type(node)}
        for child in ast.iter_child_nodes(node):
            _walk(child, new_types)

    _walk(tree, set())
    return ancestors


def _find_main_guard_nodes(tree: ast.AST) -> set[int]:
    """Return IDs of all AST nodes inside ``if __name__ == "__main__"`` blocks.

    Matches both orderings of the comparison:

    - ``__name__ == "__main__"`` (canonical form)
    - ``"__main__" == __name__`` (reversed form; rare but legal)
    """
    guarded_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = node.test
            is_main_guard = False
            if isinstance(test, ast.Compare):
                if (len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
                        and len(test.comparators) == 1):
                    # Canonical: __name__ == "__main__"
                    if (isinstance(test.left, ast.Name) and test.left.id == "__name__"
                            and isinstance(test.comparators[0], ast.Constant)
                            and test.comparators[0].value == "__main__"):
                        is_main_guard = True
                    # Reversed: "__main__" == __name__
                    elif (isinstance(test.left, ast.Constant)
                            and test.left.value == "__main__"
                            and isinstance(test.comparators[0], ast.Name)
                            and test.comparators[0].id == "__name__"):
                        is_main_guard = True
            if is_main_guard:
                for child in ast.walk(node):
                    guarded_ids.add(id(child))
    return guarded_ids


def _enclosing_functions(tree: ast.AST) -> dict[int, list[ast.AST]]:
    """Map each node id to its chain of enclosing function/lambda/class nodes.

    Innermost first. ``ast.walk`` is breadth-first and loses this, but "does
    this call actually run?" is a question about the chain, not about whether
    *some* function encloses it.
    """
    chains: dict[int, list[ast.AST]] = {}

    def visit(node: ast.AST, stack: list[ast.AST]) -> None:
        for child in ast.iter_child_nodes(node):
            chains[id(child)] = list(stack)
            if isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
            ):
                visit(child, [child, *stack])
            else:
                visit(child, stack)

    visit(tree, [])
    return chains


def _top_level_invocation_counts(
    tree: ast.AST, ancestors: dict[int, set], main_guard_ids: set[int]
) -> dict[str, int]:
    """How many times each name is called from module scope.

    Module scope only: a call nested in another body is counted for *that*
    body's own run count, which is what the recursive check in
    :func:`_runs_exactly_once` walks. Calls in loops are deliberately excluded
    rather than counted as one, since the iteration count is unknown.

    ``if __name__ == "__main__":`` bodies are excluded too, and missing that was
    a real bug in the first cut of this: data-engineering.md ends with
    ``run("s3://…")`` under such a guard, so the harness counted a pipeline that
    never executes and inferred a miss for ``aggregate`` that could not happen.
    """
    counts: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        anc = ancestors.get(id(node), set())
        if {ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef} & anc:
            continue
        if {ast.For, ast.While} & anc:
            continue
        if id(node) in main_guard_ids:
            continue
        counts[node.func.id] = counts.get(node.func.id, 0) + 1
    return counts


def _runs_exactly_once(
    node: ast.AST,
    enclosing: dict[int, list[ast.AST]],
    top_level_calls: dict[str, int],
) -> bool:
    """Whether *node* executes exactly once per page run.

    True only when every function enclosing it is a plain ``def``/``async def``
    invoked exactly once from module scope — the ``asyncio.run(main())`` shape.
    A lambda or class body anywhere in the chain returns False: those are not
    invoked by name, so the run count cannot be established.
    """
    chain = enclosing.get(id(node))
    if chain is None:
        return False
    for frame in chain:
        if not isinstance(frame, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False
        if top_level_calls.get(frame.name, 0) != 1:
            return False
    return True


def called_but_uninferable(md_path: Path) -> dict[str, str]:
    """Cached functions the page CALLS but whose calls claim inference can't see.

    Returns ``{function_name: reason}``. Empty is the healthy state.

    This is the narrow question, and getting it narrow matters. A page that
    *defines* a cached function without calling it is a different, already
    triaged category (``unexercised_cached_functions`` + ``_EXERCISE_REQUIRED``),
    and a sweep found most such cases legitimate — a signature reference, a
    network example that must not run. Gating on "this page verifies no claim"
    re-litigates that decision and flags 13 pages.

    What is genuinely wrong is a page that *does* call its cached function — so
    it teaches runtime behaviour — where nothing checks the behaviour, because
    every call site is invisible to :func:`infer_claims`. Two causes:

    * ``shadowed`` — the name is redefined later and all calls sit above the
      last definition. Rebinding makes the earlier function object unreachable,
      so its ``cache_info()`` can never be read. A real limit, not a bug.
    * ``calls-inside-body`` — every call is inside a function or class body,
      which inference skips because such code usually does not run.
    """
    try:
        fences = extract_fences(md_path)
        script = _strip_magic_lines(_apply_inject_comments(
            "\n\n".join(f.code for f in fences if not f.skip)
        ))
        tree = ast.parse(script)
    except (OSError, SyntaxError, ValueError):
        return {}

    ancestors = _get_ancestor_types(tree)
    # Same allowance infer_claims makes, from the same helpers: a call inside a
    # function that is itself invoked exactly once at top level DOES run and IS
    # counted. Sharing the helpers is deliberate -- a detector that disagreed
    # with the inference it reports on would list cases that are actually fine.
    enclosing = _enclosing_functions(tree)
    main_guard_ids = _find_main_guard_nodes(tree)
    top_level_calls = _top_level_invocation_counts(tree, ancestors, main_guard_ids)
    defs: dict[str, list[int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            _is_cache_decorator(d) for d in node.decorator_list
        ):
            defs.setdefault(node.name, []).append(node.lineno)
    if not defs:
        return {}

    calls: dict[str, list[tuple[int, bool, bool]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = (
            node.func.id if isinstance(node.func, ast.Name)
            else node.func.attr if isinstance(node.func, ast.Attribute)
            else None
        )
        if name not in defs:
            continue
        anc = ancestors.get(id(node), set())
        in_body = bool(
            {ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef} & anc
        )
        if in_body and _runs_exactly_once(node, enclosing, top_level_calls):
            in_body = False
        if id(node) in main_guard_ids:
            in_body = True  # a __main__ guard never runs here
        in_loop = bool({ast.For, ast.While} & anc)
        calls.setdefault(name, []).append((node.lineno, in_body, in_loop))

    out: dict[str, str] = {}
    for name, linenos in defs.items():
        sites = calls.get(name, [])
        if not sites:
            continue  # never called - the other category, not ours
        last_def = max(linenos)
        after = [s for s in sites if s[0] > last_def]
        if any(not in_body and not in_loop for _, in_body, in_loop in after):
            continue  # at least one usable call site: inference sees this one
        if not after:
            out[name] = (
                f"shadowed - {len(linenos)} definitions, all {len(sites)} call(s) "
                f"above the last one (line {last_def})"
            )
        elif all(in_body for _, in_body, _ in after):
            out[name] = f"every call ({len(after)}) is inside a function or class body"
        else:
            out[name] = f"every call ({len(after)}) is inside a loop"
    return out


def infer_claims(
    source: str,
    exclude_line_ranges: list[tuple[int, int]] | None = None,
) -> list[CacheClaim]:
    """Parse the source, find @cash.cache-decorated functions, and infer the
    expected (hits, misses) per function based on call counts, unique argument
    tuples, and any inline-comment overrides.

    ``exclude_line_ranges`` is a list of (start, end) inclusive line-number
    ranges in *source* (1-based) whose calls should be excluded from claim
    inference — used for fences annotated with ``test:expect-raises`` where
    the calls may never run.
    """
    tree = ast.parse(source)
    ancestors = _get_ancestor_types(tree)
    main_guard_ids = _find_main_guard_nodes(tree)
    exclude_ranges = list(exclude_line_ranges or [])

    def _in_exclude_range(lineno: int) -> bool:
        return any(start <= lineno <= end for start, end in exclude_ranges)

    # Find all function definitions with their decorators and line numbers.
    # Track the LAST definition line for each function name.
    cached_funcs: set[str] = set()
    stateful_funcs: set[str] = set()
    last_def_line: dict[str, int] = {}  # func_name -> last definition lineno

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            has_cache = any(_is_cache_decorator(d) for d in node.decorator_list)
            has_stateful = any(_is_stateful_decorator(d) for d in node.decorator_list)
            if has_cache:
                cached_funcs.add(node.name)
                last_def_line[node.name] = max(last_def_line.get(node.name, 0), node.lineno)
            elif has_stateful:
                stateful_funcs.add(node.name)
                last_def_line[node.name] = max(last_def_line.get(node.name, 0), node.lineno)

    # Map each call to its function name and arg tuple (text representation).
    enclosing = _enclosing_functions(tree)
    top_level_calls = _top_level_invocation_counts(tree, ancestors, main_guard_ids)

    lines = source.splitlines()
    calls: dict[str, list[tuple[str, int]]] = {}  # func -> [(args_repr, lineno)]
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Skip calls inside loops
            node_ancestors = ancestors.get(id(node), set())
            if ast.For in node_ancestors or ast.While in node_ancestors:
                continue
            # Calls inside a function or class body only run if the enclosing
            # function is called — usually it isn't, so these are skipped.
            #
            # The exception is the standard async doc shape:
            #
            #     async def main():
            #         await fetch_user(1)      # <- the call we care about
            #     asyncio.run(main())          # <- and it definitely runs
            #
            # Here the call runs exactly once, so skipping it threw away the
            # only evidence those pages offer. ``_runs_exactly_once`` allows the
            # call through only when every enclosing function is itself invoked
            # exactly once at top level; anything else (a lambda, a class body,
            # a helper called twice, or not called at all) still skips, because
            # then the run count is not one and the inferred claim would be
            # wrong in a way that is worse than absent.
            if (ast.Lambda in node_ancestors or ast.ClassDef in node_ancestors):
                continue
            if (ast.FunctionDef in node_ancestors
                    or ast.AsyncFunctionDef in node_ancestors):
                if not _runs_exactly_once(node, enclosing, top_level_calls):
                    continue
            # Skip calls inside if __name__ == "__main__" guards
            if id(node) in main_guard_ids:
                continue
            # Skip calls inside expect-raises fences (may not actually run)
            if _in_exclude_range(node.lineno):
                continue

            target_name = None
            if isinstance(node.func, ast.Name):
                target_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                target_name = node.func.attr
            if target_name is None:
                continue
            if target_name not in cached_funcs and target_name not in stateful_funcs:
                continue
            # Only count calls AFTER the last definition of this function.
            if node.lineno <= last_def_line.get(target_name, 0):
                continue
            args_repr = ",".join(
                ast.unparse(a) if hasattr(ast, "unparse") else repr(a) for a in node.args
            )
            calls.setdefault(target_name, []).append((args_repr, node.lineno))

    claims: list[CacheClaim] = []
    for func, sites in calls.items():
        # Bug 3: @stateful-only functions don't have cache_info(), so don't
        # generate a claim for them (only generate when also @cache-decorated).
        if func in stateful_funcs and func not in cached_funcs:
            continue
        if func in stateful_funcs:
            claims.append(
                CacheClaim(function=func, expected_hits=0, expected_misses=len(sites))
            )
            continue
        # Count comment-tagged misses/hits and infer the rest from arg uniqueness.
        explicit_miss = 0
        explicit_hit = 0
        ambiguous: list[tuple[str, int]] = []
        for args_repr, lineno in sites:
            line_text = lines[lineno - 1] if 1 <= lineno <= len(lines) else ""
            if _MISS_HINT.search(line_text):
                explicit_miss += 1
            elif _HIT_HINT.search(line_text):
                explicit_hit += 1
            else:
                ambiguous.append((args_repr, lineno))
        # For ambiguous calls, infer: first call per unique args = miss, rest = hits.
        seen: dict[str, int] = {}
        ambig_hits = 0
        ambig_misses = 0
        for args_repr, _ in ambiguous:
            n = seen.get(args_repr, 0)
            if n == 0:
                ambig_misses += 1
            else:
                ambig_hits += 1
            seen[args_repr] = n + 1
        claims.append(
            CacheClaim(
                function=func,
                expected_hits=explicit_hit + ambig_hits,
                expected_misses=explicit_miss + ambig_misses,
            )
        )
    return claims
