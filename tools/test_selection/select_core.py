"""Pick the core integration set from a baseline run.

A test earns a place if it covers something no already-picked test covers, in
any of four kinds of element:

* ``L`` lines of cash code run inside a function body (import-time lines such
  as ``def`` and module constants are ignored: every test "runs" those);
* ``F``/``P`` features, and pairs of features, the test touched; a feature is a
  cash subsystem (loops, file deps, restore, ...) whose function bodies ran;
* ``S`` the order of the test's own steps (run, edit, restart, write a file,
  ...), as 1-, 2- and 3-step sequences;
* ``X`` a feature combined with a disruptive step (edit, restart, file write,
  partial rerun), so "restart while file deps are live" is its own element.

Selection is greedy weighted set cover: repeatedly take the passing test with
the most new elements per second of runtime. Tests in CI's smoke list are
always taken first. Failed, errored and skipped tests are never picked.

Usage::

    python tools/test_selection/select_core.py --out .testsel
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# Always in the core set: today's CI smoke subset.
ALWAYS = [
    "tests/test_notebook_integration/test_basic_flow.py",
    "tests/test_notebook_integration/test_disk_restore_after_restart.py",
    "tests/test_notebook_integration/test_file_invalidation_real.py",
    "tests/test_notebook_integration/test_decorator_bridge_integration.py",
    "tests/test_notebook_integration/test_badge_integration.py",
    "tests/test_notebook_integration/test_forward_probe_upstream_skip.py",
    "tests/test_notebook_integration/test_file_dep_path_fallback.py",
]

# cash module path (relative to the package, "/"-separated, no .py) prefix -> feature.
# Longest matching prefix wins. Modules with no entry count only as lines.
FEATURES = {
    "notebook/control_structures/for_handler": "loops",
    "notebook/loop_split": "loops",
    "notebook/control_structures/if_handler": "branches",
    "notebook/control_structures/try_handler": "try_blocks",
    "tracking/file_tracker": "file_deps",
    "tracking/file_dep_snapshot": "file_deps",
    "notebook/statement/file_deps": "file_deps",
    "notebook/write_observer": "file_writes",
    "notebook/module_invalidator": "modules",
    "tracking/function_tracker": "functions",
    "tracking/module_symbols": "modules",
    "tracking/randomness": "randomness",
    "notebook/restore": "restore",
    "notebook/statement/restore": "restore",
    "notebook/call_unit": "call_caching",
    "notebook/call_interception": "call_caching",
    "notebook/call_refs": "call_caching",
    "notebook/consumables": "consumables",
    "notebook/upstream/reexecution_planner": "reexecution",
    "notebook/upstream/mismatch_classifier": "mismatch",
    "notebook/upstream/stateful_carriers": "stateful_carriers",
    "notebook/carrier_history": "stateful_carriers",
    "notebook/statement/derivation_edges": "derivation_edges",
    "notebook/statement/miss_guard": "miss_guard",
    "notebook/live_cells": "cell_sources",
    "notebook/vscode_backup": "cell_sources",
    "notebook/server_discovery": "cell_sources",
    "notebook/badge_renderer": "badges",
    "analysis/annotations": "annotations",
    "purity": "purity",
    "analysis/cacheability_decision": "cacheability",
    "notebook/staleness": "staleness",
    "notebook/compute_baselines": "cost_model",
    "cost_model": "cost_model",
    "core": "decorator",
    "purity_analyzer": "decorator_purity",
    "effect_observer": "effects",
    "remote_source": "remote_sources",
    "backends/file_backend": "disk_tier",
    "backends/tiered_backend": "tiers",
    "backends/versions": "versions",
    "backends/value_policy": "value_policy",
}

DISRUPTIVE = {"edit", "restart", "reset", "write_file", "run_subset", "add_cell"}

# runner method (or file method) -> step name. A second start_kernel() in the
# same test is a restart: most tests restart by shutting down and starting again.
STEPS = {
    "start_kernel": "start",
    "run_all": "run_all",
    "run_cell": "run_subset",
    "run_cells": "run_subset",
    "set_cell_source": "edit",
    "restart": "restart",
    "reset_cash_state": "reset",
    "enable_persist": "persist",
    "write_text": "write_file",
    "write_bytes": "write_file",
    "add_cell": "add_cell",
}

# A feature counts as touched only when the test ran at least this many of the
# feature's lines that are not "ubiquitous" (run by UBIQUITOUS share of tests or
# more). Loading the extension touches a little of every subsystem.
FEATURE_MIN_LINES = 3
UBIQUITOUS = 0.5


# ---------------------------------------------------------------------------
# Static: the order of each test's steps


def _steps_of(func: ast.AST) -> list[str]:
    calls = []
    for node in ast.walk(func):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            step = STEPS.get(node.func.attr)
            if step:
                calls.append((node.lineno, node.col_offset, step))
    steps: list[str] = []
    started = False
    for _, _, step in sorted(calls):
        if step == "start":
            step = "restart" if started else "start"
            started = True
        if not steps or steps[-1] != step:
            steps.append(step)
    return steps


def static_steps(test_files: set[str]) -> dict[str, list[str]]:
    """'file::Class::func' / 'file::func' -> ordered steps."""
    out: dict[str, list[str]] = {}
    for rel in sorted(test_files):
        path = ROOT / rel
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except (OSError, SyntaxError):
            continue
        helpers = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef) and not n.name.startswith("test")}

        def with_helpers(fn):
            steps = _steps_of(fn)
            # Inline module-level helpers the test calls (one level deep).
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in helpers:
                    steps += _steps_of(helpers[node.func.id])
            return steps

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                out[f"{rel}::{node.name}"] = with_helpers(node)
            elif isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name.startswith("test"):
                        out[f"{rel}::{node.name}::{sub.name}"] = with_helpers(sub)
    return out


# ---------------------------------------------------------------------------
# Coverage: which function-body lines each test ran


def _cash_rel(path: str) -> str | None:
    p = path.replace("\\", "/")
    i = p.rfind("/cash/")
    if i < 0 or not p.endswith(".py"):
        return None
    return p[i + len("/cash/") : -3]


def _body_lines(path: str) -> set[int]:
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, SyntaxError):
        return set()
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            lines.update(range(node.body[0].lineno, node.end_lineno + 1))
    return lines


def _feature_of(mod: str) -> str | None:
    best = None
    for prefix, feat in FEATURES.items():
        if (mod == prefix or mod.startswith(prefix + "/")) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, feat)
    return best[1] if best else None


def coverage_by_test(cov_file: Path):
    from coverage import CoverageData

    data = CoverageData(basename=str(cov_file))
    data.read()
    lines: dict[str, set[tuple[str, int]]] = defaultdict(set)
    by_line: dict[tuple[str, int], set[str]] = {}
    for path in data.measured_files():
        mod = _cash_rel(path)
        if mod is None:
            continue
        body = _body_lines(path)
        for lineno, contexts in data.contexts_by_lineno(path).items():
            if lineno not in body:
                continue
            tests = {c for c in contexts if c}
            by_line[(mod, lineno)] = tests
            for ctx in tests:
                lines[ctx].add((mod, lineno))

    n_tests = max(len(lines), 1)
    hits: dict[str, Counter] = defaultdict(Counter)
    for (mod, _), tests in by_line.items():
        feat = _feature_of(mod)
        if feat is None or len(tests) >= UBIQUITOUS * n_tests:
            continue
        for ctx in tests:
            hits[ctx][feat] += 1
    feats = {ctx: {f for f, c in cnt.items() if c >= FEATURE_MIN_LINES} for ctx, cnt in hits.items()}
    return lines, feats


# ---------------------------------------------------------------------------


def _func_id(nodeid: str) -> str:
    return re.sub(r"\[.*\]$", "", nodeid)


def elements_for(nodeid, lines, feats, steps):
    el: set = set()
    el.update(("L",) + x for x in lines.get(nodeid, ()))
    fs = sorted(feats.get(nodeid, ()))
    el.update(("F", f) for f in fs)
    el.update(("P", a, b) for a, b in combinations(fs, 2))
    st = steps.get(_func_id(nodeid), [])
    for n in (1, 2, 3):
        el.update(("S",) + tuple(st[i : i + n]) for i in range(len(st) - n + 1))
    for s in set(st) & DISRUPTIVE:
        el.update(("X", f, s) for f in fs)
    return el


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / ".testsel"))
    ap.add_argument(
        "--overhead", type=float, default=1.0, help="seconds added to every test's cost (kernel handover etc.)"
    )
    args = ap.parse_args()
    out = Path(args.out)

    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    info = json.loads((out / "run_info.json").read_text(encoding="utf-8")) if (out / "run_info.json").exists() else {}
    lines, feats = coverage_by_test(out / "cov" / ".coverage")
    test_files = {n.split("::", 1)[0] for n in results}
    steps = static_steps(test_files)

    passing = {n for n, r in results.items() if r["outcome"] == "passed"}
    els = {n: elements_for(n, lines, feats, steps) for n in passing}
    cost = {n: results[n]["duration"] + args.overhead for n in passing}

    chosen: list[str] = []
    covered: set = set()
    for n in sorted(passing):
        if any(n.startswith(f + "::") for f in ALWAYS):
            chosen.append(n)
            covered |= els[n]
    remaining = {n for n in passing if n not in set(chosen)}
    while True:
        best, best_score = None, 0.0
        for n in remaining:
            gain = len(els[n] - covered)
            if gain and gain / cost[n] > best_score:
                best, best_score = n, gain / cost[n]
        if best is None:
            break
        chosen.append(best)
        covered |= els[best]
        remaining.discard(best)

    universe = set().union(*els.values()) if els else set()
    kinds = Counter(e[0] for e in universe)
    kinds_cov = Counter(e[0] for e in covered)
    missing_ctx = sorted(n for n in passing if not lines.get(n))
    total_time = sum(results[n]["duration"] for n in results)
    core_time = sum(results[n]["duration"] for n in chosen)

    (out / "core_set.txt").write_text("\n".join(sorted(chosen)) + "\n", encoding="utf-8")
    outcomes = Counter(r["outcome"] for r in results.values())
    flaky = sorted(n for n, r in results.items() if r["reruns"] and r["outcome"] == "passed")
    failing = sorted(n for n, r in results.items() if r["outcome"] in ("failed", "error"))

    names = {
        "L": "function-body lines",
        "F": "features",
        "P": "feature pairs",
        "S": "step sequences",
        "X": "feature x disruptive step",
    }
    rep = [
        "# Core integration set",
        "",
        f"Baseline: commit `{info.get('commit', '?')[:10]}` on {info.get('platform', '?')}, "
        f"{info.get('cpu_count', '?')} CPUs, wall clock {info.get('elapsed_seconds', 0) / 60:.0f} min.",
        "",
        f"- Tests run: {len(results)} ({', '.join(f'{k} {v}' for k, v in sorted(outcomes.items()))})",
        f"- Passed after a rerun (flaky here): {len(flaky)}",
        f"- Core set: **{len(chosen)} tests**, {core_time / 60:.1f} min of test time "
        f"vs {total_time / 60:.1f} min for the full suite ({core_time / max(total_time, 1):.0%})",
        f"- Passing tests with no coverage recorded (labelling failed?): {len(missing_ctx)}",
        "",
        "| Element | In suite | Covered by core set |",
        "|---|---|---|",
    ]
    rep += [f"| {names[k]} | {kinds[k]} | {kinds_cov[k]} |" for k in "LFPSX"]
    feat_count = Counter(f for n in passing for f in feats.get(n, ()))
    rep += ["", "## Tests touching each feature", ""]
    rep += [f"- {f}: {c}" for f, c in feat_count.most_common()]
    if failing:
        rep += ["", "## Failed or errored in the baseline", ""] + [f"- `{n}`" for n in failing]
    if flaky:
        rep += ["", "## Needed a rerun", ""] + [f"- `{n}`" for n in flaky]
    (out / "core_set_report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep[:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
