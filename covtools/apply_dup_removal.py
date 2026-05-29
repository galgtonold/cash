"""Apply the duplicate-test removal list produced by dup_detect.py --emit.

Dry-run by default: prints exactly which test methods, classes, and files
would be removed. Pass --apply to actually rewrite files.

Removal is line-span based (never ast.unparse — that would reformat surviving
tests and drop their comments). For each target we delete from its first
decorator (or ``def`` line) through ``end_lineno``, bottom-up so earlier line
numbers stay valid. A class whose every function is removed and which holds
nothing but a docstring/pass is removed whole; a file left with zero test
functions is deleted entirely.

Usage:
    python covtools/apply_dup_removal.py                 # dry run
    python covtools/apply_dup_removal.py --apply
"""
from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REMOVAL = ROOT / "covrun" / "dup_removal.json"


def _func_span(fn: ast.FunctionDef) -> tuple[int, int]:
    """1-based inclusive (start, end) line span, including any decorators."""
    start = fn.lineno
    if fn.decorator_list:
        start = min(start, min(d.lineno for d in fn.decorator_list))
    return start, fn.end_lineno


def _class_is_empty_without(cls: ast.ClassDef, removed: set[str]) -> bool:
    """True if removing `removed` methods leaves only docstring/pass behind."""
    for stmt in cls.body:
        if isinstance(stmt, ast.FunctionDef):
            if stmt.name not in removed:
                return False  # a surviving method
        elif isinstance(stmt, ast.Pass):
            continue
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue  # docstring
        else:
            return False  # class-level attribute / other statement -> keep class
    return True


def _has_remaining_tests(tree: ast.Module, removed_top: set[str],
                         removed_cls: dict[str, set[str]],
                         dropped_classes: set[str]) -> bool:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            if node.name not in removed_top:
                return True
        elif isinstance(node, ast.ClassDef) and node.name not in dropped_classes:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name.startswith("test_"):
                    if sub.name not in removed_cls.get(node.name, set()):
                        return True
    return False


def plan_file(path: Path, targets: list[dict]) -> dict:
    src = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)

    removed_top = {t["test"] for t in targets if t["class"] is None}
    removed_cls: dict[str, set[str]] = defaultdict(set)
    for t in targets:
        if t["class"] is not None:
            removed_cls[t["class"]].add(t["test"])

    spans: list[tuple[int, int]] = []
    dropped_classes: set[str] = set()
    method_count = 0

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in removed_top:
            spans.append(_func_span(node))
            method_count += 1
        elif isinstance(node, ast.ClassDef) and node.name in removed_cls:
            rm = removed_cls[node.name]
            if _class_is_empty_without(node, rm):
                spans.append((node.lineno, node.end_lineno))  # drop whole class
                dropped_classes.add(node.name)
                method_count += len(rm)
            else:
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef) and sub.name in rm:
                        spans.append(_func_span(sub))
                        method_count += 1

    delete_file = not _has_remaining_tests(tree, removed_top, removed_cls, dropped_classes)

    return {
        "path": path,
        "lines": lines,
        "spans": sorted(spans, reverse=True),
        "dropped_classes": dropped_classes,
        "method_count": method_count,
        "delete_file": delete_file,
    }


def apply_plan(plan: dict) -> None:
    if plan["delete_file"]:
        plan["path"].unlink()
        return
    lines = plan["lines"]
    for start, end in plan["spans"]:  # already sorted descending
        del lines[start - 1:end]
    new_src = "".join(lines)
    ast.parse(new_src)  # fail loudly rather than write broken Python
    plan["path"].write_text(new_src, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually rewrite/delete files")
    ap.add_argument("--removal", type=str, default=None,
                    help="path to removal JSON (default: covrun/dup_removal.json)")
    args = ap.parse_args()

    removal_path = Path(args.removal) if args.removal else REMOVAL
    removals = json.loads(removal_path.read_text(encoding="utf-8"))
    by_file: dict[str, list[dict]] = defaultdict(list)
    for r in removals:
        by_file[r["file"]].append(r)

    total_methods = 0
    files_deleted = 0
    classes_dropped = 0
    for rel in sorted(by_file):
        plan = plan_file(ROOT / rel, by_file[rel])
        total_methods += plan["method_count"]
        files_deleted += int(plan["delete_file"])
        classes_dropped += len(plan["dropped_classes"])
        tag = "DELETE FILE" if plan["delete_file"] else f"-{plan['method_count']} tests"
        extra = (f"  drop classes: {sorted(plan['dropped_classes'])}"
                 if plan["dropped_classes"] and not plan["delete_file"] else "")
        print(f"  {tag:<14} {rel.split('/')[-1]}{extra}")
        if args.apply:
            apply_plan(plan)

    mode = "APPLIED" if args.apply else "DRY RUN"
    print(f"\n[{mode}] {total_methods} tests across {len(by_file)} files | "
          f"{files_deleted} files deleted whole | {classes_dropped} classes dropped")
    if not args.apply:
        print("Re-run with --apply to perform the removal.")


if __name__ == "__main__":
    main()
