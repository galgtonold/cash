"""Evidence inventory for the notebook integration test suite.

Classifies every test function by the *cash feature-combination signature* it
exercises — NOT by the stdlib module it happens to print. The point is to find
true duplicates (same code paths, same combination, generic picklable value)
while explicitly protecting tests that exercise *rare combinations* of features
(e.g. file-deps + code-edit + restart), which carry real coverage even though
each feature is individually common.

Outputs:
  - integration_inventory.json   (one record per test function)
  - integration_inventory.md     (human-readable summary)

Run:  python analyze_integration_tests.py
"""
from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
TEST_DIR = ROOT / "tests" / "test_notebook_integration"


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------
# Each test function's source text is scanned for substrings that map to a
# distinct cash code-path. We deliberately scan the WHOLE function body (cell
# string literals + runner calls) because a cell that does `open(...)` and a
# test that asserts on file-dep behavior both indicate the file-dep path.

def collect_string_literals(func: ast.AST) -> str:
    """Concatenate every string constant in the function (the cell sources)."""
    parts: list[str] = []
    for node in ast.walk(func):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            parts.append(node.value)
    return "\n".join(parts)


def count_calls(func: ast.AST, name: str) -> int:
    n = 0
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr == name:
                n += 1
    return n


def has_nonsequential_run_cells(func: ast.AST) -> bool:
    """True if any run_cells([...]) gets a non-strictly-increasing list."""
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr == "run_cells" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.List):
                    nums = [e.value for e in arg.elts
                            if isinstance(e, ast.Constant) and isinstance(e.value, int)]
                    if nums and any(b <= a for a, b in zip(nums, nums[1:])):
                        return True
    return False


# Object-type buckets: which serialization / content-hash code path the cached
# value lands in. Order matters (first match wins, most-specific first).
#
# `set` and `large_collection` were added after a coverage diff proved they hit
# distinct branches in object_hashing.py that the plain "generic" pickle path
# never reaches: _hash_collection's >200 sampling branch (52-57) and
# _estimate_set (209-216). Without these, every set / big-list test collapsed
# into "generic" and got dropped, silently losing those branches.
OBJECT_BUCKET_PATTERNS = [
    ("pandas",        re.compile(r"\b(pd\.|pandas|DataFrame|read_csv|read_parquet|\.Series\()")),
    ("numpy",         re.compile(r"\b(np\.|numpy|ndarray|\.nbytes)")),
    ("scipy_sparse",  re.compile(r"(csr_matrix|csc_matrix|coo_matrix|scipy\.sparse)")),
    ("unpicklable",   re.compile(r"(\blambda\b|\byield\b|weakref|threading|Lock\(|"
                                 r"\bsocket\b|asyncio|\basync def\b|generator|io\.BytesIO|io\.StringIO|"
                                 r"open\(|file=|\.fileno\(|tempfile)")),
    ("set",           re.compile(r"\bset\(|\bfrozenset\(|\{[^{}:\n]*,[^{}:\n]*\}")),
]

# A collection big enough (>200 elements) to trip object_hashing's sampling
# branch. Anchored to actual collection construction (constructor call, list/
# set/dict comprehension, or sequence repetition) so a bare ``sum(i for i in
# range(500_000))`` — which stores an int, not a collection — is NOT flagged.
_LARGE_COLL_RE = re.compile(
    r"(?:list|set|tuple|dict|frozenset)\(\s*(?:sorted\()?\s*range\(\s*(?:[\d_]+\s*,\s*)?([\d_]+)\s*\)"
    r"|\[[^\]\n]*\bfor\b[^\]\n]*\brange\(\s*(?:[\d_]+\s*,\s*)?([\d_]+)\s*\)"
    r"|\{[^}\n]*\bfor\b[^}\n]*\brange\(\s*(?:[\d_]+\s*,\s*)?([\d_]+)\s*\)"
    r"|[\]\)\'\"]\s*\*\s*([\d_]+)"
    r"|\b([\d_]+)\s*\*\s*[\[\(\'\"]"
)


def _is_large_collection(text: str) -> bool:
    for m in _LARGE_COLL_RE.finditer(text):
        for g in m.groups():
            if g and int(g.replace("_", "")) > 200:
                return True
    return False

FEATURE_PATTERNS = {
    # File-dependency tracking (open/read_csv/path-based reads inside cells)
    "file_dep": re.compile(r"(open\(|read_csv|to_csv|\.csv\b|\.txt\b|\.json'|\.parquet|"
                           r"Path\(|with open|loadtxt|savetxt|read_excel|read_parquet|"
                           r"\.write\(|writelines|tempfile|os\.path|pathlib)"),
    # Module import / reload tracking
    "module_track": re.compile(r"(importlib|\.reload\(|sys\.modules|write_text\(.*def |"
                               r"create_module|\.py['\"]|module_file)"),
    # In-cell function / class / decorator definition. Exercises function_tracker
    # (source + bytecode hashing) and purity analysis of the user's own callable —
    # a path plain value-caching never touches. ~40% of the giant generic
    # signatures hide one of these.
    "defines_func": re.compile(r"(^|\n)\s*(def |class |@\w)", re.MULTILINE),
    # The @cash decorator-bridge introspection API (cache_info/cache_clear/explain).
    # Drives core.py's _delete_backend_entries + backend.delete + cache_info, none
    # of which the notebook auto-cache path reaches.
    "decorator_api": re.compile(r"cache_info\(|cache_clear\(|@cash\b|\.explain\("),
    # Mutation tracking (in-place mutation of a tracked container)
    "mutation": re.compile(r"(\.append\(|\.extend\(|\.update\(|\.add\(|\.insert\(|"
                           r"\.pop\(|\.remove\(|\.sort\(\)|\+=|\-=|\*=)"),
    # Control-structure body caching
    "control": re.compile(r"^\s*(for |while |if |elif |else:|try:|with )", re.MULTILINE),
}

# Caching-behavior assertions (vs. merely asserting the printed value)
STATUS_RE = re.compile(r"(get_status|cash_status|_cash_|\.lineage|badge|cache_stats|"
                       r"was_cached|from_cache|cached=|\bskip|\bhit\b|\bmiss\b|recompute|"
                       r"restore|invalidat|reexec|re-exec|upstream|cache_info|cache_clear)",
                       re.IGNORECASE)


def classify_object_bucket(text: str) -> str:
    for name, pat in OBJECT_BUCKET_PATTERNS:
        if pat.search(text):
            return name
    if _is_large_collection(text):
        return "large_collection"
    return "generic"


def analyze_func(func: ast.FunctionDef, file_rel: str, cls: str | None) -> dict:
    # Use the string literals (cell sources) for object/feature detection, and
    # the structural calls for edit/restart/order detection.
    text = collect_string_literals(func)

    n_edits = count_calls(func, "set_cell_source")
    n_start = count_calls(func, "start_kernel")
    n_reset = count_calls(func, "reset_cash_state")
    n_shutdown = count_calls(func, "shutdown")

    flags = {}
    flags["edits_code"] = n_edits > 0
    flags["multi_edit"] = n_edits > 1
    for fname, pat in FEATURE_PATTERNS.items():
        flags[fname] = bool(pat.search(text))
    flags["out_of_order"] = has_nonsequential_run_cells(func)
    # restart/restore: explicit reset, OR a second kernel start, OR name hints
    name_l = (func.name + " " + (cls or "") + " " + file_rel).lower()
    flags["restart"] = (n_reset > 0 or n_start > 1 or n_shutdown > 0
                        or "restart" in name_l or "restore" in name_l
                        or "disk" in name_l)
    flags["asserts_status"] = bool(STATUS_RE.search(
        ast.unparse(func) if hasattr(ast, "unparse") else text))

    bucket = classify_object_bucket(text)

    # Active behavioral features (exclude pure object bucket)
    active = sorted(k for k, v in flags.items() if v and k not in ("multi_edit",))

    # Signature = the COMBINATION of code-paths this test exercises.
    signature = "+".join(active) + f"|{bucket}" if active else f"none|{bucket}"

    # A "smoke-only generic" test: runs cells, asserts printed output, does NOT
    # edit code and does NOT assert any caching behavior, generic value. These
    # only prove "cash doesn't corrupt normal execution" — the weakest class.
    smoke_only = (not flags["edits_code"]
                  and not flags["asserts_status"]
                  and bucket == "generic"
                  and not flags["restart"]
                  and not flags["file_dep"]
                  and not flags["module_track"]
                  and not flags["out_of_order"]
                  and not flags["defines_func"]
                  and not flags["decorator_api"])

    return {
        "file": file_rel,
        "class": cls,
        "test": func.name,
        "bucket": bucket,
        "n_edits": n_edits,
        "flags": {k: v for k, v in flags.items() if v},
        "signature": signature,
        "smoke_only": smoke_only,
    }


def main() -> None:
    records: list[dict] = []
    parse_errors: list[str] = []

    for path in sorted(TEST_DIR.glob("test_*.py")):
        rel = path.relative_to(ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except SyntaxError as e:
            parse_errors.append(f"{rel}: {e}")
            continue

        # Top-level test functions
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                records.append(analyze_func(node, rel, None))
            elif isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef) and sub.name.startswith("test_"):
                        records.append(analyze_func(sub, rel, node.name))

    # ---- Aggregations -----------------------------------------------------
    sig_counts = Counter(r["signature"] for r in records)
    bucket_counts = Counter(r["bucket"] for r in records)
    feature_counts = Counter()
    for r in records:
        for k in r["flags"]:
            feature_counts[k] += 1

    smoke = [r for r in records if r["smoke_only"]]

    # Rare combinations = signatures held by <=2 tests => PROTECT (these are the
    # combinatorial-coverage tests the user warned about).
    rare = [r for r in records if sig_counts[r["signature"]] <= 2]
    rare_multi = [r for r in rare if len(r["flags"]) >= 2]

    # Redundancy candidates: generic-bucket tests whose signature is shared by
    # many tests (the template clones). We keep K representatives per signature.
    K = 3
    seen_per_sig: Counter = Counter()
    redundant: list[dict] = []
    keep: list[dict] = []
    for r in sorted(records, key=lambda x: (x["signature"], x["file"], str(x["test"]))):
        sig = r["signature"]
        # Never mark non-generic buckets or status-asserting or rare sigs as redundant
        protect = (r["bucket"] != "generic"
                   or r["flags"].get("asserts_status")
                   or sig_counts[sig] <= K)
        if protect:
            keep.append(r)
            continue
        seen_per_sig[sig] += 1
        if seen_per_sig[sig] <= K:
            keep.append(r)
        else:
            redundant.append(r)

    out = {
        "totals": {
            "files": len({r["file"] for r in records}),
            "tests": len(records),
            "unique_signatures": len(sig_counts),
            "smoke_only_generic": len(smoke),
            "rare_combo_tests_protected": len(rare),
            "rare_multifeature_protected": len(rare_multi),
            "redundant_candidates": len(redundant),
            "keep_estimate": len(keep),
            "parse_errors": parse_errors,
        },
        "bucket_counts": bucket_counts.most_common(),
        "feature_counts": feature_counts.most_common(),
        "top_signatures": sig_counts.most_common(40),
        "records": records,
    }

    (ROOT / "integration_inventory.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ---- Markdown summary -------------------------------------------------
    md: list[str] = []
    md.append("# Integration test inventory\n")
    t = out["totals"]
    md.append(f"- **Tests:** {t['tests']} across {t['files']} files")
    md.append(f"- **Unique feature-combination signatures:** {t['unique_signatures']}")
    md.append(f"- **Smoke-only generic** (no edit, no caching assertion, picklable value): "
              f"**{t['smoke_only_generic']}**  ← weakest tests")
    md.append(f"- **Redundant candidates** (generic + shared signature, keeping {K}/sig): "
              f"**{t['redundant_candidates']}**")
    md.append(f"- **Protected rare combinations** (signature held by ≤2 tests): "
              f"{t['rare_combo_tests_protected']} (of which {t['rare_multifeature_protected']} "
              f"combine ≥2 features)")
    md.append(f"- **Estimated keep:** ~{t['keep_estimate']} tests\n")

    md.append("## Object-type buckets (serialization/hash code path)\n")
    md.append("| bucket | tests |\n|---|---|")
    for name, c in out["bucket_counts"]:
        md.append(f"| {name} | {c} |")

    md.append("\n## Feature flags (how many tests touch each cash path)\n")
    md.append("| feature | tests |\n|---|---|")
    for name, c in out["feature_counts"]:
        md.append(f"| {name} | {c} |")

    md.append("\n## Top 40 signatures by frequency (combination → count)\n")
    md.append("| count | signature |\n|---|---|")
    for sig, c in out["top_signatures"]:
        md.append(f"| {c} | `{sig}` |")

    md.append("\n## Rare multi-feature combinations (PROTECT — combinatorial coverage)\n")
    md.append("These have a signature shared by ≤2 tests AND combine ≥2 features. "
              "Do not delete without inspection.\n")
    md.append("| file | test | signature |\n|---|---|---|")
    for r in sorted(rare_multi, key=lambda x: x["file"])[:80]:
        md.append(f"| {r['file'].split('/')[-1]} | {r['test']} | `{r['signature']}` |")

    (ROOT / "integration_inventory.md").write_text("\n".join(md), encoding="utf-8")

    # ---- Console digest ---------------------------------------------------
    print(f"tests={t['tests']} files={t['files']} unique_signatures={t['unique_signatures']}")
    print(f"smoke_only_generic={t['smoke_only_generic']}")
    print(f"redundant_candidates(keep {K}/sig)={t['redundant_candidates']}  keep_estimate={t['keep_estimate']}")
    print(f"protected_rare_combos={t['rare_combo_tests_protected']} (multi-feature={t['rare_multifeature_protected']})")
    print(f"parse_errors={len(parse_errors)}")
    print("\nbuckets:", out["bucket_counts"])
    print("\nfeatures:", out["feature_counts"])
    print("\ntop 15 signatures:")
    for sig, c in out["top_signatures"][:15]:
        print(f"  {c:5d}  {sig}")


if __name__ == "__main__":
    main()
