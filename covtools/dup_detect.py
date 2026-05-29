"""Detect *logically* duplicate integration tests at three leniency tiers.

Non-destructive: this reports, it never edits or deletes. It answers the
question "how many kernel-spinning tests are the same scenario wearing a
different stdlib costume?" — at three increasingly lenient definitions of
"same".

Why three tiers? "Logically the same" is a dial, not a threshold. Rather than
hard-code one notion of sameness, we fingerprint every test three ways and let
the reader pick where the line sits:

  T0  exact      Normalized source — comments / docstrings / whitespace
                 stripped via ast round-trip. Pure copy-paste only.
  T1  alpha      Structural isomorphism: literals collapse to their *type*
                 (<int>, <str>, ...) and every identifier / attribute /
                 keyword name is alpha-renamed in first-appearance order,
                 consistently across the whole test. So `bisect.bisect_left(d, 25)`
                 and `bisect.bisect_right(s, 5)` fold together — cash hashes
                 cell source opaquely, so the specific stdlib call is payload,
                 not behavior.
  T2  skeleton   cash-behavioral spine only: the ordered runner op-sequence
                 (create / run / edit / rerun / reset) + a per-cell feature
                 vector (imports? defines func? loop? mutates? ...) + the
                 assertion target shape. Cell *contents* are ignored entirely.
                 This is the loosest reading: "same scenario, any payload".

The regression-corpus guardrails are honored at every tier: a test is only
*collapse-eligible* if its inventory record is generic-bucket, does NOT assert
caching behavior, and its feature-signature is not rare (held by > K tests).
Non-generic buckets (pandas/numpy/scipy/set/large_collection/unpicklable),
status-asserting tests, and rare combinations are reported for context but
never counted as removable — they carry scenario coverage a clone does not.

Usage:
    python covtools/dup_detect.py                  # summary across all tiers
    python covtools/dup_detect.py --tier 1 --show 25   # list biggest T1 groups
    python covtools/dup_detect.py --tier 2 --show 25
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import textwrap
import tokenize
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "tests" / "test_notebook_integration"
INVENTORY = ROOT / "integration_inventory.json"
RUNNER = "nb_runner"
K_RARE = 3  # signatures held by <= K tests are protected (matches analyzer)


# ---------------------------------------------------------------------------
# Runner op-sequence + cell/edit extraction (the cash-behavioral spine)
# ---------------------------------------------------------------------------

OPAQUE = "\x00OPAQUE\x00"


def _const_str(node: ast.AST) -> str | None:
    """Resolve a cell-source expression to its string value, or None if dynamic.

    Handles the three forms used across the suite: bare string literals,
    ``textwrap.dedent("...")`` / ``dedent("...")`` wrappers (dedented so the
    block actually parses), and ``"a" + "b"`` constant concatenation.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Call):
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        if name == "dedent" and node.args:
            inner = _const_str(node.args[0])
            if inner is not None:
                return textwrap.dedent(inner)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        lo, ro = _const_str(node.left), _const_str(node.right)
        if lo is not None and ro is not None:
            return lo + ro
    return None


def _str_list(node: ast.AST) -> list[str] | None:
    """Return the string elements of a List literal, or None if not extractable."""
    if not isinstance(node, ast.List):
        return None
    out: list[str] = []
    for e in node.elts:
        s = _const_str(e)
        out.append(s if s is not None else OPAQUE)  # keep slot for dynamic cells
    return out


def runner_spine(func: ast.AST) -> tuple[list[str], list[str], list[str]]:
    """Walk the test in source order, returning (ops, cells, edit_sources).

    ops    ordered tokens for cash-relevant runner calls (reads omitted)
    cells  the first create_notebook([...]) cell sources
    edits  the new sources passed to each set_cell_source(...)
    """
    calls = [n for n in ast.walk(func)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == RUNNER]
    calls.sort(key=lambda n: (n.lineno, n.col_offset))

    ops: list[str] = []
    cells: list[str] = []
    edits: list[str] = []
    for c in calls:
        m = c.func.attr
        if m == "create_notebook":
            cl = _str_list(c.args[0]) if c.args else None
            n = len(cl) if cl is not None else "?"
            ops.append(f"NB{n}")
            if cl is not None and not cells:
                cells = cl
        elif m == "add_cell":
            ops.append("ADD")
        elif m == "start_kernel":
            ops.append("START")
        elif m == "run_all":
            ops.append("RUNALL")
        elif m == "run_cell":
            ops.append("RUN1")
        elif m == "run_cells":
            ln = len(c.args[0].elts) if c.args and isinstance(c.args[0], ast.List) else "?"
            ops.append(f"RUNN{ln}")
        elif m == "set_cell_source":
            ops.append("EDIT")
            src = _const_str(c.args[1]) if len(c.args) >= 2 else None
            edits.append(src if src is not None else OPAQUE)
        elif m in ("shutdown", "reset_cash_state"):
            ops.append("RESET")
        elif m == "enable_debug":
            ops.append("DEBUG")
        elif m == "load":
            ops.append("LOAD")
        # get_output / get_raw_output / get_cell are reads — not part of the spine
    return ops, cells, edits


# ---------------------------------------------------------------------------
# Tier-1 alpha canonicalization: structural isomorphism over a whole test
# ---------------------------------------------------------------------------

class _Alpha(ast.NodeTransformer):
    """Rename identifiers/attrs/kwargs to first-appearance tokens; literals -> type.

    One instance per *test* so a variable shared across cells stays consistent.
    """

    def __init__(self) -> None:
        self.names: dict[str, str] = {}
        self.attrs: dict[str, str] = {}
        self.kw: dict[str, str] = {}

    @staticmethod
    def _m(d: dict, key: str, prefix: str) -> str:
        if key not in d:
            d[key] = f"{prefix}{len(d)}"
        return d[key]

    def visit_Name(self, node: ast.Name):
        return ast.copy_location(
            ast.Name(id=self._m(self.names, node.id, "v"), ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg):
        node.arg = self._m(self.names, node.arg, "v")
        node.annotation = None
        return node

    def visit_Attribute(self, node: ast.Attribute):
        self.generic_visit(node)
        node.attr = self._m(self.attrs, node.attr, "a")
        return node

    def visit_keyword(self, node: ast.keyword):
        self.generic_visit(node)
        if node.arg is not None:
            node.arg = self._m(self.kw, node.arg, "k")
        return node

    def visit_Constant(self, node: ast.Constant):
        return ast.copy_location(
            ast.Constant(value=f"<{type(node.value).__name__}>"), node)

    # Drop docstrings consistently (they are Expr(Constant) -> already typed,
    # but blanking keeps them from distinguishing otherwise-identical tests).


def _cell_comments(src: str) -> tuple[str, ...]:
    """Verbatim comments in a cell, in order.

    Cash hashes raw cell source and parses ``# @cash:`` directives from comments
    (annotations.py), so comments are behaviorally significant — a fingerprint
    that drops them (every AST round-trip does) would wrongly merge a
    ``# @cash:ttl=60`` cell with a plain one. Folding comments back in keeps
    directive / comment-only-edit tests from collapsing into non-directive ones.
    """
    out: list[str] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                out.append(tok.string.strip())
    except (tokenize.TokenError, IndentationError, SyntaxError):
        out.append("<TOKERR>")
    return tuple(out)


def _ws_layout(src: str) -> tuple[int, ...]:
    """Per-line trailing-whitespace widths — invisible to the AST, but cash sees
    them (whitespace-only edits are their own regression scenario)."""
    return tuple(len(ln) - len(ln.rstrip()) for ln in src.splitlines())


def _exact_canon(src: str) -> str:
    """T0: raw text, newline-normalized only. Preserves comments and whitespace
    so 'exact' means genuine copy-paste, not merely same-AST."""
    return src.replace("\r\n", "\n").replace("\r", "\n").strip("\n")


def _alpha_canon(src: str, tr: _Alpha) -> str:
    """T1: structural isomorph form for one cell, plus the source-text details
    (comments + trailing whitespace) that cash hashes but the AST discards."""
    detail = f"##C{_cell_comments(src)!r}##W{_ws_layout(src)!r}"
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return "ERR:" + src.strip() + detail
    return ast.unparse(tr.visit(tree)) + detail


# ---------------------------------------------------------------------------
# Tier-2 skeleton: per-cell feature vector (cash-relevant statement shape)
# ---------------------------------------------------------------------------

_DIRECTIVE = __import__("re").compile(r"#\s*@cash:([\w-]+)(?:=(\d+))?")


def _directives(src: str) -> tuple[str, ...]:
    """Normalized ``@cash:`` directive tokens (value blanked) — matches the regex
    cash itself uses in annotations.py. Significant at every tier."""
    return tuple(sorted(
        f"@cash:{m.group(1).lower()}" + ("=N" if m.group(2) else "")
        for m in _DIRECTIVE.finditer(src)))


def _cell_features(src: str) -> tuple[str, ...]:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ("ERR",) + _directives(src)
    feats: set[str] = set(_directives(src))
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            feats.add("imp")
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            feats.add("def")
        elif isinstance(n, ast.ClassDef):
            feats.add("class")
        elif isinstance(n, (ast.For, ast.AsyncFor, ast.While)):
            feats.add("loop")
        elif isinstance(n, ast.If):
            feats.add("if")
        elif isinstance(n, (ast.With, ast.AsyncWith)):
            feats.add("with")
        elif isinstance(n, ast.Try):
            feats.add("try")
        elif isinstance(n, ast.AugAssign):
            feats.add("aug")
        elif isinstance(n, ast.Assign):
            feats.add("assign")
        elif isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            feats.add("comp")
        elif isinstance(n, ast.Call):
            feats.add("call")
    return tuple(sorted(feats))


def _target_shape(t: ast.AST) -> str:
    """Structural shape of an assignment target, recursively.

    cash's lineage/upstream analyzer parses assignment targets, so a flat swap
    ``a, b = b, a`` and a nested unpack ``(a, (b, c)) = ...`` are *different*
    cash scenarios even though both are 'an assignment'. Encoding the shape
    keeps T2c from merging them."""
    if isinstance(t, ast.Name):
        return "n"
    if isinstance(t, ast.Starred):
        return "*" + _target_shape(t.value)
    if isinstance(t, ast.Subscript):
        return "sub"
    if isinstance(t, ast.Attribute):
        return "attr"
    if isinstance(t, (ast.Tuple, ast.List)):
        return "(" + ",".join(_target_shape(e) for e in t.elts) + ")"
    return "?"


def _cell_ops_sig(src: str) -> str:
    """T2c operation identity: the *multiset* of called function/method names,
    attribute accesses, operator kinds, and assignment-target shapes in a cell,
    plus its feature vector and directives.

    This is the discriminator T2 lacks. T2 folds any two cells sharing a feature
    category ("has a call") — so complex-arithmetic, dict-views and string-split
    collapse together. Keeping the actual operation names (``sorted``/``split``/
    ``bisect_left``/...) and assignment shapes splits those apart, while still
    ignoring variable names and literal values: cash hashes cell source opaquely
    for cache keys, so two cells that run the *same* operations with the *same*
    binding structure on *different data* are not distinct cash scenarios.
    Counts are preserved (multiset, not set) to stay on the conservative side."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return "ERR:" + ",".join(_directives(src))
    toks: list[str] = list(_cell_features(src))  # already includes directives
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                toks.append(f"f:{f.id}")
            elif isinstance(f, ast.Attribute):
                toks.append(f"m:{f.attr}")
        elif isinstance(n, ast.Attribute):
            toks.append(f"a:{n.attr}")
        elif isinstance(n, ast.Assign):
            toks.append("asgn:" + "=".join(_target_shape(tg) for tg in n.targets))
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
            toks.append("asgn1:" + _target_shape(n.target))
        elif isinstance(n, ast.BinOp):
            toks.append(f"op:{type(n.op).__name__}")
        elif isinstance(n, ast.UnaryOp):
            toks.append(f"u:{type(n.op).__name__}")
        elif isinstance(n, ast.BoolOp):
            toks.append(f"b:{type(n.op).__name__}")
        elif isinstance(n, ast.Compare):
            toks.extend(f"c:{type(o).__name__}" for o in n.ops)
        elif isinstance(n, ast.Subscript):
            toks.append("sub")
    # Comments + trailing whitespace are invisible to the ops walk but cash
    # hashes them (and parses @cash: directives from comments). Folding them in
    # keeps comment-only / whitespace-only scenarios — whose entire purpose is
    # cash's source-hash sensitivity — from merging into substantive-edit tests.
    detail = f"#C{_cell_comments(src)!r}#W{_ws_layout(src)!r}"
    return "|".join(sorted(toks)) + detail


# ---------------------------------------------------------------------------
# Assertion shape
# ---------------------------------------------------------------------------

def _assert_shapes(func: ast.AST, tier: int, tr: _Alpha | None) -> list[str]:
    out: list[str] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Assert):
            continue
        t = node.test
        if tier == 2:
            # Only the *kind* of assertion, not its payload.
            if isinstance(t, ast.Compare) and t.ops and isinstance(t.ops[0], ast.In):
                out.append("IN")
            elif isinstance(t, ast.Compare) and t.ops and isinstance(t.ops[0], ast.Eq):
                out.append("EQ")
            else:
                out.append("OTHER")
        elif tier == 1 and tr is not None:
            try:
                out.append(ast.unparse(tr.visit(ast.parse(ast.unparse(t)))))
            except SyntaxError:
                out.append("ERR")
        else:  # T0
            out.append(ast.unparse(t))
    return out


# ---------------------------------------------------------------------------
# Fingerprint a test at a given tier
# ---------------------------------------------------------------------------

def fingerprint(func: ast.AST, tier: int, uniq: str) -> str:
    ops, cells, edits = runner_spine(func)
    spine = "|".join(ops)
    # A test with a dynamically-built cell can't be safely compared — give it a
    # unique fingerprint so it never collapses into another test by accident.
    salt = uniq if (OPAQUE in cells or OPAQUE in edits) else ""

    if tier == 2:
        cellsig = [",".join(_cell_features(c)) for c in cells]
        editsig = [",".join(_cell_features(e)) for e in edits]
        asserts = _assert_shapes(func, 2, None)
        payload = f"S[{spine}] C[{';'.join(cellsig)}] E[{';'.join(editsig)}] A[{Counter(asserts)}]"
    elif tier == 3:  # T2c: call-aware skeleton (same ops, any data)
        cellsig = [_cell_ops_sig(c) for c in cells]
        editsig = [_cell_ops_sig(e) for e in edits]
        asserts = _assert_shapes(func, 2, None)  # coarse IN/EQ/OTHER — data folds
        payload = f"S[{spine}] C[{';'.join(cellsig)}] E[{';'.join(editsig)}] A[{Counter(asserts)}]"
    elif tier == 1:
        tr = _Alpha()
        cellsig = [_alpha_canon(c, tr) for c in cells]
        editsig = [_alpha_canon(e, tr) for e in edits]
        asserts = _assert_shapes(func, 1, tr)
        payload = f"S[{spine}] C[{chr(10).join(cellsig)}] E[{chr(10).join(editsig)}] A[{chr(10).join(asserts)}]"
    else:  # T0
        cellsig = [_exact_canon(c) for c in cells]
        editsig = [_exact_canon(e) for e in edits]
        asserts = _assert_shapes(func, 0, None)
        payload = f"S[{spine}] C[{chr(10).join(cellsig)}] E[{chr(10).join(editsig)}] A[{chr(10).join(asserts)}]"

    return hashlib.sha1((salt + payload).encode("utf-8", "replace")).hexdigest()


# ---------------------------------------------------------------------------
# Load inventory metadata for guardrails
# ---------------------------------------------------------------------------

def load_inventory() -> tuple[dict, Counter]:
    inv = json.loads(INVENTORY.read_text(encoding="utf-8"))
    recs = inv["records"]
    sig_counts = Counter(r["signature"] for r in recs)
    by_key = {(r["file"], r["class"], r["test"]): r for r in recs}
    return by_key, sig_counts


def eligible(rec: dict, sig_counts: Counter) -> bool:
    """Collapse-eligible iff generic bucket, no caching assertion, common signature."""
    if rec is None:
        return False
    return (rec["bucket"] == "generic"
            and not rec["flags"].get("asserts_status")
            and sig_counts[rec["signature"]] > K_RARE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect() -> list[dict]:
    """Return one entry per test function: keys + fingerprints at all tiers."""
    by_key, sig_counts = load_inventory()
    entries: list[dict] = []
    for path in sorted(TEST_DIR.glob("test_*.py")):
        rel = path.relative_to(ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except SyntaxError:
            continue

        def handle(fn: ast.FunctionDef, cls: str | None) -> None:
            rec = by_key.get((rel, cls, fn.name))
            uniq = f"{rel}::{cls}::{fn.name}"
            entries.append({
                "file": rel,
                "class": cls,
                "test": fn.name,
                "rec": rec,
                "eligible": eligible(rec, sig_counts),
                "fp": {t: fingerprint(fn, t, uniq) for t in (0, 1, 2, 3)},
            })

        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                handle(node, None)
            elif isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef) and sub.name.startswith("test_"):
                        handle(sub, node.name)
    return entries


def _key(e: dict) -> tuple:
    return (e["file"], e["class"], e["test"])


def _dup_groups(entries: list[dict], tier: int) -> list[list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        groups[e["fp"][tier]].append(e)
    return [v for v in groups.values() if len(v) > 1]


def _removable_in_group(members: list[dict]) -> tuple[list[dict], dict | None]:
    """Return (members_to_remove, kept_representative).

    Protected members are never removed and act as implicit survivors; if a
    group has at least one, every *eligible* clone is removable. Otherwise the
    sorted-first eligible test is kept as the representative.
    """
    elig = sorted((m for m in members if m["eligible"]),
                  key=lambda m: (m["file"], str(m["class"]), m["test"]))
    if not elig:
        return [], None
    if any(not m["eligible"] for m in members):
        return elig, None
    return elig[1:], elig[0]


def emit_removal(entries: list[dict], tiers: tuple[int, ...], out_path: Path) -> dict:
    removal: dict[tuple, int] = {}
    reps: set[tuple] = set()
    for t in tiers:
        for members in _dup_groups(entries, t):
            rem, rep = _removable_in_group(members)
            for m in rem:
                removal.setdefault(_key(m), t)
            if rep is not None:
                reps.add(_key(rep))

    # Safety invariant: every duplicate group at the loosest requested tier must
    # retain at least one surviving member after removal.
    loosest = max(tiers)
    for members in _dup_groups(entries, loosest):
        survivors = [m for m in members if _key(m) not in removal]
        if not survivors:
            raise SystemExit(
                f"INVARIANT VIOLATION: group would be fully deleted: "
                f"{[_key(m) for m in members]}")

    payload = sorted(
        ({"file": k[0], "class": k[1], "test": k[2], "tier": t}
         for k, t in removal.items()),
        key=lambda d: (d["file"], str(d["class"]), d["test"]))
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    files = {d["file"] for d in payload}
    by_tier = Counter(d["tier"] for d in payload)
    print(f"emitted {len(payload)} removals across {len(files)} files -> {out_path}")
    print(f"  by flagging tier: {dict(by_tier)}")
    return {"removal": payload, "files": files}


def tier_report(entries: list[dict], tier: int) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        groups[e["fp"][tier]].append(e)

    dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
    removable = 0
    eligible_dup_tests = 0
    for members in dup_groups.values():
        elig = [m for m in members if m["eligible"]]
        protected_present = any(not m["eligible"] for m in members)
        eligible_dup_tests += len(elig)
        if not elig:
            continue
        # Keep one representative per group. If a protected member already
        # stays, every eligible clone is removable; else keep one eligible rep.
        removable += len(elig) if protected_present else len(elig) - 1

    return {
        "tier": tier,
        "total_groups": len(groups),
        "dup_groups": len(dup_groups),
        "tests_in_dup_groups": sum(len(v) for v in dup_groups.values()),
        "eligible_dup_tests": eligible_dup_tests,
        "removable": removable,
        "groups": dup_groups,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, choices=(0, 1, 2, 3), default=None)
    ap.add_argument("--show", type=int, default=0, help="list the N biggest dup groups")
    ap.add_argument("--emit", type=str, default=None,
                    help="write the removal list (JSON) to this path")
    ap.add_argument("--emit-tiers", type=str, default="0,1",
                    help="comma-separated tiers to union for --emit (default 0,1)")
    args = ap.parse_args()

    entries = collect()

    if args.emit:
        tiers = tuple(int(x) for x in args.emit_tiers.split(","))
        emit_removal(entries, tiers, Path(args.emit))
        return
    total = len(entries)
    n_elig = sum(1 for e in entries if e["eligible"])
    print(f"tests analyzed   = {total}")
    print(f"collapse-eligible = {n_elig}  "
          f"(generic bucket, no caching assertion, signature held by > {K_RARE})")
    print(f"protected         = {total - n_elig}  (non-generic / status / rare)\n")

    names = {0: "T0 exact", 1: "T1 alpha-equiv", 3: "T2c call-aware", 2: "T2 skeleton"}
    print(f"{'tier':<16}{'dup groups':>11}{'tests in dups':>15}{'removable*':>12}")
    reports = {}
    for t in (0, 1, 3, 2):  # display in leniency order: exact < alpha < call-aware < skeleton
        r = tier_report(entries, t)
        reports[t] = r
        print(f"{names[t]:<16}{r['dup_groups']:>11}{r['tests_in_dup_groups']:>15}{r['removable']:>12}")
    print("\n* removable = clones droppable keeping 1 representative per group, "
          "counting ONLY collapse-eligible tests.")

    if args.tier is not None and args.show:
        r = reports[args.tier]
        big = sorted(r["groups"].values(), key=len, reverse=True)[:args.show]
        print(f"\n=== {names[args.tier]}: {len(big)} largest duplicate groups ===")
        for members in big:
            elig = sum(1 for m in members if m["eligible"])
            print(f"\n[{len(members)} tests | {elig} eligible] "
                  f"buckets={Counter(m['rec']['bucket'] if m['rec'] else '?' for m in members)}")
            for m in sorted(members, key=lambda x: (x["file"], str(x["test"])))[:8]:
                tag = "elig" if m["eligible"] else "PROT"
                short = m["file"].split("/")[-1]
                print(f"    [{tag}] {short}::{m['class'] or '-'}::{m['test']}")
            if len(members) > 8:
                print(f"    ... +{len(members) - 8} more")


if __name__ == "__main__":
    main()
