#!/usr/bin/env python3
"""Speed scoring for one round-34 project.

usage: speed.py RUNDIR

Reads RUNDIR/{plain1,plain2,cash1,cash2}.jsonl and writes RUNDIR/speed.json
and RUNDIR/speed.md. All numbers are wall seconds of the replayed steps
(kernel start, restarts, cells, scripts, shells; setup and the harness's own
digesting are not counted; plain script steps use wall minus the call hook's
own time).

  P  = plain total, min of plain1/plain2.     C = cash total, min of cash1/cash2.
  Z1 = P - plain time of every statement/call the cash run served from cache.
       Notebook: statements matched by step + position (aligned on statement
       text); RESTORED and SKIPPED count fully, MIXED by its saved share.
       Scripts: cached calls matched by step + order of completion (aligned on
       function name); a HIT subtracts that call's plain time (children of a
       hit are not counted again).
  Z2 = P - plain time of every statement/call that repeats an earlier one of
       the session with the same code and the same inputs (perfect cache).
       Statement inputs: the value digests of the names it reads (before the
       cell, or as written by an earlier statement of the same cell), the code
       identity of functions/modules it reads, and the content of project files
       whose path appears in its text or in a string variable it reads.
       Call inputs: function identity (code, globals it reads, its module's
       source) and argument digest (with file contents for path arguments).
  Z2 is then taken per step as max(model free time, Z1 served time): a perfect
  cache saves at least what cash saved (cash also reuses intercepted helper
  calls inside a statement, which the statement-level model cannot see).
  Z2_model (in speed.json) is the model alone.
  Served (Z1) = label-based + wall-based credit for inner hits: a statement
  cash labels COMPUTED/MIXED whose cash time is below 0.7 x its plain time and
  more than 0.2 s shorter is credited plain - cash; cash's sub-call hits when
  the replay recorded them; the step walls when statements cannot be matched.
  C - Z1 = cash overhead;  Z1 - Z2 = missed reuse.
The plain timings come from the plain run with the smaller total and the
served statements from the cash run with the smaller total.
"""

from __future__ import annotations

import ast
import difflib
import fnmatch
import hashlib
import json
import os
import re
import sys
from collections import defaultdict

RUNS = ("plain1", "plain2", "cash1", "cash2")


def load(path):
    meta, recs = None, []
    if not os.path.exists(path):
        return None, None
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("op") == "end":  # cache size record (stats.py reads it)
            continue
        if r.get("op") == "meta":
            meta = r
        else:
            recs.append(r)
    return meta, {r["i"]: r for r in recs}


def wall(r):
    return float(r.get("wall_adj", r.get("wall")) or 0.0)


def total(recs):
    return sum(wall(r) for r in recs.values())


def H(*a):
    return hashlib.sha1(repr(a).encode()).hexdigest()[:16]


def norm_code(s):
    try:
        return ast.unparse(ast.parse(s))
    except Exception:  # noqa: BLE001
        return re.sub(r"\s+", " ", s or "").strip()


def excerpt(s, n=90):
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[: n - 3] + "..."


# ---------------------------------------------------------------- Z1


# Wall-based credit: a statement cash labels COMPUTED/MIXED can still have been
# served from inside (intercepted helper calls or @cash.cache calls that hit, a
# loop whose iterations' calls hit). Credit plain - cash when cash was clearly
# faster: below WALL_RATIO x plain and by more than WALL_MIN seconds (noise).
WALL_RATIO = 0.7
WALL_MIN = 0.2
_ITER = re.compile(r"^#\s*__iteration_context__[^\n]*\n?")


def wall_credit(plain_t, cash_t):
    if cash_t is None or plain_t <= 0:
        return 0.0
    if cash_t < WALL_RATIO * plain_t and plain_t - cash_t > WALL_MIN:
        return max(0.0, plain_t - cash_t)
    return 0.0


def _cash_code(c):
    """cash's statement code without its loop-iteration marker, normalized (cash keeps 100 chars)."""
    code = _ITER.sub("", c.get("code", "") or "")
    return re.sub(r"\s+", " ", code).strip(), bool(_ITER.match(c.get("code", "") or ""))


def assign_cash_stmts(texts, cst):
    """Map each cash statement to the plain top-level statement it belongs to.

    cash reports a loop body once per iteration (code prefixed with
    '# __iteration_context__'), truncates code at 100 characters and may add
    statements of its own; plain has one entry per top-level statement. Returns
    {plain index: [cash statements]}."""
    plain = [re.sub(r"\s+", " ", norm_code(t)).strip() for t in texts]
    out = defaultdict(list)
    if len(plain) == len(cst) and not any(_cash_code(c)[1] for c in cst):
        for k, c in enumerate(cst):  # same count, no loop split: position is the match
            out[k].append(c)
        return out
    cur = -1
    for c in cst:
        code, is_iter = _cash_code(c)
        head = code[:90]
        q = None
        if is_iter and 0 <= cur < len(plain) and head[:60] in plain[cur]:
            q = cur  # another iteration of the loop already being consumed
        if q is None:
            for j in range(cur + 1, len(plain)):
                if plain[j].startswith(head) or (head and head[:60] in plain[j]):
                    q = j
                    break
        if q is None and 0 <= cur < len(plain) and head and head[:60] in plain[cur]:
            q = cur
        if q is None:
            q = min(cur + 1, len(plain) - 1) if plain else None
        if q is None:
            continue
        out[q].append(c)
        cur = q
    return out


def served_stmts(prec, crec):
    """[(stmt index, plain secs, served secs, text, how)] for one notebook step.

    served = label-based credit (RESTORED/SKIPPED in full, MIXED by its saved
    share) or, when larger, the credit from cash's per-call hits inside the
    statement (`subcalls`, newer replays) or the wall-based credit for inner
    hits (cash statement time clearly below the plain statement time).
    Falls back to the step walls when the statements cannot be matched."""
    st = prec.get("stmts") or {}
    texts, times = st.get("texts") or [], st.get("times") or []
    n = min(len(texts), len(times)) if texts else 0
    cash = (crec or {}).get("cash") or {}
    allc = cash.get("statements") or []
    subcalls = cash.get("subcalls")  # per entry of `statements`, if the replay captured them
    cst = [c for c in allc if not c.get("is_upstream")]
    if subcalls is not None and len(subcalls) == len(allc):
        for c, sc in zip(allc, subcalls):
            c["_sub"] = sc
    out = []
    if not n or not cst:
        # nothing to match statement by statement: compare the step walls
        pw, cw = wall(prec), (wall(crec) if crec else None)
        sv = wall_credit(pw, cw) if crec and not cash.get("statements") else 0.0
        if sv:
            stmt_sum = sum(times) if times else pw
            out.append((0, min(pw, stmt_sum or pw), min(sv, stmt_sum or pw), texts[0] if texts else (prec.get("src") or "?"), "step-wall"))
        return out
    groups = assign_cash_stmts(texts[:n], cst)
    for k in range(n):
        t, txt = times[k], texts[k]
        cs = groups.get(k, [])
        label = sub = wallc = 0.0
        if cs:
            stats = [c.get("status") for c in cs]
            if all(x in ("RESTORED", "SKIPPED") for x in stats):
                label = t
            else:
                sv = sum(float(c.get("saved_time") or 0) for c in cs)
                ex = sum(float(c.get("execution_time") or 0) for c in cs)
                if sv > 0:
                    label = t * sv / (sv + ex) if sv + ex > 0 else 0.0
            cash_t = sum(float(c.get("execution_time") or 0) for c in cs)
            wallc = wall_credit(t, cash_t)
            hits = [e for c in cs for e in (c.get("_sub") or []) if e.get("cache_hit") and not e.get("ran_plain")]
            if hits:
                sub = sum(float(e.get("time_saved") or 0) for e in hits)
        served = min(t, max(label, sub, wallc))
        how = "label" if served and served == label else "subcalls" if served and served == sub else "wall" if served else ""
        out.append((k, t, served, txt, how))
    return out


def call_tree(calls):
    """Plain calls by (pid, id), with children lists."""
    nodes = {(c.get("pid"), c["id"]): c for c in calls if "id" in c}
    kids = defaultdict(list)
    roots = []
    for key, c in sorted(nodes.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        if c.get("p", -1) == -1 or (key[0], c["p"]) not in nodes:
            roots.append(key)
        else:
            kids[(key[0], c["p"])].append(key)
    return nodes, kids, roots


def served_calls(prec, crec):
    """[(plain call, served secs)] for one script/shell step."""
    pcalls = [c for c in (prec.get("calls") or []) if "id" in c]
    ccalls = (crec or {}).get("calls") or []
    nodes, kids, _ = call_tree(pcalls)
    # plain records are in completion order per process, like cash's verbose lines
    a = [c["q"] for c in pcalls]
    b = [c.get("q") or c.get("fn", "").rsplit(".", 1)[-1] for c in ccalls]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    served = {}
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            cc = ccalls[blk.b + k]
            if cc.get("outcome") == "HIT":
                served[k + blk.a] = True
    out = []
    done = set()

    def mark(key):
        done.add(key)
        for ch in kids.get(key, []):
            mark(ch)

    # pre-order (ids are given at call start), so a served parent is seen before its children
    for k, c in sorted(enumerate(pcalls), key=lambda kc: (str(kc[1].get("pid")), kc[1]["id"])):
        key = (c.get("pid"), c["id"])
        if key in done:
            continue
        if served.get(k):
            # a hit replaces the whole subtree; children were matched only by accident
            mark(key)
            out.append((c, float(c["t"])))
    # inclusive times of children already counted under a served parent
    return out


# ---------------------------------------------------------------- Z2

def reads_writes(node):
    reads, writes = set(), set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            (writes if isinstance(n.ctx, (ast.Store, ast.Del)) else reads).add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            writes.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                writes.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            b = n.func.value
            while isinstance(b, (ast.Attribute, ast.Subscript, ast.Call)):
                b = b.func if isinstance(b, ast.Call) else b.value
            if isinstance(b, ast.Name):
                writes.add(b.id)  # a method call may mutate its receiver
        elif isinstance(n, (ast.Attribute, ast.Subscript)) and isinstance(n.ctx, ast.Store):
            b = n.value
            while isinstance(b, (ast.Attribute, ast.Subscript)):
                b = b.value
            if isinstance(b, ast.Name):
                writes.add(b.id)
    bound = set()
    for n in ast.walk(node):
        tg = [n.target] if isinstance(n, (ast.For, ast.AsyncFor, ast.comprehension)) else []
        if isinstance(n, (ast.With, ast.AsyncWith)):
            tg += [w.optional_vars for w in n.items if w.optional_vars]
        for t in tg:
            bound |= {x.id for x in ast.walk(t) if isinstance(x, ast.Name)}
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            bound |= {a.arg for a in ast.walk(n.args) if isinstance(a, ast.arg)}
            if not isinstance(n, ast.Lambda):
                bound |= {x.id for x in ast.walk(n) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)}
        if isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
    return reads - bound, writes


def file_inputs(text, node, read_names, strs, files):
    lits = set()
    if node is not None:
        lits = {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str) and 0 < len(n.value) < 400}
    lits |= {strs[n] for n in read_names if n in strs}
    if node is not None:
        for n in ast.walk(node):
            mods = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module] if isinstance(n, ast.ImportFrom) and n.module else []
            for m in mods:
                lits.add(m.replace(".", "/") + ".py")
                lits.add(m.replace(".", "/") + "/__init__.py")
    out = []
    for p, d in files.items():
        base = os.path.basename(p)
        hit = p in text or (len(base) > 3 and base in text)
        if not hit:
            for s in lits:
                s2 = s.lstrip("./").rstrip("/")
                if not s2:
                    continue
                if p == s2 or p.startswith(s2 + "/") or s2.endswith("/" + p) or (any(ch in s2 for ch in "*?[") and fnmatch.fnmatch(p, s2)):
                    hit = True
                    break
                if os.path.basename(s2) == base and len(base) > 3:
                    hit = True
                    break
        if hit:
            out.append((p, d))
    return sorted(out)


def z2_calls(calls, seen):
    """Rows for decorated calls: free when the same function identity and
    argument digest already ran (ok) earlier in the session."""
    rows = []
    nodes, kids, roots = call_tree(calls)

    def visit(key):
        c = nodes[key]
        ck = H(c.get("fn"), c.get("f"), c.get("a"))
        if ck in seen:
            rows.append(("call", c["id"], float(c["t"]), float(c["t"]), c.get("fn")))
            return
        if c.get("ok"):
            seen.add(ck)
        rows.append(("call", c["id"], float(c["t"]), 0.0, c.get("fn")))
        for ch in kids.get(key, []):
            visit(ch)

    for root in roots:
        visit(root)
    return rows


def z2_session(meta, recs):
    """{step: [(kind, idx, plain secs, free secs, text)]} for the perfect-cache bound."""
    seen = set()
    files = dict((meta or {}).get("files0") or {})
    vals, codes, strs = {}, {}, {}
    per = {}
    for i in sorted(recs):
        r = recs[i]
        op = r["op"]
        rows = []
        if op == "run":
            st = r.get("stmts") or {}
            texts, times = st.get("texts") or [], st.get("times") or []
            local = {}
            for k, t in enumerate(times[: len(texts)]):
                text = texts[k]
                try:
                    node = ast.parse(text)
                    rd, wr = reads_writes(node)
                except SyntaxError:
                    node, rd, wr = None, set(), set()
                # codes/values are refreshed only at run steps, so a module or function's
                # identity can predate a write_file/shell that edited its source (and
                # %autoreload picks the edit up): pair code identities with the current
                # project .py files.
                pyfiles = H(sorted((p, d) for p, d in files.items() if p.endswith(".py")))
                def ident(n):
                    if n in local:
                        return local[n]
                    if vals.get(n) == "module":  # its identity from the last run step may predate the edit
                        return H("module", n, pyfiles)
                    if n in codes:
                        return H(codes[n], pyfiles)
                    return vals.get(n) or "?"

                inputs = tuple(sorted((n, ident(n)) for n in rd))
                fin = tuple(file_inputs(text, node, rd, strs, files))
                key = H(text, inputs, fin)
                free = 0.0
                is_import = node is not None and node.body and all(isinstance(n, (ast.Import, ast.ImportFrom)) for n in node.body)
                if key in seen and not is_import:  # no cache can import a module into a new process for free
                    free = t
                else:
                    seen.add(key)
                for n in wr:
                    local[n] = H(key, n)
                rows.append(("stmt", k, t, free, text))
                if not free:
                    rows += z2_calls([c for c in (r.get("calls") or []) if "id" in c and c.get("s") == k], seen)
        elif op in ("script", "shell"):
            rows += z2_calls([c for c in (r.get("calls") or []) if "id" in c], seen)
        per[i] = rows
        for p, d in (r.get("files") or {}).items():
            if d is None:
                files.pop(p, None)
            else:
                files[p] = d
        if isinstance(r.get("values"), dict):
            vals, codes, strs = r["values"], r.get("codes") or {}, r.get("strs") or {}
    return per


# ---------------------------------------------------------------- report


def parallel_factor(pr):
    """Calls run in worker processes overlap in time: plain call times summed over
    workers exceed the step's wall (round 35, p25 joblib / p29 Pool). Scale a step's
    call-based credit by (parent's top-level call time + the busiest worker's) /
    (sum over all processes); 1.0 when the step ran in one process."""
    cs = [c for c in (pr.get("calls") or []) if "id" in c and c.get("d", 0) == 0]
    byp = defaultdict(float)
    for c in cs:
        byp[str(c.get("pid", "0"))] += float(c.get("t") or 0)
    if len(byp) < 2:
        return 1.0
    tot = sum(byp.values())
    parent = min(byp, key=lambda p: int(p) if p.isdigit() else 0)
    eff = byp[parent] + max(v for p, v in byp.items() if p != parent)
    return min(1.0, eff / tot) if tot else 1.0


def main():
    rundir = sys.argv[1] if len(sys.argv) > 1 else sys.exit(__doc__)
    data = {n: load(os.path.join(rundir, n + ".jsonl")) for n in RUNS}
    have = {n: d for n, d in data.items() if d[1] is not None}
    tot = {n: round(total(d[1]), 3) for n, d in have.items()}
    plains = [n for n in ("plain1", "plain2") if n in have]
    cashes = [n for n in ("cash1", "cash2") if n in have]
    if not plains:
        sys.exit("no plain run")
    pbest = min(plains, key=lambda n: tot[n])
    cbest = min(cashes, key=lambda n: tot[n]) if cashes else None
    pmeta, precs = have[pbest]
    P = tot[pbest]
    C = tot[cbest] if cbest else None
    pfac = {i: parallel_factor(pr) for i, pr in precs.items()}

    def z1_for(cname):
        crecs = have[cname][1]
        credit_how = defaultdict(float)
        served_total, per = 0.0, {}
        for i, pr in precs.items():
            cr = crecs.get(i)
            s, items = 0.0, []
            if pr["op"] == "run":
                served_k = set()
                for k, t, sv, txt, how in served_stmts(pr, cr):
                    s += sv
                    items.append((k, t, sv, txt))
                    if sv > 0:
                        served_k.add(k)
                    if how:
                        credit_how[how] += sv
                for c, sv in served_calls(pr, cr):
                    sv *= pfac[i]
                    if c.get("s") not in served_k:  # a call inside a statement cash ran, served by the decorator cache
                        s += sv
                        credit_how["calls"] += sv
                        items.append(("call", float(c["t"]), sv, c.get("fn")))
            elif pr["op"] in ("script", "shell"):
                for c, sv in served_calls(pr, cr):
                    sv *= pfac[i]
                    s += sv
                    credit_how["calls"] += sv
                    items.append((c["id"], float(c["t"]), sv, c.get("fn")))
            if pr["op"] in ("script", "shell") and cr:
                # parallel workers make per-call credit approximate: a clearly faster cash step
                # was served at least plain - cash
                wc = wall_credit(wall(pr), wall(cr))
                if wc > s:
                    credit_how["wall"] += wc - s
                    s = wc
            s = min(s, wall(pr))  # never credit more than the step took
            per[i] = (s, items)
            served_total += s
        return P - served_total, per, {k: round(v, 3) for k, v in credit_how.items()}

    z1 = {c: z1_for(c) for c in cashes}
    Z1, z1per, how = z1[cbest] if cbest else (None, {}, {})
    z2per = z2_session(pmeta, precs)
    # parallel workers: scale call rows like the served credit, cap each step at its wall
    for i, rows in list(z2per.items()):
        rows = [(kind, k, t, f * pfac.get(i, 1.0), txt) if kind == "call" else (kind, k, t, f, txt)
                for kind, k, t, f, txt in rows]
        cap, tot_f = wall(precs[i]), sum(r[3] for r in rows)
        if tot_f > cap > 0:
            rows = [(kind, k, t, f * cap / tot_f, txt) for kind, k, t, f, txt in rows]
        z2per[i] = rows
    free_total = sum(f for rows in z2per.values() for _, _, _, f, _ in rows)
    Z2_model = P - free_total
    # A perfect cache saves at least what cash saved: per step, the larger of the
    # model's free time and cash's served time (cash also reuses at a finer grain
    # than the model sees, e.g. intercepted helper calls inside a statement).
    Z2 = P - sum(
        max(sum(x[3] for x in z2per.get(i, [])), z1per.get(i, (0.0, []))[0]) for i in precs
    ) if cbest else Z2_model

    steps = []
    crecs = have[cbest][1] if cbest else {}
    for i in sorted(precs):
        pr = precs[i]
        cr = crecs.get(i, {})
        pw, cw = wall(pr), wall(cr) if cr else None
        sv = z1per.get(i, (0.0, []))[0]
        fr_model = sum(x[3] for x in z2per.get(i, []))
        fr = max(fr_model, z1per.get(i, (0.0, []))[0])
        row = {
            "i": i, "op": pr["op"], "cell": pr.get("cell") or pr.get("cmd") or pr.get("path"),
            "plain": round(pw, 4), "cash": None if cw is None else round(cw, 4),
            "served": round(sv, 4), "free": round(fr, 4), "free_model": round(fr_model, 4),
            "overhead": None if cw is None else round(cw - (pw - sv), 4),
            "missed": round(fr - sv, 4),
        }
        if pr["op"] == "run":
            cst = (cr.get("cash") or {})
            up = cst.get("upstream_metrics") or []
            row["upstream"] = len(up)
            # largest cash-side statement time, for the overhead excerpt
            cs = [s for s in (cst.get("statements") or []) if not s.get("is_upstream")]
            big = max(cs, key=lambda s: float(s.get("execution_time") or 0), default=None)
            row["overhead_excerpt"] = excerpt(big.get("code")) if big else excerpt(pr.get("src"))
            served_k = {k: s for k, _, s, _ in z1per.get(i, (0, []))[1] if s > 0 and k != "call"}
            served_fn = [txt for k, _, s, txt in z1per.get(i, (0, []))[1] if s > 0 and k == "call"]
            missed = []
            for kind, k, t, f, txt in z2per.get(i, []):
                if kind == "stmt":
                    f = f - served_k.get(k, 0.0)  # missed reuse is what is left after the credit
                if f <= 0.005:
                    continue
                if kind == "call" and txt in served_fn:
                    served_fn.remove(txt)
                    continue
                missed.append((f, txt))
            missed.sort(reverse=True)
            row["missed_excerpt"] = "; ".join(f"{excerpt(t, 60)} ({f:.2f}s)" for f, t in missed[:3])
        elif pr["op"] in ("script", "shell"):
            calls = (cr.get("calls") or [])
            row["overhead_excerpt"] = pr.get("cmd", "") + (
                " | cash calls: " + ", ".join(f"{c['outcome']} {c['q']}" for c in calls[:6]) if calls else "")
            missed = [(f, txt) for kind, k, t, f, txt in z2per.get(i, []) if f > 0]
            row["missed_excerpt"] = "; ".join(f"{t} ({f:.2f}s)" for f, t in sorted(missed, reverse=True)[:3])
        steps.append(row)

    noise = {}
    if len(plains) == 2:
        a, b = tot["plain1"], tot["plain2"]
        noise["plain"] = {"plain1": a, "plain2": b, "spread": round(abs(a - b), 3), "rel": round(abs(a - b) / max(min(a, b), 1e-9), 4)}
    if len(cashes) == 2:
        a, b = tot["cash1"], tot["cash2"]
        noise["cash"] = {"cash1": a, "cash2": b, "spread": round(abs(a - b), 3), "rel": round(abs(a - b) / max(min(a, b), 1e-9), 4)}
    step_spread = []
    for i in sorted(precs):
        for mode, pair in (("plain", plains), ("cash", cashes)):
            if len(pair) == 2:
                ra, rb = have[pair[0]][1].get(i), have[pair[1]][1].get(i)
                if ra and rb:
                    step_spread.append({"mode": mode, "i": i, "cell": ra.get("cell") or ra.get("cmd"), "a": wall(ra), "b": wall(rb), "spread": round(abs(wall(ra) - wall(rb)), 4)})
    step_spread.sort(key=lambda r: -r["spread"])
    noise["top_step_spread"] = step_spread[:10]

    top_over = sorted([s for s in steps if s["overhead"] is not None], key=lambda s: -s["overhead"])[:10]
    top_missed = sorted(steps, key=lambda s: -s["missed"])[:10]
    res = {
        "rundir": os.path.abspath(rundir),
        "totals": tot, "plain_run": pbest, "cash_run": cbest,
        "P": round(P, 3), "C": None if C is None else round(C, 3),
        "Z1": None if Z1 is None else round(Z1, 3), "Z2": round(Z2, 3), "Z2_model": round(Z2_model, 3),
        "Z1_by_cash_run": {c: round(z[0], 3) for c, z in z1.items()},
        "served_by_method": how,
        "served_method": "served = label-based (RESTORED/SKIPPED, MIXED saved share) + wall-based credit for inner hits "
        f"(COMPUTED/MIXED statement with cash < {WALL_RATIO} x plain and plain - cash > {WALL_MIN}s credits plain - cash; "
        "cash's own sub-call hits when the replay captured them; step walls when statements cannot be matched)",
        "overhead_C_minus_Z1": None if Z1 is None else round(C - Z1, 3),
        "missed_Z1_minus_Z2": None if Z1 is None else round(Z1 - Z2, 3),
        "ratios": {} if C is None else {"C/P": round(C / P, 4), "Z1/P": round(Z1 / P, 4), "Z2/P": round(Z2 / P, 4), "C/Z1": round(C / Z1, 4) if Z1 else None},
        "noise": noise,
        "top_overhead": top_over, "top_missed": top_missed,
        "steps": steps,
    }
    json.dump(res, open(os.path.join(rundir, "speed.json"), "w", encoding="utf-8"), indent=1)

    f = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
    md = [f"# speed: {os.path.basename(os.path.abspath(rundir.rstrip('/')))}", "",
          "| P plain | C cash | Z1 cash at zero overhead | Z2 perfect cache | C-Z1 overhead | Z1-Z2 missed reuse |",
          "|---|---|---|---|---|---|",
          f"| {f(P)} | {f(C)} | {f(Z1)} | {f(Z2)} | {f(res['overhead_C_minus_Z1'])} | {f(res['missed_Z1_minus_Z2'])} |", ""]
    if res["ratios"]:
        md.append("Ratios: " + ", ".join(f"{k} = {v}" for k, v in res["ratios"].items()))
    md.append(f"Runs: {', '.join(f'{k} {v:.2f}s' for k, v in tot.items())} (P from {pbest}, C and Z1 from {cbest}; Z1 by cash run: {res['Z1_by_cash_run']})")
    md.append(f"Served = label-based + wall-based credit for inner hits: RESTORED/SKIPPED count in full, MIXED by its saved share; "
              f"a COMPUTED/MIXED statement with cash < {WALL_RATIO} x plain and plain - cash > {WALL_MIN}s is credited plain - cash "
              f"(or cash's own sub-call hits, when the replay captured them, if larger); step walls when statements cannot be matched. "
              f"Seconds served by method: {json.dumps(how)}")
    md.append(f"Z2 per step = max(model free, served); the model alone gives Z2_model = {Z2_model:.2f}")
    md.append("Noise: " + "; ".join(f"{k}: spread {v['spread']}s ({v['rel']*100:.1f}%)" for k, v in noise.items() if k in ("plain", "cash")))
    md += ["", "## Top 10 steps by overhead (cash time minus plain time plus served time)", "",
           "| step | op | cell | plain | cash | served | overhead | excerpt |", "|---|---|---|---|---|---|---|---|"]
    for s in top_over:
        md.append(f"| {s['i']} | {s['op']} | {s['cell']} | {f(s['plain'])} | {f(s['cash'])} | {f(s['served'])} | {f(s['overhead'])} | {(s.get('overhead_excerpt') or '').replace('|', '/')} |")
    md += ["", "## Top 10 steps by missed reuse (Z2 free minus served by cash)", "",
           "| step | op | cell | plain | free (Z2) | served (Z1) | missed | statements/calls |", "|---|---|---|---|---|---|---|---|"]
    for s in top_missed:
        if s["missed"] <= 0:
            continue
        md.append(f"| {s['i']} | {s['op']} | {s['cell']} | {f(s['plain'])} | {f(s['free'])} | {f(s['served'])} | {f(s['missed'])} | {(s.get('missed_excerpt') or '').replace('|', '/')} |")
    md += ["", "## Noise: steps with the largest run-to-run spread", "", "| mode | step | cell | run a | run b | spread |", "|---|---|---|---|---|---|"]
    for s in noise["top_step_spread"]:
        md.append(f"| {s['mode']} | {s['i']} | {s['cell']} | {s['a']:.2f} | {s['b']:.2f} | {s['spread']:.2f} |")
    open(os.path.join(rundir, "speed.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md[:9]))


if __name__ == "__main__":
    main()
