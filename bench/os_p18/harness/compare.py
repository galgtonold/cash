#!/usr/bin/env python3
"""Correctness oracle for one round-34 project.

usage: compare.py RUNDIR

Reads RUNDIR/{plain1,plain2,cash1,cash2}.jsonl (written by replay.py) and
writes RUNDIR/compare.json and RUNDIR/compare.md.

Compared per step: status and error text, normalized stdout ('out'), displayed
values ('disp': text/plain plus image digests), every user-namespace value
digest ('value'), and the content digest of every file under the project dir
after the step ('file', cumulative state, not only the files the step wrote).

Everything that differs between plain1 and plain2 is nondeterministic and
masked. Every remaining plain-vs-cash difference (and cash1-vs-cash2
difference) is a WRONG candidate. A step that is ok in both plain runs but an
error under cash is a BLOCKING candidate.
"""

from __future__ import annotations

import difflib
import json
import os
import sys
from collections import defaultdict

RUNS = ("plain1", "plain2", "cash1", "cash2")


def load(path):
    recs = {}
    meta = None
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
            recs[r["i"]] = r
    return meta, recs


def items(meta, recs):
    """{(i, kind, key): value} for every comparable item of one run."""
    out = {}
    files = dict((meta or {}).get("files0") or {})
    for i in sorted(recs):
        r = recs[i]
        for path, d in (r.get("files") or {}).items():
            if d is None:
                files.pop(path, None)
            else:
                files[path] = d
        if r["op"] == "cell":
            continue
        out[(i, "status", "")] = r.get("status")
        out[(i, "error", "")] = r.get("error") or ""
        if "out" in r:
            out[(i, "out", "")] = r.get("out") or ""
        if "disp" in r:
            out[(i, "disp", "")] = json.dumps(r.get("disp") or [], sort_keys=True)
        if "rc" in r:
            out[(i, "rc", "")] = r.get("rc")
        vals = r.get("values")
        if isinstance(vals, dict):
            out[(i, "names", "")] = " ".join(sorted(vals))
            for k, v in vals.items():
                out[(i, "value", k)] = v
        for path, d in files.items():
            out[(i, "file", path)] = d
        out[(i, "files", "")] = " ".join(sorted(files))
    return out


def excerpt(a, b, la="plain", lb="cash", n=12):
    a, b = "" if a is None else str(a), "" if b is None else str(b)
    if len(a) < 120 and len(b) < 120 and "\n" not in a + b:
        return f"{la}: {a!r}  {lb}: {b!r}"
    d = list(difflib.unified_diff(a.splitlines(), b.splitlines(), la, lb, n=1, lineterm=""))
    return "\n".join(d[2 : 2 + n])[:1500]


def main():
    rundir = sys.argv[1] if len(sys.argv) > 1 else sys.exit(__doc__)
    data = {name: load(os.path.join(rundir, name + ".jsonl")) for name in RUNS}
    present = [n for n in RUNS if data[n][1] is not None]
    res = {"rundir": os.path.abspath(rundir), "runs": {}, "notes": []}
    for n in present:
        meta, recs = data[n]
        res["runs"][n] = {
            "steps": len([i for i in recs if i >= 0]),
            "expected": (meta or {}).get("n_steps"),
            "errors": sum(1 for r in recs.values() if r.get("status") == "error"),
            "problems": (meta or {}).get("problems"),
        }
        if meta and res["runs"][n]["steps"] != meta.get("n_steps"):
            res["notes"].append(f"{n}: only {res['runs'][n]['steps']} of {meta.get('n_steps')} steps recorded (crashed or still running)")
    it = {n: items(*data[n]) for n in present}
    recs = {n: data[n][1] for n in present}

    masked = set()
    if "plain1" in it and "plain2" in it:
        p1, p2 = it["plain1"], it["plain2"]
        for k in set(p1) | set(p2):
            if p1.get(k) != p2.get(k):
                masked.add(k)
    else:
        res["notes"].append("plain2 missing: nothing is masked")
    # A step both plain runs did not reach cannot be compared.
    base = "plain1" if "plain1" in it else None
    if base is None:
        res["notes"].append("plain1 missing: no comparison")
    diffs = []
    pairs = [(base, c) for c in ("cash1", "cash2") if c in it and base] + (
        [("cash1", "cash2")] if "cash1" in it and "cash2" in it else []
    )
    for a, b in pairs:
        A, B = it[a], it[b]
        steps_both = set(recs[a]) & set(recs[b])
        for k in sorted(set(A) | set(B), key=lambda k: (k[0], k[1], k[2])):
            if k in masked or k[0] not in steps_both:
                continue
            va, vb = A.get(k), B.get(k)
            if va == vb:
                continue
            i, kind, key = k
            r = recs[b].get(i) or recs[a].get(i) or {}
            d = {
                "pair": f"{a}-{b}",
                "i": i,
                "op": r.get("op"),
                "cell": r.get("cell") or r.get("cmd"),
                "kind": kind,
                "key": key,
                a: va if kind not in ("out", "disp", "error") else None,
                b: vb if kind not in ("out", "disp", "error") else None,
            }
            if kind in ("out", "disp", "error", "names", "files", "status"):
                d["excerpt"] = excerpt(va, vb, a, b)
            diffs.append(d)

    blocking = []
    if base:
        for c in ("cash1", "cash2"):
            if c not in recs:
                continue
            for i, r in sorted(recs[c].items()):
                if r.get("status") != "error":
                    continue
                pl = [recs[p].get(i, {}).get("status") for p in ("plain1", "plain2") if p in recs]
                if pl and all(s == "ok" for s in pl):
                    blocking.append(
                        {"run": c, "i": i, "op": r.get("op"), "cell": r.get("cell") or r.get("cmd"), "error": r.get("error"),
                         "src": (r.get("src") or r.get("cmd") or "")[:400]}
                    )

    # group persisting value/file differences: first step + how many steps
    groups = defaultdict(list)
    for d in diffs:
        groups[(d["pair"], d["kind"], d["key"])].append(d["i"])
    res.update(
        masked=len(masked),
        masked_items=[list(k) for k in sorted(masked, key=lambda k: (k[0], k[1], k[2]))][:500],
        n_diffs=len(diffs),
        n_wrong_candidates=len({(g[1], g[2], v[0]) for g, v in groups.items() if g[0] != "cash1-cash2"}),
        blocking=blocking,
        diffs=diffs,
        groups=[{"pair": g[0], "kind": g[1], "key": g[2], "first_step": v[0], "steps": len(v)} for g, v in groups.items()],
    )
    json.dump(res, open(os.path.join(rundir, "compare.json"), "w", encoding="utf-8"), indent=1, default=str)

    md = [f"# compare: {os.path.basename(os.path.abspath(rundir.rstrip('/')))}", ""]
    for n, info in res["runs"].items():
        md.append(f"- {n}: {info['steps']}/{info['expected']} steps, {info['errors']} error steps"
                  + (f", problems: {info['problems']}" if info["problems"] else ""))
    for note in res["notes"]:
        md.append(f"- NOTE: {note}")
    md.append(f"- masked (plain1 != plain2): {len(masked)} items"
              + (": " + ", ".join(f"{k[0]}/{k[1]}/{k[2]}" for k in sorted(masked)[:15]) if masked else ""))
    md.append(f"- differences after masking: {len(diffs)} item-steps in {len(groups)} groups")
    md.append(f"- BLOCKING candidates (error only under cash): {len(blocking)}")
    md.append("")
    if blocking:
        md += ["## BLOCKING candidates", "", "| run | step | op | cell | error |", "|---|---|---|---|---|"]
        for b in blocking:
            md.append(f"| {b['run']} | {b['i']} | {b['op']} | {b['cell']} | {str(b['error'])[:150].replace('|', '/')} |")
        md.append("")
    if groups:
        md += ["## Differences (first step where each starts)", "",
               "| pair | first step | steps | op | cell | kind | name/file | plain vs cash |", "|---|---|---|---|---|---|---|---|"]
        seen = set()
        for d in diffs:
            g = (d["pair"], d["kind"], d["key"])
            if g in seen:
                continue
            seen.add(g)
            a, b = d["pair"].split("-")
            what = d.get("excerpt") or f"{d.get(a)} vs {d.get(b)}"
            what = what.replace("\n", "<br>").replace("|", "/")[:400]
            md.append(f"| {d['pair']} | {d['i']} | {len(groups[g])} | {d['op']} | {d['cell']} | {d['kind']} | {d['key']} | {what} |")
        md.append("")
    else:
        md.append("No differences between plain and cash after masking.")
    open(os.path.join(rundir, "compare.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md[:12]))


if __name__ == "__main__":
    main()
