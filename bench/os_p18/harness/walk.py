"""Commit walk of one GitHub repo, plain vs cash with the cache carried from commit to commit.

usage: walk.py WALK_DIR RUNS_DIR [--venv VENV] [--first N] [--only-cash-ref-label LABEL]

WALK_DIR/walk.json:
  {"id": "G1-networkx",
   "repo": "https://github.com/networkx/networkx",
   "commits": ["<sha oldest>", ..., "<sha newest>"],     # walked in this order
   "venv": "/home/user/cs/venvs/G1",                      # python with the frozen deps + cash editable
   "install": "editable" | "pythonpath:<subdir>",          # how the walked repo is made importable
   "decorate": "decorate.py",                             # run as `python decorate.py CHECKOUT` after each checkout
   "driver": "driver.py",                                 # run as `python driver.py OUT_DIR` from a fixed work dir
   "extra_steps": {"<sha>": ["uv pip install --python {venv} scipy==1.16.2"]},  # optional, run BEFORE that commit's runs
   "timeout": 1800}

Per commit: check out (detached, forced), run decorate.py, install, then
  plain: CASH_DISABLE=1, OUT_DIR fresh            -> stdout, files digests, wall
  cash : CASH_CACHE_DIR kept for the whole walk, verify on (CASH_HITPATH_VERIFY=1, CASH_CWATCH_VERIFY=1),
         CASH_VERBOSE=1 (HIT/MISS lines counted)  -> same records
A commit where cash differs from plain gets a second plain run; a difference plain shows too is
masked as nondeterministic. Anything else is WRONG. An error counts as wrong unless plain raises it too.
Records: RUNS_DIR/walk.jsonl (one line per commit and mode), RUNS_DIR/summary.md.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

ADDR = re.compile(r"0x[0-9a-fA-F]{6,}")
TIMING = re.compile(r"\b\d+(\.\d+)?\s?(s|ms|us|µs|sec|seconds)\b")


def sh(cmd, cwd=None, env=None, timeout=None):
    p = subprocess.run(cmd, cwd=cwd, env=env, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout)
    return p


def norm(text):
    lines = []
    for line in text.splitlines():
        if line.startswith(("[Cash]", "cash.", "cash:", "Cash ")):
            continue
        lines.append(TIMING.sub("<t>", ADDR.sub("0x?", line)))
    return "\n".join(lines)


def digest_dir(d):
    out = {}
    for root, _, files in os.walk(d):
        for f in sorted(files):
            p = os.path.join(root, f)
            with open(p, "rb") as fh:
                out[os.path.relpath(p, d)] = hashlib.sha256(fh.read()).hexdigest()[:16]
    return out


def clean_env(venv):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("CASH_", "PYTHON", "VIRTUAL_ENV"))}
    e.update(VIRTUAL_ENV=venv, PATH=os.path.join(venv, "bin") + os.pathsep + e.get("PATH", ""),
             PYTHONHASHSEED="0", MPLBACKEND="Agg", OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    return e


def run_driver(w, walk_dir, venv, mode, workdir, cachedir, timeout, extra_env=None):
    out = os.path.join(workdir, "out")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    e = clean_env(venv)
    e.update(extra_env or {})
    if mode == "plain":
        e["CASH_DISABLE"] = "1"
    else:
        e.update(CASH_CACHE_DIR=cachedir, CASH_VERBOSE="1", CASH_HITPATH_VERIFY="1", CASH_CWATCH_VERIFY="1")
    t = time.perf_counter()
    try:
        p = subprocess.run([os.path.join(venv, "bin", "python"), os.path.join(walk_dir, w["driver"]), out],
                           cwd=workdir, env=e, capture_output=True, text=True, timeout=timeout)
        rc, so, se = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as ex:
        rc, so, se = "TIMEOUT", (ex.stdout or b"").decode() if isinstance(ex.stdout, bytes) else (ex.stdout or ""), ""
    wall = time.perf_counter() - t
    err_tail = [l for l in se.splitlines() if l.strip() and not l.startswith(("[Cash]", "cash."))][-3:]
    hits = len(re.findall(r"^cash\.calls: HIT\b", se, re.M))
    misses = len(re.findall(r"^cash\.calls: MISS\b", se, re.M))
    verify_fail = [l for l in se.splitlines() if "HitMemoMismatch" in l or "WatchMismatch" in l][:5]
    warns = sorted(set(re.findall(r"\b([A-Z]+(?:-[A-Z]+)+)\b", se)))
    return {"mode": mode, "rc": rc, "wall": round(wall, 3), "stdout": norm(so), "files": digest_dir(out),
            "err": err_tail if rc != 0 else [], "hits": hits, "misses": misses, "verify_fail": verify_fail,
            "warnings": warns[:40]}


def same(a, b):
    return a["rc"] == b["rc"] and a["stdout"] == b["stdout"] and a["files"] == b["files"] and (a["rc"] == 0 or a["err"][-1:] == b["err"][-1:])


def diff_text(a, b):
    parts = []
    if a["rc"] != b["rc"]:
        parts.append(f"exit {a['rc']} vs {b['rc']} ({(b['err'] or a['err'] or [''])[-1][:200]})")
    if a["stdout"] != b["stdout"]:
        al, bl = a["stdout"].splitlines(), b["stdout"].splitlines()
        for i, (x, y) in enumerate(zip(al + [""] * len(bl), bl + [""] * len(al))):
            if x != y:
                parts.append(f"stdout line {i}: plain {x[:150]!r} cash {y[:150]!r}")
                break
    fa, fb = a["files"], b["files"]
    for k in sorted(set(fa) | set(fb)):
        if fa.get(k) != fb.get(k):
            parts.append(f"file {k}: plain {fa.get(k)} cash {fb.get(k)}")
    return "; ".join(parts[:6])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("walk_dir")
    ap.add_argument("runs_dir")
    ap.add_argument("--venv")
    ap.add_argument("--first", type=int, default=0, help="walk only the first N commits")
    a = ap.parse_args()
    walk_dir = os.path.abspath(a.walk_dir)
    w = json.load(open(os.path.join(walk_dir, "walk.json")))
    venv = a.venv or w["venv"]
    timeout = w.get("timeout", 1800)
    commits = w["commits"][: a.first or None]
    os.makedirs(a.runs_dir, exist_ok=True)
    tmp = subprocess.check_output(["mktemp", "-d", "/tmp/cs-walk-XXXX"], text=True).strip()
    assert re.fullmatch(r"/tmp/cs-walk-[A-Za-z0-9]{4}", tmp)
    try:
        repo = os.path.join(tmp, "repo")
        cachedir = os.path.join(tmp, "cache")
        work = {m: os.path.join(tmp, "work-" + m) for m in ("plain", "cash")}
        for d in (cachedir, *work.values()):
            os.makedirs(d)
        src = w["local_clone"] if os.path.isdir(w.get("local_clone") or "/nonexistent") else w["repo"]
        r = sh(["git", "clone", "-q", src, repo])
        if r.returncode:
            sys.exit("clone failed: " + r.stderr)
        sh(["git", "fetch", "-q", "origin", *commits], cwd=repo)
        cash_rev = sh([os.path.join(venv, "bin", "python"), "-c", "import cash,os,subprocess;d=os.path.dirname(cash.__file__);print(subprocess.run(['git','-C',d,'rev-parse','--short','HEAD'],capture_output=True,text=True).stdout.strip())"], env=clean_env(venv)).stdout.strip()
        log = open(os.path.join(a.runs_dir, "walk.jsonl"), "w")
        log.write(json.dumps({"op": "meta", "id": w["id"], "cash": cash_rev, "n": len(commits), "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
        rows = []
        for i, sha in enumerate(commits):
            r = sh(["git", "checkout", "-q", "-f", "--detach", sha], cwd=repo)
            sh(["git", "clean", "-qfdx", "-e", "*.egg-info"], cwd=repo)
            if r.returncode:
                rows.append({"i": i, "sha": sha, "verdict": "CHECKOUT-FAILED", "detail": r.stderr[-200:]})
                continue
            for cmd in w.get("extra_steps", {}).get(sha, []):
                sh(cmd.format(venv=venv, repo=repo), env=clean_env(venv))
            d = sh([os.path.join(venv, "bin", "python"), os.path.join(walk_dir, w["decorate"]), repo], env=clean_env(venv))
            if d.returncode:
                rows.append({"i": i, "sha": sha, "verdict": "DECORATE-FAILED", "detail": d.stderr[-300:]})
                log.write(json.dumps(rows[-1]) + "\n"); log.flush()
                continue
            inst = w.get("install", "editable")
            extra_env = {}
            if inst == "editable":
                sh(["uv", "pip", "install", "-q", "--no-deps", "--python", os.path.join(venv, "bin", "python"), "-e", repo], env=clean_env(venv))
            elif inst.startswith("pythonpath:"):
                # not a package (e.g. nanoGPT): the checkout (or a subdir) goes on PYTHONPATH for the driver
                extra_env["PYTHONPATH"] = os.path.normpath(os.path.join(repo, inst.split(":", 1)[1]))
            rec = {}
            for mode in ("plain", "cash"):
                rec[mode] = run_driver(w, walk_dir, venv, mode, work[mode], cachedir, timeout, extra_env)
            verdict, detail = "OK", ""
            if not same(rec["plain"], rec["cash"]):
                p2 = run_driver(w, walk_dir, venv, "plain", work["plain"], cachedir, timeout, extra_env)
                rec["plain2"] = p2
                if same(p2, rec["cash"]) or not same(p2, rec["plain"]):
                    verdict, detail = "NONDET", diff_text(rec["plain"], rec["cash"])
                else:
                    verdict, detail = "WRONG", diff_text(rec["plain"], rec["cash"])
            if rec["cash"]["verify_fail"]:
                verdict, detail = "WRONG", "verify: " + " | ".join(rec["cash"]["verify_fail"])
            subj = sh(["git", "log", "-1", "--format=%s"], cwd=repo).stdout.strip()[:80]
            files_changed = sh(["git", "diff", "--name-only", f"{sha}~1", sha], cwd=repo).stdout.split()
            row = {"i": i, "sha": sha[:10], "subject": subj, "changed": files_changed[:30], "verdict": verdict, "detail": detail,
                   "plain_s": rec["plain"]["wall"], "cash_s": rec["cash"]["wall"], "hits": rec["cash"]["hits"],
                   "misses": rec["cash"]["misses"], "warnings": rec["cash"]["warnings"], "runs": rec}
            rows.append(row)
            log.write(json.dumps(row) + "\n"); log.flush()
            print(f"[{i+1}/{len(commits)}] {sha[:10]} {verdict:7s} plain {row['plain_s']:7.1f}s cash {row['cash_s']:7.1f}s hits {row['hits']} misses {row['misses']} {detail[:120]}", flush=True)
        log.close()
        write_summary(w, rows, cash_rev, a.runs_dir)
    finally:
        if re.fullmatch(r"/tmp/cs-walk-[A-Za-z0-9]{4}", tmp):
            shutil.rmtree(tmp, ignore_errors=True)


def write_summary(w, rows, cash_rev, runs_dir):
    ok = [r for r in rows if "plain_s" in r]
    wrong = [r for r in rows if r["verdict"] == "WRONG"]
    nondet = [r for r in rows if r["verdict"] == "NONDET"]
    broken = [r for r in rows if r["verdict"] not in ("OK", "WRONG", "NONDET")]
    P = sum(r["plain_s"] for r in ok)
    C = sum(r["cash_s"] for r in ok)
    L = [f"# Walk {w['id']} on cash {cash_rev}", "",
         f"{len(rows)} commits: {len(wrong)} WRONG, {len(nondet)} nondeterministic (masked), {len(broken)} not run.",
         f"Plain {P:.0f} s, cash {C:.0f} s (ratio {C / P if P else 0:.2f}; cash runs had verify on, so direction only).",
         f"Cache hits {sum(r['hits'] for r in ok)}, misses {sum(r['misses'] for r in ok)}.", "",
         "| # | commit | verdict | plain s | cash s | hits | misses | subject / detail |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['i']} | {r['sha'][:10]} | {r['verdict']} | {r.get('plain_s', '')} | {r.get('cash_s', '')} | {r.get('hits', '')} | {r.get('misses', '')} | {(r.get('subject', '') + ' ' + r.get('detail', ''))[:200].replace('|', '/')} |")
    with open(os.path.join(runs_dir, "summary.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L[:5]))


if __name__ == "__main__":
    main()
