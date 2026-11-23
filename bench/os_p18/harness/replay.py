#!/usr/bin/env python3
"""Replay one round-34 session trace, plain or with cash, and log every step.

usage: replay.py TRACE PROJECT_DIR {plain,cash} OUT.jsonl

* Copies PROJECT_DIR (without .venv, runs/, __pycache__, .git, .cash*) to a
  fresh temp dir (r34-XXXXXXXX in the system temp folder) and deletes that exact dir at the end.
* Python: PROJECT_DIR/.venv if it exists (used in place, not copied), else the
  interpreter running this script. It needs ipykernel for notebook steps and
  cash for cash mode.
* Runs the trace's setup commands (CASH_DISABLE=1 in both modes, not timed),
  then the steps in order. One JSON line per step is appended to OUT as the
  step finishes. See README.md for the record fields.
* Env for everything: empty CASH_CACHE_DIR in the temp dir (kept for the whole
  replay), MPLBACKEND=Agg, PYTHONHASHSEED=0, an empty IPYTHONDIR, inherited
  CASH_* variables removed. Plain mode adds CASH_DISABLE=1 everywhere (kernel
  included, so a 'both' project's helper module does not cache either).
* Per-step timeout R34_STEP_TIMEOUT seconds (default 1800); a timeout is
  recorded as an error (the cell is interrupted, the kernel restarted if it
  does not go idle, a script's process group is killed).
"""

from __future__ import annotations

import ast
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from queue import Empty

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r34digest  # noqa: E402

STEP_TIMEOUT = float(os.environ.get("R34_STEP_TIMEOUT", 1800))
BADGE_MODE = os.environ.get("R34_BADGE", "")  # "" keeps cash's default (html); "print" for the text badge
OUT_CAP = 200_000  # characters of stdout kept per step
ERR_CAP = 20_000_000  # characters of a cell's stderr read for cash.calls lines (only the tail is stored)
DISP_CAP = 20_000  # characters of one display's text/plain
EXCLUDE_DIRS = {".venv", "runs", "__pycache__", ".git", ".ipynb_checkpoints", ".mypy_cache", ".pytest_cache", ".ruff_cache"}

# ---------------------------------------------------------------- normalization

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07")
_ADDR = re.compile(r"0x[0-9a-fA-F]{4,}")
_TIME = re.compile(
    r"(?<![\w.])\d+(?:\.\d+)?(?:e-?\d+)?[ ]?(?:s|sec|secs|seconds|ms|µs|us|ns|min|mins|minutes)\b(?:[ ]*±[ ]*\d+(?:\.\d+)?[ ]?(?:s|ms|µs|us|ns))?"
)
_WALL = re.compile(r"^(CPU times|Wall time):.*$", re.M)
_BADGE_STATUS = (
    r"EXECUTED|CACHED|NOT CACHED|SKIPPED|ERROR|MIXED|FUNC CHANGED|MODULE RELOADED|WARNING|BYPASSED|RESTORED|COMPUTED"
)
_BADGE_ROW = re.compile(r"^\s+-?\s*(?:" + _BADGE_STATUS + r")\b[^:\n]*:")
_CASH_LINE = re.compile(r"^(?:\[Cash\]|cash[.:]|cash \(pid|Cash enabled|Badge mode)", re.I)


def strip_cash_lines(s: str) -> str:
    out, in_badge = [], False
    for line in s.split("\n"):
        if _CASH_LINE.match(line):
            in_badge = line.startswith("[Cash]")
            continue
        if in_badge and (_BADGE_ROW.match(line) or line.startswith("    ") or line.startswith("  ")):
            continue
        in_badge = False
        out.append(line)
    return "\n".join(out)


def normalize(s: str | None, tmp: str = "") -> str:
    if not s:
        return ""
    s = _ANSI.sub("", s)
    s = s.replace("\r\n", "\n")
    # progress bars rewrite the line with \r: keep the final state of each line
    s = "\n".join(line.split("\r")[-1] for line in s.split("\n"))
    s = strip_cash_lines(s)
    if tmp:
        s = s.replace(tmp, "<TMP>")
    s = _ADDR.sub("0x?", s)
    s = _WALL.sub("<TIMING>", s)
    s = _TIME.sub("<T>", s)
    return s.strip()


def is_cash_display(data: dict) -> bool:
    html = data.get("text/html", "") or ""
    plain = data.get("text/plain", "") or ""
    if isinstance(html, list):
        html = "".join(html)
    return (
        "c3-wrap" in html
        or 'class="c3' in html
        or "cash-badge" in html
        or "cash_badge" in html
        or plain.startswith("[Cash]")
        or "application/vnd.cash" in "".join(data.keys())
        or ("Cash" in html[:4000] and ("EXECUTED" in html or "CACHED" in html))
    )


# ---------------------------------------------------------------- files


def scan_files(root: str, memo: dict) -> dict:
    """{relpath: digest} of every file under root (skipping EXCLUDE_DIRS)."""
    out = {}
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in EXCLUDE_DIRS and not x.startswith(".cash"))
        for f in sorted(files):
            p = os.path.join(d, f)
            if os.path.islink(p):
                continue
            try:
                st = os.stat(p)
            except OSError:
                continue
            key = (st.st_size, st.st_mtime_ns)
            rel = os.path.relpath(p, root).replace(os.sep, "/")
            if memo.get(rel, (None,))[0] != key:
                try:
                    memo[rel] = (key, r34digest.file_digest(p))
                except OSError:
                    continue
            out[rel] = memo[rel][1]
    for rel in list(memo):
        if rel not in out:
            del memo[rel]
    return out


def file_changes(before: dict, after: dict) -> dict:
    ch = {k: v for k, v in after.items() if before.get(k) != v}
    ch.update({k: None for k in before if k not in after})
    return ch


# ---------------------------------------------------------------- kernel


KERNEL_HELPER = r"""
import sys as _r34_sys
_r34_sys.path.insert(0, {hookdir!r})
import r34digest as _r34d
_r34_sys.path.remove({hookdir!r})
def _r34_state():
    return _r34d.namespace_state(globals())
"""

PLAIN_TIMER = r"""
import ast as _r34_ast, time as _r34_time
_r34_texts, _r34_times = [], []
class _R34Rec(_r34_ast.NodeTransformer):
    def visit(self, node):
        if isinstance(node, _r34_ast.Module):
            _r34_texts.append([_r34_ast.unparse(n) for n in node.body])
        return node
get_ipython().ast_transformers.append(_R34Rec())
_r34_orig_run_code = get_ipython().run_code
async def _r34_run_code(code_obj, result=None, *, async_=False):
    _c = getattr(_r34_sys, '_r34_calls', None)
    if _c is not None:
        _c['stmt'] = len(_r34_times)
    _t = _r34_time.perf_counter()
    try:
        return await _r34_orig_run_code(code_obj, result, async_=async_)
    finally:
        _r34_times.append(_r34_time.perf_counter() - _t)
get_ipython().run_code = _r34_run_code
def _r34_stmts():
    import json
    texts = _r34_texts[0] if _r34_texts else []
    _c = getattr(_r34_sys, '_r34_calls', None)
    calls, over = [], 0.0
    if _c is not None:
        calls, over = list(_c['buf']), _c['overhead']
        _c['buf'].clear(); _c['overhead'] = 0.0; _c['stmt'] = None
    out = json.dumps({"texts": texts, "times": list(_r34_times), "n_cells": len(_r34_texts), "calls": calls, "hook_overhead": over})
    _r34_texts.clear(); _r34_times.clear()
    return out
"""

CASH_SNAP = r"""
def _r34_metrics():
    import copy
    m = get_ipython().magics_manager.registry.get('CashMagics')
    return copy.deepcopy(getattr(m, '_last_cell_metrics', None)) if m is not None else None
def _r34_snap(result=None):
    # cash fills _last_cell_metrics before IPython's own pre/post_run_cell events
    # fire; a cell cash did not record (e.g. bypassed) leaves the previous cell's
    # metrics in place, which shows as equal to the last snapshot.
    global _r34_last, _r34_prev, _r34_sub, _r34_raw
    try:
        cur = _r34_metrics()
        _r34_last = cur if cur != _r34_prev else {'stale': True, 'statements': []}
        _r34_prev = cur
        if cur is not None and _r34_last is cur and _r34_sub is not None and len(_r34_sub) == len(cur.get('statements') or []):
            _r34_last = dict(cur, subcalls=_r34_sub)
        if cur is not None and _r34_raw is not None and cur.get('stale') is not True and _r34_last is not None and not _r34_last.get('stale'):
            _r34_last = dict(_r34_last, raw=_r34_raw)
        _r34_sub = _r34_raw = None
    except Exception as e:
        _r34_last = {'error': repr(e), 'statements': []}
_r34_last = _r34_prev = _r34_sub = None
_r34_raw = None
_R34_RAW_KEYS = ('status', 'execution_time', 'saved_time', 'miss_reason', 'uncacheable_reasons', 'skipped_reason',
                 'guard_cause', 'storage', 'cash_tax', 'compute_cost', 'cost_model_size_bytes', 'cost_model_restore_seconds',
                 'cost_model_family', 'is_upstream')
def _r34_short(v):
    if isinstance(v, (list, tuple)):
        return [_r34_short(x) for x in v[:8]]
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    return str(v)[:200]
_R34_SUB_KEYS = ('func_name', 'cache_hit', 'execution_time', 'time_saved', 'intercepted', 'ran_plain', 'calls', 'call_source', 'loop_header', 'miss_reason')
def _r34_wrap_update():
    # Read-only tap on the raw per-statement results cash builds _last_cell_metrics
    # from: they carry each statement's 'decorator_calls' (intercepted helper calls
    # and @cash.cache calls, with cache_hit / time_saved), which the metrics dict drops.
    m = get_ipython().magics_manager.registry.get('CashMagics')
    orig = getattr(m, '_update_last_cell_metrics', None)
    if orig is None or getattr(orig, '_r34', False):
        return 'no'
    def wrapped(all_metrics, *a, **k):
        global _r34_sub, _r34_raw
        try:
            _r34_sub = [[{kk: (str(e.get(kk))[:120] if kk in ('call_source', 'loop_header') else e.get(kk)) for kk in _R34_SUB_KEYS if kk in e}
                         for e in (x.get('decorator_calls') or []) if isinstance(e, dict)] for x in all_metrics]
        except Exception:
            _r34_sub = None
        try:
            # cash's own decision record per statement: why it was not served / not stored
            _r34_raw = [{kk: _r34_short(x.get(kk)) for kk in _R34_RAW_KEYS if x.get(kk) not in (None, '', [], {})}
                        for x in all_metrics if isinstance(x, dict)]
        except Exception as e:
            _r34_raw = [{'error': repr(e)[:200]}]
        return orig(all_metrics, *a, **k)
    wrapped._r34 = True
    m._update_last_cell_metrics = wrapped
    return 'ok'

def _r34_stmts():
    import json
    out = dict(_r34_last) if isinstance(_r34_last, dict) else {'statements': [], 'none': True}
    try:
        import cs_stats as _cs
        out['work'] = _cs.take()
    except Exception as e:
        out['work'] = {'error': repr(e)[:200]}
    return json.dumps(out, default=str)
def _r34_cs_install():
    import sys
    sys.path.insert(0, {hookdir!r})
    try:
        import cs_stats
    finally:
        sys.path.remove({hookdir!r})
    return cs_stats.install()
"""


def _venv_python(venv: str) -> str:
    if os.name == "nt":
        return os.path.join(venv, "Scripts", "python.exe")
    return os.path.join(venv, "bin", "python")


def _is_own_tmp(t: str) -> bool:
    return os.path.dirname(os.path.abspath(t)) == os.path.abspath(tempfile.gettempdir()) and re.fullmatch(
        r"r34-[A-Za-z0-9_]{6,12}", os.path.basename(t)
    ) is not None


class Kernel:
    def __init__(self, rp: "Replay"):
        self.rp = rp
        self.km = None
        self.kc = None

    def start(self) -> dict:
        from jupyter_client import KernelManager

        t0 = time.perf_counter()
        km = KernelManager(kernel_name="r34py")
        km.start_kernel(cwd=self.rp.proj, env=self.rp.env(kernel=True))
        kc = km.client()
        kc.start_channels()
        kc.wait_for_ready(timeout=180)
        self.km, self.kc = km, kc
        notes = []
        r = self.execute(KERNEL_HELPER.format(hookdir=self.rp.hookdir), timeout=120, setup=True)
        if r.get("error"):
            notes.append("helper: " + r["error"])
        if self.rp.mode == "plain":
            r = self.execute(PLAIN_TIMER, timeout=120, setup=True)
            if r.get("error"):
                notes.append("timer: " + r["error"])
            self.fetch_stmts()
        else:
            # the snapshot hooks go in before %cash_on, so cash never processes the harness's code
            r2 = self.execute(CASH_SNAP.replace("{hookdir!r}", repr(self.rp.hookdir)), timeout=120, setup=True)
            if r2.get("error"):
                notes.append("snap: " + r2["error"])
            init = "import cash\n%cash_on" + ("\n%cash_badge " + BADGE_MODE if BADGE_MODE else "")
            r = self.execute(init, timeout=600, setup=True)
            if r.get("error"):
                notes.append("cash init: " + r["error"])
            notes.append("cash init stdout: " + (r.get("raw_stdout") or "")[:400])
            # Registered after %cash_on (cash fills its metrics in its own post_run_cell
            # handler, which must run first) through user_expressions, which do not go
            # through run_cell, so cash never sees this either.
            x = self.user_expr(
                "[_r34_snap(), get_ipython().events.register('post_run_cell', _r34_snap), _r34_wrap_update()] and 'ok'",
                timeout=120,
            )
            if x != "ok":
                notes.append(f"snap register: {x}")
            # cash's work counters (cs_stats.py): installed after %cash_on so cash's modules are loaded
            x = self.user_expr("_r34_cs_install()", timeout=120)
            notes.append(f"cs_stats: {x}")
        return {"secs": time.perf_counter() - t0, "notes": notes}

    def alive(self) -> bool:
        return self.km is not None and self.km.is_alive()

    def stop(self):
        if self.km is None:
            return
        try:
            self.kc.stop_channels()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.km.shutdown_kernel(now=True)
        except Exception:  # noqa: BLE001
            pass
        self.km = self.kc = None

    def execute(self, code: str, timeout: float = STEP_TIMEOUT, setup: bool = False) -> dict:
        kc, km = self.kc, self.km
        t0 = time.perf_counter()
        msg_id = kc.execute(code, store_history=not setup)
        res = {"stdout": "", "stderr": "", "disp": [], "error": None, "raw_stdout": ""}
        deadline = t0 + timeout
        idle = reply = False
        status = None
        while not (idle and reply):
            now = time.perf_counter()
            if now > deadline:
                res["error"] = "TIMEOUT after %ds" % timeout
                res["timeout"] = True
                try:
                    km.interrupt_kernel()
                except Exception:  # noqa: BLE001
                    pass
                if not self._drain(msg_id, 120):
                    res["need_restart"] = True
                break
            if not idle:
                try:
                    m = kc.get_iopub_msg(timeout=min(2.0, max(0.05, deadline - now)))
                except Empty:
                    if not km.is_alive():
                        res["error"] = "KERNEL_DIED"
                        res["need_restart"] = True
                        break
                    continue
                if m["parent_header"].get("msg_id") != msg_id:
                    continue
                t, c = m["msg_type"], m["content"]
                if t == "stream":
                    key = "stdout" if c["name"] == "stdout" else "stderr"
                    cap = OUT_CAP if key == "stdout" else ERR_CAP  # stderr carries cash's per-call lines
                    if len(res[key]) < cap:
                        res[key] += c["text"][: cap - len(res[key])]
                elif t in ("execute_result", "display_data", "update_display_data"):
                    data = c.get("data", {})
                    if is_cash_display(data):
                        res.setdefault("cash_disp", 0)
                        res["cash_disp"] += 1
                        continue
                    d = {"text": (data.get("text/plain", "") or "")[:DISP_CAP]}
                    for mime in ("image/png", "image/svg+xml", "image/jpeg"):
                        if mime in data:
                            d["img"] = r34digest._hex(mime, str(data[mime]))
                    if "text/html" in data and not d["text"]:
                        d["html"] = r34digest._hex(str(data["text/html"]))
                    res["disp"].append(d)
                elif t == "error":
                    res["error"] = f"{c['ename']}: {c['evalue']}"
                elif t == "status" and c["execution_state"] == "idle":
                    idle = True
            if idle and not reply:
                try:
                    r = kc.get_shell_msg(timeout=max(0.05, min(5.0, deadline - time.perf_counter())))
                except Empty:
                    if not km.is_alive():
                        res["error"] = "KERNEL_DIED"
                        res["need_restart"] = True
                        break
                    continue
                if r["parent_header"].get("msg_id") == msg_id:
                    reply = True
                    status = r["content"].get("status")
                    if status == "error" and not res["error"]:
                        res["error"] = f"{r['content'].get('ename')}: {r['content'].get('evalue')}"
        res["secs"] = time.perf_counter() - t0
        res["raw_stdout"] = res["stdout"][:2000] if setup else ""
        return res

    def _drain(self, msg_id: str, secs: float) -> bool:
        end = time.perf_counter() + secs
        while time.perf_counter() < end:
            try:
                m = self.kc.get_iopub_msg(timeout=2)
            except Empty:
                if not self.km.is_alive():
                    return False
                continue
            if (
                m["parent_header"].get("msg_id") == msg_id
                and m["msg_type"] == "status"
                and m["content"]["execution_state"] == "idle"
            ):
                try:
                    self.kc.get_shell_msg(timeout=5)
                except Empty:
                    pass
                return True
        return False

    def user_expr(self, expr: str, timeout: float = 1800) -> str | None:
        """Evaluate *expr* in a silent request (no run_cell, no history)."""
        mid = self.kc.execute("", silent=True, store_history=False, user_expressions={"x": expr})
        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            try:
                m = self.kc.get_shell_msg(timeout=5)
            except Empty:
                if not self.km.is_alive():
                    return None
                continue
            if m["parent_header"].get("msg_id") != mid:
                continue
            x = m["content"].get("user_expressions", {}).get("x", {})
            if x.get("status") != "ok":
                return json.dumps({"_error": f"{x.get('ename')}: {x.get('evalue')}"})
            txt = x.get("data", {}).get("text/plain")
            try:
                return ast.literal_eval(txt)
            except Exception:  # noqa: BLE001
                return txt
        return None

    def fetch_stmts(self):
        s = self.user_expr("_r34_stmts()", timeout=120)
        try:
            return json.loads(s) if s else None
        except Exception:  # noqa: BLE001
            return {"_raw": str(s)[:500]}

    def fetch_state(self):
        s = self.user_expr("_r34_state()")
        try:
            return json.loads(s) if s else None
        except Exception:  # noqa: BLE001
            return {"_raw": str(s)[:500]}


# ---------------------------------------------------------------- replay

_VERBOSE = re.compile(
    r"^(?:\S+ )*?cash\.calls: (HIT|MISS|RAISE|BYPASS\w*|SKIP\w*|NOT STORED|\w+)\s+(\S+)\s*(?:\[([0-9a-f]+)\])?(.*)$"
)
_RAN = re.compile(r"\((?:ran|saved) ([\d.]+)s\)\s*$")


_WARN_CODE = re.compile(r"(\w*Warning): \[([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+|[A-Z]\d+)\]")


def parse_warnings(stderr: str) -> dict:
    """{"Category [CODE]": count} for the warnings a cell printed (cash's carry a [CODE])."""
    out: dict = {}
    for m in _WARN_CODE.finditer(stderr or ""):
        k = f"{m.group(1)} [{m.group(2)}]"
        out[k] = out.get(k, 0) + 1
    return out


def dir_bytes(root: str) -> list:
    """[bytes, files] under root (the cash cache at the end of a replay)."""
    n = f = 0
    for dp, _dn, fns in os.walk(root):
        for fn in fns:
            try:
                n += os.path.getsize(os.path.join(dp, fn))
                f += 1
            except OSError:
                pass
    return [n, f]


def parse_cash_calls(stderr: str) -> list:
    calls = []
    for line in stderr.splitlines():
        m = _VERBOSE.match(line.strip())
        if not m:
            continue
        kind, fn, key, rest = m.groups()
        t = _RAN.search(rest)
        calls.append(
            {
                "fn": fn,
                "q": fn.rsplit(".", 1)[-1] if "." in fn else fn,
                "outcome": kind,
                "key": key,
                "t": float(t.group(1)) if t else None,
                "why": rest.strip()[:300],
            }
        )
    return calls


class Replay:
    def __init__(self, trace_path: str, project: str, mode: str, out: str):
        self.trace = json.load(open(trace_path, encoding="utf-8"))
        self.src_project = os.path.abspath(project)
        # BRIEF layout: pNN-<slug>/{trace.json, project/}; replay the project/ subfolder then.
        if os.path.isdir(os.path.join(self.src_project, "project")) and not os.path.exists(os.path.join(self.src_project, "setup.py")):
            self.src_project = os.path.join(self.src_project, "project")
        self.mode = mode
        self.out = os.path.abspath(out)
        self.cells: dict[str, str] = {}
        self.kernel: Kernel | None = None
        self.file_memo: dict = {}
        self.files_state: dict = {}

    # -- setup

    def make_tmp(self):
        self.tmp = tempfile.mkdtemp(prefix="r34-")
        assert _is_own_tmp(self.tmp), self.tmp
        self.proj = os.path.join(self.tmp, "proj")
        self.cache = os.path.join(self.tmp, "cache")
        self.ipydir = os.path.join(self.tmp, "ipython")
        self.hookdir = os.path.join(self.tmp, "hook")
        self.callsdir = os.path.join(self.tmp, "calls")
        self.jupdir = os.path.join(self.tmp, "jupyter")
        self.kcallsdir = os.path.join(self.tmp, "kcalls")
        for d in (self.cache, self.ipydir, self.hookdir, self.callsdir, self.kcallsdir):
            os.makedirs(d)

        def ignore(d, names):
            return [n for n in names if n in EXCLUDE_DIRS or n.startswith(".cash")]

        shutil.copytree(self.src_project, self.proj, symlinks=True, ignore=ignore)
        shutil.copy(os.path.join(HERE, "r34digest.py"), self.hookdir)
        shutil.copy(os.path.join(HERE, "cs_stats.py"), self.hookdir)
        shutil.copy(os.path.join(HERE, "r34hook_site.py"), os.path.join(self.hookdir, "sitecustomize.py"))
        # The shared folder has no symlinks, so a project venv may live outside it:
        # $R34_VENV, then trace.json "venv", then PROJECT_DIR/.venv.
        venv = os.environ.get("R34_VENV") or self.trace.get("venv") or os.path.join(self.src_project, ".venv")
        if os.path.exists(_venv_python(venv)):
            self.venv = venv
            self.python = _venv_python(venv)
        else:
            self.venv = os.path.dirname(os.path.dirname(sys.executable)) if sys.prefix != sys.base_prefix else None
            self.python = sys.executable
        ks = os.path.join(self.jupdir, "kernels", "r34py")
        os.makedirs(ks)
        json.dump(
            {
                "argv": [self.python, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "r34",
                "language": "python",
            },
            open(os.path.join(ks, "kernel.json"), "w", encoding="utf-8"),
        )
        os.environ["JUPYTER_PATH"] = self.jupdir

    def cleanup(self):
        if self.kernel:
            self.kernel.stop()
        t = getattr(self, "tmp", None)
        if t and _is_own_tmp(t) and os.path.isdir(t):
            shutil.rmtree(t, ignore_errors=True)

    def env(self, kernel: bool = False, hook: bool = False, setup: bool = False) -> dict:
        e = {k: v for k, v in os.environ.items() if not k.startswith(("CASH_", "R34_", "PYTHON", "VIRTUAL_ENV", "JPY_"))}
        e.update(
            CASH_CACHE_DIR=self.cache,
            MPLBACKEND="Agg",
            PYTHONHASHSEED="0",
            PYTHONUNBUFFERED="1",
            IPYTHONDIR=self.ipydir,
            JUPYTER_PATH=self.jupdir,
        )
        if self.venv:
            e["VIRTUAL_ENV"] = self.venv
            e["PATH"] = os.path.dirname(_venv_python(self.venv)) + os.pathsep + e.get("PATH", "")
        else:
            e["PATH"] = os.path.dirname(self.python) + os.pathsep + e.get("PATH", "")
        if self.mode == "plain" or setup:
            e["CASH_DISABLE"] = "1"
        else:
            e["CASH_VERBOSE"] = "1"  # cash's per-call lines (cash.calls: HIT/MISS ...) on stderr
            # correctness set: CS_CASH_ENV="CASH_HITPATH_VERIFY=1 CASH_CWATCH_VERIFY=1" turns verify on (cash mode only)
            for kv in os.environ.get("CS_CASH_ENV", "").split():
                k, _, v = kv.partition("=")
                if k.startswith("CASH_"):
                    e[k] = v
        if (hook or kernel) and self.mode == "plain" and not setup:
            e["PYTHONPATH"] = self.hookdir
            e["R34_CALLS_DIR"] = self.kcallsdir if kernel else self.callsdir
        return e

    def emit(self, rec: dict):
        with open(self.out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    def check_python(self) -> list:
        need = ["ipykernel"] if any(s.get("op") in ("run", "restart") for s in self.trace["steps"]) else []
        if self.mode == "cash":
            need.append("cash")
        bad = []
        for mod in need:
            r = subprocess.run([self.python, "-c", f"import {mod}"], capture_output=True, text=True, env=self.env(setup=True))
            if r.returncode:
                bad.append(f"{mod} not importable by {self.python}: {r.stderr.strip()[-300:]}")
        return bad

    # -- steps

    def shell(self, cmd: str, hook: bool, setup: bool = False) -> dict:
        for f in os.listdir(self.callsdir):
            os.remove(os.path.join(self.callsdir, f))
        t0 = time.perf_counter()
        # Trace commands are bash; on Windows they need bash on PATH (Git for Windows), else cmd.exe runs them.
        if os.name == "nt":
            argv = [shutil.which("bash"), "-c", cmd] if shutil.which("bash") else cmd
            group = dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP, shell=not shutil.which("bash"))
        else:
            argv, group = ["bash", "-c", cmd], dict(start_new_session=True)
        p = subprocess.Popen(
            argv,
            cwd=self.proj,
            env=self.env(hook=hook, setup=setup),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **group,
        )
        timed_out = False
        try:
            out, err = p.communicate(timeout=STEP_TIMEOUT)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
                else:
                    os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                pass
            out, err = p.communicate()
        secs = time.perf_counter() - t0
        res = {"secs": secs, "rc": p.returncode, "stdout": out[:OUT_CAP], "stderr": err[-OUT_CAP:]}
        if timed_out:
            res["error"] = "TIMEOUT after %ds" % STEP_TIMEOUT
        elif p.returncode:
            lines = [ln for ln in err.strip().splitlines() if ln.strip()]
            # The exception line, not a trailing cash summary/notice line (CASH_SUMMARY, logging handlers).
            exc = [ln for ln in lines if re.match(r"^[A-Za-z_][\w.]*(Error|Exception|Exit|Interrupt|Warning)\b", ln)]
            tail = exc or [ln for ln in lines if not ln.startswith("cash")]
            res["error"] = f"exit {p.returncode}: " + (tail[-1][:500] if tail else "")
        calls, over = [], 0.0
        if hook and self.mode == "plain":
            for f in sorted(os.listdir(self.callsdir)):
                pid = f.split("-")[-1].split(".")[0]
                for line in open(os.path.join(self.callsdir, f), encoding="utf-8"):
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if "overhead" in r:
                        over += r["overhead"]
                    else:
                        r["pid"] = pid
                        calls.append(r)
        elif self.mode == "cash":
            calls = parse_cash_calls(err)
        res["calls"] = calls
        res["hook_overhead"] = round(over, 6)
        return res

    def ensure_kernel(self) -> dict | None:
        if self.kernel is None or not self.kernel.alive():
            if self.kernel:
                self.kernel.stop()
            self.kernel = Kernel(self)
            return self.kernel.start()
        return None

    def state_record(self, rec: dict):
        st = self.kernel.fetch_state() if self.kernel and self.kernel.alive() else None
        if isinstance(st, dict) and "v" in st:
            rec["values"], rec["codes"], rec["strs"] = st["v"], st["c"], st["s"]
            rec["digest_secs"] = st.get("secs")
        else:
            rec["values"] = None
            rec["state_error"] = str(st)[:500]

    def files_record(self, rec: dict):
        now = scan_files(self.proj, self.file_memo)
        rec["files"] = file_changes(self.files_state, now)
        self.files_state = now

    def run(self):
        open(self.out, "w", encoding="utf-8").close()
        self.make_tmp()
        try:
            self._run()
            if self.mode == "cash":
                b, f = dir_bytes(self.cache)
                self.emit({"i": -3, "op": "end", "cache_bytes": b, "cache_files": f, "wall": 0.0, "status": "ok"})
        finally:
            self.cleanup()

    def _run(self):
        tr = self.trace
        bad = self.check_python()
        head = {
            "i": -2,
            "op": "meta",
            "mode": self.mode,
            "project": tr.get("project"),
            "python": self.python,
            "tmp": self.tmp,
            "n_steps": len(tr["steps"]),
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": platform.node(),
            "platform": sys.platform,
            "problems": bad,
        }
        try:
            head["cash_commit"] = subprocess.check_output(
                [self.python, "-c", "import cash,sys; print(cash.__version__, cash.__file__)"], text=True, env=self.env(setup=True)
            ).strip()
            src = head["cash_commit"].split(" ", 1)[-1]
            head["cash_git"] = subprocess.run(
                ["git", "-C", os.path.dirname(src), "rev-parse", "HEAD"], capture_output=True, text=True
            ).stdout.strip()
        except Exception as e:  # noqa: BLE001
            head["cash_commit"] = "?" + repr(e)[:100]
        setup = []
        for cmd in tr.get("setup", []):
            r = self.shell(cmd, hook=False, setup=True)
            setup.append({"cmd": cmd, "secs": round(r["secs"], 3), "rc": r["rc"], "err": r.get("error")})
            if r["rc"]:
                head["problems"].append(f"setup failed: {cmd}: {r.get('error')} {r['stderr'][-500:]}")
        head["setup"] = setup
        self.files_state = scan_files(self.proj, self.file_memo)
        head["files0"] = self.files_state
        self.emit(head)
        if bad:
            print("cannot replay:", bad, file=sys.stderr)
            sys.exit(2)
        needs_kernel = any(s.get("op") in ("run", "restart") for s in tr["steps"])
        if needs_kernel:
            st = self.ensure_kernel()
            rec = {"i": -1, "op": "start", "wall": round(st["secs"], 4), "status": "ok", "notes": st["notes"]}
            self.state_record(rec)
            self.emit(rec)
        for i, step in enumerate(tr["steps"]):
            op = step.get("op")
            rec = {"i": i, "op": op}
            if op == "cell":
                self.cells[step["id"]] = step["src"]
                rec.update(cell=step["id"], wall=0.0, status="ok")
            elif op == "run":
                rec["cell"] = step["id"]
                rec.update(self.do_run(step["id"]))
            elif op == "restart":
                self.kernel.stop() if self.kernel else None
                self.kernel = None
                st = self.ensure_kernel()
                rec.update(wall=round(st["secs"], 4), status="ok", notes=st["notes"])
                self.state_record(rec)
            elif op == "write_file":
                t0 = time.perf_counter()
                p = os.path.join(self.proj, step["path"])
                os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
                with open(p, "w", encoding="utf-8", newline="") as fh:
                    fh.write(step["content"])
                rec.update(path=step["path"], wall=round(time.perf_counter() - t0, 4), status="ok")
            elif op in ("shell", "script"):
                r = self.shell(step["cmd"], hook=True)
                rec.update(
                    cmd=step["cmd"],
                    wall=round(r["secs"], 4),
                    wall_adj=round(r["secs"] - r["hook_overhead"], 4),
                    hook_overhead=r["hook_overhead"],
                    rc=r["rc"],
                    status="error" if r.get("error") else "ok",
                    error=r.get("error"),
                    out=normalize(r["stdout"], self.tmp),
                    stderr=normalize(r["stderr"], self.tmp)[-3000:],
                    calls=r["calls"],
                )
            else:
                rec.update(status="error", error=f"unknown op {op!r}", wall=0.0)
            self.files_record(rec)
            self.emit(rec)
            print(
                f"[{self.mode}] {i:4d} {op:10s} {str(step.get('id') or step.get('cmd') or step.get('path') or ''):30.30s}"
                f" {rec.get('wall', 0):8.2f}s {rec.get('status')} {(rec.get('error') or '')[:80]}",
                flush=True,
            )

    def do_run(self, cell_id: str) -> dict:
        src = self.cells.get(cell_id)
        if src is None:
            return {"status": "error", "error": f"trace error: cell {cell_id!r} run before it was defined", "wall": 0.0}
        rec = {"src": src}
        st = self.ensure_kernel()
        if st:
            rec["kernel_started"] = round(st["secs"], 3)
        r = self.kernel.execute(src)
        rec["wall"] = round(r["secs"], 4)
        rec["status"] = "error" if r["error"] else "ok"
        rec["error"] = normalize(r["error"], self.tmp) if r["error"] else None
        rec["out"] = normalize(r["stdout"], self.tmp)
        rec["stderr"] = normalize(r["stderr"], self.tmp)[-3000:]
        rec["disp"] = [{**d, "text": normalize(d.get("text"), self.tmp)} for d in r["disp"]]
        if r.get("cash_disp"):
            rec["cash_disp"] = r["cash_disp"]
        if r.get("need_restart"):
            # A cell that would not stop, or a dead kernel: start over like the user would.
            self.kernel.stop()
            self.kernel = None
            st = self.ensure_kernel()
            rec["forced_restart"] = round(st["secs"], 3)
            rec["stmts"] = None
        else:
            s = self.kernel.fetch_stmts()
            if self.mode == "plain":
                if isinstance(s, dict):
                    rec["calls"] = s.pop("calls", [])
                    rec["hook_overhead"] = round(s.pop("hook_overhead", 0.0), 6)
                    rec["wall_adj"] = round(rec["wall"] - rec["hook_overhead"], 4)
                rec["stmts"] = s
            else:
                rec["cash"] = s
                rec["calls"] = parse_cash_calls(r["stderr"])
                w = parse_warnings(r["stderr"])
                if w:
                    rec["warnings"] = w
        self.state_record(rec)
        return rec


def main():
    if len(sys.argv) != 5 or sys.argv[3] not in ("plain", "cash"):
        sys.exit(__doc__)
    trace, project, mode, out = sys.argv[1:]
    Replay(trace, project, mode, out).run()


if __name__ == "__main__":
    main()
