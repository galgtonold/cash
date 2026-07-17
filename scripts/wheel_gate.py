r"""CAS-190 wheel-gate harness: an automated, assertion-driven reproduction of
the manual gate rounds.

WHY THIS EXISTS
---------------
The fast test suite (`tests/test_notebook_integration/`) is structurally BLIND
to two whole classes of bug, proven four times this release (CAS-185, CAS-196,
CAS-202, the packaging P0):

  1. KERNEL-RESTART behaviour. `nb_runner` boots a fresh kernel per test and has
     NO restart method (`grep 'def restart' conftest.py` -> nothing). Every
     restart-path bug is invisible: CAS-196's restart re-fire and CAS-202's
     restart-retrain both shipped green.
  2. WHEEL-VENV install layout. The suite runs the EDITABLE dev install against
     C:\Python314; testers run a FRESH WHEEL VENV. `importlib.metadata` phantom
     file-dep probes (81 in a venv vs 0 in dev) only exist in the venv. Every
     install-layout bug is invisible. (The dev env has no `jupyter_server`
     installed at all -- so whatever the CAS-171 sweep drove, it was not a real
     Jupyter server.)

The manual gate rounds catch these (fresh wheel venv + real `jupyter server` +
`BlockingKernelClient` + real kernel restart) but take five human-like agents
~30 min. This harness automates that methodology so a developer or CI can run it
with one command.

WHAT IT DOES
------------
  1. Builds a wheel from the current tree (or accepts `--wheel <path>`).
  2. Creates a FRESH venv on a SHORT path (C:\Temp\wheelgate\venv -- MAX_PATH!)
     and installs `<wheel>[all]` + pandas numpy scikit-learn jupyter-server
     jupyter-client nbformat ipykernel. NEVER installs into C:\Python314.
  3. Registers a UNIQUE `wheelgate` kernelspec INTO the venv
     (`ipykernel install --sys-prefix`, never `--user`) so the kernel can only
     resolve to the venv interpreter -- and a guard cell asserts `sys.prefix`
     really is the venv (the install-layout is genuinely exercised).
  4. Drives a REAL `jupyter server` + `BlockingKernelClient` via the proven
     `wheel_gate_driver.py`, with a real kernel `restart` between run phases.
  5. Runs four assertion-driven scenarios. Each proves a ground-truth invariant
     with an EXTERNAL signal -- a counter written from INSIDE a function/cell to
     a file, read from OUTSIDE the kernel. Never a badge or a print, which are
     themselves restored on a cache hit and so cannot witness a silent re-run.
  6. Cleans up: quit + tree-kill the server (idempotent, rerun-safe).

THE SCENARIOS (each a separate assertion; RED = invariant violated = bug present)
  S1  restart survival of @cash.cache on the sklearn pipeline (CAS-202) -> RED
  S2  to_csv audit-log not re-fired by a downstream reader     (CAS-196) -> RED
  S3  same sklearn pipeline warm-re-runs WITHIN a session       (CAS-185) -> GREEN
  S4  a plain @cash.cache int fn survives a restart (control)             -> GREEN

The S1(RED)/S3(GREEN) contrast isolates the failure to "sklearn pipeline +
restart"; S4(GREEN) shows restart-survival works when no sklearn import poisons
the file-dep set. A harness that passed everything would be the exact CAS-190
blindness it is meant to cure -- so S1/S2 going RED is the proof of non-vacuity.

RUN IT
------
    python scripts/wheel_gate.py                 # build wheel, full run
    python scripts/wheel_gate.py --wheel dist/cash_lib-0.5.0b1-py3-none-any.whl
    python scripts/wheel_gate.py --scenarios S1,S4     # subset
    python scripts/wheel_gate.py --reuse-venv          # skip venv rebuild

Exit code 0 iff the observed RED/GREEN matrix matches the recorded baseline
(S1 RED, S2 RED, S3 GREEN, S4 GREEN). A mismatch (a green invariant regressed,
or a known bug got fixed and the baseline needs updating) exits 1. Takes roughly
6-12 min cold (wheel build + venv install dominate) and ~2-4 min with
--reuse-venv. Kept OUT of the default pytest collection because it is slow; see
tests/test_wheel_gate/ for the CI shim (skipped unless CASH_WHEEL_GATE=1).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DRIVER_SRC = Path(__file__).resolve().parent / "wheel_gate_driver.py"

# SHORT paths only: a venv under the deep repo path blows MAX_PATH (260) and
# yields bogus ModuleNotFoundErrors. C:\Temp\wheelgate keeps every path short.
GATE_ROOT = Path(os.environ.get("CASH_WHEELGATE_ROOT", r"C:\Temp\wheelgate"))
VENV_DIR = GATE_ROOT / "venv"
DIST_DIR = GATE_ROOT / "dist"
WORK_ROOT = GATE_ROOT / "work"

KERNEL_NAME = "wheelgate"  # unique -> cannot be shadowed by a system python3 spec
DEV_PYTHON = sys.executable  # used ONLY to build the wheel + create the venv


def _venv_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


# ---------------------------------------------------------------------------
# small shell helpers
# ---------------------------------------------------------------------------

def run(cmd, *, cwd=None, env=None, check=True, quiet=False) -> subprocess.CompletedProcess:
    cmd = [str(c) for c in cmd]
    if not quiet:
        print("  $", " ".join(cmd))
    cp = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True,
                        encoding="utf-8", errors="replace")
    if cp.returncode != 0 and check:
        print(cp.stdout[-3000:])
        print(cp.stderr[-3000:])
        raise SystemExit(f"command failed ({cp.returncode}): {' '.join(cmd)}")
    return cp


def _free_port(port: int) -> None:
    """Kill any process still LISTENING on *port* (a leaked prior driver)."""
    if os.name != "nt":
        return
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "tcp"],
                             capture_output=True, text=True).stdout
    except Exception:
        return
    pids = set()
    for line in out.splitlines():
        if f":{port} " in line and "LISTEN" in line.upper():
            pid = line.split()[-1]
            if pid.isdigit() and pid != "0":
                pids.add(pid)
    for pid in pids:
        subprocess.run(["taskkill", "/F", "/T", "/PID", pid], capture_output=True)


# ---------------------------------------------------------------------------
# phase 1: build wheel
# ---------------------------------------------------------------------------

def build_wheel() -> Path:
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[build] python -m build --wheel --outdir {DIST_DIR}")
    cp = run([DEV_PYTHON, "-m", "build", "--wheel", "--outdir", DIST_DIR, REPO_ROOT],
             check=False)
    if cp.returncode != 0:
        print("[build] isolated build failed; retrying --no-isolation")
        cp = run([DEV_PYTHON, "-m", "build", "--wheel", "--no-isolation",
                  "--outdir", DIST_DIR, REPO_ROOT], check=False)
    wheels = sorted(DIST_DIR.glob("cash_lib-*.whl"), key=lambda p: p.stat().st_mtime)
    if cp.returncode != 0 or not wheels:
        raise SystemExit("wheel build failed and no prebuilt wheel in DIST_DIR")
    wheel = wheels[-1]
    print(f"[build] wheel: {wheel}")
    return wheel


# ---------------------------------------------------------------------------
# phase 2/3: venv + kernelspec
# ---------------------------------------------------------------------------

def _venv_provisioned(py: Path) -> bool:
    if not py.exists():
        return False
    probe = "import cash, jupyter_server, sklearn, pandas, numpy, ipykernel, nbformat"
    cp = run([py, "-c", probe], check=False, quiet=True)
    if cp.returncode != 0:
        return False
    spec = VENV_DIR / "share" / "jupyter" / "kernels" / KERNEL_NAME / "kernel.json"
    return spec.exists()


def setup_venv(wheel: Path, *, reuse: bool) -> Path:
    GATE_ROOT.mkdir(parents=True, exist_ok=True)
    py = _venv_python()
    if reuse and _venv_provisioned(py):
        print(f"[venv] reusing provisioned venv at {VENV_DIR}")
        return py
    if VENV_DIR.exists():
        print(f"[venv] removing stale venv {VENV_DIR}")
        shutil.rmtree(VENV_DIR, ignore_errors=True)
    print(f"[venv] creating fresh venv at {VENV_DIR}")
    run([DEV_PYTHON, "-m", "venv", VENV_DIR])
    py = _venv_python()
    run([py, "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
    # extras on a local wheel path: pip supports "<path>.whl[extra]"
    print("[venv] installing wheel[all] + gate deps (this is the slow step)")
    run([py, "-m", "pip", "install", f"{wheel}[all]",
         "pandas", "numpy", "scikit-learn",
         "jupyter-server", "jupyter-client", "nbformat", "ipykernel",
         "--quiet"])
    # Register a UNIQUE kernelspec INTO the venv (--sys-prefix, NEVER --user):
    # guarantees the kernel resolves to the venv interpreter.
    run([py, "-m", "ipykernel", "install", "--sys-prefix",
         "--name", KERNEL_NAME, "--display-name", KERNEL_NAME])
    if not _venv_provisioned(py):
        raise SystemExit("venv provisioning check failed after install")
    print("[venv] provisioned OK")
    return py


# ---------------------------------------------------------------------------
# driver client (file-based inbox/outbox protocol)
# ---------------------------------------------------------------------------

class Driver:
    def __init__(self, py: Path, work: Path, port: int):
        self.py = py
        self.work = work
        self.port = port
        self.inbox = work / "inbox"
        self.outbox = work / "outbox"
        self.proc: subprocess.Popen | None = None
        self._seq = 0

    def start(self, ready_timeout: float = 300.0) -> None:
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.outbox.mkdir(parents=True, exist_ok=True)
        _free_port(self.port)
        env = dict(os.environ)
        env.update(
            CASH_WORK_DIR=str(self.work),
            CASH_DRIVER_PORT=str(self.port),
            CASH_DRIVER_TOKEN="wheelgate",
            CASH_KERNEL_NAME=KERNEL_NAME,
            PYTHONUTF8="1",
            PYTHONIOENCODING="utf-8",
            JUPYTER_PLATFORM_DIRS="1",
        )
        out = open(self.work / "driver.out", "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [str(self.py), str(DRIVER_SRC)],
            cwd=str(self.work), env=env, stdout=out, stderr=subprocess.STDOUT)
        # readiness: poll driver.out for the DRIVER READY banner
        t0 = time.time()
        log = self.work / "driver.out"
        while time.time() - t0 < ready_timeout:
            if self.proc.poll() is not None:
                raise SystemExit(f"driver died during boot; see {log}\n"
                                 f"{log.read_text(encoding='utf-8', errors='replace')[-2000:]}")
            try:
                if "DRIVER READY" in log.read_text(encoding="utf-8", errors="replace"):
                    return
            except OSError:
                pass
            time.sleep(0.3)
        raise SystemExit(f"driver never signalled READY within {ready_timeout}s; see {log}")

    def _send(self, cmd: dict, timeout: float) -> dict:
        self._seq += 1
        name = f"{self._seq:06d}_{uuid.uuid4().hex[:6]}.json"
        (self.inbox / name).write_text(json.dumps(cmd), encoding="utf-8")
        outp = self.outbox / name
        t0 = time.time()
        while time.time() - t0 < timeout:
            if outp.exists():
                time.sleep(0.05)
                try:
                    return json.loads(outp.read_text(encoding="utf-8"))
                except Exception:
                    time.sleep(0.1)
                    continue
            if self.proc and self.proc.poll() is not None:
                raise SystemExit("driver died mid-command; see driver.out")
            time.sleep(0.1)
        raise SystemExit(f"timeout ({timeout}s) waiting for {cmd}")

    def set(self, index: int, source: str) -> dict:
        return self._send({"action": "set", "index": index, "source": source}, 60)

    def run(self, index: int, timeout: float = 900.0) -> dict:
        r = self._send({"action": "run", "index": index, "timeout": timeout}, timeout + 60)
        if not r.get("ok"):
            raise SystemExit(f"run(cell {index}) failed: {r.get('msg')}")
        return r

    def restart(self) -> dict:
        return self._send({"action": "restart"}, 180)

    def quit(self) -> None:
        try:
            self._send({"action": "quit"}, 30)
        except Exception:
            pass

    def close(self) -> None:
        self.quit()
        if self.proc:
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                                   capture_output=True)
                else:
                    self.proc.terminate()
            except Exception:
                pass
        _free_port(self.port)


# ---------------------------------------------------------------------------
# scenario scaffolding
# ---------------------------------------------------------------------------

@dataclass
class Result:
    name: str
    title: str
    expected: str            # "RED" or "GREEN"
    status: str = "ERROR"    # observed: "RED" / "GREEN" / "ERROR"
    detail: str = ""
    evidence: dict = field(default_factory=dict)

    @property
    def matches(self) -> bool:
        return self.status == self.expected


def _fresh_work(name: str) -> Path:
    w = WORK_ROOT / name
    if w.exists():
        shutil.rmtree(w, ignore_errors=True)
    w.mkdir(parents=True, exist_ok=True)
    return w


def _guard_venv(drv: Driver, venv_dir: Path) -> None:
    """Prove the kernel really runs in the venv (install-layout is exercised)."""
    drv.set(0, "import sys, cash\n"
               "print('GUARD_PREFIX', sys.prefix)\n"
               "print('GUARD_CASH', cash.__file__)")
    out = drv.run(0).get("out", "")
    vp = str(venv_dir).lower()
    if vp not in out.lower():
        raise SystemExit(f"GUARD FAILED: kernel not in venv.\n{out}")


def _count(path: Path) -> int:
    """External signal: number of characters the in-kernel counter wrote."""
    try:
        return len(path.read_bytes())
    except OSError:
        return 0


def _lines(path: Path) -> int:
    try:
        txt = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    return len([ln for ln in txt.splitlines() if ln.strip()])


# ----- shared sklearn-pipeline cells (S1 + S3) -----------------------------

_CELL_ON = 'import cash\n%cash_on\nprint("cash on")'

def _pipeline_cells(counter_file: str):
    """make_classification -> DataFrame -> train_test_split -> @cash.cache train."""
    data = (
        "import pandas as pd, numpy as np\n"
        "from sklearn.datasets import make_classification\n"
        "X_arr, y_arr = make_classification(n_samples=8000, n_features=14,\n"
        "    n_informative=8, n_redundant=3, weights=[0.75, 0.25], random_state=42)\n"
        "cols = [f'f{i}' for i in range(X_arr.shape[1])]\n"
        "df = pd.DataFrame(X_arr, columns=cols)\n"
        "df['churned'] = y_arr\n"
        "print('DATA', df.shape)"
    )
    split = (
        "from sklearn.model_selection import train_test_split\n"
        "X = df[cols].fillna(0)\n"
        "y = df['churned']\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.2, random_state=42)\n"
        "print('SPLIT', X_train.shape)"
    )
    train = (
        "import cash, time\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "@cash.cache\n"
        "def train_model(X_train, y_train, n_estimators):\n"
        f"    open('{counter_file}', 'a').write('T')   # external counter: proves body ran\n"
        "    m = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=1)\n"
        "    m.fit(X_train, y_train)\n"
        "    return m\n"
        "_t = time.perf_counter()\n"
        "model = train_model(X_train, y_train, n_estimators=120)\n"
        "print('TRAIN trees', model.n_estimators, 'wall', round(time.perf_counter()-_t, 3))"
    )
    # NOTE: no downstream "evaluate" cell. A cell consuming `model`
    # (model.predict(...)) triggers cash to re-run the train cell within the
    # SAME session (observed cold_steps [0,0,0,1,2] -- a second fit fired when
    # eval ran), which pollutes the restart signal. The counter is measured at
    # the train cell so it witnesses only the restart-survival invariant.
    return [_CELL_ON, data, split, train]


# ---------------------------------------------------------------------------
# S1 -- restart survival of @cash.cache on the sklearn pipeline (CAS-202)
# ---------------------------------------------------------------------------

def scenario_s1(py: Path, port: int) -> Result:
    r = Result("S1", "restart survival of @cash.cache sklearn pipeline (CAS-202)", "RED")
    work = _fresh_work("s1")
    counter = work / "calls.log"
    cells = _pipeline_cells("calls.log")
    drv = Driver(py, work, port)
    try:
        drv.start()
        _guard_venv(drv, VENV_DIR)          # cell 0 becomes the guard...
        for i, src in enumerate(cells):     # ...then overwrite 0..4 with the pipeline
            drv.set(i, src)
        # cold run-all
        cold_steps = []
        for i in range(len(cells)):
            drv.run(i)
            cold_steps.append(_count(counter))
        r.evidence["cold_steps"] = cold_steps
        n_cold = _count(counter)
        # restart, then restart-and-run-all (the flagship "restart in seconds" flow)
        drv.restart()
        for i in range(len(cells)):
            drv.run(i)
        n_warm = _count(counter)
        r.evidence.update(fit_calls_cold=n_cold, fit_calls_after_restart=n_warm)
        if n_cold != 1:
            r.status, r.detail = "ERROR", f"expected 1 fit on cold run, got {n_cold}"
        elif n_warm == n_cold:
            r.status = "GREEN"
            r.detail = f"model restored after restart (fit ran {n_warm}x total)"
        else:
            r.status = "RED"
            r.detail = (f"@cash.cache re-trained after restart: fit body ran "
                        f"{n_warm}x (cold {n_cold} + retrain {n_warm - n_cold})")
    except SystemExit as e:
        r.status, r.detail = "ERROR", str(e)
    except Exception:
        r.status, r.detail = "ERROR", traceback.format_exc()
    finally:
        drv.close()
    return r


# ---------------------------------------------------------------------------
# S3 -- SINGLE-CELL self-contained sklearn @cash.cache survives a restart.
#
# CAS-202's own diagnostic clue: "A self-contained SINGLE-CELL version survives
# restart fine -- only the documented multi-cell load/split/train layout
# triggers it." So the SAME sklearn work, packed into one cell, is the control
# that isolates S1's failure to multi-cell provenance. This is the GREEN that
# proves the harness distinguishes pass from fail on an sklearn pipeline (not
# just the trivial int fn of S4), and that it isn't blanket-failing everything.
# ---------------------------------------------------------------------------

_S3_CELL = (
    "import cash, time\n"
    "import pandas as pd, numpy as np\n"
    "from sklearn.datasets import make_classification\n"
    "from sklearn.ensemble import RandomForestClassifier\n"
    "from sklearn.model_selection import train_test_split\n"
    "@cash.cache\n"
    "def train_all():\n"
    "    open('calls.log', 'a').write('T')   # external counter: proves body ran\n"
    "    X, y = make_classification(n_samples=8000, n_features=14, n_informative=8,\n"
    "        n_redundant=3, weights=[0.75, 0.25], random_state=42)\n"
    "    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])\n"
    "    df['churned'] = y\n"
    "    cols = [c for c in df.columns if c != 'churned']\n"
    "    Xtr, Xte, ytr, yte = train_test_split(df[cols], df['churned'],\n"
    "        test_size=0.2, random_state=42)\n"
    "    m = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=1)\n"
    "    m.fit(Xtr, ytr)\n"
    "    return m\n"
    "_t = time.perf_counter()\n"
    "model = train_all()\n"
    "print('TRAIN trees', model.n_estimators, 'wall', round(time.perf_counter()-_t, 3))"
)


def scenario_s3(py: Path, port: int) -> Result:
    r = Result("S3", "single-cell sklearn @cash.cache survives a restart (CAS-202 control)", "GREEN")
    work = _fresh_work("s3")
    counter = work / "calls.log"
    cells = [_CELL_ON, _S3_CELL]
    drv = Driver(py, work, port)
    try:
        drv.start()
        _guard_venv(drv, VENV_DIR)
        for i, src in enumerate(cells):
            drv.set(i, src)
        for i in range(len(cells)):
            drv.run(i)
        n_cold = _count(counter)
        drv.restart()
        for i in range(len(cells)):
            drv.run(i)
        n_warm = _count(counter)
        r.evidence = {"fit_calls_cold": n_cold, "fit_calls_after_restart": n_warm}
        if n_cold != 1:
            r.status, r.detail = "ERROR", f"expected 1 fit on cold run, got {n_cold}"
        elif n_warm == n_cold:
            r.status = "GREEN"
            r.detail = f"single-cell model restored after restart (fit ran {n_warm}x total)"
        else:
            r.status = "RED"
            r.detail = f"single-cell sklearn re-trained after restart: fit ran {n_warm}x"
    except SystemExit as e:
        r.status, r.detail = "ERROR", str(e)
    except Exception:
        r.status, r.detail = "ERROR", traceback.format_exc()
    finally:
        drv.close()
    return r


# ---------------------------------------------------------------------------
# S4 -- a plain @cash.cache int fn survives a restart (control) -> GREEN
# ---------------------------------------------------------------------------

def scenario_s4(py: Path, port: int) -> Result:
    r = Result("S4", "plain @cash.cache int fn survives a restart (control)", "GREEN")
    work = _fresh_work("s4")
    counter = work / "s4calls.log"
    cell1 = (
        "import cash, time\n"
        "@cash.cache\n"
        "def slow_square(n):\n"
        "    open('s4calls.log', 'a').write('S')   # external counter\n"
        "    s = 0\n"
        "    for i in range(n):\n"
        "        s += i * i\n"
        "    return s\n"
        "_t = time.perf_counter()\n"
        "r = slow_square(6_000_000)\n"
        "print('S4 result', r, 'wall', round(time.perf_counter()-_t, 3))"
    )
    cells = [_CELL_ON, cell1]
    drv = Driver(py, work, port)
    try:
        drv.start()
        _guard_venv(drv, VENV_DIR)
        for i, src in enumerate(cells):
            drv.set(i, src)
        for i in range(len(cells)):
            drv.run(i)
        n_cold = _count(counter)
        drv.restart()
        for i in range(len(cells)):
            drv.run(i)
        n_warm = _count(counter)
        r.evidence = {"calls_cold": n_cold, "calls_after_restart": n_warm}
        if n_cold != 1:
            r.status, r.detail = "ERROR", f"expected 1 call on cold run, got {n_cold}"
        elif n_warm == n_cold:
            r.status = "GREEN"
            r.detail = f"restored after restart (body ran {n_warm}x total)"
        else:
            r.status = "RED"
            r.detail = f"plain @cash.cache re-ran after restart: body ran {n_warm}x"
    except SystemExit as e:
        r.status, r.detail = "ERROR", str(e)
    except Exception:
        r.status, r.detail = "ERROR", traceback.format_exc()
    finally:
        drv.close()
    return r


# ---------------------------------------------------------------------------
# S2 -- to_csv audit-log not re-fired by a downstream reader (CAS-196) -> RED
# ---------------------------------------------------------------------------

def _write_etl_data(work: Path, n: int = 400_000) -> None:
    """Deterministic retail-ETL CSVs (a shrunk gen_data.py) for the S2 pipeline."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(20260717)
    n_prod, cats, n_months = 800, ["Electronics", "Grocery", "Apparel", "Home", "Toys"], 24
    pid_dim = np.arange(1, n_prod + 1)
    cat_idx = rng.integers(0, len(cats), size=n_prod)
    products = pd.DataFrame({
        "product_id": pid_dim,
        "category": [cats[i] for i in cat_idx],
        "base_price": np.round(rng.uniform(5, 200, size=n_prod), 2),
    })
    products.to_csv(work / "products.csv", index=False)
    pid = rng.integers(1, n_prod + 1, size=n)
    month = rng.integers(0, n_months, size=n)
    year = 2024 + (month // 12)
    mon = (month % 12) + 1
    day = rng.integers(1, 29, size=n)
    order_date = pd.to_datetime(dict(year=year, month=mon, day=day))
    qty = rng.integers(1, 11, size=n)
    base = products.set_index("product_id").loc[pid, "base_price"].to_numpy()
    unit_price = np.round(base * rng.uniform(0.9, 1.3, size=n), 2)
    cat_of_pid = products.set_index("product_id").loc[pid, "category"].to_numpy()
    erode = np.zeros(n)
    erode[cat_of_pid == "Electronics"] = 0.010
    erode[cat_of_pid == "Toys"] = 0.008
    erode[cat_of_pid == "Grocery"] = 0.003
    cost_frac = np.clip(0.60 + erode * month + rng.normal(0, 0.02, size=n), 0.3, 0.98)
    unit_cost = np.round(unit_price * cost_frac, 2)
    pd.DataFrame({
        "order_id": np.arange(1, n + 1),
        "product_id": pid,
        "order_date": order_date.dt.strftime("%Y-%m-%d"),
        "quantity": qty,
        "unit_price": unit_price,
        "unit_cost": unit_cost,
    }).to_csv(work / "sales.csv", index=False)


# Cells mirror the round-7 P1 CAS-196 repro (nb.py). The DOWNSTREAM READER (cell
# 10) depends on `receipt`, which the AUDIT WRITER (cell 8) produces alongside
# its non-idempotent `to_csv(..., mode='a')` append -- so reconstructing the
# reader re-fires the append.
_S2_CELLS = [
    'import cash\n%cash_on\nprint("cash on")',                                       # 0
    "import pandas as pd\nimport numpy as np\nprint('imports ok')",                  # 1
    "sales = pd.read_csv('sales.csv')\nprint('sales', sales.shape)",                 # 2
    "products = pd.read_csv('products.csv')\nprint('products', products.shape)",     # 3
    ("sales['order_date'] = pd.to_datetime(sales['order_date'])\n"
     "sales['revenue'] = sales['quantity'] * sales['unit_price']\n"
     "sales['cost'] = sales['quantity'] * sales['unit_cost']\n"
     "sales['month'] = sales['order_date'].dt.to_period('M').astype(str)\n"
     "print('clean rev_sum', round(float(sales['revenue'].sum()), 2))"),            # 4
    ("merged = sales.merge(products[['product_id','category']], on='product_id', how='left')\n"
     "print('merged', merged.shape)"),                                              # 5
    ("agg = merged.groupby(['category','month'], as_index=False).agg("
     "revenue=('revenue','sum'), cost=('cost','sum'))\n"
     "agg['margin'] = (agg['revenue'] - agg['cost']) / agg['revenue']\n"
     "print('agg', agg.shape)"),                                                    # 6
    ("slopes = {}\n"
     "for cat, g in agg.groupby('category'):\n"
     "    g = g.sort_values('month'); x = np.arange(len(g)); yv = g['margin'].to_numpy()\n"
     "    slopes[cat] = float(np.polyfit(x, yv, 1)[0])\n"
     "worst = min(slopes, key=slopes.get)\n"
     "print('FASTEST_DECLINING', worst, round(slopes[worst], 6))"),                 # 7
    ("audit_row = pd.DataFrame([{'answer': worst, 'slope': round(slopes[worst], 6)}])\n"
     "audit_row.to_csv('audit.log', mode='a', header=False, index=False)\n"
     "receipt = 'audit-appended'\n"
     "print('AUDIT_APPENDED', worst)"),                                             # 8  WRITER
    ("agg.to_csv('summary.csv', index=False)\nprint('SUMMARY_WRITTEN', agg.shape)"), # 9
    ("print('READER worst=', worst, 'receipt=', receipt, 'n_slopes=', len(slopes))\n"
     "reader_out = f'{worst}:{round(slopes[worst],6)}'\n"
     "print('READER_OUT', reader_out)"),                                            # 10 READER
    ("sales_consumer = int(sales['quantity'].sum())\nprint('SALES_CONSUMER', sales_consumer)"),  # 11
]
_S2_READER = 10


def _s2_baseline_lines(py: Path, port: int) -> int:
    """Ground-truth: run the SAME pipeline with %cash_off; count audit.log lines.

    Mirrors the gate's cmd_off_write vs cmd_on_write comparison. A plain kernel
    executes the writer cell exactly once -> exactly one appended line.
    """
    work = _fresh_work("s2_off")
    _write_etl_data(work)
    (work / "audit.log").unlink(missing_ok=True)
    cells = list(_S2_CELLS)
    cells[0] = 'import cash\n%cash_off\nprint("cash off")'
    drv = Driver(py, work, port)
    try:
        drv.start()
        for i, src in enumerate(cells):
            drv.set(i, src)
        for i in range(len(cells)):
            drv.run(i)
        return _lines(work / "audit.log")
    finally:
        drv.close()


def scenario_s2(py: Path, port: int) -> Result:
    r = Result("S2", "to_csv audit-log not re-fired during reconstruction (CAS-196)", "RED")
    try:
        baseline = _s2_baseline_lines(py, port)      # %cash_off ground truth
    except SystemExit as e:
        r.status, r.detail = "ERROR", f"baseline failed: {e}"
        return r

    work = _fresh_work("s2")
    audit = work / "audit.log"
    _write_etl_data(work)
    audit.unlink(missing_ok=True)
    drv = Driver(py, work, port)
    try:
        drv.start()
        _guard_venv(drv, VENV_DIR)               # guard reuses cell 0 slot...
        for i, src in enumerate(_S2_CELLS):      # ...then S2 overwrites with %cash_on
            drv.set(i, src)
        for i in range(len(_S2_CELLS)):
            drv.run(i)
        n_runall = _lines(audit)
        # restart, re-enable cash, then run ONLY the downstream reader. The reader
        # needs `receipt` (produced by the writer cell) -> cash reconstructs
        # upstream. A correct system must NOT re-fire the non-idempotent append.
        drv.restart()
        drv.run(0)                               # %cash_on
        reader_out = drv.run(_S2_READER).get("out", "")
        n_reader = _lines(audit)
        r.evidence = {
            "baseline_off_lines": baseline,
            "on_runall_lines": n_runall,
            "on_after_reader_lines": n_reader,
            "loud_refusal": "UpstreamStateError" in reader_out,
        }
        if baseline != 1:
            r.status, r.detail = "ERROR", f"%cash_off baseline should be 1 line, got {baseline}"
        elif n_runall == baseline and n_reader == baseline:
            r.status = "GREEN"
            r.detail = f"audit.log byte-stable vs no-cash ({baseline} line) through run-all + reader"
        elif "UpstreamStateError" in reader_out and n_runall == baseline:
            r.status = "RED"
            r.detail = "downstream reader raised UpstreamStateError reconstructing the writer"
        else:
            r.status = "RED"
            r.detail = (f"cash re-fired the non-idempotent to_csv append: no-cash={baseline} "
                        f"line vs cash run-all={n_runall}, after-reader={n_reader} lines")
    except SystemExit as e:
        r.status, r.detail = "ERROR", str(e)
    except Exception:
        r.status, r.detail = "ERROR", traceback.format_exc()
    finally:
        drv.close()
    return r


SCENARIOS = {
    "S1": scenario_s1,
    "S2": scenario_s2,
    "S3": scenario_s3,
    "S4": scenario_s4,
}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="CAS-190 wheel-gate harness")
    ap.add_argument("--wheel", help="use a prebuilt wheel instead of building")
    ap.add_argument("--reuse-venv", action="store_true",
                    help="reuse an already-provisioned venv (skip rebuild/install)")
    ap.add_argument("--scenarios", default="S1,S2,S3,S4",
                    help="comma list of scenarios to run (default all)")
    ap.add_argument("--port-base", type=int, default=8921)
    args = ap.parse_args()

    names = [s.strip().upper() for s in args.scenarios.split(",") if s.strip()]
    for n in names:
        if n not in SCENARIOS:
            raise SystemExit(f"unknown scenario {n}; choose from {list(SCENARIOS)}")

    t_start = time.time()
    print("=" * 78)
    print("CAS-190 WHEEL-GATE HARNESS")
    print("=" * 78)

    wheel = Path(args.wheel).resolve() if args.wheel else build_wheel()
    if not wheel.exists():
        raise SystemExit(f"wheel not found: {wheel}")
    py = setup_venv(wheel, reuse=args.reuse_venv)

    results: list[Result] = []
    for idx, name in enumerate(names):
        port = args.port_base + idx
        print("\n" + "-" * 78)
        print(f"[{name}] {SCENARIOS[name].__doc__ or ''}".splitlines()[0])
        print(f"[{name}] port {port}")
        res = SCENARIOS[name](py, port)
        tag = "OK " if res.matches else "!! "
        print(f"[{name}] -> {res.status} (expected {res.expected}) {tag}{res.detail}")
        print(f"[{name}] evidence: {res.evidence}")
        results.append(res)

    # summary
    print("\n" + "=" * 78)
    print("SUMMARY (RED = invariant violated = bug present)")
    print("=" * 78)
    print(f"{'id':<4}{'status':<8}{'expected':<10}{'match':<7}title")
    print("-" * 78)
    for r in results:
        print(f"{r.name:<4}{r.status:<8}{r.expected:<10}"
              f"{('yes' if r.matches else 'NO'):<7}{r.title}")
    for r in results:
        print(f"\n[{r.name}] {r.detail}\n      evidence={r.evidence}")

    mism = [r for r in results if not r.matches]
    dur = round(time.time() - t_start, 1)
    print("\n" + "=" * 78)
    reds = [r.name for r in results if r.status == "RED"]
    greens = [r.name for r in results if r.status == "GREEN"]
    print(f"RED (bug reproduced): {reds or 'none'}")
    print(f"GREEN (invariant held): {greens or 'none'}")
    print(f"elapsed {dur}s")
    if mism:
        print(f"BASELINE MISMATCH on {[r.name for r in mism]} -- "
              "a green invariant regressed OR a known bug was fixed (update baseline).")
        return 1
    print("baseline matched: known-open bugs still RED, controls still GREEN.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
