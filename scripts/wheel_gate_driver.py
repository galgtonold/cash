"""Persistent-kernel notebook driver, backed by a REAL Jupyter server.

This is the CAS-190 wheel-gate driver: a parametrized copy of the proven
`C:/Temp/cashut/driver_reference.py` (the harness the manual gate rounds used).
cash's notebook mode needs a live Jupyter server (it maps kernel_id -> notebook
path via the server's /api/sessions), so plain nbclient can't drive it. This
spins up `jupyter server`, opens a session bound to work.ipynb, and talks ZMQ to
that server's kernel -- i.e. exactly what JupyterLab does, minus the browser.

Differences from driver_reference.py (all env-driven, behaviour identical when
the env is unset):
  * CASH_WORK_DIR   -- the work dir (inbox/outbox/runtime/work.ipynb live here).
                       Defaults to this file's own directory.
  * CASH_KERNEL_NAME-- the kernelspec name to bind the session to. Defaults to
                       "python3". The gate registers a UNIQUE `wheelgate`
                       kernelspec into the venv (`ipykernel install --sys-prefix`)
                       and drives with that name, so a stray system/user
                       `python3` kernelspec can never shadow the venv interpreter
                       (that shadow would silently defeat the whole
                       install-layout test -- the CAS-190 point).
  * CASH_DRIVER_PORT / CASH_DRIVER_TOKEN -- server port + token.

Run this WITH THE VENV PYTHON so `sys.executable` (used to spawn the server) is
the venv interpreter and the kernel resolves to venv site-packages.

Protocol: JSON files in inbox/ -> results in outbox/.
  {"action":"set","index":N,"source":"..."}
  {"action":"run","index":N}
  {"action":"restart"}
  {"action":"delete","index":N}
  {"action":"quit"}
"""
import json
import os
import queue
import subprocess
import sys
import time
import traceback
import urllib.request

# The kernel emits UTF-8 (cash badges contain checkmark/lightning emoji). This
# driver, and any tester that prints captured cell output, run on Windows where
# the default stdout is cp1252 and re-printing those bytes raises
# UnicodeEncodeError -- it silently corrupted testers' %cash_on measurements (a
# phantom "no cache" reading). Force UTF-8 so captured emoji never crash the
# harness (CAS-192).
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import nbformat
from jupyter_client import BlockingKernelClient

WORK = os.environ.get("CASH_WORK_DIR") or os.path.dirname(os.path.abspath(__file__))
WORK = os.path.abspath(WORK)
NB_NAME = "work.ipynb"
NB_PATH = os.path.join(WORK, NB_NAME)
INBOX = os.path.join(WORK, "inbox")
OUTBOX = os.path.join(WORK, "outbox")
RUNTIME = os.path.join(WORK, "runtime")
PORT = int(os.environ.get("CASH_DRIVER_PORT", "8899"))
TOKEN = os.environ.get("CASH_DRIVER_TOKEN", "cashtest")
KERNEL_NAME = os.environ.get("CASH_KERNEL_NAME", "python3")
BASE = f"http://127.0.0.1:{PORT}"
for d in (INBOX, OUTBOX, RUNTIME):
    os.makedirs(d, exist_ok=True)

if os.path.exists(NB_PATH):
    nb = nbformat.read(NB_PATH, as_version=4)
else:
    nb = nbformat.v4.new_notebook()
    nb.metadata.kernelspec = {"name": KERNEL_NAME, "display_name": KERNEL_NAME,
                              "language": "python"}
    nbformat.write(nb, NB_PATH)


def save():
    nbformat.write(nb, NB_PATH)


def api(path, method="GET", body=None):
    req = urllib.request.Request(
        f"{BASE}/api/{path}", method=method,
        data=json.dumps(body).encode() if body else None,
        headers={"Authorization": f"token {TOKEN}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read()
    return json.loads(raw) if raw else None


env = dict(os.environ, JUPYTER_RUNTIME_DIR=RUNTIME)
server = subprocess.Popen(
    [sys.executable, "-m", "jupyter", "server", "--no-browser",
     f"--port={PORT}", f"--IdentityProvider.token={TOKEN}",
     f"--ServerApp.root_dir={WORK}", "--ServerApp.disable_check_xsrf=True"],
    env=env, stdout=open(os.path.join(WORK, "server.log"), "w"),
    stderr=subprocess.STDOUT)


def _kill_server_tree():
    """Force-kill the jupyter server AND the kernel subprocesses it spawned.

    ``server.terminate()`` alone leaks the kernels on Windows. Registered with
    atexit so the server dies even if the driver is killed WITHOUT a ``quit``
    command -- the failure mode that actually leaked in the gate rounds.
    """
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(server.pid)],
                           capture_output=True)
        else:
            server.terminate()
    except Exception:
        pass


import atexit
atexit.register(_kill_server_tree)

for _ in range(240):
    try:
        api("status")
        break
    except Exception:
        time.sleep(0.5)
else:
    raise SystemExit("jupyter server never came up")

kc = None
KERNEL_ID = None
SESSION_ID = None

# A kernel that dies during startup is a HARNESS failure, not a verdict about
# cash -- and it happens: two consecutive full-gate runs each lost a different
# scenario to "Kernel died before replying to kernel_info", while every
# scenario passed when run alone. Booting six kernels at once, each paying a
# multi-second ``import cash``, is simply enough contention to lose one.
#
# Retrying the whole session is the honest response: the gate exists to decide
# whether a cash invariant held, and reporting ERROR because a kernel failed to
# start answers a different question. Bounded, and every attempt is printed, so
# a kernel that dies REPEATEDLY still surfaces instead of being retried away.
BOOT_ATTEMPTS = 3


# Fault injection, so the retry above can be OBSERVED recovering rather than
# assumed to work. A boot death is rare and load-dependent, so waiting for a
# real one to prove the path is not a plan. Set CASH_DRIVER_FAIL_BOOTS=N to
# make the first N attempts raise.
_FAIL_BOOTS = int(os.environ.get("CASH_DRIVER_FAIL_BOOTS", "0"))
_boots_attempted = 0


def _start_session():
    global KERNEL_ID, SESSION_ID, kc, _boots_attempted
    _boots_attempted += 1
    if _boots_attempted <= _FAIL_BOOTS:
        raise RuntimeError(
            f"injected boot failure {_boots_attempted}/{_FAIL_BOOTS} "
            "(CASH_DRIVER_FAIL_BOOTS)")
    session = api("sessions", "POST", {
        "path": NB_NAME, "type": "notebook", "name": NB_NAME,
        "kernel": {"name": KERNEL_NAME}})
    KERNEL_ID = session["kernel"]["id"]
    SESSION_ID = session["id"]
    print("session", SESSION_ID, "kernel", KERNEL_ID,
          "kernelspec", KERNEL_NAME, flush=True)

    cf = os.path.join(RUNTIME, f"kernel-{KERNEL_ID}.json")
    for _ in range(240):
        if os.path.exists(cf):
            break
        time.sleep(0.25)
    else:
        raise RuntimeError("no connection file " + cf)
    kc = BlockingKernelClient()
    kc.load_connection_file(cf)
    kc.start_channels()
    kc.wait_for_ready(timeout=120)


def _discard_dead_session():
    """Best effort: stop channels and delete the session so the retry is clean.

    A half-started kernel left registered would keep its ports and its entry in
    the server, and the next attempt would be racing it.
    """
    global kc
    if kc is not None:
        try:
            kc.stop_channels()
        except Exception:
            pass
        kc = None
    if SESSION_ID:
        try:
            api(f"sessions/{SESSION_ID}", "DELETE")
        except Exception:
            pass


def connect():
    for attempt in range(1, BOOT_ATTEMPTS + 1):
        try:
            _start_session()
            return
        except Exception as exc:
            print(f"[boot] attempt {attempt}/{BOOT_ATTEMPTS} failed: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            _discard_dead_session()
            if attempt == BOOT_ATTEMPTS:
                raise SystemExit(
                    f"kernel failed to start {BOOT_ATTEMPTS} times: {exc}")
            time.sleep(2.0 * attempt)


connect()


def execute(cell, timeout=3600):
    cell["outputs"] = []
    cell["execution_count"] = None
    msg_id = kc.execute(cell.source, store_history=True)
    outs = cell["outputs"]
    done = False
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            msg = kc.get_iopub_msg(timeout=1)
        except queue.Empty:
            if done:
                break
            continue
        if msg.get("parent_header", {}).get("msg_id") != msg_id:
            continue
        mt, c = msg["msg_type"], msg["content"]
        if mt == "status" and c["execution_state"] == "idle":
            done = True
        elif mt == "stream":
            outs.append(nbformat.v4.new_output("stream", name=c["name"], text=c["text"]))
        elif mt in ("display_data", "update_display_data"):
            outs.append(nbformat.v4.new_output("display_data", data=c["data"],
                                               metadata=c.get("metadata", {})))
        elif mt == "execute_result":
            cell["execution_count"] = c.get("execution_count")
            outs.append(nbformat.v4.new_output(
                "execute_result", data=c["data"], metadata=c.get("metadata", {}),
                execution_count=c.get("execution_count")))
        elif mt == "error":
            outs.append(nbformat.v4.new_output(
                "error", ename=c["ename"], evalue=c["evalue"], traceback=c["traceback"]))
    return cell


def render(cell):
    parts = []
    for out in cell.get("outputs", []):
        t = out.get("output_type")
        if t == "stream":
            parts.append(out.get("text", ""))
        elif t in ("execute_result", "display_data"):
            data = out.get("data", {})
            if "text/plain" in data:
                parts.append(data["text/plain"])
            for k in data:
                if k.startswith("image/"):
                    parts.append(f"<{k} rendered>")
            if "text/html" in data:
                parts.append("[HTML] " + " ".join(data["text/html"].split())[:700])
        elif t == "error":
            parts.append("ERROR %s: %s\n%s" % (
                out.get("ename"), out.get("evalue"),
                "\n".join(out.get("traceback", []))[-2500:]))
    return "\n".join(parts)


print("DRIVER READY", flush=True)

while True:
    cmds = sorted(f for f in os.listdir(INBOX) if f.endswith(".json"))
    if not cmds:
        time.sleep(0.15)
        continue
    name = cmds[0]
    p = os.path.join(INBOX, name)
    try:
        with open(p, encoding="utf-8") as fh:
            cmd = json.load(fh)
    except Exception:
        time.sleep(0.1)
        continue
    os.remove(p)
    res = {"cmd": cmd}
    try:
        a = cmd["action"]
        if a == "quit":
            try:
                api(f"sessions/{SESSION_ID}", "DELETE")
            except Exception:
                pass
            _kill_server_tree()  # tree kill, not server.terminate() (leaks kernels on Windows)
            with open(os.path.join(OUTBOX, name), "w") as fh:
                json.dump({"ok": True, "msg": "bye"}, fh)
            break
        elif a == "set":
            i = cmd["index"]
            while len(nb.cells) <= i:
                nb.cells.append(nbformat.v4.new_code_cell(""))
            nb.cells[i] = nbformat.v4.new_code_cell(cmd["source"])
            save()
            res.update(ok=True, msg=f"cell {i} set ({len(nb.cells)} cells)")
        elif a == "delete":
            nb.cells.pop(cmd["index"])
            save()
            res.update(ok=True, msg=f"deleted; {len(nb.cells)} cells")
        elif a == "restart":
            kc.stop_channels()
            api(f"kernels/{KERNEL_ID}/restart", "POST")
            time.sleep(2)
            connect()
            res.update(ok=True, msg="kernel restarted")
        elif a == "run":
            i = cmd["index"]
            t0 = time.perf_counter()
            execute(nb.cells[i], timeout=cmd.get("timeout", 3600))
            dt = time.perf_counter() - t0
            save()
            res.update(ok=True, wall=round(dt, 3), out=render(nb.cells[i]))
        else:
            res.update(ok=False, msg="unknown action")
    except Exception:
        res.update(ok=False, msg=traceback.format_exc())
    with open(os.path.join(OUTBOX, name), "w", encoding="utf-8") as fh:
        json.dump(res, fh)
    print(f"done {name}", flush=True)
