"""Set up (and inspect) the manual end-to-end check for the JupyterLab live-cell push.

The `cash-live-cells` frontend extension can only be proven end to end in a real
browser against a real JupyterLab: every layer below it has unit tests, but
"does an unsaved edit reach the kernel before the execute_request" is a property
of the browser, the notebook model, and the ZMQ shell channel together. This
script builds the environment that check needs, so redoing it after any change
to ``labextension/src/index.ts`` costs minutes instead of an hour.

It does NOT drive the browser -- that part is manual (or agent-driven). What it
does is remove the three traps that silently invalidate the result:

1. **`cash autoload` must be OFF.** A developer machine may have
   ``~/.ipython/profile_default/startup/00-cash.py`` pre-importing cash into
   every kernel. That hides the real cold-start behaviour, so the launch command
   below points ``IPYTHONDIR`` at an empty directory, and cell 0 of the probe
   notebook ASSERTS the hook did not run. If cell 0 prints
   ``AUTOLOAD_OFF: False``, the whole run is void.

2. **JupyterLab autosave must be OFF.** Autosave fires on a timer; if it lands
   between the edit and the run, the saved ``.ipynb`` already holds the edit and
   a stale-file read would pass by accident. This writes an ``overrides.json``
   that turns it off (it needs a server restart to take effect).

3. **The FIRST execution burst on a fresh kernel legitimately falls back to the
   saved file.** The extension's ``comm_open`` necessarily arrives before the
   cell that runs ``import cash``, so the target does not exist yet and the open
   is refused; the extension re-opens on the next ``executionScheduled``. Run
   the import in its own execution and start measuring from the burst after it.
   For the same reason a cold-kernel "Run All" gets no live cells at all --
   every ``execute_request`` is queued up front. Do not test via Run All.

Protocol (run each cell with Ctrl+Enter; wait for one to finish before the next)::

    cell 0  guard        -- must print AUTOLOAD_OFF: True
    cell 1  import cash + %cash_on
    cell 2  import t4probe          (this burst is the one that opens the comm)
    cell 3  K = 1                   UPSTREAM
    cell 4  RESULT = t4probe.work(K)  DOWNSTREAM   -> baseline, log grows to 1
    cell 4  again, no edit          -- NEGATIVE CONTROL: log must NOT grow
    edit cell 3 to `K = 2`, do NOT save, run cell 4  -- log MUST grow, k=2
    cell 5  source probe            -- LAST_CELL_SOURCE must be "extension"

The oracle is deliberately NOT the badge. ``t4probe.work`` appends a line to
``t4log.txt`` with ``os.write`` (not ``builtins.open``, which cash shims) and a
fresh ``time.time_ns()``, so a replayed line is distinguishable from a real one,
and the file is readable from outside the kernel and outside the browser.
``print(len(t4probe.CALLS))`` is a cheap statement cash does not cache, so it
always reports the LIVE invocation count as a second, in-kernel oracle. Read the
badge too, as corroboration -- but trust the side effect.

Usage::

    python scripts/jupyterlab_live_cells_probe.py setup [--workdir DIR]
    python scripts/jupyterlab_live_cells_probe.py check  [--workdir DIR]

``setup`` builds the venv and prints the launch command; ``check`` prints the
on-disk notebook sources, the notebook's mtime (proof no save happened) and the
side-effect log, which is what you compare against the screen.

The default workdir is deliberately SHORT and outside the repo: a venv nested
under a long path trips Windows' 260-character path limit while pip is unpacking
itself, and the venv must never be committed.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_WORKDIR = Path(os.environ.get("TEMP", "/tmp")) / "cash-jlab-probe"

PROBE_MODULE = '''\
"""Side-effect oracle for the JupyterLab live-cell end-to-end check.

`work()` is the only thing the downstream cell calls. Every REAL invocation
appends to CALLS and writes one line to LOG via os.write -- deliberately NOT
builtins.open, so cash's writer shim cannot replay it on a cache restore.

Each line carries time.time_ns(), so a replayed line is distinguishable from a
fresh one: a replay would repeat an ns value that is already in the file.
"""
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "t4log.txt")

CALLS = []


def _emit(line):
    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    try:
        os.write(fd, (line + "\\n").encode("ascii"))
    finally:
        os.close(fd)


def work(k):
    """Expensive enough for cash to cache; loud enough to prove it did not."""
    time.sleep(0.35)
    ns = time.time_ns()
    CALLS.append((k, ns))
    _emit("work k=%s ns=%s n=%s" % (k, ns, len(CALLS)))
    return k * 100
'''

CELLS = [
    ("guard", '''import sys, os, pathlib
print("AUTOLOAD_OFF:", "cash" not in sys.modules)
print("IPYTHONDIR:", os.environ.get("IPYTHONDIR"))
print("STARTUP_DIR_FILES:", sorted(p.name for p in pathlib.Path(get_ipython().profile_dir.startup_dir).iterdir()))
print("PY:", sys.version.split()[0])'''),
    ("enable", '''import cash
%cash_on'''),
    ("warmup", '''import t4probe
print("PROBE_LOADED", t4probe.LOG)'''),
    ("upstream", '''K = 1'''),
    ("downstream", '''RESULT = t4probe.work(K)
print("RESULT", RESULT, "LIVE_CALLS", len(t4probe.CALLS))'''),
    ("source", '''from cash.notebook import server_discovery as _sd
from cash.notebook import live_cells as _lc
print("LAST_CELL_SOURCE:", _sd.last_cell_source())
_c = _lc.latest_cells()
print("SEQ:", _lc._store["seq"], "NCELLS:", None if _c is None else len(_c))
for _x in (_c or []):
    if _x.get("source", "").startswith("K ="):
        print("KERNEL_SEES_UPSTREAM:", repr(_x["source"]))'''),
]


def _notebook() -> dict:
    return {
        "cells": [
            {"cell_type": "code", "execution_count": None, "id": cid,
             "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}
            for cid, src in CELLS
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _build_wheel(dist: Path) -> Path:
    dist.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "build", "--wheel", "--outdir", str(dist)],
                   cwd=str(_repo_root()), check=True)
    wheels = sorted(dist.glob("*.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels:
        raise SystemExit("no wheel produced")
    return wheels[-1]


def setup(workdir: Path) -> None:
    nb_dir = workdir / "nb"
    venv = workdir / "venv"
    nb_dir.mkdir(parents=True, exist_ok=True)
    (workdir / "ipythondir").mkdir(parents=True, exist_ok=True)

    wheel = _build_wheel(workdir / "dist")
    print(f"[probe] built {wheel.name}")

    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run([str(py), "-m", "pip", "install", "-q", str(wheel), "jupyterlab"],
                   check=True)
    print("[probe] installed wheel + jupyterlab")

    (nb_dir / "t4probe.py").write_text(PROBE_MODULE, encoding="ascii")
    (nb_dir / "t4.ipynb").write_text(json.dumps(_notebook(), indent=1), encoding="utf-8")
    log = nb_dir / "t4log.txt"
    if log.exists():
        log.unlink()

    # Trap 2: autosave off. Read at server START, so this must precede launch.
    settings = venv / "share/jupyter/lab/settings"
    settings.mkdir(parents=True, exist_ok=True)
    (settings / "overrides.json").write_text(json.dumps(
        {"@jupyterlab/docmanager-extension:plugin": {"autosave": False}}, indent=2),
        encoding="utf-8")

    jlab = venv / ("Scripts/jupyter-lab.exe" if os.name == "nt" else "bin/jupyter-lab")
    print("\n[probe] launch with (note IPYTHONDIR -- trap 1):\n")
    print(f'  IPYTHONDIR="{workdir / "ipythondir"}" \\\n'
          f'    "{jlab}" --no-browser --port=8899 \\\n'
          f'    --IdentityProvider.token=cashprobe \\\n'
          f'    --ServerApp.root_dir="{nb_dir}" --ServerApp.open_browser=False\n')
    print("  then open  http://localhost:8899/lab/tree/t4.ipynb?token=cashprobe")
    print("  and follow the protocol in this file's module docstring.")
    print(f"\n[probe] inspect with:  python {Path(__file__).name} check "
          f"--workdir {workdir}")


def check(workdir: Path) -> None:
    nb = workdir / "nb" / "t4.ipynb"
    log = workdir / "nb" / "t4log.txt"
    if not nb.exists():
        raise SystemExit(f"no notebook at {nb} -- run `setup` first")
    st = nb.stat()
    print(f"NB mtime={st.st_mtime:.3f} size={st.st_size}   "
          f"(an unchanged mtime proves no save happened)")
    for cell in json.loads(nb.read_text(encoding="utf-8"))["cells"]:
        if cell["id"] in ("upstream", "downstream"):
            print(f"  DISK[{cell['id']}] = {''.join(cell['source'])!r}")
    lines = log.read_text(encoding="ascii").splitlines() if log.exists() else []
    print(f"LOG lines={len(lines)}   (one line per REAL work() invocation)")
    for line in lines:
        print("  " + line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("setup", "check"))
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR,
                        help=f"default: {DEFAULT_WORKDIR} (keep it SHORT on Windows)")
    args = parser.parse_args()
    (setup if args.action == "setup" else check)(args.workdir)


if __name__ == "__main__":
    main()
