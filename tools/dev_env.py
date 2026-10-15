"""Sets up, checks and removes an isolated development environment for cash.

One work stream gets one directory: a git worktree with its own ``.venv``,
cash installed editable from that worktree, and the C extension compiled
from that worktree's sources. ``check`` then proves the pieces fit together,
so a test run never grades code from somewhere else::

    python tools/dev_env.py create ../cash-feature --branch my-feature --python 3.13
    python tools/dev_env.py create --here             # a .venv in this checkout
    python tools/dev_env.py check ../cash-feature
    python tools/dev_env.py clean ../cash-feature

``check`` fails (exit 1) when:

* the venv's python is missing, is a broken link, or does not run;
* ``import cash`` in that python finds a cash other than this checkout's
  ``src/cash`` (a stray ``PYTHONPATH``, a ``.pth`` file, another install);
* the C extension should load on this interpreter but does not, or the
  compiled module is older than a C source (rebuild: ``python hatch_build.py``);
* a Jupyter kernel started the way the integration tests start one (kernel
  name ``python3``, from the venv's python) runs another interpreter or
  imports another cash;
* the disk holding the checkout or the temp dir has less free space than
  ``--min-free-gb``, or the temp dir is not writable.

A checkout on a network or FUSE file system is only a warning.

``clean`` removes only what ``create`` made: it reads the marker ``create``
leaves in the venv and refuses any directory without one.

Standard library only; runs on Linux, macOS and Windows. Needs ``uv`` on PATH.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The file ``create`` writes into the venv; ``clean`` acts only where it finds one.
MARKER_NAME = ".cash-dev-env.json"
MARKER_TOOL = "cash-dev-env"
DEFAULT_EXTRAS = "dev,docs-test"
DEFAULT_MIN_FREE_GB = 5.0
KERNEL_TIMEOUT_S = 15
#: File systems a checkout should not live on: slow, and their locks and
#: mtimes are not what the cache or the build expect.
NETWORK_FS_TYPES = frozenset(
    {"nfs", "nfs4", "cifs", "smb3", "smbfs", "9p", "afs", "ncpfs", "sshfs", "davfs", "glusterfs", "ceph"}
)


# ---------------------------------------------------------------------------
# Pure helpers (unit tested in tests/test_tooling/test_dev_env.py)
# ---------------------------------------------------------------------------


def canon(path: str | os.PathLike) -> str:
    """*path* made comparable: absolute, links resolved, case folded where the OS folds it."""
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def is_within(path: str | os.PathLike, root: str | os.PathLike) -> bool:
    """True when *path* is *root* or lies under it (after :func:`canon`)."""
    p, r = canon(path), canon(root)
    return p == r or p.startswith(r.rstrip(os.sep) + os.sep)


def same_file(a: str | os.PathLike, b: str | os.PathLike) -> bool:
    """True when *a* and *b* name the same path, without resolving links.

    A venv's python is a link to the base interpreter, so resolving links would
    make every venv's python look like every other's.
    """
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def free_gb(path: str | os.PathLike) -> float:
    return shutil.disk_usage(path).free / 1024**3


def disk_problem(path: str | os.PathLike, min_free_gb: float, free: float | None = None) -> str | None:
    """Why the disk under *path* is too full, or None. *free* (GB) overrides the measurement."""
    free = free_gb(path) if free is None else free
    if free < min_free_gb:
        return f"only {free:.1f} GB free on the disk holding {path} (want {min_free_gb:g} GB; --min-free-gb)"
    return None


def stale_sources(built: Path, sources: list[Path]) -> list[Path]:
    """The *sources* modified after *built* was written."""
    built_mtime = built.stat().st_mtime
    return [s for s in sources if s.stat().st_mtime > built_mtime]


def mount_fs_type(path: str | os.PathLike, mounts_text: str) -> str | None:
    """The file system type of the mount holding *path*, from ``/proc/mounts`` text."""
    target = canon(path)
    best, best_type = "", None
    for line in mounts_text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        point = parts[1].replace("\\040", " ")
        if (target == point or target.startswith(point.rstrip("/") + "/")) and len(point) >= len(best):
            best, best_type = point, parts[2]
    return best_type


def is_network_fs(fs_type: str | None) -> bool:
    if not fs_type:
        return False
    return fs_type in NETWORK_FS_TYPES or fs_type.startswith("fuse.") or fs_type == "fuse"


def network_mount_note(path: Path) -> str | None:
    """A note when *path* is on a network or FUSE mount; None when it is not or we cannot tell."""
    if os.name == "nt":
        p = str(path.resolve())
        if p.startswith("\\\\"):
            return f"{p} is a network path"
        try:
            import ctypes

            drive = os.path.splitdrive(p)[0] + "\\"
            if ctypes.windll.kernel32.GetDriveTypeW(drive) == 4:  # DRIVE_REMOTE
                return f"{p} is on a network drive"
        except (AttributeError, OSError):
            return None
        return None
    try:
        mounts = Path("/proc/mounts").read_text(encoding="utf-8")
    except OSError:
        return None  # macOS and others: no cheap answer, so no warning
    fs_type = mount_fs_type(path, mounts)
    if is_network_fs(fs_type):
        return f"{path} is on a {fs_type} mount; builds and file watching there are slow and flaky"
    return None


def read_marker(target: Path) -> dict | None:
    try:
        data = json.loads((target / ".venv" / MARKER_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("tool") == MARKER_TOOL else None


def clean_refusal(target: Path, marker: dict | None, worktrees: list[str], repo_root: Path) -> str | None:
    """Why ``clean`` must not touch *target*, or None when it may.

    *worktrees* are the paths ``git worktree list`` reports for *repo_root*.
    """
    if not target.is_dir():
        return f"{target} is not a directory"
    if marker is None:
        return f"{target} has no {MARKER_NAME} in .venv; it was not made by this tool, so it is left alone"
    if not same_file(canon(marker.get("path", "")), canon(target)):
        return f"the marker in {target} was written for {marker.get('path')!r}; refusing a copied or moved environment"
    if marker.get("worktree"):
        if is_within(repo_root, target):
            return f"{target} holds the main checkout {repo_root}; refusing to remove it"
        if not any(same_file(canon(w), canon(target)) for w in worktrees):
            return f"{target} is not a git worktree of {repo_root}"
    return None


# ---------------------------------------------------------------------------
# Running things
# ---------------------------------------------------------------------------


def run(cmd: list[str], cwd: Path | None = None, check: bool = True, **kw) -> subprocess.CompletedProcess:
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=cwd, check=check, **kw)


def git(*args: str, cwd: Path = REPO_ROOT) -> str:
    out = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8")
    return out.stdout.strip()


def git_ok(*args: str, cwd: Path = REPO_ROOT) -> bool:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True).returncode == 0


def worktree_paths(repo_root: Path) -> list[str]:
    out = git("worktree", "list", "--porcelain", cwd=repo_root)
    return [line[len("worktree ") :] for line in out.splitlines() if line.startswith("worktree ")]


def need_uv() -> str:
    uv = shutil.which("uv")
    if not uv:
        sys.exit("dev_env: uv is not on PATH. Install it: https://docs.astral.sh/uv/getting-started/installation/")
    return uv


#: Runs in the venv's python. Prints one JSON object.
_PROBE = r"""
import json, sys, sysconfig
info = {"executable": sys.executable, "version": sys.version.split()[0],
        "ext_suffix": sysconfig.get_config_var("EXT_SUFFIX")}
try:
    import cash
    info["cash_file"] = cash.__file__
    info["cash_version"] = getattr(cash, "__version__", "?")
    from cash import _native
    info["native_status"] = _native.status_line()
    info["native_available"] = _native.AVAILABLE
    info["native_file"] = getattr(_native.cwatch, "__file__", None)
except Exception as exc:
    info["import_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(info))
"""

#: Starts a kernel as tests/_nbharness/kernels.py does and reports what it imports.
_KERNEL_PROBE = r"""
import json, os, sys, tempfile
os.environ["IPYTHONDIR"] = tempfile.mkdtemp(prefix="cash-devenv-ipython-")
os.environ.pop("CASH_CACHE_DIR", None)
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
timeout = float(sys.argv[1])
info = {}
try:
    spec = KernelSpecManager().get_kernel_spec("python3")
    info["spec_argv0"] = spec.argv[0] if spec.argv else None
    info["spec_dir"] = spec.resource_dir
except Exception as exc:
    info["spec_error"] = f"{type(exc).__name__}: {exc}"
km = KernelManager(kernel_name="python3")
try:
    km.start_kernel()
    kc = km.client()
    kc.start_channels()
    kc.wait_for_ready(timeout=timeout)
    out = []
    def hook(msg):
        if msg["msg_type"] == "stream":
            out.append(msg["content"]["text"])
        elif msg["msg_type"] == "error":
            out.append("ERROR " + msg["content"]["ename"] + ": " + msg["content"]["evalue"] + "\n")
    code = "import sys, json, cash\nprint(json.dumps([sys.executable, cash.__file__]))"
    kc.execute_interactive(code, timeout=timeout, output_hook=hook)
    text = "".join(out).strip()
    if text.startswith("ERROR"):
        info["kernel_error"] = text
    else:
        info["kernel_executable"], info["kernel_cash_file"] = json.loads(text.splitlines()[-1])
    kc.stop_channels()
except Exception as exc:
    info["kernel_error"] = f"{type(exc).__name__}: {exc}"
finally:
    try:
        km.shutdown_kernel(now=True)
    except Exception:
        pass
print(json.dumps(info))
"""


def _probe(python: Path, script: str, *args: str, timeout: float) -> dict:
    """Runs *script* in *python* from a neutral directory and returns its JSON."""
    with tempfile.TemporaryDirectory(prefix="cash-devenv-probe-") as neutral:
        proc = subprocess.run(
            [str(python), "-c", script, *args],
            cwd=neutral,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    lines = proc.stdout.strip().splitlines()
    if proc.returncode != 0 or not lines:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        raise RuntimeError(f"exit {proc.returncode}: " + " | ".join(tail))
    return json.loads(lines[-1])


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


@dataclass
class Report:
    rows: list[tuple[str, str, str]] = field(default_factory=list)

    def ok(self, name: str, detail: str) -> None:
        self.rows.append(("ok", name, detail))

    def warn(self, name: str, detail: str) -> None:
        self.rows.append(("warn", name, detail))

    def fail(self, name: str, detail: str) -> None:
        self.rows.append(("FAIL", name, detail))

    @property
    def failed(self) -> bool:
        return any(status == "FAIL" for status, _, _ in self.rows)

    def print(self) -> None:
        for status, name, detail in self.rows:
            print(f"  [{status:>4}] {name:<10} {detail}")
        print("check: FAILED" if self.failed else "check: all good")


def check_python(report: Report, python: Path) -> dict | None:
    if not python.exists():
        if python.is_symlink():
            report.fail("python", f"{python} is a broken link to {os.readlink(python)}; recreate the venv")
        else:
            report.fail("python", f"{python} does not exist; run `create` (or `uv venv`) first")
        return None
    try:
        info = _probe(python, _PROBE, timeout=60)
    except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError) as exc:
        report.fail("python", f"{python} does not run: {exc}")
        return None
    report.ok("python", f"{python} (Python {info['version']})")
    return info


def check_import(report: Report, info: dict, target: Path) -> bool:
    """True when the venv imports this checkout's cash."""
    if "import_error" in info:
        report.fail("import", f"`import cash` fails: {info['import_error']}")
        return False
    cash_file = info["cash_file"]
    if is_within(cash_file, target / "src" / "cash"):
        report.ok("import", f"cash {info['cash_version']} from {Path(cash_file).parent}")
        return True
    else:
        hint = " (PYTHONPATH is set)" if os.environ.get("PYTHONPATH") else ""
        report.fail(
            "import",
            f"`import cash` loads {cash_file}, not {target / 'src' / 'cash'}{hint}; "
            f"reinstall with `uv pip install -e .` in {target} and unset PYTHONPATH",
        )
        return False


def extension_expected(version: str) -> str | None:
    """Why no C extension is expected for Python *version*, or None when one is."""
    if os.environ.get("CASH_PURE_PYTHON", "").strip().lower() in ("1", "true", "yes", "on"):
        return "CASH_PURE_PYTHON is set"
    major, minor = (int(x) for x in version.split(".")[:2])
    if (major, minor) < (3, 12):
        return f"Python {version} is older than 3.12"
    return None


def check_extension(report: Report, info: dict, target: Path) -> None:
    skip = extension_expected(info["version"])
    sources = sorted((target / "src" / "cash" / "_c").glob("*.[ch]"))
    if skip or not sources:
        report.ok("c-ext", f"not expected ({skip or 'no C sources'}); {info['native_status']}")
        return
    built = target / "src" / "cash" / ("_cwatch" + (info.get("ext_suffix") or ""))
    if not built.is_file():
        report.fail("c-ext", f"{built.name} is not built; run `python hatch_build.py` in {target}")
        return
    stale = stale_sources(built, sources)
    if stale:
        names = ", ".join(s.name for s in stale)
        report.fail("c-ext", f"{built.name} is older than {names}; rebuild with `python hatch_build.py` in {target}")
        return
    if not info.get("native_available"):
        report.fail("c-ext", f"built but not loaded: {info['native_status']}")
        return
    if info.get("native_file") and not is_within(info["native_file"], built):
        report.fail("c-ext", f"loads {info['native_file']}, not {built}")
        return
    report.ok("c-ext", f"{info['native_status']}, {built.name} newer than its sources")


def check_kernel(report: Report, python: Path, target: Path, timeout: float) -> None:
    started = time.monotonic()
    try:
        info = _probe(python, _KERNEL_PROBE, str(timeout), timeout=timeout + 10)
    except subprocess.TimeoutExpired:
        report.fail("kernel", f"the python3 kernel did not answer within {timeout:g} s")
        return
    except (OSError, RuntimeError, ValueError) as exc:
        report.fail("kernel", f"could not start the python3 kernel: {exc}")
        return
    took = time.monotonic() - started
    where = f"spec {info.get('spec_dir')}"
    repoint = f"remove that spec, or repoint it with {python} -m ipykernel install --user"
    argv0 = info.get("spec_argv0")
    # A bare `python` in the spec means the launching interpreter: the venv's.
    if argv0 and os.path.isabs(argv0) and not same_file(argv0, python):
        report.fail("kernel", f"the python3 kernel spec runs {argv0}, not {python} ({where}); {repoint}")
    elif "kernel_error" in info:
        report.fail("kernel", f"the python3 kernel failed: {info['kernel_error']} ({where})")
    elif not same_file(info["kernel_executable"], python):
        report.fail("kernel", f"the python3 kernel runs {info['kernel_executable']}, not {python} ({where}); {repoint}")
    elif not is_within(info["kernel_cash_file"], target / "src" / "cash"):
        report.fail("kernel", f"the kernel imports cash from {info['kernel_cash_file']}, not this checkout")
    else:
        report.ok("kernel", f"python3 kernel imports this checkout's cash ({took:.1f} s)")


def check_disk(report: Report, target: Path, min_free_gb: float) -> None:
    tmp = Path(tempfile.gettempdir())
    for label, path in (("disk", target), ("tmp-disk", tmp)):
        problem = disk_problem(path, min_free_gb)
        if problem:
            report.fail(label, problem)
        else:
            report.ok(label, f"{free_gb(path):.1f} GB free under {path}")
    try:
        fd, name = tempfile.mkstemp(prefix="cash-devenv-", dir=tmp)
        os.write(fd, b"x")
        os.close(fd)
        os.unlink(name)
        report.ok("tmp", f"{tmp} is writable")
    except OSError as exc:
        report.fail("tmp", f"cannot write to {tmp}: {exc}")
    note = network_mount_note(target)
    if note:
        report.warn("mount", note)


def do_check(target: Path, min_free_gb: float, kernel: bool = True, kernel_timeout: float = KERNEL_TIMEOUT_S) -> int:
    target = target.resolve()
    print(f"check {target}")
    report = Report()
    if not (target / "pyproject.toml").is_file() or not (target / "src" / "cash").is_dir():
        report.fail("checkout", f"{target} is not a cash checkout")
        report.print()
        return 1
    python = venv_python(target / ".venv")
    info = check_python(report, python)
    # The extension and kernel checks say nothing useful about another cash.
    if info is not None and check_import(report, info, target):
        check_extension(report, info, target)
        if kernel:
            check_kernel(report, python, target, kernel_timeout)
    check_disk(report, target, min_free_gb)
    report.print()
    return 1 if report.failed else 0


# ---------------------------------------------------------------------------
# create / clean
# ---------------------------------------------------------------------------


def _add_worktree(target: Path, branch: str | None, base: str) -> bool:
    """Adds the worktree; returns True when it also created *branch*."""
    if target.exists() and any(target.iterdir()):
        sys.exit(f"dev_env: {target} exists and is not empty")
    if branch is None:
        run(["git", "worktree", "add", "--detach", target, base], cwd=REPO_ROOT)
        return False
    if git_ok("show-ref", "--verify", "--quiet", f"refs/heads/{branch}"):
        run(["git", "worktree", "add", target, branch], cwd=REPO_ROOT)
        return False
    if git_ok("show-ref", "--verify", "--quiet", f"refs/remotes/origin/{branch}"):
        base = f"origin/{branch}"
    run(["git", "worktree", "add", "-b", branch, target, base], cwd=REPO_ROOT)
    return True


def do_create(args: argparse.Namespace) -> int:
    uv = need_uv()
    if args.here:
        target = REPO_ROOT
        created_branch = False
        if (target / ".venv").exists():
            sys.exit(f"dev_env: {target / '.venv'} exists; `clean --here` it first or use it as it is")
    else:
        if args.dir is None:
            sys.exit("dev_env: create needs a directory, or --here")
        target = Path(args.dir).expanduser().resolve()
        created_branch = _add_worktree(target, args.branch, args.base)
    venv = target / ".venv"
    venv_cmd = [uv, "venv", venv]
    if args.python:
        venv_cmd += ["--python", args.python]
    run(venv_cmd, cwd=target)
    marker = {
        "tool": MARKER_TOOL,
        "path": str(target),
        "worktree": not args.here,
        "repo": str(REPO_ROOT),
        "branch": args.branch,
        "created_branch": created_branch,
    }
    (venv / MARKER_NAME).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    python = venv_python(venv)
    spec = f".[{args.extras}]" if args.extras else "."
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")}
    run([uv, "pip", "install", "--python", python, "-e", spec], cwd=target, env=env)
    # The editable install compiles the extension too, but a cached build or a
    # missing compiler can skip it quietly; compile in place so check sees it.
    build = run([python, "hatch_build.py"], cwd=target, check=False, env=env)
    if build.returncode != 0:
        print("dev_env: the C extension was not built (see above); cash will run its pure-Python path")
    print()
    return do_check(target, args.min_free_gb)


def _rmtree(path: Path) -> None:
    def _retry_writable(func, p, _exc):  # Windows: read-only files in .git and venvs
        os.chmod(p, stat.S_IWRITE)
        func(p)

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_retry_writable)
    else:
        shutil.rmtree(path, onerror=_retry_writable)


def do_clean(args: argparse.Namespace) -> int:
    target = REPO_ROOT if args.here else Path(args.dir).expanduser().resolve() if args.dir else None
    if target is None:
        sys.exit("dev_env: clean needs a directory, or --here")
    marker = read_marker(target)
    worktrees = worktree_paths(REPO_ROOT) if marker and marker.get("worktree") else []
    if args.here and marker and marker.get("worktree"):
        sys.exit("dev_env: refusing: --here names a worktree environment; pass its directory")
    refusal = clean_refusal(target, marker, worktrees, REPO_ROOT)
    if refusal:
        print(f"dev_env: refusing: {refusal}", file=sys.stderr)
        return 2
    print(f"removing {target / '.venv'}")
    _rmtree(target / ".venv")
    if not marker.get("worktree"):
        return 0
    remove = ["git", "worktree", "remove", target]
    if args.force:
        remove.insert(3, "--force")
    if run(remove, cwd=REPO_ROOT, check=False).returncode != 0:
        print("dev_env: git kept the worktree (uncommitted changes?); --force removes it anyway", file=sys.stderr)
        return 1
    branch = marker.get("branch")
    if marker.get("created_branch") and branch:
        # -d, not -D: a branch with commits that are nowhere else stays.
        if run(["git", "branch", "-d", branch], cwd=REPO_ROOT, check=False).returncode != 0:
            print(f"dev_env: kept branch {branch}; it has unmerged commits")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dev_env.py", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, dir_help: str) -> None:
        p.add_argument("dir", nargs="?", help=dir_help)
        p.add_argument("--here", action="store_true", help="this checkout instead of a new worktree")

    p_create = sub.add_parser("create", help="worktree + .venv + editable install + C extension, then check")
    common(p_create, "where the new worktree goes")
    p_create.add_argument("--branch", help="branch to check out (made from --base, or origin/BRANCH, if new)")
    p_create.add_argument("--base", default="HEAD", help="where a new branch starts (default HEAD)")
    p_create.add_argument("--python", help="interpreter for uv venv, e.g. 3.13")
    p_create.add_argument("--extras", default=DEFAULT_EXTRAS, help=f"extras to install (default {DEFAULT_EXTRAS})")
    p_create.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)

    p_check = sub.add_parser("check", help="verify an environment (exit 1 on any failure)")
    common(p_check, "the environment's directory (default: the current directory)")
    p_check.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)
    p_check.add_argument("--no-kernel", action="store_true", help="skip starting a Jupyter kernel")

    p_clean = sub.add_parser("clean", help="remove an environment made by create")
    common(p_clean, "the environment's directory")
    p_clean.add_argument("--force", action="store_true", help="remove the worktree even with local changes")

    args = parser.parse_args(argv)
    if args.command == "create":
        return do_create(args)
    if args.command == "check":
        target = REPO_ROOT if args.here else Path(args.dir or os.getcwd())
        return do_check(target, args.min_free_gb, kernel=not args.no_kernel)
    return do_clean(args)


if __name__ == "__main__":
    sys.exit(main())
