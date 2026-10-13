"""Where "here" is for this process: the project it works on, and where it caches.

Decided from how the process was launched -- a script, an installed console
script or ``python -m`` module, pytest and its xdist workers, an interactive
shell -- because the working directory alone made the cache a property of
where you were standing rather than of what you were running. `config` asks
`project_anchor` for the default cache location and the ``pyproject.toml`` to
read, and `installed_entry_point_cache_dir` for a tool run outside any project.
The file backend calls `announce_new_cache_dir` when it creates that default
folder.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path

from .install_paths import is_installed_path

logger = logging.getLogger(__name__)

__all__ = [
    "PROJECT_MARKERS",
    "announce_new_cache_dir",
    "default_project_config_path",
    "default_user_config_path",
    "installed_entry_point_cache_dir",
    "interactive_shell_is_running",
    "per_user_cache_root",
    "project_anchor",
]

#: Files that mean "the project starts here".
PROJECT_MARKERS = ("pyproject.toml", "setup.py", "setup.cfg", ".git")


def interactive_shell_is_running() -> bool:
    """True inside IPython, a Jupyter kernel, or anything else hosting one.

    ``__main__.__file__`` cannot be trusted there. IPython SETS it, temporarily,
    while it executes each of the profile's startup scripts -- so a kernel that
    imports cash from a startup file resolves an anchor inside
    ``~/.ipython/profile_default/startup`` and writes the session's cache there.
    Measured exactly that: a notebook whose entries went to the profile
    directory after a kernel restart, so nothing hit.

    There is no "running script" in an interactive session anyway. The cwd is
    the right answer, and it is the one a notebook has always had.

    A running shell has imported IPython, so a process that has not is not in
    one: a plain script is answered without importing IPython (50-160 ms).
    """
    ipython = sys.modules.get("IPython")
    if ipython is None:
        return False
    try:
        return ipython.get_ipython() is not None
    except Exception:  # noqa: BLE001 - a half-initialised IPython is not one
        return False


def _running_cash_cli() -> bool:
    """Is ``__main__`` cash's own command line (``python -m cash``)?

    It is a tool acting on the project you are standing in, wherever its
    source lives. From an editable checkout it looked like a local script,
    anchored to cash's own repository, and ``python -m cash clear --all`` run
    inside another project cleared the cash checkout's cache instead.
    """
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    return getattr(spec, "name", None) == "cash.__main__"


def _running_script_dir() -> Path | None:
    """The directory of the script being run, or None if that is meaningless.

    None for an interactive interpreter, a notebook, ``python -c``, and for any
    ``__main__`` that lives inside the interpreter's own installation -- an
    installed console entry point, ``python -m pytest``, the Jupyter kernel
    launcher. Those all report a ``__file__`` somewhere under ``sys.prefix`` or
    site-packages, and anchoring a user's cache inside their virtualenv because
    they ran an installed tool would be a worse answer than the cwd.
    """
    if interactive_shell_is_running() or _running_cash_cli():
        return None
    main = sys.modules.get("__main__")
    raw = getattr(main, "__file__", None)
    if _pool_launcher_main():
        # A joblib (loky) worker runs ``python -m ...popen_loky_posix``: its
        # ``__main__`` is joblib's launcher, an installed module, while
        # ``sys.argv`` is the parent's. Taken at its word, the worker was
        # "the installed tool joblib" and cached in the per-user
        # ``~/.cache/cash/joblib`` (or the cwd's project) while its parent
        # cached beside the script.
        raw = None
    if not raw:
        # A spawned multiprocessing worker has no ``__main__.__file__`` and no
        # ``__spec__`` -- but it does inherit the parent's ``sys.argv[0]``.
        # Without this the parent would anchor to its project and every pool
        # worker to the cwd: one run writing into two cache directories,
        # neither seeing the other's entries.
        #
        # Only a real file counts, which is what keeps the interpreter's own
        # invocations out: ``python -c`` leaves ``-c`` here, a REPL leaves the
        # empty string, and an installed console script is filtered below like
        # any other path inside the interpreter's installation.
        raw = sys.argv[0] if sys.argv else None
        if raw and not os.path.isabs(raw):
            # Relative to where the parent started, which multiprocessing
            # and loky both hand their workers (the cwd may have moved since).
            original = getattr(sys.modules.get("multiprocessing.process"), "ORIGINAL_DIR", None)
            if original:
                raw = os.path.join(original, raw)
        if not raw or not str(raw).endswith(".py") or not os.path.isfile(raw):
            return None
    try:
        path = Path(raw).resolve()
    except OSError:
        return None
    if is_installed_path(path):
        return None
    return path.parent


def _is_installed_file(raw: str) -> bool:
    try:
        return is_installed_path(Path(raw).resolve())
    except OSError:
        return False


def _pool_launcher_main() -> bool:
    """Is this a pool worker whose ``__main__`` is not the parent's program?

    A joblib (loky) worker runs ``python -m <joblib's launcher>``: installed
    code. A spawned worker of ``python -m pkg`` has a bare ``__main__``.
    ``sys.argv`` is still the parent's in both, so the program is the one
    ``sys.argv[0]`` names.
    """
    raw = getattr(sys.modules.get("__main__"), "__file__", None)
    return _multiprocessing_parent_pid() is not None and (not raw or _is_installed_file(raw))


def _argv_installed_module_name() -> str | None:
    """The top-level module of ``sys.argv[0]`` when that is an installed file:
    what a pool worker of ``python -m <installed module>`` was started by."""
    raw = sys.argv[0] if sys.argv else ""
    if not raw.endswith(".py") or not _is_installed_file(raw):
        return None
    try:
        path = Path(raw).resolve()
    except OSError:
        return None
    best: tuple[str, ...] = ()
    for entry in sys.path:
        try:
            parts = path.relative_to(Path(entry or ".").resolve()).parts
        except (OSError, ValueError):
            continue
        if parts and (not best or len(parts) < len(best)):
            best = parts
    if not best:
        return None
    return best[0][:-3] if len(best) == 1 else best[0]


def _multiprocessing_parent_pid() -> int | None:
    """The pid of the process that started this one as a multiprocessing or
    joblib (loky) worker, or None when this is no such worker.

    Read only if multiprocessing is already imported: a worker always has it.
    """
    process = sys.modules.get("multiprocessing.process")
    if process is None:
        return None
    try:
        parent = process.parent_process()
    except Exception:  # noqa: BLE001 - an odd multiprocessing is not a worker
        return None
    pid = getattr(parent, "pid", None)
    return pid if isinstance(pid, int) else None


#: The environment variable through which a process hands its pool workers
#: where it resolved "here" (`project_anchor`, `installed_entry_point_cache_dir`).
LAUNCH_ENV = "_CASH_PARENT_LAUNCH"


#: (pid, `_read_parent_launch()`) once read: this process re-exports
#: `LAUNCH_ENV` under its own pid, so it cannot read its parent's again.
_PARENT_LAUNCH: tuple[int, dict | None] | None = None


def _parent_launch() -> dict | None:
    global _PARENT_LAUNCH
    if _PARENT_LAUNCH is None or _PARENT_LAUNCH[0] != os.getpid():
        _PARENT_LAUNCH = (os.getpid(), _read_parent_launch())
    return _PARENT_LAUNCH[1]


def _read_parent_launch() -> dict | None:
    """What the parent resolved, when this process is its pool worker.

    A worker is not launched the way its parent was -- joblib starts it as
    ``python -m <joblib's launcher>``, a forkserver from wherever that server
    was started, an interactive parent has no script at all -- so it cannot
    re-derive the parent's answer; it reads it. Only from its own parent (by
    pid): a script that a worker merely runs with ``subprocess`` resolves as
    itself.
    """
    raw = os.environ.get(LAUNCH_ENV)
    if not raw:
        return None
    parent = _multiprocessing_parent_pid()
    if parent is None:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, dict) or data.get("pid") != parent or not isinstance(data.get("anchor"), str):
        return None
    return data


def _launch() -> tuple[Path, Path | None]:
    """This process's anchor and per-user tool cache, handed on to its workers.

    Recorded in the environment on every resolution, so a pool started
    afterwards inherits the answer the parent is using now.
    """
    inherited = _parent_launch()
    if inherited is not None:
        anchor = Path(inherited["anchor"])
        entry = inherited.get("entry")
        entry_dir = Path(entry) if isinstance(entry, str) else None
    else:
        anchor = _own_project_anchor()
        entry_dir = _own_installed_entry_point_cache_dir()
    try:
        os.environ[LAUNCH_ENV] = json.dumps(
            {"pid": os.getpid(), "anchor": str(anchor), "entry": None if entry_dir is None else str(entry_dir)}
        )
    except (TypeError, ValueError, OSError):  # pragma: no cover - a path that cannot be stored
        pass
    return anchor, entry_dir


def _running_installed_module() -> bool:
    """``python -m <module installed in site-packages>`` -- ``python -m pytest``."""
    if _pool_launcher_main():
        return _argv_installed_module_name() is not None
    main = sys.modules.get("__main__")
    if getattr(main, "__spec__", None) is None:
        return False
    if _running_cash_cli():
        return True
    raw = getattr(main, "__file__", None)
    if not raw:
        return False
    try:
        return is_installed_path(Path(raw).resolve())
    except OSError:
        return False


def _running_installed_code() -> bool:
    """Is the program itself installed code -- a console script or ``-m`` module?

    Then the code being cached is the user's project code it runs, and the
    project the user is standing in is the best anchor there is. Not true of a
    notebook, a REPL or ``python -c``, which keep the cwd.
    """
    if interactive_shell_is_running():
        return False
    return _running_console_script() is not None or _running_installed_module()


#: The tables that make a ``pyproject.toml`` a project's. A ``tests/pyproject.toml``
#: holding only ``[tool.ruff]`` does not make ``tests/`` a project of its own
#: (a second, cold cache when pytest runs from there, and the repository's
#: ``[tool.cash]`` ignored).
_PYPROJECT_PROJECT_TABLE = re.compile(
    r"^\s*\[\[?\s*(project|build-system|tool\.cash|tool\.poetry)\s*[\].]", re.MULTILINE
)


def _marks_project(directory: Path, marker: str) -> bool:
    path = directory / marker
    if marker != "pyproject.toml":
        return path.exists()
    try:
        return bool(_PYPROJECT_PROJECT_TABLE.search(path.read_text(encoding="utf-8-sig")))
    except (OSError, UnicodeDecodeError):
        return False


def _home() -> Path | None:
    try:
        return Path.home().resolve()
    except (OSError, RuntimeError):  # no home directory to speak of
        return None


def _holds_existing_cache(directory: Path, home: Path | None) -> bool:
    """Does *directory* hold a cache cash made (``.cash/CACHE_VERSION``)?

    A notebook folder, or a project whose only ``pyproject.toml`` configures a
    linter, has no other marker, and a script in a subfolder of it started a
    second, cold cache beside itself. Never the filesystem root, the home
    directory or a directory above it: a stray ``~/.cash`` must not claim
    every script under the home directory.
    """
    from .backends.cache_dir import VERSION_FILENAME

    if directory.parent == directory:
        return False
    if home is not None and home.is_relative_to(directory):
        return False
    return (directory / ".cash" / VERSION_FILENAME).is_file()


def _project_root_above(start: Path, *, existing_cache: bool = False) -> Path | None:
    """The first directory at or above *start* holding a project marker.

    A ``pyproject.toml`` counts when it describes a project -- ``[project]``,
    ``[build-system]``, ``[tool.poetry]`` -- or configures cash; one that only
    configures a linter does not. With *existing_cache*, so does a cache cash
    already made there (`_holds_existing_cache`).
    """
    home = _home() if existing_cache else None
    for d in [start, *start.parents]:
        try:
            if any(_marks_project(d, marker) for marker in PROJECT_MARKERS):
                return d
            if existing_cache and _holds_existing_cache(d, home):
                return d
        except OSError:
            continue
    return None


def _cwd_project_root() -> Path | None:
    """The first directory at or above the cwd holding a project marker."""
    try:
        here = Path.cwd()
    except OSError:
        return None
    return _project_root_above(here)


def _running_pytest() -> bool:
    """Is this a pytest process -- the one that was typed, or an xdist worker?

    A worker is started as ``python -c``, so nothing in its ``argv`` or
    ``__main__`` says pytest; the variable xdist sets for it does, together
    with that ``-c``. ``pytest`` must also be imported, which keeps out a
    plain subprocess that merely inherited the variable from a test; and an
    interactive shell is never pytest, whatever it has imported.
    """
    if "pytest" not in sys.modules or interactive_shell_is_running():
        return False
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0 == "-c" and os.environ.get("PYTEST_XDIST_WORKER"):
        return True
    if _running_console_script() in ("pytest", "py.test"):
        return True
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    return getattr(spec, "name", None) in ("pytest", "pytest.__main__")


def _calling_code_project_root() -> Path | None:
    """The project of the nearest code on the stack that is neither cash's nor
    installed -- under pytest, the test module being collected or run."""
    own = Path(__file__).resolve().parent
    frame = sys._getframe(1)
    while frame is not None:
        name = frame.f_code.co_filename
        frame = frame.f_back
        if not name or name.startswith("<"):
            continue
        try:
            path = Path(name).resolve()
            if path.is_relative_to(own) or is_installed_path(path):
                continue
        except (OSError, ValueError):
            continue
        root = _project_root_above(path.parent)
        if root is not None:
            return root
    return None


def _invocation_project_root() -> Path | None:
    """The project an installed program -- pytest above all -- is working on.

    The one the cwd is in. Failing that, under pytest, the one the tests
    belong to, so ``pytest proj/tests`` typed from the directory above reads
    ``proj/pyproject.toml`` and caches where ``python -m pytest`` does.

    Only a project BELOW the cwd: a test that changes into a scratch
    directory keeps the scratch directory, as it always has.
    """
    root = _cwd_project_root()
    if root is None and _running_pytest():
        below = _calling_code_project_root()
        try:
            if below is not None and below.is_relative_to(Path.cwd().resolve()):
                root = below
        except (OSError, ValueError):
            pass
    return root


def project_anchor() -> Path:
    """The directory cash treats as "here" (see `_own_project_anchor`).

    A pool worker -- multiprocessing, ``concurrent.futures``, joblib -- answers
    as the process that started it, so both use one cache directory.
    """
    return _launch()[0]


def _own_project_anchor() -> Path:
    """The directory cash treats as "here" -- for the DEFAULT cache location
    and for finding ``pyproject.toml``.

    Not ``os.getcwd()``, which would make the cache a property of where you
    are standing rather than of what you are running: the same script run
    from another directory -- a cron job, a CI step, a colleague -- would
    silently build a second cache, and ``[tool.cash] cache_dir`` would move
    with it.

    The anchor walks up from the RUNNING SCRIPT to its project root, so
    ``python /srv/etl/run.py`` uses the same cache from anywhere on the machine.
    A directory holding a cache cash made (``.cash/CACHE_VERSION``) counts as
    a project root on that walk, below the home directory, so a script in
    ``proj/scratch`` uses ``proj/.cash`` when ``proj`` has no other marker.
    Without a script (a notebook, a REPL) or without a project marker above it,
    the answer is the cwd or the script's own directory respectively -- both
    stable for the case they describe.

    When the program itself is INSTALLED code -- ``pytest``, ``cash``, a
    ``python -m`` module in site-packages -- there is no script of the user's to
    anchor to, but there is usually a project the user is standing in, and the
    code being cached is that project's. So it walks up from the cwd instead.
    That puts a test suite's cache beside its project whichever way ``pytest``
    was typed and from whichever subdirectory, instead of one per-user cache
    shared by every project on the machine.
    """
    start = _running_script_dir()
    if start is None:
        # An xdist worker is ``python -c``, not installed code, and anchored
        # to the cwd while the pytest that started it anchored to the project:
        # one run, two caches.
        if _running_installed_code() or _running_pytest():
            root = _invocation_project_root()
            if root is not None:
                return root
        return Path.cwd()
    return _project_root_above(start, existing_cache=True) or start


def _running_console_script() -> str | None:
    """The name of the installed entry point being run, if that is what this is.

    A ``[project.scripts]`` console script lives in the interpreter's own
    ``bin`` / ``Scripts`` directory, so it has no project to anchor to and
    ``_running_script_dir`` correctly returns None for it -- leaving it on the
    cwd, where a `pip install`ed tool would drop a fresh ``.cash`` in every
    directory it is run from and never reuse one.

    Detected from ``sys.argv[0]`` rather than from the absence of an anchor,
    because that absence also covers a notebook, a REPL and ``python -c``,
    where the cwd is the right answer and always was.

    Deliberately NOT ``python -m tool``: its ``argv[0]`` is a module path
    inside site-packages, so it looks similar, but the invocation is a
    developer standing in a project far more often than it is an installed
    tool -- ``python -m pytest`` most of all. That shape keeps today's
    behaviour.

    Generic on purpose: it names ANY launcher in the script directory,
    ``pytest`` and ``cash`` included. Deciding what that means is
    ``installed_entry_point_cache_dir``'s job.
    """
    argv0 = sys.argv[0] if sys.argv else None
    if not argv0:
        return None
    try:
        path = Path(argv0).resolve()
    except OSError:
        return None
    script_dirs = {Path(sys.prefix) / d for d in ("bin", "Scripts")}
    script_dirs |= {Path(sys.base_prefix) / d for d in ("bin", "Scripts")}
    if path.parent not in script_dirs:
        return None
    name = re.sub(r"[^A-Za-z0-9._-]", "-", path.stem).strip("-.")
    return name or None


def _running_installed_module_name() -> str | None:
    """The top-level package of ``python -m <installed module>``, or None.

    The same tool as its console script, launched the other way, so cron's
    ``python -m nightly`` uses the per-user cache ``nightly`` itself uses
    rather than a fresh ``.cash`` wherever it was started. Inside
    a project it anchors to the project like any installed code; this name is
    only asked for outside one.
    """
    if interactive_shell_is_running() or not _running_installed_module():
        return None
    if _pool_launcher_main():
        top = _argv_installed_module_name() or ""
    else:
        spec = getattr(sys.modules.get("__main__"), "__spec__", None)
        top = (getattr(spec, "name", "") or "").split(".", 1)[0]
    name = re.sub(r"[^A-Za-z0-9._-]", "-", top).strip("-.")
    return name or None


def per_user_cache_root() -> Path:
    """The platform's own place for caches, where a cache survives ``cd``."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")  # not APPDATA: caches do not roam
        if base:
            return Path(base) / "cash"
        return Path.home() / "AppData" / "Local" / "cash"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "cash"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "cash"
    return Path.home() / ".cache" / "cash"


def installed_entry_point_cache_dir() -> Path | None:
    """Where an installed console script should cache, or None if not one
    (see `_own_installed_entry_point_cache_dir`); a pool worker answers as the
    process that started it."""
    return _launch()[1]


def _own_installed_entry_point_cache_dir() -> Path | None:
    """Where an installed console script should cache, or None if not one.

    Per tool, under the platform cache root, so two installed tools do not
    share one directory and neither inherits the other's eviction pressure.

    Only reached when nothing else claimed ``cache_dir``: an explicit setting
    of any kind wins, and so does a project ``pyproject.toml`` found by walking
    up from the cwd -- which is how a tool run inside a project that declares
    ``[tool.cash] cache_dir`` still caches beside that project's code. The
    per-user location is the answer for "nothing here claims this run", not a
    blanket override.

    Two refinements:

    * **Never for ``cash`` itself.** cash's own CLI is an installed console
      script too, and a per-user ``…/cash/cash`` is a cache nothing writes to.
      The CLI resolves like the context it is run in; ``--tool NAME`` reaches
      an installed tool's cache.
    * **Not inside a project.** ``pytest`` is a console script as well, and
      must not take every project's test suite into one per-user cache. Any
      launcher run inside a project anchors to that project instead (see
      ``project_anchor``); the per-user location is for a tool run from
      somewhere no project claims -- a home directory, a scratch directory, a
      drive root.
    """
    name = _running_console_script() or _running_installed_module_name()
    if name is None or name.lower() == "cash":
        return None
    if _invocation_project_root() is not None:
        return None
    try:
        return per_user_cache_root() / name
    except (OSError, RuntimeError):  # no home directory to speak of
        return None


def default_project_config_path(anchor: Path | None = None) -> Path | None:
    """Walk upward from the project anchor (or *anchor*) to find a ``pyproject.toml``.

    The first directory containing one wins. None if we never find one (a
    standalone script with no project structure).
    """
    if anchor is None:
        anchor = project_anchor()
    for d in [anchor, *anchor.parents]:
        candidate = d / "pyproject.toml"
        if candidate.exists():
            return candidate
    return None


def default_user_config_path() -> Path:
    """The XDG-spec user config location."""
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "cash" / "config.toml"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "cash" / "config.toml"
    return Path.home() / ".config" / "cash" / "config.toml"


#: The cache folders this process announced (CACHE-DIR-NEW): once each.
_ANNOUNCED: set[str] = set()


def _existing_cache_near(cache_dir: str) -> str | None:
    """A cache cash made (``.cash/CACHE_VERSION``) in the cwd or a folder above
    it, or above *cache_dir*: the one this run probably meant to use.

    A few ``stat`` calls: ancestors only, up to the home directory, and never
    the filesystem root.
    """
    from .backends.cache_dir import VERSION_FILENAME

    own = Path(cache_dir).resolve()
    home = _home()
    starts = [own.parent]
    try:
        starts.append(Path.cwd().resolve())
    except OSError:
        pass
    for start in starts:
        for d in [start, *start.parents]:
            if d.parent == d or (home is not None and d != home and home.is_relative_to(d)):
                break
            candidate = d / ".cash"
            if candidate != own and (candidate / VERSION_FILENAME).is_file():
                return str(candidate)
    return None


def announce_new_cache_dir(cache_dir: str) -> None:
    """Say, once per folder and process, that cash just created its default
    cache folder, and where (CACHE-DIR-NEW).

    A run that lands somewhere new starts cold, which looks exactly like a
    cache that does not work. Naming the folder, and an existing cache nearby,
    makes that visible. Not in an interactive shell: a notebook's cache is the
    folder it runs in, which is what its user expects.
    """
    key = os.path.normcase(os.path.abspath(cache_dir))
    if key in _ANNOUNCED or interactive_shell_is_running():
        return
    _ANNOUNCED.add(key)
    try:
        shown = os.path.relpath(cache_dir)
        if shown.startswith(os.pardir):
            shown = os.path.abspath(cache_dir)
    except ValueError:  # another drive
        shown = os.path.abspath(cache_dir)
    script_dir = _running_script_dir()
    beside_script = (
        script_dir is not None
        and Path(cache_dir).resolve().parent == script_dir
        and _project_root_above(script_dir) is None
    )
    reason = " (no project found above the script)" if beside_script else ""
    nearby = _existing_cache_near(cache_dir)
    # The cache found nearby goes in the first sentence: the one-line form
    # keeps only that, and it is the line's point.
    seen = f"; {nearby} exists" if nearby else ""
    fix = (
        "set CASH_CACHE_DIR or [tool.cash] cache_dir to use it."
        if nearby
        else "set CASH_CACHE_DIR or [tool.cash] cache_dir to cache elsewhere."
    )
    from .diagnostics import warn_diagnostic
    from .exceptions import CashWarning

    try:
        warn_diagnostic(CashWarning, "CACHE-DIR-NEW", f"new cache at {shown}{reason}{seen}.", fix)
    except Exception:  # a notice must not become the failure
        logger.debug("Could not announce %s", cache_dir, exc_info=True)
