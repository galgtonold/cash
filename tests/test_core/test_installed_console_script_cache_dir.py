"""An installed console script caches per user, not per directory.

CAS-104, reported blocking by a round-16 tester. A ``[project.scripts]`` entry
point lives in the interpreter's own ``bin``/``Scripts`` directory, so it has
no project to anchor to and fell back to the cwd -- which meant a `pip
install`ed tool dropped a fresh ``.cash`` in every directory it was run from
and never reused one. Anchoring inside the virtualenv would be worse: the cache
would land in site-packages, shared by every project using that environment and
wiped by a reinstall.

So: the platform's own cache root, one directory per tool
(``%LOCALAPPDATA%\\cash\\<tool>``, ``~/Library/Caches/cash/<tool>``,
``$XDG_CACHE_HOME/cash/<tool>``), and an explicit setting of any kind still
wins -- including a ``pyproject.toml`` found by walking up from the cwd, so a
tool run inside a project that declares ``[tool.cash] cache_dir`` still caches
beside that project's code.

These tests build a real distribution and `pip install` it into a throwaway
venv, because the shape under test IS the installed entry point: a script run
by path, or a function called in-process, does not reproduce it. That is slow
(one install), so the arms share the venv.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = [pytest.mark.core, pytest.mark.slow]

_PKG = "cashprobe"


def _write_distribution(root):
    """A tiny console-script package that prints the cache dir cash resolved."""
    src = root / "src" / _PKG
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "cli.py").write_text(textwrap.dedent("""
        import json
        import sys
        import time

        import cash
        from cash.config import get_config


        @cash.cache
        def slow(n):
            time.sleep(0.3)          # over the persistence floor
            return n * 2


        def main():
            if sys.argv[1:] == ["compute"]:
                print(json.dumps({"value": slow(21)}))
                return
            print(json.dumps({
                "cache_dir": str(get_config().cache_dir),
                "argv0": sys.argv[0],
            }))
    """), encoding="utf-8")
    (root / "pyproject.toml").write_text(textwrap.dedent(f"""
        [build-system]
        requires = ["hatchling<1.28"]
        build-backend = "hatchling.build"

        [project]
        name = "{_PKG}"
        version = "0.0.1"

        [project.scripts]
        {_PKG} = "{_PKG}.cli:main"

        [tool.hatch.build.targets.wheel]
        packages = ["src/{_PKG}"]
    """), encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def installed_tool(tmp_path_factory):
    """A venv with cash and a console-script package installed into it."""
    base = tmp_path_factory.mktemp("consolescript")
    venv = base / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True,
                   capture_output=True)
    bindir = venv / ("Scripts" if os.name == "nt" else "bin")
    python = bindir / ("python.exe" if os.name == "nt" else "python")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dist = _write_distribution(base / "dist")
    # `tomllib` is 3.11+ and cash has no required dependencies, so on 3.10 a
    # bare install cannot read a [tool.cash] section at all -- which is what
    # CONFIG-TOML-UNREADABLE now says out loud. The project-wins arm below is
    # about config PRECEDENCE, not about whether a parser exists, so give the
    # environment one.
    extra = ["tomli"] if sys.version_info < (3, 11) else []
    install = subprocess.run(
        [str(python), "-m", "pip", "install", "-q", repo_root, str(dist), *extra],
        capture_output=True, text=True,
    )
    if install.returncode != 0:
        pytest.skip(f"could not build the probe distribution:\n{install.stderr[-2000:]}")

    exe = bindir / (f"{_PKG}.exe" if os.name == "nt" else _PKG)
    assert exe.exists(), f"no console script at {exe}"
    return base, exe


def _run(exe, cwd, env=None):
    environ = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    environ.update(env or {})
    out = subprocess.run([str(exe)], cwd=str(cwd), capture_output=True,
                         text=True, env=environ)
    assert out.returncode == 0, out.stdout + out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_the_same_tool_uses_one_cache_from_any_directory(installed_tool, tmp_path):
    """THE BUG: a `.cash` per directory you happened to run the tool from."""
    _, exe = installed_tool
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()

    first = _run(exe, a)["cache_dir"]
    second = _run(exe, b)["cache_dir"]

    assert first == second, "the tool cached per directory"
    assert os.path.isabs(first)
    assert not os.path.exists(a / ".cash"), "a cache was left in the cwd"
    assert not os.path.exists(b / ".cash")
    assert _PKG in first, f"the location should name the tool: {first}"


def test_an_explicit_setting_still_wins(installed_tool, tmp_path):
    """The control: nothing here overrides what a user actually asked for."""
    _, exe = installed_tool
    workdir = tmp_path / "work"
    workdir.mkdir()
    wanted = str(tmp_path / "chosen")

    got = _run(exe, workdir, env={"CASH_CACHE_DIR": wanted})["cache_dir"]

    assert got == wanted


def test_a_project_that_claims_the_run_still_wins(installed_tool, tmp_path):
    """Run inside a project that declares [tool.cash], cache beside its code.

    This is the half that makes the per-user location safe to default to: a
    tool invoked in a project that has an opinion follows the project.
    """
    _, exe = installed_tool
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "0"\n\n'
        '[tool.cash]\ncache_dir = "build/.cash"\n',
        encoding="utf-8",
    )

    got = _run(exe, project)["cache_dir"]

    assert got == os.path.normpath(str(project / "build" / ".cash")), got


def test_a_plain_script_is_unaffected(installed_tool, tmp_path):
    """The other control: this must not reach an ordinary `python script.py`.

    That shape anchors to its project, which is CAS-84's fix and stays put.
    """
    base, _ = installed_tool
    bindir = base / "venv" / ("Scripts" if os.name == "nt" else "bin")
    python = bindir / ("python.exe" if os.name == "nt" else "python")

    project = tmp_path / "scripted"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "0"\n', encoding="utf-8")
    script = project / "run.py"
    script.write_text(
        "from cash.config import get_config\nprint(get_config().cache_dir)\n",
        encoding="utf-8")

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    environ = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    out = subprocess.run([str(python), str(script)], cwd=str(elsewhere),
                         capture_output=True, text=True, env=environ)
    assert out.returncode == 0, out.stderr

    assert out.stdout.strip() == os.path.normpath(str(project / ".cash")), out.stdout


def _private_user_cache(tmp_path):
    """Point the platform cache root somewhere private for this test."""
    root = tmp_path / "usercache"
    root.mkdir()
    return root, {
        "LOCALAPPDATA": str(root),        # Windows
        "XDG_CACHE_HOME": str(root),      # Linux
        "HOME": str(root),                # macOS: ~/Library/Caches
    }


def _cash_cli(base, *argv, cwd, env):
    """The INSTALLED `cash` console script -- the shape round 17 found broken."""
    bindir = base / "venv" / ("Scripts" if os.name == "nt" else "bin")
    exe = bindir / ("cash.exe" if os.name == "nt" else "cash")
    environ = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    environ.update(env)
    return subprocess.run([str(exe), *argv], cwd=str(cwd), capture_output=True,
                          text=True, env=environ)


def test_the_installed_cash_cli_agrees_with_python_m_cash(installed_tool, tmp_path):
    """Round 17, all five testers: `cash info` said `…/cash/cash`."""
    base, _ = installed_tool
    project = tmp_path / "project"
    (project / "sub").mkdir(parents=True)
    (project / "pyproject.toml").write_text('[project]\nname = "p"\nversion = "0"\n',
                                            encoding="utf-8")
    _, env = _private_user_cache(tmp_path)
    python = base / "venv" / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python")

    via_script = _cash_cli(base, "info", cwd=project / "sub", env=env)
    environ = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    environ.update(env)
    via_module = subprocess.run([str(python), "-m", "cash", "info"], cwd=str(project / "sub"),
                                capture_output=True, text=True, env=environ)

    def cache_line(out):
        return [ln for ln in out.stdout.splitlines() if "Cache dir" in ln][0].split(":", 1)[1].strip()

    assert via_script.returncode == 0, via_script.stderr
    assert cache_line(via_script) == os.path.normpath(str(project / ".cash"))
    assert cache_line(via_module) == cache_line(via_script)


def test_the_cli_reaches_a_tools_per_user_cache_by_name(installed_tool, tmp_path):
    """`--tool NAME` is how an operator gets at a cache the CLI cannot infer."""
    base, exe = installed_tool
    nowhere = tmp_path / "nowhere"
    nowhere.mkdir()
    root, env = _private_user_cache(tmp_path)

    environ = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    environ.update(env)
    ran = subprocess.run([str(exe), "compute"], cwd=str(nowhere), capture_output=True,
                         text=True, env=environ)
    assert ran.returncode == 0, ran.stdout + ran.stderr
    tool_dir = next(root.rglob(_PKG), None)
    assert tool_dir is not None and tool_dir.is_dir(), "the tool cached nowhere private"

    info = _cash_cli(base, "info", cwd=nowhere, env=env)
    assert _PKG in info.stdout, info.stdout

    inspect = _cash_cli(base, "inspect", "--tool", _PKG, cwd=nowhere, env=env)
    assert inspect.returncode == 0, inspect.stdout + inspect.stderr
    assert "slow" in inspect.stdout, inspect.stdout

    cleared = _cash_cli(base, "clear", "--tool", _PKG, cwd=nowhere, env=env)
    assert cleared.returncode == 0, cleared.stdout + cleared.stderr
    assert not tool_dir.exists()
