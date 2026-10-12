"""Creating the default cache folder says where it is (CACHE-DIR-NEW).

A script that lands somewhere new starts with an empty cache, which looks
exactly like a cache that does not work: a cold run, every time, and nothing
said. The first time cash creates its default ``.cash`` it names the folder,
and an existing cache in the current directory or above, which the run
probably meant to use. A folder the user chose is where they expect it, so
nothing is said then.

Fresh interpreters: what is announced is what a process creates.
"""

from __future__ import annotations

import textwrap

from tests._scripts import run_python

_SCRIPT = textwrap.dedent(
    """
    import cash

    @cash.cache
    def work(n):
        return n * 2

    print(work(21), work(22))
    """
)


def _notices(stderr: str) -> list[str]:
    return [line for line in stderr.splitlines() if "CACHE-DIR-NEW" in line]


def _layout(tmp_path):
    proj = tmp_path / "proj"
    outside = tmp_path / "outside"
    proj.mkdir()
    outside.mkdir()
    (proj / "job.py").write_text(_SCRIPT, encoding="utf-8")
    (outside / "job.py").write_text(_SCRIPT, encoding="utf-8")
    return proj, outside


def test_the_first_run_names_the_new_folder_once(tmp_path):
    proj, _ = _layout(tmp_path)
    first = run_python("job.py", cwd=proj, cache_dir=None)
    notices = _notices(first.stderr)
    assert len(notices) == 1, first.stderr
    assert "new cache at .cash" in notices[0] and "no project found above the script" in notices[0]
    assert (proj / ".cash").is_dir()

    again = run_python("job.py", cwd=proj, cache_dir=None)
    assert _notices(again.stderr) == [], "an existing folder is not new"


def test_an_existing_cache_nearby_is_named(tmp_path):
    """``cd proj && python ../outside/job.py``: the script's own folder gets a
    new cache, while the one in the current directory holds the results."""
    proj, outside = _layout(tmp_path)
    run_python("job.py", cwd=proj, cache_dir=None)
    run = run_python(outside / "job.py", cwd=proj, cache_dir=None)
    notices = _notices(run.stderr)
    assert len(notices) == 1, run.stderr
    assert str(outside / ".cash") in notices[0]
    assert f"{proj / '.cash'} exists" in run.stderr, run.stderr


def test_a_folder_the_user_chose_is_not_announced(tmp_path):
    proj, _ = _layout(tmp_path)
    by_env = run_python("job.py", cwd=proj, cache_dir=tmp_path / "chosen")
    assert _notices(by_env.stderr) == [], by_env.stderr
    assert (tmp_path / "chosen").is_dir(), "control: the run must have created its folder"

    (proj / "pyproject.toml").write_text('[tool.cash]\ncache_dir = "var/cache"\n', encoding="utf-8")
    by_config = run_python("job.py", cwd=proj, cache_dir=None)
    assert _notices(by_config.stderr) == [], by_config.stderr
    assert (proj / "var" / "cache").is_dir(), "control: the run must have created its folder"
