"""``tools/dev_env.py``: the parts of the environment check that need no build.

The full ``create`` / ``check`` / ``clean`` cycle installs packages and
compiles the C extension, so it is run by hand; these pin the decisions it
rests on: which cash counts as this checkout's, when the disk is too full,
when the extension is stale, and that ``clean`` never removes a directory it
did not make.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tools import dev_env


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


# -- which cash is this checkout's -------------------------------------------


def test_a_file_under_the_checkouts_src_is_within_it(tmp_path):
    root = tmp_path / "wt" / "src" / "cash"
    root.mkdir(parents=True)
    assert dev_env.is_within(root / "__init__.py", root)
    assert dev_env.is_within(root, root)


def test_a_sibling_checkout_with_a_shared_prefix_is_not_within(tmp_path):
    (tmp_path / "wt" / "src" / "cash").mkdir(parents=True)
    (tmp_path / "wt2" / "src" / "cash").mkdir(parents=True)
    assert not dev_env.is_within(tmp_path / "wt2" / "src" / "cash" / "__init__.py", tmp_path / "wt" / "src" / "cash")


@pytest.mark.skipif(not hasattr(os, "symlink") or os.name == "nt", reason="needs unprivileged symlinks")
def test_a_link_into_the_checkout_counts_as_the_checkout(tmp_path):
    real = tmp_path / "real" / "src" / "cash"
    real.mkdir(parents=True)
    (tmp_path / "link").symlink_to(tmp_path / "real")
    assert dev_env.is_within(tmp_path / "link" / "src" / "cash" / "__init__.py", real)


@pytest.mark.skipif(os.name == "nt", reason="venv pythons are copies, not links, on Windows")
def test_two_venv_pythons_linking_one_interpreter_are_different_pythons(tmp_path):
    for venv in ("a", "b"):
        (tmp_path / venv).mkdir()
        (tmp_path / venv / "python").symlink_to(sys.executable)
    assert not dev_env.same_file(tmp_path / "a" / "python", tmp_path / "b" / "python")
    assert dev_env.same_file(tmp_path / "a" / "python", tmp_path / "a" / ".." / "a" / "python")


# -- disk ---------------------------------------------------------------------


def test_a_disk_below_the_threshold_is_a_problem_naming_both_numbers(tmp_path):
    problem = dev_env.disk_problem(tmp_path, 5, free=1.25)
    assert problem is not None
    assert "1.2 GB free" in problem and "5 GB" in problem and "--min-free-gb" in problem


def test_a_disk_at_or_above_the_threshold_is_fine(tmp_path):
    assert dev_env.disk_problem(tmp_path, 5, free=5.0) is None
    assert dev_env.disk_problem(tmp_path, 0, free=0.0) is None


def test_the_measured_free_space_is_used_without_an_override(tmp_path):
    assert dev_env.disk_problem(tmp_path, 0) is None
    assert dev_env.disk_problem(tmp_path, 10**9) is not None


# -- C extension freshness ----------------------------------------------------


def test_a_source_edited_after_the_build_is_stale(tmp_path):
    built, older, newer = tmp_path / "_cwatch.so", tmp_path / "a.c", tmp_path / "b.h"
    for f in (older, built, newer):
        f.write_text("x", encoding="utf-8")
    os.utime(older, (1000, 1000))
    os.utime(built, (2000, 2000))
    os.utime(newer, (3000, 3000))
    assert dev_env.stale_sources(built, [older, newer]) == [newer]


def test_no_extension_is_expected_before_python_3_12_or_when_switched_off(monkeypatch):
    monkeypatch.delenv("CASH_PURE_PYTHON", raising=False)
    assert dev_env.extension_expected("3.11.9") is not None
    assert dev_env.extension_expected("3.13.1") is None
    monkeypatch.setenv("CASH_PURE_PYTHON", "1")
    assert dev_env.extension_expected("3.13.1") == "CASH_PURE_PYTHON is set"


# -- network mounts -----------------------------------------------------------

MOUNTS = """\
/dev/vda / ext4 rw,relatime 0 0
server:/export /mnt/shared nfs4 rw 0 0
remote:files /mnt/attach/files fuse.rclone rw 0 0
"""


@pytest.mark.skipif(os.name == "nt", reason="/proc/mounts paths are POSIX")
@pytest.mark.parametrize(
    ("path", "fs_type", "network"),
    [
        ("/home/user/cash", "ext4", False),
        ("/mnt/shared/cash", "nfs4", True),
        ("/mnt/attach/files/x", "fuse.rclone", True),
        ("/mnt/sharedother", "ext4", False),
    ],
)
def test_the_longest_mount_holding_a_path_decides_its_file_system(path, fs_type, network):
    found = dev_env.mount_fs_type(path, MOUNTS)
    assert found == fs_type
    assert dev_env.is_network_fs(found) is network


# -- clean refuses what create did not make -----------------------------------


def _marker(target, **extra):
    data = {"tool": dev_env.MARKER_TOOL, "path": str(target), "worktree": True, **extra}
    (target / ".venv").mkdir(parents=True, exist_ok=True)
    (target / ".venv" / dev_env.MARKER_NAME).write_text(json.dumps(data), encoding="utf-8")
    return dev_env.read_marker(target)


def test_clean_refuses_a_directory_without_a_marker(tmp_path):
    (tmp_path / ".venv").mkdir()
    refusal = dev_env.clean_refusal(tmp_path, dev_env.read_marker(tmp_path), [str(tmp_path)], tmp_path / "repo")
    assert refusal is not None and "not made by this tool" in refusal


def test_clean_refuses_a_marker_from_another_tool(tmp_path):
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / dev_env.MARKER_NAME).write_text('{"tool": "other"}', encoding="utf-8")
    assert dev_env.read_marker(tmp_path) is None


def test_clean_refuses_an_environment_copied_from_elsewhere(tmp_path):
    target = tmp_path / "copy"
    target.mkdir()
    marker = _marker(target)
    marker["path"] = str(tmp_path / "original")
    refusal = dev_env.clean_refusal(target, marker, [str(target)], tmp_path / "repo")
    assert refusal is not None and "copied or moved" in refusal


def test_clean_refuses_a_directory_git_does_not_list_as_a_worktree(tmp_path):
    target = tmp_path / "wt"
    target.mkdir()
    refusal = dev_env.clean_refusal(target, _marker(target), [str(tmp_path / "elsewhere")], tmp_path / "repo")
    assert refusal is not None and "not a git worktree" in refusal


def test_clean_refuses_the_main_checkout_even_with_a_marker(tmp_path):
    marker = _marker(tmp_path)
    refusal = dev_env.clean_refusal(tmp_path, marker, [str(tmp_path)], tmp_path)
    assert refusal is not None and "main checkout" in refusal


def test_clean_accepts_a_worktree_it_made(tmp_path):
    target = tmp_path / "wt"
    target.mkdir()
    assert dev_env.clean_refusal(target, _marker(target), [str(target)], tmp_path / "repo") is None


def test_clean_on_a_foreign_directory_removes_nothing(tmp_path, capsys):
    (tmp_path / ".venv").mkdir()
    keep = tmp_path / ".venv" / "keep.txt"
    keep.write_text("x", encoding="utf-8")
    assert dev_env.main(["clean", str(tmp_path)]) == 2
    assert keep.is_file()
    assert "refusing" in capsys.readouterr().err


def test_clean_removes_a_real_worktree_and_the_branch_it_made(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-q", cwd=repo)
    _git("-c", "user.name=n", "-c", "user.email=e@x", "commit", "-q", "--allow-empty", "-m", "init", cwd=repo)
    target = tmp_path / "wt"
    _git("worktree", "add", "-q", "-b", "feature", str(target), cwd=repo)
    marker = _marker(target.resolve(), branch="feature", created_branch=True)
    assert dev_env.clean_refusal(target.resolve(), marker, dev_env.worktree_paths(repo), repo) is None

    monkeypatch.setattr(dev_env, "REPO_ROOT", repo)
    assert dev_env.main(["clean", str(target)]) == 0
    assert not target.exists()
    branches = subprocess.run(["git", "branch"], cwd=repo, capture_output=True, text=True, check=True).stdout
    assert "feature" not in branches
