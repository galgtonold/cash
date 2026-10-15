"""``tools/test_selection/impacted.py`` picks the tests a diff can affect.

Each test builds a tiny git repo and a synthetic coverage map by hand, so the
selection is checked against known coverage: a changed line selects the tests
that ran it, import-time lines and new code widen as documented, a map built at
an older commit is read through ``git diff``, and node ids with spaces and
brackets reach pytest intact.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.test_selection import impacted
from tools.test_selection.impacted import (
    CoverageMap,
    Hunk,
    Selector,
    convert_baseline,
    existing_id,
    lines_to_numbits,
    map_back,
    parse_diff,
    parse_range,
    write_map,
)

MOD = """\
import os

CONST = 1


def f(x):
    y = x + 1
    return y


def g(x):
    if x:
        return 1
    return 2


class K:
    def m(self):
        return 3
"""
# line numbers in MOD: f's body 7-8, g's body 12-14, K.m's body 19

TEST_F = "from cash.mod import f\n\n\ndef test_a():\n    assert f(1) == 2\n"
TEST_K = "from cash import mod\n\n\ndef test_k():\n    assert mod.K().m() == 3\n"
TEST_G = "def test_g():\n    pass\n"
ID_A, ID_G, ID_K = "tests/test_f.py::test_a", "tests/test_g.py::test_g", "tests/test_k.py::test_k"


def sh(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def commit(root: Path, message: str) -> str:
    sh(root, "add", "-A")
    sh(root, "commit", "-q", "-m", message)
    return sh(root, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "src" / "cash").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "cash" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "cash" / "mod.py").write_text(MOD, encoding="utf-8")
    (root / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (root / "tests" / "conftest.py").write_text("", encoding="utf-8")
    (root / "tests" / "test_f.py").write_text(TEST_F, encoding="utf-8")
    (root / "tests" / "test_g.py").write_text(TEST_G, encoding="utf-8")
    (root / "tests" / "test_k.py").write_text(TEST_K, encoding="utf-8")
    sh(root, "init", "-q", "-b", "main")
    commit(root, "start")
    return root


def make_map(root: Path, commit_sha: str, cover: dict[str, list[int]]) -> Path:
    """A map where each test ran its listed lines of src/cash/mod.py, plus the import-time lines."""
    path = root.parent / "map.sqlite"
    rows = [(nid, "src/cash/mod.py", lines_to_numbits([1, 3, 6, 11, 17, 18, *lines])) for nid, lines in cover.items()]
    write_map(path, {"commit": commit_sha}, {nid: ("passed", 1.0) for nid in cover}, rows)
    return path


COVER = {ID_A: [7, 8], ID_G: [12, 14], ID_K: [19]}


def select(root: Path, map_path: Path | None, spec: str = "HEAD") -> tuple[list[str], list[str]]:
    cmap = CoverageMap(map_path) if map_path else None
    sel = Selector(root, parse_range(root, spec), cmap).run()
    return sel.ids(), sel.notes


def edit(root: Path, old: str, new: str, path: str = "src/cash/mod.py") -> None:
    p = root / path
    text = p.read_text(encoding="utf-8")
    assert text.count(old) == 1
    p.write_text(text.replace(old, new), encoding="utf-8")


# --- diff parsing ------------------------------------------------------------


def test_hunk_headers_default_a_missing_count_to_one():
    text = (
        "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n"
        "@@ -3 +3 @@\n-a\n+b\n@@ -10,0 +11,2 @@\n+c\n+d\n@@ -20,3 +21,0 @@\n-e\n-f\n-g\n"
    )
    (fd,) = parse_diff(text)
    assert fd.old_path == fd.new_path == "x.py"
    assert fd.hunks == [Hunk(3, 1, 3, 1), Hunk(10, 0, 11, 2), Hunk(20, 3, 21, 0)]
    assert list(fd.hunks[1].old_lines()) == [] and list(fd.hunks[1].new_lines()) == [11, 12]


def test_added_deleted_renamed_and_quoted_paths():
    text = (
        "diff --git a/new.py b/new.py\nnew file mode 100644\n--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+x\n"
        "diff --git a/old.py b/old.py\ndeleted file mode 100644\n--- a/old.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n"
        "diff --git a/a.py b/b.py\nsimilarity index 100%\nrename from a.py\nrename to b.py\n"
        'diff --git "a/sp ace\\"q.py" "b/sp ace\\"q.py"\n--- "a/sp ace\\"q.py"\n+++ "b/sp ace\\"q.py"\n@@ -1 +1 @@\n'
    )
    new, old, renamed, quoted = parse_diff(text)
    assert (new.old_path, new.new_path) == (None, "new.py")
    assert (old.old_path, old.new_path) == ("old.py", None)
    assert (renamed.old_path, renamed.new_path, renamed.hunks) == ("a.py", "b.py", [])
    assert quoted.path == 'sp ace"q.py'


@pytest.mark.parametrize(
    ("hunks", "line", "expected"),
    [
        ([], 5, 5),
        ([Hunk(0, 0, 1, 3)], 4, 1),  # three lines inserted at the top
        ([Hunk(0, 0, 1, 3)], 2, None),  # an inserted line has no old number
        ([Hunk(1, 2, 0, 0)], 1, 3),  # two lines deleted from the top
        ([Hunk(5, 1, 5, 1)], 5, None),  # a rewritten line
        ([Hunk(5, 1, 5, 1)], 6, 6),
        ([Hunk(2, 0, 3, 1), Hunk(8, 2, 8, 0)], 9, 10),  # +1 then -2
        ([Hunk(2, 0, 3, 1), Hunk(8, 2, 8, 0)], 4, 3),
    ],
)
def test_map_back_carries_a_line_to_the_old_side(hunks, line, expected):
    assert map_back(hunks, line) == expected


# --- selection with a map at the diff base -----------------------------------


def test_a_changed_line_selects_only_the_tests_that_ran_it(repo):
    map_path = make_map(repo, sh(repo, "rev-parse", "HEAD"), COVER)
    edit(repo, "    y = x + 1\n", "    y = x + 2\n")
    ids, _ = select(repo, map_path)
    # every test also "ran" the import-time lines; they must not count
    assert ids == [ID_A]


def test_a_module_level_change_widens_to_every_test_of_the_file(repo):
    map_path = make_map(repo, sh(repo, "rev-parse", "HEAD"), COVER)
    edit(repo, "CONST = 1\n", "CONST = 2\n")
    ids, notes = select(repo, map_path)
    assert ids == [ID_A, ID_G, ID_K]
    assert any("whole file" in n for n in notes)


def test_a_changed_signature_counts_as_its_function(repo):
    map_path = make_map(repo, sh(repo, "rev-parse", "HEAD"), COVER)
    edit(repo, "def g(x):\n", "def g(x, z=0):\n")
    assert select(repo, map_path)[0] == [ID_G]


def test_new_code_widens_to_the_enclosing_function_and_says_so(repo):
    map_path = make_map(repo, sh(repo, "rev-parse", "HEAD"), COVER)
    edit(repo, "    return 2\n", "    print(x)\n    return 2\n")
    ids, notes = select(repo, map_path)
    assert ids == [ID_G]
    assert any("enclosing function/class (g)" in n for n in notes), notes


def test_a_changed_line_no_test_runs_widens_to_its_function(repo):
    map_path = make_map(repo, sh(repo, "rev-parse", "HEAD"), COVER)
    edit(repo, "        return 1\n", "        return 11\n")  # line 13: COVER never ran it
    assert select(repo, map_path)[0] == [ID_G]


def test_a_new_method_widens_to_its_class(repo):
    map_path = make_map(repo, sh(repo, "rev-parse", "HEAD"), COVER)
    edit(repo, "        return 3\n", "        return 3\n\n    def n(self):\n        return 4\n")
    assert select(repo, map_path)[0] == [ID_K]


def test_committed_and_uncommitted_changes_both_count(repo):
    base = sh(repo, "rev-parse", "HEAD")
    map_path = make_map(repo, base, COVER)
    edit(repo, "        return 3\n", "        return 33\n")
    commit(repo, "change m")
    edit(repo, "    y = x + 1\n", "    y = x + 2\n")
    assert select(repo, map_path, base)[0] == [ID_A, ID_K]
    sel = Selector(repo, parse_range(repo, base, worktree=False), CoverageMap(map_path)).run()
    assert sel.ids() == [ID_K]


# --- a map built at another commit ----------------------------------------------


def test_lines_are_carried_back_to_an_older_map(repo):
    map_path = make_map(repo, sh(repo, "rev-parse", "HEAD"), COVER)
    edit(repo, "import os\n", "import os\nimport re\nimport sys\nimport json\n")
    base = commit(repo, "three more imports")
    edit(repo, "    y = x + 1\n", "    y = x + 2\n")  # line 10 now, line 7 in the map
    ids, notes = select(repo, map_path, base)
    assert ids == [ID_A]
    assert any("line numbers are carried over" in n for n in notes)


def test_lines_changed_since_the_map_widen_to_the_file_and_are_counted(repo):
    map_path = make_map(repo, sh(repo, "rev-parse", "HEAD"), COVER)
    edit(repo, "    y = x + 1\n", "    y = x + 3\n")
    base = commit(repo, "rewrite the line after the map")
    edit(repo, "    y = x + 3\n", "    y = x + 4\n")
    ids, notes = select(repo, map_path, base)
    assert ids == [ID_A, ID_G, ID_K]
    assert any(n.startswith("1 changed line(s) could not be placed") for n in notes), notes


def test_a_map_from_a_commit_this_clone_lacks_widens_to_file_level(repo):
    map_path = make_map(repo, "0123456789" * 4, COVER)
    edit(repo, "    y = x + 1\n", "    y = x + 2\n")
    ids, notes = select(repo, map_path)
    assert ids == [ID_A, ID_G, ID_K]
    assert any("not in this clone" in n for n in notes)


def test_a_stale_parameter_id_runs_its_whole_function(repo):
    (repo / "tests" / "test_p.py").write_text(
        "import pytest\n\n\n@pytest.mark.parametrize('v', [1, 2])\ndef test_p(v):\n    pass\n", encoding="utf-8"
    )
    assert existing_id(repo, "tests/test_p.py::test_p[3]") == "tests/test_p.py::test_p"
    assert existing_id(repo, "tests/test_p.py::test_gone[3]") is None


# --- fallbacks without the map's help ------------------------------------------


def test_without_a_map_the_tool_says_how_to_build_one(repo, tmp_path, capsys):
    code = impacted.main(["HEAD", "--repo", str(repo), "--map", str(tmp_path / "missing.sqlite")])
    assert code == 2
    err = capsys.readouterr().err
    assert "impacted.py build" in err and "--file-level" in err


def test_file_level_selects_the_tests_that_import_the_module(repo):
    edit(repo, "    y = x + 1\n", "    y = x + 2\n")
    assert select(repo, None)[0] == ["tests/test_f.py", "tests/test_k.py"]


def test_a_new_module_falls_back_to_its_importers_and_new_tests(repo):
    map_path = make_map(repo, sh(repo, "rev-parse", "HEAD"), COVER)
    (repo / "src" / "cash" / "fresh.py").write_text("def h():\n    return 1\n", encoding="utf-8")
    (repo / "tests" / "helper.py").write_text("from cash.fresh import h\n", encoding="utf-8")
    (repo / "tests" / "test_uses_helper.py").write_text("from tests.helper import h\n", encoding="utf-8")
    ids, _ = select(repo, map_path)
    # the untracked helper and the test that imports it
    assert ids == ["tests/test_uses_helper.py"]


def test_a_module_nothing_covers_or_imports_falls_back_to_its_name(repo):
    map_path = make_map(repo, sh(repo, "rev-parse", "HEAD"), COVER)
    (repo / "src" / "cash" / "widgets").mkdir()
    (repo / "src" / "cash" / "widgets" / "knob.py").write_text("X = 1\n", encoding="utf-8")
    (repo / "tests" / "test_widgets").mkdir()
    (repo / "tests" / "test_widgets" / "test_dials.py").write_text("def test_d():\n    pass\n", encoding="utf-8")
    ids, notes = select(repo, map_path)
    assert ids == ["tests/test_widgets"]
    assert any("name heuristic" in n for n in notes)


def test_changed_test_files_run_whole_and_a_conftest_runs_its_directory(repo):
    map_path = make_map(repo, sh(repo, "rev-parse", "HEAD"), COVER)
    edit(repo, "    pass\n", "    assert True\n", path="tests/test_g.py")
    assert select(repo, map_path)[0] == ["tests/test_g.py"]
    (repo / "tests" / "conftest.py").write_text("X = 1\n", encoding="utf-8")
    assert select(repo, map_path)[0] == ["tests"]


def test_a_c_source_selects_the_tests_that_reach_the_extension(repo):
    native_user = "from cash._native import cwatch as _cw\n\n\ndef q():\n    return _cw.f()\n"
    (repo / "src" / "cash" / "user.py").write_text(native_user, encoding="utf-8")
    (repo / "tests" / "test_n.py").write_text("import cash._native\n", encoding="utf-8")
    sha = commit(repo, "native")
    path = repo.parent / "map.sqlite"
    write_map(
        path,
        {"commit": sha},
        {},
        [
            ("tests/test_q.py::test_q", "src/cash/user.py", lines_to_numbits([1, 4, 5])),
            *[(nid, "src/cash/mod.py", lines_to_numbits(lines)) for nid, lines in COVER.items()],
        ],
    )
    (repo / "tests" / "test_q.py").write_text("def test_q():\n    pass\n", encoding="utf-8")
    commit(repo, "the test")
    (repo / "src" / "cash" / "_c").mkdir()
    (repo / "src" / "cash" / "_c" / "cwatch.c").write_text("int x;\n", encoding="utf-8")
    ids, notes = select(repo, path)
    assert ids == ["tests/test_n.py", "tests/test_q.py::test_q"]
    assert any("C source" in n for n in notes)


# --- ids reach pytest intact ------------------------------------------------------


def test_ids_with_spaces_and_brackets_run_as_recorded(repo, capfd):
    (repo / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (repo / "tests" / "test_odd.py").write_text(
        "import pytest\n\n\n@pytest.mark.parametrize('v', ['a b', 'c[d]', 'other'])\ndef test_odd(v):\n    pass\n",
        encoding="utf-8",
    )
    sha = commit(repo, "odd ids")
    odd = ["tests/test_odd.py::test_odd[a b]", "tests/test_odd.py::test_odd[c[d]]"]
    path = repo.parent / "map.sqlite"
    write_map(path, {"commit": sha}, {}, [(nid, "src/cash/mod.py", lines_to_numbits([7])) for nid in odd])
    edit(repo, "    y = x + 1\n", "    y = x + 2\n")

    out = repo.parent / "ids.txt"
    assert impacted.main(["HEAD", "--repo", str(repo), "--map", str(path), "-o", str(out)]) == 0
    assert out.read_text(encoding="utf-8").splitlines() == odd

    code = impacted.main(
        ["HEAD", "--repo", str(repo), "--map", str(path), "--run", "-n", "0", "--", "-q", "-p", "no:cacheprovider"]
    )
    captured = capfd.readouterr()
    assert code == 0, captured.out + captured.err
    assert "2 passed" in captured.out


def test_printed_ids_are_the_only_stdout(repo, capsys):
    map_path = make_map(repo, sh(repo, "rev-parse", "HEAD"), COVER)
    edit(repo, "    y = x + 1\n", "    y = x + 2\n")
    assert impacted.main(["HEAD", "--repo", str(repo), "--map", str(map_path)]) == 0
    captured = capsys.readouterr()
    assert captured.out == ID_A + "\n"
    assert "1 id(s) selected" in captured.err


def test_nothing_selected_runs_nothing(repo, capsys):
    map_path = make_map(repo, sh(repo, "rev-parse", "HEAD"), COVER)
    assert impacted.main(["HEAD", "--repo", str(repo), "--map", str(map_path), "--run"]) == 0
    assert "nothing to run" in capsys.readouterr().err


# --- the map file ----------------------------------------------------------------


def test_a_map_of_another_format_is_refused(tmp_path):
    path = tmp_path / "m.sqlite"
    write_map(path, {"commit": "x"}, {}, [])
    import sqlite3

    db = sqlite3.connect(path)
    db.execute("UPDATE meta SET value = '0' WHERE key = 'format_version'")
    db.commit()
    db.close()
    with pytest.raises(impacted.UsageError, match="rebuild"):
        CoverageMap(path)


def test_a_baseline_run_converts_into_a_map(repo, tmp_path):
    coverage = pytest.importorskip("coverage")
    out = tmp_path / "baseline"
    (out / "cov").mkdir(parents=True)
    data = coverage.CoverageData(basename=str(out / "cov" / ".coverage"))
    mod = str(repo / "src" / "cash" / "mod.py")
    data.set_context("")
    data.add_lines({mod: [1, 3]})  # before any test: not a test's
    for nid, lines in COVER.items():
        data.set_context(nid)
        data.add_lines({mod: lines})
    data.write()
    sha = sh(repo, "rev-parse", "HEAD")
    (out / "run_info.json").write_text(json.dumps({"commit": sha, "started": "now"}), encoding="utf-8")
    (out / "results.json").write_text(json.dumps({ID_A: {"outcome": "passed", "duration": 2.5}}), encoding="utf-8")
    map_path = tmp_path / "m.sqlite"
    convert_baseline(out, map_path, repo)
    cmap = CoverageMap(map_path)
    assert cmap.commit == sha
    assert cmap.files() == ["src/cash/mod.py"]
    assert cmap.tests_for("src/cash/mod.py", [7]) == {ID_A}
    assert cmap.tests_for("src/cash/mod.py", [1, 3]) == set()
    assert cmap.durations[ID_A] == 2.5


def test_the_cli_runs_as_a_script(repo, tmp_path):
    map_path = make_map(repo, sh(repo, "rev-parse", "HEAD"), COVER)
    edit(repo, "        return 3\n", "        return 4\n")
    script = Path(impacted.__file__)
    proc = subprocess.run(
        [sys.executable, str(script), "HEAD", "--repo", str(repo), "--map", str(map_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == [ID_K]
