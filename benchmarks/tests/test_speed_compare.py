"""speed_compare: verdicts from paired runs, and the worktree + PYTHONPATH swap."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from benchmarks import speed_compare as sc


def _run(**ratios):
    return {"meta": {}, "rows": [{"name": n, "ratio": r, "notes": [], "skipped": None} for n, r in ratios.items()]}


def _verdicts(a_runs, b_runs, threshold=0.2):
    return {v.name: v for v in sc.compare(a_runs, b_runs, threshold)}


def test_a_ratio_up_in_every_round_is_a_regression():
    v = _verdicts([_run(x=2.0), _run(x=2.2)], [_run(x=3.0), _run(x=2.9)])["x"]
    assert v.verdict == "regression"
    assert v.change == pytest.approx((1.5 + 2.9 / 2.2) / 2)


def test_a_round_that_disagrees_is_not_a_regression():
    v = _verdicts([_run(x=2.0), _run(x=2.0), _run(x=2.0)], [_run(x=3.0), _run(x=1.9), _run(x=3.0)])["x"]
    assert v.change > 1.2 and v.verdict == "same"


def test_small_moves_and_improvements():
    v = _verdicts([_run(a=1.0, b=4.0)], [_run(a=1.1, b=2.0)])
    assert v["a"].verdict == "same"
    assert v["b"].verdict == "improved"


def test_missing_skipped_and_broken_rows():
    a = _run(x=1.0, y=1.0, z=1.0)
    b = _run(x=1.0, w=1.0, z=1.0)
    b["rows"][2]["skipped"] = "error: KeyError: 'k'"
    v = _verdicts([a], [b])
    assert v["y"].verdict == "only in A" and v["w"].verdict == "only in B"
    assert v["z"].verdict == "broken"
    text = sc.format_verdicts(list(v.values()), "a", "b", 0.2)
    assert "1 regressed: z" in text


def test_from_json_exit_code(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(_run(x=1.0)), encoding="utf-8")
    b.write_text(json.dumps(_run(x=1.5)), encoding="utf-8")
    assert sc.main(["--from-json", str(a), str(b)]) == 1
    assert sc.main(["--from-json", str(a), str(b), "--threshold", "0.6"]) == 0


FAKE_SPEED_SET = textwrap.dedent(
    """
    import argparse, json, sys
    from pathlib import Path
    import cash
    p = argparse.ArgumentParser()
    p.add_argument("--repeats"); p.add_argument("--json"); p.add_argument("-q", action="store_true")
    p.add_argument("--only", nargs="+")
    a = p.parse_args()
    meta = {"cash_path": str(Path(cash.__file__).resolve().parent)}
    rows = [{"name": "s", "ratio": cash.RATIO, "notes": [], "skipped": None}]
    Path(a.json).write_text(json.dumps({"meta": meta, "rows": rows}), encoding="utf-8")
    """
)


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def two_commit_repo(tmp_path):
    repo = tmp_path / "repo"
    pkg = repo / "src" / "cash"
    pkg.mkdir(parents=True)
    _git(repo, "init", "-q")
    for ratio in (2.0, 3.0):
        (pkg / "__init__.py").write_text(f"RATIO = {ratio}\n", encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", f"ratio {ratio}")
    fake = tmp_path / "fake_speed_set.py"
    fake.write_text(FAKE_SPEED_SET, encoding="utf-8")
    return repo, fake


def test_each_ref_runs_against_its_own_src_and_worktrees_are_removed(two_commit_repo, monkeypatch, capsys):
    repo, fake = two_commit_repo
    monkeypatch.setattr(sc, "repo_root", lambda: repo)
    code = sc.main(["HEAD~1", "HEAD", "--rounds", "2", "--speed-set", str(fake)])
    out = capsys.readouterr()
    assert code == 1, out.err
    assert "REGRESSION" in out.out and "2.00x" in out.out and "3.00x" in out.out
    # A B, then B A
    assert [line.split(": ")[1] for line in out.err.splitlines() if line.startswith("round")] == [
        "A = HEAD~1",
        "B = HEAD",
        "B = HEAD",
        "A = HEAD~1",
    ]
    worktrees = subprocess.run(["git", "worktree", "list"], cwd=repo, capture_output=True, text=True).stdout
    assert len(worktrees.strip().splitlines()) == 1


def test_worktree_means_the_checkout_on_disk(two_commit_repo, monkeypatch, capsys):
    repo, fake = two_commit_repo
    (repo / "src" / "cash" / "__init__.py").write_text("RATIO = 2.0\n", encoding="utf-8")
    monkeypatch.setattr(sc, "repo_root", lambda: repo)
    assert sc.main(["HEAD", sc.WORKTREE, "--rounds", "1", "--speed-set", str(fake)]) == 0
    assert "improved" in capsys.readouterr().out


def test_an_unknown_ref_is_a_usage_error(two_commit_repo, monkeypatch):
    repo, fake = two_commit_repo
    monkeypatch.setattr(sc, "repo_root", lambda: repo)
    assert sc.main(["no-such-ref", "HEAD", "--speed-set", str(fake)]) == 2


def test_the_real_speed_set_runs_from_this_checkout():
    assert sc.SPEED_SET == Path(__file__).resolve().parents[1] / "speed_set.py"
    assert sys.executable
