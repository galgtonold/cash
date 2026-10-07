"""durations_diff: read JUnit times, save runs, flag tests > 2x AND > 1 s slower."""

from __future__ import annotations

import json

from benchmarks import durations_diff as dd


def _junit(path, times: dict[str, float], skipped=()):
    cases = []
    for tid, t in times.items():
        cls, name = tid.split("::")
        body = "<skipped/>" if tid in skipped else ""
        cases.append(f'<testcase classname="{cls}" name="{name}" time="{t}">{body}</testcase>')
    path.write_text(
        f'<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest">{"".join(cases)}</testsuite></testsuites>',
        encoding="utf-8",
    )
    return path


def test_both_thresholds_must_be_crossed():
    before = {"a": 0.2, "b": 3.0, "c": 3.0, "d": 1.0}
    after = {"a": 0.9, "b": 5.0, "c": 7.0, "d": 1.0, "new": 50.0}
    flagged = dd.diff(before, after, factor=2.0, min_seconds=1.0)
    # a: 4.5x but only 0.7 s; b: 2 s but 1.7x; new: no baseline.
    assert [s.test for s in flagged] == ["c"]


def test_reruns_keep_the_longest_time_and_skips_are_left_out(tmp_path):
    x1 = _junit(tmp_path / "1.xml", {"tests.m::t1": 1.0, "tests.m::t2": 0.5}, skipped={"tests.m::t2"})
    x2 = _junit(tmp_path / "2.xml", {"tests.m::t1": 2.5})
    assert dd.read_junit([x1, x2]) == {"tests.m::t1": 2.5}


def test_first_run_saves_a_baseline_and_the_second_compares(tmp_path, capsys):
    store = tmp_path / "store"
    run1 = _junit(tmp_path / "r1.xml", {"tests.m::slow": 1.0, "tests.m::ok": 1.0})
    run2 = _junit(tmp_path / "r2.xml", {"tests.m::slow": 4.0, "tests.m::ok": 1.1})
    assert dd.main([str(run1), "--store", str(store), "--label", "abc"]) == 0
    assert "becomes the baseline" in capsys.readouterr().out
    (saved,) = store.glob("durations-*-abc.json")
    assert json.loads(saved.read_text(encoding="utf-8"))["durations"]["tests.m::slow"] == 1.0
    assert dd.main([str(run2), "--store", str(store)]) == 1
    out = capsys.readouterr().out
    assert "1 slower" in out and "tests.m::slow" in out and "tests.m::ok" not in out.split("slower:")[1]
    # The second run is now the latest: running it again flags nothing.
    assert dd.main([str(run2), "--store", str(store), "--no-save"]) == 0
    assert len(list(store.glob("durations-*.json"))) == 2


def test_against_a_junit_file_and_warn_only(tmp_path):
    old = _junit(tmp_path / "old.xml", {"tests.m::t": 1.0})
    new = _junit(tmp_path / "new.xml", {"tests.m::t": 3.0})
    assert dd.main([str(new), "--against", str(old), "--no-save", "--store", str(tmp_path / "s")]) == 1
    assert dd.main([str(new), "--against", str(old), "--no-save", "--warn-only", "--store", str(tmp_path / "s")]) == 0
    assert not (tmp_path / "s").exists()


def test_unreadable_input_is_a_usage_error(tmp_path):
    bad = tmp_path / "bad.xml"
    bad.write_text("not xml", encoding="utf-8")
    assert dd.main([str(bad), "--store", str(tmp_path / "s")]) == 2
