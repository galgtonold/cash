"""The speed set's harness: interleaved samples, medians, ratios, notebook sessions."""

from __future__ import annotations

import json

import pytest

from benchmarks import _speed_harness as h
from benchmarks import speed_set
from benchmarks._speed_scenarios import SCENARIOS


def _fixed(times: dict[str, list[float]]):
    """A sample function that hands out the given times in order, per side."""
    order = []

    def sample(side, ctx):
        order.append(side)
        t = times[side].pop(0)
        return {"": h.Timing(t, t)}

    return sample, order


def test_sides_alternate_after_a_warm_up_and_the_ratio_is_of_medians():
    sample, order = _fixed({"plain": [9.0, 1.0, 2.0, 3.0], "cash": [9.0, 4.0, 8.0, 6.0]})
    (row,) = h.measure(h.Scenario("toy", "g", "what", sample, clock="cpu"), repeats=3)
    assert order == ["plain", "cash", "plain", "cash", "cash", "plain", "plain", "cash"]
    assert row.plain_samples == [1.0, 2.0, 3.0] and row.cash_samples == [4.0, 8.0, 6.0]
    assert row.ratio == pytest.approx(6.0 / 2.0)
    # per-round ratios 4, 4, 2: deviations 0, 0, 2 from 4
    assert row.spread == 0.0


def test_a_scenario_whose_module_is_missing_is_skipped():
    sc = h.Scenario("toy", "g", "what", lambda side, ctx: {}, needs=("no_such_module_xyz",))
    (row,) = h.measure(sc, repeats=1)
    assert row.skipped == "needs no_such_module_xyz"
    assert "skipped" in h.format_table([row])


def test_each_metric_is_a_row():
    def sample(side, ctx):
        return {"early": h.Timing(1.0, 1.0), "late": h.Timing(2.0 if side == "cash" else 1.0, 1.0)}

    sc = h.Scenario("nb", "g", "w", sample, clock="cpu", metrics=(("early", "e"), ("late", "l")))
    rows = h.measure(sc, repeats=1)
    assert [(r.name, r.ratio) for r in rows] == [("nb/early", 1.0), ("nb/late", 2.0)]


def test_the_clock_is_cpu_only_when_it_is_fine_grained():
    assert h.default_clock() == ("cpu" if h.cpu_clock_step() < 1e-4 else "wall")


def test_a_notebook_session_runs_cells_with_and_without_cash(tmp_path):
    cells = ["a = 1", "b = a + 1"]
    plain = h.Notebook(cells, cash_on=False, workdir=tmp_path / "p")
    assert len(plain.run()) == 2 and plain.peek("b") == 2
    plain.close()
    on = h.Notebook(cells, cash_on=True, workdir=tmp_path / "c")
    on.run()
    assert on.peek("b") == 2
    assert "b" in on.magics.tracking_state.variable_lineage
    # The upstream check reads this session's notebook file.
    assert on.shell.user_ns["__vsc_ipynb_file__"] == str(on.path)
    on.close()


def test_a_failing_cell_fails_the_sample(tmp_path):
    nb = h.Notebook(["1 / 0"], cash_on=False, workdir=tmp_path)
    with pytest.raises(RuntimeError, match="cell 1 failed"):
        nb.run()
    nb.close()


def test_scenario_names_are_unique_and_there_are_about_two_dozen_rows():
    names = [s.name for s in SCENARIOS]
    assert len(names) == len(set(names))
    rows = sum(len(s.metrics) for s in SCENARIOS)
    assert 20 <= rows <= 40


def test_the_cli_writes_json_and_a_table(tmp_path, capsys):
    out = tmp_path / "r.json"
    assert speed_set.main(["--only", "dec_hit_small_dict", "--repeats", "1", "--json", str(out), "-q"]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    (row,) = data["rows"]
    assert row["name"] == "dec_hit_small_dict" and row["ratio"] > 0 and not row["skipped"]
    assert data["meta"]["cash_path"].endswith("cash")
    assert "dec_hit_small_dict" in capsys.readouterr().out


def test_a_scenario_that_raises_is_reported_not_fatal(monkeypatch, tmp_path):
    def boom(side, ctx):
        raise ValueError("old cash lacks this")

    monkeypatch.setattr("benchmarks._speed_scenarios.SCENARIOS", [h.Scenario("broken", "g", "w", boom)])
    result = speed_set.run(None, repeats=1, quick=True)
    (row,) = result["rows"]
    assert row["skipped"].startswith("error: ValueError: old cash lacks this")


@pytest.mark.parametrize("name", ["nb_seeded_rerun", "dec_first_call"])
def test_real_scenarios_run(name):
    (sc,) = [s for s in SCENARIOS if s.name == name]
    rows = h.measure(sc, repeats=1)
    assert all(r.ratio > 0 and not r.skipped for r in rows)
    assert not any("VALUE DIFFERS" in n for r in rows for n in r.notes)
