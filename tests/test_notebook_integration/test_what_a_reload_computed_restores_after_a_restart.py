"""What was computed after a helper reload restores in the next kernel.

Round 29, r29s1 (2/2 in ``repro/repro_restart_after_reload.py``, and their
Wednesday morning: 0 hits with nothing changed since Tuesday) and r29s3 (2/2):
edit your helper module, let cash reload it and recompute, restart the next
morning -- nothing restores. Restart once more and everything does. The
entries the session wrote after the reload were keyed differently from what a
fresh kernel computes for the same code and the same file.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(300)]

SLOW = "    _ = sum(i * i for i in range(6_000_000))\n"


def _module(op, extra="", clock=False):
    head, start, report = (
        ("import time\n", "    t0 = time.perf_counter()\n", "    print('took', round(time.perf_counter() - t0, 1))\n")
        if clock
        else ("", "", "")
    )
    return head + "def summary(rows):\n" + start + SLOW + report + "    return " + op + "(rows)\n" + extra


def _cells(import_line, call):
    return [
        "import cash\n%cash_on\n%cash_persist on\n%cash_badge print",
        import_line + "\nROWS = [1, 2, 3, 4]",
        "tbl = " + call + "(ROWS)",
        "print('R', tbl)",
    ]


@pytest.mark.parametrize(
    "import_line, call",
    [
        ("import {m} as hm", "hm.summary"),
        ("from {m} import summary", "summary"),
    ],
    ids=["aliased", "from_import"],
)
@pytest.mark.parametrize("edit", ["changes_the_function", "appends_an_unrelated_one"])
@pytest.mark.parametrize("clock", [False, True], ids=["plain", "timed"])
def test_the_next_kernel_restores_what_the_reload_computed(nb_runner, tmp_path, import_line, call, edit, clock):
    # A name of its own per case: the integration suite shares one project
    # cache, and two cases ending with the same file would serve each other.
    name = f"helperrr_{'t' if clock else 'p'}_{edit[:4]}_{'a' if ' as ' in import_line else 'f'}"
    import_line = import_line.format(m=name)
    mod = tmp_path / f"{name}.py"
    mod.write_text(_module("sum", clock=clock), encoding="utf-8")
    nb_runner.create_notebook(_cells(import_line, call))
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "R 10" in nb_runner.get_output(4), nb_runner.get_raw_output(4)

    if edit == "changes_the_function":
        mod.write_text(_module("max", clock=clock), encoding="utf-8")
        expected = "R 4"
    else:
        mod.write_text(_module("sum", "\n\ndef unrelated(x):\n    return x * 2\n", clock=clock), encoding="utf-8")
        expected = "R 10"
    nb_runner.run_cell(4)
    assert expected in nb_runner.get_output(4), nb_runner.get_raw_output(4)

    nb_runner.restart()
    nb_runner.run_all()
    assert expected in nb_runner.get_output(4), nb_runner.get_raw_output(4)
    raw = nb_runner.get_raw_output(3)
    assert "CACHED" in raw or "RESTORED" in raw, (
        "the morning after the edit, the value computed after the reload was recomputed instead of restored:\n" + raw
    )


@pytest.mark.parametrize(
    "import_line, call",
    [
        ("import {m} as hm", "hm.summary"),
        ("from {m} import summary", "summary"),
    ],
    ids=["aliased", "from_import"],
)
def test_an_unrelated_edit_to_a_timed_helper_re_runs_nothing(nb_runner, tmp_path, import_line, call):
    """r29s1 (8/8 edits) and r29s3: a helper that times its own steps was keyed
    whole, so appending an unrelated function re-ran everything built on it
    -- the 60-day load and the map-match, 57 s + 100 s per edit."""
    name = "helpertimed_" + ("a" if " as " in import_line else "f")
    mod = tmp_path / f"{name}.py"
    mod.write_text(_module("sum", clock=True), encoding="utf-8")
    nb_runner.create_notebook(_cells(import_line.format(m=name), call))
    nb_runner.start_kernel()
    nb_runner.run_all()
    mod.write_text(_module("sum", "\n\ndef unrelated(x):\n    return x * 2\n", clock=True), encoding="utf-8")
    nb_runner.run_cell(4)
    raw = nb_runner.get_raw_output(4)
    assert "R 10" in nb_runner.get_output(4), raw
    assert "EXECUTED: tbl =" not in raw, raw
