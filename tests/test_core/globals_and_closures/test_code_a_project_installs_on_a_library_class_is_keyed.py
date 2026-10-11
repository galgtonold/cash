"""Code a project installs on a library class is keyed when the cached
function reaches it through an argument.

``@pd.api.extensions.register_dataframe_accessor("money")`` puts the user's
class on ``pd.DataFrame``; ``df.money.net()`` runs it. So does a function
patched on (``pd.DataFrame.report = report``). The argument's pickle names
none of it and the helper walk never reaches it, so editing the accessor, or
a module setting it reads, served the old result.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

pytest.importorskip("pandas")

MOD = textwrap.dedent("""
    import pandas as pd
    import cash

    FEE = 1.0

    @pd.api.extensions.register_dataframe_accessor("money")
    class Money:
        def __init__(self, df):
            self._df = df
        def net(self):
            return self._df["amount"] - FEE * BODY

    @pd.api.extensions.register_series_accessor("cents")
    class Cents:
        def __init__(self, s):
            self._s = s
        def total(self):
            return float(self._s.sum()) * 100 * BODY

    def report(df):
        return float(df["amount"].max()) * BODY

    pd.DataFrame.report = report

    def helper(df):
        return float(df.money.net().sum())

    @cash.cache
    def total_net(df):
        return float(df.money.net().sum())

    @cash.cache
    def via_helper(df):
        return helper(df)

    @cash.cache
    def column_cents(df):
        return df["amount"].cents.total()

    @cash.cache
    def series_cents(s):
        return s.cents.total()

    @cash.cache
    def patched(df):
        return df.report()

    DF = pd.DataFrame({"amount": [10.0, 20.0, 30.0]})

    def show():
        return " ".join(
            str(v) for v in (total_net(DF), via_helper(DF), column_cents(DF), series_cents(DF["amount"]), patched(DF))
        )
""")


def _write(tmp_path, body):
    (tmp_path / "mod.py").write_text(MOD.replace("BODY", body), encoding="utf-8")


def test_editing_the_installed_code_recomputes_in_the_next_run(tmp_path):
    _write(tmp_path, "1")
    first = run_python("-c", "import mod; print(mod.show())", cwd=tmp_path).stdout.strip()
    _write(tmp_path, "2")
    second = run_python("-c", "import mod; print(mod.show())", cwd=tmp_path).stdout.strip()
    assert first == "57.0 57.0 6000.0 6000.0 30.0"
    assert second == "54.0 54.0 12000.0 12000.0 60.0"


def test_a_setting_the_accessor_reads_changed_in_this_process_recomputes(tmp_path):
    _write(tmp_path, "1")
    code = (
        "import mod\n"
        "a = mod.total_net(mod.DF), mod.via_helper(mod.DF)\n"
        "mod.FEE = 5.0\n"
        "b = mod.total_net(mod.DF), mod.via_helper(mod.DF)\n"
        "c = mod.total_net(mod.DF)\n"
        "print(a, b, c, mod.total_net.cache_info()['hits'])\n"
    )
    out = run_python("-c", code, cwd=tmp_path).stdout.strip()
    # The last call is a hit: an unchanged accessor does not miss.
    assert out == "(57.0, 57.0) (45.0, 45.0) 45.0 1"
