"""A callee's ATTRIBUTE names are not globals it reads.

``called_function_dependencies`` walks each callee's ``co_names``, which holds
attribute names too (``m.forecast(h)`` puts ``forecast`` there). In a
user's notebook ``forecast = run_forecast(series, H)`` called a helper that
calls ``m.forecast(h)``: the key held ``forecast:ABSENT`` before the first run
and ``forecast:<lineage>`` after, so the simulation never found the entry and
the export cell re-ran the forecast.
"""

from cash.notebook.cache_key import called_function_dependencies
from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell


def _ns(src):
    ns: dict = {}
    exec(src, ns)
    return ns


HELPER = """
def run_forecast(series, h):
    return series.forecast(h)
"""


def test_an_attribute_named_like_a_variable_is_not_a_dependency():
    ns = _ns(HELPER)
    before = called_function_dependencies(["run_forecast", "series"], ns, {})
    ns["forecast"] = object()  # the statement's own output, after it ran
    after = called_function_dependencies(["run_forecast", "series"], ns, {"forecast": "L1"})
    assert before == after == ["forecast:ABSENT"]


def test_a_global_the_callee_reads_is_still_a_dependency():
    """Control: the same name read as a GLOBAL moves the key."""
    ns = _ns("""
def run_forecast(series, h):
    return forecast_model(series, h)
""")
    ns["forecast_model"] = object()
    deps = called_function_dependencies(["run_forecast"], ns, {"forecast_model": "L2"})
    assert deps == ["forecast_model:L2"]


def test_a_name_read_as_a_global_by_another_callee_still_counts():
    """Attribute in one callee, global in another: the global read wins."""
    ns = _ns("""
def a(x):
    return x.rate
def b():
    return rate * 2
def both(x):
    return a(x) + b()
""")
    ns["rate"] = 3
    deps = called_function_dependencies(["both"], ns, {"rate": "L3"})
    assert "rate:L3" in deps


def test_keys_without_a_collision_are_unchanged():
    """An attribute name no variable shares keeps contributing ABSENT, as before."""
    ns = _ns(HELPER)
    assert called_function_dependencies(["run_forecast"], ns, {}) == ["forecast:ABSENT"]


def test_a_string_naming_an_attribute_is_not_a_global_read():
    """``hasattr(o, "u")`` loads the constant ``"u"``; only ``o.u`` puts ``u``
    in ``co_names``. scipy's ``factorized`` does this, so a notebook that
    called it and bound ``u`` keyed ``u:ABSENT`` before its first run and
    ``u:<lineage>`` after."""
    ns = _ns("""
def solve(o):
    if hasattr(o, "u"):
        return o.u
    return 0
""")
    before = called_function_dependencies(["solve"], ns, {})
    ns["u"] = object()
    after = called_function_dependencies(["solve"], ns, {"u": "L4"})
    assert before == after == ["u:ABSENT"]


def test_a_local_named_like_an_attribute_is_not_a_global_read():
    ns = _ns("""
def solve(o):
    u = o.u
    return u
""")
    ns["u"] = object()
    assert called_function_dependencies(["solve"], ns, {"u": "L5"}) == ["u:ABSENT"]


def _statuses(magics):
    return [m.get("status") for m in magics.cash_status("dict")["last_cell"]["statements"]]


def test_a_statement_binding_a_name_its_callee_uses_as_a_string_hits_on_rerun(cash_magics):
    """The same case run as cells: the first re-run of ``u, v = run(3)`` missed."""
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    run_cash_cell(
        cash_magics,
        # The sleep makes the call worth caching.
        "import time\ndef run(o):\n    time.sleep(0.05)\n    if hasattr(o, 'u'):\n        return o.u, 0\n    return o, o + 1",
    )
    run_cash_cell(cash_magics, "u, v = run(3)")
    assert _statuses(cash_magics) == [CacheStatus.COMPUTED]
    run_cash_cell(cash_magics, "u, v = run(3)")
    assert _statuses(cash_magics) == [CacheStatus.RESTORED], "the first re-run missed"
