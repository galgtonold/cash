"""A callee's ATTRIBUTE names are not globals it reads.

``called_function_dependencies`` walks each callee's ``co_names``, which holds
attribute names too (``m.forecast(h)`` puts ``forecast`` there). In a
user's notebook ``forecast = run_forecast(series, H)`` called a helper that
calls ``m.forecast(h)``: the key held ``forecast:ABSENT`` before the first run
and ``forecast:<lineage>`` after, so the simulation never found the entry and
the export cell re-ran the forecast.
"""

from cash.notebook.cache_key import called_function_dependencies


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
