"""`%cash_stats` cannot see decorator hits on a custom Cash instance (CAS-222).

`_get_cash_instance` drains the notebook's own `Cash` or the module-level
`cash._global_cash` singleton, and `Cash._decorator_call_log` is a per-instance
list — so a `@c.cache` hit on a separately-constructed `c = cash.Cash()` is
never credited. That instance is the pattern several feature guides teach for
`register_hasher`, custom backends and `file_depends_on`.

Measured, identical workload both ways:

    @cash.cache (global)          cache_info saved 0.401s   %cash_stats 0.401s
    @c.cache on cash.Cash()       cache_info saved 0.403s   %cash_stats 0.0s

This pins the CURRENT behaviour and the doc caveat that describes it
(`docs/tutorials/feature-guides/debugging-and-monitoring.md`). If the blind
spot is ever closed, this test fails — delete it and the caveat together.
"""
import json
import re

import pytest

pytestmark = [pytest.mark.integration]

_SLEEP = 0.4          # well above the cost-model floor


def _cells(setup, decorator):
    return [
        "import cash, time\n%cash_on\n%cash_badge off",
        setup,
        f"{decorator}\ndef slow(x):\n    time.sleep({_SLEEP})\n    return x + 1",
        "print('R', slow(1), slow(1))",                       # miss, then hit
        "i = slow.cache_info()\n"
        "print('INFO %s %.3f' % (i['hits'], i['total_time_saved']))",
        "%cash_stats json",
    ]


def _run(nb_runner, setup, decorator):
    nb_runner.create_notebook(_cells(setup, decorator))
    nb_runner.start_kernel()
    nb_runner.run_all()

    info = next(ln for ln in nb_runner.get_output(5).splitlines()
                if ln.strip().startswith("INFO"))
    _, hits, saved = info.split()

    blob = re.search(r"\{.*\}", nb_runner.get_output(6), re.S)
    assert blob, "%cash_stats json printed nothing parseable"
    return int(hits), float(saved), json.loads(blob.group(0))["total_time_saved"]


def test_global_decorator_is_credited(nb_runner):
    """Positive control. Without it, a stats system that reports 0 for
    everything would satisfy the blind-spot test below."""
    hits, saved, gross = _run(nb_runner, "pass", "@cash.cache")
    assert hits == 1, f"the second call was not a cache hit ({hits})"
    assert saved > _SLEEP / 2, f"cache_info recorded no real saving ({saved})"
    assert gross > _SLEEP / 2, (
        f"%cash_stats did not credit a GLOBAL decorator hit ({gross}); the "
        "CAS-222 fix has regressed"
    )


def test_custom_instance_decorator_is_not_credited(nb_runner):
    hits, saved, gross = _run(nb_runner, "c = cash.Cash()", "@c.cache")
    assert hits == 1, f"the second call was not a cache hit ({hits})"
    assert saved > _SLEEP / 2, (
        f"cache_info recorded no real saving ({saved}) — this test would then "
        "prove nothing about what %cash_stats can see"
    )
    assert gross == 0.0, (
        f"%cash_stats now credits custom-instance hits (gross={gross}). That is "
        "an improvement — delete this test and the 'default Cash instance' "
        "caveat in docs/tutorials/feature-guides/debugging-and-monitoring.md."
    )
