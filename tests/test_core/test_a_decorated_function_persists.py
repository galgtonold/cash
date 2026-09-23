"""Decorating a function means caching it, floor or no floor.

Found while stress-testing the decorator: a function under the 0.1 s
persistence floor is cached in RAM only, so running a script twice recomputes
every time and a user sees no caching at all. Two of six agents' first sweeps
reported "no bugs found" for that reason alone -- nothing was ever stored, so
no staleness could appear. The floor belongs to the notebook, where cash caches
every statement by itself; ``@cash.cache`` is the user saying to cache this one.

The cost model does not get a vote either. It was still gating decorated entries
after the floor went, which put the whole decision on one fitted intercept --
measured at 10.4 ms against a real small read of ~1.3 ms, so nothing under about
13 ms of body reached disk however often it was called. Decorating a function is
a decision the caller has already made; cash's job is to honour it, not to
second-guess it with a number it cannot measure reliably. The per-tier size caps
still apply: a value with nowhere to fit still has nowhere to fit.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

PROGRAM = textwrap.dedent("""
    import json
    import cash
    cash.configure(cache_dir=CACHE)

    calls = []

    @cash.cache
    def quick(n):
        import time
        time.sleep(0.02)          # well under the old 0.1s floor
        calls.append(n)
        return n * 2

    value = quick(21)
    print(json.dumps({"value": value, "ran": len(calls),
                      "misses": quick.cache_info()["misses"]}))
""")


def _run(tmp_path):
    script = tmp_path / "run.py"
    script.write_text(PROGRAM.replace("CACHE", repr(str(tmp_path / ".cash"))), encoding="utf-8")
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=180, cwd=str(tmp_path))
    assert done.returncode == 0, done.stderr[-1500:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.timeout(300)
def test_a_quick_function_survives_a_second_process(tmp_path):
    assert _run(tmp_path) == {"value": 42, "ran": 1, "misses": 1}
    assert _run(tmp_path) == {"value": 42, "ran": 0, "misses": 0}, "nothing reached disk"


@pytest.mark.timeout(300)
def test_the_entry_is_on_disk(tmp_path):
    _run(tmp_path)
    assert list((tmp_path / ".cash").glob("*.entry")), sorted(p.name for p in (tmp_path / ".cash").iterdir())


RESTORE_IS_SLOWER = textwrap.dedent("""
    import json
    import cash
    cash.configure(cache_dir=CACHE)

    @cash.cache
    def wide(n):
        return list(range(4_000_000))       # instant to build, slow to restore

    wide(1)
    print(json.dumps({"ok": True}))
""")


@pytest.mark.timeout(300)
def test_a_result_slower_to_restore_than_to_rebuild_is_still_stored(tmp_path):
    """Even the case the cost model would refuse.

    This asserted the opposite until the cost model stopped voting on decorated
    entries. The reasoning behind the old assertion was sound in the abstract --
    do not put a value on disk that costs more to read back than to recompute --
    but acting on it meant trusting a prediction, and the prediction was made by
    an intercept that priced a small read at 10.4 ms when it measures ~1.3 ms.
    A wrong refusal is silent and permanent: the caller decorated the function
    and gets nothing across processes, with no way to tell why.

    So the decision goes back to the caller. `@cash.cache` stores it; a caller
    who does not want that removes the decorator.
    """
    script = tmp_path / "wide.py"
    script.write_text(RESTORE_IS_SLOWER.replace("CACHE", repr(str(tmp_path / ".cash"))), encoding="utf-8")
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=180, cwd=str(tmp_path))
    assert done.returncode == 0, done.stderr[-1500:]
    assert list((tmp_path / ".cash").glob("*.entry")), "a decorated result was not stored"


QUICK = textwrap.dedent("""
    import json
    import cash
    cash.configure(cache_dir=CACHE)

    calls = []

    @cash.cache
    def barely_anything(n):
        import time
        time.sleep(0.002)         # 2ms: cheaper to redo than to read back
        calls.append(n)
        return n + 1

    @cash.cache
    def quick_but_bulky(n):
        import time
        time.sleep(0.005)
        calls.append(n)
        return [{"i": i, "s": "x" * 16} for i in range(n)]

    small = barely_anything(1)
    bulky = quick_but_bulky(50_000)
    print(json.dumps({"small": small, "bulky": len(bulky), "ran": len(calls)}))
""")


def _run_quick(tmp_path):
    script = tmp_path / "quick.py"
    script.write_text(QUICK.replace("CACHE", repr(str(tmp_path / ".cash"))), encoding="utf-8")
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=180, cwd=str(tmp_path))
    assert done.returncode == 0, done.stderr[-1500:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.timeout(300)
def test_a_two_millisecond_call_still_reaches_disk(tmp_path):
    """The cheapest thing anyone would bother decorating.

    Restoring it costs more than running it, and cash caches it anyway, because
    the caller asked. The alternative -- deciding per call whether the user
    meant it -- is what left a decorated function recomputing in every process
    with nothing to show for the decorator.
    """
    assert _run_quick(tmp_path)["ran"] == 2
    assert _run_quick(tmp_path)["ran"] == 0, "a second process recomputed both"


@pytest.mark.timeout(300)
def test_a_quick_call_with_a_bulky_result_reaches_disk_too(tmp_path):
    """The case the cost model used to refuse: 5ms of work, a result that takes
    longer than that to read back."""
    _run_quick(tmp_path)
    entries = list((tmp_path / ".cash").glob("*.entry"))
    assert len(entries) >= 2, [p.name for p in (tmp_path / ".cash").iterdir()]
