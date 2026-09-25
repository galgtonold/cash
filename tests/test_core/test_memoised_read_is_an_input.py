"""A file read once and memoised is still an input of every call that uses it.

A parse memoised with ``functools.lru_cache`` (or a module
dict), aggregated by two cached functions. The first recorded the file; the
second got the memoised rows, read nothing, and stored no file dependency --
after the data changed it kept serving the old total, 2 of 2, while the same
job without the memo was right.
"""

from __future__ import annotations

import functools
import os

import pytest

from cash import Cash

pytestmark = [pytest.mark.core]


def _parse_plain(path):
    with open(path, encoding="utf-8") as fh:
        return [int(line) for line in fh if line.strip()]


_MEMO: dict = {}


def _parse_dict(path):
    if path not in _MEMO:
        _MEMO[path] = _parse_plain(path)
    return _MEMO[path]


def _write(path, values):
    path.write_text("\n".join(str(v) for v in values) + "\n", encoding="utf-8")
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))


@pytest.fixture(params=["lru_cache", "dict"])
def parse(request):
    _MEMO.clear()
    if request.param == "lru_cache":
        fn = functools.lru_cache(maxsize=8)(_parse_plain)
        yield fn
        fn.cache_clear()
    else:
        yield _parse_dict
        _MEMO.clear()


def _consumers(c, parse):
    calls = {"a": 0, "b": 0}

    @c.cache
    def total_a(path):
        calls["a"] += 1
        return sum(parse(path))

    @c.cache
    def total_b(path):
        calls["b"] += 1
        return max(parse(path))

    return total_a, total_b, calls


def test_the_second_consumer_of_a_memoised_parse_invalidates_when_the_file_changes(tmp_path, parse):
    data = tmp_path / "data.txt"
    _write(data, [1, 2, 3])
    total_a, total_b, calls = _consumers(Cash(cache_dir=str(tmp_path / "cache")), parse)

    assert (total_a(str(data)), total_b(str(data))) == (6, 3)
    assert (total_a(str(data)), total_b(str(data))) == (6, 3)
    assert calls == {"a": 1, "b": 1}, "an unchanged file did not hit"

    _write(data, [10, 20, 30, 40])
    if hasattr(parse, "cache_clear"):
        parse.cache_clear()  # what a new process starts with
    _MEMO.clear()
    assert total_b(str(data)) == 40, "the memo's consumer served the old maximum"
    assert calls["b"] == 2


def test_a_memo_filled_from_an_older_file_is_not_stored_as_the_new_answer(tmp_path, parse):
    """In a long-lived process the memo still holds the old
    parse after the file changed. Without cash only this process is wrong; cash
    stored that answer against the NEW file, so a restart served it too."""
    data = tmp_path / "data.txt"
    _write(data, [1, 2, 3])
    total_a, total_b, calls = _consumers(Cash(cache_dir=str(tmp_path / "cache")), parse)
    assert total_a(str(data)) == 6  # fills the memo from [1, 2, 3]

    _write(data, [10, 20, 30, 40])  # the memo is NOT cleared
    assert total_b(str(data)) == 3  # the memo's answer, as without cash
    if hasattr(parse, "cache_clear"):
        parse.cache_clear()  # what a new process starts with
    _MEMO.clear()
    assert total_b(str(data)) == 40, "the old memo's answer was stored for the new file"


def test_a_memo_keyed_by_path_does_not_tie_one_file_to_another(tmp_path, parse):
    """Control: the memo read two files over the process. The consumer of the
    first must not start depending on the second -- only on the file it was
    given -- or every unrelated edit would recompute it."""
    first, second = tmp_path / "first.txt", tmp_path / "second.txt"
    _write(first, [1, 2])
    _write(second, [5, 6])
    total_a, total_b, calls = _consumers(Cash(cache_dir=str(tmp_path / "cache")), parse)

    total_a(str(second))  # the memo reads `second` here
    assert total_b(str(first)) == 2  # reads `first` through... the memo reads it live
    total_a(str(first))  # memoised: nothing read in this call
    _write(second, [50, 60])
    assert total_a(str(first)) == 3
    assert calls["a"] == 2, "an edit to another file invalidated a memoised consumer"


_SETTINGS_APP = """
import functools, json, os, sys, time
import cash

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

@functools.lru_cache(maxsize=None)
def get_settings():
    with open(PATH, encoding="utf-8") as f:
        return json.load(f)

@cash.cache
def convert(amount):
    print("RAN", file=sys.stderr)
    time.sleep(0.25)          # over the persistence floor: the entry must reach disk
    return round(amount * get_settings()["rate"], 2)

if __name__ == "__main__":
    print("starting with rate", get_settings()["rate"], file=sys.stderr)
    print(convert(100.0))
"""


def test_settings_a_memo_read_before_the_first_cached_call_are_an_input(tmp_path):
    """`main()` logs its settings -- through the memo --
    before the first cached call, the way a CLI prints its config. No cached
    call ever read the file, so every later edit of it was ignored. Fresh
    processes, because that is where the memo starts empty."""
    import json
    import subprocess
    import sys

    app = tmp_path / "app.py"
    app.write_text(_SETTINGS_APP, encoding="utf-8")
    settings = tmp_path / "settings.json"
    env = dict(os.environ, CASH_CACHE_DIR=str(tmp_path / "cache"), PYTHONWARNINGS="ignore")
    env.pop("CASH_DISABLE", None)
    got = []
    for rate in (1.1, 1.3, 1.4):
        settings.write_text(json.dumps({"rate": rate}), encoding="utf-8")
        st = os.stat(settings)
        os.utime(settings, ns=(st.st_atime_ns, st.st_mtime_ns + int(rate * 10) * 2_000_000_000))
        run = subprocess.run(
            [sys.executable, str(app)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
        )
        assert run.returncode == 0, run.stderr[-2000:]
        got.append((float(run.stdout.strip().splitlines()[-1]), "RAN" in run.stderr))
    assert got == [(110.0, True), (130.0, True), (140.0, True)], "a settings edit was ignored"
    # Control: nothing changed, so the fourth run is served from disk.
    run = subprocess.run([sys.executable, str(app)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert "RAN" not in run.stderr, "the entry never reached disk: this test proves nothing"


_IMPORT_TIME_CONFIG = """
import functools, json, os, sys

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfg.json")

@functools.lru_cache(maxsize=None)
def load_cfg():
    with open(PATH, encoding="utf-8") as fh:
        return json.load(fh)

print("starting", load_cfg()["name"], file=sys.stderr)   # a banner at import
"""

_IMPORT_TIME_MAIN = """
import sys, time
import cash
from config import load_cfg

@cash.cache
def f(x):
    print("RAN", file=sys.stderr)
    time.sleep(0.25)          # over the persistence floor: the entry must reach disk
    return x * load_cfg()["k"]

print(f(2))
"""


def test_a_memo_filled_at_import_before_any_decoration_is_an_input(tmp_path):
    """The memo was filled while the app's modules were imported, before any
    function was decorated -- and the read watch started only at the first
    decoration, so the read was nobody's and a config edit was served stale.
    It starts when cash is imported."""
    import json
    import subprocess
    import sys

    (tmp_path / "config.py").write_text(_IMPORT_TIME_CONFIG, encoding="utf-8")
    (tmp_path / "main.py").write_text(_IMPORT_TIME_MAIN, encoding="utf-8")
    cfg = tmp_path / "cfg.json"
    env = dict(os.environ, CASH_CACHE_DIR=str(tmp_path / "cache"), PYTHONWARNINGS="ignore")
    env.pop("CASH_DISABLE", None)
    got = []
    for step, k in enumerate((10, 10, 11)):
        cfg.write_text(json.dumps({"name": "demo", "k": k}), encoding="utf-8")
        st = os.stat(cfg)
        os.utime(cfg, ns=(st.st_atime_ns, st.st_mtime_ns + (step + 1) * 2_000_000_000))
        run = subprocess.run(
            [sys.executable, "main.py"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
        )
        assert run.returncode == 0, run.stderr[-2000:]
        got.append((run.stdout.strip().splitlines()[-1], "RAN" in run.stderr))
    assert got[2] == ("22", True), f"a config edit was served the old result: {got}"
