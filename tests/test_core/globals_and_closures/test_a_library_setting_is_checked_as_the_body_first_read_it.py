"""A library setting is kept as the body first read it, not as it left it.

A library's module dict the body fills as it runs (torch's
``eval_frame.cached_backends``, which ``_TorchDynamoContext.__init__`` fills
with ``setdefault`` on an optimizer step) was kept as the body left it. A new
process starts it empty again, so the call was never served there. The
result came of the value the body read, so each setting is now kept as the
first code reading it started (`library_reads.first_reads`): a new process
reading the same value is served, and one whose setting was changed from
outside still runs the body. Every call is compared with plain Python.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytestmark = [
    pytest.mark.core,
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
]

needs_recording = pytest.mark.skipif(
    sys.version_info < (3, 12), reason="a body's reads are recorded with sys.monitoring, Python 3.12 or newer"
)

LIBRARY = """
cached_backends = {}
_seen = []
_opts = {"scale": 1}


class _Context:
    def __init__(self, backend):
        cached_backends.setdefault(id(backend), backend)


def step(x):
    _Context(None)
    return x * 2


def count_then_fill(x):
    n = len(_seen)
    _seen.append(x)
    return n


def scaled(x):
    return x * _opts["scale"]


n = 0
state = {"n": 0}
calls = []


def bump_global():
    global n
    n += 1
    return n - 1


def bump_entry():
    state["n"] += 1
    return state["n"] - 1


def bump_list():
    k = len(calls)
    calls.append(k)
    return k
"""

COUNTER = """
import sys
sys.path.insert(0, {site!r})
import cash_ut_firstread as lib
from cash import Cash


def by_attribute():
    lib.n += 1
    return lib.n - 1


bodies = {{
    "attribute": by_attribute,
    "global": lambda: lib.bump_global(),
    "entry": lambda: lib.bump_entry(),
    "list": lambda: lib.bump_list(),
}}
body = bodies[sys.argv[1]]
if sys.argv[2] == "plain":
    print(*[body() for _ in range(3)])
else:
    f = Cash(cache_dir={cache!r}, register_magic=False).cache(body)
    print(*[f() for _ in range(3)])
"""

SCRIPT = """
import sys
sys.path.insert(0, {site!r})
import cash_ut_firstread as lib
from cash import Cash

def body(x):
    return lib.step(x), lib.count_then_fill(x), lib.scaled(x)

lib._opts["scale"] = int(sys.argv[1])
if sys.argv[2] == "plain":
    print(*body(7), 0)
else:
    c = Cash(cache_dir={cache!r}, register_magic=False)
    f = c.cache(body)
    print(*f(7), f.cache_info()["hits"])
"""

FOREIGN = """
import sys
sys.path.insert(0, {site!r})
import cash_ut_firstread as lib
from cash import Cash

# Another key the body never put there: what it reads may differ.
lib.cached_backends["other"] = 1
c = Cash(cache_dir={cache!r}, register_magic=False)
f = c.cache(lambda x: lib.step(x))
first = f(7)
hits = f.cache_info()["hits"]
# Called again in this process, the dict now filled as the call filled it.
assert f(7) == first
print(first, hits, f.cache_info()["hits"] - hits)
"""


@needs_recording
def test_a_dict_the_body_fills_is_kept_as_first_read_and_a_changed_setting_still_misses(tmp_path):
    site = tmp_path / "venv" / "lib" / "site-packages"
    site.mkdir(parents=True)
    (site / "cash_ut_firstread.py").write_text(LIBRARY, encoding="utf-8")
    script = tmp_path / "run.py"
    script.write_text(textwrap.dedent(SCRIPT.format(site=str(site), cache=str(tmp_path / "cache"))), encoding="utf-8")

    def run(scale, how="cached"):
        out = subprocess.run(
            [sys.executable, str(script), str(scale), how], capture_output=True, text=True, check=True, timeout=120
        )
        *result, hits = out.stdout.split()
        return result, int(hits)

    plain1, _ = run(1, "plain")
    plain3, _ = run(3, "plain")
    assert plain1 == ["14", "0", "7"]
    assert plain3 == ["14", "0", "21"]
    assert run(1) == (plain1, 0)
    # The body filled cached_backends and _seen; a new process starts them
    # empty again, as the body first read them: served.
    assert run(1) == (plain1, 1)
    # A setting the body read, changed from outside: the body runs.
    assert run(3) == (plain3, 0)
    assert run(3) == (plain3, 1)
    # Back to the first value: the entry made under 3 took the key's place.
    assert run(1) == (plain1, 0)


@needs_recording
def test_a_dict_the_body_fills_holding_another_key_misses(tmp_path):
    site = tmp_path / "venv" / "lib" / "site-packages"
    site.mkdir(parents=True)
    (site / "cash_ut_firstread.py").write_text(LIBRARY, encoding="utf-8")
    main = tmp_path / "main.py"
    cache = str(tmp_path / "cache")
    main.write_text(
        textwrap.dedent(FOREIGN.format(site=str(site), cache=cache)).replace('lib.cached_backends["other"] = 1', ""),
        encoding="utf-8",
    )
    foreign = tmp_path / "foreign.py"
    foreign.write_text(textwrap.dedent(FOREIGN.format(site=str(site), cache=cache)), encoding="utf-8")

    def run(script):
        out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=True, timeout=120)
        return out.stdout.split()

    assert run(main) == ["14", "0", "1"]
    assert run(main) == ["14", "1", "1"]
    # Another key the body never put there: runs, then is served.
    assert run(foreign) == ["14", "0", "1"]


@needs_recording
@pytest.mark.parametrize("how", ["attribute", "global", "entry", "list"])
def test_a_counter_the_body_bumps_is_recomputed(tmp_path, how):
    # The body reads a library counter and bumps it: the next call, in the
    # same process or another, runs with the value it now reads.
    site = tmp_path / "venv" / "lib" / "site-packages"
    site.mkdir(parents=True)
    (site / "cash_ut_firstread.py").write_text(LIBRARY, encoding="utf-8")
    script = tmp_path / "counter.py"
    script.write_text(textwrap.dedent(COUNTER.format(site=str(site), cache=str(tmp_path / "cache"))), encoding="utf-8")

    def run(mode):
        out = subprocess.run(
            [sys.executable, "-W", "ignore", str(script), how, mode],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        return out.stdout.split()

    plain = run("plain")
    assert plain == ["0", "1", "2"]
    assert run("cached") == plain
    assert run("cached") == plain


EXTENSION = """
import sys
import types

# A compiled module's table of its C helpers' addresses, filled as it
# loads: other in every process.
ext = types.ModuleType("cash_ut_ext")
ext.__file__ = {so!r}
ext.c_helpers = {{"extract": id(object()), "hash": id(ext)}}
sys.modules["cash_ut_ext"] = ext
sys.path.insert(0, {site!r})
import cash_ut_tables as lib
from cash import Cash

f = Cash(cache_dir={cache!r}, register_magic=False).cache(lambda x: lib.helper(x))
first = f(3)
hits = f.cache_info()["hits"]
if sys.argv[1] == "rebind":
    # Another table under the name: what the body read moved.
    lib.c_helpers = {{"extract": 0}}
    assert f(3) == lib.helper(3) == 0
    assert f.cache_info()["hits"] == hits, "served under a rebound table"
print(first > 0, hits)
"""

TABLES = """
from cash_ut_ext import c_helpers


def helper(x):
    return x * c_helpers["extract"]
"""


@needs_recording
def test_a_compiled_modules_table_of_addresses_is_checked_in_its_process_only(tmp_path):
    site = tmp_path / "venv" / "lib" / "site-packages"
    site.mkdir(parents=True)
    (site / "cash_ut_tables.py").write_text(TABLES, encoding="utf-8")
    script = tmp_path / "tables.py"
    so = str(site / "cash_ut_ext.cpython-312-x86_64-linux-gnu.so")
    script.write_text(
        textwrap.dedent(EXTENSION.format(so=so, site=str(site), cache=str(tmp_path / "cache"))), encoding="utf-8"
    )

    def run(how):
        out = subprocess.run(
            [sys.executable, "-W", "ignore", str(script), how], capture_output=True, text=True, check=True, timeout=120
        )
        return out.stdout.split()

    assert run("plain") == ["True", "0"]
    # The addresses differ in the next process; the table is the same one.
    assert run("plain") == ["True", "1"]
    assert run("rebind") == ["True", "1"]
