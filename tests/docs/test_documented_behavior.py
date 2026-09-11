"""Behavioral grounding for the caching guarantees the docs *promise* (P2).

The fence harness runs examples and the claim-checker asserts opt-in hit/miss
counts, but neither exercises the load-bearing *behaviors* the prose sells —
and those are exactly what drifted (the data_sources ``has_changed`` bool bug,
the ``file_depends_on`` mtime-vs-content mix-up). Each test here encodes one
documented promise against a real ``Cash`` instance, so if the behavior changes
the doc claim it grounds is caught.

Call counting uses ``assume_safe=True`` + a dict counter: on a cache *hit* the
body doesn't run, so the counter is the number of real computes. Tests about the
warning contract use pure functions instead, so the only warning that can fire
is the one under test.
"""
from __future__ import annotations

import os
import pathlib
import re
import threading
import time
import warnings

import pytest

from cash import Cash
from cash.data_source import DataSource
from cash.exceptions import CashCacheIneffectiveWarning

# Module globals used by tests where a nested function must read/call a real
# module attribute via ``__globals__`` (a name defined *inside* a test would be
# a closure freevar instead, which the analyzer treats differently).
_TAX_RATE = 0.2

# NOTE: transitive helper-edit invalidation (cash folding a called helper's
# source into the key — the headline difference from joblib in
# docs/migration_guide.md) is intentionally NOT grounded here. It models an
# edit-and-rerun of a helper's *source*, which is a file/notebook operation;
# reproducing it in one in-process test is unreliable (the decorator's
# re-resolution interacts with how the test function nests its defs). That
# invariant is exercised by the core decorator suite under tests/test_core.


def _cash(tmp_path) -> Cash:
    return Cash(cache_dir=str(tmp_path), register_magic=False)


# --------------------------------------------------------------------------- #
# DataSource token contract  (docs/api/data_sources.md,                       #
#                             docs/tutorials/feature-guides/dynamic-dependencies.md)
# --------------------------------------------------------------------------- #

def test_datasource_token_invalidates_when_it_changes(tmp_path):
    """A ``depends_on`` DataSource whose ``has_changed()`` returns a state
    *token* invalidates the cache exactly when that token changes."""
    c = _cash(tmp_path)
    state = {"v": 1}

    class TokenSource(DataSource):
        def get_id(self):
            return "tok"

        def has_changed(self):
            return state["v"]      # a token (int), not a bool

        def update_state(self):
            pass

    n = {"c": 0}

    @c.cache(depends_on=[TokenSource()], assume_safe=True)
    def f():
        n["c"] += 1
        return n["c"]

    f()
    f()
    assert n["c"] == 1, "token unchanged -> cached"
    state["v"] = 2
    f()
    assert n["c"] == 2, "token changed -> recomputed"


def test_datasource_nonbool_token_does_not_warn(tmp_path):
    """A non-bool ``has_changed()`` token is valid and fires no warning."""
    c = _cash(tmp_path)

    class IntSource(DataSource):
        def get_id(self):
            return "int"

        def has_changed(self):
            return 7

        def update_state(self):
            pass

    @c.cache(depends_on=[IntSource()])
    def pure():
        return 1

    with warnings.catch_warnings():
        warnings.simplefilter("error", CashCacheIneffectiveWarning)
        pure()  # must not raise


def test_datasource_bool_has_changed_warns(tmp_path):
    """A ``bool`` ``has_changed()`` can't track changes, so cash warns — this
    is the exact misconfiguration behind the old broken DBTableSource example."""
    c = _cash(tmp_path)

    class BoolSource(DataSource):
        def get_id(self):
            return "bool"

        def has_changed(self):
            return True

        def update_state(self):
            pass

    @c.cache(depends_on=[BoolSource()])
    def pure():
        return 1

    with pytest.warns(CashCacheIneffectiveWarning):
        pure()


# --------------------------------------------------------------------------- #
# File tracking: file_depends_on = mtime, auto = content  (docs/decorator.md)  #
# --------------------------------------------------------------------------- #

def test_file_depends_on_tracks_mtime_not_content(tmp_path):
    """``file_depends_on=`` keys on the file **mtime**, not its content: a
    content edit that leaves the mtime unchanged stays cached; bumping the
    mtime recomputes. (The function must NOT read the file, or auto content
    tracking would mask the mtime-only behavior.)"""
    c = _cash(tmp_path)
    p = tmp_path / "cfg.bin"
    p.write_text("aaaa")
    st = p.stat()
    n = {"c": 0}

    @c.cache(file_depends_on=str(p), assume_safe=True)
    def g():
        n["c"] += 1
        return n["c"]

    g()
    g()
    assert n["c"] == 1

    p.write_text("bbbb")                                   # same size, new content
    os.utime(p, (st.st_atime, st.st_mtime))                # reset mtime to original
    g()
    assert n["c"] == 1, "content changed but mtime reset -> still cached (mtime-only)"

    time.sleep(0.02)
    os.utime(p, None)                                      # bump mtime only
    g()
    assert n["c"] == 2, "mtime changed -> recomputed"


def test_auto_file_tracking_is_content_hash(tmp_path):
    """Automatic file-read tracking fingerprints file **content**: a content
    edit recomputes even if the mtime is reset to its original value."""
    c = _cash(tmp_path)
    p = tmp_path / "data.csv"
    p.write_text("a,b\n1,2\n")
    st = p.stat()
    n = {"c": 0}

    @c.cache(assume_safe=True)
    def load():
        n["c"] += 1
        return p.read_text()

    load()
    load()
    assert n["c"] == 1

    p.write_text("a,b\n9,9\n")                             # same size, new content
    os.utime(p, (st.st_atime, st.st_mtime))                # reset mtime
    load()
    assert n["c"] == 2, "content changed (mtime reset) -> recomputed (content-hash)"


# --------------------------------------------------------------------------- #
# Content hashing, module globals, cache_if, helpers, unhashable args         #
# --------------------------------------------------------------------------- #

def test_content_equal_args_share_one_entry(tmp_path):
    """Two content-equal but non-identical args (here dicts differing only in
    insertion order) hit the same cache entry — cash hashes by content, unlike
    ``functools.lru_cache``. The same holds for DataFrames/arrays via hashers."""
    c = _cash(tmp_path)
    n = {"c": 0}

    @c.cache(assume_safe=True)
    def summ(d):
        n["c"] += 1
        return sum(d.values())

    summ({"a": 1, "b": 2})
    summ({"b": 2, "a": 1})
    assert n["c"] == 1


def test_read_module_global_invalidates_on_change(tmp_path):
    """A cached function that *reads* a module global recomputes when that
    global changes (docs/decorator.md 'Module globals a function reads')."""
    global _TAX_RATE
    _TAX_RATE = 0.2
    c = _cash(tmp_path)
    n = {"c": 0}

    @c.cache(assume_safe=True)
    def net(x):
        n["c"] += 1
        return x * (1 - _TAX_RATE)   # reads the module global

    assert net(100) == 80.0
    _TAX_RATE = 0.5
    assert net(100) == 50.0
    assert n["c"] == 2


def test_cache_if_falsy_result_is_not_cached(tmp_path):
    """``cache_if=`` skips caching on a falsy predicate but still returns the
    value (docs/decorator.md 'cache_if — skip caching by result')."""
    c = _cash(tmp_path)
    n = {"c": 0}

    @c.cache(cache_if=lambda r: r is not None, assume_safe=True)
    def maybe(x):
        n["c"] += 1
        return None if x < 0 else x

    assert maybe(-1) is None
    assert maybe(-1) is None
    assert n["c"] == 2, "predicate rejected -> not cached, recomputed"

    n["c"] = 0
    assert maybe(5) == 5
    assert maybe(5) == 5
    assert n["c"] == 1, "predicate accepted -> cached"


def test_unhashable_arg_warns_but_returns_correct_value(tmp_path):
    """An unpicklable argument can't be keyed, so cash warns and skips caching
    — but the call still returns the correct value (docs/decorator.md
    'Unhashable arguments')."""
    c = _cash(tmp_path)

    @c.cache
    def f(lock):
        return "computed"

    with pytest.warns(CashCacheIneffectiveWarning):
        result = f(threading.Lock())
    assert result == "computed"


# ---------------------------------------------------------------------------
# The text badge an agent is told to parse
# ---------------------------------------------------------------------------

#: One rendered statement row, as ``renderers.text._row_line`` emits it: two
#: spaces, ONE label, the code, then a timing — ``(saved Ns)`` on a restore,
#: ``(Ns)`` otherwise — with an optional ``- reason`` tail on a NOT CACHED row.
_BADGE_ROW = re.compile(
    r"^ {2,}(?:\^)?(CACHED|EXECUTED|NOT CACHED|SKIPPED): .+? {2}"
    r"\((?:saved )?\d+\.\d\ds\)(?: - .+)?$"
)

_BADGE_HEADER = re.compile(r"^\[Cash\] (?:CACHED|EXECUTED|SKIPPED)\b")


def _render_reference_badge() -> list[str]:
    """A real badge for the shape the agent guide advertises: a restore, a
    compute, and a row that ran but was not stored."""
    import io
    from contextlib import redirect_stdout

    from cash.notebook.badge_renderer._text import print_text_badge
    from cash.notebook.cache_status import CacheStatus

    metrics = [
        {"code": "df = pd.read_csv('sales.csv')", "status": str(CacheStatus.RESTORED),
         "execution_time": 0.004, "total_time": 0.004, "saved_time": 0.42,
         "storage_tiers": ["RAM", "DISK"], "variables": ["df"]},
        {"code": "summary = df.groupby('region').sum()", "status": str(CacheStatus.COMPUTED),
         "execution_time": 0.01, "total_time": 0.01, "variables": ["summary"]},
        {"code": "n = len(df)", "status": str(CacheStatus.COMPUTED),
         "execution_time": 0.0001, "total_time": 0.0001,
         "uncacheable_reasons": ["Too cheap to cache"], "variables": ["n"]},
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_text_badge(metrics, cell_total_time=0.015)
    return buf.getvalue().splitlines()


def _agent_guide_badge_block() -> list[str]:
    """The fenced ``[Cash] ...`` transcript from docs/for-coding-agents.md."""
    page = (
        pathlib.Path(__file__).resolve().parents[2]
        / "docs" / "for-coding-agents.md"
    ).read_text(encoding="utf-8")
    blocks = re.findall(r"^```\n(\[Cash\] .*?)^```", page, re.M | re.S)
    assert len(blocks) == 1, (
        f"expected exactly one [Cash] transcript in for-coding-agents.md, found {len(blocks)}"
    )
    return [ln for ln in blocks[0].splitlines() if ln.strip()]


def test_text_badge_renders_one_labelled_row_per_line():
    """Ground truth for the test below: this is the shape the renderer emits.

    Written as its own assertion so a renderer change fails *here*, naming the
    renderer, rather than looking like the docs drifted.
    """
    lines = _render_reference_badge()
    assert _BADGE_HEADER.match(lines[0]), lines[0]
    rows = lines[1:]
    assert rows, "expected statement rows under the header"
    for row in rows:
        assert _BADGE_ROW.match(row), f"renderer emitted an unexpected row shape: {row!r}"
    labels = [_BADGE_ROW.match(r).group(1) for r in rows]
    assert labels == ["CACHED", "EXECUTED", "NOT CACHED"], labels


def test_agent_guide_badge_example_matches_what_the_renderer_emits():
    """docs/for-coding-agents.md tells an agent what to parse, so it must be real.

    It once showed three rows crammed onto ONE line with no timings — a shape
    ``print_text_badge`` cannot produce. Nothing caught it: the block is an
    untagged fence, so the docs harness never executes it, and the
    ``AGENT_GUIDE`` sync test only proves the two copies of the mistake agree
    with each other. An agent writing a parser from that page wrote the wrong
    one.
    """
    doc_lines = _agent_guide_badge_block()

    assert _BADGE_HEADER.match(doc_lines[0]), (
        f"the guide's transcript must open with a real header line, got {doc_lines[0]!r}"
    )
    for row in doc_lines[1:]:
        assert _BADGE_ROW.match(row), (
            f"the guide shows a badge row the renderer cannot emit: {row!r}\n"
            "Rows are one per line: two spaces, one label, the code, then a "
            "timing. Re-render with cash.notebook.badge_renderer._text."
            "print_text_badge and paste what it prints."
        )


# ---------------------------------------------------------------------------
# How a script's cached functions are NAMED
# ---------------------------------------------------------------------------

def test_a_scripts_cached_function_is_keyed_by_filename_not_dunder_main(tmp_path):
    """`docs/decorator.md` and `docs/cli.md` show `cash inspect` / CASH_SUMMARY
    output right after telling the reader to run `python model.py`, so the
    module prefix in those transcripts is a claim about this behaviour.

    Both pages showed `__main__.` until 2026-09-02 -- pre-resolution output.
    A script's functions group under the defining FILE's name, which is what
    makes `python model.py` and `import model` share one entry
    (see how-it-works/decorator-path.md). `__main__.` survives only where
    there is no defining file: a REPL, `python -c`, a notebook kernel.

    A subprocess is required: run in-process, this file is not `__main__`.
    """
    import subprocess
    import sys

    script = tmp_path / "model.py"
    script.write_text(
        "import warnings, time\n"
        "warnings.simplefilter('ignore')\n"
        "import cash\n"
        f"c = cash.Cash(cache_dir={str(tmp_path / '.cash')!r}, register_magic=False)\n"
        "@c.cache\n"
        "def ray_component(i):\n"
        "    time.sleep(0.15)\n"
        "    return i\n"
        "ray_component(1)\n"
        "print('KEY:', next(iter(c.functions)))\n",
        encoding="utf-8",
    )
    cp = subprocess.run([sys.executable, str(script)], capture_output=True,
                        text=True, cwd=str(tmp_path), timeout=180)
    names = [ln.split("KEY:", 1)[1].strip()
             for ln in cp.stdout.splitlines() if ln.startswith("KEY:")]
    assert names, (
        "the probe observed no registered function, so it proves nothing:\n"
        f"stdout={cp.stdout!r}\nstderr={cp.stderr[-500:]!r}"
    )
    name = names[0]
    assert name == "model.ray_component", (
        f"a script's cached function is named {name!r}. The docs' `cash inspect` "
        "and CASH_SUMMARY transcripts show a `model.` prefix on the strength of "
        "this; if the __main__ resolution changed, those pages need updating too."
    )


# --------------------------------------------------------------------------- #
# Ambient reads  (docs/decorator.md "strict= and assume_safe= - purity gates", #
#                 docs/warnings.md#key-ambient-read)                          #
# --------------------------------------------------------------------------- #

def test_decorator_doc_ambient_read_bullet_actually_fires(tmp_path):
    """The doc says `datetime.now()` gets a `CashImpurityWarning`. Run it.

    This claim was FALSE for as long as it had been written: the bullet listed
    `datetime.now` among the impure calls that warn, and the analyzer had no
    rule for it at all. The claim anchor beside that bullet did not catch it --
    an anchor pins the doc to source that CHANGED, and this source never
    changed, it was never written. Only executing the sentence finds that.
    """
    from cash.exceptions import CashImpurityWarning

    c = _cash(tmp_path)
    ran = []
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")

        @c.cache
        def stamped():
            import datetime
            ran.append("call")
            return datetime.datetime.now().year

        stamped()

    impurity = [w for w in rec if issubclass(w.category, CashImpurityWarning)]
    assert impurity, "docs/decorator.md promises a CashImpurityWarning here"
    text = "\n".join(str(w.message) for w in impurity)
    assert "KEY-AMBIENT-READ" in text, (
        f"docs/warnings.md documents this code for the ambient reads:\n{text}"
    )
    # The other half of the same bullet: warned, and STILL CACHED. Counted, not
    # compared -- two calls a second apart return the same year either way.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stamped()
    assert len(ran) == 1, "the doc says the function is still cached"


# --------------------------------------------------------------------------- #
# Round-18 docs sweep: each test executes one sentence the docs now make.     #
# --------------------------------------------------------------------------- #

def _counted(c, calls, **decorator):
    @c.cache(assume_safe=True, **decorator)
    def value_of(v):
        calls.append(v)
        return v
    return value_of


def test_a_float_and_an_np_float64_are_two_entries(tmp_path):
    """decorator.md "What else is in the key": `0.5` and `np.float64(0.5)`."""
    np = pytest.importorskip("numpy")
    calls = []
    f = _counted(_cash(tmp_path), calls)
    f(0.5)
    f(np.float64(0.5))
    f(0.5)
    assert len(calls) == 2


def test_a_changed_default_recomputes_a_call_that_passes_the_argument(tmp_path):
    """decorator.md: "Calls that pass `k` explicitly recompute too"."""
    c = _cash(tmp_path)
    calls = []

    def scaled(x, k=3):
        calls.append(k)
        return x * k

    f = c.cache(scaled, assume_safe=True)
    f(2, k=5)
    f(2, k=5)
    assert calls == [5], "control: an unchanged function hits"
    scaled.__defaults__ = (4,)          # what editing `K` does to the function
    f(2, k=5)
    assert calls == [5, 5]


def test_a_data_file_switched_back_recomputes_and_code_would_not(tmp_path):
    """decorator.md: data versions overwrite one entry (every switch recomputes)."""
    c = _cash(tmp_path)
    data = tmp_path / "data.csv"
    calls = []

    @c.cache(assume_safe=True)
    def load(path):
        calls.append(1)
        with open(path) as fh:
            return fh.read()

    for content in ("v1", "v2", "v1"):
        data.write_text(content, encoding="utf-8")
        assert load(str(data)) == content
    assert len(calls) == 3


def test_a_relative_path_from_two_directories_is_right_and_recomputes(tmp_path, monkeypatch):
    """decorator.md: the same "data.csv" from two cwds -- correct answers, and
    every switch recomputes."""
    c = _cash(tmp_path / ".cash")
    for name, content in (("a", "AAA"), ("b", "BBBB")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "data.csv").write_text(content, encoding="utf-8")
    calls = []

    @c.cache(assume_safe=True)
    def read(path):
        calls.append(1)
        with open(path) as fh:
            return fh.read()

    for name, expected in (("a", "AAA"), ("b", "BBBB"), ("a", "AAA")):
        monkeypatch.chdir(tmp_path / name)
        assert read("data.csv") == expected
    assert len(calls) == 3


def test_the_parameter_sweep_counts(tmp_path):
    """scientific-computing.md: 12 runs, then 12 hits, then 3 for a new alpha."""
    c = _cash(tmp_path)
    calls = []

    @c.cache(assume_safe=True)
    def simulate(n, dt, alpha, seed):
        calls.append((alpha, seed))
        return alpha * seed

    def sweep(alphas):
        for alpha in alphas:
            for seed in range(3):
                simulate(1_000, 0.01, alpha, seed)

    sweep([0.1, 0.5, 1.0, 2.0])
    assert len(calls) == 12
    sweep([0.1, 0.5, 1.0, 2.0])
    assert len(calls) == 12
    sweep([0.1, 0.5, 1.0, 2.0, 4.0])
    assert len(calls) == 15


def test_cache_if_decides_writes_not_what_is_served(tmp_path):
    """decorator.md: a None stored before `cache_if` was added is still served."""
    c = _cash(tmp_path)
    calls = []

    def lookup(x):
        calls.append(x)
        return None

    c.cache(lookup, assume_safe=True)(1)
    c.cache(lookup, assume_safe=True, cache_if=lambda r: r is not None)(1)
    assert calls == [1], "adding cache_if changed what an existing entry serves"


def test_a_hit_returns_a_copy_where_a_miss_returned_a_view(tmp_path):
    """decorator.md "A hit returns a copy"."""
    np = pytest.importorskip("numpy")
    c = _cash(tmp_path)

    @c.cache(assume_safe=True)
    def head(a):
        return a[:3]

    first = np.arange(10)
    head(first)[0] = 99
    assert first[0] == 99, "on a miss the result is the function's own view"
    second = np.arange(10)
    head(second)[0] = 99
    assert second[0] == 0, "on a hit the result is rebuilt from the stored bytes"


def test_clearing_an_inner_function_leaves_its_caller_serving(tmp_path):
    """cli.md `cash clear --function`: the caller is keyed on the inner code,
    so clearing the inner function does not make its caller recompute."""
    import subprocess
    import sys
    import textwrap

    script = tmp_path / "pipeline.py"
    script.write_text(textwrap.dedent("""
        import sys, time
        import cash

        @cash.cache(assume_safe=True)
        def inner(x):
            print("RUN inner", file=sys.stderr)
            time.sleep(0.15)
            return x

        @cash.cache(assume_safe=True)
        def outer(x):
            print("RUN outer", file=sys.stderr)
            time.sleep(0.15)
            return inner(x) + 1

        print(outer(1))
    """), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env.update(CASH_CACHE_DIR=str(tmp_path / ".cash"), PYTHONDONTWRITEBYTECODE="1")

    def run(*argv):
        return subprocess.run([sys.executable, *argv], cwd=str(tmp_path), env=env,
                              capture_output=True, text=True, timeout=120)

    first = run(str(script))
    assert "RUN inner" in first.stderr and "RUN outer" in first.stderr
    cleared = run("-m", "cash", "clear", "--function", "inner", str(tmp_path / ".cash"))
    assert "Cleared 1 entry" in cleared.stdout, cleared.stdout + cleared.stderr
    again = run(str(script))
    assert again.stdout.strip() == "2"
    assert "RUN" not in again.stderr, "clearing the inner function recomputed its caller"


def test_the_debug_line_format_matches_the_guide(tmp_path):
    """debugging-and-monitoring.md: the `cash.calls:` lines, id and all."""
    import subprocess
    import sys
    import textwrap

    script = tmp_path / "model.py"
    script.write_text(textwrap.dedent("""
        import cash

        @cash.cache(assume_safe=True)
        def build_grid(n):
            return list(range(n))

        build_grid(3)
        build_grid(3)
    """), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env.update(CASH_DEBUG="1", CASH_CACHE_DIR=str(tmp_path / ".cash"))
    err = subprocess.run([sys.executable, str(script)], cwd=str(tmp_path), env=env,
                         capture_output=True, text=True, timeout=120).stderr
    lines = [line for line in err.splitlines() if line.startswith("cash.calls:")]
    assert re.match(r"cash\.calls: MISS model\.build_grid  \[[0-9a-f]{12}\]  no entry yet", lines[0]), lines
    assert re.match(r"cash\.calls: HIT  model\.build_grid  \[[0-9a-f]{12}\]  \(saved ", lines[1]), lines
