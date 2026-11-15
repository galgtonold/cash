"""A settings object read only through attributes is keyed by those attributes.

Reported against scanpy checked out as the user's code: its ``settings``
object holds the time it was imported and the time it last logged, beside
``n_jobs``, ``verbosity`` and the preset its functions read. Keyed whole,
it changed in every process, and no call was ever served in the next one.
A global every load of which, in a reader's code, heads an attribute chain
(``settings.preset.flavor``, ``getattr(settings, "n")``) is keyed by what
those chains reach, following only data the object holds; any other use
keys it whole, as does an attribute reading would compute.

Each case is run with and without cash, and must print the same.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

HEAD = textwrap.dedent("""
    import json, sys, time
    import cash

    CONF = json.loads(sys.argv[1])


    class Settings:
        extra = 100  # a class attribute an instance may shadow

        def __init__(self, conf):
            self.n = conf["n"]
            self.m = conf.get("m", 1)
            self.opts = {"k": conf.get("k", 1)}
            self.started = conf.get("started", time.time())
            if "extra" in conf:
                self.extra = conf["extra"]


    settings = Settings(CONF)
    NAME = "n"
    BRANCH = bool(CONF.get("branch"))
""")

READERS = {
    "chain": """
        @cash.cache
        def f(x):
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe
            return x * settings.n + settings.opts["k"]
    """,
    "constant_getattr": """
        @cash.cache
        def f(x):
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe
            return x * getattr(settings, "n")
    """,
    "nested_scopes": """
        @cash.cache
        def f(x):
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe

            def inner():
                return settings.m

            scale = (lambda: settings.n)()
            return sum(v * scale for v in [x]) + [settings.opts["k"] for _ in "a"][0] + inner()
    """,
    "branch": """
        @cash.cache
        def f(x):
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe
            if BRANCH:
                return settings.m * 1000
            return x * settings.n
    """,
    "shadowed": """
        @cash.cache
        def f(x):
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe
            return x * settings.n + settings.extra
    """,
    # Whole: a name read at run time, the global handed on.
    "runtime_getattr": """
        @cash.cache
        def f(x):
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe
            return x * getattr(settings, NAME)
    """,
    "handed_on": """
        def helper(s, x):
            return x * s.n


        @cash.cache
        def f(x):
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe
            return helper(settings, x)
    """,
}

TAIL = textwrap.dedent("""
    print(f(2))
    if CONF.get("rebind"):
        settings = Settings({**CONF, "n": CONF["n"] + 10})
        print(f(2))
    if CONF.get("set"):
        settings.n = 9
        settings.m = 9
        print(f(2))
    if CONF.get("mutate"):
        settings.opts["k"] = 50
        print(f(2))
""")


def _script(reader: str, cached: bool) -> str:
    text = HEAD + textwrap.dedent(READERS[reader]) + TAIL
    return text if cached else text.replace("import cash\n", "").replace("@cash.cache\n", "")


def _run(tmp_path, reader, conf, cached=True):
    import json

    (tmp_path / "main.py").write_text(_script(reader, cached), encoding="utf-8")
    proc = run_python("main.py", json.dumps(conf), cwd=tmp_path)
    return proc.stdout.split(), proc.stderr.count("[RUN]")


def _same_as_plain(tmp_path, reader, conf):
    plain, _ = _run(tmp_path, reader, conf, cached=False)
    out, runs = _run(tmp_path, reader, conf)
    assert out == plain, f"{reader} {conf}: {out} where plain Python prints {plain}"
    return runs


@pytest.mark.parametrize("reader", ["chain", "constant_getattr", "nested_scopes", "branch"])
def test_an_attribute_no_reader_reads_does_not_key_it(tmp_path, reader):
    assert _same_as_plain(tmp_path, reader, {"n": 3, "started": 1.0}) == 1
    assert _same_as_plain(tmp_path, reader, {"n": 3, "started": 2.0}) == 0, "the start time is read by no one"
    assert _same_as_plain(tmp_path, reader, {"n": 4, "started": 3.0}) == 1


@pytest.mark.parametrize("reader", ["runtime_getattr", "handed_on"])
def test_any_other_use_keys_it_whole(tmp_path, reader):
    assert _same_as_plain(tmp_path, reader, {"n": 3, "started": 1.0}) == 1
    assert _same_as_plain(tmp_path, reader, {"n": 3, "started": 2.0}) == 1
    assert _same_as_plain(tmp_path, reader, {"n": 5, "started": 2.0}) == 1


def test_a_read_in_a_branch_not_taken_is_keyed(tmp_path):
    _same_as_plain(tmp_path, "branch", {"n": 3, "m": 1, "started": 1.0})
    _same_as_plain(tmp_path, "branch", {"n": 3, "m": 1, "started": 1.0, "branch": True})
    assert _same_as_plain(tmp_path, "branch", {"n": 3, "m": 2, "started": 1.0, "branch": True}) == 1


@pytest.mark.parametrize("reader", READERS)
def test_a_rebound_or_changed_value_recomputes_in_the_process(tmp_path, reader):
    _same_as_plain(tmp_path, reader, {"n": 3, "started": 1.0, "rebind": True})
    _same_as_plain(tmp_path, reader, {"n": 3, "started": 1.0, "mutate": True})
    _same_as_plain(tmp_path, reader, {"n": 3, "started": 1.0, "set": True, "branch": True})
    _same_as_plain(tmp_path, reader, {"n": 3, "started": 1.0, "set": True})


def test_a_class_attribute_shadowed_then_dropped_is_followed(tmp_path):
    """Read from the class, the chain stops at the object, keyed whole;
    once an instance shadows it, by the instance's value."""
    for conf in ({"n": 3}, {"n": 3, "extra": 7}, {"n": 3, "extra": 7}, {"n": 3}, {"n": 3, "extra": 100}):
        _same_as_plain(tmp_path, "shadowed", {**conf, "started": 1.0})
