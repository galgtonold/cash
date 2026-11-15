"""A settings object read through its module is keyed by the attributes read.

``from m import settings; settings.n_jobs`` keyed ``settings`` by the
attributes read off it, but ``m.settings.n_jobs`` keyed the whole object,
which holds the time it was made: every new process missed. The rule is the
same through a module (``m.settings.a.b``, ``import pkg.m as mm;
mm.settings.x``, ``pkg.m.settings.x``): every load that reaches the object
heads an attribute chain, or it is keyed whole.

Each case is run with and without cash, and must print the same.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

CONF = textwrap.dedent("""
    import json, sys, time


    class Preset:
        def __init__(self, flavor):
            self.flavor = flavor


    class Settings:
        def __init__(self, conf):
            self.n = conf["n"]
            self.m = conf.get("m", 1)
            self.opts = {"k": conf.get("k", 1)}
            self.preset = Preset(conf.get("flavor", "a"))
            self.started = conf.get("started", time.time())


    settings = Settings(json.loads(sys.argv[1]))
""")

HEAD = textwrap.dedent("""
    import json, sys
    import cash

    CONF = json.loads(sys.argv[1])
""")

READERS = {
    "module": """
        import conf

        @cash.cache
        def f(x):
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe
            return x * conf.settings.n + conf.settings.opts["k"] + len(conf.settings.preset.flavor)
    """,
    "package_alias": """
        import pkg.conf as cc

        @cash.cache
        def f(x):
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe
            return x * cc.settings.n + len(cc.settings.preset.flavor)
    """,
    "package_path": """
        import pkg.conf

        @cash.cache
        def f(x):
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe
            return x * pkg.conf.settings.n + len(pkg.conf.settings.preset.flavor)
    """,
    "mixed": """
        import conf
        from conf import settings

        @cash.cache
        def f(x):
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe
            return x * settings.n + conf.settings.m * 10 + len(conf.settings.preset.flavor)
    """,
    # Whole: the object handed on, the module handed on.
    "handed_on": """
        import conf

        def helper(s, x):
            return x * s.n


        @cash.cache
        def f(x):
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe
            return helper(conf.settings, x)
    """,
    "module_handed_on": """
        import conf

        def helper(mod, x):
            return x * mod.settings.n


        @cash.cache
        def f(x):
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe
            return helper(conf, x) + conf.settings.m
    """,
}

#: How each reader names the module that holds ``settings``.
MODULE = {
    "module": "conf",
    "package_alias": "cc",
    "package_path": "pkg.conf",
    "mixed": "conf",
    "handed_on": "conf",
    "module_handed_on": "conf",
}

TAIL = """
print(f(2))
if CONF.get("set"):
    {mod}.settings.n = 9
    {mod}.settings.m = 9
    print(f(2))
if CONF.get("mutate"):
    {mod}.settings.opts["k"] = 50
    {mod}.settings.preset.flavor = "long"
    print(f(2))
if CONF.get("unread"):
    {mod}.settings.started = 0.5
    print(f(2))
"""


def _script(reader: str, cached: bool) -> str:
    text = HEAD + textwrap.dedent(READERS[reader]) + TAIL.format(mod=MODULE[reader])
    return text if cached else text.replace("import cash\n", "").replace("@cash.cache\n", "")


def _run(tmp_path, reader, conf, cached=True):
    (tmp_path / "conf.py").write_text(CONF, encoding="utf-8")
    (tmp_path / "pkg").mkdir(exist_ok=True)
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "conf.py").write_text(CONF, encoding="utf-8")
    (tmp_path / "main.py").write_text(_script(reader, cached), encoding="utf-8")
    proc = run_python("main.py", json.dumps(conf), cwd=tmp_path)
    return proc.stdout.split(), proc.stderr.count("[RUN]")


def _same_as_plain(tmp_path, reader, conf):
    plain, _ = _run(tmp_path, reader, conf, cached=False)
    out, runs = _run(tmp_path, reader, conf)
    assert out == plain, f"{reader} {conf}: {out} where plain Python prints {plain}"
    return runs


@pytest.mark.parametrize("reader", ["module", "package_alias", "package_path", "mixed"])
def test_an_attribute_no_reader_reads_does_not_key_it(tmp_path, reader):
    assert _same_as_plain(tmp_path, reader, {"n": 3, "started": 1.0}) == 1
    assert _same_as_plain(tmp_path, reader, {"n": 3, "started": 2.0}) == 0, "the start time is read by no one"
    assert _same_as_plain(tmp_path, reader, {"n": 4, "started": 3.0}) == 1, "n is read"
    assert _same_as_plain(tmp_path, reader, {"n": 4, "flavor": "bb", "started": 4.0}) == 1, "a nested chain is read"


def test_each_attribute_read_through_the_module_or_by_name_keys_it(tmp_path):
    assert _same_as_plain(tmp_path, "mixed", {"n": 3, "m": 1, "started": 1.0}) == 1
    assert _same_as_plain(tmp_path, "mixed", {"n": 3, "m": 2, "started": 2.0}) == 1, "read through the module"
    assert _same_as_plain(tmp_path, "mixed", {"n": 5, "m": 2, "started": 3.0}) == 1, "read by name"
    assert _same_as_plain(tmp_path, "mixed", {"n": 5, "m": 2, "k": 7, "started": 4.0}) == 0, "read by no one"


@pytest.mark.parametrize("reader", ["handed_on", "module_handed_on"])
def test_any_other_use_keys_it_whole(tmp_path, reader):
    assert _same_as_plain(tmp_path, reader, {"n": 3, "started": 1.0}) == 1
    assert _same_as_plain(tmp_path, reader, {"n": 3, "started": 2.0}) == 1
    assert _same_as_plain(tmp_path, reader, {"n": 5, "started": 2.0}) == 1


@pytest.mark.parametrize("reader", READERS)
def test_a_changed_value_recomputes_in_the_process(tmp_path, reader):
    _same_as_plain(tmp_path, reader, {"n": 3, "started": 1.0, "set": True})
    _same_as_plain(tmp_path, reader, {"n": 3, "started": 1.0, "mutate": True})


@pytest.mark.parametrize("reader", ["module", "package_alias", "package_path", "mixed"])
def test_an_unread_attribute_changed_in_the_process_still_hits(tmp_path, reader):
    _same_as_plain(tmp_path, reader, {"n": 3, "started": 1.0})
    assert _same_as_plain(tmp_path, reader, {"n": 3, "started": 2.0, "unread": True}) == 0
