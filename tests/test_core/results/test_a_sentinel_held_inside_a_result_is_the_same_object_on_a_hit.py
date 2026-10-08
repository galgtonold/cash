"""A module-level sentinel held INSIDE a cached result is that sentinel on a hit.

``[TABLE.get(k, MISSING) for k in keys]``: a hit is a copy, and the copy of
``MISSING`` inside it was a different object, so ``v is MISSING`` held on
the call that computed the list and failed on every RAM and disk hit. Only a
result that IS the sentinel was handed back as itself.
"""

from __future__ import annotations

import textwrap

from tests._scripts import run_python

SCRIPT = textwrap.dedent(
    """
    import collections, dataclasses
    import cash

    MISSING = object()
    DEFAULT = object()

    class _Unset:
        pass

    UNSET = _Unset()
    TABLE = {"a": 1, "c": 3}
    Pair = collections.namedtuple("Pair", "key value")

    @dataclasses.dataclass
    class Row:
        key: str
        value: object

    @cash.cache
    def lookup(keys):
        return [TABLE.get(k, MISSING) for k in keys]

    @cash.cache
    def nested(keys):
        return {"rows": [(k, TABLE.get(k, MISSING), DEFAULT) for k in keys],
                "pairs": [Pair(k, TABLE.get(k, UNSET)) for k in keys],
                "objs": [Row(k, TABLE.get(k, MISSING)) for k in keys]}

    def make(default):
        @cash.cache
        def pick(keys):
            return [TABLE.get(k, default) for k in keys]
        return pick

    @cash.cache
    def own_token(keys):
        token = object()  # not a global: a hit's copy is a new object, as a new call's is
        return [token] + [TABLE.get(k, MISSING) for k in keys]

    pick = make(UNSET)
    keys = ("a", "b", "c")
    for _ in range(2):
        assert [v is MISSING for v in lookup(keys)] == [False, True, False]
        got = nested(keys)
        assert [(r[1] is MISSING, r[2] is DEFAULT) for r in got["rows"]] == [(False, True), (True, True), (False, True)]
        assert [p.value is UNSET for p in got["pairs"]] == [False, True, False]
        assert type(got["pairs"][0]) is Pair
        assert [r.value is MISSING for r in got["objs"]] == [False, True, False]
        assert [v is UNSET for v in pick(keys)] == [False, True, False]
        got = own_token(keys)
        assert got[0] is not MISSING and type(got[0]) is object
        assert [v is MISSING for v in got[1:]] == [False, True, False]
    print(lookup.cache_info()["hits"], nested.cache_info()["hits"], pick.cache_info()["hits"])
    """
)


def test_a_held_sentinel_is_itself_on_ram_and_disk_hits(tmp_path):
    script = tmp_path / "job.py"
    script.write_text(SCRIPT, encoding="utf-8")
    hits = []
    for _ in range(2):
        p = run_python(script, cwd=tmp_path, cache_dir=tmp_path / "cache")
        assert p.returncode == 0, p.stderr
        hits.append(p.stdout.strip().splitlines()[-1])
    assert hits == ["1 1 1", "2 2 2"], "the second process did not hit the disk entries"
