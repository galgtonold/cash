"""A module-level sentinel returned by a cached function is that sentinel on a hit.

``MISSING = object()`` and ``return d.get(key, MISSING)``: a hit is a copy,
and a copy of an object compared by identity is a different object, so
``lookup(k) is MISSING`` was False on every hit and in every later process.
A result that IS a module global or closure variable of an identity-compared
type is handed back as that variable's object, and CACHE-RESULT-SHARED no
longer claims the hit will be a separate object.
"""

from __future__ import annotations

import textwrap

from tests._scripts import run_python

SCRIPT = textwrap.dedent(
    """
    import cash

    MISSING = object()

    class _Unset:
        pass

    UNSET = _Unset()

    @cash.cache
    def lookup(key):
        return {"a": [1]}.get(key, MISSING)

    def make(default):
        @cash.cache
        def pick(key):
            return {"a": 1}.get(key, default)
        return pick

    @cash.cache
    def fresh():
        return _Unset()

    pick = make(UNSET)
    for _ in range(2):
        assert lookup("zz") is MISSING
        assert lookup("a") == [1]
        assert pick("zz") is UNSET
        assert fresh() is not UNSET
    print(lookup.cache_info()["hits"], pick.cache_info()["hits"])
    """
)


def test_a_sentinel_is_itself_on_ram_and_disk_hits(tmp_path):
    script = tmp_path / "job.py"
    script.write_text(SCRIPT, encoding="utf-8")
    hits = []
    for _ in range(2):
        p = run_python(script, cwd=tmp_path, cache_dir=tmp_path / "cache")
        hits.append(p.stdout.strip().splitlines()[-1])
        # A hit hands back the global itself, so "they will be separate
        # objects" would be false.
        assert "CACHE-RESULT-SHARED" not in p.stderr, p.stderr
    assert hits == ["2 1", "4 2"], "the second process did not hit the disk entries"
