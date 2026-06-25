"""Finding #8: explain() should make a dynamic-dependency miss self-explanatory.

A file tracked via ``dynamic_depends_on`` that changes produces a NEW cache key,
so the miss surfaces as ``no_entry`` rather than ``file_changed``. The
explanation must say so and list the tracked dynamic dependencies, instead of
only "first call with these arguments".
"""
from __future__ import annotations

import time

from cash import Cash, FileBackend, FileDataSource


def _build(c: Cash, base_dir):
    def dep(name):
        return FileDataSource(str(base_dir / f"{name}.parquet"))

    @c.cache(dynamic_depends_on=dep)
    def load(name):
        with open(base_dir / f"{name}.parquet") as f:
            return f.read()

    return load


def test_dynamic_dep_change_is_explained(tmp_path):
    (tmp_path / "AAA.parquet").write_text("v1")
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path / "cache")))
    load = _build(c, tmp_path)

    load("AAA")
    assert load.explain("AAA").reason == "hit"

    time.sleep(0.02)
    (tmp_path / "AAA.parquet").write_text("v2-changed")

    e = load.explain("AAA")
    assert e.reason == "no_entry"
    # The details now carry the tracked dynamic dependency...
    dyn = e.details.get("dynamic_dependencies")
    assert dyn and any("AAA.parquet" in d for d in dyn), dyn
    # ...and the hint explains the dynamic-dep -> new-key behaviour.
    assert "dynamic" in e.details["hint"].lower()


def test_no_dynamic_section_when_not_used(tmp_path):
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path / "cache")))

    @c.cache
    def f(x):
        return x * 2

    e = f.explain(5)            # first call -> no_entry, but no dynamic deps
    assert e.reason == "no_entry"
    assert "dynamic_dependencies" not in e.details
