"""Unit tests for the extracted dependency-state seam.

Exercises ``DependencyStateHasher`` (the fold) and
``SysModulesHelperResolver`` (the live-environment port) in isolation,
with plain dicts standing in for Cash's registries and a fake resolver
standing in for the impure helper re-resolution. The fold's whole job
is deterministic serialization, so it is tested without touching Cash,
the cache backend, or sys.modules.
"""
from __future__ import annotations

import hashlib

from cash.dependency_state import DependencyStateHasher, SysModulesHelperResolver
from cash.graph import DependencyGraph
from cash.purity_analyzer import PurityReport

from . import _purity_helper_module as hm


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class FakeResolver:
    """Returns a preset qual->hash map; records the reports it saw."""

    def __init__(self, mapping: dict[str, str] | None = None):
        self._mapping = mapping or {}
        self.seen: list[PurityReport] = []

    def current_hashes(self, report: PurityReport) -> dict[str, str]:
        self.seen.append(report)
        return dict(self._mapping)


def make_hasher(
    *,
    functions=None,
    data_sources=None,
    source_hashes=None,
    purity_reports=None,
    graph=None,
    resolver=None,
):
    return DependencyStateHasher(
        functions=functions if functions is not None else {},
        data_sources=data_sources if data_sources is not None else {},
        source_hashes=source_hashes if source_hashes is not None else {},
        purity_reports=purity_reports if purity_reports is not None else {},
        graph=graph if graph is not None else DependencyGraph(),
        helper_resolver=resolver if resolver is not None else FakeResolver(),
    )


def test_lone_function_folds_to_sha256_of_source_hash():
    h = make_hasher(functions={"f": object()}, source_hashes={"f": "H"})
    assert h.compute("f") == _sha("H")


def test_unknown_node_folds_to_sha256_of_empty_join():
    # Node is neither a function nor a data source, has no deps, no report.
    assert make_hasher().compute("ghost") == _sha("")


def test_dependencies_fold_in_sorted_order():
    g = DependencyGraph()
    for n in ("parent", "a", "b"):
        g.add_node(n)
    g.add_dependency("parent", "b")
    g.add_dependency("parent", "a")

    h = make_hasher(
        functions={"parent": 1, "a": 1, "b": 1},
        source_hashes={"parent": "P", "a": "A", "b": "B"},
        graph=g,
    )
    # Children first computed independently, then folded sorted (a before b).
    expected = _sha(":".join(["P", _sha("A"), _sha("B")]))
    assert h.compute("parent") == expected


def test_visited_short_circuits_cycle():
    g = DependencyGraph()
    g.add_node("a")
    g.add_node("b")
    g.add_dependency("a", "b")
    g.add_dependency("b", "a")  # cycle

    h = make_hasher(
        functions={"a": 1, "b": 1},
        source_hashes={"a": "A", "b": "B"},
        graph=g,
    )
    # Should terminate. b's recursion back into a hits visited -> "".
    result = h.compute("a")
    assert isinstance(result, str) and len(result) == 64


def test_helper_tokens_prefer_resolver_then_fall_back_to_snapshot():
    report = PurityReport(
        helper_source_hashes={"h1": "snap1", "h2": "snap2"},
    )
    resolver = FakeResolver({"h1": "live1"})  # h2 unresolved -> snapshot
    h = make_hasher(
        functions={"f": 1},
        source_hashes={"f": "F"},
        purity_reports={"f": report},
        resolver=resolver,
    )
    expected = _sha(":".join(["F", "helper:h1:live1", "helper:h2:snap2"]))
    assert h.compute("f") == expected
    assert resolver.seen == [report]


def test_node_own_qualname_excluded_from_helper_tokens():
    report = PurityReport(helper_source_hashes={"f": "self", "h": "snap"})
    h = make_hasher(
        functions={"f": 1},
        source_hashes={"f": "F"},
        purity_reports={"f": report},
        resolver=FakeResolver({}),
    )
    # "f" token skipped; only "h" contributes.
    expected = _sha(":".join(["F", "helper:h:snap"]))
    assert h.compute("f") == expected


def test_data_source_node_uses_has_changed_marker():
    class DS:
        def has_changed(self):
            return True

    h = make_hasher(data_sources={"src": DS()})
    assert h.compute("src") == _sha("True")


def test_data_source_node_prefers_mtime_when_available():
    class DS:
        def _get_mtime(self):
            return 1234.5

        def has_changed(self):  # pragma: no cover - mtime path wins
            raise AssertionError("should not be consulted")

    h = make_hasher(data_sources={"src": DS()})
    assert h.compute("src") == _sha("1234.5")


def test_registry_mutation_is_visible_by_reference():
    src = {}
    funcs = {}
    h = make_hasher(functions=funcs, source_hashes=src)
    funcs["late"] = 1
    src["late"] = "H"
    # Hasher borrowed the live dicts, so the late registration is seen.
    assert h.compute("late") == _sha("H")


# --- SysModulesHelperResolver (the live-environment port) -------------------


def test_sysmodules_resolver_hashes_resolved_callable():
    qual = hm.helper.__qualname__
    report = PurityReport(
        helper_source_hashes={qual: "snap"},
        helper_resolution_paths={qual: (hm.__name__, (qual,))},
    )
    resolver = SysModulesHelperResolver(hash_callable=lambda fn: "HASHED")
    assert resolver.current_hashes(report) == {qual: "HASHED"}


def test_sysmodules_resolver_skips_missing_module():
    report = PurityReport(
        helper_source_hashes={"x": "snap"},
        helper_resolution_paths={"x": ("no.such.module", ("x",))},
    )
    resolver = SysModulesHelperResolver(hash_callable=lambda fn: "HASHED")
    assert resolver.current_hashes(report) == {}
