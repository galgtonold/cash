"""Wire-contract characterization for the dependency-state hash.

These tests pin the *serialization* of the dependency-state hash —
the value that becomes the ``state_hash`` segment of a decorator cache
key (``func:state:dynamic:args``). Changing the byte format silently
invalidates every cache on disk, so the format is a contract.

They are deliberately written against the observable produced by Cash
today and must keep producing the same digest after the fold is
extracted into ``cash.dependency_state.DependencyStateHasher``. The
single ``state_hash`` indirection below is the one line that gets
repointed at the new call path once the extraction lands; the asserted
digests do not change.
"""
from __future__ import annotations

import hashlib
import inspect

from cash import Cash
from cash.source_norm import normalize_source_for_hash

from . import _purity_helper_module as hm


def state_hash(c: Cash, name: str) -> str:
    """The one seam repointed after extraction (old method -> compute)."""
    return c._state_hasher.compute(name)


def test_lone_function_hash_is_sha256_of_its_source_hash(tmp_path):
    """A function with no dependencies and no helpers folds to exactly
    ``sha256(source_hash)`` — nothing else enters the join."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    def lonely(x):
        return x + 1

    name = c._get_func_key(lonely)
    lonely(1)  # trigger registration + lazy analysis

    expected = hashlib.sha256(c.source_hashes[name].encode("utf-8")).hexdigest()
    assert state_hash(c, name) == expected


def test_helper_token_format_is_helper_qual_hash(tmp_path):
    """A captured module helper contributes a ``helper:{qual}:{hash}``
    token, ``:``-joined after the function's own source hash, the whole
    thing sha256'd. Reconstructed from the documented resolution path
    using ``inspect.getsource`` (not the private hasher) to stay
    independent of the implementation under test.

    The helper digest is over the NORMALIZED source: a comment or a
    reformat inside a helper must not invalidate its callers. That is a
    deliberate part of the contract, not an implementation detail, so it
    is reconstructed here through the same public normalizer."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    def main(x):
        return hm.helper(x)

    name = c._get_func_key(main)
    main(5)  # trigger analysis -> populates the purity report

    report = c._purity_reports[name]
    assert report.helper_source_hashes, "fixture expected to capture hm.helper"

    import sys as _sys
    live: dict[str, str] = {}
    for qual, (mod_name, attr_chain) in report.helper_resolution_paths.items():
        obj = _sys.modules.get(mod_name)
        for attr in attr_chain:
            obj = getattr(obj, attr, None)
        if callable(obj):
            live[qual] = hashlib.sha256(
                normalize_source_for_hash(inspect.getsource(obj)).encode("utf-8")
            ).hexdigest()

    parts = [c.source_hashes[name]]
    for qual in sorted(report.helper_source_hashes):
        if qual == name:
            continue
        parts.append(f"helper:{qual}:{live.get(qual, report.helper_source_hashes[qual])}")
    expected = hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()

    assert state_hash(c, name) == expected


def test_state_hash_is_stable_across_cash_instances(tmp_path):
    """Two independent Cash instances registering the same source must
    produce an identical state hash — this is the on-disk-cache-validity
    guarantee. Pins purity (no instance identity, mtime, or ordering
    leaks into the digest)."""
    def build(cache_dir):
        c = Cash(cache_dir=str(cache_dir), register_magic=False)

        @c.cache
        def dep(x):
            return x * 10

        @c.cache(depends_on=[dep])
        def parent(x):
            return dep(x) + 1

        parent(2)  # warm + analyze both nodes
        return c, c._get_func_key(parent)

    c1, p1 = build(tmp_path / "a")
    c2, p2 = build(tmp_path / "b")

    assert p1 == p2
    assert state_hash(c1, p1) == state_hash(c2, p2)
