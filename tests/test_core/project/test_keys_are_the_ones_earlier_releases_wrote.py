"""A cache written by an earlier release is read by this one.

A key is found again only when it is built to the same bytes. The digests
below were written down from a released cash; a change that moves one of
them leaves every cache on disk unread, silently, unless the cache format
(``cache_dir.CACHE_FORMAT_VERSION``) moves with it. So when a test here
fails: either the change was not meant to move keys, or bump the format and
write the new digests down.

Only keys that are the same on every supported Python and platform are
pinned: the value hash, and a notebook statement's key. A decorated
function's key folds in the running interpreter (``sys.version``), so no
digest of one can be written down here.
"""

from __future__ import annotations

import pytest

from cash.notebook.cache_key import CacheKeyContext, compute_cache_key
from cash.value_hash import compute_hash

# The value hash is BLAKE3 where it loads (a dependency on CPython) and
# SHA-256 where it does not: the digests are the BLAKE3 ones.
pytest.importorskip("blake3")

VALUES = {
    "an int": (7, "20aef26d0b1917604dd2462e3d82a1bc1d0ce7d3981bb4aeadc20302096261b7"),
    "a text": ("héllo", "67f597040f48b5f5bc04456fa40362038ea895e0c5267def612898b6d2284f79"),
    "a nested collection": (
        {"a": [1, 2.5, None], "b": ("x", b"y", frozenset({1, 2}))},
        "b81f3ee20bd48c6296acae8b7e3c5278ae950f730d5d70c227bca7917871e27e",
    ),
    "a slice": (slice(1, 5, 2), "76ab02015197891b69a8029d67d597909b4e2daff03c9a8b403cf66398783afa"),
    "a big int": (10**40, "67a73364009cb5317975ea97142846d668ff3e325765ed32bc7528a5ac496a67"),
    "a complex": (3 + 4j, "52478f7b9dc42b2f80ee6f5b8b2dd931425b2c3850df2e676cbf01d17486c445"),
    "a range": (range(3, 30, 3), "eebecf747cfc4b383529cb931dfad4607f87d8e57df7257f1b1699a9f546a807"),
    "a bytearray": (bytearray(b"abc"), "6f4de63ef30b126418b693e724632a26a79ee3a8ad4bf74cbb7d3bb94e2dd9d2"),
}

SOURCE = (
    "def scale(v):\n"
    "    return v * 2\n"
    "class Box:\n"
    "    def __init__(self, items):\n"
    "        self.items = items\n"
    "    def fit(self):\n"
    "        return scale(len(self.items))\n"
)


@pytest.mark.parametrize("name", sorted(VALUES))
def test_a_value_hashes_as_it_did(name):
    value, digest = VALUES[name]
    assert compute_hash(value) == digest


def _context() -> CacheKeyContext:
    ns: dict = {"__name__": "__main__"}
    exec(compile(SOURCE, "<cell>", "exec"), ns)
    ns["pre"] = ns["Box"]([1, 2, 3])
    ns["n"] = 4
    lineages = {"Box": "box0", "scale": "scale0", "pre": "pre0", "n": "n0"}
    return CacheKeyContext(variable_lineage=lineages, user_ns=ns)


def test_a_statement_calling_a_method_keys_as_it_did():
    key = compute_cache_key("res = pre.fit()", {"pre"}, ctx=_context(), outputs={"res"}).cache_key
    assert key == "stmt:addcb1c4be2b3450504f26fa899e55ae41f70180cc526cd3ace49710c62ecaf8"


def test_a_statement_calling_a_function_keys_as_it_did():
    key = compute_cache_key("m = scale(n) + 1", {"scale", "n"}, ctx=_context(), outputs={"m"}).cache_key
    assert key == "stmt:7e44f901615907c229735ccdd567276c861c052200f7206c07442b01b01cbd2f"


def test_a_later_occurrence_in_another_namespace_keys_as_it_did():
    key = compute_cache_key(
        "m = scale(n) + 1", {"scale", "n"}, ctx=_context(), outputs={"m"}, occurrence_index=2, namespace="loop"
    ).cache_key
    assert key == "loop:76d77d3e9189ddb27df89feed60c4e5a6526399d52e46ed9cd1f6aa4b78db224"
