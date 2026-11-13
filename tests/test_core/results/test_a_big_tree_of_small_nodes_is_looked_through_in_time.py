"""A result that is a big tree of small objects is looked through for a
function sharing a part of it (STORE-PARTS-SHARED) in time, and stored.

A word trie of a few hundred thousand nodes (a tokenizer's dictionary) was
read a node at a time: its slots, its attribute dict and, for a library's
class, its ``__reduce_ex__``, each through a few Python calls. The look
took four times as long as building the trie, ran past its time and the
result was never stored (STORE-PARTS-UNCHECKED): every call built it again,
and paid for the look on top. The objects of one class are now read a list
at a time, so the look finishes; a function in the result whose closure
holds one node of the tree is still found.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from pathlib import Path

from cash import closure_sharing
from cash.closure_sharing import closures_sharing

#: Builds a trie of slotted nodes over generated words: about 300,000 nodes.
TRIE_SOURCE = textwrap.dedent(
    """
    class Node:
        __slots__ = ("end", "children")

        def __init__(self):
            self.end = False
            self.children = None


    class Trie:
        def __init__(self, words):
            self.root = Node()
            self.size = 0
            for word in words:
                node = self.root
                for ch in word:
                    if node.children is None:
                        node.children = {}
                    child = node.children.get(ch)
                    if child is None:
                        child = node.children[ch] = Node()
                        self.size += 1
                    node = child
                node.end = True


    def words(n):
        return ["".join(chr(97 + (i * 2654435761 >> 3 * k) % 26) for k in range(6 + i % 6)) for i in range(n)]
    """
)

namespace: dict = {}
exec(compile(TRIE_SOURCE, "<trie>", "exec"), namespace)
Node, Trie, words = namespace["Node"], namespace["Trie"], namespace["words"]


class Leaf:
    """A node with an attribute dict, no slots."""

    def __init__(self, value):
        self.value = value
        self.kids = []


class Mixed:
    __slots__ = ("a", "b", "__dict__")


def _built(n: int = 60_000):
    trie = Trie(words(n))
    assert trie.size > 250_000
    return trie


def _a_deep_node(trie):
    node = trie.root
    while node.children:
        node = next(iter(node.children.values()))
    return node


def test_a_trie_of_small_nodes_is_looked_through_within_what_building_it_took():
    started = time.process_time()
    trie = _built()
    cost = time.process_time() - started
    started = time.process_time()
    found = closures_sharing({"result": trie}, 60.0)
    looked = time.process_time() - started
    assert found == ([], True)
    # The store gives the look twice what the work took; it took four times.
    assert looked < closure_sharing.CLOSURE_CHECK_FACTOR * cost, (looked, cost)


def test_a_function_closing_over_one_node_of_the_trie_is_still_found():
    trie = _built()
    node = _a_deep_node(trie)
    found = closures_sharing({"result": (trie, lambda: node)}, 60.0)
    assert found.finished and found.names == ["result"]
    # A node no other part holds is not shared.
    lone = Node()
    found = closures_sharing({"result": (trie, lambda: lone)}, 60.0)
    assert found.finished and found.names == []


def test_nodes_with_attribute_dicts_and_empty_slots_are_read_as_one_by_one():
    target = Leaf(1)
    nodes = [Leaf(i) for i in range(5000)]
    nodes[4321].kids.append(target)
    found = closures_sharing({"result": (nodes, lambda: target)}, 60.0)
    assert found.finished and found.names == ["result"]
    in_a_slot = [Mixed() for _ in range(5000)]
    in_a_slot[17].b = target  # every other slot empty
    assert closures_sharing({"result": (in_a_slot, lambda: target)}, 60.0).names == ["result"]
    in_a_dict = [Mixed() for _ in range(5000)]
    in_a_dict[99].extra = [target]
    assert closures_sharing({"result": (in_a_dict, lambda: target)}, 60.0).names == ["result"]
    assert closures_sharing({"result": (in_a_dict, lambda: Leaf(0))}, 60.0).names == []


def test_the_trie_is_stored_and_served_in_the_next_process(tmp_path: Path):
    cache_dir = tmp_path / "cash"
    script = tmp_path / "build_trie.py"
    script.write_text(
        TRIE_SOURCE
        + textwrap.dedent(
            f"""
            import warnings
            from cash import Cash

            c = Cash(cache_dir={str(cache_dir)!r}, register_magic=False)

            @c.cache
            def word_trie(n):
                return Trie(words(n))

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                trie = word_trie(60_000)
            assert trie.size > 250_000
            codes = [getattr(w.message, "code", None) for w in caught]
            assert "STORE-PARTS-UNCHECKED" not in codes, codes
            print("hit" if any(e["cache_hit"] for e in c._calls.entries) else "miss")
            """
        ),
        encoding="utf-8",
    )
    runs = [
        subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=300, cwd=tmp_path)
        for _ in range(2)
    ]
    for run in runs:
        assert run.returncode == 0, run.stderr
    assert [run.stdout.split()[-1] for run in runs] == ["miss", "hit"]


#: What each `Reduced` holds, by its id: it has no attributes or slots.
HELD: dict[int, object] = {}


class Reduced:
    """A library's object with nothing of its own, whose ``__reduce_ex__``
    hands pickle a new list each time: a temporary the walk frees a few
    levels later."""

    __module__ = "json"  # read as a library's class, whose reduce is asked
    __slots__ = ("__weakref__",)

    def __reduce_ex__(self, protocol):
        return (Reduced, (), None, iter([[HELD[id(self)]]]))


def _chain(end, length):
    for _ in range(length):
        link = Reduced()
        HELD[id(link)] = end
        end = link
    return end


def test_a_part_behind_many_temporaries_is_still_found():
    target = Leaf(0)
    chains = [_chain(Leaf(i), 6) for i in range(3000)]
    chains.append(_chain(target, 6))
    try:
        found = closures_sharing({"result": (chains, lambda: target)}, 60.0)
    finally:
        HELD.clear()
    assert found.finished and found.names == ["result"]
