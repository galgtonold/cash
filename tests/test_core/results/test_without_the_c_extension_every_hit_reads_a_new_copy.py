"""Without the C extension every hit of a value kept as pickle bytes reads
a new copy: nothing can prove the last one untouched and unheld, so it is
never handed back (`cash.handed_copies`)."""

from __future__ import annotations

import textwrap

from tests._scripts import run_python

SCRIPT = textwrap.dedent(
    """
    from abc import ABC

    import cash
    from cash import value_watch

    assert not value_watch.enabled()
    LOADS = []


    class Trie(ABC):
        class Node:
            __slots__ = ("children", "end")

            def __init__(self):
                self.end = False
                self.children = {}

        def __init__(self, n):
            self.root = Trie.Node()
            for i in range(n):
                node = self.root
                for ch in str(i):
                    node = node.children.setdefault(ch, Trie.Node())
                node.end = True

        def __setstate__(self, state):
            LOADS.append(1)
            self.__dict__.update(state)


    @cash.cache
    def word_dict_trie():
        return Trie(5000)


    word_dict_trie()
    for _ in range(3):
        before = len(LOADS)
        trie = word_dict_trie()
        assert len(LOADS) == before + 1, "each hit reads a new copy"
        assert sorted(trie.root.children) == list("0123456789")
        del trie
    print("ok")
    """
)


def test_each_hit_reads_a_new_copy(tmp_path):
    (tmp_path / "main.py").write_text(SCRIPT, encoding="utf-8")
    done = run_python("main.py", cwd=tmp_path, env={"CASH_PURE_PYTHON": "1"})
    assert done.stdout.strip().endswith("ok"), done.stdout + done.stderr
