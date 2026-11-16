"""A docstring an object was given at run time is keyed by what it says.

xclim writes each indicator's ``__doc__`` in ``__init__`` from what the
indicator holds: a parameter's choices (a set, rendered in the order of the
process's string hashing) and a description given as a lambda (rendered
with its address). Keyed as written, the docstring moved with every new
process, so every call reaching an indicator through module data missed.

An object's own docstring is an input only where its class's code reads a
docstring, as a class's is; then a set or an address it renders of a value
the object or its class holds is written the same in every process, and
any other change to the text still changes the key.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.timeout(300)]

IND = textwrap.dedent("""
    import os


    class Indicator:
        choices = {"clip", "mask", "keep", "drop"}
        describe = staticmethod(lambda: "a description")

        def __init__(self, factor):
            self.factor = factor
            self.__doc__ = f"{os.environ['WORD']} by {factor}. Choices: {self.choices}. See {self.describe}"

        def __call__(self, x):
            return x * self.factor


    class Documented(Indicator):
        def __call__(self, x):
            return x * self.factor + len(self.__doc__)


    SCALE = Indicator(int(os.environ["FACTOR"]))
    REGISTRY = {"scale": SCALE}
    DOCUMENTED = Documented(int(os.environ["FACTOR"]))
""")

MAIN = textwrap.dedent("""
    import cash
    import ind

    c = cash.Cash(cache_dir=".cash")


    @c.cache
    def run(x):
        return ind.SCALE(x)


    @c.cache
    def run_get(x):
        return ind.REGISTRY.get("scale")(x)


    @c.cache
    def run_documented(x):
        return ind.DOCUMENTED(x)


    for f in (run, run_get, run_documented):
        print(f(5), f.cache_info()["hits"])
""")


def _run(tmp_path, seed, factor="2", word="Scale"):
    proc = run_python("main.py", cwd=tmp_path, env={"PYTHONHASHSEED": str(seed), "FACTOR": factor, "WORD": word})
    return [tuple(map(int, line.split())) for line in proc.stdout.strip().splitlines()]


def test_the_next_process_hits_and_a_change_misses(tmp_path):
    (tmp_path / "ind.py").write_text(IND, encoding="utf-8")
    (tmp_path / "main.py").write_text(MAIN, encoding="utf-8")
    first = _run(tmp_path, 1)
    assert [hits for _, hits in first] == [0, 0, 0]
    # Another string hashing, another address for the lambda: the same text.
    assert _run(tmp_path, 2) == [(value, 1) for value, _ in first]
    assert _run(tmp_path, 3) == [(value, 1) for value, _ in first]
    # What the object holds changed: every call recomputes.
    assert [hits for _, hits in _run(tmp_path, 4, factor="3")] == [0, 0, 0]
    # The docstring changed: the call whose code reads it computes the new
    # value.
    changed = _run(tmp_path, 5, word="Multiply")
    assert changed[2] == (first[2][0] + len("Multiply") - len("Scale"), 0)


class _Holder:
    choices = frozenset({"b", "a"})
    keep = staticmethod(lambda: None)

    def __init__(self, doc):
        self.__doc__ = doc


def test_only_renderings_of_held_values_are_made_the_same():
    from cash.generated_docs import doc_form

    held = _Holder("")
    address = f"{id(_Holder.keep):x}"
    assert doc_form(held, "{'b', 'a'} and <function <lambda> at 0x" + address + ">") == (
        "{'a', 'b'} and <function <lambda> at <held function>>"
    )
    # A list in the text that is not a held set keeps its order, an address
    # of nothing held keeps its digits, and a reordered sentence is new text.
    assert doc_form(held, "{'c', 'a'}") == "{'c', 'a'}"
    assert doc_form(held, "at 0x1") == "at 0x1"
    assert doc_form(held, "a then b") != doc_form(held, "b then a")


def test_a_docstring_counts_as_read_only_by_a_load():
    from cash.generated_docs import reads_own_doc

    class Writes:
        def __init__(self):
            self.__doc__ = "x"

    class Reads(Writes):
        def show(self):
            return getattr(self, "__doc__")

    assert not reads_own_doc(Writes)
    assert reads_own_doc(Reads)
