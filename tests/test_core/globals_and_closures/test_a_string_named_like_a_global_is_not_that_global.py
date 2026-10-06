"""A string that happens to spell a module global's name is not a read of it.

``frame["x"]`` or ``f"t{i}"`` in a cached body put the module's unrelated
``x`` or ``t`` in the key: every hit hashed it in full (a large array made a
sub-millisecond hit take tens of milliseconds), and the function recomputed
whenever that global moved -- a loop variable ``t`` ran the body on every
pass. A string reaches the namespace only through code that reads it by
name (``globals()[name]``), the function's own or a helper's; then it is
still a read.
"""

from __future__ import annotations

import textwrap

from tests._scripts import run_python

SCRIPT = """
    import cash

    cash.configure(cache_dir="cache")
    COUNT = [0]


    def get(name):
        return globals()[name]


    @cash.cache(assume_safe=True)
    def names(n):
        COUNT[0] += 1
        return [f"t{i}" for i in range(n)] + [{"x": 1}["x"]]


    @cash.cache
    def by_helper(n):
        return n * get("t")


    for t in range(3):
        names(2)
        print("helper", by_helper(10))
    print("names ran", COUNT[0])
"""


def test_a_string_matching_a_global_does_not_key_on_it(tmp_path):
    (tmp_path / "main.py").write_text(textwrap.dedent(SCRIPT), encoding="utf-8")
    out = run_python("main.py", cwd=tmp_path).stdout.split("\n")
    assert "names ran 1" in out, out


def test_a_string_a_helper_reads_by_name_is_still_a_read(tmp_path):
    """The control: ``get("t")`` reads ``t`` through ``globals()``, so each
    new ``t`` recomputes."""
    (tmp_path / "main.py").write_text(textwrap.dedent(SCRIPT), encoding="utf-8")
    out = run_python("main.py", cwd=tmp_path).stdout.split("\n")
    assert [line for line in out if line.startswith("helper")] == ["helper 0", "helper 10", "helper 20"], out
