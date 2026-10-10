"""A global the module binds only after a cached function's first call is
still part of that function's key.

A warm-up call whose branch does not read the constant, with the constant
defined further down the file: the first analysis saw no such global and
left it out of the key for good, so changing it (in the same run, or by
editing the file) served the old result with no warning. The same holds for
a builtin name the module shadows later.
"""

from __future__ import annotations

import textwrap

from tests._scripts import run_python

SAME_RUN = textwrap.dedent("""
    import cash

    @cash.cache
    def scale(x, mode):
        if mode == "a":
            return x * A
        return x

    @cash.cache
    def size(xs):
        return len(xs)

    scale(1, "b")          # A does not exist yet
    A = 2
    r1 = scale(1, "a")
    A = 3
    r2 = scale(1, "a")

    size([1, 2])           # len is the builtin here
    def len(xs):
        return 100
    print(r1, r2, size([1, 2]))
""")


def test_a_global_bound_after_the_first_call_is_keyed_in_the_same_run(tmp_path):
    (tmp_path / "same.py").write_text(SAME_RUN)
    out = run_python("same.py", cwd=tmp_path).stdout.split()
    assert out == ["2", "3", "100"]


def _script(k: int) -> str:
    return textwrap.dedent(f"""
        import cash

        @cash.cache
        def scale(x, mode):
            if mode == "a":
                return x * A
            return x

        scale(1, "b")
        A = {k}
        print(scale(1, "a"))
    """)


def test_editing_a_global_bound_after_the_first_call_recomputes(tmp_path):
    script = tmp_path / "script.py"
    script.write_text(_script(2))
    assert run_python("script.py", cwd=tmp_path).stdout.strip() == "2"
    script.write_text(_script(3))
    assert run_python("script.py", cwd=tmp_path).stdout.strip() == "3"


def test_an_unchanged_late_global_is_still_a_hit(tmp_path):
    # Positive control: the late global is keyed, not keyed differently on
    # every run -- a second run of the same script is served from the cache.
    script = tmp_path / "script.py"
    script.write_text(_script(2) + 'print(scale.cache_info()["hits"])\n')
    assert run_python("script.py", cwd=tmp_path).stdout.split() == ["2", "0"]
    assert run_python("script.py", cwd=tmp_path).stdout.split() == ["2", "2"]


def _helper_script(k: int) -> str:
    return textwrap.dedent(f"""
        import cash

        @cash.cache
        def scale(x, mode):
            if mode == "a":
                return helper(x)
            return x

        scale(1, "b")          # helper is not defined yet

        def helper(x):
            return x * {k}

        print(scale(1, "a"))
    """)


def test_editing_a_helper_defined_after_the_first_call_recomputes(tmp_path):
    script = tmp_path / "script.py"
    script.write_text(_helper_script(2))
    assert run_python("script.py", cwd=tmp_path).stdout.strip() == "2"
    script.write_text(_helper_script(3))
    assert run_python("script.py", cwd=tmp_path).stdout.strip() == "3"
