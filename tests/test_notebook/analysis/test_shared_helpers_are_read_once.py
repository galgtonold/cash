"""What a call does to its arguments, through a library's many shared helpers.

``from sympy import *`` binds hundreds of functions whose helpers call the same
helpers again. Following each call path on its own read a helper once per path
to it, so ``factor(e)`` was analysed for minutes. Each helper is now read once,
and what it does still reaches every caller.
"""

import ast
import time

from cash.analysis.callee_effects import function_arg_mutations, params_mutated_in_function


def _lattice(levels: int) -> dict[str, str]:
    """``f0`` calls ``f1`` twice, ``f1`` calls ``f2`` twice, ...: 2**levels paths
    to the last one, which changes its argument."""
    sources = {f"f{i}": f"def f{i}(x):\n    f{i + 1}(x)\n    return f{i + 1}(x)\n" for i in range(levels)}
    sources[f"f{levels}"] = f"def f{levels}(x):\n    x.append(1)\n"
    return sources


def test_a_helper_reached_by_many_paths_is_read_once():
    sources = _lattice(40)
    reads: list[str] = []

    def resolve(name):
        reads.append(name)
        return sources.get(name)

    started = time.perf_counter()
    assert function_arg_mutations(ast.parse("f0(rows)"), resolve) == {"rows"}
    assert time.perf_counter() - started < 5
    assert len(reads) <= 2 * len(sources)


def test_a_mutation_through_mutual_recursion_reaches_the_caller():
    sources = {
        "even": "def even(x, n):\n    if n:\n        odd(x, n - 1)\n",
        "odd": "def odd(x, n):\n    if n:\n        even(x, n - 1)\n    else:\n        x.append(n)\n",
    }
    assert function_arg_mutations(ast.parse("even(rows, 4)"), sources.get) == {"rows"}
    # Control: nothing on the cycle changes its argument.
    quiet = {k: v.replace("x.append(n)", "pass") for k, v in sources.items()}
    assert function_arg_mutations(ast.parse("even(rows, 4)"), quiet.get) == frozenset()


def test_the_matching_slot_still_decides_which_argument_changes():
    sources = {
        "outer": "def outer(a, b):\n    inner(b, a)\n",
        "inner": "def inner(p, q):\n    p.append(q)\n",
    }
    fdef = ast.parse(sources["outer"]).body[0]
    assert params_mutated_in_function(fdef, sources.get, frozenset({"outer"})) == {"b"}
    assert function_arg_mutations(ast.parse("outer(left, right)"), sources.get) == {"right"}
