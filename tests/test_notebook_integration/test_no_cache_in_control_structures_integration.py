"""Real-kernel coverage for ``@cash:`` annotations inside control structures.

CAS-135 hole 3. ``# @cash:no-cache`` is the documented escape hatch for the
deliberate "unseeded randomness is cached" policy (badges.md: *"If you want it to
re-run every time, say so explicitly"*). It worked on a top-level statement and
was **silently ignored** inside a loop body — no error, no warning, no badge
note. A documented opt-out that quietly does nothing is worse than no opt-out,
and "unseeded + expensive + in a loop" is the definition of Monte Carlo, so this
compounded the frozen-value bug into one with no working workaround.

The annotation was never unparseable — ``get_statement_annotations`` binds it to
the body statement correctly. It was computed in ``cell_executor`` and then
dropped: ``ControlStructureProcessor.process()`` was never handed it.

The controls are what make these tests worth having. An over-broad fix that
disabled caching for the whole loop would satisfy "the annotated statement
re-runs" while destroying the caching of every sibling around it — so each test
pairs the annotated statement with an unannotated one in the SAME loop body.
"""
import ast

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]

# Comfortably above min_execution_time_to_cache_seconds (0.01s) so the control
# statement is genuinely cacheable and the test measures the annotation rather
# than the cost floor.
DRAW = "float(np.random.default_rng().standard_normal(4000000).mean())"


def _vals(output: str, prefix: str) -> list:
    """Parse a ``PREFIX= [...]`` line out of cell output."""
    lines = [ln for ln in output.splitlines() if ln.strip().startswith(prefix)]
    assert lines, f"no {prefix!r} line in output: {output!r}"
    return ast.literal_eval(lines[-1].split("=", 1)[1].strip())


def test_no_cache_inside_for_body_reexecutes(nb_runner):
    """The headline: the annotated body statement must re-run; its sibling must not.

    ``a`` is annotated and unseeded, so honouring the directive means fresh draws
    every run. ``b`` is identical work WITHOUT the directive, in the same body —
    it must stay cached. Together they prove the annotation is both *honoured*
    and *scoped*.
    """
    nb_runner.create_notebook([
        "import numpy as np",
        "acc = []\n"
        "for t in range(2):\n"
        "    # @cash:no-cache\n"
        f"    a = {DRAW}\n"
        f"    b = {DRAW}\n"
        "    acc.append((a, b))\n"
        "print('A=', [round(x[0], 12) for x in acc])\n"
        "print('B=', [round(x[1], 12) for x in acc])",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    a1, b1 = _vals(nb_runner.get_output(2), "A="), _vals(nb_runner.get_output(2), "B=")

    nb_runner.run_cells([2])
    a2, b2 = _vals(nb_runner.get_output(2), "A="), _vals(nb_runner.get_output(2), "B=")

    # The directive is honoured: fresh draws.
    assert a1 != a2, (
        f"# @cash:no-cache inside a for body was ignored: annotated statement "
        f"replayed frozen values across runs ({a1} vs {a2})"
    )
    # Control: the unannotated sibling still caches. Without this the test would
    # pass for a fix that simply stopped caching the whole loop.
    assert b1 == b2, (
        f"the unannotated sibling stopped caching: the no-cache directive leaked "
        f"across the loop body ({b1} vs {b2})"
    )


def test_no_cache_above_the_for_header_disables_the_whole_loop(nb_runner):
    """The other placement the report tried: the directive above ``for``.

    Scoped to the loop, so it must reach every body statement — including the
    one that carries no annotation of its own.
    """
    nb_runner.create_notebook([
        "import numpy as np",
        "acc = []\n"
        "# @cash:no-cache\n"
        "for t in range(2):\n"
        f"    a = {DRAW}\n"
        "    acc.append(a)\n"
        "print('A=', [round(x, 12) for x in acc])",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    a1 = _vals(nb_runner.get_output(2), "A=")

    nb_runner.run_cells([2])
    assert a1 != _vals(nb_runner.get_output(2), "A=")


def test_unannotated_loop_body_still_caches(nb_runner):
    """Control: the baseline behaviour this must not disturb.

    Per-iteration loop caching is a headline feature (docs/index.md). If the fix
    made loop bodies uncacheable, this is what would catch it.
    """
    nb_runner.create_notebook([
        "import numpy as np",
        "acc = []\n"
        "for t in range(2):\n"
        f"    a = {DRAW}\n"
        "    acc.append(a)\n"
        "print('A=', [round(x, 12) for x in acc])",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    a1 = _vals(nb_runner.get_output(2), "A=")

    nb_runner.run_cells([2])
    assert a1 == _vals(nb_runner.get_output(2), "A=")


def test_no_cache_at_top_level_still_works(nb_runner):
    """Control: do not regress the placement that already worked."""
    nb_runner.create_notebook([
        "import numpy as np",
        "# @cash:no-cache\n"
        f"c = {DRAW}\n"
        "print('C=', [round(c, 12)])",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    c1 = _vals(nb_runner.get_output(2), "C=")

    nb_runner.run_cells([2])
    assert c1 != _vals(nb_runner.get_output(2), "C=")


def test_no_cache_inside_if_branch_reexecutes(nb_runner):
    """``if`` bodies are decomposed per-statement by the same dispatcher."""
    nb_runner.create_notebook([
        "import numpy as np",
        "flag = True",
        "if flag:\n"
        "    # @cash:no-cache\n"
        f"    a = {DRAW}\n"
        f"    b = {DRAW}\n"
        "print('A=', [round(a, 12)])\n"
        "print('B=', [round(b, 12)])",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    a1, b1 = _vals(nb_runner.get_output(3), "A="), _vals(nb_runner.get_output(3), "B=")

    nb_runner.run_cells([3])
    a2, b2 = _vals(nb_runner.get_output(3), "A="), _vals(nb_runner.get_output(3), "B=")

    assert a1 != a2
    assert b1 == b2, "no-cache leaked to the unannotated sibling in the if body"


def test_no_cache_inside_while_single_unit_reexecutes(nb_runner):
    """A ``while`` runs as ONE cache unit, so an annotation anywhere in it scopes
    to the whole unit — the annotation granularity follows the cache granularity.

    Deliberately avoids ``acc.append(...)`` in the body: a method mutation gets
    ``skip_cache=True`` from the mutation detector, which would make this pass
    without the annotation doing anything.
    ``test_unannotated_while_single_unit_still_caches`` pins that the unit really
    does cache when unannotated, so the re-execution here is attributable to the
    directive and nothing else.
    """
    nb_runner.create_notebook([
        "import numpy as np",
        "i = 0\n"
        "w = 0.0\n"
        "while i < 2:\n"
        "    # @cash:no-cache\n"
        f"    w = {DRAW}\n"
        "    i += 1\n"
        "print('A=', [round(w, 12)])",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    a1 = _vals(nb_runner.get_output(2), "A=")

    nb_runner.run_cells([2])
    assert a1 != _vals(nb_runner.get_output(2), "A=")


def test_unannotated_while_single_unit_still_caches(nb_runner):
    """Control for the test above: the while unit caches when left alone."""
    nb_runner.create_notebook([
        "import numpy as np",
        "i = 0\n"
        "w = 0.0\n"
        "while i < 2:\n"
        f"    w = {DRAW}\n"
        "    i += 1\n"
        "print('A=', [round(w, 12)])",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    a1 = _vals(nb_runner.get_output(2), "A=")

    nb_runner.run_cells([2])
    assert a1 == _vals(nb_runner.get_output(2), "A=")


def test_ttl_annotation_reaches_a_loop_body(nb_runner):
    """The hole was in the threading, not in ``no-cache`` specifically — every
    directive was being dropped. A ttl that has not expired must still cache.
    """
    nb_runner.create_notebook([
        "import numpy as np",
        "acc = []\n"
        "for t in range(2):\n"
        "    # @cash:ttl=600\n"
        f"    a = {DRAW}\n"
        "    acc.append(a)\n"
        "print('A=', [round(x, 12) for x in acc])",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    a1 = _vals(nb_runner.get_output(2), "A=")

    nb_runner.run_cells([2])
    assert a1 == _vals(nb_runner.get_output(2), "A=")
