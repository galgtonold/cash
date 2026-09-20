"""A notebook that only works because you ran the cells out of order is broken.

Round 26, r26s5: a cell read a variable that only a cell BELOW it binds. Under
cash the notebook worked, because the later cell had been run at some point and
the name was still in the namespace. A clean in-order run died with
``NameError``, and only the uncached oracle caught it -- cash reported success
on a notebook that could not reproduce itself.

cash already knows which cell binds each name: ``_evict_orphaned_definitions``
walks every cell for exactly that, to catch the sibling case where NO cell
produces a name any more. This is the other half -- a name only a LATER cell
produces -- and it fails rather than warns, because a warning leaves cash
building on state its own in-order run would not have.

The control arms matter as much as the failing one: a name bound above, and a
name bound both above and below, must keep working. The second is the common
shape (a variable set early and reassigned later in the notebook) and failing
it would make cash unusable.
"""
import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]


def test_reading_a_name_only_a_later_cell_binds_fails(nb_runner):
    """r26s5's shape: run the later binding first, then the earlier reader."""
    nb_runner.create_notebook([
        "import cash\n%cash_on",
        "y = x + 1\nprint('Y', y)",        # cell 2 reads x
        "x = 10\nprint('X', x)",           # cell 3 binds it
    ])
    nb_runner.start_kernel()
    nb_runner.run_cell(3)                   # bind x, so the name exists
    with pytest.raises(CellExecutionError) as exc:
        nb_runner.run_cell(2)               # now read it from above

    text = str(exc.value)
    assert "ForwardReferenceError" in text, text
    assert "Y 11" not in text, (
        "the cell produced a value from a binding below it:\n" + text
    )
    # The message has to be actionable on its own: which name, which cell.
    assert "`x`" in text and "cell 3" in text, text


def test_a_name_bound_above_still_works(nb_runner):
    """The control. Ordinary downstream reads must be untouched."""
    nb_runner.create_notebook([
        "import cash\n%cash_on",
        "x = 10\nprint('X', x)",
        "y = x + 1\nprint('Y', y)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "Y 11" in nb_runner.get_output(3), nb_runner.get_raw_output(3)


def test_a_name_bound_above_and_again_below_still_works(nb_runner):
    """The shape that would break everything if this were done carelessly.

    `x` is bound in cell 2 and rebound in cell 4. Cell 3 reads it. That is an
    ordinary notebook, not a forward reference: the read resolves to the
    binding above it.
    """
    nb_runner.create_notebook([
        "import cash\n%cash_on",
        "x = 10",
        "y = x + 1\nprint('Y', y)",
        "x = 99\nprint('X', x)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "Y 11" in nb_runner.get_output(3), nb_runner.get_raw_output(3)


def test_a_function_defined_below_and_called_above_fails(nb_runner):
    """Same rule for a def, which is how it usually happens in practice."""
    nb_runner.create_notebook([
        "import cash\n%cash_on",
        "v = helper(3)\nprint('V', v)",
        "def helper(n):\n    return n * 2",
    ])
    nb_runner.start_kernel()
    nb_runner.run_cell(3)
    with pytest.raises(CellExecutionError) as exc:
        nb_runner.run_cell(2)

    text = str(exc.value)
    assert "ForwardReferenceError" in text, text
    assert "V 6" not in text, text
    assert "`helper`" in text and "cell 3" in text, text


def test_a_function_body_may_name_something_bound_below(nb_runner):
    """The distinction that makes this rule safe, and the one it got wrong.

    `def a(n): return b(n) * 2` above `def b` is ordinary Python: a name in a
    function body is resolved when the function is CALLED, and by then the
    cell below has run. Only a MODULE-LEVEL read can break an in-order run.

    Refusing this shape broke `test_downward_function_dependency`, whose
    notebook runs perfectly well from the top.
    """
    nb_runner.create_notebook([
        "import cash\n%cash_on",
        "def a(n):\n    return b(n) * 2",     # names b, defined below
        "def b(n):\n    return n + 1",
        "r = a(3)\nprint(f'r={r}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "r=8" in nb_runner.get_output(4), nb_runner.get_raw_output(4)


def test_a_decorator_bound_below_still_fails(nb_runner):
    """...but a decorator IS evaluated at definition time, so it is a read."""
    nb_runner.create_notebook([
        "import cash\n%cash_on",
        "@deco\ndef f():\n    return 1",
        "def deco(fn):\n    return fn",
    ])
    nb_runner.start_kernel()
    nb_runner.run_cell(3)
    with pytest.raises(CellExecutionError) as exc:
        nb_runner.run_cell(2)
    assert "ForwardReferenceError" in str(exc.value), str(exc.value)
    assert "`deco`" in str(exc.value), str(exc.value)
