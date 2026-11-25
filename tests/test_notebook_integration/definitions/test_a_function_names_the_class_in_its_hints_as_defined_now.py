"""A function's type hints name the class as the notebook defines it now.

``def f(x: Kind) -> Kind`` keeps the class on the function: in
``f.__annotations__``, in the bound of a type parameter, and in whatever a
decorator worked out from the hints. A class named only there was no input of
the ``def``, so after the class was edited above the function was not defined
again and went on naming the old class -- which no object made below is an
instance of. The reference is a run from the top, where the ``def`` runs after
the class; that holds for hints Python evaluates when the ``def`` runs, for
quoted ones, and for the ones Python 3.14 evaluates on first use.
"""

import sys

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

KIND, EDITED = "class Kind:\n    tag = 1", "class Kind:\n    tag = 2   # edited"

# name: (the def cell, the reader, printed before the edit, printed after)
CASES = {
    "a_parameter_and_the_return_hint": (
        "def f(x: Kind) -> Kind:\n    return x",
        "print('R', f.__annotations__['x'] is Kind, f.__annotations__['return'].tag)",
        "R True 1",
        "R True 2",
    ),
    "a_quoted_hint_a_decorator_resolves": (
        "import typing\ndef typed(f):\n    f.hints = typing.get_type_hints(f)\n    return f\n"
        "@typed\ndef f(x: 'Kind'):\n    return x",
        "print('R', f.hints['x'] is Kind, f.hints['x'].tag)",
        "R True 1",
        "R True 2",
    ),
    "a_hint_inside_a_generic": (
        "def f(xs: list[Kind]):\n    return xs",
        "print('R', f.__annotations__['xs'].__args__[0] is Kind)",
        "R True",
        "R True",
    ),
    "control_a_default_value": (
        "def f(x=Kind()):\n    return x.tag",
        "print('R', f())",
        "R 1",
        "R 2",
    ),
}
if sys.version_info >= (3, 12):
    CASES["the_bound_of_a_type_parameter"] = (
        "def ident[T: Kind](x: T) -> T:\n    return x",
        "print('R', ident.__type_params__[0].__bound__ is Kind)",
        "R True",
        "R True",
    )


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_function_follows_an_edit_to_a_class_its_hints_name(nb_runner, case):
    define, reader, before, after = CASES[case]
    nb_runner.create_notebook([KIND, define, reader])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert before in nb_runner.get_output(3), "control: the top-to-bottom run"

    nb_runner.set_cell_source(1, EDITED)
    nb_runner.run_cell(3)

    assert after in nb_runner.get_output(3), nb_runner.get_raw_output(3)
