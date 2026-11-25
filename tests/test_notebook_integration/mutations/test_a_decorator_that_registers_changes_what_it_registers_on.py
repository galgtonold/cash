"""A decorator written as a method call changes the object it is called on.

``@reg.add(PREFIX + '1')``, ``@show.register(Kind)``, ``@app.route('/x')``
above a ``def`` is the call ``reg.add(...)(h)``: it registers the function
on ``reg``. Read as binding the function's name only, the ``def`` was no
producer of the registry, so a cell below reading the registry did not bring
it up to date: after an edit to something the decorator reads, the registry
kept the entry made with the old value, and ``functools.singledispatch``
picked the base implementation for an object of the edited class.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

REGISTRY = (
    "class Reg:\n"
    "    def __init__(self): self.items = []\n"
    "    def add(self, name):\n"
    "        def d(f):\n"
    "            self.items.append(name)\n"
    "            return f\n"
    "        return d\n"
    "    def plain(self, f):\n"
    "        self.items.append(f.__name__ + SUFFIX)\n"
    "        return f\n"
    "reg = Reg()"
)

# name: (cells, the cell edited (1-based) and its new source, printed before, printed after)
CASES = {
    "a_registry_called_with_arguments": (
        ["PREFIX = 'a'\nSUFFIX = ''", REGISTRY, "@reg.add(PREFIX + '1')\ndef h(): pass", "print('R', reg.items)"],
        (1, "PREFIX = 'b'\nSUFFIX = ''"),
        "R ['a1']",
        "R ['b1']",
    ),
    "a_registry_method_used_bare": (
        ["SUFFIX = '-a'", REGISTRY, "@reg.plain\ndef h(): pass", "print('R', reg.items)"],
        (3, "@reg.plain\ndef renamed(): pass"),
        "R ['h-a']",
        "R ['renamed-a']",
    ),
    "a_class_registered": (
        ["PREFIX = 'a'\nSUFFIX = ''", REGISTRY, "@reg.add(PREFIX + 'K')\nclass K: pass", "print('R', reg.items)"],
        (1, "PREFIX = 'b'\nSUFFIX = ''"),
        "R ['aK']",
        "R ['bK']",
    ),
    "singledispatch_register_with_a_class": (
        [
            "class Kind:\n    tag = 1",
            "import functools\n@functools.singledispatch\ndef show(x): return 'base'",
            "@show.register(Kind)\ndef _(x): return 'kind'",
            "print('R', show(Kind()))",
        ],
        (1, "class Kind:\n    tag = 2   # edited"),
        "R kind",
        "R kind",
    ),
    "singledispatch_register_by_type_hint": (
        [
            "class Kind:\n    tag = 1",
            "import functools\n@functools.singledispatch\ndef show(x): return 'base'",
            "@show.register\ndef _(x: Kind): return 'kind'",
            "print('R', show(Kind()))",
        ],
        (1, "class Kind:\n    tag = 2   # edited"),
        "R kind",
        "R kind",
    ),
    "control_the_same_registration_as_a_call": (
        ["PREFIX = 'a'\nSUFFIX = ''", REGISTRY, "reg.add(PREFIX + '1')(len)", "print('R', reg.items)"],
        (1, "PREFIX = 'b'\nSUFFIX = ''"),
        "R ['a1']",
        "R ['b1']",
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_reader_of_the_registry_follows_an_edit_above(nb_runner, case):
    cells, (number, edited), before, after = CASES[case]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert before in nb_runner.get_output(len(cells)), "control: the top-to-bottom run"

    nb_runner.set_cell_source(number, edited)
    nb_runner.run_cell(len(cells))

    assert after in nb_runner.get_output(len(cells)), nb_runner.get_raw_output(len(cells))


@pytest.mark.fresh_kernel
@pytest.mark.parametrize("case", ["a_registry_called_with_arguments", "singledispatch_register_with_a_class"])
def test_the_reader_run_alone_after_a_restart_finds_what_was_registered(nb_runner, case):
    cells, _edit, before, _after = CASES[case]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert before in nb_runner.get_output(len(cells)), "control: the top-to-bottom run"

    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_cell(len(cells))

    assert before in nb_runner.get_output(len(cells)), nb_runner.get_raw_output(len(cells))
