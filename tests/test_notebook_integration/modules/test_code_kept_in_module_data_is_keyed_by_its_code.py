"""A function or object kept in a module's data is keyed by its code.

``@mylib.register def alpha(): return 1`` in one cell, ``x = mylib.call_all()``
below: editing ``alpha`` to return 2 and running the notebook again served
``[1]``. The reader's key took the registry's value hash, which names a
function only by its name. The same for an object of a notebook class set
as the module's handler, after its method was edited.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

LIB = (
    "import os, time\n"
    "REG = {}\n"
    "HANDLER = None\n"
    "def _count():\n"
    "    fd = os.open('calls.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND)\n"
    "    os.write(fd, b'x')\n"
    "    os.close(fd)\n"
    "def register(f):\n    REG[f.__name__] = f\n    return f\n"
    "def set_handler(h):\n    global HANDLER\n    HANDLER = h\n"
    "def call_all():\n    _count()\n    time.sleep(0.3)\n    return [REG[k]() for k in sorted(REG)]\n"
    "def process(x):\n    time.sleep(0.3)\n    return HANDLER.apply(x)\n"
)


@pytest.mark.parametrize(
    ("setter", "reader", "edit", "before", "after"),
    [
        ("@mylib.register\ndef alpha():\n    return 1", "x = mylib.call_all()", ("return 1", "return 2"), "[1]", "[2]"),
        (
            "def alpha():\n    return 1\nmylib.REG['alpha'] = alpha",
            "x = mylib.call_all()",
            ("return 1", "return 2"),
            "[1]",
            "[2]",
        ),
        (
            "class H:\n    def apply(self, x):\n        return x + 1\nmylib.set_handler(H())",
            "x = mylib.process(1)",
            ("x + 1", "x + 100"),
            "2",
            "101",
        ),
    ],
    ids=["registered_by_a_decorator", "assigned_into_the_registry", "handler_object"],
)
def test_editing_the_kept_code_recomputes_the_reader(nb_runner, setter, reader, edit, before, after):
    (nb_runner.work_dir / "mylib.py").write_text(LIB, encoding="utf-8")
    nb_runner.create_notebook(["import mylib", setter, reader])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("x") == before

    nb_runner.set_cell_source(2, setter.replace(*edit))
    nb_runner.run_all()

    assert nb_runner.peek("x") == after, nb_runner.get_raw_output(3)


def test_an_unedited_registry_is_served_after_a_restart(nb_runner):
    """Positive control: the code part of the key is the same in a new
    kernel, so Restart & Run All serves the reader."""
    work = nb_runner.work_dir
    (work / "mylib.py").write_text(LIB, encoding="utf-8")
    nb_runner.create_notebook(
        ["import cash\n%cash_on", "import mylib", "@mylib.register\ndef alpha():\n    return 1", "x = mylib.call_all()"]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    assert len((work / "calls.log").read_bytes()) == 1

    nb_runner.restart()
    nb_runner.run_all()

    assert nb_runner.peek("x") == "[1]"
    assert len((work / "calls.log").read_bytes()) == 1, nb_runner.get_raw_output(4)
