"""A loop's hit restores what it changed into the object it changed.

A hit that bound ``p`` to the restored copy put a new object where the old
one was: ``weakref.ref(p)()`` gave None, a ``WeakValueDictionary`` lost its
entry and ``{id(p): ...}`` no longer found ``p``, where plain Python keeps
them all. The copy's state now goes into the live object (`in_place`).

What such a restore cannot give as plain Python does runs the loop again,
with the reason: an object pickled its own way, a container holding the
object, a helper writing a module it imports, a ``while`` loop, and a loop that
last ran faster than storing it would save.
"""

import ast
import json

import pytest

pytestmark = [pytest.mark.timeout(300)]

P = (
    "import time\nclass P:\n    def __init__(self):\n        self.n = 0\n        self.seen = []\n"
    "    def update(self, i=0):\n        self.n += 1\n        self.seen.append(i)\n"
)
LOOP = "for i in range(3000):\n    time.sleep(0.0003)\n    p.update(i)"
STATS = "get_ipython().magics_manager.registry['CashMagics']._last_cell_metrics['statements']"
PEEK = f"[str(s['status']) for s in {STATS}]"


def _restart_and_run(nb_runner):
    """After a restart, run the cells one by one: what the loop cell did is
    read before the cell after it runs."""
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_cell(1)
    nb_runner.run_cell(2)
    seen = ast.literal_eval(nb_runner.peek(PEEK))
    nb_runner.run_cell(3)
    return seen


def _shown(nb_runner, cell):
    """All the cell shows, its badge with the reasons in it."""
    return json.dumps(nb_runner.get_cell(cell).get("outputs", []))


def _run(nb_runner, make, loop, read):
    nb_runner.create_notebook([make, loop, read])
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = nb_runner.get_output(3).strip()
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == first
    seen = _restart_and_run(nb_runner)
    assert nb_runner.get_output(3).strip() == first
    return first, seen


@pytest.mark.fresh_kernel
def test_weak_references_and_id_keys_still_find_the_object(nb_runner):
    make = (
        P + "import weakref\np = P()\nref = weakref.ref(p)\nby_value = weakref.WeakValueDictionary()\n"
        "by_value['p'] = p\nby_id = {id(p): 'p'}"
    )
    read = "print(ref() is p, by_value['p'] is p, by_id.get(id(p)), p.n)"
    printed, seen = _run(nb_runner, make, LOOP, read)
    assert printed == "True True p 3000"
    assert "RESTORED" in str(seen)


@pytest.mark.fresh_kernel
@pytest.mark.parametrize(
    ("make", "body", "read", "expected"),
    [
        ("out = []", "out.append(i)", "len(out), out[-1]", "3000 2999"),
        ("out = set()", "out.add(i)", "len(out), max(out)", "3000 2999"),
    ],
    ids=["a list", "a set"],
)
def test_a_container_is_restored_into(nb_runner, make, body, read, expected):
    loop = f"for i in range(3000):\n    time.sleep(0.0003)\n    {body}"
    printed, seen = _run(nb_runner, f"import time\n{make}\nkeep = id(out)", loop, f"print(keep == id(out), {read})")
    assert printed == "True " + expected
    assert "RESTORED" in str(seen)


GETSTATE = "    def __getstate__(self):\n        return {'n': self.n, 'seen': []}\n"
MODULE = "open('counter_mod.py', 'w').write('k = 0\\n')\nimport counter_mod\n"
BUMP = "def bump():\n    import counter_mod\n    counter_mod.k += 1\n"


@pytest.mark.fresh_kernel
@pytest.mark.parametrize(
    ("make", "loop", "read", "why"),
    [
        (P + GETSTATE + "p = P()", LOOP, "print(p.n, len(p.seen))", "is not restored into"),
        (P + "p = P()\nholder = {'a': [p]}", LOOP, "print(holder['a'][0].n, holder['a'][0] is p)", "container holds"),
        (
            P + MODULE + BUMP + "p = P()",
            "for i in range(3000):\n    time.sleep(0.0003)\n    p.update(i)\n    bump()",
            "print(p.n, counter_mod.k)",
            None,
        ),
        (
            P + "p = P()",
            "i = 0\nwhile i < 3000:\n    time.sleep(0.0003)\n    p.update(i)\n    i += 1",
            "print(p.n)",
            None,
        ),
        (P + "p = P()", "for i in range(30000):\n    p.update(i)", "print(p.n)", None),
    ],
    ids=["own pickling", "a container holding it", "a helper writing a module", "a while loop", "a cheap loop"],
)
def test_what_a_restore_into_cannot_give_runs_the_loop(nb_runner, make, loop, read, why):
    _, statuses = _run(nb_runner, make, loop, read)
    assert "RESTORED" not in str(statuses)
    if why is not None:
        assert why in _shown(nb_runner, 2)
