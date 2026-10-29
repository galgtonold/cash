"""A loop's hit restores what it changed into the object it changed.

A hit that bound ``p`` to the restored copy put a new object where the old
one was: ``weakref.ref(p)()`` gave None, a ``WeakValueDictionary`` lost its
entry and ``{id(p): ...}`` no longer found ``p``, where plain Python keeps
them all. The copy's state now goes into the live object (`in_place`), whether the
loop calls its methods (``p.update()``) or assigns into it (``p.n += 1``,
``d[k] = v``).

What such a restore cannot give as plain Python does runs the loop again,
with the reason: a weak reference to, or an ``id()`` key of, an object it
holds (that comes back as a copy), an object pickled its own way, a container
holding the object, a helper writing a module it imports, a ``while`` loop, and a loop that
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
def test_an_object_the_loop_assigns_into_is_restored_into(nb_runner):
    """``p.n += 1`` changes ``p`` as ``p.update()`` does: a weak reference
    to ``p`` still finds it after a hit."""
    make = "import time, weakref\nclass C:\n    n = 0\np = C()\nref = weakref.ref(p)"
    loop = "for i in range(3000):\n    time.sleep(0.0003)\n    p.n += 1"
    printed, seen = _run(nb_runner, make, loop, "print(ref() is p, p.n)")
    assert printed == "True 3000"
    assert "RESTORED" in str(seen)


@pytest.mark.fresh_kernel
@pytest.mark.parametrize(
    ("make", "body", "read", "expected"),
    [
        ("out = []", "out.append(i)", "len(out), out[-1]", "3000 2999"),
        ("out = set()", "out.add(i)", "len(out), max(out)", "3000 2999"),
        ("out = {}", "out[i * 2] = i * 3", "len(out), out[5998]", "3000 8997"),
    ],
    ids=["a list", "a set", "a dict assigned into"],
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
        (
            P + "import weakref\nclass Q:\n    pass\np = P()\np.child = Q()\nchild = weakref.ref(p.child)",
            LOOP,
            "print(child() is p.child, p.n)",
            "weak reference",
        ),
        (P + "p = P()\nreg = {id(p.seen): 'seen'}", LOOP, "print(reg.get(id(p.seen)), p.n)", "keeps the id()"),
        (
            P + "p = P()\nreg = {'m': {id(p.seen): 'seen'}}",
            LOOP,
            "print(reg['m'].get(id(p.seen)), p.n)",
            "keeps the id()",
        ),
        (P + "p = P()\nids = [id(p.seen)]", LOOP, "print(ids[0] == id(p.seen), p.n)", "keeps the id()"),
        (
            P + "class R:\n    pass\np = P()\nreg = R()\nreg.m = {id(p.seen): 'seen'}",
            LOOP,
            "print(reg.m.get(id(p.seen)), p.n)",
            "keeps the id()",
        ),
        (P + "p = P()\nreg = {'seen': id(p.seen)}", LOOP, "print(reg['seen'] == id(p.seen), p.n)", "keeps the id()"),
        (
            P + "import weakref\nclass Q:\n    pass\np = P()\np.child = Q()\nwkd = weakref.WeakKeyDictionary()\n"
            "wkd[p.child] = 'c'",
            LOOP,
            "print(wkd.get(p.child), p.n)",
            "weak reference",
        ),
        (
            P + "p = P()\nbig = [(j, j) for j in range(20000)] + [('seen', id(p.seen))]",
            LOOP,
            "print(big[-1][1] == id(p.seen), p.n)",
            "too big to look through",
        ),
        (
            P + "from collections import deque\np = P()\nids = deque([id(p.seen)])",
            LOOP,
            "print(ids[0] == id(p.seen), p.n)",
            "keeps the id()",
        ),
        (
            P + "import numpy as np\np = P()\narr = np.array([id(p.seen)], dtype=np.int64)",
            LOOP,
            "print(int(arr[0]) == id(p.seen), p.n)",
            "keeps the id()",
        ),
        (
            P + "from types import SimpleNamespace\np = P()\nns = SimpleNamespace(k=id(p.seen))",
            LOOP,
            "print(ns.k == id(p.seen), p.n)",
            "keeps the id()",
        ),
        (
            P + "class H:\n    __slots__ = ('k',)\n    def __init__(self, k):\n        self.k = k\np = P()\n"
            "h = H(id(p.seen))",
            LOOP,
            "print(h.k == id(p.seen), p.n)",
            "keeps the id()",
        ),
        (
            P + "import pandas as pd\np = P()\ndf = pd.DataFrame({'r': [id(p.seen)]}, dtype=object)",
            LOOP,
            "print(df['r'][0] == id(p.seen), p.n)",
            "keeps the id()",
        ),
        (
            P + "import functools\np = P()\nf = functools.partial(max, id(p.seen))",
            LOOP,
            "print(f(0) == id(p.seen), p.n)",
            "keeps the id()",
        ),
        (
            P + "import numpy as np\np = P()\narr = np.array([(1, id(p.seen))], dtype=[('a', 'i8'), ('b', 'i8')])",
            LOOP,
            "print(int(arr['b'][0]) == id(p.seen), p.n)",
            "keeps the id()",
        ),
        (
            P + "p = P()\nreg = {hex(id(p.seen)): 's'}",
            LOOP,
            "print(reg.get(hex(id(p.seen))), p.n)",
            "keeps the id()",
        ),
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
    ids=[
        "a weak reference to a part",
        "an id key of a part",
        "an id key in a nested dict",
        "an id in a list",
        "an id key inside an object",
        "an id as a dict value",
        "a weak reference as a dict key",
        "an id inside a big container",
        "an id in a deque",
        "an id in an integer array",
        "an id in a SimpleNamespace",
        "an id in a slots class",
        "an id in a pandas column",
        "an id in a partial",
        "an id in a structured array",
        "an id as hex text",
        "own pickling",
        "a container holding it",
        "a helper writing a module",
        "a while loop",
        "a cheap loop",
    ],
)
def test_what_a_restore_into_cannot_give_runs_the_loop(nb_runner, make, loop, read, why):
    _, statuses = _run(nb_runner, make, loop, read)
    assert "RESTORED" not in str(statuses)
    if why is not None:
        assert why in _shown(nb_runner, 2)
    if why in ("weak reference", "keeps the id()", "too big to look through"):
        assert "NOTEBOOK-IDENTITY-HELD" in _shown(nb_runner, 2)


@pytest.mark.fresh_kernel
@pytest.mark.parametrize(
    ("made_later", "read", "expected"),
    [
        ("held = weakref.ref(p.child)", "print(held is not None and held() is p.child, p.n)", "True 3000"),
        ("held = {id(p.child): 'child'}", "print(held and held.get(id(p.child)), p.n)", "child 3000"),
    ],
    ids=["a weak reference", "an id key"],
)
def test_what_points_into_it_is_asked_again_at_a_restore(nb_runner, made_later, read, expected):
    """Made after the loop was stored, in a cell whose edit does not change
    the loop's key: a restore into ``p`` would hand it a copy of the part."""
    nb_runner.create_notebook(
        [P + "import weakref\nclass Q:\n    pass\np = P()\np.child = Q()", "held = None", LOOP, read]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.set_cell_source(2, made_later)
    nb_runner.run_all()
    assert nb_runner.get_output(4).strip() == expected


@pytest.mark.fresh_kernel
@pytest.mark.parametrize(
    ("make", "body", "read", "expected"),
    [
        ("import numpy as np\nout = np.zeros(3000)", "out[i] = i * 2", "out[-1], out.sum()", "5998.0 8997000.0"),
        (
            "import pandas as pd\nout = {}",
            "out[i] = pd.DataFrame({'a': [i]})",
            "len(out), int(out[2999]['a'][0])",
            "3000 2999",
        ),
        (
            "import numpy as np, pandas as pd\nout = pd.DataFrame(np.zeros((3000, 2)))",
            "out.iloc[i, 0] = i",
            "float(out[0].sum())",
            "4498500.0",
        ),
    ],
    ids=["a numpy array", "a dict of frames", "a frame"],
)
def test_what_the_loop_assigns_into_is_restored_after_a_restart(nb_runner, make, body, read, expected):
    """A NumPy array, or a frame, is bound to the restored copy, as before
    (the weak reference pandas keeps to the frame in itself comes back with
    the copy); a dict of frames is restored into, the weak references pandas keeps between its
    own blocks coming back with it."""
    loop = f"for i in range(3000):\n    time.sleep(0.0003)\n    {body}"
    printed, seen = _run(nb_runner, f"import time\n{make}", loop, f"print({read})")
    assert printed == expected
    assert "RESTORED" in str(seen)


@pytest.mark.fresh_kernel
@pytest.mark.parametrize(
    "held",
    ["import weakref\nheld = weakref.ref(out)", "held = {id(out): 'out'}"],
    ids=["a weak reference", "an id key"],
)
def test_an_array_the_notebook_holds_by_identity_runs_the_loop_with_the_warning(nb_runner, held):
    """Bound to the stored copy, an array the notebook finds by a weak
    reference or its ``id()`` would not be the object it finds: the loop
    runs every time, and says so."""
    make = f"import time\nimport numpy as np\nout = np.zeros(3000)\n{held}"
    loop = "for i in range(3000):\n    time.sleep(0.0003)\n    out[i] = i * 2"
    printed, seen = _run(nb_runner, make, loop, "print(out[-1], out.sum())")
    assert printed == "5998.0 8997000.0"
    assert "RESTORED" not in str(seen)
    assert "NOTEBOOK-IDENTITY-HELD" in _shown(nb_runner, 2)
