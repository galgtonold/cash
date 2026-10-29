"""A class another cell changed through its object keeps that change when an
edited cell that changes the class runs again.

``s.hold(data['meta'])`` leaves ``data['meta']`` on the class through
``type(self)``, ``super()`` or a helper; ``SBox.bump(7)``, edited from 5,
changes it. Before the edited cell ran again, the class's own cell was re-run
(the edited cell was taken for the class's only changer, from the cells'
text), which dropped what ``hold`` had left on it: the bump failed on an
empty list, and the reading cell was served the value from before the edit.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

DATA = "import time\ndata = {'rows': [{'v': i} for i in range(3000)], 'meta': {'n': 1}}\ntime.sleep(0.2)"
READ = "time.sleep(0.2)\nresult = sum(r['v'] for r in data['rows']) + data['meta']['n']\nprint('R', result)"
BUMP = "@classmethod\n    def bump(cls, n):\n        cls.held[-1]['n'] += n"
SHAPES = {
    "through-super": (
        "class Base:\n    held = []\n    " + BUMP + "\nclass Sub(Base):\n    def hold(self, d):\n"
        "        super().held.append(d)",
        "s = Sub()",
        "s.hold(data['meta'])",
        "Base.bump({n})",
    ),
    "through-a-helper": (
        "class SBox:\n    held = []\n    " + BUMP + "\ndef keep(o, d):\n    type(o).held.append(d)",
        "sbox = SBox()",
        "keep(sbox, data['meta'])",
        "SBox.bump({n})",
    ),
    "a-function-on-the-base": (
        "class Base:\n    held = []\n    def hold(self, d):\n        type(self).held.append(d)\n"
        "class Sub(Base):\n    pass\ndef bump(n):\n    Base.held[-1]['n'] += n",
        "s = Sub()",
        "s.hold(data['meta'])",
        "bump({n})",
    ),
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_an_edited_bump_runs_on_what_the_other_cell_kept(nb_runner, shape):
    cls, setup, hold, bump = SHAPES[shape]
    nb_runner.create_notebook([cls, setup, DATA, hold, bump.format(n=5), READ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "R 4498506" in nb_runner.get_output(6)
    nb_runner.set_cell_source(5, bump.format(n=7))
    for _ in range(2):
        nb_runner.run_all()
        assert "R 4498508" in nb_runner.get_output(6), nb_runner.get_raw_output(5) + nb_runner.get_raw_output(6)
