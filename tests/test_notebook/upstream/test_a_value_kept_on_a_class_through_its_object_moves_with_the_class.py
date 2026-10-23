"""A value kept on a class through one of its objects moves with the class.

``sbox.hold(data['meta'])`` stores ``data['meta']`` on the class
(``type(self).held.append(d)``), and ``SBox.bump(7)`` changes it through the
class method's ``cls``. The holder link looked only at the classes a
statement read by name, which ``sbox.hold`` does not, and a class method's
change through ``cls`` was not seen as a change to the class at all: a cell
reading ``data`` after an edit to the bump was served the value from before.

So it was when the class the list is on was one the statement names nowhere:
a base class, a metaclass, or the class of an object reached through a list;
and when the bump is a method of the object (``type(self).held``) or of the
metaclass (``K.bump(7)``).
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

pytestmark = pytest.mark.core

DATA = "import time\ndata = (time.sleep(0.2), {'rows': [{'v': i} for i in range(3000)], 'meta': {'n': 1}})[1]"
READ = "result = (time.sleep(0.2), sum(r['v'] for r in data['rows']) + data['meta']['n'])[1]"
SHAPES = {
    "through-an-object": [
        "class SBox:\n    held = []\n    def hold(self, d):\n        type(self).held.append(d)\n"
        "    @classmethod\n    def bump(cls, n):\n        cls.held[-1]['n'] += n",
        "sbox = SBox()",
        DATA,
        "sbox.hold(data['meta'])",
        "SBox.bump(5)",
        READ,
    ],
    "through-the-class": [
        "class Pt:\n    held = []\n    @staticmethod\n    def hold(d):\n        Pt.held.append(d)\n"
        "    @staticmethod\n    def bump(n):\n        Pt.held[-1]['n'] += n",
        "x = 1",
        DATA,
        "Pt.hold(data['meta'])",
        "Pt.bump(5)",
        READ,
    ],
    "a method of the object": [
        "class SBox:\n    held = []\n    def hold(self, d):\n        type(self).held.append(d)\n"
        "    def bump(self, n):\n        type(self).held[-1]['n'] += n",
        "sbox = SBox()",
        DATA,
        "sbox.hold(data['meta'])",
        "sbox.bump(5)",
        READ,
    ],
    "class methods called on the object": [
        "class SBox:\n    held = []\n    @classmethod\n    def hold(cls, d):\n        cls.held.append(d)\n"
        "    @classmethod\n    def bump(cls, n):\n        cls.held[-1]['n'] += n",
        "sbox = SBox()",
        DATA,
        "sbox.hold(data['meta'])",
        "sbox.bump(5)",
        READ,
    ],
    "on the base class": [
        "class Base:\n    held = []\n    def hold(self, d):\n        type(self).held.append(d)\nclass Sub(Base):\n    pass",
        "s = Sub()",
        DATA,
        "s.hold(data['meta'])",
        "Base.held[-1]['n'] += 5",
        READ,
    ],
    "on the metaclass": [
        "class Meta(type):\n    reg = []\nclass K(metaclass=Meta):\n    def hold(self, d):\n"
        "        type(type(self)).reg.append(d)\ndef bump(n):\n    Meta.reg[-1]['n'] += n",
        "k = K()",
        DATA,
        "k.hold(data['meta'])",
        "bump(5)",
        READ,
    ],
    "a metaclass method": [
        "class Meta(type):\n    reg = []\n    def bump(cls, n):\n        cls.reg[-1]['n'] += n\n"
        "class K(metaclass=Meta):\n    def hold(self, d):\n        type(self).reg.append(d)",
        "k = K()",
        DATA,
        "k.hold(data['meta'])",
        "K.bump(5)",
        READ,
    ],
    "an object in a list": [
        "class SBox:\n    held = []\n    def hold(self, d):\n        type(self).held.append(d)\n"
        "    @classmethod\n    def bump(cls, n):\n        cls.held[-1]['n'] += n",
        "boxes = [SBox()]",
        DATA,
        "boxes[0].hold(data['meta'])",
        "SBox.bump(5)",
        READ,
    ],
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_an_edited_change_through_the_class_is_not_served_from_before(cash_magics, shape):
    cells = list(SHAPES[shape])
    for rnd in range(3):
        if rnd == 1:
            cells[4] = cells[4].replace("5", "7")
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        expected = sum(range(3000)) + (6 if rnd == 0 else 8)
        assert cash_magics.shell.user_ns["result"] == expected, f"run {rnd + 1} served a value from before the edit"
