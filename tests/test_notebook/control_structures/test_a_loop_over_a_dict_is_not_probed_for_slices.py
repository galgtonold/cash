"""A loop over a dict is not probed with a slice.

Whether a long loop can be split is asked by reading ``iterable[0:0]``. A
``defaultdict`` answers that by adding the key ``slice(0, 0, None)``, which the
loop then printed and the next one failed on (found replaying a student's
notebook of the JuNE dataset, 'student_2' steps 542-544).
"""

import ast
from collections import defaultdict
from unittest.mock import MagicMock

from cash.notebook.control_structures.split_policy import LoopSplitPolicy

LOOP = ast.parse("for key in cnts:\n    print(key)").body[0]


def test_a_defaultdict_is_left_as_it_was():
    cnts = defaultdict(int, {(i, i + 1): 1 for i in range(200)})
    assert LoopSplitPolicy(MagicMock()).eligible(LOOP, cnts, {}) is None
    assert all(isinstance(key, tuple) for key in cnts)
    assert len(cnts) == 200


def test_a_list_is_still_eligible():
    assert LoopSplitPolicy(MagicMock()).eligible(LOOP, list(range(200)), {"cnts": 1}) == 200
