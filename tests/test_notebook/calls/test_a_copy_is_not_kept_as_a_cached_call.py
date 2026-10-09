"""``copy.copy`` and ``copy.deepcopy`` are not cached as calls.

A stored copy costs about as much to restore as to make, and keeping one meant
hashing the argument twice, walking the result and the argument for shared
objects, and pickling a value as large as the argument: 11 s plain, 51 s with
cash for ``deepcopy`` of 87,000 sessions (found replaying a student's notebook
of the JuNE dataset, 'student_8' step 77).
"""

import copy

from cash.notebook.call_interception import interceptable


def test_the_copy_functions_are_left_alone():
    assert not interceptable(copy.deepcopy)
    assert not interceptable(copy.copy)


def test_another_plain_function_is_still_intercepted():
    def helper(x):
        return x

    assert interceptable(helper)
