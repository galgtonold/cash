"""A loop refused because the notebook holds a container too big to look
through names that container's variable.

E10's progress-bar loops ran every time with "the notebook holds containers
too big to look through ...", and nothing said which variable to look at.
"""

from __future__ import annotations

from cash.notebook.statement.in_place import refusal


class P:
    def __init__(self):
        self.n = 0
        self.seen = []


def test_the_refusal_names_the_variable():
    p = P()
    namespace = {"p": p, "small": [1, 2], "big": [[j] for j in range(20000)]}

    why = refusal(p, namespace)

    assert why is not None
    assert "too big to look through ('big')" in why
    assert why.name == "big"


def test_without_the_big_container_the_loop_is_restored_into():
    p = P()

    assert refusal(p, {"p": p, "small": [[1], [2]]}) is None


def test_an_id_kept_in_a_dict_is_still_found():
    p = P()
    namespace = {"p": p, "reg": {id(p.seen): "seen"}}

    assert "id()" in refusal(p, namespace)
