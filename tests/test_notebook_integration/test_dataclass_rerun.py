"""Hidden mutation in a ``@dataclass`` ``__post_init__`` must reset on isolated
re-run (CAS-79, extends CAS-73). The dataclass decorator synthesises ``__init__``
and calls the user's ``__post_init__``; a mutation there (a module list or a
``ClassVar``) is invisible to the construction site and accumulates on re-run.

Each construction mutates ONCE, so the correct value is identical on the first
run and every re-run.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream]


def _rerun(nb_runner, setup, cell, expect):
    nb_runner.create_notebook([setup, cell])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert expect in nb_runner.get_output(2), f"first: {nb_runner.get_output(2)!r}"
    nb_runner.run_cell(2)
    assert expect in nb_runner.get_output(2), f"re-run: {nb_runner.get_output(2)!r}"


# --- must reset ---------------------------------------------------------------

def test_post_init_appends_module_list(nb_runner):
    _rerun(nb_runner,
           "from dataclasses import dataclass\nlog = []\n@dataclass\nclass Reg:\n    name: str\n    def __post_init__(self):\n        log.append(self.name)",
           "r = Reg('a')\nprint('log', len(log))", "log 1")


def test_post_init_appends_class_var(nb_runner):
    _rerun(nb_runner,
           "from dataclasses import dataclass\nfrom typing import ClassVar, List\n@dataclass\nclass Node:\n    val: int\n    registry: ClassVar[List] = []\n    def __post_init__(self):\n        Node.registry.append(self.val)",
           "n = Node(7)\nprint('reg', len(Node.registry))", "reg 1")


# --- must NOT over-invalidate (pure) ------------------------------------------

def test_pure_dataclass_not_over_invalidated(nb_runner):
    _rerun(nb_runner,
           "from dataclasses import dataclass\n@dataclass\nclass Point:\n    x: int\n    y: int\n    def norm(self):\n        return self.x + self.y",
           "p = Point(3, 4)\nprint('n', p.norm())", "n 7")
