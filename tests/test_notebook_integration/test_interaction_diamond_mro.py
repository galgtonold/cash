"""Batch 369: multiple inheritance diamond pattern and MRO."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDiamondMRO:
    def test_diamond_basic(self, nb_runner):
        nb_runner.create_notebook([
            "class A:\n    def who(self):\n        return 'A'\nclass B(A):\n    def who(self):\n        return 'B'\nclass C(A):\n    def who(self):\n        return 'C'\nclass D(B, C):\n    pass",
            "d = D()\nresult = d.who()\nmro = [c.__name__ for c in D.__mro__]\nprint(f'result={result} mro={mro}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=B" in nb_runner.get_output(2)
        assert "mro=['D', 'B', 'C', 'A', 'object']" in nb_runner.get_output(2)

    def test_diamond_edit_order(self, nb_runner):
        nb_runner.create_notebook([
            "class A:\n    val = 1\nclass B(A):\n    val = 2\nclass C(A):\n    val = 3\nclass D(B, C):\n    pass",
            "result = D.val\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=2" in nb_runner.get_output(2)
        # Switch MRO order
        nb_runner.set_cell_source(1, "class A:\n    val = 1\nclass B(A):\n    val = 2\nclass C(A):\n    val = 3\nclass D(C, B):\n    pass")
        nb_runner.run_all()
        assert "result=3" in nb_runner.get_output(2)

    def test_super_cooperative(self, nb_runner):
        nb_runner.create_notebook([
            "class Base:\n    def __init__(self):\n        self.log = ['Base']\nclass Left(Base):\n    def __init__(self):\n        super().__init__()\n        self.log.append('Left')\nclass Right(Base):\n    def __init__(self):\n        super().__init__()\n        self.log.append('Right')\nclass Child(Left, Right):\n    def __init__(self):\n        super().__init__()\n        self.log.append('Child')",
            "c = Child()\nprint(f'log={c.log}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "log=['Base', 'Right', 'Left', 'Child']" in nb_runner.get_output(2)
