"""Batch 505: multiple inheritance and MRO resolution."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestMultipleInheritanceMRO:
    def test_diamond_mro(self, nb_runner):
        nb_runner.create_notebook([
            "pass  # setup",
            "class A:\n    def who(self): return 'A'\nclass B(A):\n    def who(self): return 'B'\nclass C(A):\n    def who(self): return 'C'\nclass D(B, C):\n    pass\nd = D()\nmro = [c.__name__ for c in D.__mro__]\nprint(f'who={d.who()} mro={mro}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "who=B" in out
        assert "mro=['D', 'B', 'C', 'A', 'object']" in out

    def test_super_chain(self, nb_runner):
        nb_runner.create_notebook([
            "pass  # setup",
            "class Base:\n    def greet(self): return 'Base'\nclass Left(Base):\n    def greet(self): return 'Left+' + super().greet()\nclass Right(Base):\n    def greet(self): return 'Right+' + super().greet()\nclass Child(Left, Right):\n    def greet(self): return 'Child+' + super().greet()\nresult = Child().greet()\nprint(f'chain={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "chain=Child+Left+Right+Base" in nb_runner.get_output(2)

    def test_mro_edit(self, nb_runner):
        nb_runner.create_notebook([
            "pass  # setup",
            "class X:\n    val = 10\nclass Y(X):\n    val = 20\nclass Z(Y):\n    pass\nprint(f'val={Z.val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val=20" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "class X:\n    val = 10\nclass Y(X):\n    pass\nclass Z(Y):\n    pass\nprint(f'val={Z.val}')")
        nb_runner.run_all()
        assert "val=10" in nb_runner.get_output(2)
