"""
Interaction test: class method resolution order (MRO) with cooperative super().
Tests diamond inheritance, MRO resolution, and super() chain
across cells with method override behavior.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestMROCooperativeSuper:
    """Test MRO and cooperative super() across cells."""

    def test_diamond_mro(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: define diamond hierarchy
            "class Base:\n    def greet(self):\n        return 'Base'\nclass Left(Base):\n    def greet(self):\n        return 'Left+' + super().greet()\nclass Right(Base):\n    def greet(self):\n        return 'Right+' + super().greet()\nclass Diamond(Left, Right):\n    def greet(self):\n        return 'Diamond+' + super().greet()\nprint('Diamond hierarchy defined')",
            # Cell 2: test MRO
            "d = Diamond()\nresult = d.greet()\nmro = [c.__name__ for c in Diamond.__mro__]\nprint(f'result={result}')\nprint(f'mro={mro}')",
            # Cell 3: verify chain
            "parts = result.split('+')\nprint(f'chain_len={len(parts)}')\nprint(f'starts_with_diamond={parts[0] == \"Diamond\"}')\nprint(f'ends_with_base={parts[-1] == \"Base\"}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "result=Diamond+Left+Right+Base" in out2
        assert "mro=['Diamond', 'Left', 'Right', 'Base', 'object']" in out2
        out3 = nb_runner.get_output(3)
        assert "chain_len=4" in out3
        assert "starts_with_diamond=True" in out3
        assert "ends_with_base=True" in out3

    def test_mro_edit(self, nb_runner):
        nb_runner.create_notebook([
            "class A:\n    def who(self):\n        return 'A'\nclass B(A):\n    def who(self):\n        return 'B+' + super().who()\nclass C(A):\n    def who(self):\n        return 'C+' + super().who()\nclass D(B, C):\n    def who(self):\n        return 'D+' + super().who()\nprint('ABCD defined')",
            "d = D()\nprint(f'who={d.who()}')",
            "length = len(d.who().split('+'))\nprint(f'depth={length}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "who=D+B+C+A" in nb_runner.get_output(2)
        assert "depth=4" in nb_runner.get_output(3)

        # Edit C to not call super
        nb_runner.set_cell_source(1, "class A:\n    def who(self):\n        return 'A'\nclass B(A):\n    def who(self):\n        return 'B+' + super().who()\nclass C(A):\n    def who(self):\n        return 'C_STOP'\nclass D(B, C):\n    def who(self):\n        return 'D+' + super().who()\nprint('ABCD redefined')")
        nb_runner.run_cells([1, 2, 3])
        assert "who=D+B+C_STOP" in nb_runner.get_output(2)
        assert "depth=3" in nb_runner.get_output(3)

    def test_mro_cache(self, nb_runner):
        nb_runner.create_notebook([
            "class X:\n    val = 'X'\nclass Y(X):\n    val = 'Y'\nclass Z(Y):\n    pass  # inherits from Y\nprint('XYZ defined')",
            "z = Z()\nprint(f'val={z.val}')\nprint(f'mro={[c.__name__ for c in Z.__mro__]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val=Y" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "val=Y" in nb_runner.get_output(2)
