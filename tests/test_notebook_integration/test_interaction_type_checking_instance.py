"""Batch 456: type checking with isinstance and issubclass."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestTypeCheckingInstanceSubclass:
    def test_isinstance_multi(self, nb_runner):
        nb_runner.create_notebook([
            "vals = [42, 'hello', 3.14, True, [1, 2]]",
            "types = [(v, type(v).__name__, isinstance(v, (int, float))) for v in vals]\nresults = [(t[1], t[2]) for t in types]\nprint(f'results={results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "('int', True)" in out
        assert "('str', False)" in out

    def test_issubclass(self, nb_runner):
        nb_runner.create_notebook([
            "class Base: pass\nclass Mid(Base): pass\nclass Leaf(Mid): pass",
            "r1 = issubclass(Leaf, Base)\nr2 = issubclass(Leaf, Mid)\nr3 = issubclass(Base, Leaf)\nprint(f'r1={r1} r2={r2} r3={r3}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=True" in nb_runner.get_output(2)
        assert "r2=True" in nb_runner.get_output(2)
        assert "r3=False" in nb_runner.get_output(2)

    def test_type_edit(self, nb_runner):
        nb_runner.create_notebook([
            "val = 42",
            "t = type(val).__name__\nis_num = isinstance(val, (int, float))\nprint(f'type={t} is_num={is_num}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "type=int" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "val = 'hello'")
        nb_runner.run_all()
        assert "type=str" in nb_runner.get_output(2)
        assert "is_num=False" in nb_runner.get_output(2)
