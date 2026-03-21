"""Batch 452: string isdigit/isalpha/isalnum validation."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStringValidation:
    def test_is_methods(self, nb_runner):
        nb_runner.create_notebook([
            "s1 = '12345'\ns2 = 'hello'\ns3 = 'hello123'\ns4 = 'Hello World'",
            "d = s1.isdigit()\na = s2.isalpha()\nan = s3.isalnum()\nsp = s4.isalpha()\nprint(f'd={d} a={a} an={an} sp={sp}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "d=True" in out
        assert "a=True" in out
        assert "an=True" in out
        assert "sp=False" in out

    def test_isupper_islower(self, nb_runner):
        nb_runner.create_notebook([
            "a = 'HELLO'\nb = 'hello'\nc = 'Hello'",
            "r = f'{a.isupper()},{b.islower()},{c.istitle()}'\nprint(f'r={r}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r=True,True,True" in nb_runner.get_output(2)

    def test_validation_edit(self, nb_runner):
        nb_runner.create_notebook([
            "token = 'abc123'",
            "valid = token.isalnum()\nprint(f'valid={valid}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "valid=True" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "token = 'abc 123'")
        nb_runner.run_all()
        assert "valid=False" in nb_runner.get_output(2)
