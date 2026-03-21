"""Batch 414: complex number arithmetic."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestComplexNumberArith:
    def test_complex_basic(self, nb_runner):
        nb_runner.create_notebook([
            "z1 = 3 + 4j\nz2 = 1 - 2j",
            "add = z1 + z2\nmul = z1 * z2\nprint(f'add={add} mul={mul}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "add=(4+2j)" in out
        assert "mul=(11-2j)" in out

    def test_complex_magnitude(self, nb_runner):
        nb_runner.create_notebook([
            "z = 3 + 4j",
            "mag = abs(z)\nreal = z.real\nimag = z.imag\nconj = z.conjugate()\nprint(f'mag={mag} real={real} imag={imag} conj={conj}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "mag=5.0" in out
        assert "real=3.0" in out
        assert "conj=(3-4j)" in out

    def test_complex_edit(self, nb_runner):
        nb_runner.create_notebook([
            "z = 1 + 1j",
            "squared = z ** 2\nprint(f'squared={squared}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "squared=2j" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "z = 2 + 3j")
        nb_runner.run_all()
        assert "squared=(-5+12j)" in nb_runner.get_output(2)
