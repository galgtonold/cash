"""Batch 487: cmath and complex number operations."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestCmathComplexOps:
    def test_complex_arithmetic(self, nb_runner):
        nb_runner.create_notebook([
            "import cmath",
            "z1 = 3 + 4j\nz2 = 1 - 2j\ns = z1 + z2\np = z1 * z2\nprint(f'sum={s} prod={p}')\nprint(f'abs_z1={abs(z1)} phase={round(cmath.phase(z1), 4)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "sum=(4+2j)" in out
        assert "prod=(11-2j)" in out
        assert "abs_z1=5.0" in out

    def test_polar_rect(self, nb_runner):
        nb_runner.create_notebook([
            "import cmath",
            "z = 1 + 1j\nr, phi = cmath.polar(z)\nback = cmath.rect(r, phi)\nprint(f'r={round(r, 4)} phi={round(phi, 4)}')\nprint(f'back_real={round(back.real, 4)} back_imag={round(back.imag, 4)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "r=1.4142" in out
        assert "phi=0.7854" in out
        assert "back_real=1.0" in out
        assert "back_imag=1.0" in out

    def test_complex_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import cmath",
            "z = 3 + 4j\nprint(f'abs={abs(z)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "abs=5.0" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "z = 5 + 12j\nprint(f'abs={abs(z)}')")
        nb_runner.run_all()
        assert "abs=13.0" in nb_runner.get_output(2)
