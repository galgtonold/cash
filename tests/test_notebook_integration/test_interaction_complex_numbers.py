"""Batch 352: complex number operations and cmath."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestComplexNumbers:

    def test_complex_edit(self, nb_runner):
        nb_runner.create_notebook([
            "z = 3 + 4j",
            "mag = abs(z)\nconj = z.conjugate()\nprint(f'mag={mag} conj={conj}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mag=5.0" in nb_runner.get_output(2)
        assert "conj=(3-4j)" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "z = 5 + 12j")
        nb_runner.run_all()
        assert "mag=13.0" in nb_runner.get_output(2)
        assert "conj=(5-12j)" in nb_runner.get_output(2)

    def test_cmath_polar(self, nb_runner):
        nb_runner.create_notebook([
            "import cmath\nz = 1 + 1j",
            "r, phi = cmath.polar(z)\nprint(f'r={round(r, 4)} phi={round(phi, 4)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r=1.4142" in nb_runner.get_output(2)
        assert "phi=0.7854" in nb_runner.get_output(2)
