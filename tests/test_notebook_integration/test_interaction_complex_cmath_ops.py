"""
Interaction test: complex number operations with cmath module.
Tests complex arithmetic, cmath functions (polar, rect, phase),
and cross-cell complex number manipulation.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestComplexCmathOps:
    """Test complex number operations with cmath across cells."""

    def test_complex_cmath(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: complex arithmetic
            "z1 = 3 + 4j\nz2 = 1 - 2j\nsum_z = z1 + z2\nprod_z = z1 * z2\nprint(f'sum={sum_z}')\nprint(f'prod={prod_z}')",
            # Cell 2: cmath functions
            "import cmath\nabs_z1 = abs(z1)\nphase_z1 = cmath.phase(z1)\npolar = cmath.polar(z1)\nprint(f'abs={abs_z1}')\nprint(f'phase={phase_z1:.4f}')\nprint(f'polar_r={polar[0]:.1f}')",
            # Cell 3: rect conversion
            "r, theta = cmath.polar(z1)\nback = cmath.rect(r, theta)\nprint(f'real={back.real:.1f}')\nprint(f'imag={back.imag:.1f}')\nprint(f'roundtrip={abs(back - z1) < 1e-10}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "sum=(4+2j)" in out1
        assert "prod=(11-2j)" in out1
        out2 = nb_runner.get_output(2)
        assert "abs=5.0" in out2
        out3 = nb_runner.get_output(3)
        assert "real=3.0" in out3
        assert "imag=4.0" in out3
        assert "roundtrip=True" in out3

    def test_complex_edit(self, nb_runner):
        nb_runner.create_notebook([
            "z = 3 + 4j\nmag = abs(z)\nprint(f'mag={mag}')",
            "conjugate = z.conjugate()\nprint(f'conj={conjugate}')",
            "product = z * conjugate\nprint(f'prod={product}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mag=5.0" in nb_runner.get_output(1)
        assert "conj=(3-4j)" in nb_runner.get_output(2)
        assert "prod=(25+0j)" in nb_runner.get_output(3)

        # Edit z
        nb_runner.set_cell_source(1, "z = 5 + 12j\nmag = abs(z)\nprint(f'mag={mag}')")
        nb_runner.run_cells([1, 2, 3])
        assert "mag=13.0" in nb_runner.get_output(1)
        assert "conj=(5-12j)" in nb_runner.get_output(2)
        assert "prod=(169+0j)" in nb_runner.get_output(3)

    def test_complex_cache(self, nb_runner):
        nb_runner.create_notebook([
            "import cmath\nz = cmath.sqrt(-1)\nprint(f'sqrt_neg1={z}')",
            "is_imag = z.real == 0 and z.imag == 1\nprint(f'is_i={is_imag}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sqrt_neg1=1j" in nb_runner.get_output(1)
        assert "is_i=True" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "is_i=True" in nb_runner.get_output(2)
