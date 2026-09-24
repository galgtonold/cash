"""Complex numbers and cmath across cells."""

import pytest


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestComplexNumbers:
    """complex number operations and cmath."""

    def test_complex_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "z = 3 + 4j",
                "mag = abs(z)\nconj = z.conjugate()\nprint(f'mag={mag} conj={conj}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "import cmath\nz = 1 + 1j",
                "r, phi = cmath.polar(z)\nprint(f'r={round(r, 4)} phi={round(phi, 4)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r=1.4142" in nb_runner.get_output(2)
        assert "phi=0.7854" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestComplexNumberArith:
    """complex number arithmetic."""

    def test_complex_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "z1 = 3 + 4j\nz2 = 1 - 2j",
                "add = z1 + z2\nmul = z1 * z2\nprint(f'add={add} mul={mul}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "add=(4+2j)" in out
        assert "mul=(11-2j)" in out

    def test_complex_magnitude(self, nb_runner):
        nb_runner.create_notebook(
            [
                "z = 3 + 4j",
                "mag = abs(z)\nreal = z.real\nimag = z.imag\nconj = z.conjugate()\nprint(f'mag={mag} real={real} imag={imag} conj={conj}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "mag=5.0" in out
        assert "real=3.0" in out
        assert "conj=(3-4j)" in out

    def test_complex_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "z = 1 + 1j",
                "squared = z ** 2\nprint(f'squared={squared}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "squared=2j" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "z = 2 + 3j")
        nb_runner.run_all()
        assert "squared=(-5+12j)" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestComplexNumberPolar:
    """Test complex number arithmetic and polar conversion across cells."""

    def test_complex_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: basic complex ops
                "z1 = complex(3, 4)\nz2 = complex(1, -2)\nz_sum = z1 + z2\nz_prod = z1 * z2\nprint(f'z1={z1}')\nprint(f'sum={z_sum}')\nprint(f'prod={z_prod}')",
                # Cell 2: conjugate and abs
                "conj = z1.conjugate()\nmag = abs(z1)\nprint(f'conjugate={conj}')\nprint(f'magnitude={mag}')",
                # Cell 3: polar form
                "import cmath\nimport math\nr, theta = cmath.polar(z1)\nback = cmath.rect(r, theta)\nprint(f'r={r}')\nprint(f'theta_deg={math.degrees(theta):.4f}')\nprint(f'roundtrip_real={back.real:.6f}')\nprint(f'roundtrip_imag={back.imag:.6f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "z1=(3+4j)" in out1
        assert "sum=(4+2j)" in out1
        assert "prod=(11-2j)" in out1
        out2 = nb_runner.get_output(2)
        assert "conjugate=(3-4j)" in out2
        assert "magnitude=5.0" in out2
        out3 = nb_runner.get_output(3)
        assert "r=5.0" in out3
        assert "roundtrip_real=3.0" in out3

    def test_complex_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "z = complex(0, 1)  # i\nprint(f'z={z}')",
                "z_sq = z * z\nprint(f'z_squared={z_sq}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z_squared=(-1+0j)" in nb_runner.get_output(2)

        # Edit to different complex number
        nb_runner.set_cell_source(1, "z = complex(1, 1)\nprint(f'z={z}')")
        nb_runner.run_cells([1, 2])
        assert "z_squared=2j" in nb_runner.get_output(2)

    def test_complex_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "z = complex(5, 12)\nmag = abs(z)\nprint(f'mag={mag}')",
                "normalized = z / mag\nprint(f'norm_real={normalized.real:.4f}')\nprint(f'norm_imag={normalized.imag:.4f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mag=13.0" in nb_runner.get_output(1)
        out2 = nb_runner.get_output(2)
        assert "norm_real=0.3846" in out2
        assert "norm_imag=0.9231" in out2

        # Re-run - cache
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "norm_real=0.3846" in out2


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestCmathComplexOps:
    """cmath and complex number operations."""

    def test_complex_arithmetic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import cmath",
                "z1 = 3 + 4j\nz2 = 1 - 2j\ns = z1 + z2\np = z1 * z2\nprint(f'sum={s} prod={p}')\nprint(f'abs_z1={abs(z1)} phase={round(cmath.phase(z1), 4)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "sum=(4+2j)" in out
        assert "prod=(11-2j)" in out
        assert "abs_z1=5.0" in out

    def test_polar_rect(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import cmath",
                "z = 1 + 1j\nr, phi = cmath.polar(z)\nback = cmath.rect(r, phi)\nprint(f'r={round(r, 4)} phi={round(phi, 4)}')\nprint(f'back_real={round(back.real, 4)} back_imag={round(back.imag, 4)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "r=1.4142" in out
        assert "phi=0.7854" in out
        assert "back_real=1.0" in out
        assert "back_imag=1.0" in out

    def test_complex_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import cmath",
                "z = 3 + 4j\nprint(f'abs={abs(z)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "abs=5.0" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "z = 5 + 12j\nprint(f'abs={abs(z)}')")
        nb_runner.run_all()
        assert "abs=13.0" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestComplexCmathOps:
    """Test complex number operations with cmath across cells."""

    def test_complex_cmath(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: complex arithmetic
                "z1 = 3 + 4j\nz2 = 1 - 2j\nsum_z = z1 + z2\nprod_z = z1 * z2\nprint(f'sum={sum_z}')\nprint(f'prod={prod_z}')",
                # Cell 2: cmath functions
                "import cmath\nabs_z1 = abs(z1)\nphase_z1 = cmath.phase(z1)\npolar = cmath.polar(z1)\nprint(f'abs={abs_z1}')\nprint(f'phase={phase_z1:.4f}')\nprint(f'polar_r={polar[0]:.1f}')",
                # Cell 3: rect conversion
                "r, theta = cmath.polar(z1)\nback = cmath.rect(r, theta)\nprint(f'real={back.real:.1f}')\nprint(f'imag={back.imag:.1f}')\nprint(f'roundtrip={abs(back - z1) < 1e-10}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "z = 3 + 4j\nmag = abs(z)\nprint(f'mag={mag}')",
                "conjugate = z.conjugate()\nprint(f'conj={conjugate}')",
                "product = z * conjugate\nprint(f'prod={product}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "import cmath\nz = cmath.sqrt(-1)\nprint(f'sqrt_neg1={z}')",
                "is_imag = z.real == 0 and z.imag == 1\nprint(f'is_i={is_imag}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sqrt_neg1=1j" in nb_runner.get_output(1)
        assert "is_i=True" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "is_i=True" in nb_runner.get_output(2)
