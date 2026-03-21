"""
Interaction test: complex number arithmetic and polar form.
Tests complex addition, multiplication, conjugate,
polar conversion, and cross-cell complex math pipelines.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestComplexNumberPolar:
    """Test complex number arithmetic and polar conversion across cells."""

    def test_complex_ops(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: basic complex ops
            "z1 = complex(3, 4)\nz2 = complex(1, -2)\nz_sum = z1 + z2\nz_prod = z1 * z2\nprint(f'z1={z1}')\nprint(f'sum={z_sum}')\nprint(f'prod={z_prod}')",
            # Cell 2: conjugate and abs
            "conj = z1.conjugate()\nmag = abs(z1)\nprint(f'conjugate={conj}')\nprint(f'magnitude={mag}')",
            # Cell 3: polar form
            "import cmath\nimport math\nr, theta = cmath.polar(z1)\nback = cmath.rect(r, theta)\nprint(f'r={r}')\nprint(f'theta_deg={math.degrees(theta):.4f}')\nprint(f'roundtrip_real={back.real:.6f}')\nprint(f'roundtrip_imag={back.imag:.6f}')",
        ])
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
        nb_runner.create_notebook([
            "z = complex(0, 1)  # i\nprint(f'z={z}')",
            "z_sq = z * z\nprint(f'z_squared={z_sq}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z_squared=(-1+0j)" in nb_runner.get_output(2)

        # Edit to different complex number
        nb_runner.set_cell_source(1, "z = complex(1, 1)\nprint(f'z={z}')")
        nb_runner.run_cells([1, 2])
        assert "z_squared=2j" in nb_runner.get_output(2)

    def test_complex_cache(self, nb_runner):
        nb_runner.create_notebook([
            "z = complex(5, 12)\nmag = abs(z)\nprint(f'mag={mag}')",
            "normalized = z / mag\nprint(f'norm_real={normalized.real:.4f}')\nprint(f'norm_imag={normalized.imag:.4f}')",
        ])
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
