"""Batch 492: math ceil floor log gcd lcm."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestMathCeilFloorLogGcd:
    def test_ceil_floor(self, nb_runner):
        nb_runner.create_notebook([
            "import math",
            "vals = [3.2, -3.2, 0.5, -0.5]\nceils = [math.ceil(v) for v in vals]\nfloors = [math.floor(v) for v in vals]\nprint(f'ceils={ceils} floors={floors}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "ceils=[4, -3, 1, 0]" in out
        assert "floors=[3, -4, 0, -1]" in out

    def test_log_gcd_lcm(self, nb_runner):
        nb_runner.create_notebook([
            "import math",
            "log2_8 = math.log2(8)\nlog10_1000 = math.log10(1000)\ng = math.gcd(48, 18)\nlcm_val = math.lcm(12, 18)\nprint(f'log2_8={log2_8} log10_1000={log10_1000} gcd={g} lcm={lcm_val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "log2_8=3.0" in out
        assert "log10_1000=3.0" in out
        assert "gcd=6" in out
        assert "lcm=36" in out

