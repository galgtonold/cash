"""
Batch 316: math module chain operations with caching and edit propagation.
Tests math.sqrt, math.pow, math.log chains and invalidation on edit.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.timeout(90)]


class TestMathChainEdit:
    """Test math module chain operations caching."""

    def test_math_sqrt_chain(self, nb_runner):
        """Chain math.sqrt operations, verify caching."""
        nb_runner.create_notebook([
            "import math",
            "x = 256",
            "y = math.sqrt(x)\nz = math.sqrt(y)",
            "print(f'z={z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "z=4.0" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "z=4.0" in out2

    def test_math_pow_log_edit(self, nb_runner):
        """Edit base value, propagate through pow/log chain."""
        nb_runner.create_notebook([
            "import math",
            "base = 2",
            "powered = math.pow(base, 10)\nlog_val = math.log2(powered)",
            "result = int(log_val)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=10" in out

        nb_runner.set_cell_source(2, "base = 3")
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        # log2(3^10) = 10 * log2(3) ≈ 15.849
        val = int(float(out2.split("result=")[1].strip()))
        assert val == 15  # int truncation of 15.849

    def test_math_trig_chain(self, nb_runner):
        """Trigonometric chain with pi."""
        nb_runner.create_notebook([
            "import math",
            "angle = math.pi / 4",
            "s = math.sin(angle)\nc = math.cos(angle)\nidentity = round(s**2 + c**2, 10)",
            "print(f'identity={identity}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "identity=1.0" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "identity=1.0" in out2
