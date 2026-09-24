"""Editing numeric, math, string and regex computations."""

import pytest

pytestmark = [pytest.mark.stress]


# Numeric precision and math interaction tests.
#
# Tests with floating point, integer overflow, precision changes,
# and mathematical operations combined with cell edits.
@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestFloatingPointEdits:
    """Floating point operations with edits."""

    def test_edit_precision(self, nb_runner):
        """Edit precision of rounding."""
        nb_runner.create_notebook(
            [
                "value = 3.141592653589793",
                "rounded = round(value, 2)\nprint(f'rounded = {rounded}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "rounded = 3.14" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "rounded = round(value, 4)\nprint(f'rounded = {rounded}')")
        nb_runner.run_all()
        assert "rounded = 3.1416" in nb_runner.get_output(2)

    def test_edit_math_operation(self, nb_runner):
        """Edit mathematical operation."""
        nb_runner.create_notebook(
            [
                "import math\nx = 16",
                "result = math.sqrt(x)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 4.0" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "result = math.log2(x)\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 4.0" in nb_runner.get_output(2)

    def test_accumulate_with_precision(self, nb_runner):
        """Accumulation with float precision."""
        nb_runner.create_notebook(
            [
                "values = [0.1] * 10",
                "total = sum(values)\nprint(f'total = {round(total, 1)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 1.0" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "values = [0.1] * 100")
        nb_runner.run_all()
        assert "total = 10.0" in nb_runner.get_output(2)


@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestLargeNumberEdits:
    """Large number operations with edits."""

    def test_edit_exponent(self, nb_runner):
        """Edit exponentiation."""
        nb_runner.create_notebook(
            [
                "base = 2\nexp = 10",
                "result = base ** exp\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 1024" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "base = 2\nexp = 20")
        nb_runner.run_all()
        assert "result = 1048576" in nb_runner.get_output(2)

    def test_factorial_edit(self, nb_runner):
        """Edit factorial input."""
        nb_runner.create_notebook(
            [
                "import math\nn = 5",
                "result = math.factorial(n)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 120" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "import math\nn = 10")
        nb_runner.run_all()
        assert "result = 3628800" in nb_runner.get_output(2)


# math module chain operations with caching and edit propagation.
# Tests math.sqrt, math.pow, math.log chains and invalidation on edit.
@pytest.mark.integration
@pytest.mark.timeout(90)
class TestMathChainEdit:
    """Test math module chain operations caching."""

    def test_math_sqrt_chain(self, nb_runner):
        """Chain math.sqrt operations, verify caching."""
        nb_runner.create_notebook(
            [
                "import math",
                "x = 256",
                "y = math.sqrt(x)\nz = math.sqrt(y)",
                "print(f'z={z}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "import math",
                "base = 2",
                "powered = math.pow(base, 10)\nlog_val = math.log2(powered)",
                "result = int(log_val)\nprint(f'result={result}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "import math",
                "angle = math.pi / 4",
                "s = math.sin(angle)\nc = math.cos(angle)\nidentity = round(s**2 + c**2, 10)",
                "print(f'identity={identity}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "identity=1.0" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "identity=1.0" in out2


# String operations and formatting interaction tests.
#
# Tests where users perform string operations across cells,
# edit string content and formatting, and verify caching
# handles string changes correctly.
@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestStringEdits:
    """String manipulation with cell edits."""

    def test_edit_format_string(self, nb_runner):
        """Edit the format string itself."""
        nb_runner.create_notebook(
            [
                "x = 42",
                "msg = f'The answer is {x}'\nprint(msg)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "The answer is 42" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "msg = f'Value: {x} (hex: {hex(x)})'\nprint(msg)")
        nb_runner.run_all()
        assert "Value: 42 (hex: 0x2a)" in nb_runner.get_output(2)

    def test_string_concatenation_chain(self, nb_runner):
        """Chain of string concatenation, edit source."""
        nb_runner.create_notebook(
            [
                "first = 'Hello'",
                "second = first + ' World'",
                "third = second + '!'\nprint(third)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Hello World!" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "first = 'Goodbye'")
        nb_runner.run_all()
        assert "Goodbye World!" in nb_runner.get_output(3)

    def test_string_method_chain_edit(self, nb_runner):
        """String methods, edit the method call."""
        nb_runner.create_notebook(
            [
                "text = '  Hello World  '",
                "processed = text.strip()\nprint(f'|{processed}|')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "|Hello World|" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "processed = text.strip().upper()\nprint(f'|{processed}|')")
        nb_runner.run_all()
        assert "|HELLO WORLD|" in nb_runner.get_output(2)


@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestStringParsingEdits:
    """String parsing patterns with cell edits."""

    def test_split_and_join_edit_delimiter(self, nb_runner):
        """Split/join with delimiter change."""
        nb_runner.create_notebook(
            [
                "raw = 'a,b,c,d'",
                "parts = raw.split(',')\nresult = '-'.join(parts)\nprint(result)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a-b-c-d" in nb_runner.get_output(2)

        # Change join delimiter
        nb_runner.set_cell_source(2, "parts = raw.split(',')\nresult = ' | '.join(parts)\nprint(result)")
        nb_runner.run_all()
        assert "a | b | c | d" in nb_runner.get_output(2)

    def test_regex_pattern_edit(self, nb_runner):
        """Regex pattern change."""
        nb_runner.create_notebook(
            [
                "import re\ntext = 'abc 123 def 456'",
                "nums = re.findall(r'\\d+', text)\nprint(f'nums = {nums}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "nums = ['123', '456']" in nb_runner.get_output(2)

        # Change to match words
        nb_runner.set_cell_source(2, "words = re.findall(r'[a-z]+', text)\nprint(f'words = {words}')")
        nb_runner.run_all()
        assert "words = ['abc', 'def']" in nb_runner.get_output(2)


@pytest.mark.timeout(90)
class TestRegexPatternEdit:
    """re (regex) pattern matching with cell edits."""

    def test_regex_findall(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import re\ntext = 'Call 555-1234 or 555-5678 for info'",
                "pattern = r'\\d{3}-\\d{4}'\nnumbers = re.findall(pattern, text)\nprint(f'numbers={numbers}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "numbers=['555-1234', '555-5678']" in nb_runner.get_output(2)

    def test_regex_edit_text(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import re\ntext = 'apple 3 banana 7 cherry 12'",
                "nums = [int(x) for x in re.findall(r'\\d+', text)]\ntotal = sum(nums)\nprint(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=22" in nb_runner.get_output(2)
        # Edit text
        nb_runner.set_cell_source(1, "import re\ntext = 'x 100 y 200 z 300'")
        nb_runner.run_all()
        assert "total=600" in nb_runner.get_output(2)
