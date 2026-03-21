"""Batch 222 – Validator pattern interaction tests.

Tests editing cells with input validation patterns
and verifying correct error/success propagation.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestValidatorPatternEdits:
    """Editing input validation patterns."""

    def test_edit_validation_rules(self, nb_runner):
        """Edit validation rules and check result."""
        nb_runner.create_notebook([
            "def validate_age(age):\n    if age < 0:\n        return 'invalid: negative'\n    if age > 150:\n        return 'invalid: too large'\n    return 'valid'",
            "result = validate_age(25)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = valid" in nb_runner.get_output(2)

        # Test with invalid
        nb_runner.set_cell_source(2, "result = validate_age(-5)\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = invalid: negative" in nb_runner.get_output(2)

    def test_edit_validator_function(self, nb_runner):
        """Edit the validator function itself."""
        nb_runner.create_notebook([
            "def check(value):\n    return 'ok' if isinstance(value, int) else 'fail'",
            "r1 = check(42)\nprint(f'r1 = {r1}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1 = ok" in nb_runner.get_output(2)

        # Make validator stricter
        nb_runner.set_cell_source(1, "def check(value):\n    return 'ok' if isinstance(value, int) and value > 0 else 'fail'")
        nb_runner.run_all()
        assert "r1 = ok" in nb_runner.get_output(2)

    def test_edit_multi_field_validation(self, nb_runner):
        """Edit validation for multiple fields."""
        nb_runner.create_notebook([
            "def validate(name, age):\n    errors = []\n    if not name:\n        errors.append('name required')\n    if age < 0:\n        errors.append('age invalid')\n    return errors if errors else ['all good']",
            "result = validate('Alice', 30)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = ['all good']" in nb_runner.get_output(2)

        # Test with bad data
        nb_runner.set_cell_source(2, "result = validate('', -1)\nprint(f'result = {result}')")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "name required" in out
        assert "age invalid" in out
