"""Batch 211 – Complex f-string interaction tests.

Tests editing cells that contain complex f-string
expressions and verifying proper output propagation.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestFStringComplexEdits:
    """Editing complex f-string patterns."""

    def test_edit_fstring_expression(self, nb_runner):
        """Edit data used in f-string with embedded expression."""
        nb_runner.create_notebook([
            "price = 19.99\nqty = 3",
            "total = price * qty\nprint(f'Total: ${total:.2f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Total: $59.97" in nb_runner.get_output(2)

        # Change price
        nb_runner.set_cell_source(1, "price = 24.99\nqty = 3")
        nb_runner.run_all()
        assert "Total: $74.97" in nb_runner.get_output(2)

    def test_edit_fstring_conditional(self, nb_runner):
        """Edit data used in f-string with conditional."""
        nb_runner.create_notebook([
            "score = 85",
            "grade = 'A' if score >= 90 else 'B' if score >= 80 else 'C'\nprint(f'Score {score} => grade {grade}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Score 85 => grade B" in nb_runner.get_output(2)

        # Raise score
        nb_runner.set_cell_source(1, "score = 95")
        nb_runner.run_all()
        assert "Score 95 => grade A" in nb_runner.get_output(2)

    def test_edit_fstring_multiline(self, nb_runner):
        """Edit data used in multi-line f-string output."""
        nb_runner.create_notebook([
            "name = 'Alice'\nage = 30\ncity = 'NYC'",
            "info = f'Name: {name}, Age: {age}, City: {city}'\nprint(info)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Name: Alice, Age: 30, City: NYC" in nb_runner.get_output(2)

        # Update all fields
        nb_runner.set_cell_source(1, "name = 'Bob'\nage = 25\ncity = 'LA'")
        nb_runner.run_all()
        assert "Name: Bob, Age: 25, City: LA" in nb_runner.get_output(2)

    def test_edit_fstring_nested_access(self, nb_runner):
        """Edit dict data used in f-string with nested access."""
        nb_runner.create_notebook([
            "user = {'name': 'Alice', 'scores': [90, 85, 78]}",
            "avg = sum(user['scores']) / len(user['scores'])\nprint(f'{user[\"name\"]}: avg={avg:.1f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Alice: avg=84.3" in nb_runner.get_output(2)

        # Change user data
        nb_runner.set_cell_source(1, "user = {'name': 'Bob', 'scores': [100, 95, 90]}")
        nb_runner.run_all()
        assert "Bob: avg=95.0" in nb_runner.get_output(2)
