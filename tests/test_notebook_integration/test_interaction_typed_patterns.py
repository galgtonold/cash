"""Batch 251 – Type annotation and typed data patterns.

Tests typed function signatures with edits, verifying runtime behavior.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestTypeAnnotationPatterns:
    """Typed function and data patterns with edits."""

    def test_typed_function_edit(self, nb_runner):
        """Edit typed function body, return type stays consistent."""
        nb_runner.create_notebook([
            "def process(items: list[int]) -> int:\n    return sum(items)",
            "data: list[int] = [10, 20, 30]\nresult: int = process(data)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 60" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "def process(items: list[int]) -> int:\n    return max(items) - min(items)",
        )
        nb_runner.run_all()
        assert "result = 20" in nb_runner.get_output(2)

    def test_typed_dict_pattern(self, nb_runner):
        """Edit TypedDict-like dict, downstream reflects."""
        nb_runner.create_notebook([
            "record: dict[str, int | str] = {'name': 'Alice', 'age': 30, 'score': 95}",
            "summary = f\"{record['name']}: age={record['age']}, score={record['score']}\"\nprint(f'summary = {summary}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "summary = Alice: age=30, score=95" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "record: dict[str, int | str] = {'name': 'Bob', 'age': 25, 'score': 88}",
        )
        nb_runner.run_all()
        assert "summary = Bob: age=25, score=88" in nb_runner.get_output(2)

    def test_optional_type_edit(self, nb_runner):
        """Edit function with Optional param."""
        nb_runner.create_notebook([
            "def greet(name: str, title: str | None = None) -> str:\n    if title:\n        return f'{title} {name}'\n    return name",
            "msg = greet('Alice', 'Dr')\nprint(f'msg = {msg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "msg = Dr Alice" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "def greet(name: str, title: str | None = None) -> str:\n    if title:\n        return f'{title}. {name}'\n    return f'Dear {name}'",
        )
        nb_runner.run_all()
        assert "msg = Dr. Alice" in nb_runner.get_output(2)
