"""
Exception handling, try/except, custom exceptions, error propagation,
and conditional error recovery across cells.
"""

import textwrap

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress]


class TestExceptionHandlingCaching:
    """Test that exception handling patterns cache correctly."""

    def test_exception_change_propagation(self, nb_runner):
        """Change exception class → downstream catches update."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class MyError(Exception):
                    def __init__(self, code):
                        self.code = code
                        super().__init__(f"Error {code}")
            """),
                textwrap.dedent("""\
                def process():
                    raise MyError(404)
            """),
                textwrap.dedent("""\
                try:
                    process()
                except MyError as e:
                    print(f"code={e.code}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "code=404" in nb_runner.get_output(3)

        # Change exception class
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            class MyError(Exception):
                def __init__(self, code, detail=""):
                    self.code = code
                    self.detail = detail
                    super().__init__(f"Error {code}: {detail}")
        """),
        )
        nb_runner.set_cell_source(
            2,
            textwrap.dedent("""\
            def process():
                raise MyError(500, "server error")
        """),
        )
        nb_runner.set_cell_source(
            3,
            textwrap.dedent("""\
            try:
                process()
            except MyError as e:
                print(f"code={e.code} detail={e.detail}")
        """),
        )
        nb_runner.run_all()
        assert "code=500 detail=server error" in nb_runner.get_output(3)


class TestMultiCellErrorRecovery:
    """Test error recovery spanning multiple cells."""

    def test_sentinel_value_propagation(self, nb_runner):
        """Sentinel values propagate correctly through cache."""
        nb_runner.create_notebook(
            [
                "_MISSING = object()",
                textwrap.dedent("""\
                cache = {'a': 1, 'b': 2}
                def lookup(key):
                    return cache.get(key, _MISSING)
            """),
                textwrap.dedent("""\
                v1 = lookup('a')
                v2 = lookup('z')
                print(f"a={v1} z_missing={v2 is _MISSING}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=1 z_missing=True" in nb_runner.get_output(3)
