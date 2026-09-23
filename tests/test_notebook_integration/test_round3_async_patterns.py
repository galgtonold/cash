"""Async/coroutine patterns — cash caching with asyncio code."""

import textwrap

import pytest


@pytest.mark.stress
class TestAsyncBasics:
    """Test async function definition and result caching."""

    def test_async_function_definition_and_call(self, nb_runner):
        """Define async function in one cell, call in another with top-level await."""
        nb_runner.create_notebook(
            [
                "import asyncio",
                textwrap.dedent("""\
                async def fetch_value(x):
                    await asyncio.sleep(0.01)
                    return x * 10
            """),
                textwrap.dedent("""\
                result = await fetch_value(5)
                print(f"result={result}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=50" in nb_runner.get_output(3)

        # Second run should use cache
        nb_runner.reset_cash_state()
        nb_runner.run_all()
        assert "result=50" in nb_runner.get_output(3)

    def test_async_function_change_propagates(self, nb_runner):
        """Changing async function body invalidates downstream."""
        nb_runner.create_notebook(
            [
                "import asyncio",
                textwrap.dedent("""\
                async def compute(x):
                    return x + 1
            """),
                textwrap.dedent("""\
                result = await compute(10)
                print(f"result={result}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=11" in nb_runner.get_output(3)

        # Change async function
        nb_runner.set_cell_source(
            2,
            textwrap.dedent("""\
            async def compute(x):
                return x + 100
        """),
        )
        nb_runner.run_all()
        assert "result=110" in nb_runner.get_output(3)

    def test_async_with_gather(self, nb_runner):
        """asyncio.gather for concurrent tasks via top-level await."""
        nb_runner.create_notebook(
            [
                "import asyncio",
                textwrap.dedent("""\
                async def square(n):
                    return n ** 2
            """),
                textwrap.dedent("""\
                tasks = [square(i) for i in range(5)]
                results = list(await asyncio.gather(*tasks))
                print(f"results={results}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[0, 1, 4, 9, 16]" in nb_runner.get_output(3)
