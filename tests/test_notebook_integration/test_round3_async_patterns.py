"""Batch 44: Async/coroutine patterns — cash caching with asyncio code."""
import textwrap
import pytest


@pytest.mark.stress
class TestAsyncBasics:
    """Test async function definition and result caching."""

    def test_async_function_definition_and_call(self, nb_runner):
        """Define async function in one cell, call in another with top-level await."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=50" in nb_runner.get_output(3)

        # Second run should use cache
        nb_runner.reset_cash_state()
        nb_runner.run_all()
        assert "result=50" in nb_runner.get_output(3)

    def test_async_function_change_propagates(self, nb_runner):
        """Changing async function body invalidates downstream."""
        nb_runner.create_notebook([
            "import asyncio",
            textwrap.dedent("""\
                async def compute(x):
                    return x + 1
            """),
            textwrap.dedent("""\
                result = await compute(10)
                print(f"result={result}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=11" in nb_runner.get_output(3)

        # Change async function
        nb_runner.set_cell_source(2, textwrap.dedent("""\
            async def compute(x):
                return x + 100
        """))
        nb_runner.run_all()
        assert "result=110" in nb_runner.get_output(3)

    def test_multiple_async_functions(self, nb_runner):
        """Multiple async functions composed together."""
        nb_runner.create_notebook([
            "import asyncio",
            textwrap.dedent("""\
                async def step1(x):
                    return x * 2
                async def step2(x):
                    return x + 10
            """),
            textwrap.dedent("""\
                a = await step1(5)
                result = await step2(a)
                print(f"result={result}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=20" in nb_runner.get_output(3)

    def test_async_with_gather(self, nb_runner):
        """asyncio.gather for concurrent tasks via top-level await."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[0, 1, 4, 9, 16]" in nb_runner.get_output(3)


@pytest.mark.stress
class TestAsyncGenerators:
    """Test async generator patterns."""

    def test_async_generator_collected(self, nb_runner):
        """Async generator collected to list via top-level await."""
        nb_runner.create_notebook([
            "import asyncio",
            textwrap.dedent("""\
                async def gen_values(n):
                    for i in range(n):
                        yield i * 3

                async def collect():
                    return [v async for v in gen_values(5)]

                values = await collect()
                print(f"values={values}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "values=[0, 3, 6, 9, 12]" in nb_runner.get_output(2)

    def test_async_context_manager(self, nb_runner):
        """Async context manager with __aenter__/__aexit__."""
        nb_runner.create_notebook([
            "import asyncio",
            textwrap.dedent("""\
                class AsyncResource:
                    def __init__(self, name):
                        self.name = name
                        self.log = []
                    async def __aenter__(self):
                        self.log.append(f"open:{self.name}")
                        return self
                    async def __aexit__(self, *args):
                        self.log.append(f"close:{self.name}")

                async def use_resource():
                    async with AsyncResource("db") as r:
                        r.log.append("query")
                    return r.log

                log = await use_resource()
                print(f"log={log}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "log=['open:db', 'query', 'close:db']" in nb_runner.get_output(2)


@pytest.mark.stress
class TestCoroutinePatterns:
    """Test coroutine-specific patterns."""

    def test_coroutine_result_cached(self, nb_runner):
        """Result of await is properly cached."""
        nb_runner.create_notebook([
            "import asyncio",
            textwrap.dedent("""\
                async def expensive():
                    total = sum(range(10000))
                    return total
                result = await expensive()
                print(f"result={result}")
            """),
            textwrap.dedent("""\
                doubled = result * 2
                print(f"doubled={doubled}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=49995000" in nb_runner.get_output(2)
        assert "doubled=99990000" in nb_runner.get_output(3)

    def test_async_exception_handling(self, nb_runner):
        """Async function with exception handling."""
        nb_runner.create_notebook([
            "import asyncio",
            textwrap.dedent("""\
                async def safe_divide(a, b):
                    try:
                        return a / b
                    except ZeroDivisionError:
                        return float('inf')

                r1 = await safe_divide(10, 2)
                r2 = await safe_divide(10, 0)
                print(f"r1={r1} r2={r2}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=5.0 r2=inf" in nb_runner.get_output(2)
