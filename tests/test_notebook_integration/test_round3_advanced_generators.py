"""Batch 84 – advanced generators: send(), throw(), close(), yield from."""

import textwrap, pytest

pytestmark = [pytest.mark.stress, pytest.mark.integration]


class TestGeneratorProtocol:
    """Generator send/throw/close protocol."""

    def test_generator_send(self, nb_runner):
        """Generator with send() for two-way communication."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def accumulator():
                    total = 0
                    while True:
                        value = yield total
                        if value is None:
                            break
                        total += value

                gen = accumulator()
                next(gen)  # prime
                r1 = gen.send(10)
                r2 = gen.send(20)
                r3 = gen.send(5)
                results = [r1, r2, r3]
            """),
            "print(f'results={results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "10" in out
        assert "30" in out
        assert "35" in out

    def test_generator_yield_from(self, nb_runner):
        """yield from delegation."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def inner():
                    yield 'a'
                    yield 'b'
                    return 'inner_done'

                def outer():
                    result = yield from inner()
                    yield result
                    yield 'c'

                values = list(outer())
            """),
            "print(f'values={values}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "a" in out
        assert "b" in out
        assert "inner_done" in out
        assert "c" in out

    def test_generator_pipeline(self, nb_runner):
        """Coroutine-style generator pipeline."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def producer(n):
                    for i in range(n):
                        yield i * i

                def filterer(source, pred):
                    for item in source:
                        if pred(item):
                            yield item

                def mapper(source, fn):
                    for item in source:
                        yield fn(item)

                pipe = mapper(filterer(producer(10), lambda x: x % 2 == 0), lambda x: x + 1)
                output = list(pipe)
            """),
            "print(f'output={output}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        # 0,1,4,9,16,25,36,49,64,81 → even: 0,4,16,36,64 → +1: 1,5,17,37,65
        assert "1" in out
        assert "5" in out
        assert "17" in out

    def test_generator_propagation(self, nb_runner):
        """Generator that depends on upstream variable, change propagation."""
        nb_runner.create_notebook([
            "multiplier = 2",
            textwrap.dedent("""\
                def scaled_range(n, scale):
                    for i in range(n):
                        yield i * scale

                collected = list(scaled_range(5, multiplier))
            """),
            "print(f'collected={collected}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[0, 2, 4, 6, 8]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "multiplier = 10")
        nb_runner.run_cells([1, 2, 3])
        assert "[0, 10, 20, 30, 40]" in nb_runner.get_output(3)

    def test_infinite_generator_islice(self, nb_runner):
        """Infinite generator consumed via islice."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from itertools import islice

                def fibonacci():
                    a, b = 0, 1
                    while True:
                        yield a
                        a, b = b, a + b

                fibs = list(islice(fibonacci(), 10))
            """),
            "print(f'fibs={fibs}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]" in out
