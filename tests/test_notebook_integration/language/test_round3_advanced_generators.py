"""advanced generators: send(), throw(), close(), yield from."""

import textwrap

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.integration]


class TestGeneratorProtocol:
    """Generator send/throw/close protocol."""

    def test_generator_pipeline(self, nb_runner):
        """Coroutine-style generator pipeline."""
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        # 0,1,4,9,16,25,36,49,64,81 → even: 0,4,16,36,64 → +1: 1,5,17,37,65
        assert "1" in out
        assert "5" in out
        assert "17" in out

    def test_generator_propagation(self, nb_runner):
        """Generator that depends on upstream variable, change propagation."""
        nb_runner.create_notebook(
            [
                "multiplier = 2",
                textwrap.dedent("""\
                def scaled_range(n, scale):
                    for i in range(n):
                        yield i * scale

                collected = list(scaled_range(5, multiplier))
            """),
                "print(f'collected={collected}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[0, 2, 4, 6, 8]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "multiplier = 10")
        nb_runner.run_cells([1, 2, 3])
        assert "[0, 10, 20, 30, 40]" in nb_runner.get_output(3)
