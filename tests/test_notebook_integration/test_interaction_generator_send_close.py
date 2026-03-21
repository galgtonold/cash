"""Batch 460: generator send and close protocol."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestGeneratorSendClose:
    def test_generator_send(self, nb_runner):
        nb_runner.create_notebook([
            "def accumulator():\n    total = 0\n    while True:\n        val = yield total\n        if val is None: break\n        total += val",
            "gen = accumulator()\nnext(gen)\ngen.send(10)\ngen.send(20)\nresult = gen.send(30)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=60" in nb_runner.get_output(2)

    def test_yield_from(self, nb_runner):
        nb_runner.create_notebook([
            "def inner():\n    yield 1\n    yield 2\ndef outer():\n    yield 0\n    yield from inner()\n    yield 3",
            "result = list(outer())\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[0, 1, 2, 3]" in nb_runner.get_output(2)

    def test_generator_edit(self, nb_runner):
        nb_runner.create_notebook([
            "def counter(start, step):\n    val = start\n    while True:\n        yield val\n        val += step",
            "gen = counter(0, 5)\nfirst5 = [next(gen) for _ in range(5)]\nprint(f'first5={first5}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "first5=[0, 5, 10, 15, 20]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "gen = counter(100, 10)\nfirst5 = [next(gen) for _ in range(5)]\nprint(f'first5={first5}')")
        nb_runner.run_all()
        assert "first5=[100, 110, 120, 130, 140]" in nb_runner.get_output(2)
