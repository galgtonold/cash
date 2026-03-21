"""Batch 366: multi-cell class evolution with method additions."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestClassEvolution:
    def test_class_add_method(self, nb_runner):
        nb_runner.create_notebook([
            "class Calculator:\n    def __init__(self):\n        self.result = 0\n    def add(self, x):\n        self.result += x\n        return self",
            "c = Calculator().add(5).add(3)\nprint(f'result={c.result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=8" in nb_runner.get_output(2)
        # Add multiply method
        nb_runner.set_cell_source(1, "class Calculator:\n    def __init__(self):\n        self.result = 0\n    def add(self, x):\n        self.result += x\n        return self\n    def multiply(self, x):\n        self.result *= x\n        return self")
        nb_runner.set_cell_source(2, "c = Calculator().add(5).multiply(3)\nprint(f'result={c.result}')")
        nb_runner.run_all()
        assert "result=15" in nb_runner.get_output(2)

    def test_class_rename_attr(self, nb_runner):
        nb_runner.create_notebook([
            "class Config:\n    def __init__(self, host, port):\n        self.host = host\n        self.port = port",
            "cfg = Config('localhost', 8080)\naddr = f'{cfg.host}:{cfg.port}'\nprint(f'addr={addr}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "addr=localhost:8080" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "class Config:\n    def __init__(self, host, port):\n        self.host = host\n        self.port = port\n    def url(self):\n        return f'http://{self.host}:{self.port}'")
        nb_runner.set_cell_source(2, "cfg = Config('example.com', 443)\naddr = cfg.url()\nprint(f'addr={addr}')")
        nb_runner.run_all()
        assert "addr=http://example.com:443" in nb_runner.get_output(2)

    def test_class_default_edit(self, nb_runner):
        nb_runner.create_notebook([
            "class Greeter:\n    def __init__(self, greeting='Hello'):\n        self.greeting = greeting\n    def greet(self, name):\n        return f'{self.greeting}, {name}!'",
            "g = Greeter()\nmsg = g.greet('World')\nprint(f'msg={msg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "msg=Hello, World!" in nb_runner.get_output(2)
