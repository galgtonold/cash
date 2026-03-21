"""Batch 521: descriptor protocol __get__ __set__."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDescriptorProtocol:
    def test_validated_descriptor(self, nb_runner):
        nb_runner.create_notebook([
            "pass  # setup",
            "class Positive:\n    def __init__(self, name): self.name = name\n    def __set_name__(self, owner, name): self.name = name\n    def __get__(self, obj, objtype=None):\n        return getattr(obj, f'_{self.name}', 0)\n    def __set__(self, obj, value):\n        if value < 0: raise ValueError\n        setattr(obj, f'_{self.name}', value)\nclass Account:\n    balance = Positive('balance')\na = Account()\na.balance = 100\nprint(f'balance={a.balance}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "balance=100" in nb_runner.get_output(2)

    def test_cached_property_desc(self, nb_runner):
        nb_runner.create_notebook([
            "pass  # setup",
            "class CachedProp:\n    def __init__(self, fn): self.fn = fn; self.name = fn.__name__\n    def __get__(self, obj, objtype=None):\n        if obj is None: return self\n        val = self.fn(obj)\n        setattr(obj, self.name, val)\n        return val\nclass Data:\n    def __init__(self, n): self.n = n\n    @CachedProp\n    def expensive(self): return sum(range(self.n))\nd = Data(100)\nprint(f'result={d.expensive}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=4950" in nb_runner.get_output(2)

    def test_descriptor_edit(self, nb_runner):
        nb_runner.create_notebook([
            "pass  # setup",
            "class TypedField:\n    def __init__(self, typ): self.typ = typ\n    def __set_name__(self, owner, name): self.name = f'_{name}'\n    def __get__(self, obj, t=None): return getattr(obj, self.name, None)\n    def __set__(self, obj, val):\n        if not isinstance(val, self.typ): raise TypeError\n        setattr(obj, self.name, val)\nclass Config:\n    port = TypedField(int)\nc = Config()\nc.port = 8080\nprint(f'port={c.port}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "port=8080" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "class TypedField:\n    def __init__(self, typ): self.typ = typ\n    def __set_name__(self, owner, name): self.name = f'_{name}'\n    def __get__(self, obj, t=None): return getattr(obj, self.name, None)\n    def __set__(self, obj, val):\n        if not isinstance(val, self.typ): raise TypeError\n        setattr(obj, self.name, val)\nclass Config:\n    port = TypedField(int)\nc = Config()\nc.port = 3000\nprint(f'port={c.port}')")
        nb_runner.run_all()
        assert "port=3000" in nb_runner.get_output(2)
