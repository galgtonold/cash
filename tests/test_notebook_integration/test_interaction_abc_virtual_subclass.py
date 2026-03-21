"""
Interaction test: ABC with virtual subclass registration.
Tests abc.ABC with register() for virtual subclasses,
__subclasshook__, and cross-cell polymorphism patterns.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestAbcVirtualSubclass:
    """Test ABC virtual subclass registration across cells."""

    def test_abc_virtual(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: define ABC and register virtual subclass
            "from abc import ABC, abstractmethod\n\nclass Serializable(ABC):\n    @abstractmethod\n    def serialize(self):\n        pass\n\nclass JsonData:\n    def __init__(self, data):\n        self.data = data\n    def serialize(self):\n        import json\n        return json.dumps(self.data)\n\nSerializable.register(JsonData)\nj = JsonData({'key': 'value'})\nprint(f'is_serializable={isinstance(j, Serializable)}')\nprint(f'serialized={j.serialize()}')",
            # Cell 2: another registered class
            "class CsvData:\n    def __init__(self, rows):\n        self.rows = rows\n    def serialize(self):\n        return '\\n'.join(','.join(str(c) for c in row) for row in self.rows)\n\nSerializable.register(CsvData)\ncsv = CsvData([[1, 2], [3, 4]])\nprint(f'csv_is_ser={isinstance(csv, Serializable)}')\nprint(f'csv_out={csv.serialize()}')",
            # Cell 3: polymorphic use
            "items = [j, csv]\nfor item in items:\n    print(f'type={type(item).__name__} check={isinstance(item, Serializable)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "is_serializable=True" in out1
        out2 = nb_runner.get_output(2)
        assert "csv_is_ser=True" in out2
        out3 = nb_runner.get_output(3)
        assert "type=JsonData check=True" in out3
        assert "type=CsvData check=True" in out3

    def test_abc_virtual_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from abc import ABC, abstractmethod\nclass Printable(ABC):\n    @abstractmethod\n    def display(self):\n        pass\n\nclass Report:\n    def __init__(self, title):\n        self.title = title\n    def display(self):\n        return f'Report: {self.title}'\n\nPrintable.register(Report)\nrep = Report('Q1')\nprint(f'display={rep.display()}')",
            "output = rep.display()\nprint(f'output={output}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "output=Report: Q1" in nb_runner.get_output(2)

        # Edit title
        nb_runner.set_cell_source(1, "from abc import ABC, abstractmethod\nclass Printable(ABC):\n    @abstractmethod\n    def display(self):\n        pass\n\nclass Report:\n    def __init__(self, title):\n        self.title = title\n    def display(self):\n        return f'Report: {self.title}'\n\nPrintable.register(Report)\nrep = Report('Q2 Summary')\nprint(f'display={rep.display()}')")
        nb_runner.run_cells([1, 2])
        assert "output=Report: Q2 Summary" in nb_runner.get_output(2)

    def test_abc_virtual_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from abc import ABC\nclass Container(ABC):\n    pass\n\nContainer.register(list)\nContainer.register(tuple)\nresults = [isinstance([], Container), isinstance((), Container), isinstance({}, Container)]\nprint(f'results={results}')",
            "all_true = all(results[:2])\nprint(f'lists_tuples_are_containers={all_true}')\nprint(f'dict_is_container={results[2]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[True, True, False]" in nb_runner.get_output(1)
        assert "lists_tuples_are_containers=True" in nb_runner.get_output(2)
        assert "dict_is_container=False" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "lists_tuples_are_containers=True" in nb_runner.get_output(2)
