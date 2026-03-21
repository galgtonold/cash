"""
Interaction test: collections.UserDict custom dictionary.
Tests UserDict subclass with custom __setitem__,
__getitem__ override, and cross-cell custom dict behavior.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestUserDictCustom:
    """Test UserDict custom dictionary across cells."""

    def test_userdict_ops(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: custom UserDict
            "from collections import UserDict\n\nclass TypedDict(UserDict):\n    def __init__(self, key_type, val_type, *args, **kwargs):\n        self._key_type = key_type\n        self._val_type = val_type\n        super().__init__(*args, **kwargs)\n    def __setitem__(self, key, value):\n        if not isinstance(key, self._key_type):\n            raise TypeError(f'Key must be {self._key_type}')\n        if not isinstance(value, self._val_type):\n            raise TypeError(f'Value must be {self._val_type}')\n        super().__setitem__(key, value)\n\ntd = TypedDict(str, int)\ntd['a'] = 1\ntd['b'] = 2\ntd['c'] = 3\nprint(f'data={dict(td)}')",
            # Cell 2: read from typed dict
            "total = sum(td.values())\nkeys = sorted(td.keys())\nprint(f'total={total}')\nprint(f'keys={keys}')",
            # Cell 3: verify type enforcement
            "try:\n    td[42] = 'bad'\n    print('error=none')\nexcept TypeError:\n    print('error=caught_key_type')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "'a': 1" in out1
        out2 = nb_runner.get_output(2)
        assert "total=6" in out2
        assert "keys=['a', 'b', 'c']" in out2
        out3 = nb_runner.get_output(3)
        assert "error=caught_key_type" in out3

    def test_userdict_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import UserDict\nclass CaseInsensitiveDict(UserDict):\n    def __setitem__(self, key, value):\n        super().__setitem__(key.lower(), value)\n    def __getitem__(self, key):\n        return super().__getitem__(key.lower())\n\ncid = CaseInsensitiveDict()\ncid['Hello'] = 1\ncid['WORLD'] = 2\nprint(f'keys={sorted(cid.keys())}')",
            "val = cid['hello'] + cid['World']\nprint(f'val={val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "keys=['hello', 'world']" in nb_runner.get_output(1)
        assert "val=3" in nb_runner.get_output(2)

        # Edit to add more items
        nb_runner.set_cell_source(1, "from collections import UserDict\nclass CaseInsensitiveDict(UserDict):\n    def __setitem__(self, key, value):\n        super().__setitem__(key.lower(), value)\n    def __getitem__(self, key):\n        return super().__getitem__(key.lower())\n\ncid = CaseInsensitiveDict()\ncid['Hello'] = 10\ncid['WORLD'] = 20\ncid['Python'] = 30\nprint(f'keys={sorted(cid.keys())}')")
        nb_runner.run_cells([1, 2])
        assert "val=30" in nb_runner.get_output(2)

    def test_userdict_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import UserDict\nclass DefaultDict(UserDict):\n    def __missing__(self, key):\n        return 0\n\ndd = DefaultDict({'x': 10, 'y': 20})\nresult = dd['x'] + dd['z']  # z returns 0\nprint(f'result={result}')",
            "is_ten = result == 10\nprint(f'is_ten={is_ten}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=10" in nb_runner.get_output(1)
        assert "is_ten=True" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "is_ten=True" in nb_runner.get_output(2)
