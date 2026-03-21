"""Batch 349: typing module usage patterns and class annotations."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestTypingPatterns:
    def test_typed_dict_usage(self, nb_runner):
        nb_runner.create_notebook([
            "from typing import Dict, List, Tuple\ndef aggregate(records: List[Tuple[str, int]]) -> Dict[str, int]:\n    result: Dict[str, int] = {}\n    for name, val in records:\n        result[name] = result.get(name, 0) + val\n    return result",
            "data = [('a', 10), ('b', 20), ('a', 5)]\nagg = aggregate(data)\nprint(f'agg={dict(sorted(agg.items()))}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "agg={'a': 15, 'b': 20}" in nb_runner.get_output(2)

    def test_typed_function_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from typing import Optional\ndef safe_div(a: float, b: float) -> Optional[float]:\n    if b == 0:\n        return None\n    return a / b",
            "r1 = safe_div(10, 3)\nr2 = safe_div(5, 0)\nprint(f'r1={round(r1, 2) if r1 else r1} r2={r2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=3.33 r2=None" in nb_runner.get_output(2)
        # Edit function
        nb_runner.set_cell_source(1, "from typing import Optional\ndef safe_div(a: float, b: float) -> Optional[float]:\n    if b == 0:\n        return -1.0\n    return a / b")
        nb_runner.run_all()
        assert "r1=3.33 r2=-1.0" in nb_runner.get_output(2)

    def test_generic_container(self, nb_runner):
        nb_runner.create_notebook([
            "from typing import List\ndef flatten(nested: List[List[int]]) -> List[int]:\n    return [item for sub in nested for item in sub]",
            "data = [[1, 2], [3, 4], [5]]\nresult = flatten(data)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[1, 2, 3, 4, 5]" in nb_runner.get_output(2)
