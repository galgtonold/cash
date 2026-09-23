"""Slicing and tuple operations across cells."""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


# List slicing and indexing interaction tests.
#
# Tests editing list slice operations, negative indexing,
# step slicing, and their propagation.
@pytest.mark.upstream
class TestSlicingEdits:
    """Editing list slicing patterns."""

    def test_edit_slice_range(self, nb_runner):
        """Edit slice start/stop."""
        nb_runner.create_notebook(
            [
                "data = list(range(10))  # slice source",
                "result = data[2:5]\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [2, 3, 4]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "result = data[5:8]\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = [5, 6, 7]" in nb_runner.get_output(2)

    def test_edit_slice_step(self, nb_runner):
        """Edit slice step."""
        nb_runner.create_notebook(
            [
                "nums = list(range(20))  # slice step source",
                "result = nums[::2]\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "result = nums[::5]\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = [0, 5, 10, 15]" in nb_runner.get_output(2)

    def test_edit_negative_index(self, nb_runner):
        """Edit negative indexing."""
        nb_runner.create_notebook(
            [
                "items = ['a', 'b', 'c', 'd', 'e']  # neg index source",
                "result = items[-1]\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = e" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "result = items[-3:]\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = ['c', 'd', 'e']" in nb_runner.get_output(2)

    def test_edit_source_then_slice(self, nb_runner):
        """Edit the source list, verify slice updates."""
        nb_runner.create_notebook(
            [
                "seq = [10, 20, 30, 40, 50]  # slice propagation source",
                "first_half = seq[:3]\nprint(f'first_half = {first_half}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "first_half = [10, 20, 30]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "seq = [100, 200, 300, 400]  # slice propagation source v2")
        nb_runner.run_all()
        assert "first_half = [100, 200, 300]" in nb_runner.get_output(2)


class TestListSlicingAdvanced:
    """list slicing advanced patterns."""

    def test_slice_step(self, nb_runner):
        nb_runner.create_notebook(
            [
                "data = list(range(20))",
                "evens = data[::2]\nrev = data[::-1]\nchunk = data[5:15:3]\nprint(f'evens={evens[:5]} rev5={rev[:5]} chunk={chunk}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "evens=[0, 2, 4, 6, 8]" in out
        assert "rev5=[19, 18, 17, 16, 15]" in out
        assert "chunk=[5, 8, 11, 14]" in out

    def test_slice_assignment(self, nb_runner):
        nb_runner.create_notebook(
            [
                "items = [1, 2, 3, 4, 5]",
                "items[1:3] = [20, 30]\nprint(f'items={items}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "items=[1, 20, 30, 4, 5]" in nb_runner.get_output(2)

    def test_slice_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "seq = list(range(10))",
                "last3 = seq[-3:]\nfirst3 = seq[:3]\nprint(f'last3={last3} first3={first3}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "last3=[7, 8, 9]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "seq = list(range(5))")
        nb_runner.run_all()
        assert "last3=[2, 3, 4]" in nb_runner.get_output(2)
        assert "first3=[0, 1, 2]" in nb_runner.get_output(2)


class TestListSlicingStep:
    """list slicing with step and negative indices."""

    def test_step_slice(self, nb_runner):
        nb_runner.create_notebook(
            [
                "data = list(range(20))",
                "evens = data[::2]\nodds = data[1::2]\nreversed_data = data[::-1]\nprint(f'evens={evens[:5]}')\nprint(f'odds={odds[:5]}')\nprint(f'last3={reversed_data[:3]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "evens=[0, 2, 4, 6, 8]" in out
        assert "odds=[1, 3, 5, 7, 9]" in out
        assert "last3=[19, 18, 17]" in out

    def test_slice_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "items = ['a', 'b', 'c', 'd', 'e', 'f']",
                "middle = items[1:-1]\nprint(f'middle={middle}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "middle=['b', 'c', 'd', 'e']" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "items = [10, 20, 30, 40, 50]")
        nb_runner.run_all()
        assert "middle=[20, 30, 40]" in nb_runner.get_output(2)

    def test_negative_index(self, nb_runner):
        nb_runner.create_notebook(
            [
                "data = [10, 20, 30, 40, 50]",
                "last = data[-1]\nsecond_last = data[-2]\nslice_neg = data[-3:]\nprint(f'last={last} second_last={second_last} slice={slice_neg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "last=50" in out
        assert "second_last=40" in out
        assert "slice=[30, 40, 50]" in out


class TestTupleOpsImmutable:
    """tuple operations immutability and named access."""

    def test_tuple_count_index(self, nb_runner):
        nb_runner.create_notebook(
            [
                "t = (1, 2, 3, 2, 1, 2)",
                "c = t.count(2)\ni = t.index(3)\nprint(f'count={c} index={i}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=3" in nb_runner.get_output(2)
        assert "index=2" in nb_runner.get_output(2)

    def test_tuple_concat_repeat(self, nb_runner):
        nb_runner.create_notebook(
            [
                "a = (1, 2)\nb = (3, 4)",
                "combined = a + b\nrepeated = a * 3\nprint(f'combined={combined} repeated={repeated}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "combined=(1, 2, 3, 4)" in nb_runner.get_output(2)
        assert "repeated=(1, 2, 1, 2, 1, 2)" in nb_runner.get_output(2)

    def test_tuple_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "coords = (10, 20, 30)",
                "x, y, z = coords\ntotal = x + y + z\nprint(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=60" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "coords = (100, 200, 300)")
        nb_runner.run_all()
        assert "total=600" in nb_runner.get_output(2)


class TestListSlicingObject:
    """list slicing and slice object usage."""

    def test_slice_patterns(self, nb_runner):
        nb_runner.create_notebook(
            [
                "data = list(range(10))",
                "first3 = data[:3]\nlast3 = data[-3:]\nevens = data[::2]\nreversed_list = data[::-1]\nprint(f'first3={first3} last3={last3}')\nprint(f'evens={evens}')\nprint(f'rev={reversed_list}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "first3=[0, 1, 2]" in out
        assert "last3=[7, 8, 9]" in out
        assert "evens=[0, 2, 4, 6, 8]" in out
        assert "rev=[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]" in out

    def test_slice_object(self, nb_runner):
        nb_runner.create_notebook(
            [
                "data = list(range(20))",
                "s = slice(2, 10, 3)\nresult = data[s]\nprint(f'result={result} start={s.start} stop={s.stop} step={s.step}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "result=[2, 5, 8]" in out
        assert "start=2" in out
        assert "stop=10" in out


# Slice and index pattern edits.
#
# Tests list slicing, indexing operations with edits.
class TestSliceIndexEdits:
    """Slice and index edit patterns."""

    def test_slice_params_edit(self, nb_runner):
        """Edit slice parameters, result updates."""
        nb_runner.create_notebook(
            [
                "data = list(range(10, 21))",
                "start = 2\nstop = 7",
                "sliced = data[start:stop]\nprint(f'sliced = {sliced}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sliced = [12, 13, 14, 15, 16]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "start = 0\nstop = 3")
        nb_runner.run_all()
        assert "sliced = [10, 11, 12]" in nb_runner.get_output(3)

    def test_step_slice_edit(self, nb_runner):
        """Edit step in slice."""
        nb_runner.create_notebook(
            [
                "nums = list(range(20))",
                "step = 2",
                "selected = nums[::step]\nprint(f'selected = {selected}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "selected = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "step = 5")
        nb_runner.run_all()
        assert "selected = [0, 5, 10, 15]" in nb_runner.get_output(3)
