"""Unit tests for the stateful-carrier classifier (CAS-175/178)."""
import pytest

from cash.notebook.upstream.stateful_carriers import stateful_carrier_kind


class TestCarriersDetected:
    def test_numpy_generator(self):
        np = pytest.importorskip("numpy")
        assert stateful_carrier_kind(np.random.default_rng(7)) == 'numpy Generator'

    def test_numpy_random_state(self):
        np = pytest.importorskip("numpy")
        assert stateful_carrier_kind(np.random.RandomState(0)) == 'numpy RandomState'

    def test_numpy_bit_generator_matches_via_base(self):
        """PCG64 leafs in ``_pcg64`` but inherits ``BitGenerator`` -- we match bases."""
        np = pytest.importorskip("numpy")
        bg = np.random.default_rng(7).bit_generator
        assert stateful_carrier_kind(bg) == 'numpy BitGenerator'

    def test_stdlib_random(self):
        import random
        assert stateful_carrier_kind(random.Random(1)) == 'random.Random'

    def test_matplotlib_figure_and_axes(self):
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        try:
            assert stateful_carrier_kind(fig) == 'matplotlib Figure'
            assert stateful_carrier_kind(ax) == 'matplotlib Axes'
        finally:
            plt.close(fig)

    def test_user_subclass_of_a_carrier_still_matches(self):
        """We match BASE classes, so a subclass leafing in ``__main__`` is covered."""
        import random

        class MyRandom(random.Random):
            pass

        assert stateful_carrier_kind(MyRandom()) == 'random.Random'


class TestNonCarriersIgnored:
    @pytest.mark.parametrize("value", [
        None, 1, 1.5, "s", b"b", [1, 2], {"a": 1}, {1, 2}, (1, 2), range(3),
    ])
    def test_plain_values(self, value):
        assert stateful_carrier_kind(value) is None

    def test_ndarray_is_not_a_carrier(self):
        """An ndarray lives in ``numpy`` and so passes the module pre-filter --
        the MRO table must still reject it, or every array would drag its
        producer into the plan."""
        np = pytest.importorskip("numpy")
        assert stateful_carrier_kind(np.zeros(3)) is None

    def test_dataframe_is_not_a_carrier(self):
        pd = pytest.importorskip("pandas")
        assert stateful_carrier_kind(pd.DataFrame({"a": [1]})) is None

    def test_matplotlib_line2d_is_not_a_carrier(self):
        """``Line2D`` is a matplotlib Artist but carries no accumulated state of
        its own -- the CAS-144 table draws the same line."""
        pytest.importorskip("matplotlib")
        from matplotlib.lines import Line2D
        assert stateful_carrier_kind(Line2D([0, 1], [0, 1])) is None

    def test_consumables_are_left_to_their_own_channel(self):
        """Generators/queues are carriers, but ``consumables.py`` owns them: its
        producer re-execution is gated on a divergence probe that self-disables
        on run_all. Classifying them here too made this pass re-derive their
        producers unconditionally and regressed 12 integration tests."""
        import queue

        def gen():
            yield 1

        assert stateful_carrier_kind(gen()) is None
        assert stateful_carrier_kind(queue.Queue()) is None

    def test_module_is_not_a_carrier(self):
        """``np`` appears as an input of ``rng = np.random.default_rng(7)``."""
        np = pytest.importorskip("numpy")
        assert stateful_carrier_kind(np) is None

    def test_arbitrary_object_without_mro_is_safe(self):
        class Weird:
            __mro__ = property(lambda self: (_ for _ in ()).throw(AttributeError))

        assert stateful_carrier_kind(Weird()) is None
