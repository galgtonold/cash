"""
Descriptor, property, slots, dataclass, and protocol patterns.
"""

import textwrap

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress]


class TestDescriptorPatterns:
    """Test caching with Python descriptors."""

    def test_property_change_class(self, nb_runner):
        """Change property logic → downstream updates."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class Rect:
                    def __init__(self, w, h):
                        self.w = w
                        self.h = h
                    
                    @property
                    def area(self):
                        return self.w * self.h
            """),
                "r = Rect(4, 5)",
                "print(r.area)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "20" in nb_runner.get_output(3)

        # Change property to include perimeter
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            class Rect:
                def __init__(self, w, h):
                    self.w = w
                    self.h = h
                
                @property
                def area(self):
                    return self.w * self.h
                
                @property
                def perimeter(self):
                    return 2 * (self.w + self.h)
        """),
        )
        nb_runner.set_cell_source(3, "print(f'{r.area} {r.perimeter}')")
        nb_runner.run_all()
        assert "20 18" in nb_runner.get_output(3)


class TestContextManagerPatterns:
    """Test caching with context managers."""

    def test_custom_context_manager_class(self, nb_runner):
        """Custom __enter__/__exit__ context manager."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class Timer:
                    def __enter__(self):
                        import time
                        self.start = time.time()
                        return self
                    def __exit__(self, *args):
                        import time
                        self.elapsed = time.time() - self.start
            """),
                textwrap.dedent("""\
                import time
                with Timer() as t:
                    time.sleep(0.01)
                print(f"elapsed={t.elapsed > 0}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "elapsed=True" in nb_runner.get_output(2)
