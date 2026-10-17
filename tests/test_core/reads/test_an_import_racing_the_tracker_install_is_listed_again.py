"""The tracker's install lists ``sys.modules`` again when an import changes it
mid-list.

Listing allocates a tuple per module; an allocation can run the collector,
whose finalizers let another thread import. On macOS 3.10 CI that raised
"dictionary changed size during iteration" inside ``install_patches``, and
the worker's later reads went unrecorded.
"""

from __future__ import annotations

from cash.tracking import reader_patches


class _Racing(dict):
    """Changes size while its items are listed, the first *times* times."""

    def __init__(self, *args, times: int = 1):
        super().__init__(*args)
        self.left = times

    def items(self):
        if self.left:
            self.left -= 1
            raise RuntimeError("dictionary changed size during iteration")
        return super().items()


def test_a_listing_an_import_raced_is_taken_again():
    modules = _Racing({"a": 1, "b": 2}, times=3)
    assert reader_patches._modules_now(modules) == [("a", 1), ("b", 2)]


def test_a_listing_raced_every_time_is_copied_whole():
    modules = _Racing({"a": 1}, times=10**6)
    assert reader_patches._modules_now(modules) == [("a", 1)]
