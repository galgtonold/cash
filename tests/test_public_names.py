"""Tests for the public import paths: cash, cash.ui and cash.backends."""

import importlib

import pytest


class TestOneImportPath:
    """Each public name has one import path, with no aliases in between."""

    def test_cash_ui_re_exports_nothing(self):
        import cash.ui

        for name in ("CacheExplorer", "DependencyGraph"):
            assert not hasattr(cash.ui, name)

    def test_the_experimental_namespace_is_gone(self):
        with pytest.raises(ImportError):
            importlib.import_module("cash.experimental")

    def test_the_markers_have_one_spelling(self):
        import cash

        for name in ("mark_opaque", "mark_pure", "mark_stateful", "CascadingBackend"):
            assert not hasattr(cash, name)

    def test_opaque_leaves_the_class_unmodified(self):
        """`cash.opaque` marks a class once, in the registry, and does not
        write to the class -- so it works on one that refuses new attributes,
        which is the kind of class it is called on from outside."""
        import cash

        class Frozen(type):
            def __setattr__(cls, name, value):
                raise TypeError("this class takes no new attributes")

        class Vendor(metaclass=Frozen):
            pass

        assert cash.opaque(Vendor) is Vendor
        assert cash.Cash._is_opaque(Vendor) is True
        assert "__cash_opaque__" not in vars(Vendor)

    def test_tiered_backend_is_exported_at_the_top_level(self):
        import cash
        from cash.backends.tiered_backend import TieredBackend

        assert cash.TieredBackend is TieredBackend


class TestBackendsInit:
    """Test the cash.backends __init__ exports."""

    def test_import_core_backends(self):
        """Core backends are importable from cash.backends."""
        from cash.backends import (
            CacheBackend,
            FileBackend,
            InMemoryBackend,
            TieredBackend,
        )

        assert all(
            cls is not None
            for cls in [
                CacheBackend,
                InMemoryBackend,
                FileBackend,
                TieredBackend,
            ]
        )

    def test_serializers_importable(self):
        """Serializers are importable from cash.backends."""
        from cash.backends import PickleSerializer, Serializer

        assert Serializer is not None
        assert PickleSerializer is not None

    def test_remote_backends_resolve_to_their_classes(self):
        """The lazy names are the classes themselves, installed client or not."""
        from cash.backends import RedisBackend, S3Backend
        from cash.backends.redis_backend import RedisBackend as RedisFromModule
        from cash.backends.s3_backend import S3Backend as S3FromModule

        assert RedisBackend is RedisFromModule
        assert S3Backend is S3FromModule
