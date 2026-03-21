"""Tests for CascadingBackend functionality."""
import pytest
from cash.backends.backend import InMemoryBackend, CascadingBackend


@pytest.fixture
def cascading_backend():
    l1 = InMemoryBackend()
    l2 = InMemoryBackend()
    backend = CascadingBackend([l1, l2])
    yield backend, l1, l2
    l1.clear()
    l2.clear()


def test_cascading_precedence(cascading_backend):
    backend, l1, l2 = cascading_backend
    
    l1.set("key", "value1")
    l2.set("key", "value2")
    
    _, value = backend.get("key")
    assert value == "value1", "Should get value from first backend (L1)"
    
    l1.delete("key")
    _, value = backend.get("key")
    assert value == "value2", "Should get value from second backend (L2) after L1 delete"


def test_cascading_read_repair(cascading_backend):
    backend, l1, l2 = cascading_backend
    
    l2.set("key", "value")
    
    _, value = l1.get("key")
    assert value is None, "L1 should not have value initially"
    
    _, value = backend.get("key")
    assert value == "value", "Should get value from L2"
    
    _, value = l1.get("key")
    assert value == "value", "L1 should now have value (read repair)"


def test_cascading_set(cascading_backend):
    backend, l1, l2 = cascading_backend
    
    backend.set("key", "value")
    
    _, v1 = l1.get("key")
    _, v2 = l2.get("key")
    assert v1 == "value", "L1 should have value"
    assert v2 == "value", "L2 should have value"


def test_cascading_delete(cascading_backend):
    backend, l1, l2 = cascading_backend
    
    l1.set("key", "value")
    l2.set("key", "value")
    
    backend.delete("key")
    
    _, v1 = l1.get("key")
    _, v2 = l2.get("key")
    assert v1 is None, "L1 should not have value after delete"
    assert v2 is None, "L2 should not have value after delete"
