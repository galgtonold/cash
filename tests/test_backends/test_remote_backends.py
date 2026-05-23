"""Tests for remote backend implementations (Redis, S3)."""
import pytest
from unittest.mock import MagicMock, patch
import pickle

from cash.backends.redis_backend import RedisBackend
from cash.backends.s3_backend import S3Backend


@pytest.fixture
def redis_backend():
    with patch('redis.Redis'):
        backend = RedisBackend()
    backend.client = MagicMock()
    backend.prefix = 'test:'
    return backend


@pytest.fixture
def s3_backend():
    with patch('boto3.client'):
        backend = S3Backend(bucket='test-bucket')
    backend.s3 = MagicMock()
    backend.bucket = 'test-bucket'
    return backend


def test_redis_get(redis_backend):
    metadata = {'key': 'k', 'size': 10}
    meta_bytes = pickle.dumps(metadata)
    original_data = 'data'
    data_bytes = pickle.dumps(original_data)
    
    pipe = redis_backend.client.pipeline.return_value
    pipe.execute.return_value = [meta_bytes, data_bytes]
    
    m, d = redis_backend.get('k')
    # get() now injects source=REDIS on bare-backend reads (see
    # test_label_consistency.py), so check the stored fields survive and
    # source is present rather than asserting full-dict equality.
    assert m is not None
    assert m['key'] == 'k' and m['size'] == 10
    assert m['source'] == 'REDIS'
    assert d == original_data

    pipe.get.assert_any_call('test:k:meta')
    pipe.get.assert_any_call('test:k:data')


def test_redis_set(redis_backend):
    pipe = redis_backend.client.pipeline.return_value
    
    redis_backend.set('k', 'data', {'ttl': 60})
    
    assert pipe.set.call_count == 2, 'Should set both meta and data'
    assert pipe.expire.call_count == 2, 'Should set TTL on both'
    pipe.execute.assert_called_once()
    
    args, _ = pipe.set.call_args_list[1]
    assert args[0] == 'test:k:data'
    assert pickle.loads(args[1]) == 'data'


def test_s3_get(s3_backend):
    metadata = {'key': 'k', 'size': 10}
    meta_bytes = pickle.dumps(metadata)
    original_data = 'data'
    data_bytes = pickle.dumps(original_data)
    
    def get_object_side_effect(Bucket, Key):
        body = MagicMock()
        if Key.endswith('.meta'):
            body.read.return_value = meta_bytes
        elif Key.endswith('.data'):
            body.read.return_value = data_bytes
        return {'Body': body}
    
    s3_backend.s3.get_object.side_effect = get_object_side_effect
    
    m, d = s3_backend.get('k')
    # get() now injects source=S3 on bare-backend reads (see
    # test_label_consistency.py), so check the stored fields survive and
    # source is present rather than asserting full-dict equality.
    assert m is not None
    assert m['key'] == 'k' and m['size'] == 10
    assert m['source'] == 'S3'
    assert d == original_data


def test_s3_set(s3_backend):
    s3_backend.set('k', 'data', {})
    
    assert s3_backend.s3.put_object.call_count == 2
    
    calls = s3_backend.s3.put_object.call_args_list
    keys = [c[1]['Key'] for c in calls]
    assert 'cash/k.data' in keys
    assert 'cash/k.meta' in keys
