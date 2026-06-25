import unittest
from unittest.mock import MagicMock, patch
import pickle

import pytest

from cash.backends.redis_backend import RedisBackend
from cash.backends.s3_backend import S3Backend

class TestRedisBackend(unittest.TestCase):
    def setUp(self):
        # patch('redis.Redis') imports redis; skip cleanly if it's not installed.
        pytest.importorskip("redis")
        with patch('redis.Redis'):
            self.backend = RedisBackend()
        self.backend.client = MagicMock()
        self.backend.prefix = "test:"

    def test_get(self):
        # Setup mock return
        metadata = {'key': 'k', 'size': 10}
        meta_bytes = pickle.dumps(metadata)
        # Data must be valid pickle because get() will try to deserialize it
        original_data = "data"
        data_bytes = pickle.dumps(original_data)
        
        # Pipeline mock
        pipe = self.backend.client.pipeline.return_value
        pipe.execute.return_value = [meta_bytes, data_bytes]
        
        m, d = self.backend.get("k")
        # get() now injects source=REDIS on bare-backend reads, so check
        # the stored fields survive instead of full-dict equality.
        self.assertIsNotNone(m)
        self.assertEqual(m['key'], 'k')
        self.assertEqual(m['size'], 10)
        self.assertEqual(m['source'], 'REDIS')
        self.assertEqual(d, original_data)

        # Verify calls
        pipe.get.assert_any_call("test:k:meta")
        pipe.get.assert_any_call("test:k:data")

    def test_set(self):
        pipe = self.backend.client.pipeline.return_value
        
        self.backend.set("k", "data", {'ttl': 60})
        
        # Verify calls
        # We can't easily check args for pickle dumps, but we check calls happened
        self.assertEqual(pipe.set.call_count, 2) # meta and data
        self.assertEqual(pipe.expire.call_count, 2) # ttl
        pipe.execute.assert_called_once()
        
        # Check that data was serialized
        # The second call to set should be data
        # pipe.set(data_key, serialized_value)
        args, _ = pipe.set.call_args_list[1]
        self.assertEqual(args[0], "test:k:data")
        # args[1] should be pickled "data"
        self.assertEqual(pickle.loads(args[1]), "data")

class TestS3Backend(unittest.TestCase):
    def setUp(self):
        # patch('boto3.client') imports boto3; skip cleanly if it's not installed.
        pytest.importorskip("boto3")
        with patch('boto3.client'):
            self.backend = S3Backend(bucket="test-bucket")
        self.backend.s3 = MagicMock()
        self.backend.bucket = "test-bucket"

    def test_get(self):
        metadata = {'key': 'k', 'size': 10}
        meta_bytes = pickle.dumps(metadata)
        original_data = "data"
        data_bytes = pickle.dumps(original_data)
        
        # Mock get_object response
        def get_object_side_effect(Bucket, Key):
            if Key.endswith('.meta'):
                body = MagicMock()
                body.read.return_value = meta_bytes
                return {'Body': body}
            elif Key.endswith('.data'):
                body = MagicMock()
                body.read.return_value = data_bytes
                return {'Body': body}
            return None
            
        self.backend.s3.get_object.side_effect = get_object_side_effect
        
        m, d = self.backend.get("k")
        # get() now injects source=S3 on bare-backend reads.
        self.assertIsNotNone(m)
        self.assertEqual(m['key'], 'k')
        self.assertEqual(m['size'], 10)
        self.assertEqual(m['source'], 'S3')
        self.assertEqual(d, original_data)

    def test_set(self):
        self.backend.set("k", "data", {})
        
        self.assertEqual(self.backend.s3.put_object.call_count, 2)
        # Check calls
        calls = self.backend.s3.put_object.call_args_list
        # One for data, one for meta
        keys = [c[1]['Key'] for c in calls]
        self.assertIn("cash/k.data", keys)
        self.assertIn("cash/k.meta", keys)
        
        # Verify data was serialized
        # Find the call for data
        data_call = [c for c in calls if c[1]['Key'] == "cash/k.data"][0]
        self.assertEqual(pickle.loads(data_call[1]['Body']), "data")

if __name__ == '__main__':
    unittest.main()
