import unittest
import socket
import threading
import time
import torch
from unittest.mock import patch, MagicMock

from src.worker.node import WorkerNode, ModelShard

class TestModelShard(unittest.TestCase):
    def setUp(self):
        self.shard_id = 1
        self.parameters = {"weight": torch.randn(3, 3)}
        self.shard = ModelShard(self.shard_id, self.parameters)
    
    def test_init(self):
        self.assertEqual(self.shard.shard_id, self.shard_id)
        self.assertEqual(self.shard.parameters.keys(), self.parameters.keys())
    
    def test_to_device(self):
        self.shard.to_device()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for param in self.shard.parameters.values():
            self.assertEqual(param.device.type, device)
    
    def test_compute(self):
        inputs = {"weight": torch.randn(1, 3)}
        result = self.shard.compute(inputs)
        self.assertIn("weight", result)
        self.assertEqual(result["weight"].shape, (1, 3))

class TestWorkerNode(unittest.TestCase):
    @patch('socket.socket')
    def setUp(self, mock_socket):
        self.worker = WorkerNode()
        self.mock_socket = mock_socket.return_value
    
    def test_init(self):
        self.assertEqual(self.worker.host, "0.0.0.0")
        self.assertEqual(self.worker.port, 65433)
        self.assertIsNone(self.worker.master_address)
        self.assertEqual(self.worker.shards, {})
        self.assertFalse(self.worker.running)
    
    @patch('threading.Thread')
    def test_start(self, mock_thread):
        self.worker.start()
        self.assertTrue(self.worker.running)
        self.mock_socket.bind.assert_called_once()
        self.mock_socket.listen.assert_called_once()
        mock_thread.assert_called()
    
    def test_stop(self):
        self.worker.running = True
        self.worker.server_socket = self.mock_socket
        self.worker.stop()
        self.assertFalse(self.worker.running)
        self.mock_socket.close.assert_called_once()
    
    @patch('src.worker.node.MessageProtocol.send_message')
    def test_connect_to_master(self, mock_send):
        self.worker.master_address = "localhost:1234"
        self.worker._connect_to_master()
        self.mock_socket.connect.assert_called_once_with(("localhost", 1234))
        mock_send.assert_called_once()
    
    def test_load_shard(self):
        shard_id = 1
        params = {"weight": torch.randn(3, 3)}
        shard_data = torch.save(params, "test_shard.pt")
        
        with patch('torch.load', return_value=params):
            result = self.worker.load_shard(shard_id, shard_data)
            
        self.assertTrue(result)
        self.assertIn(shard_id, self.worker.shards)
    
    def test_unload_shard(self):
        shard_id = 1
        self.worker.shards[shard_id] = MagicMock()
        
        result = self.worker.unload_shard(shard_id)
        
        self.assertTrue(result)
        self.assertNotIn(shard_id, self.worker.shards)
    
    def test_schedule_computation(self):
        shard_id = 1
        inputs = {"weight": torch.randn(1, 3)}
        params = {"weight": torch.randn(3, 3)}
        
        self.worker.shards[shard_id] = ModelShard(shard_id, params)
        results = self.worker.schedule_computation(inputs, [shard_id])
        
        self.assertIn("weight", results)
        self.assertEqual(results["weight"].shape, (1, 3))
        
    def test_multiple_shard_computation(self):
        shard1 = 1
        shard2 = 2
        inputs = {"weight1": torch.randn(1, 3), "weight2": torch.randn(1, 3)}
        params1 = {"weight1": torch.randn(3, 3)}
        params2 = {"weight2": torch.randn(3, 3)}
        
        self.worker.shards[shard1] = ModelShard(shard1, params1)
        self.worker.shards[shard2] = ModelShard(shard2, params2)
        results = self.worker.schedule_computation(inputs, [shard1, shard2])
        
        self.assertIn("weight1", results)
        self.assertIn("weight2", results)
        self.assertEqual(results["weight1"].shape, (1, 3))
        self.assertEqual(results["weight2"].shape, (1, 3))
        
    def test_shard_unloading_during_computation(self):
        shard_id = 1
        inputs = {"weight": torch.randn(1, 3)}
        params = {"weight": torch.randn(3, 3)}
        
        self.worker.shards[shard_id] = ModelShard(shard_id, params)
        self.worker.unload_shard(shard_id)
        
        with self.assertRaises(KeyError):
            self.worker.schedule_computation(inputs, [shard_id])

if __name__ == '__main__':
    unittest.main()