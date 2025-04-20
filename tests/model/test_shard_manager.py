import unittest
import os
import torch
import json
from unittest.mock import patch, MagicMock

from src.model.shard_manager import ModelShardManager

class TestModelShardManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = "./test_model"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create dummy model files
        torch.save({"weight1": torch.randn(3,3), "weight2": torch.randn(3,3)}, 
                   os.path.join(self.test_dir, "pytorch_model.bin"))
        with open(os.path.join(self.test_dir, "config.json"), "w") as f:
            json.dump({"model_type": "test"}, f)
        
        self.shard_manager = ModelShardManager(self.test_dir, num_shards=2)
    
    def tearDown(self):
        # Clean up test files
        for root, dirs, files in os.walk(self.test_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.test_dir)
    
    def test_shard_model(self):
        # Test model sharding
        shard_dir = self.shard_manager.shard_model()
        
        # Verify shard directory was created
        self.assertTrue(os.path.exists(shard_dir))
        
        # Verify shard files were created
        shard_files = [f for f in os.listdir(shard_dir) if f.startswith("shard_")]
        self.assertEqual(len(shard_files), 2)
        
        # Verify shard info file was created
        self.assertTrue(os.path.exists(os.path.join(shard_dir, "shard_info.json")))
    
    def test_get_shard_paths(self):
        # Test getting shard paths
        self.shard_manager.shard_model()
        paths = self.shard_manager.get_shard_paths()
        
        self.assertEqual(len(paths), 2)
        self.assertTrue(all(p.startswith(self.test_dir) for p in paths))
    
    def test_reconstruct_model(self):
        # Test model reconstruction
        self.shard_manager.shard_model()
        shard_paths = self.shard_manager.get_shard_paths()
        config_path = os.path.join(self.test_dir, "shards", "config.json")
        
        state_dict, config = ModelShardManager.reconstruct_model(shard_paths, config_path)
        
        self.assertIn("weight1", state_dict)
        self.assertIn("weight2", state_dict)
        self.assertEqual(config["model_type"], "test")

if __name__ == '__main__':
    unittest.main()