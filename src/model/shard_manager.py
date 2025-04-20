# src/model/shard_manager.py
import os
import json
import torch
from typing import Dict, List, Any

class ModelShardManager:
    def __init__(self, model_path: str, num_shards: int):
        self.model_path = model_path
        self.num_shards = num_shards
        self.shard_dir = os.path.join(model_path, "shards")
        self.shard_info = {}  # Maps shard_id -> parameter keys
        
    def shard_model(self):
        """Shard the model into multiple parts"""
        # Load model state dict
        if os.path.exists(os.path.join(self.model_path, "pytorch_model.bin")):
            state_dict_path = os.path.join(self.model_path, "pytorch_model.bin")
        else:
            # Handle sharded HF models
            state_dict_path = os.path.join(self.model_path, "model.safetensors")
        
        # Get model state dict
        state_dict = torch.load(state_dict_path, map_location="cpu")
        
        # Get model config for rebuilding
        with open(os.path.join(self.model_path, "config.json"), "r") as f:
            config = json.load(f)
        
        # Create shards directory
        os.makedirs(self.shard_dir, exist_ok=True)
        
        # Group parameters by layer for layer-wise sharding
        layer_params = {}
        for key in state_dict.keys():
            # Extract layer number from parameter name (e.g., 'layer.0.weight')
            if '.' in key:
                layer_num = key.split('.')[1]
                if layer_num.isdigit():
                    if layer_num not in layer_params:
                        layer_params[layer_num] = {}
                    layer_params[layer_num][key] = state_dict[key]
        
        # Distribute layers across shards
        shards = [{} for _ in range(self.num_shards)]
        shard_sizes = [0] * self.num_shards
        
        # Assign complete layers to shards for better locality
        for layer_num, params in layer_params.items():
            # Find shard with minimum current size
            target_shard = shard_sizes.index(min(shard_sizes))
            
            # Add all parameters from this layer to the shard
            for key, tensor in params.items():
                shards[target_shard][key] = tensor
                shard_sizes[target_shard] += tensor.nelement() * tensor.element_size()
                
                # Track which parameters are in which shard
                if target_shard not in self.shard_info:
                    self.shard_info[target_shard] = []
                self.shard_info[target_shard].append(key)
        
        # Save shards
        for i, shard_dict in enumerate(shards):
            shard_path = os.path.join(self.shard_dir, f"shard_{i}.pt")
            torch.save(shard_dict, shard_path)
            print(f"Saved shard {i} with {len(shard_dict)} parameters")
        
        # Save shard info and config
        with open(os.path.join(self.shard_dir, "shard_info.json"), "w") as f:
            json.dump(self.shard_info, f)
        
        with open(os.path.join(self.shard_dir, "config.json"), "w") as f:
            json.dump(config, f)
            
        return self.shard_dir
    
    def get_shard_paths(self):
        """Get paths to all shards"""
        return [os.path.join(self.shard_dir, f"shard_{i}.pt") for i in range(self.num_shards)]
    
    @staticmethod
    def reconstruct_model(shard_paths, config_path):
        """Reconstruct a full model from shards"""
        with open(config_path, "r") as f:
            config = json.load(f)
            
        full_state_dict = {}
        
        for path in shard_paths:
            shard_dict = torch.load(path, map_location="cpu")
            full_state_dict.update(shard_dict)
            
        return full_state_dict, config