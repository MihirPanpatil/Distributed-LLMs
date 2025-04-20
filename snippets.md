Here are several key code snippets for implementing the distributed LLM inference system. These cover the most critical components of the system:

## 1. Model Sharding Implementation

```python
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
        
        # Categorize parameters by size for better balancing
        keys = list(state_dict.keys())
        param_sizes = {k: state_dict[k].nelement() * state_dict[k].element_size() for k in keys}
        
        # Sort parameters by size descending
        sorted_params = sorted(param_sizes.items(), key=lambda x: x[1], reverse=True)
        sorted_keys = [k for k, _ in sorted_params]
        
        # Initialize shards with empty dicts
        shards = [{} for _ in range(self.num_shards)]
        shard_sizes = [0] * self.num_shards
        
        # Greedy assignment: place largest parameters first
        for key in sorted_keys:
            # Find shard with minimum current size
            target_shard = shard_sizes.index(min(shard_sizes))
            shards[target_shard][key] = state_dict[key]
            shard_sizes[target_shard] += param_sizes[key]
            
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
```

## 2. Network Communication Protocol

```python
# src/network/protocol.py
import socket
import pickle
import struct
from typing import Dict, Any, Optional, Tuple

HEADER_SIZE = 10  # Size of the header length field

class MessageProtocol:
    @staticmethod
    def send_message(sock: socket.socket, command: str, payload: Optional[bytes] = None, 
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send a message with the specified command and optional payload"""
        try:
            # Prepare header
            header = {
                "command": command,
            }
            
            if metadata:
                header.update(metadata)
                
            if payload is not None:
                header["payload_size"] = len(payload)
            
            # Serialize header
            header_bytes = pickle.dumps(header)
            header_len = len(header_bytes)
            
            # Send header length + header
            sock.sendall(f"{header_len:<{HEADER_SIZE}}".encode() + header_bytes)
            
            # Send payload if exists
            if payload is not None:
                sock.sendall(payload)
                
            return True
            
        except Exception as e:
            print(f"Error sending message: {e}")
            return False
    
    @staticmethod
    def receive_message(sock: socket.socket, timeout: int = 60) -> Tuple[Dict[str, Any], Optional[bytes]]:
        """Receive a message and return header and payload"""
        # Set timeout
        sock.settimeout(timeout)
        
        try:
            # Receive header length
            header_len_bytes = sock.recv(HEADER_SIZE)
            if not header_len_bytes:
                return {}, None
                
            header_len = int(header_len_bytes.decode().strip())
            
            # Receive header
            header_bytes = sock.recv(header_len)
            header = pickle.loads(header_bytes)
            
            # Receive payload if exists
            payload = None
            if "payload_size" in header:
                payload_size = header["payload_size"]
                payload = b""
                
                # Receive in chunks to handle large payloads
                bytes_received = 0
                while bytes_received < payload_size:
                    chunk_size = min(4096, payload_size - bytes_received)
                    chunk = sock.recv(chunk_size)
                    if not chunk:
                        raise ConnectionError("Connection closed while receiving payload")
                    payload += chunk
                    bytes_received += len(chunk)
            
            return header, payload
            
        except socket.timeout:
            raise TimeoutError("Timeout while receiving message")
        except Exception as e:
            print(f"Error receiving message: {e}")
            raise
```

## 3. Master Node Core Implementation

```python
# src/master/node.py
import socket
import threading
import time
import json
from typing import Dict, List, Any, Optional
from queue import Queue

from ..network.protocol import MessageProtocol
from ..model.shard_manager import ModelShardManager

class MasterNode:
    def __init__(self, host: str = "0.0.0.0", port: int = 65432):
        self.host = host
        self.port = port
        self.workers = {}  # worker_id -> WorkerInfo
        self.model_path = None
        self.tokenizer = None
        self.shard_manager = None
        self.shard_assignments = {}  # worker_id -> [shard_ids]
        self.running = False
        self.server_socket = None
        self.task_queue = Queue()
        self.result_queue = Queue()
        
    def start(self):
        """Start the master server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        
        self.running = True
        print(f"Master node listening on {self.host}:{self.port}")
        
        # Thread for accepting connections
        self.accept_thread = threading.Thread(target=self._accept_connections)
        self.accept_thread.daemon = True
        self.accept_thread.start()
        
        # Thread for processing tasks
        self.task_thread = threading.Thread(target=self._process_tasks)
        self.task_thread.daemon = True
        self.task_thread.start()
        
    def stop(self):
        """Stop the master server"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
            
    def initialize_model(self, model_path: str, num_shards: int):
        """Initialize and shard a model"""
        self.model_path = model_path
        self.shard_manager = ModelShardManager(model_path, num_shards)
        return self.shard_manager.shard_model()
    
    def assign_shards(self):
        """Assign shards to connected workers"""
        if not self.workers:
            raise ValueError("No workers connected")
            
        # Get worker capacities (simplified: equally capable)
        workers = list(self.workers.keys())
        num_shards = self.shard_manager.num_shards
        
        # Round-robin assignment
        self.shard_assignments = {}
        for i in range(num_shards):
            worker_idx = i % len(workers)
            worker_id = workers[worker_idx]
            
            if worker_id not in self.shard_assignments:
                self.shard_assignments[worker_id] = []
                
            self.shard_assignments[worker_id].append(i)
        
        return self.shard_assignments
    
    def distribute_shards(self):
        """Send shards to workers"""
        shard_paths = self.shard_manager.get_shard_paths()
        
        for worker_id, shard_ids in self.shard_assignments.items():
            worker = self.workers[worker_id]
            
            for shard_id in shard_ids:
                shard_path = shard_paths[shard_id]
                self._send_shard(worker, shard_id, shard_path)
                
    def run_inference(self, input_text: str, timeout: int = 60):
        """Run inference with the distributed model"""
        # Queue the task
        task_id = f"task_{int(time.time() * 1000)}"
        self.task_queue.put((task_id, input_text))
        
        # Wait for result
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check for result
            try:
                result_id, result = self.result_queue.get(block=False)
                if result_id == task_id:
                    return result
                else:
                    # Put it back if it's not our result
                    self.result_queue.put((result_id, result))
            except:
                # Queue empty, wait a bit
                time.sleep(0.1)
                
        raise TimeoutError(f"Inference timed out after {timeout} seconds")
    
    def _accept_connections(self):
        """Thread for accepting worker connections"""
        while self.running:
            try:
                client_socket, address = self.server_socket.accept()
                client_thread = threading.Thread(
                    target=self._handle_worker,
                    args=(client_socket, address)
                )
                client_thread.daemon = True
                client_thread.start()
            except Exception as e:
                if self.running:
                    print(f"Error accepting connection: {e}")
    
    def _handle_worker(self, client_socket, address):
        """Handle messages from a worker"""
        worker_id = f"{address[0]}:{address[1]}"
        print(f"New connection from {worker_id}")
        
        # Create worker info
        self.workers[worker_id] = {
            "socket": client_socket,
            "address": address,
            "connected_time": time.time(),
            "last_heartbeat": time.time(),
            "status": "connected"
        }
        
        while self.running:
            try:
                header, payload = MessageProtocol.receive_message(client_socket)
                if not header:
                    print(f"Worker {worker_id} disconnected")
                    break
                    
                command = header.get("command", "")
                self._handle_worker_command(worker_id, command, header, payload)
                
            except Exception as e:
                print(f"Error handling worker {worker_id}: {e}")
                break
                
        # Clean up
        if worker_id in self.workers:
            del self.workers[worker_id]
        client_socket.close()
    
    def _handle_worker_command(self, worker_id, command, header, payload):
        """Process commands from workers"""
        if command == "REGISTER":
            # Update worker info with capabilities
            capabilities = header.get("capabilities", {})
            self.workers[worker_id].update(capabilities)
            print(f"Worker {worker_id} registered with capabilities: {capabilities}")
            
        elif command == "HEARTBEAT":
            # Update last heartbeat time
            self.workers[worker_id]["last_heartbeat"] = time.time()
            
        elif command == "SHARD_LOADED":
            # Worker confirmed shard loading
            shard_id = header.get("shard_id")
            print(f"Worker {worker_id} loaded shard {shard_id}")
            
        elif command == "RESULT":
            # Process inference result
            task_id = header.get("task_id")
            self.result_queue.put((task_id, payload))
    
    def _send_shard(self, worker, shard_id, shard_path):
        """Send a model shard to a worker"""
        print(f"Sending shard {shard_id} to worker")
        
        with open(shard_path, "rb") as f:
            shard_data = f.read()
            
        MessageProtocol.send_message(
            worker["socket"], 
            "LOAD_SHARD", 
            payload=shard_data,
            metadata={"shard_id": shard_id}
        )
    
    def _process_tasks(self):
        """Thread for processing inference tasks"""
        while self.running:
            try:
                # Get a task
                task_id, input_text = self.task_queue.get(block=True, timeout=1)
                
                # TODO: Tokenize input
                input_data = input_text.encode('utf-8')
                
                # Distribute to all workers
                for worker_id, worker in self.workers.items():
                    shard_ids = self.shard_assignments.get(worker_id, [])
                    if not shard_ids:
                        continue
                        
                    MessageProtocol.send_message(
                        worker["socket"],
                        "RUN_INFERENCE",
                        payload=input_data,
                        metadata={
                            "task_id": task_id,
                            "shard_ids": shard_ids
                        }
                    )
                
            except TimeoutError:
                # No tasks in queue
                pass
            except Exception as e:
                print(f"Error processing tasks: {e}")
```

## 4. Worker Node Implementation

```python
# src/worker/node.py
import socket
import threading
import time
import torch
import pickle
import os
from typing import Dict, List, Any, Optional

from ..network.protocol import MessageProtocol

class ModelShard:
    def __init__(self, shard_id: int, parameters: Dict[str, torch.Tensor]):
        self.shard_id = shard_id
        self.parameters = parameters
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def to_device(self):
        """Move parameters to appropriate device"""
        for key in self.parameters:
            self.parameters[key] = self.parameters[key].to(self.device)
    
    def compute(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute partial activations for this shard"""
        # This is a simplified placeholder
        # Real implementation would perform actual computation
        results = {}
        for key in self.parameters:
            if key in inputs:
                results[key] = torch.matmul(inputs[key], self.parameters[key])
        return results

class WorkerNode:
    def __init__(self, host: str = "0.0.0.0", port: int = 65433, master_address: Optional[str] = None):
        self.host = host
        self.port = port
        self.master_address = master_address
        self.shards = {}  # shard_id -> ModelShard
        self.running = False
        self.server_socket = None
        self.master_socket = None
        
    def start(self):
        """Start the worker"""
        # Start server for master to connect
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        
        self.running = True
        print(f"Worker listening on {self.host}:{self.port}")
        
        # Connect to master if specified
        if self.master_address:
            self._connect_to_master()
        
        # Thread for accepting connections
        self.accept_thread = threading.Thread(target=self._accept_connections)
        self.accept_thread.daemon = True
        self.accept_thread.start()
        
        # Heartbeat thread
        self.heartbeat_thread = threading.Thread(target=self._send_heartbeat)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()
        
        # Keep main thread alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop the worker"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        if self.master_socket:
            self.master_socket.close()
    
    def _connect_to_master(self):
        """Connect to the master node"""
        try:
            host, port = self.master_address.split(':')
            port = int(port)
            
            self.master_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.master_socket.connect((host, port))
            
            # Register with master
            capabilities = {
                "has_gpu": torch.cuda.is_available(),
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "memory": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
            }
            
            MessageProtocol.send_message(
                self.master_socket,
                "REGISTER",
                metadata={"capabilities": capabilities}
            )
            
            # Start thread to handle master messages
            master_thread = threading.Thread(target=self._handle_master)
            master_thread.daemon = True
            master_thread.start()
            
        except Exception as e:
            print(f"Error connecting to master: {e}")
    
    def _accept_connections(self):
        """Accept incoming connections"""
        while self.running:
            try:
                client_socket, address = self.server_socket.accept()
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address)
                )
                client_thread.daemon = True
                client_thread.start()
            except Exception as e:
                if self.running:
                    print(f"Error accepting connection: {e}")
    
    def _handle_client(self, client_socket, address):
        """Handle incoming client connection"""
        print(f"Handling connection from {address}")
        
        try:
            while self.running:
                header, payload = MessageProtocol.receive_message(client_socket)
                if not header:
                    break
                    
                command = header.get("command", "")
                self._handle_command(client_socket, command, header, payload)
                
        except Exception as e:
            print(f"Error handling client {address}: {e}")
        finally:
            client_socket.close()
    
    def _handle_master(self):
        """Handle messages from master"""
        try:
            while self.running:
                header, payload = MessageProtocol.receive_message(self.master_socket)
                if not header:
                    print("Master disconnected")
                    break
                    
                command = header.get("command", "")
                self._handle_command(self.master_socket, command, header, payload)
                
        except Exception as e:
            print(f"Error handling master messages: {e}")
        finally:
            self.master_socket.close()
            self.master_socket = None
    
    def _handle_command(self, sock, command, header, payload):
        """Process commands from master or clients"""
        if command == "LOAD_SHARD":
            shard_id = header.get("shard_id")
            self._load_shard(sock, shard_id, payload)
            
        elif command == "RUN_INFERENCE":
            task_id = header.get("task_id")
            shard_ids = header.get("shard_ids", [])
            self._run_inference(sock, task_id, shard_ids, payload)
    
    def _load_shard(self, sock, shard_id, shard_data):
        """Load a model shard"""
        try:
            print(f"Loading shard {shard_id}")
            
            # Deserialize the shard
            shard_dict = pickle.loads(shard_data)
            
            # Create and store shard
            self.shards[shard_id] = ModelShard(shard_id, shard_dict)
            
            # Move to appropriate device
            self.shards[shard_id].to_device()
            
            # Acknowledge loading
            MessageProtocol.send_message(
                sock,
                "SHARD_LOADED",
                metadata={"shard_id": shard_id}
            )
            
        except Exception as e:
            print(f"Error loading shard {shard_id}: {e}")
            MessageProtocol.send_message(
                sock,
                "ERROR",
                metadata={"shard_id": shard_id, "error": str(e)}
            )
    
    def _run_inference(self, sock, task_id, shard_ids, input_data):
        """Run inference on specified shards"""
        try:
            print(f"Running inference for task {task_id} on shards {shard_ids}")
            
            # Decode input data
            input_text = input_data.decode('utf-8')
            
            # Create dummy input tensor for demonstration
            # In a real implementation, this would be properly tokenized
            inputs = {"input": torch.ones((1, 768)).to(self.shards[shard_ids[0]].device)}
            
            # Process with each shard
            results = {}
            for shard_id in shard_ids:
                if shard_id in self.shards:
                    shard_results = self.shards[shard_id].compute(inputs)
                    results[shard_id] = shard_results
            
            # Serialize results
            result_data = pickle.dumps(results)
            
            # Send results back
            MessageProtocol.send_message(
                sock,
                "RESULT",
                payload=result_data,
                metadata={"task_id": task_id}
            )
            
        except Exception as e:
            print(f"Error running inference: {e}")
            MessageProtocol.send_message(
                sock,
                "ERROR",
                metadata={"task_id": task_id, "error": str(e)}
            )
    
    def _send_heartbeat(self):
        """Send periodic heartbeats to master"""
        while self.running:
            if self.master_socket:
                try:
                    MessageProtocol.send_message(
                        self.master_socket,
                        "HEARTBEAT"
                    )
                except:
                    pass
            time.sleep(10)  # Send heartbeat every 10 seconds
```

## 5. Quantization Implementation

```python
# src/model/quantization.py
import torch
import os
import sys
from typing import Dict, Any, Optional

def quantize_model(model_or_state_dict, bits=8, device="cpu"):
    """Quantize a model or state dict to lower precision"""
    if isinstance(model_or_state_dict, dict):
        # State dict quantization
        return _quantize_state_dict(model_or_state_dict, bits, device)
    else:
        # Full model quantization
        return _quantize_model(model_or_state_dict, bits, device)

def _quantize_state_dict(state_dict: Dict[str, torch.Tensor], bits: int, device: str) -> Dict[str, torch.Tensor]:
    """Quantize a state dict"""
    quantized_dict = {}
    
    try:
        # Try to use bitsandbytes for advanced quantization
        import bitsandbytes as bnb
        has_bnb = True
    except ImportError:
        has_bnb = False
    
    # Process each parameter
    for key, param in state_dict.items():
        param = param.to(device)
        
        # Skip non-floating point tensors
        if not param.is_floating_point():
            quantized_dict[key] = param
            continue
            
        if bits == 8:
            if has_bnb:
                # Use bitsandbytes for INT8 quantization
                quantized_dict[key] = bnb.nn.Int8Params(
                    param.contiguous(), 
                    requires_grad=False
                ).to(device)
            else:
                # Fallback to simple quantization
                scale = param.abs().max() / 127.0
                quantized_dict[key] = (param / scale).round().clamp(-127, 127).to(torch.int8)
                # Store scale as an attribute
                quantized_dict[key].scale = scale
                
        elif bits == 4:
            if has_bnb and hasattr(bnb.nn, "Int4Params"):
                # Use bitsandbytes for INT4 quantization if available
                quantized_dict[key] = bnb.nn.Int4Params(
                    param.contiguous(),
                    requires_grad=False
                ).to(device)
            else:
                # Simple 4-bit quantization
                scale = param.abs().max() / 7.0
                quantized = (param / scale).round().clamp(-7, 7)
                # Pack two 4-bit values into each byte
                half_size = list(param.size())
                if half_size[-1] % 2 == 1:
                    half_size[-1] += 1  # Ensure even size for packing
                half_size[-1] = half_size[-1] // 2
                packed = torch.zeros(half_size, dtype=torch.int8, device=device)
                
                # Pack values (simplified)
                even_indices = torch.arange(0, param.numel(), 2)
                if even_indices.numel() > 0:
                    packed.view(-1)[...] = (
                        (quantized.view(-1)[even_indices] & 0xF) | 
                        ((quantized.view(-1)[even_indices + 1] & 0xF) << 4)
                    )
                
                quantized_dict[key] = packed
                # Store scale and original shape as attributes
                quantized_dict[key].scale = scale
                quantized_dict[key].original_shape = param.size()
        else:
            # For other bit widths, keep as is
            quantized_dict[key] = param
    
    return quantized_dict

def _quantize_model(model, bits: int, device: str):
    """Quantize a full model"""
    try:
        import bitsandbytes as bnb
        has_bnb = True
    except ImportError:
        has_bnb = False
    
    # For full model quantization with bitsandbytes
    if has_bnb and bits == 8:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Get parent module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model
                if parent_name:
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                
                # Replace with 8-bit linear
                int8_linear = bnb.nn.Linear8bitLt(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None
                )
                
                # Copy weights
                int8_linear.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    int8_linear.bias.data.copy_(module.bias.data)
                
                # Replace module
                setattr(parent, child_name, int8_linear)
    
    return model

def dequantize_state_dict(quantized_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Dequantize a quantized state dict"""
    dequantized_dict = {}
    
    for key, param in quantized_dict.items():
        # Check if this is a quantized tensor with scale attribute
        if hasattr(param, 'scale'):
            if param.dtype == torch.int8:
                # Check if this is packed 4-bit
                if hasattr(param, 'original_shape'):
                    # Unpack 4-bit values
                    original_shape = param.original_shape
                    unpacked = torch.zeros(original_shape, dtype=torch.int8, device=param.device)
                    
                    # Unpack (simplified)
                    for i in range(param.numel()):
                        byte = param.view(-1)[i].item()
                        unpacked.view(-1)[i*2] = byte & 0xF
                        if i*2+1 < unpacked.numel():
                            unpacked.view(-1)[i*2+1] = (byte >> 4) & 0xF
                    
                    # Dequantize
                    dequantized_dict[key] = unpacked.float() * param.scale
                else:
                    # Regular 8-bit quantization
                    dequantized_dict[key] = param.float() * param.scale
            else:
                dequantized_dict[key] = param
        else:
            # Not quantized
            dequantized_dict[key] = param
    
    return dequantized_dict
```

## 6. Main Application Script

```python
# src/main.py
import argparse
import os
import sys
import torch
import time
import multiprocessing as mp

from master.node import MasterNode
from worker.node import Worker