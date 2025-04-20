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
                    break
                    
                command = header.get("command", "")
                self._handle_command(self.master_socket, command, header, payload)
                
        except Exception as e:
            print(f"Error handling master connection: {e}")
        finally:
            self.master_socket.close()
    
    def _handle_command(self, socket, command, header, payload):
        """Process commands from master"""
        if command == "LOAD_SHARD":
            # Load a model shard
            shard_id = header.get("shard_id")
            parameters = pickle.loads(payload)
            self.shards[shard_id] = ModelShard(shard_id, parameters)
            self.shards[shard_id].to_device()
            
            # Send confirmation
            MessageProtocol.send_message(
                socket,
                "SHARD_LOADED",
                metadata={"shard_id": shard_id}
            )
            
        elif command == "UNLOAD_SHARD":
            # Unload a model shard
            shard_id = header.get("shard_id")
            if shard_id in self.shards:
                del self.shards[shard_id]
                
            # Send confirmation
            MessageProtocol.send_message(
                socket,
                "SHARD_UNLOADED",
                metadata={"shard_id": shard_id}
            )
            
        elif command == "RUN_INFERENCE":
            # Run computation on loaded shards
            task_id = header.get("task_id")
            shard_ids = header.get("shard_ids", [])
            
            # Process input (simplified)
            inputs = pickle.loads(payload)
            
            # Compute results for each shard
            results = {}
            for shard_id in shard_ids:
                if shard_id in self.shards:
                    results.update(self.shards[shard_id].compute(inputs))
            
            # Send results back
            MessageProtocol.send_message(
                socket,
                "RESULT",
                payload=pickle.dumps(results),
                metadata={"task_id": task_id}
            )
            
        elif command == "SCHEDULE_COMPUTATION":
            # Schedule computation with priority
            task_id = header.get("task_id")
            shard_ids = header.get("shard_ids", [])
            priority = header.get("priority", 0)
            
            # Process input
            inputs = pickle.loads(payload)
            
            # Compute with priority scheduling
            results = {}
            for shard_id in sorted(shard_ids, key=lambda x: -priority):
                if shard_id in self.shards:
                    results.update(self.shards[shard_id].compute(inputs))
            
            # Send results back
            MessageProtocol.send_message(
                socket,
                "RESULT",
                payload=pickle.dumps(results),
                metadata={"task_id": task_id}
            )
    
    def _send_heartbeat(self):
        """Send periodic heartbeat messages to master"""
        while self.running:
            try:
                if self.master_socket:
                    MessageProtocol.send_message(
                        self.master_socket,
                        "HEARTBEAT",
                        metadata={"timestamp": time.time()}
                    )
                time.sleep(5)
            except Exception as e:
                print(f"Error sending heartbeat: {e}")
                time.sleep(1)

    def load_shard(self, shard_id: int, shard_data: bytes) -> bool:
        """Load a model shard into memory"""
        try:
            parameters = pickle.loads(shard_data)
            self.shards[shard_id] = ModelShard(shard_id, parameters)
            self.shards[shard_id].to_device()
            return True
        except Exception as e:
            print(f"Error loading shard {shard_id}: {e}")
            return False

    def unload_shard(self, shard_id: int) -> bool:
        """Unload a model shard from memory"""
        if shard_id in self.shards:
            del self.shards[shard_id]
            return True
        return False

    def schedule_computation(self, inputs: Dict[str, torch.Tensor], shard_ids: List[int]) -> Dict[str, torch.Tensor]:
        """Schedule computation on specified shards"""
        results = {}
        for shard_id in shard_ids:
            if shard_id in self.shards:
                results.update(self.shards[shard_id].compute(inputs))
        return results