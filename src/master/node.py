# src/master/node.py
import socket
import threading
import time
import json
import os
import zmq
from typing import Dict, List, Any, Optional
from queue import Queue

from src.network.protocol import MessageProtocol
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
        self.zmq_context = zmq.Context()
        
    def start(self):
        """Start the master server"""
        # Setup ZMQ socket
        protocol = MessageProtocol(zmq_context=self.zmq_context)
        self.server_socket = protocol.setup_zmq_socket(zmq.ROUTER, f"tcp://{self.host}:{self.port}")
        
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
            
    def initialize_model(self, model_id: str, num_shards: int, cache_dir: str = "./models"):
        """Download, initialize and shard a model
        
        Args:
            model_id: HuggingFace model ID or local path
            num_shards: Number of shards to split the model into
            cache_dir: Directory to store downloaded models
            
        Returns:
            Path to the directory containing model shards
        """
        from ..model.downloader import download_model
        from ..model.loader import load_model
        
        # Download model if it's a HuggingFace model ID
        if not os.path.exists(model_id):
            print(f"Downloading model {model_id}...")
            self.model_path = download_model(model_id, cache_dir=cache_dir)
        else:
            self.model_path = model_id
            
        # Load the model and tokenizer
        print(f"Loading model from {self.model_path}...")
        model, self.tokenizer = load_model(self.model_path)
        
        # Initialize shard manager and shard the model
        print(f"Sharding model into {num_shards} parts...")
        self.shard_manager = ModelShardManager(self.model_path, num_shards)
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
                # ZMQ message reception
                # ZMQ message handling
                # ZMQ message reception
                msg_parts = self.server_socket.recv_multipart()
                identity = msg_parts[0]
                threading.Thread(
                    target=self._handle_worker,
                    args=(identity, msg_parts[1:]),
                    daemon=True
                ).start()
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
                
                # Tokenize input if tokenizer is available
                if self.tokenizer:
                    try:
                        input_data = self.tokenizer(input_text, return_tensors="pt")
                        input_data = {k: v.to('cpu').numpy().tobytes() for k, v in input_data.items()}
                        input_data = json.dumps(input_data).encode('utf-8')
                    except Exception as e:
                        print(f"Error tokenizing input: {e}")
                        continue
                else:
                    # Fallback if tokenizer not available
                    input_data = input_text.encode('utf-8')
                
                # Distribute to all workers
                if not self.workers:
                    print(f"No workers connected to process task {task_id}")
                    continue
                    
                if not self.shard_assignments:
                    print(f"No shard assignments available for task {task_id}")
                    continue
                    
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
                # Sleep briefly to prevent tight error loops
                time.sleep(0.5)