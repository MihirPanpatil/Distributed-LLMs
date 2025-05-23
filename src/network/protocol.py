# src/network/protocol.py
import socket
import pickle
import struct
import zmq
from typing import Dict, Any, Optional, Tuple

HEADER_SIZE = 10  # Size of the header length field

class MessageProtocol:
    # Supported message types
    MESSAGE_TYPES = {
        "REGISTER": "Worker registration",
        "LOAD_SHARD": "Transfer model shard",
        "RUN_INFERENCE": "Execute computation",
        "RESULT": "Return computation results",
        "HEARTBEAT": "Health check",
        "SHARD_REQUEST": "Request specific shard",
        "TASK_ASSIGN": "Assign computation task"
    }
    
    def __init__(self, zmq_context=None):
        self.zmq_context = zmq_context or zmq.Context()
        self.zmq_socket = None
        self.use_zmq = False  # Flag to track if ZMQ is being used
    
    def setup_zmq_socket(self, socket_type, address=None):
        """Setup a ZeroMQ socket for communication"""
        socket = self.zmq_context.socket(socket_type)
        if address:
            if socket_type == zmq.PUB or socket_type == zmq.PUSH or socket_type == zmq.REP:
                socket.bind(address)
            else:
                socket.connect(address)
        self.zmq_socket = socket
        return socket
    
    def send_message(self, sock: socket.socket, command: str, payload: Optional[bytes] = None, 
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send a message with the specified command and optional payload"""
        # Use ZeroMQ if available
        if self.zmq_socket:
            return self._send_zmq_message(command, payload, metadata)
        # Fallback to regular socket
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
            
        except socket.timeout:
            print("Timeout while sending message")
            return False
        except Exception as e:
            print(f"Error sending message: {e}")
            return False
    
    def receive_message(self, sock: socket.socket, timeout: int = 60) -> Tuple[Dict[str, Any], Optional[bytes]]:
        """Receive a message and return header and payload"""
        # Use ZeroMQ if available
        if self.zmq_socket:
            return self._receive_zmq_message(timeout)
            
        # Fallback to regular socket
        try:
            sock.settimeout(timeout)
            
            # Receive header length
            header_len_bytes = b""
            while len(header_len_bytes) < HEADER_SIZE:
                chunk = sock.recv(HEADER_SIZE - len(header_len_bytes))
                if not chunk:
                    return {}, None
                header_len_bytes += chunk
                
            header_len = int(header_len_bytes.decode().strip())
            
            # Receive header
            header_bytes = b""
            while len(header_bytes) < header_len:
                chunk = sock.recv(header_len - len(header_bytes))
                if not chunk:
                    raise ConnectionError("Connection closed while receiving header")
                header_bytes += chunk
            
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
            
    def _send_zmq_message(self, command: str, payload: Optional[bytes] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send message using ZeroMQ socket with identity frame"""
        try:
            header = {"command": command}
            if metadata:
                header.update(metadata)
            
            if payload is not None:
                header["payload_size"] = len(payload)
            
            # Get identity from metadata if present
            identity = metadata.get('identity', b'')
            
            # Send as multipart message with identity
            self.zmq_socket.send_multipart([
                identity,
                pickle.dumps(header),
                payload if payload is not None else b""
            ])
            return True
            
        except Exception as e:
            print(f"Error sending ZMQ message: {e}")
            return False
            
    def _receive_zmq_message(self, timeout: int = 60) -> Tuple[bytes, Dict[str, Any], Optional[bytes]]:
        """Receive message using ZeroMQ socket with identity frame"""
        try:
            self.zmq_socket.setsockopt(zmq.RCVTIMEO, timeout * 1000)
            parts = self.zmq_socket.recv_multipart()
            
            if len(parts) < 2:
                return b'', {}, None
                
            identity = parts[0]
            header = pickle.loads(parts[1])
            payload = parts[2] if len(parts) > 2 else None
            
            return identity, header, payload
            
        except zmq.Again:
            raise TimeoutError("Timeout while receiving ZMQ message")
        except Exception as e:
            print(f"Error receiving ZMQ message: {e}")
            raise