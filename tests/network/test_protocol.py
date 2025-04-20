import unittest
import socket
import pickle
from unittest.mock import patch, MagicMock

from src.network.protocol import MessageProtocol

class TestMessageProtocol(unittest.TestCase):
    @patch('socket.socket')
    def setUp(self, mock_socket):
        self.mock_socket = mock_socket.return_value
        self.protocol = MessageProtocol
    
    def test_send_message(self):
        # Test sending message without payload
        result = self.protocol.send_message(self.mock_socket, "TEST_COMMAND")
        self.assertTrue(result)
        self.mock_socket.sendall.assert_called()
    
    def test_send_message_with_payload(self):
        # Test sending message with payload
        payload = b"test_payload"
        result = self.protocol.send_message(self.mock_socket, "TEST_COMMAND", payload=payload)
        self.assertTrue(result)
        self.assertEqual(self.mock_socket.sendall.call_count, 2)
    
    def test_send_message_with_metadata(self):
        # Test sending message with metadata
        metadata = {"key": "value"}
        result = self.protocol.send_message(self.mock_socket, "TEST_COMMAND", metadata=metadata)
        self.assertTrue(result)
        self.mock_socket.sendall.assert_called()
    
    @patch('socket.socket.recv')
    def test_receive_message(self, mock_recv):
        # Mock header and payload
        header = {"command": "TEST_COMMAND"}
        header_bytes = pickle.dumps(header)
        header_len = len(header_bytes)
        
        # Configure mock to return header length and header
        mock_recv.side_effect = [
            f"{header_len:<10}".encode(),
            header_bytes
        ]
        
        # Test receiving message
        received_header, received_payload = self.protocol.receive_message(self.mock_socket)
        self.assertEqual(received_header["command"], "TEST_COMMAND")
        self.assertIsNone(received_payload)
    
    @patch('socket.socket.recv')
    def test_receive_message_with_payload(self, mock_recv):
        # Mock header and payload
        header = {"command": "TEST_COMMAND", "payload_size": 12}
        header_bytes = pickle.dumps(header)
        header_len = len(header_bytes)
        payload = b"test_payload"
        
        # Configure mock to return header length, header, and payload
        mock_recv.side_effect = [
            f"{header_len:<10}".encode(),
            header_bytes,
            payload
        ]
        
        # Test receiving message with payload
        received_header, received_payload = self.protocol.receive_message(self.mock_socket)
        self.assertEqual(received_header["command"], "TEST_COMMAND")
        self.assertEqual(received_payload, payload)

    @patch('socket.socket.recv')
    def test_receive_message_timeout(self, mock_recv):
        # Test timeout handling
        mock_recv.side_effect = socket.timeout()
        with self.assertRaises(TimeoutError):
            self.protocol.receive_message(self.mock_socket)
    
    @patch('socket.socket.recv')
    def test_receive_message_invalid_header(self, mock_recv):
        # Test invalid header length
        mock_recv.side_effect = [b"invalid", b"header"]
        with self.assertRaises(ValueError):
            self.protocol.receive_message(self.mock_socket)
    
    @patch('socket.socket.recv')
    def test_receive_message_connection_closed(self, mock_recv):
        # Test connection closed during receive
        mock_recv.return_value = b""
        with self.assertRaises(ConnectionError):
            self.protocol.receive_message(self.mock_socket)

if __name__ == '__main__':
    unittest.main()