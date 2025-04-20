import unittest
import torch
from unittest.mock import patch, MagicMock

from src.model.loader import load_model

class TestModelLoader(unittest.TestCase):
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_load_model(self, mock_tokenizer, mock_model):
        # Mock model and tokenizer
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        
        # Test loading model
        model, tokenizer = load_model("test_model")
        
        # Verify calls
        mock_model.assert_called_once()
        mock_tokenizer.assert_called_once()
        self.assertIsNotNone(model)
        self.assertIsNotNone(tokenizer)
    
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_load_model_device_mapping(self, mock_tokenizer, mock_model):
        # Test device mapping
        model, _ = load_model("test_model", device_map="auto")
        mock_model.assert_called_with(
            "test_model",
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    
    @patch('transformers.AutoModel.from_pretrained', side_effect=Exception("Invalid path"))
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_load_model_invalid_path(self, mock_tokenizer, mock_model):
        # Test error handling
        with self.assertRaises(Exception):
            load_model("invalid_path")

if __name__ == '__main__':
    unittest.main()