# src/model/loader.py
from transformers import AutoModel, AutoTokenizer
import torch

def load_model(model_id, device_map="auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    return model, tokenizer