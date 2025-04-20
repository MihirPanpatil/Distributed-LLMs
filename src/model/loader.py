# src/model/loader.py
from transformers import AutoModel, AutoTokenizer, OPTForCausalLM
import torch

def load_model(model_id, device_map="auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Special handling for OPT models
    if "facebook/opt" in model_id.lower():
        model = OPTForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            offload_folder="offload",
            offload_state_dict=True
        )
    else:
        model = AutoModel.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    
    return model, tokenizer