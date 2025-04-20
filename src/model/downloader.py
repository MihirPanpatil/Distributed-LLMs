# src/model/downloader.py
from huggingface_hub import snapshot_download

def download_model(model_id, cache_dir="./models"):
    """Download model from Hugging Face Hub"""
    return snapshot_download(repo_id=model_id, cache_dir=cache_dir)