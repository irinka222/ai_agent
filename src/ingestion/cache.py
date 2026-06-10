import hashlib
import json
from pathlib import Path
from src.config import MAX_CHUNK_SIZE, CHUNK_OVERLAP, INCLUDE_REFERENCES

def compute_cache_key(pdf_path: str) -> str:
    with open(pdf_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    config_str = json.dumps({
        'max_chunk_size': MAX_CHUNK_SIZE,
        'overlap': CHUNK_OVERLAP,
        'include_references': INCLUDE_REFERENCES,
    }, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    return f"{file_hash}_{config_hash}"