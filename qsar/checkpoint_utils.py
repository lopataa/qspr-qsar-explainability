from pathlib import Path
import pickle
import hashlib

def file_signature(path, block_size=65536):
    path = Path(path)
    if not path.exists():
        return None
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            hasher.update(block)
    return f"sha256:{hasher.hexdigest()}"

def save_pickle_cache(path, metadata, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache_obj = {"metadata": metadata, "data": data}
    with open(path, "wb") as f:
        pickle.dump(cache_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle_cache(path, expected_metadata):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        try:
            cache_obj = pickle.load(f)
            if cache_obj.get("metadata") == expected_metadata:
                return cache_obj.get("data")
        except (pickle.UnpicklingError, EOFError):
            pass
    return None
