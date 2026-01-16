import os
import hashlib
import numpy as np
from typing import Dict, Any, Optional

def compute_cache_key(Hx, Hz, Lx, Lz, num_cycles, error_rate) -> str:
    hasher = hashlib.sha256()
    for arr in [Hx, Hz, Lx, Lz]: hasher.update(arr.tobytes())
    hasher.update(str(num_cycles).encode())
    hasher.update(f"{error_rate:.6f}".encode())
    return hasher.hexdigest()[:16]

def save_matrices(cache_dir, cache_key, matrices):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"matrices_{cache_key}.npz")
    np.savez_compressed(
        path,
        HdecZ=matrices['HdecZ'], HdecX=matrices['HdecX'],
        channel_probsZ=matrices['channel_probsZ'], channel_probsX=matrices['channel_probsX'],
        HZ_full=matrices['HZ_full'], HX_full=matrices['HX_full'],
        first_logical_rowZ=np.array([matrices['first_logical_rowZ']]),
        first_logical_rowX=np.array([matrices['first_logical_rowX']]),
        num_cycles=np.array([matrices['num_cycles']]),
        k=np.array([matrices['k']]),
    )
    return path

def load_matrices(cache_dir, cache_key) -> Optional[Dict[str, Any]]:
    path = os.path.join(cache_dir, f"matrices_{cache_key}.npz")
    if not os.path.exists(path): return None
    try:
        data = np.load(path)
        return {
            'HdecZ': data['HdecZ'], 'HdecX': data['HdecX'],
            'channel_probsZ': data['channel_probsZ'], 'channel_probsX': data['channel_probsX'],
            'HZ_full': data['HZ_full'], 'HX_full': data['HX_full'],
            'first_logical_rowZ': int(data['first_logical_rowZ'][0]),
            'first_logical_rowX': int(data['first_logical_rowX'][0]),
            'num_cycles': int(data['num_cycles'][0]),
            'k': int(data['k'][0]),
        }
    except Exception: return None
