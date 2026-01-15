"""
Matrix caching utilities for BB code decoding matrices.

This module provides functions to save and load precomputed decoding matrices
to avoid recomputation across simulation runs.
"""

import os
import hashlib
import numpy as np
from typing import Dict, Any, Optional


def get_cache_dir() -> str:
    """Get the cache directory path, creating it if it doesn't exist."""
    cache_dir = os.path.join(os.path.dirname(__file__), "matrix_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def compute_cache_key(
    Hx: np.ndarray,
    Hz: np.ndarray,
    Lx: np.ndarray,
    Lz: np.ndarray,
    num_cycles: int,
    error_rate: float
) -> str:
    """
    Compute a unique cache key based on the code parameters and settings.
    
    The key is a hash of:
    - Hx, Hz parity check matrices
    - Lx, Lz logical operators
    - Number of syndrome cycles
    - Error rate (rounded to avoid floating point issues)
    
    Returns:
        A hex string that uniquely identifies this configuration.
    """
    # Create a combined representation of all parameters
    hasher = hashlib.sha256()
    
    # Add matrices (as bytes)
    hasher.update(Hx.tobytes())
    hasher.update(Hz.tobytes())
    hasher.update(Lx.tobytes())
    hasher.update(Lz.tobytes())
    
    # Add numeric parameters
    hasher.update(str(num_cycles).encode())
    # Round error rate to 6 decimal places to avoid floating point issues
    hasher.update(f"{error_rate:.6f}".encode())
    
    return hasher.hexdigest()[:16]  # Use first 16 characters for brevity


def get_cache_path(cache_key: str) -> str:
    """Get the full path to the cache file for a given key."""
    return os.path.join(get_cache_dir(), f"matrices_{cache_key}.npz")


def save_matrices(cache_key: str, matrices: Dict[str, Any]) -> str:
    """
    Save precomputed matrices to disk.
    
    Args:
        cache_key: Unique identifier for this configuration
        matrices: Dictionary of matrices returned by build_decoding_matrices
        
    Returns:
        Path to the saved file
    """
    cache_path = get_cache_path(cache_key)
    
    # Convert all values to arrays for npz storage
    np.savez_compressed(
        cache_path,
        HdecZ=matrices['HdecZ'],
        HdecX=matrices['HdecX'],
        channel_probsZ=matrices['channel_probsZ'],
        channel_probsX=matrices['channel_probsX'],
        HZ_full=matrices['HZ_full'],
        HX_full=matrices['HX_full'],
        first_logical_rowZ=np.array([matrices['first_logical_rowZ']]),
        first_logical_rowX=np.array([matrices['first_logical_rowX']]),
        num_cycles=np.array([matrices['num_cycles']]),
        k=np.array([matrices['k']]),
    )
    
    return cache_path


def load_matrices(cache_key: str) -> Optional[Dict[str, Any]]:
    """
    Load precomputed matrices from disk if they exist.
    
    Args:
        cache_key: Unique identifier for this configuration
        
    Returns:
        Dictionary of matrices, or None if not cached
    """
    cache_path = get_cache_path(cache_key)
    
    if not os.path.exists(cache_path):
        return None
    
    try:
        data = np.load(cache_path)
        matrices = {
            'HdecZ': data['HdecZ'],
            'HdecX': data['HdecX'],
            'channel_probsZ': data['channel_probsZ'],
            'channel_probsX': data['channel_probsX'],
            'HZ_full': data['HZ_full'],
            'HX_full': data['HX_full'],
            'first_logical_rowZ': int(data['first_logical_rowZ'][0]),
            'first_logical_rowX': int(data['first_logical_rowX'][0]),
            'num_cycles': int(data['num_cycles'][0]),
            'k': int(data['k'][0]),
        }
        return matrices
    except Exception as e:
        print(f"Warning: Failed to load cached matrices: {e}")
        return None


def get_or_compute_matrices(
    circuit_builder,
    Lx: np.ndarray,
    Lz: np.ndarray,
    error_rate: float,
    Hx: np.ndarray,
    Hz: np.ndarray,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Get decoding matrices, loading from cache if available or computing if not.
    
    This is the main entry point for cached matrix access.
    
    Args:
        circuit_builder: BBCodeCircuit instance
        Lx: X logical operators matrix
        Lz: Z logical operators matrix
        error_rate: Physical error rate
        Hx: X-type parity check matrix (for cache key computation)
        Hz: Z-type parity check matrix (for cache key computation)
        verbose: Print progress information
        
    Returns:
        Dictionary containing decoding matrices
    """
    from circuit_noise import build_decoding_matrices
    
    # Compute cache key
    cache_key = compute_cache_key(
        Hx, Hz, Lx, Lz,
        circuit_builder.num_cycles,
        error_rate
    )
    
    # Try to load from cache
    cached = load_matrices(cache_key)
    if cached is not None:
        if verbose:
            print(f"Loaded matrices from cache (key: {cache_key})")
        return cached
    
    # Compute matrices
    if verbose:
        print(f"Computing matrices (will be cached with key: {cache_key})")
    
    matrices = build_decoding_matrices(
        circuit_builder, Lx, Lz, error_rate, verbose=verbose, num_workers=8
    )
    
    # Save to cache
    cache_path = save_matrices(cache_key, matrices)
    if verbose:
        print(f"Saved matrices to {cache_path}")
    
    return matrices


def clear_cache() -> int:
    """
    Clear all cached matrices.
    
    Returns:
        Number of files deleted
    """
    cache_dir = get_cache_dir()
    count = 0
    
    for filename in os.listdir(cache_dir):
        if filename.startswith("matrices_") and filename.endswith(".npz"):
            os.remove(os.path.join(cache_dir, filename))
            count += 1
    
    return count


def list_cached_matrices() -> list:
    """
    List all cached matrix files.
    
    Returns:
        List of cache keys that have been saved
    """
    cache_dir = get_cache_dir()
    keys = []
    
    if os.path.exists(cache_dir):
        for filename in os.listdir(cache_dir):
            if filename.startswith("matrices_") and filename.endswith(".npz"):
                # Extract key from filename
                key = filename[9:-4]  # Remove "matrices_" prefix and ".npz" suffix
                keys.append(key)
    
    return keys
