"""
GPU-accelerated decoding functions using PyTorch MPS (Metal Performance Shaders).

This module provides GPU-accelerated versions of the decoding algorithms
optimized for Apple Silicon (M1/M2/M3) GPUs.

Usage:
    from decoding_gpu import BatchMinSumDecoder
    
    decoder = BatchMinSumDecoder(H, device='mps')
    results = decoder.decode_batch(syndromes, initial_beliefs)
"""

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
    MPS_AVAILABLE = torch.backends.mps.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    MPS_AVAILABLE = False


class BatchMinSumDecoder:
    """
    GPU-accelerated batch Min-Sum decoder using PyTorch MPS.
    
    Decodes multiple syndromes in parallel on the GPU for maximum throughput.
    Best used when you have many syndromes to decode at once (batch_size >= 16).
    """
    
    def __init__(self, H, device='mps', alpha=1.0, max_iter=50, clip_llr=50.0):
        """
        Initialize the decoder with parity check matrix.
        
        Args:
            H: Parity check matrix (m x n)
            device: PyTorch device ('mps' for Apple GPU, 'cpu' for CPU)
            alpha: Scaling factor for Min-Sum normalization
            max_iter: Maximum BP iterations
            clip_llr: LLR clipping value
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for GPU decoding. Install with: pip install torch")
        
        if device == 'mps' and not MPS_AVAILABLE:
            print("Warning: MPS not available, falling back to CPU")
            device = 'cpu'
        
        self.device = torch.device(device)
        self.alpha = alpha
        self.max_iter = max_iter
        self.clip_llr = clip_llr
        
        # Pre-convert H to GPU tensor
        H_np = np.asarray(H, dtype=np.float32)
        self.H = torch.tensor(H_np, device=self.device, dtype=torch.float32)
        self.mask = self.H != 0
        self.m, self.n = H_np.shape
        
        # Pre-compute sparse structure for syndrome check
        self.H_sparse = torch.tensor(H_np, device=self.device, dtype=torch.float32)
    
    def decode_batch(self, syndromes, initial_beliefs, alpha=None):
        """
        Decode a batch of syndromes in parallel on GPU.
        
        Args:
            syndromes: Batch of syndromes, shape (batch_size, m)
            initial_beliefs: Batch of initial LLRs, shape (batch_size, n)
            alpha: Optional override for scaling factor
            
        Returns:
            Tuple of (candidate_errors, success_flags, final_values)
            - candidate_errors: shape (batch_size, n), int
            - success_flags: shape (batch_size,), bool
            - final_values: shape (batch_size, n), float
        """
        if alpha is None:
            alpha = self.alpha
        
        syndromes = np.atleast_2d(syndromes)
        initial_beliefs = np.atleast_2d(initial_beliefs)
        batch_size = syndromes.shape[0]
        
        # Move to GPU
        syn = torch.tensor(syndromes, device=self.device, dtype=torch.float32)
        beliefs = torch.tensor(initial_beliefs, device=self.device, dtype=torch.float32)
        
        # syndrome_sign: shape (batch, m, 1)
        syndrome_sign = (1 - 2 * syn).unsqueeze(-1)
        
        # Expand mask for batch: (m, n) -> (batch, m, n)
        mask_batch = self.mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Initialize Q: (batch, m, n)
        Q = torch.where(mask_batch, beliefs.unsqueeze(1).expand(-1, self.m, -1), 
                       torch.zeros_like(mask_batch, dtype=torch.float32))
        Q_old = Q.clone()
        
        # Iterative decoding
        for it in range(self.max_iter):
            # === Check-to-variable messages (Min-Sum) ===
            # Sign of Q
            sign_Q = torch.sign(Q)
            sign_Q = torch.where(sign_Q == 0, torch.ones_like(sign_Q), sign_Q)
            sign_Q = torch.where(mask_batch, sign_Q, torch.ones_like(sign_Q))
            
            # Product of signs per row
            row_sign_prod = torch.prod(sign_Q, dim=2, keepdim=True)
            r_signs = row_sign_prod * sign_Q
            
            # Absolute values
            abs_Q = torch.abs(Q)
            abs_Q_masked = torch.where(mask_batch, abs_Q, torch.full_like(abs_Q, float('inf')))
            
            # Find min1 and min2
            min1_vals, min1_idx = torch.min(abs_Q_masked, dim=2, keepdim=True)
            
            # Set min1 positions to inf and find min2
            temp_Q = abs_Q_masked.clone()
            temp_Q.scatter_(2, min1_idx, float('inf'))
            min2_vals, _ = torch.min(temp_Q, dim=2, keepdim=True)
            
            # Choose min2 for min1 position, min1 otherwise
            is_min1 = (abs_Q_masked == min1_vals)
            magnitudes = torch.where(is_min1, min2_vals, min1_vals)
            
            # Compute R messages
            R_new = alpha * syndrome_sign * r_signs * magnitudes
            R_new = torch.where(mask_batch, R_new, torch.zeros_like(R_new))
            
            # === Variable-to-check messages ===
            R_sum = R_new.sum(dim=1)  # (batch, n)
            values = R_sum + beliefs  # (batch, n)
            
            Q_new = torch.where(mask_batch, 
                               values.unsqueeze(1) - R_new, 
                               torch.zeros_like(R_new))
            
            # Damping and clipping
            Q = Q_new  # No damping for simplicity
            Q = torch.clamp(Q, -self.clip_llr, self.clip_llr)
            Q_old = Q.clone()
            
            # Check for convergence
            candidate = (values < 0).float()  # (batch, n)
            calculated_syn = (candidate @ self.H.T) % 2  # (batch, m)
            
            converged = torch.all(calculated_syn == syn, dim=1)  # (batch,)
            if torch.all(converged):
                break
        
        # Final results
        candidate_errors = (values < 0).cpu().numpy().astype(np.int8)
        success_flags = converged.cpu().numpy()
        final_values = values.cpu().numpy()
        
        return candidate_errors, success_flags, final_values
    
    def decode_single(self, syndrome, initial_belief, alpha=None):
        """Decode a single syndrome (convenience wrapper)."""
        candidates, successes, values = self.decode_batch(
            syndrome.reshape(1, -1), 
            initial_belief.reshape(1, -1),
            alpha
        )
        return candidates[0], successes[0], values[0]


def check_gpu_available():
    """Check if MPS GPU acceleration is available."""
    if not TORCH_AVAILABLE:
        return False, "PyTorch not installed"
    if not MPS_AVAILABLE:
        return False, "MPS backend not available"
    return True, f"MPS available: {torch.backends.mps.is_built()}"


# Convenience function for quick GPU check
def get_device():
    """Get the best available device for computation."""
    if TORCH_AVAILABLE and MPS_AVAILABLE:
        return 'mps'
    return 'cpu'
