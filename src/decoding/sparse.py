import numpy as np
from scipy.sparse import csr_matrix
from .kernels import minsum_core_sparse, syndrome_check, minsum_decoder_full, minsum_decoder_full_autoregressive

def performMinSum_Symmetric_Sparse(
    H_csr, 
    syndrome, 
    initialBelief, 
    maxIter=100, 
    alpha=1.0,
    alpha_mode="dynamical",
    damping=1.0, 
    clip_llr=20.0
):
    """
    Highly optimized Sparse Min-Sum Algorithm using fully JIT-compiled decoder.
    """
    if alpha_mode is None:
        use_dynamic_alpha = (alpha == 0)
    elif alpha_mode == "dynamical":
        use_dynamic_alpha = True
    elif alpha_mode == "alvarado":
        if alpha <= 0:
            raise ValueError("alpha must be > 0 when alpha_mode='alvarado'")
        use_dynamic_alpha = False
    elif alpha_mode == "alvarado-autoregressive":
        use_dynamic_alpha = False
    else:
        raise ValueError(f"Unsupported alpha_mode: {alpha_mode}")
    syndrome = np.asarray(syndrome, dtype=np.int8)
    initialBelief = np.asarray(initialBelief, dtype=np.float64)
    
    H_indices = H_csr.indices.astype(np.int32)
    H_indptr = H_csr.indptr.astype(np.int32)
    
    if alpha_mode == "alvarado-autoregressive":
        alpha_seq = np.asarray(alpha, dtype=np.float64)
        if alpha_seq.ndim != 1 or alpha_seq.size == 0:
            raise ValueError("alpha must be a non-empty 1D sequence for alvarado-autoregressive")
        candidateError, converged, values, final_iter = minsum_decoder_full_autoregressive(
            H_indices, H_indptr,
            syndrome, initialBelief,
            maxIter, alpha_seq, alpha_seq.size,
            damping, clip_llr
        )
    else:
        # Use fully JIT-compiled decoder
        candidateError, converged, values, final_iter = minsum_decoder_full(
            H_indices, H_indptr,
            syndrome, initialBelief,
            maxIter, use_dynamic_alpha, alpha,
            damping, clip_llr
        )
    
    return candidateError, converged, values, final_iter