import numpy as np
from scipy.sparse import csr_matrix
from .kernels import minsum_core_sparse, syndrome_check, minsum_decoder_full

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
    else:
        raise ValueError(f"Unsupported alpha_mode: {alpha_mode}")
    syndrome = np.asarray(syndrome, dtype=np.int8)
    initialBelief = np.asarray(initialBelief, dtype=np.float64)
    
    H_indices = H_csr.indices.astype(np.int32)
    H_indptr = H_csr.indptr.astype(np.int32)
    
    # Use fully JIT-compiled decoder
    candidateError, converged, values, final_iter = minsum_decoder_full(
        H_indices, H_indptr,
        syndrome, initialBelief,
        maxIter, use_dynamic_alpha, alpha,
        damping, clip_llr
    )
    
    return candidateError, converged, values, final_iter


def performMinSum_Symmetric_Sparse_Legacy(
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
    Legacy implementation with Python loop (kept for reference/debugging).
    """
    if alpha_mode is None:
        use_dynamic_alpha = (alpha == 0)
    elif alpha_mode == "dynamical":
        use_dynamic_alpha = True
    elif alpha_mode == "alvarado":
        if alpha <= 0:
            raise ValueError("alpha must be > 0 when alpha_mode='alvarado'")
        use_dynamic_alpha = False
    else:
        raise ValueError(f"Unsupported alpha_mode: {alpha_mode}")
    syndrome = np.asarray(syndrome, dtype=np.int8)
    initialBelief = np.asarray(initialBelief, dtype=np.float64)
    
    m, n = H_csr.shape
    H_data = H_csr.data
    if H_data.dtype != np.int8:
        H_data = H_data.astype(np.int8)
    H_indices = H_csr.indices
    H_indptr = H_csr.indptr
    
    syndrome_sign = (1 - 2 * syndrome).astype(np.float64)
    
    # Initialize messages
    Q_flat = initialBelief[H_indices].copy()
    Q_flat_old = Q_flat.copy()
    
    candidateError = np.zeros(n, dtype=np.int8)
    values = initialBelief.copy()
    
    for currentIter in range(maxIter):
        current_alpha = (1.0 - 2.0 ** (-(currentIter + 1))) if use_dynamic_alpha else alpha
        
        R_flat, R_sum = minsum_core_sparse(
            H_data, H_indices, H_indptr, Q_flat, syndrome_sign, current_alpha, m, n
        )
        
        values = R_sum + initialBelief
        
        with np.errstate(invalid='ignore'):
            Q_flat_new = values[H_indices] - R_flat
        Q_flat_new = np.nan_to_num(Q_flat_new, nan=0.0, posinf=clip_llr, neginf=-clip_llr)
        Q_flat_new = np.clip(Q_flat_new, -clip_llr, clip_llr)
        
        Q_flat = damping * Q_flat_new + (1 - damping) * Q_flat_old
        Q_flat = np.clip(Q_flat, -clip_llr, clip_llr)
        Q_flat_old = Q_flat.copy()
        
        candidateError = (values < 0).astype(np.int8)
        calculateSyndrome = syndrome_check(H_data, H_indices, H_indptr, candidateError, m)
        
        if np.array_equal(calculateSyndrome, syndrome):
            return candidateError, True, values, currentIter
            
    return candidateError, False, values, currentIter
