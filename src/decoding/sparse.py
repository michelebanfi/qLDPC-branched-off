import numpy as np
from scipy.sparse import csr_matrix
from .kernels import minsum_core_sparse, syndrome_check

def performMinSum_Symmetric_Sparse(
    H_csr, 
    syndrome, 
    initialBelief, 
    maxIter=100, 
    alpha=1.0, 
    damping=1.0, 
    clip_llr=20.0
):
    """
    Highly optimized Sparse Min-Sum Algorithm using CSR components.
    """
    use_dynamic_alpha = (alpha == 0)
    syndrome = np.asarray(syndrome, dtype=np.int8)
    initialBelief = np.asarray(initialBelief, dtype=np.float64)
    
    m, n = H_csr.shape
    H_data = H_csr.data.astype(np.int8)
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
