import numpy as np
from scipy.sparse import csr_matrix
from .kernels import bp_core, minsum_core

def performMinSum_Symmetric(
    H,
    syndrome,
    initialBelief,
    maxIter=50,
    alpha=1.0,
    alpha_mode="dynamical",
    damping=1.0,
    clip_llr=20.0,
    alpha_estimation=False,
):
    """
    Normalized Min-Sum Algorithm (Numba-accelerated).
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
    H = np.asarray(H, dtype=np.float64)
    syndrome = np.asarray(syndrome, dtype=np.int8)
    initialBelief = np.asarray(initialBelief, dtype=np.float64)
    
    H_sparse = csr_matrix(H)
    mask = H != 0
    syndrome_sign = (1 - 2 * syndrome).astype(np.float64).reshape(-1, 1)
    
    Q = np.where(mask, initialBelief, 0.0)
    Q_old = Q.copy()
    m, n = H.shape
    candidateError = np.zeros(n, dtype=np.int8)
    
    for currentIter in range(maxIter):
        current_alpha = (1.0 - 2.0 ** (-(currentIter + 1))) if use_dynamic_alpha else alpha
        R = minsum_core(H, Q, syndrome_sign, mask, current_alpha)

        if alpha_estimation and currentIter == 0:
            scale = current_alpha if current_alpha != 0 else 1.0
            return np.zeros(n, dtype=np.int8), False, R / scale, 0

        R_sum = np.sum(R, axis=0)
        values = R_sum + initialBelief
        with np.errstate(invalid='ignore'):
            Q_new = np.where(mask, values - R, 0.0)
        Q_new = np.nan_to_num(Q_new, nan=0.0, posinf=clip_llr, neginf=-clip_llr)
        Q = damping * Q_new + (1 - damping) * Q_old
        Q = np.clip(Q, -clip_llr, clip_llr)
        Q_old = Q.copy()
        
        candidateError = (values < 0).astype(np.int8)
        calculateSyndrome = H_sparse.dot(candidateError) % 2
        
        if np.array_equal(calculateSyndrome, syndrome) and not alpha_estimation:
            return candidateError, True, values, currentIter
            
    return candidateError, False, values, currentIter

def performBeliefPropagationFast(H, syndrome, initialBelief, maxIter=50):
    """Fast Belief Propagation (Numba-accelerated)."""
    H = np.asarray(H, dtype=np.float64)
    syndrome = np.asarray(syndrome, dtype=np.int8)
    initialBelief = np.asarray(initialBelief, dtype=np.float64)
    H_sparse = csr_matrix(H)
    mask = H != 0
    syndrome_sign = (1 - 2 * syndrome).astype(np.float64).reshape(-1, 1)
    Q = np.where(mask, initialBelief, 0.0)
    CLIP_VAL = 0.9999999
    m, n = H.shape
    for currentIter in range(maxIter):
        R = bp_core(H, Q, syndrome_sign, mask, CLIP_VAL)
        R_sum = np.sum(R, axis=0)
        values = R_sum + initialBelief
        with np.errstate(invalid='ignore'):
            Q = np.where(mask, values - R, 0.0)
        candidateError = (values < 0).astype(np.int8)
        calculateSyndrome = H_sparse.dot(candidateError) % 2
        if np.array_equal(calculateSyndrome, syndrome):
            return candidateError, True, values, currentIter
    return candidateError, False, values, currentIter
