import numpy as np
from itertools import combinations
from scipy.sparse import csr_matrix
from numba import njit, prange

# ============================================================================
# Numba JIT-compiled kernels for performance-critical operations
# ============================================================================

@njit(cache=True, nogil=True)
def _gf2_elimination_numba(A, b):
    """
    JIT-compiled GF2 Gaussian elimination.
    Returns: (reduced matrix A, reduced vector b, pivot_rows, pivot_cols)
    """
    m, n = A.shape
    
    pivot_rows = np.empty(min(m, n), dtype=np.int64)
    pivot_cols = np.empty(min(m, n), dtype=np.int64)
    num_pivots = 0
    
    row = 0
    for col in range(n):
        if row >= m:
            break
        
        # Find pivot
        pivot_row = -1
        for r in range(row, m):
            if A[r, col] == 1:
                pivot_row = r
                break
        
        if pivot_row == -1:
            continue
        
        # Swap rows
        if pivot_row != row:
            for j in range(n):
                A[row, j], A[pivot_row, j] = A[pivot_row, j], A[row, j]
            b[row], b[pivot_row] = b[pivot_row], b[row]
        
        pivot_rows[num_pivots] = row
        pivot_cols[num_pivots] = col
        num_pivots += 1
        
        # Eliminate
        for r in range(m):
            if r != row and A[r, col] == 1:
                for j in range(n):
                    A[r, j] ^= A[row, j]
                b[r] ^= b[row]
        
        row += 1
    
    return A, b, pivot_rows[:num_pivots], pivot_cols[:num_pivots]


@njit(cache=True, nogil=True)
def _minsum_core(H, Q, syndrome_sign, mask, alpha):
    """
    Core Min-Sum message computation - JIT compiled.
    Returns the R matrix (check-to-variable messages).
    """
    m, n = H.shape
    R = np.zeros((m, n), dtype=np.float64)
    
    for i in range(m):
        # Compute sign product and find positions with H[i,j] != 0
        sign_prod = syndrome_sign[i, 0]
        min1 = np.inf
        min2 = np.inf
        min1_idx = -1
        
        for j in range(n):
            if mask[i, j]:
                val = Q[i, j]
                if val >= 0:
                    sign_prod *= 1.0
                else:
                    sign_prod *= -1.0
                
                abs_val = abs(val)
                if abs_val < min1:
                    min2 = min1
                    min1 = abs_val
                    min1_idx = j
                elif abs_val < min2:
                    min2 = abs_val
        
        # Compute messages
        for j in range(n):
            if mask[i, j]:
                val = Q[i, j]
                sign_j = 1.0 if val >= 0 else -1.0
                if sign_j == 0:
                    sign_j = 1.0
                
                # Exclude j from sign product
                row_sign_excl_j = sign_prod * sign_j
                
                # Use min2 if j is the position of min1
                if j == min1_idx:
                    mag = min2
                else:
                    mag = min1
                
                R[i, j] = alpha * row_sign_excl_j * mag
    
    return R


@njit(cache=True, nogil=True)
def _bp_core(H, Q, syndrome_sign, mask, clip_val):
    """
    Core BP tanh-based message computation - JIT compiled.
    Returns the R matrix (check-to-variable messages).
    """
    m, n = H.shape
    R = np.zeros((m, n), dtype=np.float64)
    
    for i in range(m):
        # Compute product of tanh(Q/2) for all j with H[i,j] != 0
        row_prod = 1.0
        for j in range(n):
            if mask[i, j]:
                tanh_val = np.tanh(Q[i, j] * 0.5)
                if abs(tanh_val) < 1e-15:
                    tanh_val = 1e-15 if tanh_val >= 0 else -1e-15
                row_prod *= tanh_val
        
        # Compute messages
        for j in range(n):
            if mask[i, j]:
                tanh_val = np.tanh(Q[i, j] * 0.5)
                if abs(tanh_val) < 1e-15:
                    tanh_val = 1e-15 if tanh_val >= 0 else -1e-15
                
                prod_others = row_prod / tanh_val
                prod_clipped = prod_others * syndrome_sign[i, 0]
                
                # Clip to avoid arctanh domain error
                if prod_clipped > clip_val:
                    prod_clipped = clip_val
                elif prod_clipped < -clip_val:
                    prod_clipped = -clip_val
                
                R[i, j] = 2.0 * np.arctanh(prod_clipped)
    
    return R


@njit(cache=True, nogil=True)
def _compute_metric_numba(solution, llr_abs, syndrome_weight):
    """JIT-compiled metric computation for OSD."""
    if syndrome_weight > 0:
        metric = 1e10 + syndrome_weight * 1e8
    else:
        metric = 0.0
    
    n = len(solution)
    for i in range(n):
        metric += solution[i] * llr_abs[i]
    
    return metric


@njit(cache=True, nogil=True)
def _recompute_solution_numba(H_permuted, s_reduced, e_permuted, pivot_rows, pivot_cols):
    """JIT-compiled solution recomputation for OSD."""
    m, n = H_permuted.shape
    e_full = e_permuted.copy()
    
    num_pivots = len(pivot_rows)
    for idx in range(num_pivots):
        r = pivot_rows[idx]
        c = pivot_cols[idx]
        
        row_contribution = 0
        for col in range(n):
            if col != c and H_permuted[r, col] == 1:
                row_contribution ^= e_full[col]
        
        e_full[c] = s_reduced[r] ^ row_contribution
    
    return e_full


@njit(cache=True, nogil=True)
def _syndrome_check(H_data, H_indices, H_indptr, candidate, m):
    """JIT-compiled sparse syndrome computation."""
    syndrome = np.zeros(m, dtype=np.int8)
    for i in range(m):
        s = 0
        for idx in range(H_indptr[i], H_indptr[i + 1]):
            j = H_indices[idx]
            s ^= candidate[j]
        syndrome[i] = s
    return syndrome

# ============================================================================
# Main decoding functions (unchanged signatures, but now using JIT kernels)
# ============================================================================

def performMinSum_Symmetric(H, syndrome, initialBelief, maxIter=50, alpha=1.0, damping=1.0, clip_llr=20.0, alpha_estimation=False):
    """
    Normalized Min-Sum Algorithm (Numba-accelerated).
    
    alpha: Scaling factor (usually 0.6 - 0.9 for Min-Sum).
           alpha=1.0 makes this standard UMP-BP (Min-Sum).
           alpha=0 enables dynamic scaling: alpha = 1.0 - 2^(-iteration)
    """
    use_dynamic_alpha = (alpha == 0)
    
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
        # Dynamic alpha: increases from 0.5 towards 1.0 as iterations progress
        if use_dynamic_alpha:
            current_alpha = 1.0 - 2.0 ** (-(currentIter + 1))
        else:
            current_alpha = alpha
        
        # Use JIT-compiled core
        R_new = _minsum_core(H, Q, syndrome_sign, mask, current_alpha)
        
        if alpha_estimation:
            return np.zeros(n, dtype=np.int8), False, R_new / current_alpha, 0

        R_sum = np.sum(R_new, axis=0)
        values = R_sum + initialBelief
        
        # Suppress warnings from inf-inf operations (edge cases)
        with np.errstate(invalid='ignore'):
            Q_new = np.where(mask, values - R_new, 0.0)
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
    """
    Fast Belief Propagation (Numba-accelerated).
    """
    H = np.asarray(H, dtype=np.float64)
    syndrome = np.asarray(syndrome, dtype=np.int8)
    initialBelief = np.asarray(initialBelief, dtype=np.float64)
    
    H_sparse = csr_matrix(H)
    mask = H != 0
    syndrome_sign = (1 - 2 * syndrome).astype(np.float64).reshape(-1, 1)
    
    Q = np.where(mask, initialBelief, 0.0)
    
    CLIP_VAL = 0.9999999
    m, n = H.shape
    candidateError = np.zeros(n, dtype=np.int8)
    
    for currentIter in range(maxIter):
        # Use JIT-compiled core
        R = _bp_core(H, Q, syndrome_sign, mask, CLIP_VAL)
        
        R_sum = np.sum(R, axis=0)
        values = R_sum + initialBelief
        
        Q = np.where(mask, values - R, 0.0)
        
        candidateError = (values < 0).astype(np.int8)
        calculateSyndrome = H_sparse.dot(candidateError) % 2
        
        if np.array_equal(calculateSyndrome, syndrome):
            return candidateError, True, values, currentIter
    
    return candidateError, False, values, currentIter


def performBeliefPropagation_Symmetric(H, syndrome, initialBelief, maxIter=50, alpha=1.0, damping=0.8, clip_llr=20.0, alpha_estimation=False):
    """
    BP with Damping and Symmetric Clipping (Numba-accelerated).
    
    :param damping: Factor (0.0 to 1.0). 1.0 = Standard BP. 0.8 = Recommended for QEC.
    :param clip_llr: Maximum magnitude for LLRs.
    """
    H = np.asarray(H, dtype=np.float64)
    syndrome = np.asarray(syndrome, dtype=np.int8)
    initialBelief = np.asarray(initialBelief, dtype=np.float64)
    
    H_sparse = csr_matrix(H)
    mask = H != 0
    syndrome_sign = (1 - 2 * syndrome).astype(np.float64).reshape(-1, 1)
    
    Q = np.where(mask, initialBelief, 0.0)
    Q_old = Q.copy()
    
    CLIP_CHECK_VAL = 0.9999999
    m, n = H.shape
    candidateError = np.zeros(n, dtype=np.int8)
    
    for currentIter in range(maxIter):
        # Use JIT-compiled core
        R = _bp_core(H, Q, syndrome_sign, mask, CLIP_CHECK_VAL)

        if alpha_estimation and currentIter == 10:
            return np.zeros(n, dtype=np.int8), False, R, 0

        R_scaled = R * alpha
        
        R_sum = np.sum(R_scaled, axis=0)
        values = R_sum + initialBelief
        
        Q_new = np.where(mask, values - R_scaled, 0.0)
        
        Q = damping * Q_new + (1 - damping) * Q_old
        Q = np.clip(Q, -clip_llr, clip_llr)
        Q_old = Q.copy()
        
        candidateError = (values < 0).astype(np.int8)
        calculateSyndrome = H_sparse.dot(candidateError) % 2
        
        if np.array_equal(calculateSyndrome, syndrome) and not alpha_estimation:
            return candidateError, True, values, currentIter
    
    return candidateError, False, values, currentIter


def performOSD_enhanced(H, syndrome, llr, hard, order=0, max_combinations=None):
    """OSD decoder with Numba-accelerated internals."""
    _, n = H.shape
    
    current_syndrome = (hard @ H.T) % 2
    residual_syndrome = (syndrome + current_syndrome) % 2
    
    llr_abs = np.abs(llr)
    ordering = np.argsort(llr_abs)
    H_permuted = H[:, ordering].astype(np.int64)
    
    # Use JIT-compiled GF2 elimination
    H_work = H_permuted.copy()
    s_work = residual_syndrome.copy().astype(np.int64)
    _, s_reduced, pivot_rows, pivot_cols = _gf2_elimination_numba(H_work, s_work)
    
    e_permuted = np.zeros(n, dtype=np.int64)
    for r, c in zip(pivot_rows, pivot_cols):
        e_permuted[c] = s_reduced[r]
    
    e_correction = np.zeros(n, dtype=np.int64)
    e_correction[ordering] = e_permuted
    osd0_solution = (hard + e_correction) % 2
    
    osd0_syndrome = (osd0_solution @ H.T) % 2
    if np.all(osd0_syndrome == syndrome):
        return osd0_solution
    
    if order == 0:
        return osd0_solution
    
    pivot_set = set(pivot_cols)
    non_pivot_positions = [i for i in range(n) if i not in pivot_set]
    
    if len(non_pivot_positions) == 0:
        return osd0_solution
    
    non_pivot_llr = llr_abs[ordering[non_pivot_positions]]
    non_pivot_sorted_idx = np.argsort(non_pivot_llr)
    non_pivot_positions = [non_pivot_positions[i] for i in non_pivot_sorted_idx]
    
    num_test_positions = min(len(non_pivot_positions), order + 10)
    test_positions = non_pivot_positions[:num_test_positions]
    
    best_solution = osd0_solution.copy()
    osd0_syn_weight = np.sum(osd0_syndrome != syndrome)
    best_metric = _compute_metric_numba(osd0_solution.astype(np.float64), llr_abs, osd0_syn_weight)
    found_valid = np.all(osd0_syndrome == syndrome)
    
    combinations_tested = 0
    for w in range(1, min(order + 1, len(test_positions) + 1)):
        if max_combinations and combinations_tested >= max_combinations:
            break
        
        for flip_positions in combinations(test_positions, w):
            if max_combinations and combinations_tested >= max_combinations:
                break
            
            e_test = e_permuted.copy()
            for pos in flip_positions:
                e_test[pos] ^= 1
            
            # Use JIT-compiled recompute
            e_test_full = _recompute_solution_numba(
                H_permuted, s_reduced, e_test, pivot_rows, pivot_cols
            )
            
            e_test_correction = np.zeros(n, dtype=np.int64)
            e_test_correction[ordering] = e_test_full
            test_solution = (hard + e_test_correction) % 2
            
            test_syndrome = (test_solution @ H.T) % 2
            is_valid = np.all(test_syndrome == syndrome)
            
            if is_valid and not found_valid:
                best_solution = test_solution.copy()
                test_syn_weight = 0
                best_metric = _compute_metric_numba(test_solution.astype(np.float64), llr_abs, test_syn_weight)
                found_valid = True
            elif is_valid or not found_valid:
                test_syn_weight = np.sum(test_syndrome != syndrome)
                test_metric = _compute_metric_numba(test_solution.astype(np.float64), llr_abs, test_syn_weight)
                if test_metric < best_metric:
                    best_solution = test_solution.copy()
                    best_metric = test_metric
            
            combinations_tested += 1
    
    return best_solution


def recompute_solution(H_permuted, s_reduced, e_permuted, pivots):
    """Python wrapper for compatibility - calls JIT version internally."""
    pivot_rows, pivot_cols = pivots
    return _recompute_solution_numba(
        H_permuted.astype(np.int64),
        s_reduced.astype(np.int64),
        e_permuted.astype(np.int64),
        np.array(pivot_rows, dtype=np.int64),
        np.array(pivot_cols, dtype=np.int64)
    )


def compute_metric(solution, llr, H, target_syndrome):
    """Python wrapper for compatibility."""
    syndrome = (solution @ H.T) % 2
    syndrome_weight = np.sum(syndrome != target_syndrome)
    return _compute_metric_numba(solution.astype(np.float64), np.abs(llr), syndrome_weight)


def gf2_elimination(H, s):
    """Python wrapper for GF2 elimination - calls JIT version internally."""
    A = H.copy().astype(np.int64)
    b = s.copy().astype(np.int64)
    A_out, b_out, pivot_rows, pivot_cols = _gf2_elimination_numba(A, b)
    return A_out, b_out, (list(pivot_rows), list(pivot_cols))