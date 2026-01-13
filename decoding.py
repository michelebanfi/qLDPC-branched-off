import numpy as np
from itertools import combinations
from scipy.sparse import csr_matrix

def performMinSum_Symmetric(H, syndrome, initialBelief, maxIter=50, alpha=1.0, damping=1.0, clip_llr=20.0, alpha_estimation=False):
    """
    Normalized Min-Sum Algorithm.
    
    alpha: Scaling factor (usually 0.6 - 0.9 for Min-Sum).
           alpha=1.0 makes this standard UMP-BP (Min-Sum).
           alpha=0 enables dynamic scaling: alpha = 1.0 - 2^(-iteration)
           (This matches the ldpc library's variable scaling factor method)
    """
    use_dynamic_alpha = (alpha == 0)
    
    H = np.asarray(H, dtype=np.float64)
    syndrome = np.asarray(syndrome, dtype=np.int8)
    initialBelief = np.asarray(initialBelief, dtype=np.float64)
    
    H_sparse = csr_matrix(H)
    mask = H != 0
    syndrome_sign = (1 - 2 * syndrome).reshape(-1, 1)
    
    Q = np.where(mask, initialBelief, 0)
    Q_old = Q.copy()
    
    R = np.zeros_like(Q)
    
    for currentIter in range(maxIter):
        # Dynamic alpha: increases from 0.5 towards 1.0 as iterations progress
        if use_dynamic_alpha:
            current_alpha = 1.0 - 2.0 ** (-(currentIter + 1))
        else:
            current_alpha = alpha
        
        sign_Q = np.sign(Q)
    
        sign_Q = np.where(sign_Q == 0, 1.0, sign_Q)
        sign_Q = np.where(mask, sign_Q, 1.0)
        
        row_sign_prod = np.prod(sign_Q, axis=1, keepdims=True)

        r_signs = row_sign_prod * sign_Q
        
        abs_Q = np.abs(Q)

        abs_Q_masked = np.where(mask, abs_Q, np.inf)
        
        min1_idx = np.argmin(abs_Q_masked, axis=1)
        min1_vals = abs_Q_masked[np.arange(abs_Q.shape[0]), min1_idx]
        
        temp_Q = abs_Q_masked.copy()
        temp_Q[np.arange(abs_Q.shape[0]), min1_idx] = np.inf
        min2_vals = np.min(temp_Q, axis=1)
        
        min1_vals_broad = min1_vals.reshape(-1, 1)
        min2_vals_broad = min2_vals.reshape(-1, 1)
        
        is_min1 = (abs_Q_masked == min1_vals_broad)
        
        magnitudes = np.where(is_min1, min2_vals_broad, min1_vals_broad)
        
        R_new = current_alpha * syndrome_sign * r_signs * magnitudes
        R_new = np.where(mask, R_new, 0)
        
        if alpha_estimation:
            return 0, 0, R_new / current_alpha, 0

        R_sum = np.sum(R_new, axis=0)
        values = R_sum + initialBelief
        Q_new = np.where(mask, values - R_new, 0)
        
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
    Docstring for performBeliefPropagationFast
    
    :param H: Description
    :param syndrome: Description
    :param initialBelief: Description
    :param maxIter: Description
    :return: Description
    :rtype: tuple[Any, Literal[True], Any] | tuple[Any, Literal[False], Any, Any | int]
    """
    
    H = np.asarray(H, dtype=np.float64)
    syndrome = np.asarray(syndrome, dtype=np.int8)
    initialBelief = np.asarray(initialBelief, dtype=np.float64)
    
    H_sparse = csr_matrix(H)
    
    mask = H != 0
    
    syndrome_sign = (1 - 2 * syndrome).reshape(-1, 1)
    
    Q = np.where(mask, initialBelief, 0)
    R = np.zeros_like(Q)
    
    CLIP_VAL = 0.9999999
    
    for currentIter in range(maxIter):
        
        tanh_Q = np.tanh(Q * 0.5)
        tanh_Q = np.where(mask, tanh_Q, 1.0)
        
        row_prod = np.prod(tanh_Q, axis=1, keepdims=True)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            tanh_Q_safe = np.where(np.abs(tanh_Q) < 1e-15, 1e-15, tanh_Q)
            prod_others = row_prod / tanh_Q_safe
        
        prod_clipped = np.clip(prod_others * syndrome_sign, -CLIP_VAL, CLIP_VAL)
        R = np.where(mask, 2.0 * np.arctanh(prod_clipped), 0)
        
        R_sum = np.sum(R, axis=0)
        values = R_sum + initialBelief
        
        Q = np.where(mask, values - R, 0)
        
        candidateError = (values < 0).astype(np.int8)
        calculateSyndrome = H_sparse.dot(candidateError) % 2
        
        if np.array_equal(calculateSyndrome, syndrome):
            return candidateError, True, values, currentIter
    
    return candidateError, False, values, currentIter

def performBeliefPropagation_Symmetric(H, syndrome, initialBelief, maxIter=50, alpha=1.0, damping=0.8, clip_llr=20.0, alpha_estimation=False):
    """
    BP with Damping and Symmetric Clipping to improve LLR distribution shape.
    
    :param damping: Factor (0.0 to 1.0). 1.0 = Standard BP. 0.8 = Recommended for QEC.
                    Helps maintain distribution symmetry by preventing oscillation.
    :param clip_llr: Maximum magnitude for LLRs. Prevents 'runaway' confidence.
    """
    
    H = np.asarray(H, dtype=np.float64)
    syndrome = np.asarray(syndrome, dtype=np.int8)
    initialBelief = np.asarray(initialBelief, dtype=np.float64)
    
    H_sparse = csr_matrix(H)
    mask = H != 0
    syndrome_sign = (1 - 2 * syndrome).reshape(-1, 1)
    
    Q = np.where(mask, initialBelief, 0)
    
    Q_old = Q.copy() 
    
    CLIP_CHECK_VAL = 0.9999999
    
    for currentIter in range(maxIter):
        
        tanh_Q = np.tanh(Q * 0.5)
        tanh_Q = np.where(mask, tanh_Q, 1.0)
        
        row_prod = np.prod(tanh_Q, axis=1, keepdims=True)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            tanh_Q_safe = np.where(np.abs(tanh_Q) < 1e-15, 1e-15, tanh_Q)
            prod_others = row_prod / tanh_Q_safe
        
        prod_clipped = np.clip(prod_others * syndrome_sign, -CLIP_CHECK_VAL, CLIP_CHECK_VAL)
        R = np.where(mask, 2.0 * np.arctanh(prod_clipped), 0)

        if alpha_estimation and currentIter == 10:
            return 0, 0, R, 0

        R_scaled = R * alpha
        
        R_sum = np.sum(R_scaled, axis=0)
        
        values = R_sum + initialBelief
        
        Q_new = np.where(mask, values - R_scaled, 0)
        
        Q = damping * Q_new + (1 - damping) * Q_old

        Q = np.clip(Q, -clip_llr, clip_llr)
        
        Q_old = Q.copy()
        
        candidateError = (values < 0).astype(np.int8)
        calculateSyndrome = H_sparse.dot(candidateError) % 2
        
        if np.array_equal(calculateSyndrome, syndrome) and not alpha_estimation:
            return candidateError, True, values, currentIter
    
    return candidateError, False, values, currentIter

def performOSD_enhanced(H, syndrome, llr, hard, order=0, max_combinations=None):
    
    _, n = H.shape
    
    current_syndrome = (hard @ H.T) % 2
    residual_syndrome = (syndrome + current_syndrome) % 2
    
    llr_abs = np.abs(llr)
    ordering = np.argsort(llr_abs)
    H_permuted = H[:, ordering]
    
    _, s_reduced, pivots = gf2_elimination(H_permuted, residual_syndrome)
    
    e_permuted = np.zeros(n, dtype=int)
    pivot_row, pivot_cols = pivots
    
    for r, c in zip(pivot_row, pivot_cols):
        e_permuted[c] = s_reduced[r]
    
    e_correction = np.zeros(n, dtype=int)
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
    best_metric = compute_metric(osd0_solution, llr, H, syndrome)
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
            
            e_test_full = recompute_solution(H_permuted, s_reduced, e_test, pivots)
            
            e_test_correction = np.zeros(n, dtype=int)
            e_test_correction[ordering] = e_test_full
            test_solution = (hard + e_test_correction) % 2
            
            test_syndrome = (test_solution @ H.T) % 2
            is_valid = np.all(test_syndrome == syndrome)
            
            if is_valid and not found_valid:

                best_solution = test_solution.copy()
                best_metric = compute_metric(test_solution, llr, H, syndrome)
                found_valid = True
            elif is_valid or not found_valid:
                
                test_metric = compute_metric(test_solution, llr, H, syndrome)
                if test_metric < best_metric:
                    best_solution = test_solution.copy()
                    best_metric = test_metric
            
            combinations_tested += 1
    
    return best_solution

def recompute_solution(H_permuted, s_reduced, e_permuted, pivots):
    _, n = H_permuted.shape
    e_full = e_permuted.copy()
    pivot_rows, pivot_cols = pivots
    
    for r, c in zip(pivot_rows, pivot_cols):
        row_contribution = 0
        for col in range(n):
            if col != c and H_permuted[r, col] == 1:
                row_contribution ^= e_full[col]
        
        e_full[c] = s_reduced[r] ^ row_contribution
    
    return e_full

def compute_metric(solution, llr, H, target_syndrome):
    
    syndrome = (solution @ H.T) % 2
    syndrome_weight = np.sum(syndrome != target_syndrome)
    
    if syndrome_weight > 0:
        metric = 1e10 + syndrome_weight * 1e8
    else:
        metric = 0.0
    
    llr_cost = np.sum(solution * np.abs(llr))
    metric += llr_cost
    
    return metric

def gf2_elimination(H, s):

    m, n = H.shape
    A = H.copy()
    b = s.copy()
    
    pivot_rows = []
    pivot_cols = []
    
    row = 0
    for col in range(n):
        
        if row >= m:
            break
        
        pivot_row = -1
        for r in range(row, m):
            if A[r, col] == 1:
                pivot_row = r
                break
        
        if pivot_row == -1:
            continue
        
        if pivot_row != row:
            A[[row, pivot_row]] = A[[pivot_row, row]]
            b[[row, pivot_row]] = b[[pivot_row, row]]
        
        pivot_rows.append(row)
        pivot_cols.append(col)
        
        for r in range(m):
            if r != row and A[r, col] == 1:
                A[r, :] ^= A[row, :]
                b[r] ^= b[row]
        
        row += 1
    
    return A, b, (pivot_rows, pivot_cols)