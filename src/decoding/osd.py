from itertools import combinations
import numpy as np
from .kernels import gf2_elimination_packed, compute_metric, recompute_solution

def performOSD_enhanced(H, syndrome, llr, hard, order=0, max_combinations=None):
    """OSD decoder with Numba-accelerated internals."""
    _, n = H.shape
    current_syndrome = (hard @ H.T) % 2
    residual_syndrome = (syndrome + current_syndrome) % 2
    
    llr_abs = np.abs(llr)
    ordering = np.argsort(llr_abs)
    H_permuted = H[:, ordering].astype(np.int64)
    
    H_work = H_permuted.copy()
    s_work = residual_syndrome.copy().astype(np.int64)
    _, s_reduced, pivot_rows, pivot_cols = gf2_elimination_packed(H_work, s_work)
    
    e_permuted = np.zeros(n, dtype=np.int64)
    for r, c in zip(pivot_rows, pivot_cols):
        e_permuted[c] = s_reduced[r]
    
    e_correction = np.zeros(n, dtype=np.int64)
    e_correction[ordering] = e_permuted
    osd0_solution = (hard + e_correction) % 2
    
    osd0_syndrome = (osd0_solution @ H.T) % 2
    if np.all(osd0_syndrome == syndrome) or order == 0:
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
    best_metric = compute_metric(osd0_solution.astype(np.float64), llr_abs, np.sum(osd0_syndrome != syndrome))
    found_valid = np.all(osd0_syndrome == syndrome)
    
    combinations_tested = 0
    for w in range(1, min(order + 1, len(test_positions) + 1)):
        if max_combinations and combinations_tested >= max_combinations: break
        for flip_positions in combinations(test_positions, w):
            if max_combinations and combinations_tested >= max_combinations: break
            
            e_test = e_permuted.copy()
            for pos in flip_positions: e_test[pos] ^= 1
            
            e_test_full = recompute_solution(H_permuted, s_reduced, e_test, pivot_rows, pivot_cols)
            e_test_correction = np.zeros(n, dtype=np.int64)
            e_test_correction[ordering] = e_test_full
            test_solution = (hard + e_test_correction) % 2
            test_syndrome = (test_solution @ H.T) % 2
            is_valid = np.all(test_syndrome == syndrome)
            
            if is_valid:
                test_metric = compute_metric(test_solution.astype(np.float64), llr_abs, 0)
                if not found_valid or test_metric < best_metric:
                    best_solution = test_solution.copy()
                    best_metric = test_metric
                    found_valid = True
            elif not found_valid:
                test_metric = compute_metric(test_solution.astype(np.float64), llr_abs, np.sum(test_syndrome != syndrome))
                if test_metric < best_metric:
                    best_solution = test_solution.copy()
                    best_metric = test_metric
            
            combinations_tested += 1
            
    return best_solution
