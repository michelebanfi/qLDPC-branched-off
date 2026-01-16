import numpy as np
from numba import njit

@njit(cache=True, nogil=True)
def gf2_elimination(A, b):
    """JIT-compiled GF2 Gaussian elimination."""
    m, n = A.shape
    pivot_rows = np.empty(min(m, n), dtype=np.int64)
    pivot_cols = np.empty(min(m, n), dtype=np.int64)
    num_pivots = 0
    row = 0
    for col in range(n):
        if row >= m: break
        pivot_row = -1
        for r in range(row, m):
            if A[r, col] == 1:
                pivot_row = r
                break
        if pivot_row == -1: continue
        if pivot_row != row:
            for j in range(n):
                A[row, j], A[pivot_row, j] = A[pivot_row, j], A[row, j]
            b[row], b[pivot_row] = b[pivot_row], b[row]
        pivot_rows[num_pivots] = row
        pivot_cols[num_pivots] = col
        num_pivots += 1
        for r in range(m):
            if r != row and A[r, col] == 1:
                for j in range(n):
                    A[r, j] ^= A[row, j]
                b[r] ^= b[row]
        row += 1
    return A, b, pivot_rows[:num_pivots], pivot_cols[:num_pivots]

@njit(cache=True, nogil=True)
def minsum_core(H, Q, syndrome_sign, mask, alpha):
    """Core Min-Sum message computation."""
    m, n = H.shape
    R = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        sign_prod = syndrome_sign[i, 0]
        min1 = np.inf
        min2 = np.inf
        min1_idx = -1
        for j in range(n):
            if mask[i, j]:
                val = Q[i, j]
                sign_prod *= (1.0 if val >= 0 else -1.0)
                abs_val = abs(val)
                if abs_val < min1:
                    min2 = min1
                    min1 = abs_val
                    min1_idx = j
                elif abs_val < min2:
                    min2 = abs_val
        for j in range(n):
            if mask[i, j]:
                val = Q[i, j]
                sign_j = 1.0 if val >= 0 else -1.0
                row_sign_excl_j = sign_prod * sign_j
                mag = min2 if j == min1_idx else min1
                R[i, j] = alpha * row_sign_excl_j * mag
    return R

@njit(cache=True, nogil=True)
def minsum_core_sparse(H_data, H_indices, H_indptr, Q_flat, syndrome_sign, alpha, m, n):
    """Sparse-aware Min-Sum message computation."""
    nnz = len(H_data)
    R_flat = np.zeros(nnz, dtype=np.float64)
    R_sum = np.zeros(n, dtype=np.float64)
    for i in range(m):
        row_start = H_indptr[i]
        row_end = H_indptr[i + 1]
        if row_start == row_end: continue
        sign_prod = syndrome_sign[i]
        min1, min2 = np.inf, np.inf
        min1_pos = -1
        for pos in range(row_start, row_end):
            val = Q_flat[pos]
            sign_prod *= (1.0 if val >= 0 else -1.0)
            abs_val = abs(val)
            if abs_val < min1:
                min2 = min1
                min1 = abs_val
                min1_pos = pos
            elif abs_val < min2:
                min2 = abs_val
        for pos in range(row_start, row_end):
            val = Q_flat[pos]
            sign_j = 1.0 if val >= 0 else -1.0
            row_sign_excl_j = sign_prod * sign_j
            mag = min2 if pos == min1_pos else min1
            msg = alpha * row_sign_excl_j * mag
            R_flat[pos] = msg
            R_sum[H_indices[pos]] += msg
    return R_flat, R_sum

@njit(cache=True, nogil=True)
def bp_core(H, Q, syndrome_sign, mask, clip_val):
    """Core BP tanh-based message computation."""
    m, n = H.shape
    R = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        row_prod = 1.0
        for j in range(n):
            if mask[i, j]:
                tanh_val = np.tanh(Q[i, j] * 0.5)
                if abs(tanh_val) < 1e-15:
                    tanh_val = 1e-15 if tanh_val >= 0 else -1e-15
                row_prod *= tanh_val
        for j in range(n):
            if mask[i, j]:
                tanh_val = np.tanh(Q[i, j] * 0.5)
                if abs(tanh_val) < 1e-15:
                    tanh_val = 1e-15 if tanh_val >= 0 else -1e-15
                prod_others = row_prod / tanh_val
                prod_clipped = prod_others * syndrome_sign[i, 0]
                prod_clipped = np.clip(prod_clipped, -clip_val, clip_val)
                R[i, j] = 2.0 * np.arctanh(prod_clipped)
    return R

@njit(cache=True, nogil=True)
def compute_metric(solution, llr_abs, syndrome_weight):
    """JIT-compiled metric computation for OSD."""
    if syndrome_weight > 0:
        metric = 1e10 + syndrome_weight * 1e8
    else:
        metric = 0.0
    for i in range(len(solution)):
        metric += solution[i] * llr_abs[i]
    return metric

@njit(cache=True, nogil=True)
def recompute_solution(H_permuted, s_reduced, e_permuted, pivot_rows, pivot_cols):
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
def syndrome_check(H_data, H_indices, H_indptr, candidate, m):
    """JIT-compiled sparse syndrome computation."""
    syndrome = np.zeros(m, dtype=np.int8)
    for i in range(m):
        s = 0
        for idx in range(H_indptr[i], H_indptr[i + 1]):
            s ^= candidate[H_indices[idx]]
        syndrome[i] = s
    return syndrome
