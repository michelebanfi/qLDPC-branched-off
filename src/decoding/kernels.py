import numba
import numpy as np
from numba import njit

@njit(cache=True, nogil=True, fastmath=True)
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

def _pack_rows_uint64(A):
    """Pack rows of a binary matrix into uint64 words using NumPy packbits."""
    m, n = A.shape
    A_uint8 = np.ascontiguousarray(A, dtype=np.uint8)
    packed_bytes = np.packbits(A_uint8, axis=1, bitorder="little")
    nbytes = packed_bytes.shape[1]
    pad = (-nbytes) % 8
    if pad:
        packed_bytes = np.pad(packed_bytes, ((0, 0), (0, pad)), mode="constant")
    A_packed = packed_bytes.view(np.uint64)
    return A_packed, n

@njit(cache=True, nogil=True, fastmath=True)
def gf2_elimination_packed_core(A_packed, b, n):
    """
    GF(2) Gaussian elimination on a bit-packed matrix.

    Expects A packed into uint64 words (one bit per column) and performs
    word-wise XOR row operations. Returns packed matrix, updated b, and pivots.
    """
    m = A_packed.shape[0]
    nwords = A_packed.shape[1]

    pivot_rows = np.empty(min(m, n), dtype=np.int64)
    pivot_cols = np.empty(min(m, n), dtype=np.int64)
    num_pivots = 0
    row = 0

    for col in range(n):
        if row >= m:
            break

        word_idx = col >> 6
        bit_mask = np.uint64(1) << np.uint64(col & 63)

        pivot_row = -1
        for r in range(row, m):
            if (A_packed[r, word_idx] & bit_mask) != 0:
                pivot_row = r
                break
        if pivot_row == -1:
            continue

        if pivot_row != row:
            for k in range(nwords):
                A_packed[row, k], A_packed[pivot_row, k] = A_packed[pivot_row, k], A_packed[row, k]
            b[row], b[pivot_row] = b[pivot_row], b[row]

        pivot_rows[num_pivots] = row
        pivot_cols[num_pivots] = col
        num_pivots += 1

        for r in range(m):
            if r != row and (A_packed[r, word_idx] & bit_mask) != 0:
                for k in range(nwords):
                    A_packed[r, k] ^= A_packed[row, k]
                b[r] ^= b[row]

        row += 1

    return A_packed, b, pivot_rows[:num_pivots], pivot_cols[:num_pivots]

def gf2_elimination_packed(A, b):
    """
    GF(2) Gaussian elimination using NumPy packbits + packed core.

    Packs rows into uint64 with NumPy for faster packing, then runs the
    compiled packed elimination core. Returns packed matrix, updated b, and pivots.
    """
    A_packed, n = _pack_rows_uint64(A)
    return gf2_elimination_packed_core(A_packed, b, n)

@njit(cache=True, nogil=True, fastmath=True)
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

@njit(cache=True, nogil=True, fastmath=True)
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

@njit(cache=True, nogil=True, fastmath=True)
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

@njit(cache=True, nogil=True, fastmath=True)
def compute_metric(solution, llr_abs, syndrome_weight):
    """JIT-compiled metric computation for OSD."""
    if syndrome_weight > 0:
        metric = 1e10 + syndrome_weight * 1e8
    else:
        metric = 0.0
    for i in range(len(solution)):
        metric += solution[i] * llr_abs[i]
    return metric

@njit(cache=True, nogil=True, fastmath=True)
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

@njit(cache=True, nogil=True, fastmath=True)
def syndrome_check(H_data, H_indices, H_indptr, candidate, m):
    """JIT-compiled sparse syndrome computation."""
    syndrome = np.zeros(m, dtype=np.int8)
    for i in range(m):
        s = 0
        for idx in range(H_indptr[i], H_indptr[i + 1]):
            s ^= candidate[H_indices[idx]]
        syndrome[i] = s
    return syndrome


@njit(cache=True, nogil=True, fastmath=True, parallel=True)
def minsum_decoder_full(
    H_indices, H_indptr, 
    syndrome, initialBelief,
    maxIter, use_dynamic_alpha, alpha_val,
    damping, clip_llr
):
    """
    Fully JIT-compiled Min-Sum decoder.
    
    Runs the entire decoding loop in compiled code to eliminate Python overhead.
    """
    m = len(H_indptr) - 1
    n = len(initialBelief)
    nnz = len(H_indices)
    
    syndrome_sign = np.empty(m, dtype=np.float64)
    for i in numba.prange(m):
        syndrome_sign[i] = 1.0 - 2.0 * syndrome[i]
    
    # Pre-allocate all buffers
    Q_flat = np.empty(nnz, dtype=np.float64)
    Q_flat_old = np.empty(nnz, dtype=np.float64)
    R_flat = np.empty(nnz, dtype=np.float64)
    R_sum = np.zeros(n, dtype=np.float64)
    values = np.empty(n, dtype=np.float64)
    candidateError = np.zeros(n, dtype=np.int8)
    
    # Initialize Q messages from initial beliefs
    for idx in range(nnz):
        Q_flat[idx] = initialBelief[H_indices[idx]]
        Q_flat_old[idx] = Q_flat[idx]
    
    final_iter = maxIter - 1
    converged = False
    
    for currentIter in range(maxIter):
        # Compute alpha
        if use_dynamic_alpha:
            current_alpha = 1.0 - 2.0 ** (-(currentIter + 1))
        else:
            current_alpha = alpha_val
        
        # Reset R_sum
        for j in numba.prange(n):
            R_sum[j] = 0.0
        
        # Min-Sum core: compute R messages
        for i in range(m):
            row_start = H_indptr[i]
            row_end = H_indptr[i + 1]
            if row_start == row_end:
                continue
            
            # First pass: compute sign product and find min1, min2
            sign_prod = syndrome_sign[i]
            min1 = np.inf
            min2 = np.inf
            min1_pos = -1
            
            for pos in range(row_start, row_end):
                val = Q_flat[pos]
                if val >= 0:
                    sign_prod *= 1.0
                else:
                    sign_prod *= -1.0
                abs_val = abs(val)
                if abs_val < min1:
                    min2 = min1
                    min1 = abs_val
                    min1_pos = pos
                elif abs_val < min2:
                    min2 = abs_val
            
            # Second pass: compute R messages
            for pos in range(row_start, row_end):
                val = Q_flat[pos]
                sign_j = 1.0 if val >= 0 else -1.0
                row_sign_excl_j = sign_prod * sign_j
                mag = min2 if pos == min1_pos else min1
                msg = current_alpha * row_sign_excl_j * mag
                R_flat[pos] = msg
                R_sum[H_indices[pos]] += msg
        
        # Update values
        for j in numba.prange(n):
            values[j] = R_sum[j] + initialBelief[j]
        
        # Update Q messages with damping and clipping
        for idx in range(nnz):
            col = H_indices[idx]
            q_new = values[col] - R_flat[idx]
            
            # Handle NaN/Inf
            if q_new != q_new:  # NaN check
                q_new = 0.0
            elif q_new > clip_llr:
                q_new = clip_llr
            elif q_new < -clip_llr:
                q_new = -clip_llr
            
            # Damping
            q_damped = damping * q_new + (1.0 - damping) * Q_flat_old[idx]
            
            # Clip again
            if q_damped > clip_llr:
                q_damped = clip_llr
            elif q_damped < -clip_llr:
                q_damped = -clip_llr
            
            Q_flat[idx] = q_damped
            Q_flat_old[idx] = q_damped
        
        # Compute candidate error and check syndrome
        for j in numba.prange(n):
            candidateError[j] = 1 if values[j] < 0 else 0
        
        # Check syndrome
        syndrome_ok = True
        for i in range(m):
            s = 0
            for idx in range(H_indptr[i], H_indptr[i + 1]):
                s ^= candidateError[H_indices[idx]]
            if s != syndrome[i]:
                syndrome_ok = False
                break
        
        if syndrome_ok:
            final_iter = currentIter
            converged = True
            break
    
    return candidateError, converged, values, final_iter


@njit(cache=True, nogil=True, fastmath=True, parallel=True)
def minsum_decoder_full_autoregressive(
    H_indices, H_indptr,
    syndrome, initialBelief,
    maxIter, alpha_seq, alpha_len,
    damping, clip_llr
):
    """
    Fully JIT-compiled Min-Sum decoder with per-iteration alpha sequence.
    """
    m = len(H_indptr) - 1
    n = len(initialBelief)
    nnz = len(H_indices)

    syndrome_sign = np.empty(m, dtype=np.float64)
    for i in numba.prange(m):
        syndrome_sign[i] = 1.0 - 2.0 * syndrome[i]

    Q_flat = np.empty(nnz, dtype=np.float64)
    Q_flat_old = np.empty(nnz, dtype=np.float64)
    R_flat = np.empty(nnz, dtype=np.float64)
    R_sum = np.zeros(n, dtype=np.float64)
    values = np.empty(n, dtype=np.float64)
    candidateError = np.zeros(n, dtype=np.int8)

    for idx in range(nnz):
        Q_flat[idx] = initialBelief[H_indices[idx]]
        Q_flat_old[idx] = Q_flat[idx]

    final_iter = maxIter - 1
    converged = False

    for currentIter in range(maxIter):
        if currentIter < alpha_len:
            current_alpha = alpha_seq[currentIter]
        else:
            current_alpha = alpha_seq[alpha_len - 1]

        for j in numba.prange(n):
            R_sum[j] = 0.0

        for i in range(m):
            row_start = H_indptr[i]
            row_end = H_indptr[i + 1]
            if row_start == row_end:
                continue

            sign_prod = syndrome_sign[i]
            min1 = np.inf
            min2 = np.inf
            min1_pos = -1

            for pos in range(row_start, row_end):
                val = Q_flat[pos]
                if val >= 0:
                    sign_prod *= 1.0
                else:
                    sign_prod *= -1.0
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
                msg = current_alpha * row_sign_excl_j * mag
                R_flat[pos] = msg
                R_sum[H_indices[pos]] += msg

        for j in numba.prange(n):
            values[j] = R_sum[j] + initialBelief[j]

        for idx in range(nnz):
            col = H_indices[idx]
            q_new = values[col] - R_flat[idx]

            if q_new != q_new:
                q_new = 0.0
            elif q_new > clip_llr:
                q_new = clip_llr
            elif q_new < -clip_llr:
                q_new = -clip_llr

            q_damped = damping * q_new + (1.0 - damping) * Q_flat_old[idx]

            if q_damped > clip_llr:
                q_damped = clip_llr
            elif q_damped < -clip_llr:
                q_damped = -clip_llr

            Q_flat[idx] = q_damped
            Q_flat_old[idx] = q_damped

        for j in numba.prange(n):
            candidateError[j] = 1 if values[j] < 0 else 0

        syndrome_ok = True
        for i in range(m):
            s = 0
            for idx in range(H_indptr[i], H_indptr[i + 1]):
                s ^= candidateError[H_indices[idx]]
            if s != syndrome[i]:
                syndrome_ok = False
                break

        if syndrome_ok:
            final_iter = currentIter
            converged = True
            break

    return candidateError, converged, values, final_iter
