import numpy as np
from scipy.optimize import curve_fit
from scipy.sparse import csr_matrix, isspmatrix_csr

from .kernels import minsum_core_sparse


def estimate_alpha_alvarado(
    code,
    error_rate,
    trials=5000,
    bins=50,
    rng=None,
):
    """
    Estimate the Alvarado alpha using one Min-Sum iteration statistics.

    This follows the procedure from the original prototype:
    - sample errors at the given error rate
    - run one Min-Sum iteration (alpha=1.0)
    - estimate alpha from the log ratio of message histograms
    """
    if error_rate <= 0 or error_rate >= 0.5:
        raise ValueError("error_rate must be in (0, 0.5)")

    if rng is None:
        rng = np.random.default_rng()

    if isspmatrix_csr(code):
        H_csr = code
    else:
        H_csr = csr_matrix(code)

    m, n = H_csr.shape
    H_data = H_csr.data.astype(np.int8, copy=False)
    H_indices = H_csr.indices.astype(np.int32, copy=False)
    H_indptr = H_csr.indptr.astype(np.int32, copy=False)

    edge_cols = H_indices
    initial_llr = np.log((1.0 - error_rate) / error_rate)
    initialBeliefs = np.full(n, initial_llr, dtype=np.float64)

    true_0 = []
    true_1 = []

    for _ in range(trials):
        error = (rng.random(n) < error_rate).astype(np.int8)
        syndrome = (H_csr.dot(error) % 2).astype(np.int8)
        syndrome_sign = (1.0 - 2.0 * syndrome).astype(np.float64)

        Q_flat = initialBeliefs[H_indices].copy()
        R_flat, _ = minsum_core_sparse(
            H_data, H_indices, H_indptr, Q_flat, syndrome_sign, 1.0, m, n
        )

        bit_values = error[edge_cols]
        true_0.append(R_flat[bit_values == 0])
        true_1.append(R_flat[bit_values == 1])

    if not true_0 or not true_1:
        raise ValueError("Insufficient samples for alpha estimation")

    true_0 = np.concatenate(true_0)
    true_1 = np.concatenate(true_1)

    min_val = min(true_0.min(), true_1.min())
    max_val = max(true_0.max(), true_1.max())
    hist_range = (min_val, max_val)

    hist_0, bin_edges = np.histogram(true_0, bins=bins, range=hist_range, density=True)
    hist_1, _ = np.histogram(true_1, bins=bins, range=hist_range, density=True)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    valid_indices = (hist_0 > 0) & (hist_1 > 0)

    if not np.any(valid_indices):
        raise ValueError("No overlapping histogram bins for alpha estimation")

    lambdas = bin_centers[valid_indices]
    f_lambdas = np.log(hist_0[valid_indices] / hist_1[valid_indices])

    def linear_model(x, alpha):
        return alpha * x

    popt, _ = curve_fit(linear_model, lambdas, f_lambdas)
    return popt[0]
