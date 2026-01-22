import numpy as np
from scipy.optimize import curve_fit
from scipy.sparse import csr_matrix, isspmatrix_csr

from .kernels import minsum_core_sparse



def estimate_scopt_beta(
    code, error_rate, trials=10000, bins=50, alpha=1.0, rng=None
):
    """
    This method is really similar to estimate_alpha_alvarado but adapted for SCOPT.
    
    SCOPT perform the same procedure but instead of estimating inside the Min-Sum decoder it works
    with the LLRs at the end on the algorithm, before passing them to the OSD stage.

    Args:
        code (_type_): _description_
        error_rate (_type_): _description_
        trials (int, optional): _description_. Defaults to 10000.
        bins (int, optional): _description_. Defaults to 50.
        rng (_type_, optional): _description_. Defaults to None.
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

    final_0 = []
    final_1 = []

    for _ in range(trials):
        error = (rng.random(n) < error_rate).astype(np.int8)
        syndrome = (H_csr.dot(error) % 2).astype(np.int8)
        syndrome_sign = (1.0 - 2.0 * syndrome).astype(np.float64)

        Q_flat = initialBeliefs[H_indices].copy()
        R_flat, _ = minsum_core_sparse(
            H_data, H_indices, H_indptr, Q_flat, syndrome_sign, alpha, m, n
        )

        # Here we collect the final LLRs after one Min-Sum iteration
        final_llrs = R_flat + initialBeliefs[edge_cols]
        bit_values = error[edge_cols]
        final_0.append(final_llrs[bit_values == 0])
        final_1.append(final_llrs[bit_values == 1])

    if not final_0 or not final_1:
        raise ValueError("Insufficient samples for beta estimation")

    final_0 = np.concatenate(final_0)
    final_1 = np.concatenate(final_1)
    
    # calculate the histograms and then estimate beta
    hist_0, bin_edges = np.histogram(final_0, bins=bins, range=(final_0.min(), final_0.max()), density=True)
    hist_1, _ = np.histogram(final_1, bins=bins, range=(final_1.min(), final_1.max()), density=True)
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    def linear_model(x, beta):
        return beta * x
    
    valid = (hist_0 > 0) & (hist_1 > 0)
    log_ratio = np.log(hist_1[valid] / hist_0[valid])
    popt, _ = curve_fit(linear_model, bin_centers[valid], log_ratio)
    return popt[0]