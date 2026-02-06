import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.sparse import csr_matrix, isspmatrix_csr

from .kernels import minsum_core_sparse


def _estimate_alpha_from_samples(true_0, true_1, bins=50, plot_path=None, title=None):
    true_0 = np.asarray(true_0, dtype=np.float64)
    true_1 = np.asarray(true_1, dtype=np.float64)
    
    # print(f"TRUE 1: {len(true_1)}")
    # print(f"TRUE 0: {len(true_0)}")
    # print(f"BINS: {bins}")
    
    # print(f"TRUE 1 min/max: {true_1.min():.3f}/{true_1.max():.3f}")
    # print(f"TRUE 0 min/max: {true_0.min():.3f}/{true_0.max():.3f}")
    
    # print(f"Unique values in TRUE 1: {len(np.unique(true_1))}")
    # print(f"Unique values in TRUE 0: {len(np.unique(true_0))}")

    true_0 = true_0[np.isfinite(true_0)]
    true_1 = true_1[np.isfinite(true_1)]

    if true_0.size == 0 or true_1.size == 0:
        raise ValueError("No finite samples for alpha estimation")

    min_val = min(true_0.min(), true_1.min())
    max_val = max(true_0.max(), true_1.max())
    hist_range = (min_val, max_val)

    hist_0, bin_edges = np.histogram(true_0, bins=bins, range=hist_range, density=True)
    hist_1, _ = np.histogram(true_1, bins=bins, range=hist_range, density=True)
    
    # show the thistogram, just for debugging
    # plt.figure(figsize=(6, 4))
    # plt.hist(true_0, bins=bins, range=hist_range, density=True, alpha=0.5, label="true 0")
    # plt.hist(true_1, bins=bins, range=hist_range, density=True, alpha=0.5, label="true 1")
    # plt.xlabel("Message value")
    # plt.ylabel("Density")
    # plt.title("Message histograms for alpha estimation")
    # plt.grid(True, ls="-", alpha=0.4)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    valid_indices = (hist_0 > 0) & (hist_1 > 0)

    if not np.any(valid_indices):
        raise ValueError("No overlapping histogram bins for alpha estimation")

    lambdas = bin_centers[valid_indices]
    f_lambdas = np.log(hist_0[valid_indices] / hist_1[valid_indices])

    def linear_model(x, alpha):
        return alpha * x

    popt, _ = curve_fit(linear_model, lambdas, f_lambdas)
    alpha = popt[0]

    fit_vals = linear_model(lambdas, alpha)
    ss_res = np.sum((f_lambdas - fit_vals) ** 2)
    ss_tot = np.sum((f_lambdas - np.mean(f_lambdas)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else np.nan)

    if plot_path is not None:
        plt.figure(figsize=(6, 4))
        plt.scatter(lambdas, f_lambdas, s=10, alpha=0.7, label="samples")
        plt.plot(lambdas, fit_vals, color="#DBA142", label=f"fit (R^2={r2:.3f})")
        plt.xlabel("Lambda")
        plt.ylabel("log(f0/f1)")
        plt.title(title or "Alpha estimation linear fit")
        plt.grid(True, ls="-", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()

    return alpha, r2


def estimate_alpha_alvarado(
    code,
    error_rate,
    trials=5000,
    bins=50,
    rng=None,
    plot_dir=None,
    plot_prefix=None,
    llrs=None
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
    # initial_llr = np.log((1.0 - error_rate) / error_rate)
    # initialBeliefs = np.full(n, initial_llr, dtype=np.float64)
    initialBeliefs = llrs

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

    plot_path = None
    if plot_dir is not None:
        prefix = plot_prefix or f"alvarado_p{error_rate:.6g}"
        plot_path = f"{plot_dir}/{prefix}_alpha_fit.png"

    return _estimate_alpha_from_samples(
        true_0,
        true_1,
        bins=bins,
        plot_path=plot_path,
        title=f"Alvarado alpha fit (p={error_rate:.6g})",
    )


def estimate_alpha_alvarado_autoregressive(
    code,
    error_rate,
    maxIter,
    trials=5000,
    bins=50,
    damping=1.0,
    clip_llr=20.0,
    rng=None,
    plot_dir=None,
    plot_prefix=None,
    llrs=None,
):
    """
    Estimate a per-iteration Alvarado alpha sequence using autoregressive updates.

    For each iteration k, the decoder state is advanced using the previously
    estimated alphas (0..k-1). Alpha_k is then estimated from the unscaled
    messages at iteration k.
    """
    if error_rate <= 0 or error_rate >= 0.5:
        raise ValueError("error_rate must be in (0, 0.5)")
    if maxIter <= 0:
        raise ValueError("maxIter must be > 0")

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
    # initial_llr = np.log((1.0 - error_rate) / error_rate)
    # initialBeliefs = np.full(n, initial_llr, dtype=np.float64)
    initialBeliefs = llrs

    alpha_values = []
    r2_values = []

    for iter_idx in range(maxIter):
        true_0 = []
        true_1 = []

        for _ in range(trials):
            error = (rng.random(n) < error_rate).astype(np.int8)
            syndrome = (H_csr.dot(error) % 2).astype(np.int8)
            syndrome_sign = (1.0 - 2.0 * syndrome).astype(np.float64)

            Q_flat = initialBeliefs[H_indices].copy()
            Q_flat_old = Q_flat.copy()

            # Advance decoder state using previously estimated alphas
            for prev_idx in range(iter_idx):
                prev_alpha = alpha_values[prev_idx]
                R_flat_prev, R_sum_prev = minsum_core_sparse(
                    H_data, H_indices, H_indptr, Q_flat, syndrome_sign, prev_alpha, m, n
                )

                values_prev = R_sum_prev + initialBeliefs

                for pos in range(len(Q_flat)):
                    col = H_indices[pos]
                    q_new = values_prev[col] - R_flat_prev[pos]
                    if q_new != q_new:
                        q_new = 0.0
                    elif q_new > clip_llr:
                        q_new = clip_llr
                    elif q_new < -clip_llr:
                        q_new = -clip_llr

                    q_damped = damping * q_new + (1.0 - damping) * Q_flat_old[pos]
                    if q_damped > clip_llr:
                        q_damped = clip_llr
                    elif q_damped < -clip_llr:
                        q_damped = -clip_llr

                    Q_flat[pos] = q_damped
                    Q_flat_old[pos] = q_damped

            # Collect unscaled messages at current iteration
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

        plot_path = None
        if plot_dir is not None:
            prefix = plot_prefix or f"autoregressive_p{error_rate:.6g}"
            plot_path = f"{plot_dir}/{prefix}_iter{iter_idx + 1}_alpha_fit.png"

        alpha_k, r2_k = _estimate_alpha_from_samples(
            true_0,
            true_1,
            bins=bins,
            plot_path=plot_path,
            title=f"Autoregressive alpha fit (p={error_rate:.6g}, iter={iter_idx + 1})",
        )
        alpha_values.append(float(alpha_k))
        r2_values.append(float(r2_k))

    return np.asarray(alpha_values, dtype=np.float64), np.asarray(r2_values, dtype=np.float64)
