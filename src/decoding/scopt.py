import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.sparse import csr_matrix, isspmatrix_csr

from .kernels import minsum_core_sparse



def estimate_scopt_beta(
    code,
    error_rate,
    trials=10000,
    bins=50,
    alpha=1.0,
    alpha_mode="dynamical",
    maxIter=50,
    damping=1.0,
    clip_llr=20.0,
    rng=None,
    plot_dir=None,
    plot_prefix=None,
    llrs=None
):
    """
    This method is really similar to estimate_alpha_alvarado but adapted for SCOPT.
    
    SCOPT perform the same procedure but instead of estimating inside the Min-Sum decoder it works
    with the LLRs at the end on the algorithm, before passing them to the OSD stage.

    Args:
        code: Parity-check matrix or CSR matrix.
        error_rate: Physical error rate.
        trials: Monte Carlo trials.
        bins: Histogram bins.
        alpha: Alpha value or per-iteration alpha sequence.
        alpha_mode: "dynamical", "alvarado", or "alvarado-autoregressive".
        maxIter: Maximum BP/Min-Sum iterations.
        damping: Damping factor used in message updates.
        clip_llr: LLR clipping value.
        rng: Optional RNG.
    """
    
    if error_rate <= 0 or error_rate >= 0.5:
        raise ValueError("error_rate must be in (0, 0.5)")

    if rng is None:
        rng = np.random.default_rng()

    if isspmatrix_csr(code):
        H_csr = code
    else:
        H_csr = csr_matrix(code)

    if maxIter <= 0:
        raise ValueError("maxIter must be > 0")

    if alpha_mode not in {"dynamical", "alvarado", "alvarado-autoregressive"}:
        raise ValueError(f"Unsupported alpha_mode: {alpha_mode}")

    if alpha_mode == "alvarado-autoregressive":
        alpha_seq = np.asarray(alpha, dtype=np.float64)
        if alpha_seq.ndim != 1 or alpha_seq.size == 0:
            raise ValueError("alpha must be a non-empty 1D sequence for alvarado-autoregressive")
    else:
        alpha_seq = None

    m, n = H_csr.shape
    H_data = H_csr.data.astype(np.int8, copy=False)
    H_indices = H_csr.indices.astype(np.int32, copy=False)
    H_indptr = H_csr.indptr.astype(np.int32, copy=False)

    # initial_llr = np.log((1.0 - error_rate) / error_rate)
    # initialBeliefs = np.full(n, initial_llr, dtype=np.float64)\
    
    initialBeliefs = llrs

    final_0 = []
    final_1 = []

    for _ in range(trials):
        error = (rng.random(n) < error_rate).astype(np.int8)
        syndrome = (H_csr.dot(error) % 2).astype(np.int8)
        syndrome_sign = (1.0 - 2.0 * syndrome).astype(np.float64)

        Q_flat = initialBeliefs[H_indices].copy()
        Q_flat_old = Q_flat.copy()

        for currentIter in range(maxIter):
            if alpha_mode == "dynamical":
                current_alpha = 1.0 - 2.0 ** (-(currentIter + 1))
            elif alpha_mode == "alvarado-autoregressive":
                current_alpha = alpha_seq[currentIter] if currentIter < alpha_seq.size else alpha_seq[-1]
            else:
                current_alpha = float(alpha)

            R_flat, R_sum = minsum_core_sparse(
                H_data, H_indices, H_indptr, Q_flat, syndrome_sign, current_alpha, m, n
            )

            values = R_sum + initialBeliefs

            for pos in range(len(Q_flat)):
                col = H_indices[pos]
                q_new = values[col] - R_flat[pos]
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

            candidateError = (values < 0).astype(np.int8)
            calculateSyndrome = H_csr.dot(candidateError) % 2
            if np.array_equal(calculateSyndrome, syndrome):
                break

        bit_values = error
        final_0.append(values[bit_values == 0])
        final_1.append(values[bit_values == 1])

    if not final_0 or not final_1:
        raise ValueError("Insufficient samples for beta estimation")

    final_0 = np.concatenate(final_0)
    final_1 = np.concatenate(final_1)

    final_0 = final_0[np.isfinite(final_0)]
    final_1 = final_1[np.isfinite(final_1)]

    if final_0.size == 0 or final_1.size == 0:
        raise ValueError("No finite samples for beta estimation")

    # calculate the histograms and then estimate beta
    min_val = min(final_0.min(), final_1.min())
    max_val = max(final_0.max(), final_1.max())
    hist_0, bin_edges = np.histogram(final_0, bins=bins, range=(min_val, max_val), density=True)
    hist_1, _ = np.histogram(final_1, bins=bins, range=(min_val, max_val), density=True)
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    def linear_model(x, beta):
        return beta * x
    
    valid = (hist_0 > 0) & (hist_1 > 0)
    log_ratio = np.log(hist_1[valid] / hist_0[valid])
    x_vals = bin_centers[valid]
    popt, _ = curve_fit(linear_model, x_vals, log_ratio)
    beta = popt[0]

    fit_vals = linear_model(x_vals, beta)
    ss_res = np.sum((log_ratio - fit_vals) ** 2)
    ss_tot = np.sum((log_ratio - np.mean(log_ratio)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else np.nan)

    if plot_dir is not None:
        prefix = plot_prefix or f"beta_p{error_rate:.6g}"
        plot_path = f"{plot_dir}/{prefix}_beta_fit.png"
        plt.figure(figsize=(6, 4))
        plt.scatter(x_vals, log_ratio, s=10, alpha=0.7, label="samples")
        plt.plot(x_vals, fit_vals, color="#64B791", label=f"fit (R^2={r2:.3f})")
        plt.xlabel("LLR")
        plt.ylabel("log(f1/f0)")
        plt.title(f"SCOPT beta fit (p={error_rate:.6g})")
        plt.grid(True, ls="-", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()

    return beta, r2