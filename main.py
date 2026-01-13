import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from decoding import performBeliefPropagation_Symmetric
from decoding import performMinSum_Symmetric
from decoding import performOSD_enhanced

def estimate_alpha_from_code(code, trials=5000, error_rate=0.05, maxIter=50, bins=50):

    true_0 = []
    true_1 = []
    
    edge_rows, edge_cols = np.nonzero(code)
    
    for _trial in range(trials):
        n = len(code[0])
        initialBeliefs = [np.log((1 - error_rate) / error_rate)] * n

        error = (np.random.random(n) < error_rate).astype(int)

        syndrome = (error @ code.T) % 2

        # _ , _, R, _ = performBeliefPropagation_Symmetric(
        #     code, syndrome, initialBeliefs, maxIter=maxIter, alpha=1.0, damping=1.0, clip_llr=np.inf, alpha_estimation=True
        # )

        _ , _, R, _ = performMinSum_Symmetric(
            code, syndrome, initialBeliefs, maxIter=maxIter, alpha=1.0, damping=1.0, clip_llr=np.inf, alpha_estimation=True
        )

        valid_messages = R[edge_rows, edge_cols]

        corresponding_bit_values = error[edge_cols]

        true_0.extend(valid_messages[corresponding_bit_values == 0])
        true_1.extend(valid_messages[corresponding_bit_values == 1])

    true_0 = np.array(true_0)
    true_1 = np.array(true_1)    

    min_val = min(true_0.min(), true_1.min())
    max_val = max(true_0.max(), true_1.max())

    hist_range = (min_val, max_val)

    hist_0, bin_edges = np.histogram(true_0, bins=bins, range=hist_range, density=True)
    hist_1, _ = np.histogram(true_1, bins=bins, range=hist_range, density=True)
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    valid_indices = (hist_0 > 0) & (hist_1 > 0)

    lambdas = bin_centers[valid_indices]
    f_lambdas = np.log(hist_0[valid_indices] / hist_1[valid_indices])
    
    def linear_model(x, alpha):
        return alpha * x
    
    popt, _ = curve_fit(linear_model, lambdas, f_lambdas)
    alpha_opt = popt[0]

    print(f"Estimated alpha for error rate {error_rate}: {alpha_opt}")
    
    return alpha_opt
        

experiment = [
    {
        "code": "[[72, 12, 6]]",
        "name": "72",
        "physicalErrorRates": [0.1, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009],
        "distance": 6,
    },
    {
        "code": "[[90, 8, 10]]",
        "name": "90",
        "physicalErrorRates": [0.1, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
        "distance": 10,
    },
    {
        "code": "[[108, 8, 10]]",
        "name": "108",
        "physicalErrorRates": [0.1, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
        "distance": 10,
    },
    {
        "code": "[[144, 12, 12]]",
        "name": "144",
        "physicalErrorRates": [0.1, 0.06, 0.05, 0.04, 0.03, 0.02],
        "distance": 12,
    },
    {
        "code": "[[288, 12, 18]]",
        "name": "288",
        "physicalErrorRates": [0.1, 0.06, 0.05, 0.04],
        "distance": 18,
    },
]

experiment_dict = {exp["name"]: exp for exp in experiment}

trials = 100

BP_maxIter = 50
OSD_order = 0

np.random.seed(0)

results = {}
llr_data = {}

for exp in experiment:
    code_name = exp["name"]
    oc = np.load(f"codes/{exp['code']}.npz")
    code = oc["Hx"]
    Lx = oc["Lx"]
    n = len(code[0])
    results[code_name] = {}

    llr_data[code_name] = {"true_0": [], "true_1": []}

    for errorRate in exp["physicalErrorRates"]:
        initialBeliefs = [np.log((1 - errorRate) / errorRate)] * n

        logicalError = 0
        OSD_invocations = 0
        degenerateErrors = 0
        weights_found_BP = []
        weights_found_OSD = []
        iterations = []
        
        weights_found_BP_error = []
        weights_found_OSD_error = []

        OSD_invocation_AND_logicalError = 0

        collect_stats = (errorRate == exp["physicalErrorRates"][0])

        alpha_estimate = estimate_alpha_from_code(code, error_rate=errorRate, maxIter=1)

        for _ in tqdm.tqdm(range(trials), desc=f"Code {code_name}, p={errorRate}"):
            error = (np.random.random(n) < errorRate).astype(int)

            syndrome = (error @ code.T) % 2

            # clipping and damping ingnored for now. We need to focus on alpha first.
            # detection, isSyndromeFound, llrs, iteration = performBeliefPropagation_Symmetric(
            #     code, syndrome, initialBeliefs, maxIter=BP_maxIter, alpha=alpha_estimate, damping=1.0, clip_llr=np.inf
            # )

            detection, isSyndromeFound, llrs, iteration = performMinSum_Symmetric(
                code, syndrome, initialBeliefs, maxIter=BP_maxIter, alpha=alpha_estimate, damping=0.7, clip_llr=25
            )

            iterations.append(iteration)

            if collect_stats:
                llr_data[code_name]["true_0"].extend(llrs[error == 0])
                
                llr_data[code_name]["true_1"].extend(llrs[error == 1])

            if not isSyndromeFound:
                detection = performOSD_enhanced(code, syndrome, llrs, detection, order=OSD_order)
                OSD_invocations += 1
                
            residual = (detection + error) % 2
            syndromeLogic = (Lx @ residual) % 2
            
            osd_syndrome_check = (detection @ code.T) % 2
            is_valid_osd = np.array_equal(osd_syndrome_check, syndrome)
            
            # we have a logical error!    
            if np.any(syndromeLogic):
                logicalError += 1
                if not isSyndromeFound:
                    weights_found_OSD_error.append(np.sum(residual))
                    OSD_invocation_AND_logicalError += 1
                if isSyndromeFound:
                    weights_found_BP_error.append(np.sum(residual))
            
            # we do not have a logical errror, but we calculate the weight of the error
            if not np.any(syndromeLogic):
                if not np.array_equal(detection, error):
                    w = np.sum(residual)
                    if not isSyndromeFound: weights_found_OSD.append(w)
                    if isSyndromeFound: weights_found_BP.append(w)

                    if is_valid_osd: degenerateErrors += 1

        logicalErrorRate = logicalError / trials
        OSD_invocationRate = OSD_invocations / trials
        degenerateErrorRate = degenerateErrors / trials
        average_iterations = np.mean(iterations)
        OSD_invocation_AND_logicalErrorRate = OSD_invocation_AND_logicalError / trials
        results[code_name][errorRate] = {
            "logical": logicalErrorRate,
            "osd": OSD_invocationRate,
            "degeneracies": degenerateErrorRate,
            "OSD_invocation_AND_logicalError": OSD_invocation_AND_logicalErrorRate,
            "weights_found_BP": weights_found_BP,
            "weights_found_OSD": weights_found_OSD,
            "weights_found_BP_error": weights_found_BP_error,
            "weights_found_OSD_error": weights_found_OSD_error,
            "average_iterations": average_iterations,
        }
        print(
            f"Code {code_name}, p={errorRate}, Logical Error Rate: {logicalErrorRate}, OSD Invocation Rate: {OSD_invocationRate}"
        )
        
np.savez("simulation_results.npz", results=results)

colors = ["2E72AE", "64B791", "DBA142", "000000", "E17792"]

fig, axes = plt.subplots(5, 1, figsize=(6, 10), sharex=True)
fig.suptitle(f"Monte Carlo trials: {trials}, BP max iterations: {BP_maxIter}, OSD order: {OSD_order} \n The y-axis shows rates calculated over all trials.")
for (code_name, code_results), color in zip(results.items(), colors):
    x = list(code_results.keys())
    axes[0].loglog(
        x,
        [v["logical"] for v in code_results.values()],
        marker="d",
        label=f"Code {code_name}",
        color=f"#{color}",
    )
    axes[1].plot(
        x,
        [v["osd"] for v in code_results.values()],
        marker="o",
        label=f"Code {code_name}",
        color=f"#{color}",
    )
    axes[2].plot(
        x,
        [v["degeneracies"] for v in code_results.values()],
        marker="s",
        label=f"Code {code_name}",
        color=f"#{color}",
    )
    axes[3].plot(
        x,
        [v["OSD_invocation_AND_logicalError"] for v in code_results.values()],
        marker="^",
        label=f"Code {code_name}",
        color=f"#{color}",
    )
    axes[4].plot(
        x,
        [v["average_iterations"] for v in code_results.values()],
        marker="o",
        label=f"Code {code_name}",
        color=f"#{color}",
    )

axes[0].set_ylabel("Logical Error Rate")
axes[0].grid(True, which="both", ls="--")
axes[0].legend()

axes[1].set_ylabel("OSD Invocation Rate")
axes[1].grid(True, which="both", ls="--")
axes[1].legend()

axes[2].set_ylabel("Degenerate Errors Rate")
axes[2].grid(True, which="both", ls="--")
axes[2].legend()

axes[3].set_ylabel("OSD Invocation & Error")
axes[3].grid(True, which="both", ls="--")
axes[3].legend()

axes[4].set_xlabel("Physical Error Rate")
axes[4].set_ylabel("Average BP Iterations")
axes[4].axhline(y=BP_maxIter, color='k', linestyle='--', label='BP Max Iter')
axes[4].grid(True, which="both", ls="--")
axes[4].legend()

plt.tight_layout()
plt.savefig("plot.png", dpi=300)
