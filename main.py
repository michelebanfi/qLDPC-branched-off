import os
import numpy as np
from datetime import datetime
from rich.console import Console

from src.codes.bb_code import BBCodeCircuit
from src.simulation.engine import run_simulation
from src.noise.builder import build_decoding_matrices
from src.utils.plotting import plot_simulation_results, plot_alpha_comparison, plot_alpha_linearity
from src.utils.caching import compute_cache_key, load_matrices, save_matrices


console = Console()

# Experiment configuration
experiments = [
    {"code": "[[72, 12, 6]]", "name": "72", "physicalErrorRates": [0.006, 0.005, 0.004, 0.003, 0.0018, 0.001, 0.0005, 0.00018], "distance": 6},
    {"code": "[[90, 8, 10]]", "name": "90", "physicalErrorRates": [0.006, 0.005, 0.004, 0.003, 0.0018, 0.001, 0.0005], "distance": 10},
    {"code": "[[108, 8, 10]]", "name": "108", "physicalErrorRates": [0.006, 0.005, 0.004, 0.003, 0.0018, 0.0001, 0.0005], "distance": 10},
    {"code": "[[144, 12, 12]]", "name": "144", "physicalErrorRates": [0.006, 0.005, 0.004, 0.003, 0.0018, 0.0001], "distance": 12},
    {"code": "[[288, 12, 18]]", "name": "288", "physicalErrorRates": [0.006, 0.005, 0.004, 0.0035], "distance": 18},
]

experiments = [
    {"code": "[[72, 12, 6]]", "name": "72", "physicalErrorRates": [0.006, 0.005, 0.004], "distance": 6},
    # {"code": "[[90, 8, 10]]", "name": "90", "physicalErrorRates": [0.006, 0.005, 0.004,], "distance": 10},
    # {"code": "[[108, 8, 10]]", "name": "108", "physicalErrorRates": [0.006, 0.005, 0.004], "distance": 10},
    # {"code": "[[144, 12, 12]]", "name": "144", "physicalErrorRates": [0.006, 0.005, 0.004], "distance": 12},
    # {"code": "[[288, 12, 18]]", "name": "288", "physicalErrorRates": [0.006, 0.005], "distance": 18},
]

experiments = [
    {"code": "[[72, 12, 6]]", "name": "72", "physicalErrorRates": [0.006], "distance": 6},
    # {"code": "[[90, 8, 10]]", "name": "90", "physicalErrorRates": [0.006, 0.005, 0.004,], "distance": 10},
    # {"code": "[[108, 8, 10]]", "name": "108", "physicalErrorRates": [0.006, 0.005, 0.004], "distance": 10},
    # {"code": "[[144, 12, 12]]", "name": "144", "physicalErrorRates": [0.006, 0.005, 0.004], "distance": 12},
    # {"code": "[[288, 12, 18]]", "name": "288", "physicalErrorRates": [0.006, 0.005], "distance": 18},
]

def main():
    target_logical_errors = 30
    max_trials = 20
    maxIter = 20
    osd_order = 2
    num_workers = 8
    
    # can be "dynamical", "alvarado", "alvarado-autoregressive"
    alpha_mode = "alvarado-autoregressive"
    scopt = False
    cache_dir = "matrix_cache"
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    estimation_plot_dir = os.path.join(output_dir, "estimation_plots")
    os.makedirs(estimation_plot_dir, exist_ok=True)

    results = {}
    
    for exp in experiments:
        code_name = exp["name"]
        console.rule(f"Running simulation for code {exp['code']}")
        
        data = np.load(f"codes/{exp['code']}.npz")
        Hx, Hz, Lx, Lz = data["Hx"], data["Hz"], data["Lx"], data["Lz"]
        bb_params = {k: data[k] for k in ["ell", "m", "a_x_powers", "a_y_powers", "b_y_powers", "b_x_powers"] if k in data}
        
        results[code_name] = {}
        
        # TODO: At this point since it's re-calculated later, we can simply move this function in the if statement
        # below
        cb = BBCodeCircuit(Hx, Hz, num_cycles=exp["distance"], **bb_params)
        
        for p in exp["physicalErrorRates"]:
            key = compute_cache_key(Hx, Hz, Lx, Lz, exp["distance"], p)
            matrices = load_matrices(cache_dir, key)
            if matrices is None:
                console.print(f"Building decoding matrices for p={p}...")
                matrices = build_decoding_matrices(cb, Lx, Lz, p, num_workers=num_workers)
                save_matrices(cache_dir, key, matrices)
            
            res = run_simulation(
                Hx, Hz, Lx, Lz, p, num_cycles=exp["distance"],
                maxIter=maxIter, osd_order=osd_order, precomputed_matrices=matrices, alpha_mode=alpha_mode,
                num_workers=num_workers, target_logical_errors=target_logical_errors,
                max_trials=max_trials, scopt=scopt, estimation_plot_dir=estimation_plot_dir, **bb_params
            )
            
            results[code_name][p] = res
            console.print(
                f"  LER: {res['logical_error_rate']:.4e} "
                f"(trials={res['num_trials']}, logical_errors={res['logical_errors']})"
            )

    # Plotting
    plot_path = f"{output_dir}/simulation_results.png"
    plot_simulation_results(results, plot_path)

    alpha_r2_values = {}
    if alpha_mode == "alvarado-autoregressive":
        alpha_plot_path = f"{output_dir}/alpha_comparison.png"
        plot_alpha_comparison(results, alpha_plot_path)

        alpha_linearity_path = f"{output_dir}/alpha_linearity.png"
        alpha_r2_values = plot_alpha_linearity(results, alpha_linearity_path)
    
    # Save results as npz
    alpha_values = {}
    beta_values = {}
    estimation_r2_values = {}
    for code_name, data in results.items():
        for p, res in data.items():
            if "alpha_values_z" in res or "alpha_values_x" in res:
                alpha_values.setdefault(code_name, {})[p] = {
                    "z": res.get("alpha_values_z"),
                    "x": res.get("alpha_values_x"),
                }
                estimation_r2_values.setdefault(code_name, {})[p] = {
                    "alpha_r2_values_z": res.get("alpha_r2_values_z"),
                    "alpha_r2_values_x": res.get("alpha_r2_values_x"),
                }
            if "alpha_r2_z" in res or "alpha_r2_x" in res:
                estimation_r2_values.setdefault(code_name, {})[p] = {
                    **estimation_r2_values.get(code_name, {}).get(p, {}),
                    "alpha_r2_z": res.get("alpha_r2_z"),
                    "alpha_r2_x": res.get("alpha_r2_x"),
                }
            if "beta_z" in res or "beta_x" in res:
                beta_values.setdefault(code_name, {})[p] = {
                    "z": res.get("beta_z"),
                    "x": res.get("beta_x"),
                }
            if "beta_r2_z" in res or "beta_r2_x" in res:
                estimation_r2_values.setdefault(code_name, {})[p] = {
                    **estimation_r2_values.get(code_name, {}).get(p, {}),
                    "beta_r2_z": res.get("beta_r2_z"),
                    "beta_r2_x": res.get("beta_r2_x"),
                }

    results_path = f"{output_dir}/results.npz"
    np.savez(
        results_path,
        results=results,
        alpha_values=alpha_values,
        beta_values=beta_values,
        alpha_r2_values=alpha_r2_values,
        estimation_r2_values=estimation_r2_values,
    )
    
    console.print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
