import numpy as np
import matplotlib.pyplot as plt

from src.codes.bb_code import BBCodeCircuit
from src.simulation.engine import run_simulation
from src.noise.builder import build_decoding_matrices
from src.utils.plotting import plot_simulation_results
from src.utils.caching import compute_cache_key, load_matrices, save_matrices

# Experiment configuration
experiments = [
    {"code": "[[72, 12, 6]]", "name": "72", "physicalErrorRates": [0.006, 0.005, 0.004, 0.003, 0.0018, 0.001, 0.0005, 0.00018], "distance": 6},
    {"code": "[[90, 8, 10]]", "name": "90", "physicalErrorRates": [0.006, 0.005, 0.004, 0.003, 0.0018, 0.001, 0.0005], "distance": 10},
    {"code": "[[108, 8, 10]]", "name": "108", "physicalErrorRates": [0.006, 0.005, 0.004, 0.003, 0.0018, 0.0001, 0.0005], "distance": 10},
    {"code": "[[144, 12, 12]]", "name": "144", "physicalErrorRates": [0.006, 0.005, 0.004, 0.003, 0.0018, 0.0001], "distance": 12},
    {"code": "[[288, 12, 18]]", "name": "288", "physicalErrorRates": [0.006, 0.005, 0.004, 0.0035], "distance": 18},
]

def main():
    target_logical_errors = 30
    max_trials = 100000
    maxIter = 200
    osd_order = 7
    num_workers = 8
    cache_dir = "matrix_cache"
    
    results = {}
    
    for exp in experiments:
        code_name = exp["name"]
        print(f"\n{'='*60}\nRunning simulation for code {exp['code']}\n{'='*60}")
        
        data = np.load(f"codes/{exp['code']}.npz")
        Hx, Hz, Lx, Lz = data["Hx"], data["Hz"], data["Lx"], data["Lz"]
        bb_params = {k: data[k] for k in ["ell", "m", "a_x_powers", "a_y_powers", "b_y_powers", "b_x_powers"] if k in data}
        
        results[code_name] = {}
        
        # At this point since it's re-calculated later, we can simply move this function in the if statement
        # below
        cb = BBCodeCircuit(Hx, Hz, num_cycles=exp["distance"], **bb_params)
        
        for p in exp["physicalErrorRates"]:
            key = compute_cache_key(Hx, Hz, Lx, Lz, exp["distance"], p)
            matrices = load_matrices(cache_dir, key)
            if matrices is None:
                print(f"Building decoding matrices for p={p}...")
                matrices = build_decoding_matrices(cb, Lx, Lz, p, num_workers=num_workers)
                save_matrices(cache_dir, key, matrices)
            
            res = run_simulation(
                Hx, Hz, Lx, Lz, p, num_cycles=exp["distance"],
                maxIter=maxIter, osd_order=osd_order, precomputed_matrices=matrices,
                num_workers=num_workers, target_logical_errors=target_logical_errors,
                max_trials=max_trials, **bb_params
            )
            
            results[code_name][p] = res
            print(f"  LER: {res['logical_error_rate']:.4e} (trials={res['num_trials']}, logical_errors={res['logical_errors']})")

    # Plotting
    plot_simulation_results(results, "simulation_results.png")

if __name__ == "__main__":
    main()
