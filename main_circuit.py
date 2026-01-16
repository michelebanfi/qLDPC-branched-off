"""
Circuit-level noise simulation for Bivariate Bicycle codes.

This script runs Monte Carlo simulations using the circuit-level depolarizing
noise model with full spatio-temporal decoding, matching the approach in Bravyi et al.

Optimized for M1 Pro with:
- Multi-core parallel trial execution
- Numba JIT-compiled decoding kernels
- Memory-efficient matrix sharing
"""

import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count, get_context
from functools import partial

from circuit_noise import (
    BBCodeCircuit,
    generate_noisy_circuit,
    simulate_circuit_Z,
    simulate_circuit_X,
    sparsify_syndrome,
    extract_data_qubit_state,
    build_decoding_matrices,
)
from matrix_cache import get_or_compute_matrices
from decoding import performMinSum_Symmetric, performMinSum_Symmetric_Sparse, performOSD_enhanced
from scipy.sparse import csr_matrix


# Experiment configuration
experiments = [
    {
        "code": "[[72, 12, 6]]",
        "name": "72",
        "physicalErrorRates": [0.005, 0.004, 0.003, 0.002, 0.001],
        "distance": 6,
    },
    {
        "code": "[[90, 8, 10]]",
        "name": "90",
        "physicalErrorRates": [0.005, 0.004, 0.003, 0.002, 0.001],
        "distance": 10,
    },
    {
        "code": "[[108, 8, 10]]",
        "name": "108",
        "physicalErrorRates": [0.005, 0.004, 0.003, 0.002, 0.001],
        "distance": 10,
    },
    {
        "code": "[[144, 12, 12]]",
        "name": "144",
        "physicalErrorRates": [0.005, 0.004, 0.003, 0.002, 0.001],
        "distance": 12,
    },
    {
        "code": "[[288, 12, 18]]",
        "name": "288",
        "physicalErrorRates": [0.004, 0.003, 0.002, 0.001],
        "distance": 18,
    },
]


def _warmup_jit():
    """Pre-compile Numba functions before spawning workers."""
    H = np.random.randint(0, 2, (10, 20)).astype(np.float64)
    syndrome = np.random.randint(0, 2, 10).astype(np.int8)
    belief = np.random.randn(20).astype(np.float64)
    performMinSum_Symmetric(H, syndrome, belief, maxIter=2)


def _run_single_trial(trial_idx, shared_data):
    """
    Worker function for a single Monte Carlo trial.
    
    This runs in a separate process and returns error flags.
    """
    # Unpack shared data
    base_circuit = shared_data['base_circuit']
    noiseless_suffix = shared_data['noiseless_suffix']
    error_rate = shared_data['error_rate']
    lin_order = shared_data['lin_order']
    data_qubits = shared_data['data_qubits']
    n = shared_data['n']
    k = shared_data['k']
    Xchecks = shared_data['Xchecks']
    Zchecks = shared_data['Zchecks']
    Lx = shared_data['Lx']
    Lz = shared_data['Lz']
    HdecZ = shared_data['HdecZ']
    HdecZ_csr = shared_data.get('HdecZ_csr')  # Sparse version (may be None)
    HdecX_csr = shared_data.get('HdecX_csr')  # Sparse version (may be None)
    use_sparse = shared_data.get('use_sparse', False)
    HdecX = shared_data['HdecX']
    llrs_z = shared_data['llrs_z']
    llrs_x = shared_data['llrs_x']
    HZ_full = shared_data['HZ_full']
    HX_full = shared_data['HX_full']
    first_logical_rowZ = shared_data['first_logical_rowZ']
    first_logical_rowX = shared_data['first_logical_rowX']
    alpha = shared_data['alpha']
    maxIter = shared_data['maxIter']
    osd_order = shared_data['osd_order']
    
    # Seed RNG for this trial (ensures reproducibility and independence)
    np.random.seed(shared_data['base_seed'] + trial_idx)
    
    # Generate noisy circuit
    noisy_circuit = generate_noisy_circuit(base_circuit, error_rate)
    full_circuit = noisy_circuit + noiseless_suffix
    
    # === Decode Z errors ===
    syndrome_hist_z, state_z, syn_map_z, _ = simulate_circuit_Z(
        full_circuit, lin_order, n, Xchecks
    )
    
    data_state_z = extract_data_qubit_state(state_z, lin_order, data_qubits)
    true_logical_z = (Lx @ data_state_z) % 2
    
    sparse_syn_z = sparsify_syndrome(syndrome_hist_z, syn_map_z, Xchecks)
    
    # Use sparse decoder for large matrices (100x+ faster)
    if use_sparse and HdecZ_csr is not None:
        detection_z, success_z, final_llrs_z, _ = performMinSum_Symmetric_Sparse(
            HdecZ_csr, sparse_syn_z, llrs_z.copy(),
            maxIter=maxIter, alpha=alpha, damping=1.0, clip_llr=50
        )
    else:
        detection_z, success_z, final_llrs_z, _ = performMinSum_Symmetric(
            HdecZ, sparse_syn_z, llrs_z.copy(),
            maxIter=maxIter, alpha=alpha, damping=1.0, clip_llr=50
        )
    
    if not success_z:
        detection_z = performOSD_enhanced(
            HdecZ, sparse_syn_z, final_llrs_z, detection_z, order=osd_order
        )
    
    decoded_syndrome_full_z = (HZ_full @ detection_z) % 2
    decoded_logical_z = decoded_syndrome_full_z[first_logical_rowZ:first_logical_rowZ + k]
    
    z_error = not np.array_equal(decoded_logical_z, true_logical_z)
    
    # === Decode X errors ===
    syndrome_hist_x, state_x, syn_map_x, _ = simulate_circuit_X(
        full_circuit, lin_order, n, Zchecks
    )
    
    data_state_x = extract_data_qubit_state(state_x, lin_order, data_qubits)
    true_logical_x = (Lz @ data_state_x) % 2
    
    sparse_syn_x = sparsify_syndrome(syndrome_hist_x, syn_map_x, Zchecks)
    
    # Use sparse decoder for large matrices (100x+ faster)
    if use_sparse and HdecX_csr is not None:
        detection_x, success_x, final_llrs_x, _ = performMinSum_Symmetric_Sparse(
            HdecX_csr, sparse_syn_x, llrs_x.copy(),
            maxIter=maxIter, alpha=alpha, damping=1.0, clip_llr=50
        )
    else:
        detection_x, success_x, final_llrs_x, _ = performMinSum_Symmetric(
            HdecX, sparse_syn_x, llrs_x.copy(),
            maxIter=maxIter, alpha=alpha, damping=1.0, clip_llr=50
        )
    
    if not success_x:
        detection_x = performOSD_enhanced(
            HdecX, sparse_syn_x, final_llrs_x, detection_x, order=osd_order
        )
    
    decoded_syndrome_full_x = (HX_full @ detection_x) % 2
    decoded_logical_x = decoded_syndrome_full_x[first_logical_rowX:first_logical_rowX + k]
    
    x_error = not np.array_equal(decoded_logical_x, true_logical_x)
    
    return (z_error, x_error, z_error or x_error)


def _worker_init(shared_data_dict):
    """Initialize worker process with shared data."""
    global _shared_data
    _shared_data = shared_data_dict
    # Warm up JIT in each worker
    _warmup_jit()


def _worker_task(trial_idx):
    """Wrapper for worker function using global shared data."""
    return _run_single_trial(trial_idx, _shared_data)


def run_spatiotemporal_simulation(
    Hx: np.ndarray,
    Hz: np.ndarray,
    Lx: np.ndarray,
    Lz: np.ndarray,
    error_rate: float,
    num_trials: int = 1000,
    num_cycles: int = 12,
    maxIter: int = 10000,
    osd_order: int = 7,
    use_dynamic_alpha: bool = True,
    precomputed_matrices: dict = None,
    num_workers: int = None,
    base_seed: int = None,
    # BB code component parameters (optional but recommended)
    ell: int = None,
    m: int = None,
    a_x_powers: np.ndarray = None,
    a_y_powers: np.ndarray = None,
    b_y_powers: np.ndarray = None,
    b_x_powers: np.ndarray = None,
) -> dict:
    """
    Run circuit-level noise simulation with full spatio-temporal decoding.
    
    Parallelized across multiple CPU cores.
    
    Args:
        Hx: X-check parity matrix
        Hz: Z-check parity matrix
        Lx: X logical operators
        Lz: Z logical operators
        error_rate: Physical error rate p
        num_trials: Number of Monte Carlo trials
        num_cycles: Number of syndrome extraction cycles
        maxIter: Maximum BP iterations
        osd_order: OSD search depth
        use_dynamic_alpha: If True, use alpha=0 for dynamic scaling
        precomputed_matrices: Pre-built decoding matrices
        num_workers: Number of parallel workers (default: 8 for M1 Pro)
        base_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with simulation results
    """
    if num_workers is None:
        # M1 Pro has 8 performance cores - use those
        num_workers = min(8, cpu_count())
    
    if base_seed is None:
        base_seed = np.random.randint(0, 2**31)
    
    # Build the circuit with component parameters for correct neighbor ordering
    circuit_builder = BBCodeCircuit(
        Hx, Hz, 
        num_cycles=num_cycles,
        ell=ell,
        m=m,
        a_x_powers=a_x_powers,
        a_y_powers=a_y_powers,
        b_y_powers=b_y_powers,
        b_x_powers=b_x_powers,
    )
    base_circuit = circuit_builder.get_full_circuit()
    noiseless_suffix = circuit_builder.cycle * 2
    
    n = Hx.shape[1]
    k = Lx.shape[0]
    
    lin_order = circuit_builder.lin_order
    data_qubits = circuit_builder.data_qubits
    Xchecks = circuit_builder.Xchecks
    Zchecks = circuit_builder.Zchecks
    
    # Build or use precomputed decoding matrices
    if precomputed_matrices is None:
        matrices = build_decoding_matrices(
            circuit_builder, Lx, Lz, error_rate, verbose=True
        )
    else:
        matrices = precomputed_matrices
    
    HdecZ = np.ascontiguousarray(matrices['HdecZ'])
    HdecX = np.ascontiguousarray(matrices['HdecX'])
    channel_probsZ = matrices['channel_probsZ']
    channel_probsX = matrices['channel_probsX']
    HZ_full = np.ascontiguousarray(matrices['HZ_full'])
    HX_full = np.ascontiguousarray(matrices['HX_full'])
    first_logical_rowZ = matrices['first_logical_rowZ']
    first_logical_rowX = matrices['first_logical_rowX']
    
    # Compute initial beliefs (suppress warnings for impossible error patterns)
    with np.errstate(divide='ignore', invalid='ignore'):
        llrs_z = np.log((1 - channel_probsZ) / channel_probsZ)
        llrs_x = np.log((1 - channel_probsX) / channel_probsX)
    llrs_z = np.clip(np.nan_to_num(llrs_z, nan=0, posinf=50, neginf=-50), -50, 50)
    llrs_x = np.clip(np.nan_to_num(llrs_x, nan=0, posinf=50, neginf=-50), -50, 50)
    
    alpha = 0 if use_dynamic_alpha else 1.0
    
    # Create sparse versions for large matrices (huge speedup for [[288,12,18]])
    # Use sparse decoder when matrix has more than 5000 columns
    use_sparse = HdecZ.shape[1] > 5000
    HdecZ_csr = None
    HdecX_csr = None
    
    if use_sparse:
        # Convert to CSR for sparse BP decoder
        HdecZ_csr = csr_matrix(HdecZ.astype(np.float64))
        HdecX_csr = csr_matrix(HdecX.astype(np.float64))
        sparsity_z = 1 - (HdecZ_csr.nnz / (HdecZ.shape[0] * HdecZ.shape[1]))
        print(f"  Using SPARSE decoder (sparsity: {sparsity_z*100:.1f}%, columns: {HdecZ.shape[1]})")
    
    # Prepare shared data for workers
    shared_data = {
        'base_circuit': base_circuit,
        'noiseless_suffix': noiseless_suffix,
        'error_rate': error_rate,
        'lin_order': lin_order,
        'data_qubits': data_qubits,
        'n': n,
        'k': k,
        'Xchecks': Xchecks,
        'Zchecks': Zchecks,
        'Lx': Lx,
        'Lz': Lz,
        'HdecZ': HdecZ,
        'HdecX': HdecX,
        'HdecZ_csr': HdecZ_csr,
        'HdecX_csr': HdecX_csr,
        'use_sparse': use_sparse,
        'llrs_z': llrs_z,
        'llrs_x': llrs_x,
        'HZ_full': HZ_full,
        'HX_full': HX_full,
        'first_logical_rowZ': first_logical_rowZ,
        'first_logical_rowX': first_logical_rowX,
        'alpha': alpha,
        'maxIter': maxIter,
        'osd_order': osd_order,
        'base_seed': base_seed,
    }
    
    # Run trials in parallel
    if num_workers == 1:
        # Sequential execution (for debugging)
        results = []
        for i in tqdm.tqdm(range(num_trials), desc="Trials"):
            results.append(_run_single_trial(i, shared_data))
    else:
        # Parallel execution using spawn context (M1 compatible)
        ctx = get_context('spawn')
        with ctx.Pool(
            processes=num_workers,
            initializer=_worker_init,
            initargs=(shared_data,)
        ) as pool:
            results = list(tqdm.tqdm(
                pool.imap(_worker_task, range(num_trials)),
                total=num_trials,
                desc=f"Trials ({num_workers} workers)"
            ))
    
    # Aggregate results
    z_logical_errors = sum(1 for r in results if r[0])
    x_logical_errors = sum(1 for r in results if r[1])
    logical_errors = sum(1 for r in results if r[2])
    
    return {
        "logical_error_rate": logical_errors / num_trials,
        "z_logical_error_rate": z_logical_errors / num_trials,
        "x_logical_error_rate": x_logical_errors / num_trials,
        "num_trials": num_trials,
        "logical_errors": logical_errors,
    }


def main():
    """Run spatio-temporal simulation for all codes and error rates."""
    
    # Simulation parameters
    num_trials = 100  # Start smaller for testing; researchers used 50000
    maxIter = 10  # BP max iterations
    osd_order = 0  # OSD-CS search depth
    num_workers = 8  # M1 Pro performance cores
    
    # Warm up JIT in main process
    print("Warming up JIT compilation...")
    _warmup_jit()
    
    np.random.seed(42)
    
    results = {}
    
    for exp in experiments:
        code_name = exp["name"]
        print(f"\n{'='*60}")
        print(f"Running spatio-temporal simulation for code {exp['code']}")
        print(f"Using {num_workers} parallel workers")
        print(f"{'='*60}")
        
        # Load code
        code_data = np.load(f"codes/{exp['code']}.npz")
        Hx = code_data["Hx"]
        Hz = code_data["Hz"]
        Lx = code_data["Lx"]
        Lz = code_data["Lz"]
        
        # Load component parameters for correct CNOT scheduling
        ell = int(code_data["ell"]) if "ell" in code_data else None
        m = int(code_data["m"]) if "m" in code_data else None
        a_x_powers = code_data["a_x_powers"] if "a_x_powers" in code_data else None
        a_y_powers = code_data["a_y_powers"] if "a_y_powers" in code_data else None
        b_y_powers = code_data["b_y_powers"] if "b_y_powers" in code_data else None
        b_x_powers = code_data["b_x_powers"] if "b_x_powers" in code_data else None

        num_cycles = exp["distance"]
        
        results[code_name] = {}
        
        # Build circuit once with component parameters
        circuit_builder = BBCodeCircuit(
            Hx, Hz, 
            num_cycles=num_cycles,
            ell=ell,
            m=m,
            a_x_powers=a_x_powers,
            a_y_powers=a_y_powers,
            b_y_powers=b_y_powers,
            b_x_powers=b_x_powers,
        )
        
        for error_rate in exp["physicalErrorRates"]:
            print(f"\n--- p = {error_rate} ---")
            
            # Get or compute decoding matrices (with caching)
            print("Loading/building decoding matrices...")
            matrices = get_or_compute_matrices(
                circuit_builder, Lx, Lz, error_rate, Hx, Hz, verbose=True
            )
            
            print(f"Running {num_trials} trials on {num_workers} cores...")
            result = run_spatiotemporal_simulation(
                Hx, Hz, Lx, Lz,
                error_rate=error_rate,
                num_trials=num_trials,
                num_cycles=num_cycles,
                maxIter=maxIter,
                osd_order=osd_order,
                precomputed_matrices=matrices,
                num_workers=num_workers,
                ell=ell,
                m=m,
                a_x_powers=a_x_powers,
                a_y_powers=a_y_powers,
                b_y_powers=b_y_powers,
                b_x_powers=b_x_powers,
            )
            
            results[code_name][error_rate] = result
            print(f"  Logical Error Rate: {result['logical_error_rate']:.4e}")
            print(f"  Z-logical errors: {result['z_logical_error_rate']:.2%}")
            print(f"  X-logical errors: {result['x_logical_error_rate']:.2%}")
    
    # Save results
    np.savez("circuit_simulation_results.npz", results=results)
    
    # Plot
    colors = ["#2E72AE", "#64B791", "#DBA142", "#000000", "#E17792"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for (code_name, code_results), color in zip(results.items(), colors):
        x = list(code_results.keys())
        y = [r["logical_error_rate"] for r in code_results.values()]
        
        # Filter out zero values for log plot
        valid = [(xi, yi) for xi, yi in zip(x, y) if yi > 0]
        if valid:
            x_valid, y_valid = zip(*valid)
            ax.loglog(x_valid, y_valid, marker='d', label=f"[[{code_name},...]]]", color=color)
    
    ax.set_xlabel("Physical Error Rate, p")
    ax.set_ylabel("Logical Error Rate, $p_L$")
    ax.set_title(f"Circuit-Level Noise: BB Codes (Spatio-Temporal Decoding)\n({num_trials} trials, {num_workers} workers)")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("circuit_plot.png", dpi=300)
    print("\nPlot saved to circuit_plot.png")


if __name__ == "__main__":
    main()
