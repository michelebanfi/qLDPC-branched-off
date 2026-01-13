"""
Circuit-level noise simulation for Bivariate Bicycle codes.

This script runs Monte Carlo simulations using the circuit-level depolarizing
noise model with full spatio-temporal decoding, matching the approach in Bravyi et al.
"""

import tqdm
import numpy as np
import matplotlib.pyplot as plt

from circuit_noise import (
    BBCodeCircuit,
    generate_noisy_circuit,
    simulate_circuit_Z,
    simulate_circuit_X,
    sparsify_syndrome,
    extract_data_qubit_state,
    build_decoding_matrices,
)
from decoding import performMinSum_Symmetric, performOSD_enhanced


# Experiment configuration - using error rates closer to researchers'
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
) -> dict:
    """
    Run circuit-level noise simulation with full spatio-temporal decoding.
    
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
        precomputed_matrices: Pre-built decoding matrices (if None, will compute)
        
    Returns:
        Dictionary with simulation results
    """
    # Build the circuit
    circuit_builder = BBCodeCircuit(Hx, Hz, num_cycles=num_cycles)
    base_circuit = circuit_builder.get_full_circuit()
    noiseless_suffix = circuit_builder.cycle * 2
    
    n = Hx.shape[1]
    n2 = n // 2
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
    
    HdecZ = matrices['HdecZ']
    HdecX = matrices['HdecX']
    channel_probsZ = matrices['channel_probsZ']
    channel_probsX = matrices['channel_probsX']
    HZ_full = matrices['HZ_full']
    HX_full = matrices['HX_full']
    first_logical_rowZ = matrices['first_logical_rowZ']
    first_logical_rowX = matrices['first_logical_rowX']
    
    # Compute initial beliefs from channel probabilities
    # LLR = log((1-p)/p) for each error location
    with np.errstate(divide='ignore'):
        llrs_z = np.log((1 - channel_probsZ) / channel_probsZ)
        llrs_x = np.log((1 - channel_probsX) / channel_probsX)
    llrs_z = np.clip(llrs_z, -50, 50)
    llrs_x = np.clip(llrs_x, -50, 50)
    
    alpha = 0 if use_dynamic_alpha else 1.0
    
    # Statistics
    logical_errors = 0
    z_logical_errors = 0
    x_logical_errors = 0
    
    for trial in range(num_trials):
        # Generate noisy circuit
        noisy_circuit = generate_noisy_circuit(base_circuit, error_rate)
        full_circuit = noisy_circuit + noiseless_suffix
        
        # === Decode Z errors ===
        syndrome_hist_z, state_z, syn_map_z, _ = simulate_circuit_Z(
            full_circuit, lin_order, n, Xchecks
        )
        
        # Get actual data qubit state (for checking logical errors)
        data_state_z = extract_data_qubit_state(state_z, lin_order, data_qubits)
        true_logical_z = (Lx @ data_state_z) % 2
        
        # Sparsify syndrome
        sparse_syn_z = sparsify_syndrome(syndrome_hist_z, syn_map_z, Xchecks)
        
        # Decode using extended matrix
        detection_z, success_z, final_llrs_z, _ = performMinSum_Symmetric(
            HdecZ, sparse_syn_z, llrs_z.copy(),
            maxIter=maxIter, alpha=alpha, damping=1.0, clip_llr=50
        )
        
        if not success_z:
            detection_z = performOSD_enhanced(
                HdecZ, sparse_syn_z, final_llrs_z, detection_z, order=osd_order
            )
        
        # Check logical error: apply detection to full matrix and check logical rows
        decoded_syndrome_full_z = (HZ_full @ detection_z) % 2
        decoded_logical_z = decoded_syndrome_full_z[first_logical_rowZ:first_logical_rowZ + k]
        
        z_error = not np.array_equal(decoded_logical_z, true_logical_z)
        if z_error:
            z_logical_errors += 1
        
        # === Decode X errors ===
        syndrome_hist_x, state_x, syn_map_x, _ = simulate_circuit_X(
            full_circuit, lin_order, n, Zchecks
        )
        
        data_state_x = extract_data_qubit_state(state_x, lin_order, data_qubits)
        true_logical_x = (Lz @ data_state_x) % 2
        
        sparse_syn_x = sparsify_syndrome(syndrome_hist_x, syn_map_x, Zchecks)
        
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
        if x_error:
            x_logical_errors += 1
        
        # Combined logical error
        if z_error or x_error:
            logical_errors += 1
            
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
    num_trials = 500  # Start smaller for testing; researchers used 50000
    num_cycles = 12
    maxIter = 1000  # BP max iterations
    osd_order = 7   # OSD-CS search depth
    
    np.random.seed(42)
    
    results = {}
    
    for exp in experiments:
        code_name = exp["name"]
        print(f"\n{'='*60}")
        print(f"Running spatio-temporal simulation for code {exp['code']}")
        print(f"{'='*60}")
        
        # Load code
        code_data = np.load(f"codes/{exp['code']}.npz")
        Hx = code_data["Hx"]
        Hz = code_data["Hz"]
        Lx = code_data["Lx"]
        Lz = code_data["Lz"]
        
        results[code_name] = {}
        
        # Build circuit once
        circuit_builder = BBCodeCircuit(Hx, Hz, num_cycles=num_cycles)
        
        for error_rate in exp["physicalErrorRates"]:
            print(f"\n--- p = {error_rate} ---")
            
            # Build decoding matrices for this error rate
            print("Building decoding matrices...")
            matrices = build_decoding_matrices(
                circuit_builder, Lx, Lz, error_rate, verbose=True
            )
            
            print(f"Running {num_trials} trials...")
            result = run_spatiotemporal_simulation(
                Hx, Hz, Lx, Lz,
                error_rate=error_rate,
                num_trials=num_trials,
                num_cycles=num_cycles,
                maxIter=maxIter,
                osd_order=osd_order,
                precomputed_matrices=matrices,
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
            ax.loglog(x_valid, y_valid, marker='d', label=f"[[{code_name},...]]", color=color)
    
    ax.set_xlabel("Physical Error Rate, p")
    ax.set_ylabel("Logical Error Rate, $p_L$")
    ax.set_title(f"Circuit-Level Noise: BB Codes (Spatio-Temporal Decoding)\n({num_trials} trials, {num_cycles} cycles)")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("circuit_plot.png", dpi=300)
    print("\nPlot saved to circuit_plot.png")


if __name__ == "__main__":
    main()
