"""
Profiling script to identify bottlenecks in the circuit-level simulation.

This script runs a small simulation and profiles where time is spent.
"""

import cProfile
import pstats
import io
import time
import numpy as np

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
from decoding import performMinSum_Symmetric, performOSD_enhanced


def run_profiled_simulation():
    """Run a small simulation for profiling purposes."""
    
    # Use a smaller code for faster profiling
    code_data = np.load("codes/[[72, 12, 6]].npz")
    Hx = code_data["Hx"]
    Hz = code_data["Hz"]
    Lx = code_data["Lx"]
    Lz = code_data["Lz"]
    
    error_rate = 0.003
    num_cycles = 6
    num_trials = 20  # Small number for profiling
    maxIter = 50
    osd_order = 0
    
    print("=" * 60)
    print("PROFILING CIRCUIT-LEVEL SIMULATION")
    print("=" * 60)
    print(f"Code: [[72, 12, 6]]")
    print(f"Error rate: {error_rate}")
    print(f"Cycles: {num_cycles}")
    print(f"Trials: {num_trials}")
    print("=" * 60)
    
    # Build circuit
    circuit_builder = BBCodeCircuit(Hx, Hz, num_cycles=num_cycles)
    base_circuit = circuit_builder.get_full_circuit()
    noiseless_suffix = circuit_builder.cycle * 2
    
    n = Hx.shape[1]
    k = Lx.shape[0]
    
    lin_order = circuit_builder.lin_order
    data_qubits = circuit_builder.data_qubits
    Xchecks = circuit_builder.Xchecks
    Zchecks = circuit_builder.Zchecks
    
    print(f"\nCircuit statistics:")
    print(f"  Base circuit gates: {len(base_circuit)}")
    print(f"  Noiseless suffix gates: {len(noiseless_suffix)}")
    print(f"  Total qubits: {circuit_builder.total_qubits}")
    print(f"  Data qubits: {len(data_qubits)}")
    
    # Get matrices (cached)
    print("\nLoading decoding matrices...")
    matrices = get_or_compute_matrices(
        circuit_builder, Lx, Lz, error_rate, Hx, Hz, verbose=False
    )
    
    HdecZ = np.ascontiguousarray(matrices['HdecZ'])
    HdecX = np.ascontiguousarray(matrices['HdecX'])
    channel_probsZ = matrices['channel_probsZ']
    channel_probsX = matrices['channel_probsX']
    HZ_full = np.ascontiguousarray(matrices['HZ_full'])
    HX_full = np.ascontiguousarray(matrices['HX_full'])
    first_logical_rowZ = matrices['first_logical_rowZ']
    first_logical_rowX = matrices['first_logical_rowX']
    
    with np.errstate(divide='ignore'):
        llrs_z = np.log((1 - channel_probsZ) / channel_probsZ)
        llrs_x = np.log((1 - channel_probsX) / channel_probsX)
    llrs_z = np.clip(llrs_z, -50, 50)
    llrs_x = np.clip(llrs_x, -50, 50)
    
    print(f"  HdecZ shape: {HdecZ.shape}")
    print(f"  HdecX shape: {HdecX.shape}")
    
    # Timing breakdown
    timings = {
        'generate_noisy_circuit': 0.0,
        'simulate_circuit_Z': 0.0,
        'simulate_circuit_X': 0.0,
        'sparsify_syndrome_Z': 0.0,
        'sparsify_syndrome_X': 0.0,
        'extract_data_state': 0.0,
        'minsum_Z': 0.0,
        'minsum_X': 0.0,
        'osd_Z': 0.0,
        'osd_X': 0.0,
        'logical_check': 0.0,
    }
    
    osd_calls_z = 0
    osd_calls_x = 0
    
    np.random.seed(42)
    
    print(f"\nRunning {num_trials} trials...")
    
    for trial_idx in range(num_trials):
        # 1. Generate noisy circuit
        t0 = time.perf_counter()
        noisy_circuit = generate_noisy_circuit(base_circuit, error_rate)
        full_circuit = noisy_circuit + noiseless_suffix
        timings['generate_noisy_circuit'] += time.perf_counter() - t0
        
        # 2. Simulate Z errors
        t0 = time.perf_counter()
        syndrome_hist_z, state_z, syn_map_z, _ = simulate_circuit_Z(
            full_circuit, lin_order, n, Xchecks
        )
        timings['simulate_circuit_Z'] += time.perf_counter() - t0
        
        # 3. Extract data state
        t0 = time.perf_counter()
        data_state_z = extract_data_qubit_state(state_z, lin_order, data_qubits)
        true_logical_z = (Lx @ data_state_z) % 2
        timings['extract_data_state'] += time.perf_counter() - t0
        
        # 4. Sparsify syndrome Z
        t0 = time.perf_counter()
        sparse_syn_z = sparsify_syndrome(syndrome_hist_z, syn_map_z, Xchecks)
        timings['sparsify_syndrome_Z'] += time.perf_counter() - t0
        
        # 5. Min-sum Z
        t0 = time.perf_counter()
        detection_z, success_z, final_llrs_z, _ = performMinSum_Symmetric(
            HdecZ, sparse_syn_z, llrs_z.copy(),
            maxIter=maxIter, alpha=0, damping=1.0, clip_llr=50
        )
        timings['minsum_Z'] += time.perf_counter() - t0
        
        # 6. OSD Z (if needed)
        if not success_z:
            t0 = time.perf_counter()
            detection_z = performOSD_enhanced(
                HdecZ, sparse_syn_z, final_llrs_z, detection_z, order=osd_order
            )
            timings['osd_Z'] += time.perf_counter() - t0
            osd_calls_z += 1
        
        # 7. Logical check Z
        t0 = time.perf_counter()
        decoded_syndrome_full_z = (HZ_full @ detection_z) % 2
        decoded_logical_z = decoded_syndrome_full_z[first_logical_rowZ:first_logical_rowZ + k]
        z_error = not np.array_equal(decoded_logical_z, true_logical_z)
        timings['logical_check'] += time.perf_counter() - t0
        
        # --- X errors ---
        
        # 8. Simulate X errors
        t0 = time.perf_counter()
        syndrome_hist_x, state_x, syn_map_x, _ = simulate_circuit_X(
            full_circuit, lin_order, n, Zchecks
        )
        timings['simulate_circuit_X'] += time.perf_counter() - t0
        
        # 9. Extract data state for X
        t0 = time.perf_counter()
        data_state_x = extract_data_qubit_state(state_x, lin_order, data_qubits)
        true_logical_x = (Lz @ data_state_x) % 2
        timings['extract_data_state'] += time.perf_counter() - t0
        
        # 10. Sparsify syndrome X
        t0 = time.perf_counter()
        sparse_syn_x = sparsify_syndrome(syndrome_hist_x, syn_map_x, Zchecks)
        timings['sparsify_syndrome_X'] += time.perf_counter() - t0
        
        # 11. Min-sum X
        t0 = time.perf_counter()
        detection_x, success_x, final_llrs_x, _ = performMinSum_Symmetric(
            HdecX, sparse_syn_x, llrs_x.copy(),
            maxIter=maxIter, alpha=0, damping=1.0, clip_llr=50
        )
        timings['minsum_X'] += time.perf_counter() - t0
        
        # 12. OSD X (if needed)
        if not success_x:
            t0 = time.perf_counter()
            detection_x = performOSD_enhanced(
                HdecX, sparse_syn_x, final_llrs_x, detection_x, order=osd_order
            )
            timings['osd_X'] += time.perf_counter() - t0
            osd_calls_x += 1
    
    # Print results
    total_time = sum(timings.values())
    
    print("\n" + "=" * 60)
    print("TIMING BREAKDOWN (percentage of total trial time)")
    print("=" * 60)
    
    # Group and sort by time
    grouped = {
        'Circuit Generation': timings['generate_noisy_circuit'],
        'Simulation (Z)': timings['simulate_circuit_Z'],
        'Simulation (X)': timings['simulate_circuit_X'],
        'Sparsify Syndrome': timings['sparsify_syndrome_Z'] + timings['sparsify_syndrome_X'],
        'Extract Data State': timings['extract_data_state'],
        'BP Min-Sum (Z)': timings['minsum_Z'],
        'BP Min-Sum (X)': timings['minsum_X'],
        'OSD (Z)': timings['osd_Z'],
        'OSD (X)': timings['osd_X'],
        'Logical Check': timings['logical_check'],
    }
    
    # Sort by time descending
    sorted_timings = sorted(grouped.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Component':<25} {'Time (s)':<12} {'%':<8} {'ms/trial':<10}")
    print("-" * 60)
    
    for name, t in sorted_timings:
        pct = 100 * t / total_time if total_time > 0 else 0
        ms_per_trial = 1000 * t / num_trials
        bar = "█" * int(pct / 2)
        print(f"{name:<25} {t:<12.4f} {pct:<7.1f}% {ms_per_trial:<10.2f} {bar}")
    
    print("-" * 60)
    print(f"{'TOTAL':<25} {total_time:<12.4f} {'100%':<8} {1000*total_time/num_trials:<10.2f}")
    
    print(f"\nAdditional stats:")
    print(f"  OSD called for Z: {osd_calls_z}/{num_trials} trials ({100*osd_calls_z/num_trials:.1f}%)")
    print(f"  OSD called for X: {osd_calls_x}/{num_trials} trials ({100*osd_calls_x/num_trials:.1f}%)")
    
    # Simulation vs Decoding breakdown
    sim_time = (timings['generate_noisy_circuit'] + 
                timings['simulate_circuit_Z'] + 
                timings['simulate_circuit_X'] +
                timings['sparsify_syndrome_Z'] + 
                timings['sparsify_syndrome_X'] +
                timings['extract_data_state'])
    
    decode_time = (timings['minsum_Z'] + timings['minsum_X'] + 
                   timings['osd_Z'] + timings['osd_X'])
    
    print(f"\n{'=' * 60}")
    print("SUMMARY: Simulation vs Decoding")
    print("=" * 60)
    print(f"  Circuit Simulation: {sim_time:.4f}s ({100*sim_time/total_time:.1f}%)")
    print(f"  Decoding (BP+OSD):  {decode_time:.4f}s ({100*decode_time/total_time:.1f}%)")
    print(f"  Other overhead:     {total_time - sim_time - decode_time:.4f}s")
    
    if sim_time > decode_time:
        ratio = sim_time / decode_time if decode_time > 0 else float('inf')
        print(f"\n→ Circuit simulation is {ratio:.1f}x slower than decoding.")
        print("  JIT-compiling generate_noisy_circuit and simulate_circuit would help!")
    else:
        ratio = decode_time / sim_time if sim_time > 0 else float('inf')
        print(f"\n→ Decoding is {ratio:.1f}x slower than circuit simulation.")
        print("  Consider optimizing BP/OSD or reducing maxIter if acceptable.")


def run_cprofile():
    """Run cProfile for detailed function-level breakdown."""
    print("\n" + "=" * 60)
    print("RUNNING CPROFILE FOR DETAILED BREAKDOWN")
    print("=" * 60)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    run_profiled_simulation()
    
    profiler.disable()
    
    # Print top functions by cumulative time
    print("\n" + "=" * 60)
    print("TOP 30 FUNCTIONS BY CUMULATIVE TIME")
    print("=" * 60)
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())


if __name__ == "__main__":
    run_profiled_simulation()
    
    print("\n\n")
    print("Run with 'python profile_simulation.py --cprofile' for detailed function breakdown")
    
    import sys
    if '--cprofile' in sys.argv:
        run_cprofile()
