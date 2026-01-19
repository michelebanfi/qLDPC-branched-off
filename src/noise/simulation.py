import numpy as np
from typing import List, Tuple, Dict, TYPE_CHECKING

from .kernels import (
    simulate_circuit_Z_jit, 
    simulate_circuit_X_jit,
    generate_noisy_circuit_jit,
    sparsify_syndrome_jit,
    extract_data_state_jit
)
from .compiled import CompiledCircuit, circuit_to_arrays

if TYPE_CHECKING:
    pass


# =============================================================================
# JIT-Compiled Fast Path (50-100x faster)
# =============================================================================

def run_trial_fast(
    compiled: CompiledCircuit,
    error_rate: float,
    Lx: np.ndarray,
    Lz: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a single trial using JIT-compiled simulation.
    
    Args:
        compiled: Pre-compiled circuit representation
        error_rate: Error probability
        Lx, Lz: Logical operator matrices
        
    Returns:
        sparse_z: Sparsified Z syndrome
        true_z: True Z logical error
        sparse_x: Sparsified X syndrome  
        true_x: True X logical error
    """
    # Generate random numbers for noisy circuit
    n_locs = compiled.num_error_locs
    random_vals = np.random.random(n_locs)
    random_paulis = np.random.randint(0, 3, n_locs, dtype=np.int32)
    random_two_qubit = np.random.randint(0, 15, n_locs, dtype=np.int32)
    
    # Generate noisy base circuit
    noisy_len = generate_noisy_circuit_jit(
        compiled.base_ops, compiled.base_q1, compiled.base_q2,
        error_rate, random_vals, random_paulis, random_two_qubit,
        compiled.out_ops, compiled.out_q1, compiled.out_q2
    )
    
    # Append noiseless suffix
    suffix_len = len(compiled.suffix_ops)
    total_len = noisy_len + suffix_len
    
    # Create full circuit arrays
    full_ops = np.empty(total_len, dtype=np.int32)
    full_q1 = np.empty(total_len, dtype=np.int32)
    full_q2 = np.empty(total_len, dtype=np.int32)
    
    full_ops[:noisy_len] = compiled.out_ops[:noisy_len]
    full_q1[:noisy_len] = compiled.out_q1[:noisy_len]
    full_q2[:noisy_len] = compiled.out_q2[:noisy_len]
    
    full_ops[noisy_len:] = compiled.suffix_ops
    full_q1[noisy_len:] = compiled.suffix_q1
    full_q2[noisy_len:] = compiled.suffix_q2
    
    # Z-error simulation
    syn_z, state_z, syn_count_z, _ = simulate_circuit_Z_jit(
        full_ops, full_q1, full_q2,
        compiled.total_qubits,
        compiled.x_check_indices, compiled.x_check_ptrs,
        compiled.max_syndromes_x
    )
    
    # Extract data state and compute true logical
    data_state_z = extract_data_state_jit(state_z, compiled.data_qubit_indices)
    true_z = (Lx @ data_state_z) % 2
    
    # Sparsify syndrome
    sparse_z = sparsify_syndrome_jit(
        syn_z, syn_count_z,
        compiled.x_syn_positions, compiled.x_syn_ptrs,
        compiled.num_x_checks
    )
    
    # X-error simulation
    syn_x, state_x, syn_count_x, _ = simulate_circuit_X_jit(
        full_ops, full_q1, full_q2,
        compiled.total_qubits,
        compiled.z_check_indices, compiled.z_check_ptrs,
        compiled.max_syndromes_z
    )
    
    data_state_x = extract_data_state_jit(state_x, compiled.data_qubit_indices)
    true_x = (Lz @ data_state_x) % 2
    
    sparse_x = sparsify_syndrome_jit(
        syn_x, syn_count_x,
        compiled.z_syn_positions, compiled.z_syn_ptrs,
        compiled.num_z_checks
    )
    
    return sparse_z, true_z.astype(np.int8), sparse_x, true_x.astype(np.int8)


# =============================================================================
# Original Pure Python Implementation (kept for compatibility/testing)
# =============================================================================

def simulate_circuit_Z(
    circuit: List[Tuple],
    lin_order: Dict[Tuple, int],
    n: int,
    Xchecks: List[Tuple]
) -> Tuple[np.ndarray, np.ndarray, Dict, int]:
    total_qubits = len(lin_order)
    state = np.zeros(total_qubits, dtype=int)
    syndrome_history = []
    syndrome_map = {c: [] for c in Xchecks}
    err_cnt = 0
    syn_cnt = 0
    
    for gate in circuit:
        gate_type = gate[0]
        if gate_type == 'CNOT':
            control_idx = lin_order[gate[1]]
            target_idx = lin_order[gate[2]]
            state[control_idx] = (state[control_idx] + state[target_idx]) % 2
        elif gate_type == 'PrepX':
            q = lin_order[gate[1]]
            state[q] = 0
        elif gate_type == 'MeasX':
            q = lin_order[gate[1]]
            check = gate[1]
            syndrome_history.append(state[q])
            syndrome_map[check].append(syn_cnt)
            syn_cnt += 1
        elif gate_type in ['Z', 'Y']:
            err_cnt += 1
            q = lin_order[gate[1]]
            state[q] = (state[q] + 1) % 2
        elif gate_type in ['ZX', 'YX']:
            err_cnt += 1
            q = lin_order[gate[1]]
            state[q] = (state[q] + 1) % 2
        elif gate_type in ['XZ', 'XY']:
            err_cnt += 1
            q = lin_order[gate[2]]
            state[q] = (state[q] + 1) % 2
        elif gate_type in ['ZZ', 'YY', 'YZ', 'ZY']:
            err_cnt += 1
            q1 = lin_order[gate[1]]
            q2 = lin_order[gate[2]]
            state[q1] = (state[q1] + 1) % 2
            state[q2] = (state[q2] + 1) % 2
            
    return np.array(syndrome_history, dtype=int), state, syndrome_map, err_cnt

def simulate_circuit_X(
    circuit: List[Tuple],
    lin_order: Dict[Tuple, int],
    n: int,
    Zchecks: List[Tuple]
) -> Tuple[np.ndarray, np.ndarray, Dict, int]:
    total_qubits = len(lin_order)
    state = np.zeros(total_qubits, dtype=int)
    syndrome_history = []
    syndrome_map = {c: [] for c in Zchecks}
    err_cnt = 0
    syn_cnt = 0
    
    for gate in circuit:
        gate_type = gate[0]
        if gate_type == 'CNOT':
            control_idx = lin_order[gate[1]]
            target_idx = lin_order[gate[2]]
            state[target_idx] = (state[target_idx] + state[control_idx]) % 2
        elif gate_type == 'PrepZ':
            q = lin_order[gate[1]]
            state[q] = 0
        elif gate_type == 'MeasZ':
            q = lin_order[gate[1]]
            check = gate[1]
            syndrome_history.append(state[q])
            syndrome_map[check].append(syn_cnt)
            syn_cnt += 1
        elif gate_type in ['X', 'Y']:
            err_cnt += 1
            q = lin_order[gate[1]]
            state[q] = (state[q] + 1) % 2
        elif gate_type in ['XZ', 'YZ']:
            err_cnt += 1
            q = lin_order[gate[1]]
            state[q] = (state[q] + 1) % 2
        elif gate_type in ['ZX', 'ZY']:
            err_cnt += 1
            q = lin_order[gate[2]]
            state[q] = (state[q] + 1) % 2
        elif gate_type in ['XX', 'YY', 'XY', 'YX']:
            err_cnt += 1
            q1 = lin_order[gate[1]]
            q2 = lin_order[gate[2]]
            state[q1] = (state[q1] + 1) % 2
            state[q2] = (state[q2] + 1) % 2
            
    return np.array(syndrome_history, dtype=int), state, syndrome_map, err_cnt

def sparsify_syndrome(
    syndrome_history: np.ndarray,
    syndrome_map: Dict[Tuple, List[int]],
    checks: List[Tuple]
) -> np.ndarray:
    result = syndrome_history.copy()
    for check in checks:
        positions = syndrome_map[check]
        for i in range(1, len(positions)):
            result[positions[i]] = (result[positions[i]] + syndrome_history[positions[i-1]]) % 2
    return result

def extract_data_qubit_state(
    state: np.ndarray,
    lin_order: Dict[Tuple, int],
    data_qubits: List[Tuple]
) -> np.ndarray:
    return np.array([state[lin_order[q]] for q in data_qubits], dtype=int)
