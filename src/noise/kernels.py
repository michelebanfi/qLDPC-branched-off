"""
JIT-compiled circuit simulation kernels using Numba.
These replace the pure Python simulate_circuit_Z/X functions for ~50-100x speedup.
"""
import numpy as np
from numba import njit
from .constants import (
    OP_CNOT, OP_PREP_X, OP_PREP_Z, OP_MEAS_X, OP_MEAS_Z, OP_IDLE,
    OP_X, OP_Y, OP_Z, OP_XX, OP_XY, OP_XZ, OP_YX, OP_YY, OP_YZ, OP_ZX, OP_ZY, OP_ZZ
)


@njit(cache=True, nogil=True)
def simulate_circuit_Z_jit(
    circuit_ops: np.ndarray,
    circuit_q1: np.ndarray, 
    circuit_q2: np.ndarray,
    total_qubits: int,
    x_check_indices: np.ndarray,
    x_check_ptrs: np.ndarray,
    max_syndromes: int
) -> tuple:
    """
    JIT-compiled Z-error propagation simulator.
    
    Tracks how Z errors propagate through the circuit and affect X-check measurements.
    
    Args:
        circuit_ops: Array of op codes (int32)
        circuit_q1: Array of first qubit indices (control for CNOT, qubit for single-qubit ops)
        circuit_q2: Array of second qubit indices (target for CNOT, -1 for single-qubit ops)
        total_qubits: Total number of qubits in linearized order
        x_check_indices: Flattened array of X-check qubit indices
        x_check_ptrs: Pointer array (CSR-style) for x_check_indices per check
        max_syndromes: Maximum number of syndrome measurements
        
    Returns:
        syndrome_history: Array of syndrome measurements
        state: Final Z-error state on all qubits
        syn_count: Number of syndromes recorded
        err_count: Number of errors encountered
    """
    state = np.zeros(total_qubits, dtype=np.int8)
    syndrome_history = np.zeros(max_syndromes, dtype=np.int8)
    syn_count = 0
    err_count = 0
    
    num_gates = len(circuit_ops)
    
    for i in range(num_gates):
        op = circuit_ops[i]
        q1 = circuit_q1[i]
        q2 = circuit_q2[i]
        
        if op == OP_CNOT:
            # Z propagates: target -> control (phase kickback in Z basis)
            state[q1] ^= state[q2]
            
        elif op == OP_PREP_X:
            # PrepX resets Z error on this qubit
            state[q1] = 0
            
        elif op == OP_MEAS_X:
            # MeasX: record syndrome (Z error flips X measurement)
            syndrome_history[syn_count] = state[q1]
            syn_count += 1
            
        elif op == OP_Z or op == OP_Y:
            # Z or Y error (Y has Z component)
            err_count += 1
            state[q1] ^= 1
            
        elif op == OP_ZX or op == OP_YX:
            # ZX or YX: Z component on control (q1)
            err_count += 1
            state[q1] ^= 1
            
        elif op == OP_XZ or op == OP_XY:
            # XZ or XY: Z component on target (q2)
            err_count += 1
            state[q2] ^= 1
            
        elif op == OP_ZZ or op == OP_YY or op == OP_YZ or op == OP_ZY:
            # Two-qubit errors with Z on both
            err_count += 1
            state[q1] ^= 1
            state[q2] ^= 1
            
        # OP_X, OP_XX, OP_PREP_Z, OP_MEAS_Z, OP_IDLE don't affect Z-error tracking
            
    return syndrome_history, state, syn_count, err_count


@njit(cache=True, nogil=True)
def simulate_circuit_X_jit(
    circuit_ops: np.ndarray,
    circuit_q1: np.ndarray,
    circuit_q2: np.ndarray,
    total_qubits: int,
    z_check_indices: np.ndarray,
    z_check_ptrs: np.ndarray,
    max_syndromes: int
) -> tuple:
    """
    JIT-compiled X-error propagation simulator.
    
    Tracks how X errors propagate through the circuit and affect Z-check measurements.
    
    Args:
        circuit_ops: Array of op codes (int32)
        circuit_q1: Array of first qubit indices
        circuit_q2: Array of second qubit indices
        total_qubits: Total number of qubits
        z_check_indices: Flattened array of Z-check qubit indices
        z_check_ptrs: Pointer array for z_check_indices
        max_syndromes: Maximum number of syndrome measurements
        
    Returns:
        syndrome_history: Array of syndrome measurements
        state: Final X-error state
        syn_count: Number of syndromes
        err_count: Number of errors
    """
    state = np.zeros(total_qubits, dtype=np.int8)
    syndrome_history = np.zeros(max_syndromes, dtype=np.int8)
    syn_count = 0
    err_count = 0
    
    num_gates = len(circuit_ops)
    
    for i in range(num_gates):
        op = circuit_ops[i]
        q1 = circuit_q1[i]
        q2 = circuit_q2[i]
        
        if op == OP_CNOT:
            # X propagates: control -> target
            state[q2] ^= state[q1]
            
        elif op == OP_PREP_Z:
            # PrepZ resets X error
            state[q1] = 0
            
        elif op == OP_MEAS_Z:
            # MeasZ: record syndrome
            syndrome_history[syn_count] = state[q1]
            syn_count += 1
            
        elif op == OP_X or op == OP_Y:
            # X or Y error (Y has X component)
            err_count += 1
            state[q1] ^= 1
            
        elif op == OP_XZ or op == OP_YZ:
            # XZ or YZ: X component on control (q1)
            err_count += 1
            state[q1] ^= 1
            
        elif op == OP_ZX or op == OP_ZY:
            # ZX or ZY: X component on target (q2)
            err_count += 1
            state[q2] ^= 1
            
        elif op == OP_XX or op == OP_YY or op == OP_XY or op == OP_YX:
            # Two-qubit errors with X on both
            err_count += 1
            state[q1] ^= 1
            state[q2] ^= 1
            
        # OP_Z, OP_ZZ, OP_PREP_X, OP_MEAS_X, OP_IDLE don't affect X-error tracking
            
    return syndrome_history, state, syn_count, err_count


@njit(cache=True, nogil=True)
def generate_noisy_circuit_jit(
    base_ops: np.ndarray,
    base_q1: np.ndarray,
    base_q2: np.ndarray,
    error_rate: float,
    random_vals: np.ndarray,
    random_paulis: np.ndarray,
    random_two_qubit: np.ndarray,
    out_ops: np.ndarray,
    out_q1: np.ndarray,
    out_q2: np.ndarray
) -> int:
    """
    JIT-compiled noisy circuit generation.
    
    Args:
        base_ops, base_q1, base_q2: Base circuit arrays
        error_rate: Probability of error per location
        random_vals: Pre-generated random values for error decisions
        random_paulis: Pre-generated random values for single-qubit Pauli selection (0-2)
        random_two_qubit: Pre-generated random values for two-qubit Pauli selection (0-14)
        out_ops, out_q1, out_q2: Output arrays (must be large enough)
        
    Returns:
        Length of output circuit
    """
    out_idx = 0
    rand_idx = 0
    
    for i in range(len(base_ops)):
        op = base_ops[i]
        q1 = base_q1[i]
        q2 = base_q2[i]
        
        if op == OP_MEAS_X:
            # Error before measurement
            if random_vals[rand_idx] < error_rate:
                out_ops[out_idx] = OP_Z
                out_q1[out_idx] = q1
                out_q2[out_idx] = -1
                out_idx += 1
            rand_idx += 1
            # Then measurement
            out_ops[out_idx] = op
            out_q1[out_idx] = q1
            out_q2[out_idx] = q2
            out_idx += 1
            
        elif op == OP_MEAS_Z:
            if random_vals[rand_idx] < error_rate:
                out_ops[out_idx] = OP_X
                out_q1[out_idx] = q1
                out_q2[out_idx] = -1
                out_idx += 1
            rand_idx += 1
            out_ops[out_idx] = op
            out_q1[out_idx] = q1
            out_q2[out_idx] = q2
            out_idx += 1
            
        elif op == OP_PREP_X:
            out_ops[out_idx] = op
            out_q1[out_idx] = q1
            out_q2[out_idx] = q2
            out_idx += 1
            if random_vals[rand_idx] < error_rate:
                out_ops[out_idx] = OP_Z
                out_q1[out_idx] = q1
                out_q2[out_idx] = -1
                out_idx += 1
            rand_idx += 1
            
        elif op == OP_PREP_Z:
            out_ops[out_idx] = op
            out_q1[out_idx] = q1
            out_q2[out_idx] = q2
            out_idx += 1
            if random_vals[rand_idx] < error_rate:
                out_ops[out_idx] = OP_X
                out_q1[out_idx] = q1
                out_q2[out_idx] = -1
                out_idx += 1
            rand_idx += 1
            
        elif op == OP_IDLE:
            if random_vals[rand_idx] < error_rate:
                pauli_choice = random_paulis[rand_idx]
                if pauli_choice == 0:
                    out_ops[out_idx] = OP_X
                elif pauli_choice == 1:
                    out_ops[out_idx] = OP_Y
                else:
                    out_ops[out_idx] = OP_Z
                out_q1[out_idx] = q1
                out_q2[out_idx] = -1
                out_idx += 1
            rand_idx += 1
            
        elif op == OP_CNOT:
            out_ops[out_idx] = op
            out_q1[out_idx] = q1
            out_q2[out_idx] = q2
            out_idx += 1
            
            if random_vals[rand_idx] < error_rate:
                err_type = random_two_qubit[rand_idx]
                # Map error type to op code
                if err_type == 0:  # X on control
                    out_ops[out_idx] = OP_X
                    out_q1[out_idx] = q1
                    out_q2[out_idx] = -1
                elif err_type == 1:  # Y on control
                    out_ops[out_idx] = OP_Y
                    out_q1[out_idx] = q1
                    out_q2[out_idx] = -1
                elif err_type == 2:  # Z on control
                    out_ops[out_idx] = OP_Z
                    out_q1[out_idx] = q1
                    out_q2[out_idx] = -1
                elif err_type == 3:  # X on target
                    out_ops[out_idx] = OP_X
                    out_q1[out_idx] = q2
                    out_q2[out_idx] = -1
                elif err_type == 4:  # Y on target
                    out_ops[out_idx] = OP_Y
                    out_q1[out_idx] = q2
                    out_q2[out_idx] = -1
                elif err_type == 5:  # Z on target
                    out_ops[out_idx] = OP_Z
                    out_q1[out_idx] = q2
                    out_q2[out_idx] = -1
                elif err_type == 6:  # XX
                    out_ops[out_idx] = OP_XX
                    out_q1[out_idx] = q1
                    out_q2[out_idx] = q2
                elif err_type == 7:  # YY
                    out_ops[out_idx] = OP_YY
                    out_q1[out_idx] = q1
                    out_q2[out_idx] = q2
                elif err_type == 8:  # ZZ
                    out_ops[out_idx] = OP_ZZ
                    out_q1[out_idx] = q1
                    out_q2[out_idx] = q2
                elif err_type == 9:  # XY
                    out_ops[out_idx] = OP_XY
                    out_q1[out_idx] = q1
                    out_q2[out_idx] = q2
                elif err_type == 10:  # YX
                    out_ops[out_idx] = OP_YX
                    out_q1[out_idx] = q1
                    out_q2[out_idx] = q2
                elif err_type == 11:  # YZ
                    out_ops[out_idx] = OP_YZ
                    out_q1[out_idx] = q1
                    out_q2[out_idx] = q2
                elif err_type == 12:  # ZY
                    out_ops[out_idx] = OP_ZY
                    out_q1[out_idx] = q1
                    out_q2[out_idx] = q2
                elif err_type == 13:  # XZ
                    out_ops[out_idx] = OP_XZ
                    out_q1[out_idx] = q1
                    out_q2[out_idx] = q2
                else:  # ZX (err_type == 14)
                    out_ops[out_idx] = OP_ZX
                    out_q1[out_idx] = q1
                    out_q2[out_idx] = q2
                out_idx += 1
            rand_idx += 1
            
        else:
            # Copy other gates unchanged
            out_ops[out_idx] = op
            out_q1[out_idx] = q1
            out_q2[out_idx] = q2
            out_idx += 1
            
    return out_idx


@njit(cache=True, nogil=True)
def sparsify_syndrome_jit(
    syndrome_history: np.ndarray,
    syn_count: int,
    check_positions: np.ndarray,
    check_ptrs: np.ndarray,
    num_checks: int
) -> np.ndarray:
    """
    Convert raw syndrome history to sparse (differential) form.
    
    For each check, XORs consecutive measurements to get change detection.
    """
    result = syndrome_history[:syn_count].copy()
    
    for c in range(num_checks):
        start = check_ptrs[c]
        end = check_ptrs[c + 1]
        for i in range(start + 1, end):
            pos_curr = check_positions[i]
            pos_prev = check_positions[i - 1]
            if pos_curr < syn_count and pos_prev < syn_count:
                result[pos_curr] ^= syndrome_history[pos_prev]
                
    return result


@njit(cache=True, nogil=True)
def extract_data_state_jit(
    state: np.ndarray,
    data_qubit_indices: np.ndarray
) -> np.ndarray:
    """Extract state of data qubits only."""
    n = len(data_qubit_indices)
    result = np.zeros(n, dtype=np.int8)
    for i in range(n):
        result[i] = state[data_qubit_indices[i]]
    return result
