"""
Circuit conversion utilities for JIT-compiled simulation.
Converts between tuple-based circuits and NumPy array format.
"""
import numpy as np
from typing import List, Tuple, Dict
from .constants import GATE_TO_OPCODE


def circuit_to_arrays(
    circuit: List[Tuple],
    lin_order: Dict[Tuple, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a tuple-based circuit to NumPy arrays for JIT compilation.
    
    Args:
        circuit: List of tuples like ('CNOT', (0,0,'X'), (0,1,'X'))
        lin_order: Dictionary mapping qubit tuples to linear indices
        
    Returns:
        ops: Array of op codes (int32)
        q1: Array of first qubit indices (int32)
        q2: Array of second qubit indices (int32), -1 for single-qubit gates
    """
    n = len(circuit)
    ops = np.empty(n, dtype=np.int32)
    q1 = np.empty(n, dtype=np.int32)
    q2 = np.full(n, -1, dtype=np.int32)
    
    for i, gate in enumerate(circuit):
        gate_type = gate[0]
        ops[i] = GATE_TO_OPCODE.get(gate_type, 0)
        
        if len(gate) >= 2 and gate[1] is not None:
            q1[i] = lin_order[gate[1]]
        else:
            q1[i] = -1
            
        if len(gate) >= 3 and gate[2] is not None:
            q2[i] = lin_order[gate[2]]
            
    return ops, q1, q2


def build_check_arrays(
    checks: List[Tuple],
    lin_order: Dict[Tuple, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build CSR-style arrays for syndrome check positions.
    
    Args:
        checks: List of check tuples (qubit identifiers)
        lin_order: Dictionary mapping qubit tuples to indices
        
    Returns:
        indices: Flattened array of check indices
        ptrs: Pointer array (length = num_checks + 1)
    """
    indices_list = []
    ptrs = [0]
    
    for check in checks:
        idx = lin_order[check]
        indices_list.append(idx)
        ptrs.append(len(indices_list))
        
    return np.array(indices_list, dtype=np.int32), np.array(ptrs, dtype=np.int32)


def build_syndrome_map_arrays(
    checks: List[Tuple],
    circuit: List[Tuple],
    meas_type: str  # 'MeasX' or 'MeasZ'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build arrays mapping checks to their syndrome measurement positions.
    
    Returns:
        positions: Flattened array of syndrome positions for each check
        ptrs: Pointer array for positions per check
    """
    # First pass: count measurements per check
    check_to_idx = {c: i for i, c in enumerate(checks)}
    check_positions = [[] for _ in checks]
    
    syn_idx = 0
    for gate in circuit:
        if gate[0] == meas_type:
            check = gate[1]
            if check in check_to_idx:
                check_positions[check_to_idx[check]].append(syn_idx)
            syn_idx += 1
            
    # Build CSR arrays
    positions = []
    ptrs = [0]
    for pos_list in check_positions:
        positions.extend(pos_list)
        ptrs.append(len(positions))
        
    return np.array(positions, dtype=np.int32), np.array(ptrs, dtype=np.int32)


def count_error_locations(circuit: List[Tuple]) -> int:
    """Count number of potential error locations in circuit."""
    count = 0
    for gate in circuit:
        gate_type = gate[0]
        if gate_type in ('MeasX', 'MeasZ', 'PrepX', 'PrepZ', 'IDLE', 'CNOT'):
            count += 1
    return count


class CompiledCircuit:
    """
    Pre-compiled circuit representation for fast JIT simulation.
    
    Converts the tuple-based circuit format to NumPy arrays once,
    then reuses them for all trials.
    """
    
    def __init__(
        self,
        base_circuit: List[Tuple],
        noiseless_suffix: List[Tuple],
        lin_order: Dict[Tuple, int],
        data_qubits: List[Tuple],
        Xchecks: List[Tuple],
        Zchecks: List[Tuple]
    ):
        self.lin_order = lin_order
        self.total_qubits = len(lin_order)
        
        # Convert base circuit
        self.base_ops, self.base_q1, self.base_q2 = circuit_to_arrays(base_circuit, lin_order)
        
        # Convert suffix
        self.suffix_ops, self.suffix_q1, self.suffix_q2 = circuit_to_arrays(noiseless_suffix, lin_order)
        
        # Combined for reference (without noise)
        full_circuit = base_circuit + noiseless_suffix
        
        # Build syndrome map arrays for sparsification
        self.x_syn_positions, self.x_syn_ptrs = build_syndrome_map_arrays(Xchecks, full_circuit, 'MeasX')
        self.z_syn_positions, self.z_syn_ptrs = build_syndrome_map_arrays(Zchecks, full_circuit, 'MeasZ')
        
        # Check arrays (not used directly in current sim, but kept for compatibility)
        self.x_check_indices, self.x_check_ptrs = build_check_arrays(Xchecks, lin_order)
        self.z_check_indices, self.z_check_ptrs = build_check_arrays(Zchecks, lin_order)
        
        # Data qubit indices
        self.data_qubit_indices = np.array([lin_order[q] for q in data_qubits], dtype=np.int32)
        
        # Count error locations for buffer allocation
        self.num_error_locs = count_error_locations(base_circuit)
        
        # Max possible circuit size (base + suffix + one error per location)
        self.max_circuit_size = len(base_circuit) + len(noiseless_suffix) + self.num_error_locs
        
        # Pre-allocate output buffers for noisy circuit generation
        self.out_ops = np.empty(self.max_circuit_size, dtype=np.int32)
        self.out_q1 = np.empty(self.max_circuit_size, dtype=np.int32)
        self.out_q2 = np.empty(self.max_circuit_size, dtype=np.int32)
        
        # Max syndromes (estimate based on circuit)
        self.max_syndromes_x = sum(1 for g in full_circuit if g[0] == 'MeasX') + 100
        self.max_syndromes_z = sum(1 for g in full_circuit if g[0] == 'MeasZ') + 100
        
        # Number of checks
        self.num_x_checks = len(Xchecks)
        self.num_z_checks = len(Zchecks)
