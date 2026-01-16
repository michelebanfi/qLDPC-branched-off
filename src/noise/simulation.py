import numpy as np
from typing import List, Tuple, Dict

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
