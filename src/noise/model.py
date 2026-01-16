import numpy as np
from typing import List, Tuple

def generate_noisy_circuit(
    circuit: List[Tuple], 
    error_rate: float
) -> List[Tuple]:
    """
    Generate a noisy version of the circuit by inserting Pauli errors.
    """
    noisy = []
    
    for gate in circuit:
        gate_type = gate[0]
        
        if gate_type == 'MeasX':
            if np.random.random() < error_rate:
                noisy.append(('Z', gate[1]))
            noisy.append(gate)
            
        elif gate_type == 'MeasZ':
            if np.random.random() < error_rate:
                noisy.append(('X', gate[1]))
            noisy.append(gate)
            
        elif gate_type == 'PrepX':
            noisy.append(gate)
            if np.random.random() < error_rate:
                noisy.append(('Z', gate[1]))
                
        elif gate_type == 'PrepZ':
            noisy.append(gate)
            if np.random.random() < error_rate:
                noisy.append(('X', gate[1]))
                
        elif gate_type == 'IDLE':
            if np.random.random() < error_rate:
                pauli = ['X', 'Y', 'Z'][np.random.randint(3)]
                noisy.append((pauli, gate[1]))
            
        elif gate_type == 'CNOT':
            noisy.append(gate)
            if np.random.random() < error_rate:
                error_type = np.random.randint(15)
                control, target = gate[1], gate[2]
                two_qubit_paulis = [
                    ('X', control), ('Y', control), ('Z', control),
                    ('X', target), ('Y', target), ('Z', target),
                    ('XX', control, target), ('YY', control, target), ('ZZ', control, target),
                    ('XY', control, target), ('YX', control, target),
                    ('YZ', control, target), ('ZY', control, target),
                    ('XZ', control, target), ('ZX', control, target),
                ]
                noisy.append(two_qubit_paulis[error_type])
        else:
            noisy.append(gate)
            
    return noisy

def compute_channel_probs_Z(circuit: List[Tuple], error_rate: float) -> np.ndarray:
    probs = []
    for gate in circuit:
        gate_type = gate[0]
        if gate_type == 'MeasX': probs.append(error_rate)
        elif gate_type == 'PrepX': probs.append(error_rate)
        elif gate_type == 'IDLE': probs.append(error_rate * 2 / 3)
        elif gate_type == 'CNOT': probs.extend([error_rate * 4/15] * 3)
    return np.array(probs)

def compute_channel_probs_X(circuit: List[Tuple], error_rate: float) -> np.ndarray:
    probs = []
    for gate in circuit:
        gate_type = gate[0]
        if gate_type == 'MeasZ': probs.append(error_rate)
        elif gate_type == 'PrepZ': probs.append(error_rate)
        elif gate_type == 'IDLE': probs.append(error_rate * 2 / 3)
        elif gate_type == 'CNOT': probs.extend([error_rate * 4/15] * 3)
    return np.array(probs)
