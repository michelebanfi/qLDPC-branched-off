"""
Circuit-level noise model for Bivariate Bicycle (BB) codes.

This module implements the circuit-level depolarizing noise model from:
Bravyi et al., "High-threshold and low-overhead fault-tolerant quantum memory"

The noise model applies errors at every location in the syndrome extraction circuit:
- CNOT gates: 15 two-qubit Pauli errors with probability p/15 each
- IDLE gates: 3 single-qubit Pauli errors with probability p/3 each  
- State prep: single Pauli error with probability p
- Measurement: single Pauli error with probability p
"""

import numpy as np
from typing import Tuple, List, Dict, Any


class BBCodeCircuit:
    """
    Constructs the syndrome extraction circuit for a Bivariate Bicycle code.
    
    The circuit structure is deduced from the Hx matrix. For BB codes:
    - Hx = [A | B] where A, B are (n/2) x (n/2) matrices
    - Each X-check connects to 6 data qubits (3 from left block, 3 from right)
    - Each Z-check also connects to 6 data qubits
    
    The syndrome extraction uses an interleaved depth-8 circuit where
    X and Z checks are measured simultaneously using CNOTs scheduled
    to avoid collisions.
    """
    
    def __init__(self, Hx: np.ndarray, Hz: np.ndarray, num_cycles: int = 12):
        """
        Initialize the circuit from the parity check matrices.
        
        Args:
            Hx: X-type parity check matrix, shape (m, n)
            Hz: Z-type parity check matrix, shape (m, n)
            num_cycles: Number of syndrome measurement cycles
        """
        self.Hx = np.asarray(Hx, dtype=int)
        self.Hz = np.asarray(Hz, dtype=int)
        self.num_cycles = num_cycles
        
        self.m, self.n = Hx.shape  # m checks, n data qubits
        self.n2 = self.n // 2  # half the data qubits (left/right blocks)
        
        assert self.m == self.n2, f"Expected square blocks: m={self.m}, n2={self.n2}"
        
        # Define qubit ordering: X checks, data_left, data_right, Z checks
        self._setup_qubit_ordering()
        
        # Compute neighbors from Hx/Hz matrices
        self._compute_neighbors()
        
        # Build the CNOT schedule
        self._build_cnot_schedule()
        
        # Build one syndrome measurement cycle
        self._build_cycle()
        
    def _setup_qubit_ordering(self):
        """Define the linear ordering of all qubits (checks + data)."""
        self.lin_order = {}
        self.data_qubits = []
        self.Xchecks = []
        self.Zchecks = []
        
        cnt = 0
        # X check ancillas
        for i in range(self.n2):
            node = ('Xcheck', i)
            self.Xchecks.append(node)
            self.lin_order[node] = cnt
            cnt += 1
        
        # Left data qubits (indices 0 to n2-1)
        for i in range(self.n2):
            node = ('data_left', i)
            self.data_qubits.append(node)
            self.lin_order[node] = cnt
            cnt += 1
            
        # Right data qubits (indices n2 to n-1)
        for i in range(self.n2):
            node = ('data_right', i)
            self.data_qubits.append(node)
            self.lin_order[node] = cnt
            cnt += 1
            
        # Z check ancillas
        for i in range(self.n2):
            node = ('Zcheck', i)
            self.Zchecks.append(node)
            self.lin_order[node] = cnt
            cnt += 1
            
        self.total_qubits = cnt
        
    def _compute_neighbors(self):
        """
        Compute the 6 neighbors of each check qubit from Hx/Hz.
        
        For X-checks: neighbors come from Hx
        For Z-checks: neighbors come from Hz
        
        Each check has exactly 6 neighbors (weight-6 rows).
        We split them: 3 from left data block, 3 from right data block.
        """
        self.nbs = {}
        
        # X-check neighbors from Hx
        for i in range(self.n2):
            check = ('Xcheck', i)
            row = self.Hx[i, :]
            
            # Left neighbors (columns 0 to n2-1)
            left_indices = np.nonzero(row[:self.n2])[0]
            # Right neighbors (columns n2 to n-1)
            right_indices = np.nonzero(row[self.n2:])[0]
            
            assert len(left_indices) == 3, f"X-check {i} has {len(left_indices)} left neighbors"
            assert len(right_indices) == 3, f"X-check {i} has {len(right_indices)} right neighbors"
            
            # Store neighbors with direction indices 0-5
            for j, idx in enumerate(left_indices):
                self.nbs[(check, j)] = ('data_left', idx)
            for j, idx in enumerate(right_indices):
                self.nbs[(check, 3 + j)] = ('data_right', idx)
                
        # Z-check neighbors from Hz
        for i in range(self.n2):
            check = ('Zcheck', i)
            row = self.Hz[i, :]
            
            left_indices = np.nonzero(row[:self.n2])[0]
            right_indices = np.nonzero(row[self.n2:])[0]
            
            assert len(left_indices) == 3, f"Z-check {i} has {len(left_indices)} left neighbors"
            assert len(right_indices) == 3, f"Z-check {i} has {len(right_indices)} right neighbors"
            
            for j, idx in enumerate(left_indices):
                self.nbs[(check, j)] = ('data_left', idx)
            for j, idx in enumerate(right_indices):
                self.nbs[(check, 3 + j)] = ('data_right', idx)
                
    def _build_cnot_schedule(self):
        """
        Build a CNOT schedule that avoids collisions.
        
        The schedule determines which direction (0-5) is used in each time step.
        For X-checks: CNOT from check to data (control=check, target=data)
        For Z-checks: CNOT from data to check (control=data, target=check)
        
        We use a greedy approach to find a collision-free schedule.
        """
        # Try to find a schedule where no two CNOTs use the same data qubit
        # in the same time step
        
        # For each time step, track which data qubits are used
        self.schedule_X = []  # directions for X-checks: list of direction indices
        self.schedule_Z = []  # directions for Z-checks
        
        # Simple approach: schedule all 6 directions sequentially, 
        # interleaving X and Z to minimize idle time
        # This gives depth 8 (similar to researchers' approach)
        
        # Researchers' schedule for [[144,12,12]]:
        # sX = ['idle', 1, 4, 3, 5, 0, 2]
        # sZ = [3, 5, 0, 1, 2, 4, 'idle']
        # 
        # We'll use a similar pattern but derive it from the connectivity
        
        # For now, use a sequential schedule
        # Round 0: Z uses direction 0, X prepares
        # Round 1-5: Both X and Z do CNOTs
        # Round 6: X uses direction 5, Z measures
        # Round 7: All idle, X measures, Z prepares
        
        self.schedule_X = ['idle', 0, 1, 2, 3, 4, 5, 'idle']  # 8 rounds
        self.schedule_Z = [0, 1, 2, 3, 4, 5, 'idle', 'idle']  # 8 rounds
        
    def _build_cycle(self):
        """Build one syndrome measurement cycle as a list of operations."""
        self.cycle = []
        
        num_rounds = 8
        
        for t in range(num_rounds):
            ops_this_round = []
            data_qubits_cnoted = set()
            
            # Round 0: Prep X checks, CNOT Z checks
            if t == 0:
                for q in self.Xchecks:
                    ops_this_round.append(('PrepX', q))
                    
            # X-check CNOTs (control=Xcheck, target=data)
            if self.schedule_X[t] != 'idle':
                direction = self.schedule_X[t]
                for control in self.Xchecks:
                    target = self.nbs[(control, direction)]
                    ops_this_round.append(('CNOT', control, target))
                    data_qubits_cnoted.add(target)
                    
            # Z-check CNOTs (control=data, target=Zcheck)
            if self.schedule_Z[t] != 'idle':
                direction = self.schedule_Z[t]
                for target in self.Zchecks:
                    control = self.nbs[(target, direction)]
                    ops_this_round.append(('CNOT', control, target))
                    data_qubits_cnoted.add(control)
                    
            # Idle for data qubits not involved in CNOTs
            for q in self.data_qubits:
                if q not in data_qubits_cnoted:
                    ops_this_round.append(('IDLE', q))
                    
            # Round 6: Measure Z checks
            if t == 6:
                for q in self.Zchecks:
                    ops_this_round.append(('MeasZ', q))
                    
            # Round 7: Measure X checks, Prep Z checks
            if t == 7:
                for q in self.Xchecks:
                    ops_this_round.append(('MeasX', q))
                for q in self.Zchecks:
                    ops_this_round.append(('PrepZ', q))
                    
            self.cycle.extend(ops_this_round)
            
    def get_full_circuit(self) -> List[Tuple]:
        """Return the full circuit for all syndrome measurement cycles."""
        return self.cycle * self.num_cycles
    
    def get_circuit_with_final_measurements(self) -> List[Tuple]:
        """
        Return the circuit with two additional noiseless cycles at the end.
        
        The researchers add two noiseless syndrome cycles at the end for
        final syndrome extraction.
        """
        noisy_circuit = self.cycle * self.num_cycles
        # Add two more cycles (these will be simulated without noise)
        noiseless_suffix = self.cycle * 2
        return noisy_circuit, noiseless_suffix


def generate_noisy_circuit(
    circuit: List[Tuple], 
    error_rate: float
) -> List[Tuple]:
    """
    Generate a noisy version of the circuit by inserting Pauli errors.
    
    Args:
        circuit: List of gate operations
        error_rate: Physical error rate p
        
    Returns:
        Noisy circuit with error operations inserted
    """
    noisy = []
    
    for gate in circuit:
        gate_type = gate[0]
        
        if gate_type == 'MeasX':
            # Z error before measurement
            if np.random.random() < error_rate:
                noisy.append(('Z', gate[1]))
            noisy.append(gate)
            
        elif gate_type == 'MeasZ':
            # X error before measurement
            if np.random.random() < error_rate:
                noisy.append(('X', gate[1]))
            noisy.append(gate)
            
        elif gate_type == 'PrepX':
            noisy.append(gate)
            # Z error after preparation
            if np.random.random() < error_rate:
                noisy.append(('Z', gate[1]))
                
        elif gate_type == 'PrepZ':
            noisy.append(gate)
            # X error after preparation
            if np.random.random() < error_rate:
                noisy.append(('X', gate[1]))
                
        elif gate_type == 'IDLE':
            # Depolarizing: X, Y, or Z with equal probability
            if np.random.random() < error_rate:
                pauli = ['X', 'Y', 'Z'][np.random.randint(3)]
                noisy.append((pauli, gate[1]))
            # Note: IDLE gates don't propagate, just mark time
            
        elif gate_type == 'CNOT':
            noisy.append(gate)
            # Two-qubit depolarizing: 15 possible Pauli errors
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
            # Pass through any other gates (including error operations)
            noisy.append(gate)
            
    return noisy


def simulate_circuit_Z(
    circuit: List[Tuple],
    lin_order: Dict[Tuple, int],
    n: int,
    Xchecks: List[Tuple]
) -> Tuple[np.ndarray, np.ndarray, Dict, int]:
    """
    Simulate the circuit tracking Z-type errors only.
    
    For decoding X-errors, we track how Z errors propagate through the circuit
    and get detected by X-checks.
    
    Args:
        circuit: List of gate operations
        lin_order: Mapping from qubit names to indices
        n: Number of data qubits
        Xchecks: List of X-check qubit names
        
    Returns:
        syndrome_history: Array of syndrome bits from X-check measurements
        state: Final Z-error state of all qubits
        syndrome_map: Dict mapping each Xcheck to its syndrome positions
        err_cnt: Number of errors in the circuit (for debugging)
    """
    total_qubits = len(lin_order)
    state = np.zeros(total_qubits, dtype=int)
    
    syndrome_history = []
    syndrome_map = {c: [] for c in Xchecks}
    err_cnt = 0
    syn_cnt = 0
    
    for gate in circuit:
        gate_type = gate[0]
        
        if gate_type == 'CNOT':
            # Z errors propagate backwards through CNOT: Z_target -> Z_control * Z_target
            control_idx = lin_order[gate[1]]
            target_idx = lin_order[gate[2]]
            state[control_idx] = (state[control_idx] + state[target_idx]) % 2
            
        elif gate_type == 'PrepX':
            # Prepares |+>, resets Z error to 0
            q = lin_order[gate[1]]
            state[q] = 0
            
        elif gate_type == 'MeasX':
            # Measures X, detects Z errors
            q = lin_order[gate[1]]
            check = gate[1]
            syndrome_history.append(state[q])
            syndrome_map[check].append(syn_cnt)
            syn_cnt += 1
            
        elif gate_type in ['Z', 'Y']:
            # Single-qubit Z or Y error (Y = iXZ, has Z component)
            err_cnt += 1
            q = lin_order[gate[1]]
            state[q] = (state[q] + 1) % 2
            
        elif gate_type in ['ZX', 'YX']:
            # Two-qubit error with Z on first qubit
            err_cnt += 1
            q = lin_order[gate[1]]
            state[q] = (state[q] + 1) % 2
            
        elif gate_type in ['XZ', 'XY']:
            # Two-qubit error with Z on second qubit
            err_cnt += 1
            q = lin_order[gate[2]]
            state[q] = (state[q] + 1) % 2
            
        elif gate_type in ['ZZ', 'YY', 'YZ', 'ZY']:
            # Two-qubit error with Z on both qubits
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
    """
    Simulate the circuit tracking X-type errors only.
    
    For decoding Z-errors, we track how X errors propagate through the circuit
    and get detected by Z-checks.
    
    Args:
        circuit: List of gate operations
        lin_order: Mapping from qubit names to indices
        n: Number of data qubits
        Zchecks: List of Z-check qubit names
        
    Returns:
        syndrome_history: Array of syndrome bits from Z-check measurements
        state: Final X-error state of all qubits
        syndrome_map: Dict mapping each Zcheck to its syndrome positions
        err_cnt: Number of errors in the circuit (for debugging)
    """
    total_qubits = len(lin_order)
    state = np.zeros(total_qubits, dtype=int)
    
    syndrome_history = []
    syndrome_map = {c: [] for c in Zchecks}
    err_cnt = 0
    syn_cnt = 0
    
    for gate in circuit:
        gate_type = gate[0]
        
        if gate_type == 'CNOT':
            # X errors propagate forwards through CNOT: X_control -> X_control * X_target
            control_idx = lin_order[gate[1]]
            target_idx = lin_order[gate[2]]
            state[target_idx] = (state[target_idx] + state[control_idx]) % 2
            
        elif gate_type == 'PrepZ':
            # Prepares |0>, resets X error to 0
            q = lin_order[gate[1]]
            state[q] = 0
            
        elif gate_type == 'MeasZ':
            # Measures Z, detects X errors
            q = lin_order[gate[1]]
            check = gate[1]
            syndrome_history.append(state[q])
            syndrome_map[check].append(syn_cnt)
            syn_cnt += 1
            
        elif gate_type in ['X', 'Y']:
            # Single-qubit X or Y error
            err_cnt += 1
            q = lin_order[gate[1]]
            state[q] = (state[q] + 1) % 2
            
        elif gate_type in ['XZ', 'YZ']:
            # Two-qubit error with X on first qubit
            err_cnt += 1
            q = lin_order[gate[1]]
            state[q] = (state[q] + 1) % 2
            
        elif gate_type in ['ZX', 'ZY']:
            # Two-qubit error with X on second qubit
            err_cnt += 1
            q = lin_order[gate[2]]
            state[q] = (state[q] + 1) % 2
            
        elif gate_type in ['XX', 'YY', 'XY', 'YX']:
            # Two-qubit error with X on both qubits
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
    """
    Apply syndrome sparsification: XOR consecutive syndrome measurements.
    
    This converts the cumulative syndrome to a differential syndrome,
    which is sparser and better for BP decoding.
    
    Args:
        syndrome_history: Raw syndrome measurements
        syndrome_map: Mapping from check names to syndrome positions
        checks: List of check qubit names
        
    Returns:
        Sparsified syndrome
    """
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
    """Extract the error state of data qubits only."""
    return np.array([state[lin_order[q]] for q in data_qubits], dtype=int)


def compute_channel_probs_Z(
    circuit: List[Tuple],
    error_rate: float
) -> np.ndarray:
    """
    Compute channel probabilities for each Z-error location.
    
    This is used to set the initial LLRs for the decoder.
    
    Returns:
        Array of error probabilities for each possible Z-error location
    """
    probs = []
    
    for gate in circuit:
        gate_type = gate[0]
        
        if gate_type == 'MeasX':
            # Z error before measurement
            probs.append(error_rate)
            
        elif gate_type == 'PrepX':
            # Z error after prep
            probs.append(error_rate)
            
        elif gate_type == 'IDLE':
            # Z or Y error (2/3 of depolarizing)
            probs.append(error_rate * 2 / 3)
            
        elif gate_type == 'CNOT':
            # Z on control, Z on target, ZZ on both
            # Each has probability 4/15 of the total CNOT error
            probs.extend([error_rate * 4/15] * 3)
            
    return np.array(probs)


def compute_channel_probs_X(
    circuit: List[Tuple],
    error_rate: float
) -> np.ndarray:
    """
    Compute channel probabilities for each X-error location.
    """
    probs = []
    
    for gate in circuit:
        gate_type = gate[0]
        
        if gate_type == 'MeasZ':
            probs.append(error_rate)
            
        elif gate_type == 'PrepZ':
            probs.append(error_rate)
            
        elif gate_type == 'IDLE':
            probs.append(error_rate * 2 / 3)
            
        elif gate_type == 'CNOT':
            probs.extend([error_rate * 4/15] * 3)
            
    return np.array(probs)


def _simulate_single_Z_circuit(args):
    """Worker function for parallel Z-circuit simulation."""
    idx, circ, noiseless_suffix, lin_order, n, Xchecks, data_qubits, Lx = args
    
    full_circ = circ + noiseless_suffix
    syndrome_history, state, syndrome_map, _ = simulate_circuit_Z(
        full_circ, lin_order, n, Xchecks
    )
    
    state_data = extract_data_qubit_state(state, lin_order, data_qubits)
    logical_syndrome = (Lx @ state_data) % 2
    sparse_syndrome = sparsify_syndrome(syndrome_history, syndrome_map, Xchecks)
    augmented = np.hstack([sparse_syndrome, logical_syndrome])
    supp = tuple(np.nonzero(augmented)[0])
    
    return idx, supp


def _simulate_single_X_circuit(args):
    """Worker function for parallel X-circuit simulation."""
    idx, circ, noiseless_suffix, lin_order, n, Zchecks, data_qubits, Lz = args
    
    full_circ = circ + noiseless_suffix
    syndrome_history, state, syndrome_map, _ = simulate_circuit_X(
        full_circ, lin_order, n, Zchecks
    )
    
    state_data = extract_data_qubit_state(state, lin_order, data_qubits)
    logical_syndrome = (Lz @ state_data) % 2
    sparse_syndrome = sparsify_syndrome(syndrome_history, syndrome_map, Zchecks)
    augmented = np.hstack([sparse_syndrome, logical_syndrome])
    supp = tuple(np.nonzero(augmented)[0])
    
    return idx, supp


def build_decoding_matrices(
    circuit_builder,
    Lx: np.ndarray,
    Lz: np.ndarray,
    error_rate: float,
    verbose: bool = True,
    num_workers: int = None
) -> Dict[str, Any]:
    """
    Build the extended spatio-temporal decoding matrices.
    
    PARALLELIZED for multi-core CPUs (M1 Pro optimized).
    
    This follows the approach from Bravyi et al.:
    1. Generate all possible single-error circuits (one error at each location)
    2. Simulate each to get its sparsified syndrome signature (PARALLELIZED)
    3. Build decoding matrix where columns are distinct syndrome signatures
    4. Merge columns with identical signatures and sum their probabilities
    
    Args:
        circuit_builder: BBCodeCircuit instance
        Lx: X logical operators matrix
        Lz: Z logical operators matrix
        error_rate: Physical error rate
        verbose: Print progress
        num_workers: Number of parallel workers (default: 8 for M1 Pro)
        
    Returns:
        Dictionary containing decoding matrices
    """
    from scipy.sparse import coo_matrix, hstack
    from multiprocessing import Pool, cpu_count
    import tqdm
    
    if num_workers is None:
        num_workers = min(8, cpu_count())
    
    num_cycles = circuit_builder.num_cycles
    n = circuit_builder.n
    n2 = circuit_builder.n2
    k = Lx.shape[0]
    
    lin_order = circuit_builder.lin_order
    data_qubits = circuit_builder.data_qubits
    Xchecks = circuit_builder.Xchecks
    Zchecks = circuit_builder.Zchecks
    
    base_circuit = circuit_builder.get_full_circuit()
    noiseless_suffix = circuit_builder.cycle * 2
    
    total_cycles = num_cycles + 2
    num_syndrome_bits = n2 * total_cycles
    
    # ========== Build Z-error decoding matrix ==========
    if verbose:
        print("Building Z-error decoding matrix...")
    
    # Generate all single-Z-error circuits
    circuitsZ = []
    probsZ = []
    
    head = []
    tail = base_circuit.copy()
    
    for gate in base_circuit:
        gate_type = gate[0]
        
        if gate_type == 'MeasX':
            circuitsZ.append(head + [('Z', gate[1])] + tail)
            probsZ.append(error_rate)
        
        head.append(gate)
        tail.pop(0)
        
        if gate_type == 'PrepX':
            circuitsZ.append(head + [('Z', gate[1])] + tail)
            probsZ.append(error_rate)
        
        if gate_type == 'IDLE':
            circuitsZ.append(head + [('Z', gate[1])] + tail)
            probsZ.append(error_rate * 2/3)
        
        if gate_type == 'CNOT':
            control, target = gate[1], gate[2]
            circuitsZ.append(head + [('Z', control)] + tail)
            probsZ.append(error_rate * 4/15)
            circuitsZ.append(head + [('Z', target)] + tail)
            probsZ.append(error_rate * 4/15)
            circuitsZ.append(head + [('ZZ', control, target)] + tail)
            probsZ.append(error_rate * 4/15)
    
    if verbose:
        print(f"  Generated {len(circuitsZ)} single-Z-error circuits")
        print(f"  Simulating in parallel ({num_workers} workers)...")
    
    # Parallel simulation of Z circuits
    HZdict = {}
    
    args_list = [
        (idx, circ, noiseless_suffix, lin_order, n, Xchecks, data_qubits, Lx)
        for idx, circ in enumerate(circuitsZ)
    ]
    
    with Pool(processes=num_workers) as pool:
        if verbose:
            results = list(tqdm.tqdm(
                pool.imap(_simulate_single_Z_circuit, args_list, chunksize=100),
                total=len(args_list),
                desc="  Z-circuits"
            ))
        else:
            results = pool.map(_simulate_single_Z_circuit, args_list, chunksize=100)
    
    # Collect results
    for idx, supp in results:
        if supp in HZdict:
            HZdict[supp].append(idx)
        else:
            HZdict[supp] = [idx]
    
    if verbose:
        print(f"  Found {len(HZdict)} distinct Z-syndrome signatures")
    
    # Build decoding matrices
    first_logical_rowZ = num_syndrome_bits
    HZ_full_cols = []
    HdecZ_cols = []
    channel_probsZ = []
    
    for supp in HZdict:
        col_full = np.zeros((num_syndrome_bits + k, 1), dtype=int)
        col_full[list(supp), 0] = 1
        HZ_full_cols.append(coo_matrix(col_full))
        
        col_dec = col_full[:num_syndrome_bits, :]
        HdecZ_cols.append(coo_matrix(col_dec))
        
        prob = sum(probsZ[i] for i in HZdict[supp])
        channel_probsZ.append(prob)
    
    HZ_full = hstack(HZ_full_cols).toarray()
    HdecZ = hstack(HdecZ_cols).toarray()
    channel_probsZ = np.array(channel_probsZ)
    
    # ========== Build X-error decoding matrix ==========
    if verbose:
        print("Building X-error decoding matrix...")
    
    circuitsX = []
    probsX = []
    
    head = []
    tail = base_circuit.copy()
    
    for gate in base_circuit:
        gate_type = gate[0]
        
        if gate_type == 'MeasZ':
            circuitsX.append(head + [('X', gate[1])] + tail)
            probsX.append(error_rate)
        
        head.append(gate)
        tail.pop(0)
        
        if gate_type == 'PrepZ':
            circuitsX.append(head + [('X', gate[1])] + tail)
            probsX.append(error_rate)
        
        if gate_type == 'IDLE':
            circuitsX.append(head + [('X', gate[1])] + tail)
            probsX.append(error_rate * 2/3)
        
        if gate_type == 'CNOT':
            control, target = gate[1], gate[2]
            circuitsX.append(head + [('X', control)] + tail)
            probsX.append(error_rate * 4/15)
            circuitsX.append(head + [('X', target)] + tail)
            probsX.append(error_rate * 4/15)
            circuitsX.append(head + [('XX', control, target)] + tail)
            probsX.append(error_rate * 4/15)
    
    if verbose:
        print(f"  Generated {len(circuitsX)} single-X-error circuits")
        print(f"  Simulating in parallel ({num_workers} workers)...")
    
    # Parallel simulation of X circuits
    HXdict = {}
    
    args_list = [
        (idx, circ, noiseless_suffix, lin_order, n, Zchecks, data_qubits, Lz)
        for idx, circ in enumerate(circuitsX)
    ]
    
    with Pool(processes=num_workers) as pool:
        if verbose:
            results = list(tqdm.tqdm(
                pool.imap(_simulate_single_X_circuit, args_list, chunksize=100),
                total=len(args_list),
                desc="  X-circuits"
            ))
        else:
            results = pool.map(_simulate_single_X_circuit, args_list, chunksize=100)
    
    for idx, supp in results:
        if supp in HXdict:
            HXdict[supp].append(idx)
        else:
            HXdict[supp] = [idx]
    
    if verbose:
        print(f"  Found {len(HXdict)} distinct X-syndrome signatures")
    
    first_logical_rowX = num_syndrome_bits
    HX_full_cols = []
    HdecX_cols = []
    channel_probsX = []
    
    for supp in HXdict:
        col_full = np.zeros((num_syndrome_bits + k, 1), dtype=int)
        col_full[list(supp), 0] = 1
        HX_full_cols.append(coo_matrix(col_full))
        
        col_dec = col_full[:num_syndrome_bits, :]
        HdecX_cols.append(coo_matrix(col_dec))
        
        prob = sum(probsX[i] for i in HXdict[supp])
        channel_probsX.append(prob)
    
    HX_full = hstack(HX_full_cols).toarray()
    HdecX = hstack(HdecX_cols).toarray()
    channel_probsX = np.array(channel_probsX)
    
    if verbose:
        print(f"HdecZ shape: {HdecZ.shape}")
        print(f"HdecX shape: {HdecX.shape}")
    
    return {
        'HdecZ': HdecZ,
        'HdecX': HdecX,
        'channel_probsZ': channel_probsZ,
        'channel_probsX': channel_probsX,
        'HZ_full': HZ_full,
        'HX_full': HX_full,
        'first_logical_rowZ': first_logical_rowZ,
        'first_logical_rowX': first_logical_rowX,
        'num_cycles': num_cycles,
        'k': k,
    }

