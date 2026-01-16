import numpy as np
import tqdm
from multiprocessing import cpu_count, get_context
from scipy.sparse import coo_matrix, hstack
from typing import Dict, Any

from .simulation import simulate_circuit_Z, simulate_circuit_X, sparsify_syndrome, extract_data_qubit_state

_worker_shared_data = None

def _init_worker_Z(base_circuit, noiseless_suffix, lin_order, n, Xchecks, data_qubits, Lx, error_specs):
    global _worker_shared_data
    _worker_shared_data = {
        'base_circuit': base_circuit,
        'noiseless_suffix': noiseless_suffix,
        'lin_order': lin_order,
        'n': n,
        'Xchecks': Xchecks,
        'data_qubits': data_qubits,
        'Lx': Lx,
        'error_specs': error_specs,
    }

def _init_worker_X(base_circuit, noiseless_suffix, lin_order, n, Zchecks, data_qubits, Lz, error_specs):
    global _worker_shared_data
    _worker_shared_data = {
        'base_circuit': base_circuit,
        'noiseless_suffix': noiseless_suffix,
        'lin_order': lin_order,
        'n': n,
        'Zchecks': Zchecks,
        'data_qubits': data_qubits,
        'Lz': Lz,
        'error_specs': error_specs,
    }

def _simulate_Z_from_spec(idx):
    global _worker_shared_data
    d = _worker_shared_data
    pos, error_tuple = d['error_specs'][idx]
    base = d['base_circuit']
    circ = base[:pos] + [error_tuple] + base[pos:]
    full_circ = circ + d['noiseless_suffix']
    syndrome_history, state, syndrome_map, _ = simulate_circuit_Z(
        full_circ, d['lin_order'], d['n'], d['Xchecks']
    )
    state_data = extract_data_qubit_state(state, d['lin_order'], d['data_qubits'])
    logical_syndrome = (d['Lx'] @ state_data) % 2
    sparse_syndrome = sparsify_syndrome(syndrome_history, syndrome_map, d['Xchecks'])
    augmented = np.hstack([sparse_syndrome, logical_syndrome])
    return idx, tuple(np.nonzero(augmented)[0])

def _simulate_X_from_spec(idx):
    global _worker_shared_data
    d = _worker_shared_data
    pos, error_tuple = d['error_specs'][idx]
    base = d['base_circuit']
    circ = base[:pos] + [error_tuple] + base[pos:]
    full_circ = circ + d['noiseless_suffix']
    syndrome_history, state, syndrome_map, _ = simulate_circuit_X(
        full_circ, d['lin_order'], d['n'], d['Zchecks']
    )
    state_data = extract_data_qubit_state(state, d['lin_order'], d['data_qubits'])
    logical_syndrome = (d['Lz'] @ state_data) % 2
    sparse_syndrome = sparsify_syndrome(syndrome_history, syndrome_map, d['Zchecks'])
    augmented = np.hstack([sparse_syndrome, logical_syndrome])
    return idx, tuple(np.nonzero(augmented)[0])

def build_decoding_matrices(
    circuit_builder,
    Lx: np.ndarray,
    Lz: np.ndarray,
    error_rate: float,
    verbose: bool = True,
    num_workers: int = None
) -> Dict[str, Any]:
    if num_workers is None:
        num_workers = min(8, cpu_count())
    
    n2 = circuit_builder.n2
    k = Lx.shape[0]
    base_circuit = circuit_builder.get_full_circuit()
    noiseless_suffix = circuit_builder.cycle * 2
    num_syndrome_bits = n2 * (circuit_builder.num_cycles + 2)
    
    # Z-error decoding matrix
    if verbose: print("Building Z-error decoding matrix...")
    error_specsZ, probsZ = [], []
    pos = 0
    for gate in base_circuit:
        gate_type = gate[0]
        if gate_type == 'MeasX':
            error_specsZ.append((pos, ('Z', gate[1])))
            probsZ.append(error_rate)
        elif gate_type == 'PrepX':
            error_specsZ.append((pos + 1, ('Z', gate[1])))
            probsZ.append(error_rate)
        elif gate_type == 'IDLE':
            error_specsZ.append((pos + 1, ('Z', gate[1])))
            probsZ.append(error_rate * 2/3)
        elif gate_type == 'CNOT':
            control, target = gate[1], gate[2]
            for err in [('Z', control), ('Z', target), ('ZZ', control, target)]:
                error_specsZ.append((pos + 1, err))
                probsZ.append(error_rate * 4/15)
        pos += 1
        
    ctx = get_context('spawn')
    with ctx.Pool(processes=num_workers, initializer=_init_worker_Z, 
                 initargs=(base_circuit, noiseless_suffix, circuit_builder.lin_order, 
                          circuit_builder.n, circuit_builder.Xchecks, circuit_builder.data_qubits, Lx, error_specsZ)) as pool:
        it = pool.imap(_simulate_Z_from_spec, range(len(error_specsZ)), chunksize=500)
        results = list(tqdm.tqdm(it, total=len(error_specsZ), desc="  Z-circuits")) if verbose else list(it)

    HZdict = {}
    for idx, supp in results: HZdict.setdefault(supp, []).append(idx)
    
    HdecZ_cols, HZ_full_cols, channel_probsZ = [], [], []
    for supp in HZdict:
        col_full = np.zeros((num_syndrome_bits + k, 1), dtype=int)
        col_full[list(supp), 0] = 1
        HZ_full_cols.append(coo_matrix(col_full))
        HdecZ_cols.append(coo_matrix(col_full[:num_syndrome_bits, :]))
        channel_probsZ.append(sum(probsZ[i] for i in HZdict[supp]))
    
    # X-error decoding matrix
    if verbose: print("Building X-error decoding matrix...")
    error_specsX, probsX = [], []
    pos = 0
    for gate in base_circuit:
        gate_type = gate[0]
        if gate_type == 'MeasZ':
            error_specsX.append((pos, ('X', gate[1])))
            probsX.append(error_rate)
        elif gate_type == 'PrepZ':
            error_specsX.append((pos + 1, ('X', gate[1])))
            probsX.append(error_rate)
        elif gate_type == 'IDLE':
            error_specsX.append((pos + 1, ('X', gate[1])))
            probsX.append(error_rate * 2/3)
        elif gate_type == 'CNOT':
            control, target = gate[1], gate[2]
            for err in [('X', control), ('X', target), ('XX', control, target)]:
                error_specsX.append((pos + 1, err))
                probsX.append(error_rate * 4/15)
        pos += 1
        
    with ctx.Pool(processes=num_workers, initializer=_init_worker_X, 
                 initargs=(base_circuit, noiseless_suffix, circuit_builder.lin_order, 
                          circuit_builder.n, circuit_builder.Zchecks, circuit_builder.data_qubits, Lz, error_specsX)) as pool:
        it = pool.imap(_simulate_X_from_spec, range(len(error_specsX)), chunksize=500)
        results = list(tqdm.tqdm(it, total=len(error_specsX), desc="  X-circuits")) if verbose else list(it)

    HXdict = {}
    for idx, supp in results: HXdict.setdefault(supp, []).append(idx)
    
    HdecX_cols, HX_full_cols, channel_probsX = [], [], []
    for supp in HXdict:
        col_full = np.zeros((num_syndrome_bits + k, 1), dtype=int)
        col_full[list(supp), 0] = 1
        HX_full_cols.append(coo_matrix(col_full))
        HdecX_cols.append(coo_matrix(col_full[:num_syndrome_bits, :]))
        channel_probsX.append(sum(probsX[i] for i in HXdict[supp]))
        
    return {
        'HdecZ': hstack(HdecZ_cols).toarray(),
        'HdecX': hstack(HdecX_cols).toarray(),
        'channel_probsZ': np.array(channel_probsZ),
        'channel_probsX': np.array(channel_probsX),
        'HZ_full': hstack(HZ_full_cols).toarray(),
        'HX_full': hstack(HX_full_cols).toarray(),
        'first_logical_rowZ': num_syndrome_bits,
        'first_logical_rowX': num_syndrome_bits,
        'num_cycles': circuit_builder.num_cycles,
        'k': k,
    }
