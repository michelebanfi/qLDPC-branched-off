import numpy as np
import tqdm
from multiprocessing import cpu_count, get_context
from scipy.sparse import csr_matrix
from typing import Dict, Any

from ..codes.bb_code import BBCodeCircuit
from ..noise.model import generate_noisy_circuit
from ..noise.simulation import simulate_circuit_Z, simulate_circuit_X, sparsify_syndrome, extract_data_qubit_state
from ..noise.builder import build_decoding_matrices
from ..decoding.dense import performMinSum_Symmetric
from ..decoding.sparse import performMinSum_Symmetric_Sparse
from ..decoding.osd import performOSD_enhanced

_shared_data = None

def _warmup_jit():
    H = np.random.randint(0, 2, (10, 20)).astype(np.float64)
    syndrome = np.random.randint(0, 2, 10).astype(np.int8)
    belief = np.random.randn(20).astype(np.float64)
    performMinSum_Symmetric(H, syndrome, belief, maxIter=2)

def _run_single_trial(trial_idx, shared_data):
    np.random.seed(shared_data['base_seed'] + trial_idx)
    
    noisy_circuit = generate_noisy_circuit(shared_data['base_circuit'], shared_data['error_rate'])
    full_circuit = noisy_circuit + shared_data['noiseless_suffix']
    
    # Z decoding
    syn_z, state_z, map_z, _ = simulate_circuit_Z(full_circuit, shared_data['lin_order'], shared_data['n'], shared_data['Xchecks'])
    true_z = (shared_data['Lx'] @ extract_data_qubit_state(state_z, shared_data['lin_order'], shared_data['data_qubits'])) % 2
    sparse_z = sparsify_syndrome(syn_z, map_z, shared_data['Xchecks'])
    
    if shared_data['use_sparse']:
        det_z, succ_z, llrs_z, _ = performMinSum_Symmetric_Sparse(shared_data['HdecZ_csr'], sparse_z, shared_data['llrs_z'], maxIter=shared_data['maxIter'], alpha=shared_data['alpha'])
    else:
        det_z, succ_z, llrs_z, _ = performMinSum_Symmetric(shared_data['HdecZ'], sparse_z, shared_data['llrs_z'], maxIter=shared_data['maxIter'], alpha=shared_data['alpha'])
    
    if not succ_z:
        det_z = performOSD_enhanced(shared_data['HdecZ'], sparse_z, llrs_z, det_z, order=shared_data['osd_order'])
    
    dec_z = (shared_data['HZ_logical'] @ det_z) % 2
    z_err = not np.array_equal(dec_z, true_z)
    
    # X decoding
    syn_x, state_x, map_x, _ = simulate_circuit_X(full_circuit, shared_data['lin_order'], shared_data['n'], shared_data['Zchecks'])
    true_x = (shared_data['Lz'] @ extract_data_qubit_state(state_x, shared_data['lin_order'], shared_data['data_qubits'])) % 2
    sparse_x = sparsify_syndrome(syn_x, map_x, shared_data['Zchecks'])
    
    if shared_data['use_sparse']:
        det_x, succ_x, llrs_x, _ = performMinSum_Symmetric_Sparse(shared_data['HdecX_csr'], sparse_x, shared_data['llrs_x'], maxIter=shared_data['maxIter'], alpha=shared_data['alpha'])
    else:
        det_x, succ_x, llrs_x, _ = performMinSum_Symmetric(shared_data['HdecX'], sparse_x, shared_data['llrs_x'], maxIter=shared_data['maxIter'], alpha=shared_data['alpha'])
    
    if not succ_x:
        det_x = performOSD_enhanced(shared_data['HdecX'], sparse_x, llrs_x, det_x, order=shared_data['osd_order'])
    
    dec_x = (shared_data['HX_logical'] @ det_x) % 2
    x_err = not np.array_equal(dec_x, true_x)
    
    return (z_err, x_err, z_err or x_err)

def _worker_init(shared_data_dict):
    global _shared_data
    _shared_data = shared_data_dict
    _warmup_jit()

def _worker_task(trial_idx):
    return _run_single_trial(trial_idx, _shared_data)

def run_simulation(
    Hx, Hz, Lx, Lz, error_rate, num_trials=1000, num_cycles=12,
    maxIter=50, osd_order=0, use_dynamic_alpha=True,
    precomputed_matrices=None, num_workers=None, base_seed=None,
    **bb_params
):
    if num_workers is None: num_workers = min(8, cpu_count())
    if base_seed is None: base_seed = np.random.randint(0, 2**31)
    
    cb = BBCodeCircuit(Hx, Hz, num_cycles=num_cycles, **bb_params)
    matrices = precomputed_matrices or build_decoding_matrices(cb, Lx, Lz, error_rate, num_workers=num_workers)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        llrs_z = np.clip(np.nan_to_num(np.log((1 - matrices['channel_probsZ']) / matrices['channel_probsZ'])), -50, 50)
        llrs_x = np.clip(np.nan_to_num(np.log((1 - matrices['channel_probsX']) / matrices['channel_probsX'])), -50, 50)
    
    use_sparse = matrices['HdecZ'].shape[1] > 5000
    shared_data = {
        'base_circuit': cb.get_full_circuit(), 'noiseless_suffix': cb.cycle * 2,
        'error_rate': error_rate, 'lin_order': cb.lin_order, 'data_qubits': cb.data_qubits,
        'n': cb.n, 'k': Lx.shape[0], 'Xchecks': cb.Xchecks, 'Zchecks': cb.Zchecks,
        'Lx': Lx, 'Lz': Lz, 'HdecZ': np.asarray(matrices['HdecZ'], dtype=np.float64, order='C'),
        'HdecX': np.asarray(matrices['HdecX'], dtype=np.float64, order='C'), 'llrs_z': llrs_z, 'llrs_x': llrs_x,
        'HZ_full': np.ascontiguousarray(matrices['HZ_full']), 'HX_full': np.ascontiguousarray(matrices['HX_full']),
        'HZ_logical': np.ascontiguousarray(matrices['HZ_full'][matrices['first_logical_rowZ']:matrices['first_logical_rowZ'] + Lx.shape[0]]),
        'HX_logical': np.ascontiguousarray(matrices['HX_full'][matrices['first_logical_rowX']:matrices['first_logical_rowX'] + Lx.shape[0]]),
        'first_logical_rowZ': matrices['first_logical_rowZ'], 'first_logical_rowX': matrices['first_logical_rowX'],
        'alpha': 0 if use_dynamic_alpha else 1.0, 'maxIter': maxIter, 'osd_order': osd_order,
        'base_seed': base_seed, 'use_sparse': use_sparse,
        'HdecZ_csr': csr_matrix(matrices['HdecZ']) if use_sparse else None,
        'HdecX_csr': csr_matrix(matrices['HdecX']) if use_sparse else None,
    }
    
    ctx = get_context('spawn')
    with ctx.Pool(processes=num_workers, initializer=_worker_init, initargs=(shared_data,)) as pool:
        results = list(tqdm.tqdm(pool.imap(_worker_task, range(num_trials)), total=num_trials, desc=f"p={error_rate}"))
        
    z_errs, x_errs, total_errs = zip(*results)
    return {
        'logical_error_rate': np.mean(total_errs),
        'z_logical_error_rate': np.mean(z_errs),
        'x_logical_error_rate': np.mean(x_errs),
        'num_trials': num_trials,
    }
