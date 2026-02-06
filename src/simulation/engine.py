import logging
import os
import numpy as np
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TaskProgressColumn
from multiprocessing import cpu_count, get_context
from scipy.sparse import csr_matrix
from typing import Dict, Any

from ..codes.bb_code import BBCodeCircuit
from ..noise.model import generate_noisy_circuit
from ..noise.simulation import (
    simulate_circuit_Z, simulate_circuit_X, 
    sparsify_syndrome, extract_data_qubit_state,
    run_trial_fast
)
from ..noise.compiled import CompiledCircuit
from ..noise.builder import build_decoding_matrices
from ..decoding.dense import performMinSum_Symmetric
from ..decoding.sparse import performMinSum_Symmetric_Sparse
from ..decoding.alpha import estimate_alpha_alvarado, estimate_alpha_alvarado_autoregressive
from ..decoding.scopt import estimate_scopt_beta
from ..decoding.osd import performOSD_enhanced

_shared_data = None
_console = Console()

_logger = logging.getLogger(__name__)
if not _logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=_console, rich_tracebacks=True, markup=True)],
    )

def _warmup_jit():
    """Warm up JIT compilation for decoders and circuit simulation."""
    from ..noise.kernels import simulate_circuit_Z_jit, simulate_circuit_X_jit, generate_noisy_circuit_jit
    
    # Warm up decoder
    H = np.random.randint(0, 2, (10, 20)).astype(np.float64)
    syndrome = np.random.randint(0, 2, 10).astype(np.int8)
    belief = np.random.randn(20).astype(np.float64)
    performMinSum_Symmetric(H, syndrome, belief, maxIter=2)
    
    # Warm up circuit simulation kernels
    dummy_ops = np.array([1, 4, 2], dtype=np.int32)  # CNOT, MeasX, PrepX
    dummy_q1 = np.array([0, 1, 2], dtype=np.int32)
    dummy_q2 = np.array([1, -1, -1], dtype=np.int32)
    dummy_check_idx = np.array([1], dtype=np.int32)
    dummy_check_ptr = np.array([0, 1], dtype=np.int32)
    
    simulate_circuit_Z_jit(dummy_ops, dummy_q1, dummy_q2, 5, dummy_check_idx, dummy_check_ptr, 10)
    simulate_circuit_X_jit(dummy_ops, dummy_q1, dummy_q2, 5, dummy_check_idx, dummy_check_ptr, 10)
    
    # Warm up noisy circuit generation
    dummy_rand = np.random.random(3)
    dummy_paulis = np.random.randint(0, 3, 3, dtype=np.int32)
    dummy_two = np.random.randint(0, 15, 3, dtype=np.int32)
    out_ops = np.empty(10, dtype=np.int32)
    out_q1 = np.empty(10, dtype=np.int32)
    out_q2 = np.empty(10, dtype=np.int32)
    generate_noisy_circuit_jit(dummy_ops, dummy_q1, dummy_q2, 0.01, dummy_rand, dummy_paulis, dummy_two, out_ops, out_q1, out_q2)


def _run_single_trial_fast(trial_idx, shared_data):
    """Run a single trial using JIT-compiled simulation (fast path)."""
    np.random.seed(shared_data['base_seed'] + trial_idx)
    
    compiled = shared_data['compiled_circuit']
    
    # Run JIT-compiled simulation
    sparse_z, true_z, sparse_x, true_x = run_trial_fast(
        compiled,
        shared_data['error_rate'],
        shared_data['Lx'],
        shared_data['Lz']
    )
    
    # Z decoding
    if shared_data['use_sparse']:
        det_z, succ_z, llrs_z, _ = performMinSum_Symmetric_Sparse(
            shared_data['HdecZ_csr'], sparse_z, shared_data['llrs_z'],
            maxIter=shared_data['maxIter'], alpha=shared_data['alpha_z'],
            alpha_mode=shared_data['alpha_mode']
        )
    else:
        det_z, succ_z, llrs_z, _ = performMinSum_Symmetric(
            shared_data['HdecZ'], sparse_z, shared_data['llrs_z'],
            maxIter=shared_data['maxIter'], alpha=shared_data['alpha_z'],
            alpha_mode=shared_data['alpha_mode']
        )
    
    if not succ_z:
        det_z = performOSD_enhanced(shared_data['HdecZ'], sparse_z, llrs_z, det_z, order=shared_data['osd_order'])
    
    dec_z = (shared_data['HZ_logical'] @ det_z) % 2
    z_err = not np.array_equal(dec_z, true_z)
    
    # X decoding
    if shared_data['use_sparse']:
        det_x, succ_x, llrs_x, _ = performMinSum_Symmetric_Sparse(
            shared_data['HdecX_csr'], sparse_x, shared_data['llrs_x'],
            maxIter=shared_data['maxIter'], alpha=shared_data['alpha_x'],
            alpha_mode=shared_data['alpha_mode']
        )
    else:
        det_x, succ_x, llrs_x, _ = performMinSum_Symmetric(
            shared_data['HdecX'], sparse_x, shared_data['llrs_x'],
            maxIter=shared_data['maxIter'], alpha=shared_data['alpha_x'],
            alpha_mode=shared_data['alpha_mode']
        )
    
    if not succ_x:
        det_x = performOSD_enhanced(shared_data['HdecX'], sparse_x, llrs_x, det_x, order=shared_data['osd_order'])
    
    dec_x = (shared_data['HX_logical'] @ det_x) % 2
    x_err = not np.array_equal(dec_x, true_x)
    
    return (z_err, x_err, z_err or x_err)

def _run_single_trial(trial_idx, shared_data):
    np.random.seed(shared_data['base_seed'] + trial_idx)
    
    noisy_circuit = generate_noisy_circuit(shared_data['base_circuit'], shared_data['error_rate'])
    full_circuit = noisy_circuit + shared_data['noiseless_suffix']
    
    # Z decoding
    syn_z, state_z, map_z, _ = simulate_circuit_Z(full_circuit, shared_data['lin_order'], shared_data['n'], shared_data['Xchecks'])
    true_z = (shared_data['Lx'] @ extract_data_qubit_state(state_z, shared_data['lin_order'], shared_data['data_qubits'])) % 2
    sparse_z = sparsify_syndrome(syn_z, map_z, shared_data['Xchecks'])
    
    if shared_data['use_sparse']:
        det_z, succ_z, llrs_z, _ = performMinSum_Symmetric_Sparse(
            shared_data['HdecZ_csr'], sparse_z, shared_data['llrs_z'],
            maxIter=shared_data['maxIter'], alpha=shared_data['alpha_z'],
            alpha_mode=shared_data['alpha_mode']
        )
    else:
        det_z, succ_z, llrs_z, _ = performMinSum_Symmetric(
            shared_data['HdecZ'], sparse_z, shared_data['llrs_z'],
            maxIter=shared_data['maxIter'], alpha=shared_data['alpha_z'],
            alpha_mode=shared_data['alpha_mode']
        )
    
    if not succ_z:
        det_z = performOSD_enhanced(shared_data['HdecZ'], sparse_z, llrs_z, det_z, order=shared_data['osd_order'])
    
    dec_z = (shared_data['HZ_logical'] @ det_z) % 2
    z_err = not np.array_equal(dec_z, true_z)
    
    # X decoding
    syn_x, state_x, map_x, _ = simulate_circuit_X(full_circuit, shared_data['lin_order'], shared_data['n'], shared_data['Zchecks'])
    true_x = (shared_data['Lz'] @ extract_data_qubit_state(state_x, shared_data['lin_order'], shared_data['data_qubits'])) % 2
    sparse_x = sparsify_syndrome(syn_x, map_x, shared_data['Zchecks'])
    
    if shared_data['use_sparse']:
        det_x, succ_x, llrs_x, _ = performMinSum_Symmetric_Sparse(
            shared_data['HdecX_csr'], sparse_x, shared_data['llrs_x'],
            maxIter=shared_data['maxIter'], alpha=shared_data['alpha_x'],
            alpha_mode=shared_data['alpha_mode']
        )
    else:
        det_x, succ_x, llrs_x, _ = performMinSum_Symmetric(
            shared_data['HdecX'], sparse_x, shared_data['llrs_x'],
            maxIter=shared_data['maxIter'], alpha=shared_data['alpha_x'],
            alpha_mode=shared_data['alpha_mode']
        )
    
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
    """Use fast JIT path by default."""
    return _run_single_trial_fast(trial_idx, _shared_data)

def _worker_task_slow(trial_idx):
    """Fallback to pure Python path."""
    return _run_single_trial(trial_idx, _shared_data)

def run_simulation(
    Hx, Hz, Lx, Lz, error_rate, num_trials=1000, num_cycles=12,
    maxIter=50, osd_order=0, use_dynamic_alpha=True,
    alpha_mode=None, alvarado_alpha=None,
    alpha_estimation_trials=5000, alpha_estimation_bins=50,
    precomputed_matrices=None, num_workers=None, base_seed=None,
    use_jit=True,  # New parameter to enable/disable JIT
    target_logical_errors=None, max_trials=None, scopt=False,
    estimation_plot_dir=None,
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

    if alpha_mode is None:
        alpha_mode = "dynamical" if use_dynamic_alpha else "alvarado"

    alpha_values_z = None
    alpha_values_x = None

    if estimation_plot_dir is not None:
        os.makedirs(estimation_plot_dir, exist_ok=True)

    def _format_rate(rate):
        return f"{rate:.6g}".replace(".", "p")

    if alpha_mode == "alvarado":
        if alvarado_alpha is None:
            # Dynamically compute trials to ensure enough true_1 samples
            # We want at least ~2000 error bits sampled for reliable histogram
            # Expected errors per trial ≈ n * error_rate
            # So trials needed ≈ min_samples / (n * error_rate)
            min_true1_samples = 2000
            n_z = matrices['HdecZ'].shape[1]
            n_x = matrices['HdecX'].shape[1]
            
            # Compute dynamic trials with a reasonable floor and ceiling
            dynamic_trials_z = max(500, min(50000, int(min_true1_samples / (n_z * error_rate))))
            dynamic_trials_x = max(500, min(50000, int(min_true1_samples / (n_x * error_rate))))
            
            # Use user-provided value if explicitly set, otherwise use dynamic
            trials_z = alpha_estimation_trials if alpha_estimation_trials != 5000 else dynamic_trials_z
            trials_x = alpha_estimation_trials if alpha_estimation_trials != 5000 else dynamic_trials_x
            
            _logger.info(
                "Alpha estimation trials: Z=%d, X=%d (dynamic based on n*p)",
                trials_z,
                trials_x,
            )
            
            alpha_z, alpha_r2_z = estimate_alpha_alvarado(
                matrices['HdecZ'], error_rate,
                trials=trials_z,
                bins=alpha_estimation_bins,
                plot_dir=estimation_plot_dir,
                plot_prefix=f"alvarado_{_format_rate(error_rate)}_z",
                llrs=llrs_z
            )
            alpha_x, alpha_r2_x = estimate_alpha_alvarado(
                matrices['HdecX'], error_rate,
                trials=trials_x,
                bins=alpha_estimation_bins,
                plot_dir=estimation_plot_dir,
                plot_prefix=f"alvarado_{_format_rate(error_rate)}_x",
                llrs=llrs_x
            )
            _logger.info(
                "Alvarado alpha (estimated) for p=%.6g: alpha_z=%.6g, alpha_x=%.6g",
                error_rate,
                alpha_z,
                alpha_x,
            )
        elif isinstance(alvarado_alpha, (list, tuple, np.ndarray)) and len(alvarado_alpha) == 2:
            alpha_z, alpha_x = float(alvarado_alpha[0]), float(alvarado_alpha[1])
            alpha_r2_z, alpha_r2_x = None, None
            _logger.info(
                "Alvarado alpha (provided) for p=%.6g: alpha_z=%.6g, alpha_x=%.6g",
                error_rate,
                alpha_z,
                alpha_x,
            )
        else:
            alpha_z = float(alvarado_alpha)
            alpha_x = float(alvarado_alpha)
            alpha_r2_z, alpha_r2_x = None, None
            _logger.info(
                "Alvarado alpha (provided) for p=%.6g: alpha_z=%.6g, alpha_x=%.6g",
                error_rate,
                alpha_z,
                alpha_x,
            )
    elif alpha_mode == "alvarado-autoregressive":
        if alvarado_alpha is not None:
            raise ValueError("alvarado_alpha must be None for alvarado-autoregressive")

        min_true1_samples = 2000
        n_z = matrices['HdecZ'].shape[1]
        n_x = matrices['HdecX'].shape[1]

        dynamic_trials_z = max(500, min(50000, int(min_true1_samples / (n_z * error_rate))))
        dynamic_trials_x = max(500, min(50000, int(min_true1_samples / (n_x * error_rate))))

        trials_z = alpha_estimation_trials if alpha_estimation_trials != 5000 else dynamic_trials_z
        trials_x = alpha_estimation_trials if alpha_estimation_trials != 5000 else dynamic_trials_x

        _logger.info(
            "Autoregressive alpha estimation trials: Z=%d, X=%d (dynamic based on n*p)",
            trials_z,
            trials_x,
        )

        alpha_values_z, alpha_r2_values_z = estimate_alpha_alvarado_autoregressive(
            matrices['HdecZ'], error_rate,
            maxIter=maxIter,
            trials=trials_z,
            bins=alpha_estimation_bins,
            plot_dir=estimation_plot_dir,
            plot_prefix=f"autoregressive_{_format_rate(error_rate)}_z",
            llrs=llrs_z
        )
        alpha_values_x, alpha_r2_values_x = estimate_alpha_alvarado_autoregressive(
            matrices['HdecX'], error_rate,
            maxIter=maxIter,
            trials=trials_x,
            bins=alpha_estimation_bins,
            plot_dir=estimation_plot_dir,
            plot_prefix=f"autoregressive_{_format_rate(error_rate)}_x",
            llrs=llrs_x
        )

        alpha_z = alpha_values_z
        alpha_x = alpha_values_x

        _logger.info(
            "Alvarado autoregressive alpha estimated for p=%.6g (len=%d)",
            error_rate,
            len(alpha_values_z),
        )
    elif alpha_mode == "dynamical":
        alpha_z = 1.0
        alpha_x = 1.0
    else:
        raise ValueError(f"Unsupported alpha_mode: {alpha_mode}")
    
    if scopt:
                
        min_true1_samples = 2000
        n_z = matrices['HdecZ'].shape[1]
        n_x = matrices['HdecX'].shape[1]
        
        dynamic_trials_z = max(500, min(50000, int(min_true1_samples / (n_z * error_rate))))
        dynamic_trials_x = max(500, min(50000, int(min_true1_samples / (n_x * error_rate))))
        
        _logger.info(
                "Beta estimation trials: Z=%d, X=%d (dynamic based on n*p)",
                dynamic_trials_z,
                dynamic_trials_x,
            )
        beta_z, beta_r2_z = estimate_scopt_beta(
            matrices['HdecZ'], error_rate,
            trials=dynamic_trials_z,
            bins=alpha_estimation_bins,
            alpha=alpha_z,
            alpha_mode=alpha_mode,
            maxIter=maxIter,
            plot_dir=estimation_plot_dir,
            plot_prefix=f"scopt_{_format_rate(error_rate)}_z",
            llrs=llrs_z
        )
        beta_x, beta_r2_x = estimate_scopt_beta(
            matrices['HdecX'], error_rate,
            trials=dynamic_trials_x,
            bins=alpha_estimation_bins,
            alpha=alpha_x,
            alpha_mode=alpha_mode,
            maxIter=maxIter,
            plot_dir=estimation_plot_dir,
            plot_prefix=f"scopt_{_format_rate(error_rate)}_x",
            llrs=llrs_x
        )
        _logger.info(
            "SCOPT beta (estimated) for p=%.6g: beta_z=%.6g, beta_x=%.6g",
            error_rate,
            beta_z,
            beta_x,
        )
        
        # TODO: implement the Beta usage in the decoder!
    
    # Build compiled circuit for JIT path
    compiled_circuit = None
    if use_jit:
        compiled_circuit = CompiledCircuit(
            base_circuit=cb.get_full_circuit(),
            noiseless_suffix=cb.cycle * 2,
            lin_order=cb.lin_order,
            data_qubits=cb.data_qubits,
            Xchecks=cb.Xchecks,
            Zchecks=cb.Zchecks
        )
    
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
        'alpha_mode': alpha_mode, 'alpha_z': alpha_z, 'alpha_x': alpha_x,
        'maxIter': maxIter, 'osd_order': osd_order,
        'base_seed': base_seed, 'use_sparse': use_sparse,
        'HdecZ_csr': csr_matrix(matrices['HdecZ']) if use_sparse else None,
        'HdecX_csr': csr_matrix(matrices['HdecX']) if use_sparse else None,
        'compiled_circuit': compiled_circuit,  # Add compiled circuit
    }
    
    # Select worker function based on JIT flag
    worker_fn = _worker_task if use_jit else _worker_task_slow
    
    if max_trials is None:
        max_trials = num_trials if num_trials is not None else 1000000
    stop_on_errors = target_logical_errors is not None and target_logical_errors > 0

    z_errs_count = 0
    x_errs_count = 0
    total_errs_count = 0
    trials_run = 0

    ctx = get_context('spawn')
    with ctx.Pool(processes=num_workers, initializer=_worker_init, initargs=(shared_data,)) as pool:
        iterator = pool.imap(worker_fn, range(max_trials))
        progress = Progress(
            TextColumn("p={task.fields[p]:.4g} | logical={task.fields[errors]}/{task.fields[target]}", justify="left"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        )
        with progress:
            task_id = progress.add_task(
                "simulate",
                total=max_trials,
                p=error_rate,
                errors=0,
                target=(target_logical_errors if stop_on_errors else "∞"),
            )
            for z_err, x_err, total_err in iterator:
                trials_run += 1
                if z_err:
                    z_errs_count += 1
                if x_err:
                    x_errs_count += 1
                if total_err:
                    total_errs_count += 1

                progress.advance(task_id, 1)
                progress.update(task_id, errors=total_errs_count)

                if stop_on_errors and total_errs_count >= target_logical_errors:
                    pool.terminate()
                    break
    
    result = {
        'logical_error_rate': total_errs_count / max(1, trials_run),
        'z_logical_error_rate': z_errs_count / max(1, trials_run),
        'x_logical_error_rate': x_errs_count / max(1, trials_run),
        'num_trials': trials_run,
        'logical_errors': total_errs_count,
    }

    if alpha_mode == "alvarado-autoregressive":
        result['alpha_values_z'] = alpha_values_z
        result['alpha_values_x'] = alpha_values_x
        result['alpha_r2_values_z'] = alpha_r2_values_z
        result['alpha_r2_values_x'] = alpha_r2_values_x
    if alpha_mode == "alvarado":
        result['alpha_r2_z'] = alpha_r2_z
        result['alpha_r2_x'] = alpha_r2_x
    if scopt:
        result['beta_z'] = beta_z
        result['beta_x'] = beta_x
        result['beta_r2_z'] = beta_r2_z
        result['beta_r2_x'] = beta_r2_x

    return result
