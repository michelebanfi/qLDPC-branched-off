"""
Noise simulation module.

Provides circuit-level noise simulation with both:
- Pure Python implementation (for compatibility/debugging)
- JIT-compiled Numba implementation (50-100x faster)
"""

from .simulation import (
    simulate_circuit_Z,
    simulate_circuit_X,
    sparsify_syndrome,
    extract_data_qubit_state,
    run_trial_fast,
)
from .compiled import CompiledCircuit
from .model import generate_noisy_circuit
from .builder import build_decoding_matrices

__all__ = [
    'simulate_circuit_Z',
    'simulate_circuit_X', 
    'sparsify_syndrome',
    'extract_data_qubit_state',
    'run_trial_fast',
    'CompiledCircuit',
    'generate_noisy_circuit',
    'build_decoding_matrices',
]