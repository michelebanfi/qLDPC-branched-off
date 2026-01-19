"""
Op codes for JIT-compiled circuit simulation.
These integer codes replace string-based gate types for Numba compatibility.
"""
import numpy as np

# Gate type op codes
OP_CNOT = 1
OP_PREP_X = 2
OP_PREP_Z = 3
OP_MEAS_X = 4
OP_MEAS_Z = 5
OP_IDLE = 6

# Single-qubit Pauli errors
OP_X = 10
OP_Y = 11
OP_Z = 12

# Two-qubit Pauli errors (first letter = control, second = target)
OP_XX = 20
OP_XY = 21
OP_XZ = 22
OP_YX = 23
OP_YY = 24
OP_YZ = 25
OP_ZX = 26
OP_ZY = 27
OP_ZZ = 28

# Mapping from string gate types to op codes
GATE_TO_OPCODE = {
    'CNOT': OP_CNOT,
    'PrepX': OP_PREP_X,
    'PrepZ': OP_PREP_Z,
    'MeasX': OP_MEAS_X,
    'MeasZ': OP_MEAS_Z,
    'IDLE': OP_IDLE,
    'X': OP_X,
    'Y': OP_Y,
    'Z': OP_Z,
    'XX': OP_XX,
    'XY': OP_XY,
    'XZ': OP_XZ,
    'YX': OP_YX,
    'YY': OP_YY,
    'YZ': OP_YZ,
    'ZX': OP_ZX,
    'ZY': OP_ZY,
    'ZZ': OP_ZZ,
}

# Two-qubit Pauli error codes for random selection (15 types)
TWO_QUBIT_ERROR_OPCODES = np.array([
    OP_X, OP_Y, OP_Z,      # Single qubit on control (indices 0-2)
    OP_X, OP_Y, OP_Z,      # Single qubit on target (indices 3-5) - handled specially
    OP_XX, OP_YY, OP_ZZ,   # Same Pauli on both (indices 6-8)
    OP_XY, OP_YX,          # Mixed (indices 9-10)
    OP_YZ, OP_ZY,          # Mixed (indices 11-12)
    OP_XZ, OP_ZX,          # Mixed (indices 13-14)
], dtype=np.int32)

# Which target for single-qubit errors in two-qubit context
# 0 = control (q1), 1 = target (q2), 2 = both
TWO_QUBIT_ERROR_TARGET = np.array([
    0, 0, 0,  # X, Y, Z on control
    1, 1, 1,  # X, Y, Z on target  
    2, 2, 2,  # XX, YY, ZZ on both
    2, 2,     # XY, YX on both
    2, 2,     # YZ, ZY on both
    2, 2,     # XZ, ZX on both
], dtype=np.int32)
