# Circuit-Level Noise Model for Bivariate Bicycle Codes

This document provides a comprehensive explanation of the circuit-level depolarizing noise model implemented in this codebase, based on the approach from **Bravyi et al., "High-threshold and low-overhead fault-tolerant quantum memory"** ([arXiv:2308.07915](https://arxiv.org/abs/2308.07915)).

---

## Table of Contents

1. [Overview](#overview)
2. [Bivariate Bicycle Code Structure](#bivariate-bicycle-code-structure)
3. [Syndrome Extraction Circuit](#syndrome-extraction-circuit)
4. [Depolarizing Noise Model](#depolarizing-noise-model)
5. [Error Propagation Through the Circuit](#error-propagation-through-the-circuit)
6. [Spatio-Temporal Decoding](#spatio-temporal-decoding)
7. [Syndrome Sparsification](#syndrome-sparsification)
8. [Building the Decoding Matrix](#building-the-decoding-matrix)
9. [Decoding Flow](#decoding-flow)
10. [Potential Issues with Your Current Implementation](#potential-issues-with-your-current-implementation)

---

## Overview

The circuit-level noise model simulates **realistic** quantum errors that occur during the syndrome extraction process. Unlike the phenomenological or code-capacity noise models, circuit-level noise:

- Applies errors at **every gate location** (CNOTs, idles, preparations, measurements)
- Models **error propagation** through the circuit (e.g., errors spreading via CNOT gates)
- Requires **spatio-temporal decoding** that considers the full syndrome history across multiple cycles

The key idea is that a single physical error can cause a complex pattern of syndrome changes across both space (different check qubits) and time (different measurement rounds).

---

## Bivariate Bicycle Code Structure

Bivariate Bicycle (BB) codes are CSS codes defined by parity check matrices:

```
Hx = [A | B]
Hz = [B^T | A^T]
```

Where `A` and `B` are `(n/2) × (n/2)` circulant matrices constructed from cyclic shift operators. For the `[[144,12,12]]` code:

- **Code length**: `n = 144` data qubits (split into left block: 0-71, right block: 72-143)
- **Logical qubits**: `k = 12`
- **Code distance**: `d = 12`

### Qubit Layout

The linear ordering of all qubits in the circuit is:

```
[X-check ancillas] [Left data qubits] [Right data qubits] [Z-check ancillas]
     (n/2)              (n/2)              (n/2)              (n/2)
```

Each check qubit connects to exactly **6 data qubits** (weight-6 checks):
- **3 neighbors in the left block** (via matrix A or B^T)
- **3 neighbors in the right block** (via matrix B or A^T)

---

## Syndrome Extraction Circuit

### Depth-8 Interleaved Schedule

The syndrome extraction uses an **interleaved depth-8 circuit** where X-checks and Z-checks are measured **simultaneously**. This is more efficient than sequential measurement.

```
┌─────────┬─────────────────────────────────────────────────────┬───────────┐
│  Round  │           Operations                                │   Notes   │
├─────────┼─────────────────────────────────────────────────────┼───────────┤
│    0    │ PrepX(all X-checks), CNOT(Z-checks, direction 0)    │ Z starts  │
│    1    │ CNOT(X-checks, dir 0), CNOT(Z-checks, dir 1)        │           │
│    2    │ CNOT(X-checks, dir 1), CNOT(Z-checks, dir 2)        │           │
│    3    │ CNOT(X-checks, dir 2), CNOT(Z-checks, dir 3)        │           │
│    4    │ CNOT(X-checks, dir 3), CNOT(Z-checks, dir 4)        │           │
│    5    │ CNOT(X-checks, dir 4), CNOT(Z-checks, dir 5)        │           │
│    6    │ CNOT(X-checks, dir 5), MeasZ(all Z-checks)          │ Z ends    │
│    7    │ MeasX(all X-checks), PrepZ(all Z-checks), IDLE data │ X ends    │
└─────────┴─────────────────────────────────────────────────────┴───────────┘
```

> [!IMPORTANT]
> In the **original Bravyi et al. code**, the schedule is specifically optimized for the `[[144,12,12]]` code:
> - `sX = ['idle', 1, 4, 3, 5, 0, 2]`
> - `sZ = [3, 5, 0, 1, 2, 4, 'idle']`
> 
> This schedule ensures no two CNOTs use the same data qubit simultaneously.

### CNOT Directions

For each check qubit, the 6 neighbors are indexed by "directions" 0-5:
- **Directions 0-2**: Neighbors in the left data block
- **Directions 3-5**: Neighbors in the right data block

### CNOT Orientation

| Check Type | CNOT Direction | Control | Target |
|------------|----------------|---------|--------|
| X-check    | Forward        | X-check ancilla | Data qubit |
| Z-check    | Backward       | Data qubit | Z-check ancilla |

This is because:
- X-checks need to accumulate X-errors from data qubits (CNOT propagates X forward: control → target)
- Z-checks need to accumulate Z-errors from data qubits (CNOT propagates Z backward: target → control)

---

## Depolarizing Noise Model

The noise model applies errors probabilistically at each gate location with a base error rate `p`:

### Error Rates by Gate Type

| Gate Type | Error Applied | Probability | When |
|-----------|--------------|-------------|------|
| **CNOT** | 15 two-qubit Paulis | `p/15` each | After gate |
| **IDLE** | Single-qubit X, Y, or Z | `p/3` each | During idle |
| **PrepX** | Z error | `p` | After prep |
| **PrepZ** | X error | `p` | After prep |
| **MeasX** | Z error | `p` | Before meas |
| **MeasZ** | X error | `p` | Before meas |

### Two-Qubit Errors (CNOT)

When a CNOT gate experiences an error, one of 15 possible two-qubit Pauli errors is applied:

```
Single-qubit on control: IX, IY, IZ  (3 errors)
Single-qubit on target:  XI, YI, ZI  (3 errors)
Two-qubit correlated:    XX, YY, ZZ, XY, YX, YZ, ZY, XZ, ZX  (9 errors)
```

> [!NOTE]
> The identity error (II) is implicitly part of the "no error" probability `(1-p)`.

---

## Error Propagation Through the Circuit

### Z-Error Propagation (Relevant for X-check detection)

Z-errors propagate **backwards** through CNOT gates:

```
Before CNOT:   Z_target → After CNOT: Z_control ⊗ Z_target
```

This is because: `CNOT · (I ⊗ Z) · CNOT = Z ⊗ Z`

**Tracking Z-errors allows us to predict X-check syndrome measurements.**

### X-Error Propagation (Relevant for Z-check detection)

X-errors propagate **forwards** through CNOT gates:

```
Before CNOT:   X_control → After CNOT: X_control ⊗ X_target  
```

This is because: `CNOT · (X ⊗ I) · CNOT = X ⊗ X`

**Tracking X-errors allows us to predict Z-check syndrome measurements.**

### Simulation Functions

The simulation separates X and Z error tracking:

```python
# For decoding Z-errors on data qubits (affects X-checks):
simulate_circuit_Z() → tracks Z-component of all errors → X-check syndrome

# For decoding X-errors on data qubits (affects Z-checks):  
simulate_circuit_X() → tracks X-component of all errors → Z-check syndrome
```

---

## Spatio-Temporal Decoding

### The Problem

In the code-capacity model, errors occur only on data qubits, and syndromes are perfect. In circuit-level noise:

1. **Errors occur everywhere**: data qubits, ancilla qubits, during gates
2. **Syndromes are noisy**: measurement errors cause spurious syndrome bits
3. **Errors propagate**: a single fault creates complex syndrome patterns

### The Solution: Extended Decoding Matrix

Instead of using the static parity check matrix `Hx` or `Hz`, we construct an **extended spatio-temporal decoding matrix** `Hdec` where:

- **Rows** = all syndrome bit positions across all cycles
- **Columns** = distinct syndrome "signatures" from all possible single-error events

Each column represents the syndrome pattern caused by a **specific single error** at a **specific location** in the circuit.

### Multi-Cycle Syndrome Structure

For `T` cycles, each check produces `T` syndrome measurements:

```
Syndrome vector = [cycle_0_syndromes | cycle_1_syndromes | ... | cycle_T-1_syndromes]
                  └──── n/2 bits ────┘└──── n/2 bits ────┘     └────  n/2 bits  ────┘
```

The two noiseless cycles at the end ensure we capture the final syndrome state.

---

## Syndrome Sparsification

Raw syndrome measurements are **cumulative** - errors persist until they're corrected. This creates dense syndrome patterns that are hard to decode.

### Sparsification: XOR with Previous

For each check, we XOR consecutive syndrome measurements:

```python
sparse_syndrome[cycle_i] = raw_syndrome[cycle_i] ⊕ raw_syndrome[cycle_i - 1]
```

This converts the syndrome to a **differential** form where:
- `1` = syndrome **changed** from previous cycle (error occurred recently)
- `0` = syndrome same as previous cycle

> [!TIP]
> Sparsified syndromes are much sparser and work better with BP-based decoders.

---

## Building the Decoding Matrix

The `build_decoding_matrices()` function constructs `HdecZ` and `HdecX` as follows:

### Step 1: Enumerate All Single-Error Locations

For each gate in the circuit, generate circuits with a single error:

```python
# For Z-type errors:
- MeasX: Z error before measurement (prob = p)
- PrepX: Z error after preparation (prob = p)
- IDLE:  Z error (prob = 2p/3, since Z and Y both have Z-component)
- CNOT:  Z_control, Z_target, ZZ_both (prob = 4p/15 each)
```

### Step 2: Simulate Each Single-Error Circuit

```python
for each single_error_circuit:
    1. Append 2 noiseless cycles at the end
    2. Simulate to get syndrome_history and final data qubit state
    3. Compute logical syndrome: logical = Lx @ data_state (mod 2)
    4. Apply sparsification to syndrome_history
    5. Create augmented signature: [sparsified_syndrome | logical]
```

### Step 3: Merge Identical Signatures

Errors at different locations can produce **identical syndrome signatures**. These are merged:

```python
HZdict = {}  # Map: signature_tuple → list of circuit indices

for idx, signature in enumerate(all_signatures):
    key = tuple(nonzero_positions(signature))
    if key in HZdict:
        HZdict[key].append(idx)  # Merge
    else:
        HZdict[key] = [idx]
```

### Step 4: Build Final Matrix

```python
for each distinct signature:
    - Add column to HdecZ (syndrome part only)
    - Add column to HZ_full (syndrome + logical)
    - Sum probabilities: channel_prob[col] = sum(prob[merged_circuits])
```

### Output Structure

```
HdecZ: shape (num_syndrome_bits, num_distinct_signatures)
HZ_full: shape (num_syndrome_bits + k, num_distinct_signatures)
         └─── syndrome rows ───┘└── logical rows ──┘

channel_probsZ: probability for each column (used for initial LLRs)
first_logical_rowZ: index where logical rows start in HZ_full
```

---

## Decoding Flow

### Step 1: Generate Noisy Circuit

```python
noisy_circuit = generate_noisy_circuit(base_circuit, error_rate)
full_circuit = noisy_circuit + noiseless_suffix  # 2 extra cycles
```

### Step 2: Simulate and Get Syndrome

```python
# For Z-error decoding:
syndrome_history_z, state_z, syn_map_z, _ = simulate_circuit_Z(full_circuit, ...)

# Get actual logical state (ground truth):
data_state_z = extract_data_qubit_state(state_z, ...)
true_logical_z = (Lx @ data_state_z) % 2

# Sparsify:
sparse_syn_z = sparsify_syndrome(syndrome_history_z, syn_map_z, Xchecks)
```

### Step 3: Decode with BP + OSD

```python
# Convert channel probabilities to LLRs:
llrs = log((1 - channel_probs) / channel_probs)

# Run BP (Min-Sum variant):
detection, success, final_llrs = performMinSum_Symmetric(
    HdecZ, sparse_syn_z, llrs, maxIter=10000
)

# If BP fails, use OSD post-processing:
if not success:
    detection = performOSD_enhanced(HdecZ, sparse_syn_z, final_llrs, detection, order=7)
```

### Step 4: Check Logical Error

```python
# Apply detection to full matrix (including logical rows):
decoded_syndrome_full = (HZ_full @ detection) % 2

# Extract decoded logical:
decoded_logical = decoded_syndrome_full[first_logical_rowZ : first_logical_rowZ + k]

# Compare with true logical:
z_error = not np.array_equal(decoded_logical, true_logical_z)
```

---

## Potential Issues with Your Current Implementation

Based on your terminal output showing very high logical error rates (e.g., 100% X-logical errors even at p=0.001), here are potential issues:

### 1. **BP/OSD Parameters**

Your current settings:
```python
maxIter = 50      # Original uses 10000
osd_order = 0     # Original uses 7
```

> [!CAUTION]
> **These are drastically lower than the original paper!**
> - `maxIter = 50` is too few iterations for BP to converge on complex syndromes
> - `osd_order = 0` provides no OSD post-processing (OSD-0 = just Gaussian elimination)

**Fix**: Use `maxIter = 10000` and `osd_order = 7` as in the original.

### 2. **Decoder Library**

The original code uses the **ldpc** library's `bposd_decoder`:
```python
from ldpc import bposd_decoder
bpdX = bposd_decoder(HdecX, channel_probs=channel_probsX, ...)
```

Your implementation uses a custom Min-Sum + OSD. Potential issues:
- LLR initialization might differ
- Min-Sum scaling factor handling
- OSD implementation correctness

### 3. **CNOT Schedule**

Your schedule in `circuit_noise.py`:
```python
self.schedule_X = ['idle', 0, 1, 2, 3, 4, 5, 'idle']  # Sequential
self.schedule_Z = [0, 1, 2, 3, 4, 5, 'idle', 'idle']  # Sequential
```

Original optimized schedule:
```python
sX = ['idle', 1, 4, 3, 5, 0, 2]
sZ = [3, 5, 0, 1, 2, 4, 'idle']
```

> [!WARNING]
> The optimized schedule matters for:
> - Minimizing circuit depth
> - Proper interleaving of X and Z checks
> - Avoiding qubit collisions

The sequential schedule may cause different error propagation patterns.

### 4. **Invalid Value Warning**

```
RuntimeWarning: invalid value encountered in subtract
  Q_new = np.where(mask, values - R_new, 0)
```

This indicates numerical issues in your BP implementation:
- Possible division by zero in LLR calculation
- Inf/NaN values propagating through the decoder

### 5. **Syndrome Bit Count Check**

Make sure:
```python
num_syndrome_bits = n2 * (num_cycles + 2)  # Including 2 noiseless cycles
assert HdecZ.shape[0] == len(sparse_syn_z)
```

---

## Summary Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    CIRCUIT-LEVEL SIMULATION                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐      ┌──────────────┐     ┌────────────────┐  │
│   │ Base Circuit│ ──▶  │ Add Noise    │ ──▶ │ + 2 Noiseless  │  │
│   │ (T cycles)  │      │ (dep. model) │     │ Cycles         │  │
│   └─────────────┘      └──────────────┘     └────────────────┘  │
│                                                     │           │
│                                                     ▼           │
│   ┌───────────────────────────────────────────────────────────┐ │
│   │              SIMULATE (track Z or X errors)               │ │
│   │  • CNOT propagation (backwards for Z, forwards for X)     │ │
│   │  • Prep resets error, Meas records syndrome               │ │
│   └───────────────────────────────────────────────────────────┘ │
│                               │                                 │
│                               ▼                                 │
│   ┌────────────────┐   ┌─────────────────┐   ┌───────────────┐  │
│   │ Syndrome       │   │ Sparsify        │   │ True Logical  │  │
│   │ History        │ ──│ (XOR consec.)   │   │ (Lx @ data)   │  │
│   └────────────────┘   └─────────────────┘   └───────────────┘  │
│           │                    │                     │          │
│           │                    ▼                     │          │
│           │         ┌──────────────────┐             │          │
│           │         │  BP + OSD Decoder│             │          │
│           │         │  (using HdecZ)   │             │          │
│           │         └──────────────────┘             │          │
│           │                    │                     │          │
│           │                    ▼                     │          │
│           │         ┌──────────────────┐             │          │
│           │         │ Apply to HZ_full │             │          │
│           │         │ ──▶ decoded_logical │          │          │
│           │         └──────────────────┘             │          │
│           │                    │                     │          │
│           │                    ▼                     ▼          │
│           │         ┌─────────────────────────────────┐         │
│           │         │  Compare: decoded vs true_logical │       │
│           │         │  ──▶ Logical Error if different   │       │
│           │         └─────────────────────────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference: Key Files

| File | Purpose |
|------|---------|
| `circuit_noise.py` | Circuit construction, noise model, simulation, matrix building |
| `main_circuit.py` | Monte Carlo simulation driver |
| `decoding.py` | BP and OSD implementations |
| `matrix_cache.py` | Caching of precomputed decoding matrices |

---

## References

1. Bravyi, S. et al. "High-threshold and low-overhead fault-tolerant quantum memory" [arXiv:2308.07915](https://arxiv.org/abs/2308.07915)
2. Original code: [github.com/sbravyi/BivariateBicycleCodes](https://github.com/sbravyi/BivariateBicycleCodes)
3. LDPC library: [pypi.org/project/ldpc](https://pypi.org/project/ldpc/)
