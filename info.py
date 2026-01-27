import numpy as np
import matplotlib.pyplot as plt

from src.utils.caching import compute_cache_key, load_matrices, save_matrices

experiments = [
    {"code": "[[72, 12, 6]]", "name": "72", "physicalErrorRates": [0.006, 0.005, 0.004], "distance": 6},
    {"code": "[[90, 8, 10]]", "name": "90", "physicalErrorRates": [0.006, 0.005, 0.004,], "distance": 10},
    {"code": "[[108, 8, 10]]", "name": "108", "physicalErrorRates": [0.006, 0.005, 0.004], "distance": 10},
    {"code": "[[144, 12, 12]]", "name": "144", "physicalErrorRates": [0.006, 0.005, 0.004], "distance": 12},
    {"code": "[[288, 12, 18]]", "name": "288", "physicalErrorRates": [0.006, 0.005], "distance": 18},
]


for exp in experiments:
    data = np.load(f"codes/{exp['code']}.npz")
    Hx, Hz = data["Hx"], data["Hz"]
    
    # for p in exp["physicalErrorRates"][:1]:  # Just first rate
    for p in exp["physicalErrorRates"]:
        key = compute_cache_key(Hx, Hz, data["Lx"], data["Lz"], exp["distance"], p)
        matrices = load_matrices("matrix_cache", key)
        
        # print(f"\n{exp['code']} (d={exp['distance']}):")
        # print(f"  Hx shape: {Hx.shape}  (m_x checks x n qubits)")
        # print(f"  HdecZ shape: {matrices['HdecZ'].shape}  (syndrome_bits x fault_classes)")
        # print(f"  Syndrome bits = {matrices['HdecZ'].shape[0]} = {Hx.shape[0]} checks x {exp['distance']} cycles")
        # print(f"  Fault classes = {matrices['HdecZ'].shape[1]}")
        
        # plot the channel_probsz distribution
        channel_probsz = matrices['channel_probsX']
        
        print(min(channel_probsz), max(channel_probsz), np.mean(channel_probsz))
        
        plt.figure(figsize=(8, 4))
        plt.hist(channel_probsz, bins=50, color='blue', alpha=0.7)
        plt.title(f"Channel Probabilities Distribution for Code {exp['code']} at p={p}")
        plt.ylim(0, 10)
        plt.xlabel("Channel Probability")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(f"info_vis/{exp['code']}_p{p}_channel_probsx_hist.png")