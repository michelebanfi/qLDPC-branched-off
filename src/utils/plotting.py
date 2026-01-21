import numpy as np
import matplotlib.pyplot as plt

def plot_simulation_results(results, filename="simulation_results.png"):
    colors = ["#2E72AE", "#64B791", "#DBA142", "#000000", "#E17792"]
    plt.figure(figsize=(10, 7))
    for i, (name, data) in enumerate(results.items()):
        ps = sorted(data.keys())
        lers = [data[p]['logical_error_rate'] for p in ps]
        color = colors[i % len(colors)]
        plt.loglog(ps, lers, 'o', label=f"n={name}", color=color)

        log_ps = np.log10(np.array(ps, dtype=float))
        log_lers = np.log10(np.array(lers, dtype=float))
        interp_log_ps = np.linspace(log_ps.min(), log_ps.max(), 200)
        interp_log_lers = np.interp(interp_log_ps, log_ps, log_lers)
        plt.loglog(10 ** interp_log_ps, 10 ** interp_log_lers, '-', color=color)
    
    plt.xlabel("Physical Error Rate p")
    plt.ylabel("Logical Error Rate LER")
    plt.xlim(1e-4, 1e-2)
    plt.ylim(1e-13, 1e-1)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.title("Spatio-Temporal Decoding Performance")
    plt.savefig(filename)
    print(f"\nResults saved to {filename}")
