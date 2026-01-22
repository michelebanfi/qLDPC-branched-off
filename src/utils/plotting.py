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
        slope, intercept = np.polyfit(log_ps, log_lers, 1)
        fit_xmin, fit_xmax = min(ps), max(ps)
        fit_log_ps = np.linspace(np.log10(fit_xmin), np.log10(fit_xmax), 200)
        fit_log_lers = slope * fit_log_ps + intercept
        plt.loglog(10 ** fit_log_ps, 10 ** fit_log_lers, '-', color=color)
    
    plt.xlabel("Physical Error Rate p")
    plt.ylabel("Logical Error Rate LER")
    plt.xlim(1e-4, 1e-2)
    plt.ylim(1e-7, 0)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.title("Spatio-Temporal Decoding Performance")
    plt.savefig(filename, dpi=300)
    print(f"\nResults saved to {filename}")
