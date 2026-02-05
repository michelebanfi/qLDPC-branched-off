import numpy as np
import matplotlib.pyplot as plt
import math

def plot_simulation_results(results, filename="simulation_results.png"):
    colors = ["#2E72AE", "#64B791", "#DBA142", "#000000", "#E17792"]
    plt.figure(figsize=(10, 7))
    for i, (name, data) in enumerate(results.items()):
        ps = sorted(data.keys())
        lers = [data[p]['logical_error_rate'] for p in ps]
        color = colors[i % len(colors)]
        plt.loglog(ps, lers, 'o', label=f"n={name}", color=color)

        ps_array = np.array(ps, dtype=float)
        lers_array = np.array(lers, dtype=float)
        mask = (ps_array > 0) & (lers_array > 0)
        log_ps = np.log10(ps_array[mask])
        log_lers = np.log10(lers_array[mask])
        slope, intercept = np.polyfit(log_ps, log_lers, 1)
        
        fit_xmin, fit_xmax = 1e-4, max(ps)
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


def plot_alpha_comparison(results, filename="alpha_comparison.png"):
    """
    Plot autoregressive alpha sequences against the dynamical schedule.
    """
    codes_with_alpha = []
    for name, data in results.items():
        if any('alpha_values_z' in res for res in data.values()):
            codes_with_alpha.append(name)

    if not codes_with_alpha:
        print("No autoregressive alpha values found to plot.")
        return

    n_codes = len(codes_with_alpha)
    ncols = 2 if n_codes > 1 else 1
    nrows = math.ceil(n_codes / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)

    for ax, name in zip(axes.flat, codes_with_alpha):
        data = results[name]
        ps = sorted(data.keys())
        dyn_plotted = False

        for p in ps:
            res = data[p]
            if 'alpha_values_z' not in res:
                continue
            alpha_z = np.asarray(res['alpha_values_z'], dtype=float)
            alpha_x = np.asarray(res.get('alpha_values_x', []), dtype=float) if 'alpha_values_x' in res else None

            iters = np.arange(1, len(alpha_z) + 1)
            ax.plot(iters, alpha_z, label=f"p={p} (Z)")
            if alpha_x is not None and alpha_x.size:
                ax.plot(iters, alpha_x, linestyle='--', label=f"p={p} (X)")

            if not dyn_plotted:
                dynamical = 1.0 - 2.0 ** (-iters)
                ax.plot(iters, dynamical, 'k:', label="dynamical")
                dyn_plotted = True

        ax.set_title(f"n={name}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Alpha")
        ax.grid(True, ls="-", alpha=0.4)
        ax.legend(fontsize=8)

    for idx in range(len(codes_with_alpha), nrows * ncols):
        fig.delaxes(axes.flat[idx])

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"\nAlpha comparison plot saved to {filename}")