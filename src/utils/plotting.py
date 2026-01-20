import matplotlib.pyplot as plt

def plot_simulation_results(results, filename="simulation_results.png"):
    colors = ["#2E72AE", "#64B791", "#DBA142", "#000000", "#E17792"]
    plt.figure(figsize=(10, 7))
    for i, (name, data) in enumerate(results.items()):
        ps = sorted(data.keys())
        lers = [data[p]['logical_error_rate'] for p in ps]
        plt.loglog(ps, lers, 'o-', label=f"n={name}", color=colors[i % len(colors)])
    
    plt.xlabel("Physical Error Rate p")
    plt.ylabel("Logical Error Rate LER")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.title("Spatio-Temporal Decoding Performance")
    plt.savefig(filename)
    print(f"\nResults saved to {filename}")
