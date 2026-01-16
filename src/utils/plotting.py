import matplotlib.pyplot as plt

def plot_simulation_results(results, filename="simulation_results.png"):
    plt.figure(figsize=(10, 7))
    for name, data in results.items():
        ps = sorted(data.keys())
        lers = [data[p]['logical_error_rate'] for p in ps]
        plt.loglog(ps, lers, 'o-', label=f"n={name}")
    
    plt.xlabel("Physical Error Rate p")
    plt.ylabel("Logical Error Rate LER")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.title("Spatio-Temporal Decoding Performance")
    plt.savefig(filename)
    print(f"\nResults saved to {filename}")
