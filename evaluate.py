# evaluate.py

import matplotlib.pyplot as plt

def plot_convergence(convergence, path):
    plt.figure(figsize=(8, 5))
    plt.plot(convergence, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Best Validation Loss")
    plt.title("Ninja Optimization Convergence")
    plt.grid(True)
    plt.savefig(path)
    plt.close()
