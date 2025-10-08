# dice_simulation.py
import numpy as np
import matplotlib.pyplot as plt
import os


os.makedirs("results", exist_ok=True)


rng = np.random.default_rng(42)  # reproducible since seed (42) is the same
N = 10000  # number of dice rolls

def simulate_die_rolls(n):
    rolls = rng.integers(1, 7, size=n)
    counts = np.bincount(rolls, minlength=7)[1:]  # only 1..6
    freqs = counts / n
    return freqs, rolls

freqs, rolls = simulate_die_rolls(N)

# Print results
print("Empirical probabilities:", freqs)
print("Expected probability per face:", 1/6)

# Plot histogram

plt.bar(range(1,7), freqs) # Plots 1 to 6 on the x axis, and the variable "freqs" on the y axis
plt.xlabel("Face")
plt.ylabel("Frequency")
plt.title(f"Dice frequencies (n={N})")

# Save figure in results folder
plt.savefig("results/dice_freqs.png", dpi=150)
plt.show()
