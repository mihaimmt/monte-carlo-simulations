import numpy as np
import matplotlib.pyplot as plt

# reproducible random numbers
rng = np.random.default_rng(42)

def simulate_die_rolls(n):
    rolls = rng.integers(1, 7, size=n)
    counts = np.bincount(rolls, minlength=7)[1:]  # index 1..6
    freqs = counts / n
    return freqs, rolls

# simulate 10,000 dice rolls
n = 10000
freqs, rolls = simulate_die_rolls(n)
print("Empirical probabilities:", freqs)
print("Expected probability per face:", 1/6)

# plot histogram
plt.bar(range(1, 7), freqs)
plt.xlabel("Face")
plt.ylabel("Frequency")
plt.title(f"Dice frequencies (n={n})")
plt.savefig("../results/dice_freqs.png", dpi=150)
plt.show()
