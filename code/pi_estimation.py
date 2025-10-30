import numpy as np
import matplotlib.pyplot as plt

# Number of samples
N = 100000
x = np.random.rand(N)
y = np.random.rand(N)

# Points inside the circle
inside = x**2 + y**2 <= 1
pi_estimate = 4 * np.mean(inside)

print(f"Monte Carlo estimate of π: {pi_estimate:.5f}")

# Plot
plt.figure(figsize=(5,5))
plt.scatter(x[inside], y[inside], s=1, color='green', label='Inside circle')
plt.scatter(x[~inside], y[~inside], s=1, color='red', label='Outside circle')
plt.title(f"Monte Carlo π estimation (N={N})")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig("results/pi_estimation.png", dpi=150)
plt.show()
