import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_steps = 1000      # number of steps in each simulation
n_simulations = 100  # number of random walks
mu = 0.0002         # expected return (drift)
sigma = 0.01        # volatility

# Simulate random walks
for i in range(n_simulations):
    steps = np.random.normal(mu, sigma, n_steps)
    price = 100 * np.exp(np.cumsum(steps))  # start price = 100
    plt.plot(price, linewidth=1)


# Plot results
plt.title("Monte Carlo Simulation of Random Walks")
plt.xlabel("Time Steps")
plt.ylabel("Simulated Price")
plt.grid(True)
plt.savefig("results/random_walks.png", dpi=150)
plt.show()

