import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# --- parameters ---
S0 = 100      # initial stock price
mu = 0.07     # drift
sigma = 0.2   # volatility
T = 1         # 1 year
n_steps = 252
n_paths = 10000

# --- simulate GBM paths ---
dt = T / n_steps
S = np.zeros((n_steps + 1, n_paths))
S[0] = S0
for t in range(1, n_steps + 1):
    Z = np.random.randn(n_paths)
    S[t] = S[t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

# --- final prices ---
final_prices = S[-1]

# --- plot histogram ---
count, bins, _ = plt.hist(final_prices, bins=30, density=True, alpha=0.6, label="Simulated Prices")

# --- theoretical log-normal curve ---
shape = sigma * np.sqrt(T)
scale = S0 * np.exp((mu - 0.5 * sigma ** 2) * T)
x = np.linspace(min(final_prices), max(final_prices), 300)
pdf = lognorm.pdf(x, s=shape, scale=scale)
plt.plot(x, pdf, 'r-', lw=2, label='Theoretical Lognormal PDF')

plt.title("GBM Final Price Distribution vs Theoretical Lognormal")
plt.xlabel("Final Price")
plt.ylabel("Probability Density")
plt.legend()
plt.show()
# --- print statistics ---
print(f"Simulated Mean Final Price: {np.mean(final_prices):.2f}")
print(f"Theoretical Mean Final Price: {scale * np.exp(0.5 * shape**2):.2f}")
print(f"Simulated Std Dev of Final Price: {np.std(final_prices):.2f}")
print(f"Theoretical Std Dev of Final Price: {scale * np.sqrt((np.exp(shape**2) - 1) * np.exp(2 * shape**2)):.2f}")
