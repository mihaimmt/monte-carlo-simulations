import numpy as np

# Parameters
S0 = 100      # initial stock price
K = 100       # strike price
T = 1.0       # time to maturity (in years)
r = 0.05      # risk-free rate
sigma = 0.2   # volatility
n_simulations = 100000  # number of Monte Carlo trials

# Step 1: simulate end-of-period prices under geometric Brownian motion
Z = np.random.standard_normal(n_simulations)
ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

# Step 2: compute option payoff
payoffs = np.maximum(ST - K, 0)

# Step 3: discount to present value
C0 = np.exp(-r * T) * np.mean(payoffs)

print(f"Estimated Call Option Price: {C0:.2f}")
