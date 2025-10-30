import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D side-effect

# ==============================
# Core parameters (edit freely)
# ==============================
S0 = 100      # initial stock price
K = 100       # strike price
T = 1.0       # time to maturity (years)
r = 0.05      # risk-free rate (continuously compounded)
sigma_base = 0.2  # base volatility for convergence plots

# N (number of paths) grid for convergence
n_sims = np.logspace(3, 5, 8, dtype=int)  # 1,000 → 100,000 in 8 steps
# Volatility grid for sensitivity and 3D surface
vols = np.linspace(0.10, 0.50, 8)         # 10% → 50%

# ------------------------------
# Black–Scholes closed-form call
# ------------------------------
def bs_call(S0, K, r, sigma, T):
    """Analytical European call price under Black–Scholes."""
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# ------------------------------
# Monte Carlo estimator (terminal-only)
# ------------------------------
def mc_call_price(N, S0, K, r, sigma, T, seed=None):
    """
    Monte Carlo European call price:
    - Simulates terminal price S_T via GBM in closed form.
    - Returns (price, stderr) where stderr is the standard error of the MC estimator.
    """
    if seed is not None:
        np.random.seed(seed)
    Z = np.random.standard_normal(N)
    ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    payoff = np.maximum(ST - K, 0.0)
    disc_payoff = np.exp(-r*T)*payoff
    price = np.mean(disc_payoff)
    stderr = np.std(disc_payoff, ddof=1) / np.sqrt(N)
    return price, stderr

# ==============================
# A) Convergence vs Black–Scholes
# ==============================
def plot_convergence(results_dir="results"):
    bs = bs_call(S0, K, r, sigma_base, T)
    prices, errors = [], []

    for N in n_sims:
        p, e = mc_call_price(N, S0, K, r, sigma_base, T)
        prices.append(p); errors.append(e)
        print(f"[Convergence] N={N:>6}  MC={p:.4f} ± {e:.5f}   BS={bs:.4f}")

    plt.figure(figsize=(10, 6))
    plt.errorbar(n_sims, prices, yerr=errors, fmt='o', color='blue',
                 ecolor='lightblue', elinewidth=2, capsize=5, markersize=6,
                 label='Monte Carlo')
    plt.plot(n_sims, prices, color='blue', alpha=0.6)
    plt.axhline(bs, color='red', linestyle='--', linewidth=1.8, label='Black–Scholes (Exact)')
    plt.xscale('log')
    plt.title("Monte Carlo Convergence to Black–Scholes Price")
    plt.xlabel("Number of Simulations (log scale)")
    plt.ylabel("Option Price")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/option_convergence.png", dpi=150)
    plt.show()

# ==============================
# B) Relative Error vs N (log–log)
# ==============================
def plot_relative_error(results_dir="results"):
    bs = bs_call(S0, K, r, sigma_base, T)
    rel_errs = []
    for N in n_sims:
        p, _ = mc_call_price(N, S0, K, r, sigma_base, T)
        rel_errs.append(abs(p - bs)/bs)

    plt.figure(figsize=(9, 5))
    plt.plot(n_sims, rel_errs, 'o-', color='purple')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel("Number of Simulations (log scale)")
    plt.ylabel("Relative Error (log scale)")
    plt.title("Monte Carlo Accuracy Improvement (Relative Error vs N)")
    plt.grid(True, which="both", linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/relative_error.png", dpi=150)
    plt.show()

# ==============================
# C) Volatility Sensitivity
# ==============================
def plot_vol_sensitivity(results_dir="results"):
    mc_prices, bs_prices = [], []
    for vol in vols:
        mc, _ = mc_call_price(50000, S0, K, r, vol, T)  # fixed large N for smoothness
        bs = bs_call(S0, K, r, vol, T)
        mc_prices.append(mc); bs_prices.append(bs)

    plt.figure(figsize=(9, 5))
    plt.plot(vols, mc_prices, 'o-', label='Monte Carlo', color='blue')
    plt.plot(vols, bs_prices, '--', label='Black–Scholes', color='red')
    plt.title("Effect of Volatility on European Call Price")
    plt.xlabel("Volatility σ")
    plt.ylabel("Option Price")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/volatility_sensitivity.png", dpi=150)
    plt.show()

# ==============================
# D) 3D Convergence Surface
# ==============================
def plot_3d_surface(results_dir="results"):
    X, Y = np.meshgrid(np.log10(n_sims), vols)
    Z = np.zeros_like(X)

    for i, vol in enumerate(vols):
        bs = bs_call(S0, K, r, vol, T)
        for j, N in enumerate(n_sims):
            mc, _ = mc_call_price(N, S0, K, r, vol, T)
            Z[i, j] = abs(mc - bs) / bs

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.95)
    ax.set_xlabel("log10(Number of Simulations)")
    ax.set_ylabel("Volatility (σ)")
    ax.set_zlabel("Relative Error")
    ax.set_title("3D Convergence Surface — Error vs σ and N")
    fig.colorbar(surf, ax=ax, shrink=0.65, aspect=12)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/3D_convergence_surface.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    # Run all static plots (saved to results/)
    plot_convergence()
    plot_relative_error()
    plot_vol_sensitivity()
    plot_3d_surface()
