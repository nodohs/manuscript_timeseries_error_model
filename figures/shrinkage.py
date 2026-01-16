import numpy as np
import matplotlib.pyplot as plt

# Publication-ready style
plt.style.use("figures.mplstyle")

# Set up the figure with three columns
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

# Sample sizes to plot
K = np.linspace(1, 50, 100)
N = np.linspace(1, 50, 100)

# ============================================
# Column 1: Drift rate (μ) shrinkage
# ============================================
# Example values
kappa_0 = 5.0
mu_j_hat = 2.0
mu_0_hat = 1.0

def shrinkage_mu(K_j, kappa_0, mu_j_hat, mu_0_hat):
    """Shrinkage estimator for drift rate."""
    w_j = K_j / (K_j + kappa_0)
    return w_j * mu_j_hat + (1 - w_j) * mu_0_hat

tilde_mu = shrinkage_mu(K, kappa_0, mu_j_hat, mu_0_hat)

axes[0].plot(K, tilde_mu, 'k-', linewidth=1.5, label=r'$\tilde{\mu}_j$')
axes[0].axhline(mu_j_hat, color='blue', linestyle='--', linewidth=1, 
                label=r'$\hat{\mu}_j$ (device estimate)')
axes[0].axhline(mu_0_hat, color='red', linestyle='--', linewidth=1, 
                label=r'$\hat{\mu}_0$ (prior mean)')
axes[0].set_xlabel(r'$K_j$ (number of checks)')
axes[0].set_ylabel(r'$\tilde{\mu}_j$')
axes[0].set_title(r'Drift Rate $\mu$ Shrinkage')
axes[0].legend(loc='best', fontsize=8)
axes[0].grid(True, alpha=0.3)

# ============================================
# Column 2: Diffusion rate (σ²) shrinkage
# ============================================
# Example values
alpha_0 = 3.0
beta_0 = 2.0
sigma_j_hat_sq = 3.0
mu_j_hat_sigma = 1.5  # device-specific drift estimate
mu_0_hat_sigma = 1.0  # population drift mean
kappa_0_sigma = 5.0

# Prior mean for sigma^2
sigma_0_sq = beta_0 / (alpha_0 - 1)

def shrinkage_sigma2(K_j, alpha_0, beta_0, kappa_0, sigma_j_hat_sq, mu_j_hat, mu_0_hat):
    """Shrinkage estimator for diffusion rate (posterior mean of inverse-gamma)."""
    # Compute beta_j for each K_j
    beta_j = (beta_0 + 
              (K_j * sigma_j_hat_sq) / 2 + 
              (kappa_0 * K_j * (mu_j_hat - mu_0_hat)**2) / (2 * (kappa_0 + K_j)))
    
    # Posterior mean of inverse-gamma
    tilde_sigma_j_sq = beta_j / (alpha_0 + K_j / 2 - 1)
    
    return tilde_sigma_j_sq

tilde_sigma_sq = shrinkage_sigma2(K, alpha_0, beta_0, kappa_0_sigma, 
                                   sigma_j_hat_sq, mu_j_hat_sigma, mu_0_hat_sigma)

axes[1].plot(K, tilde_sigma_sq, 'k-', linewidth=1.5, label=r'$\tilde{\sigma}_j^2$')
axes[1].axhline(sigma_j_hat_sq, color='blue', linestyle='--', linewidth=1, 
                label=r'$\hat{\sigma}_j^2$ (device estimate)')
axes[1].axhline(sigma_0_sq, color='red', linestyle='--', linewidth=1, 
                label=r'$\bar{\sigma}^2$ (prior mean)')
axes[1].set_xlabel(r'$K_j$ (number of checks)')
axes[1].set_ylabel(r'$\tilde{\sigma}_j^2$')
axes[1].set_title(r'Diffusion Rate $\sigma^2$ Shrinkage')
axes[1].legend(loc='best', fontsize=8)
axes[1].grid(True, alpha=0.3)

# ============================================
# Column 3: Check variance (η²) shrinkage
# ============================================
# Example values
alpha_1 = 3.0
eta_j_hat_sq = 2.5
eta_bar_sq = 1.5  # prior mean

def shrinkage_eta2(N_j, alpha_1, eta_j_hat_sq, eta_bar_sq):
    """Shrinkage estimator for check variance."""
    w_j = N_j / (N_j + 2 * (alpha_1 - 1))
    return w_j * eta_j_hat_sq + (1 - w_j) * eta_bar_sq

tilde_eta_sq = shrinkage_eta2(N, alpha_1, eta_j_hat_sq, eta_bar_sq)

axes[2].plot(N, tilde_eta_sq, 'k-', linewidth=1.5, label=r'$\tilde{\eta}_j^2$')
axes[2].axhline(eta_j_hat_sq, color='blue', linestyle='--', linewidth=1, 
                label=r'$\hat{\eta}_j^2$ (device estimate)')
axes[2].axhline(eta_bar_sq, color='red', linestyle='--', linewidth=1, 
                label=r'$\bar{\eta}^2$ (prior mean)')
axes[2].set_xlabel(r'$N_j$ (number of standard intervals)')
axes[2].set_ylabel(r'$\tilde{\eta}_j^2$')
axes[2].set_title(r'Check Variance $\eta^2$ Shrinkage')
axes[2].legend(loc='best', fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('shrinkage_three_column.pdf', bbox_inches='tight')
plt.close()

print("Figure saved as shrinkage_three_column.pdf")
