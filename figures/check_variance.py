#!/usr/bin/env python3
"""
Illustration: variance between two uncertain checkpoints.

Generates an annotated, publication-ready PDF showing:
  - process variance curve:      v_proc(t)  = sigma^2 * phi(t)
  - check uncertainty component: v_check(t) = (1-lambda)^2 v0 + lambda^2 v1
  - total variance:              v_tot(t)   = v_check(t) + v_proc(t)

Matches manuscript Eq. (variance_uncertain_bridge) with
  phi(t) = (t-t0)(t1-t)/(t1-t0) on [t0,t1],
  lambda(t) = (t-t0)/(t1-t0).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def bridge_factor(t: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """phi(t) = (t-t0)(t1-t)/(t1-t0) for t in [t0,t1]; 0 outside."""
    phi = np.zeros_like(t, dtype=float)
    mask = (t >= t0) & (t <= t1)
    denom = (t1 - t0)
    if denom <= 0:
        raise ValueError("Require t1 > t0.")
    phi[mask] = (t[mask] - t0) * (t1 - t[mask]) / denom
    return phi


def interp_weight(t: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """lambda(t) = (t-t0)/(t1-t0) for t in [t0,t1]; clipped outside."""
    denom = (t1 - t0)
    if denom <= 0:
        raise ValueError("Require t1 > t0.")
    lam = (t - t0) / denom
    return np.clip(lam, 0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", type=float, default=0.0, help="Left checkpoint time.")
    ap.add_argument("--t1", type=float, default=10.0, help="Right checkpoint time.")
    ap.add_argument("--v0", type=float, default=0.04, help="Var at left checkpoint.")
    ap.add_argument("--v1", type=float, default=0.09, help="Var at right checkpoint.")
    ap.add_argument("--sigma2", type=float, default=0.02, help="Process diffusion variance rate (sigma^2).")
    ap.add_argument("--n", type=int, default=600, help="Number of points in the curve.")
    ap.add_argument("--open-extend", type=float, default=4.0, help="Time units to extend for open intervals on either side.")
    ap.add_argument("--out", type=str, default="variance_between_uncertain_checkpoints.pdf", help="Output PDF path.")
    args = ap.parse_args()

    t0, t1 = float(args.t0), float(args.t1)
    v0, v1 = float(args.v0), float(args.v1)
    sigma2 = float(args.sigma2)
    open_extend = float(args.open_extend)

    if t1 <= t0:
        raise SystemExit("Error: require t1 > t0.")
    if v0 < 0 or v1 < 0 or sigma2 < 0:
        raise SystemExit("Error: require v0, v1, sigma2 >= 0.")

    # Time grid: open interval before, closed interval, open interval after
    t_left = t0 - open_extend
    t_right = t1 + open_extend
    t = np.linspace(t_left, t_right, args.n)

    # Initialize variance arrays
    v_check = np.zeros_like(t)
    v_proc = np.zeros_like(t)

    # Left open interval: t < t0
    mask_left = t < t0
    v_check[mask_left] = v0  # constant check uncertainty
    v_proc[mask_left] = sigma2 * (t0 - t[mask_left])  # grows backward from t0

    # Closed interval: t0 <= t <= t1
    mask_closed = (t >= t0) & (t <= t1)
    lam_closed = (t[mask_closed] - t0) / (t1 - t0)
    phi_closed = (t[mask_closed] - t0) * (t1 - t[mask_closed]) / (t1 - t0)
    v_check[mask_closed] = (1.0 - lam_closed) ** 2 * v0 + lam_closed ** 2 * v1
    v_proc[mask_closed] = sigma2 * phi_closed

    # Right open interval: t > t1
    mask_right = t > t1
    v_check[mask_right] = v1  # constant check uncertainty
    v_proc[mask_right] = sigma2 * (t[mask_right] - t1)  # grows forward from t1

    v_tot = v_check + v_proc

    # Publication-ready style
    plt.style.use("figures.mpstyle")

    # 12 cm width = 12/2.54 inches
    fig = plt.figure(figsize=(12/2.54, 7/2.54))
    ax = fig.add_subplot(1, 1, 1)

    # Shaded decomposition (stacked areas): [0, v_check] and [v_check, v_tot]
    ax.fill_between(
        t, 0.0, v_proc, alpha=0.18,
        label=r"Wiener-process variance, $v_k(t)=\sigma^2\phi_k(t)$",
    )

    ax.fill_between(
        t, v_proc, v_tot, alpha=0.18,
        label=r"check-measurement variance, $\tilde{v}_k(t)$",
    )

    # Curves
    ax.plot(
        t, v_proc, linewidth=1.6,
        # label=r"$v_k(t)$ (process only)",
    )
    # ax.plot(t, v_check, linewidth=1.6, label=r"$\tilde{v}_k(t)$ (checks only)")
    ax.plot(
        t, v_tot, linewidth=2.0,
        label=r"total variance, $\tilde{v}_k(t)+v_k(t)$",
    )

    # Vertical ma_left, t_right
    ax.axvline(t0, linewidth=1, linestyle="--", color="black")
    ax.axvline(t1, linewidth=1, linestyle="--", color="black")

    # Annotate endpoints
    ax.scatter(
        [t0, t1], [v0, v1], zorder=5,
        label=r"variance of the check, $t=0$ and $t=10$",
        color="black",
        )

    # Annotate intervals
    y_label = max(v_tot) * 0.10
    ax.text((t_left + t0) / 2, y_label, "open", 
            ha="center", va="top", fontsize=9, style="italic")
    ax.text((t0 + t1) / 2, y_label, "closed", 
            ha="center", va="top", fontsize=9, style="italic")
    ax.text((t1 + t_right) / 2, y_label, "open", 
            ha="center", va="top", fontsize=9, style="italic")

    #ax.set_title("Variance between uncertain checkpoints (decomposition)")
    ax.set_xlabel("time $t$")
    ax.set_ylabel(
        r"error variance",
        # $\operatorname{Var}[E_t]$",
        )
    ax.set_xlim(t_left, t_right)
    ax.set_ylim(0.0, max(v_tot) * 1.08)
    ax.locator_params(axis='y', nbins=4)

    # Compact legend, no frame
    ax.legend(loc="upper center", frameon=True, ncol=1, framealpha=1.0)

    # Light grid for readability
    ax.grid(True, linewidth=0.5, alpha=0.25)

    fig.tight_layout()

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
