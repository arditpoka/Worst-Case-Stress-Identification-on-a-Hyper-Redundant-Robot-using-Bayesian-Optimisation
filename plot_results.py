"""plot_results.py — Post-Optimisation Diagnostics and Visualisation
====================================================================

This script loads the trained GP model from the checkpoint saved by
``BO_ANSYS_WAIT.py`` and produces a comprehensive set of diagnostic
plots **without re-running any optimisation**.

Generated figures
-----------------
1. **Predicted vs Actual** (parity plot) on the hold-out test set.
2. **Residual distribution** — histogram and scatter of prediction errors.
3. **BO convergence** — best predicted stress and batch-level metrics
   over the course of the optimisation.
4. **ARD lengthscales** — bar chart showing learnt feature relevance.
5. **GP response surfaces** — 1-D slices through each input dimension
   with 95 % confidence bands, all other inputs held at their midpoint.
6. **Training data overview** — input and stress distributions.
7. **Prediction uncertainty** — predicted mean ± 2 σ vs actual, ranked.

Required files (produced by ``BO_ANSYS_WAIT.py``)
--------------------------------------------------
* ``gp_checkpoint_safe.pt``                           — GP model checkpoint.
* ``new_loads_results.xlsx``                          — Test set (actual stresses).
* ``all_batch_results.xlsx``       *(optional)*       — Per-candidate BO results.
* ``batch_history.xlsx``           *(optional)*       — Per-batch summary metrics.

All figures are saved as PNG files in a ``plots/`` sub-directory next to
this script.

Usage
-----
    python plot_results.py
"""

# ── Standard library ────────────────────────────────────────────────────
from pathlib import Path

# ── Numerical / data ────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ── Plotting ────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")                     # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt           # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

# ── PyTorch / GPyTorch / BoTorch ────────────────────────────────────────
import torch                              # noqa: E402
import gpytorch                           # noqa: E402
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.priors import GammaPrior
from botorch.models import SingleTaskGP


# =====================================================================
#  Configuration
# =====================================================================

DTYPE  = torch.float64
DEVICE = torch.device("cpu")  # Plotting does not benefit from GPU

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR   = SCRIPT_DIR / "plots"

# Files produced by BO_ANSYS_WAIT.py
CHECKPOINT_FILE     = SCRIPT_DIR / "gp_checkpoint_safe.pt"
TEST_FILE           = SCRIPT_DIR / "new_loads_results.xlsx"
BATCH_RESULTS_FILE  = SCRIPT_DIR / "all_batch_results.xlsx"
BATCH_HISTORY_FILE  = SCRIPT_DIR / "batch_history.xlsx"

# Fallback values used only when loading a checkpoint that predates the
# inclusion of bounds/feature-names (backwards compatibility).
_DEFAULT_LOWER = np.array([-37.35, -41.79,  658.7, -1295.0, -178.7])
_DEFAULT_UPPER = np.array([ 35.14,  46.14, 1872.5,  1185.0,  178.8])
_DEFAULT_NAMES = ["Mx", "My", "Rx", "Ry", "Rz"]

# Relative-error denominator floor (same as in BO_ANSYS_WAIT.py)
REL_DENOM_FLOOR = 1.0  # MPa

# Plot aesthetics
plt.rcParams.update({
    "figure.dpi":       150,
    "savefig.dpi":      150,
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "legend.fontsize":  10,
    "figure.facecolor": "white",
})


# =====================================================================
#  Model reconstruction helpers  (mirrors BO_ANSYS_WAIT.py)
# =====================================================================

LOG_EPS = 1e-6


def stress_to_log(y_mpa: np.ndarray) -> np.ndarray:
    """MPa → log-space (clamp positive)."""
    return np.log(np.maximum(y_mpa, LOG_EPS))


def log_to_stress(mu_log, std_log):
    """Log-space mean & std → MPa via lognormal moments."""
    v = np.maximum(std_log, 0.0) ** 2
    mean = np.exp(mu_log + 0.5 * v)
    var  = (np.exp(v) - 1.0) * np.exp(2.0 * mu_log + v)
    return mean, np.sqrt(np.maximum(var, 0.0))


def zunscale_y(y_scaled, mu, sig):
    """Reverse z-score standardisation."""
    return y_scaled * sig + mu


def unit_scale_X(X_phys, lower, upper):
    """Physical → [0, 1] using provided bounds."""
    rng = np.where((upper - lower) > 0, upper - lower, 1.0)
    return np.clip((X_phys - lower) / rng, 0.0, 1.0)


def unit_unscale_X(X_unit, lower, upper):
    """[0, 1] → physical."""
    rng = np.where((upper - lower) > 0, upper - lower, 1.0)
    return X_unit * rng + lower


def build_gp(X, Y):
    """Reconstruct the same GP architecture used during training."""
    base_kernel = MaternKernel(
        nu=2.5, ard_num_dims=5,
        lengthscale_prior=GammaPrior(3.0, 6.0),
        lengthscale_constraint=GreaterThan(1e-3),
    )
    covariance = ScaleKernel(
        base_kernel,
        outputscale_prior=GammaPrior(2.0, 0.15),
        outputscale_constraint=GreaterThan(1e-6),
    )
    likelihood = GaussianLikelihood(
        noise_prior=GammaPrior(1.1, 0.05),
        noise_constraint=GreaterThan(1e-6),
    )
    return SingleTaskGP(
        train_X=X, train_Y=Y,
        covar_module=covariance,
        mean_module=ConstantMean(),
        likelihood=likelihood,
    ).to(DEVICE)


def predict_stress(model, X_unit_np, ylog_mean, ylog_std):
    """Unit-cube inputs → predicted mean and std in MPa."""
    X_t = torch.tensor(X_unit_np, dtype=DTYPE, device=DEVICE)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        post = model.posterior(X_t)
        mu_s  = post.mean.detach().cpu().numpy().ravel()
        var_s = post.variance.detach().cpu().numpy().ravel()

    mu_log  = zunscale_y(mu_s, ylog_mean, ylog_std)
    std_log = np.sqrt(np.maximum(var_s, 0.0)) * ylog_std
    return log_to_stress(mu_log, std_log)


# =====================================================================
#  Load checkpoint
# =====================================================================

def load_checkpoint(path: Path):
    """Load the GP checkpoint and return model + metadata.

    Backwards-compatible: if bounds or feature names are missing from an
    older checkpoint, sensible defaults are used.
    """
    data = torch.load(str(path), map_location="cpu", weights_only=True)

    X_bo = data["X_bo"].to(dtype=DTYPE, device=DEVICE)
    Y_bo = data["Y_bo"].to(dtype=DTYPE, device=DEVICE)

    model = build_gp(X_bo, Y_bo)
    model.load_state_dict(data["model_state_dict"])
    model.eval()

    # Domain bounds (may be absent in older checkpoints)
    if "lower_bounds" in data:
        lower = data["lower_bounds"].numpy().ravel()
        upper = data["upper_bounds"].numpy().ravel()
    else:
        lower, upper = _DEFAULT_LOWER.copy(), _DEFAULT_UPPER.copy()

    feature_names = data.get("feature_names", list(_DEFAULT_NAMES))

    return {
        "model":         model,
        "X_bo":          X_bo,
        "Y_bo":          Y_bo,
        "y_raw_mean":    float(data["y_raw_mean"]),
        "y_raw_std":     float(data["y_raw_std"]),
        "ylog_mean":     float(data["ylog_mean"]),
        "ylog_std":      float(data["ylog_std"]),
        "input_dim":     int(data.get("input_dim", 5)),
        "lower_bounds":  lower,
        "upper_bounds":  upper,
        "feature_names": feature_names,
    }


# =====================================================================
#  Plotting functions
# =====================================================================

def plot_predicted_vs_actual(y_true, y_pred, save_path):
    """1. Parity (45° line) plot: predicted stress vs FEA stress."""
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y_true, y_pred, s=40, alpha=0.7, edgecolors="k", linewidths=0.4)

    # Perfect-prediction line
    lo = min(y_true.min(), y_pred.min()) * 0.95
    hi = max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="Perfect prediction")

    ax.set_xlabel("FEA Stress (MPa)")
    ax.set_ylabel("GP Predicted Stress (MPa)")
    ax.set_title("Predicted vs Actual Stress")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_residuals(y_true, y_pred, save_path):
    """2. Residual analysis: scatter + histogram of (predicted − actual)."""
    residuals = y_pred - y_true

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter: residual vs actual
    ax = axes[0]
    ax.scatter(y_true, residuals, s=30, alpha=0.7, edgecolors="k", linewidths=0.3)
    ax.axhline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("FEA Stress (MPa)")
    ax.set_ylabel("Residual (MPa)")
    ax.set_title("Residuals vs Actual Stress")

    # Histogram
    ax = axes[1]
    ax.hist(residuals, bins=20, edgecolor="k", alpha=0.75)
    ax.axvline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual (MPa)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_convergence(history_df, save_path):
    """3. BO convergence: best predicted stress and batch MAE over iterations."""
    batches = history_df["Batch"].values

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: best predicted stress
    ax = axes[0]
    ax.plot(batches, history_df["best_predicted_stress_MPa"], "o-", color="tab:blue")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Best Predicted Stress (MPa)")
    ax.set_title("Convergence — Best Predicted Stress")
    ax.grid(True, alpha=0.3)

    # Right: batch MAE and RMSE
    ax = axes[1]
    ax.plot(batches, history_df["batch_MAE_MPa"], "s-", label="MAE", color="tab:orange")
    ax.plot(batches, history_df["batch_RMSE"], "^-", label="RMSE", color="tab:red")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Error (MPa)")
    ax.set_title("Batch Prediction Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_batch_metrics(history_df, save_path):
    """4. Per-batch R², mean relative error, and improvement tracking."""
    batches = history_df["Batch"].values

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # R²
    ax = axes[0]
    ax.plot(batches, history_df["batch_R2"], "o-", color="tab:green")
    ax.set_xlabel("Batch")
    ax.set_ylabel("R²")
    ax.set_title("Batch R²")
    ax.grid(True, alpha=0.3)

    # Mean relative error
    ax = axes[1]
    ax.plot(batches, history_df["batch_mean_rel_error"], "s-", color="tab:purple")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Mean Relative Error")
    ax.set_title("Batch Mean Relative Error")
    ax.grid(True, alpha=0.3)

    # Relative improvement + patience counter
    ax = axes[2]
    ax.bar(batches, history_df["rel_predicted_improvement"],
           alpha=0.6, color="tab:cyan", label="Rel. improvement")
    ax.axhline(0.01, color="r", linestyle="--", linewidth=1, label="1 % threshold")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Relative Improvement")
    ax.set_title("Predicted Improvement per Batch")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_lengthscales(model, feature_names, save_path):
    """5. ARD lengthscale bar chart (feature importance proxy).

    Shorter lengthscale → the GP varies more rapidly along that dimension
    → the feature is *more* influential on the prediction.
    """
    ls = (
        model.covar_module.base_kernel.lengthscale
        .detach().cpu().numpy().ravel()
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(feature_names, ls, edgecolor="k", alpha=0.8, color="steelblue")
    ax.set_ylabel("Lengthscale (unit-cube)")
    ax.set_title("ARD Lengthscales — Feature Importance\n(shorter = more influential)")
    ax.grid(axis="y", alpha=0.3)

    # Annotate bar values
    for bar, val in zip(bars, ls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_gp_1d_slices(model, ckpt, save_path):
    """6. 1-D GP response surface slices through each input dimension.

    For each feature, the plot sweeps that feature across its full range
    while holding all other features at their domain midpoint.  The blue
    line is the GP mean and the shaded band is the 95 % confidence
    interval (± 2 σ), both in physical stress (MPa).
    """
    feature_names = ckpt["feature_names"]
    lower = ckpt["lower_bounds"]
    upper = ckpt["upper_bounds"]
    ylog_mean = ckpt["ylog_mean"]
    ylog_std  = ckpt["ylog_std"]
    n_dim = len(feature_names)
    n_pts = 200  # Resolution of each 1-D sweep

    # Midpoint in unit-cube space = 0.5 for every dimension
    midpoint = np.full(n_dim, 0.5)

    fig, axes = plt.subplots(1, n_dim, figsize=(4.5 * n_dim, 4.5), squeeze=False)

    for dim_idx in range(n_dim):
        ax = axes[0, dim_idx]

        # Build query grid: sweep dimension `dim_idx`, hold others at midpoint
        X_unit = np.tile(midpoint, (n_pts, 1))
        X_unit[:, dim_idx] = np.linspace(0.0, 1.0, n_pts)

        # Physical x-axis values for labelling
        x_phys = unit_unscale_X(X_unit, lower, upper)[:, dim_idx]

        mean_mpa, std_mpa = predict_stress(model, X_unit, ylog_mean, ylog_std)

        ax.plot(x_phys, mean_mpa, color="tab:blue", linewidth=1.5, label="GP mean")
        ax.fill_between(
            x_phys,
            mean_mpa - 2 * std_mpa,
            mean_mpa + 2 * std_mpa,
            color="tab:blue", alpha=0.2, label="95 % CI",
        )
        ax.set_xlabel(f"{feature_names[dim_idx]}")
        if dim_idx == 0:
            ax.set_ylabel("Predicted Stress (MPa)")
        ax.set_title(f"GP slice — {feature_names[dim_idx]}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "1-D GP Response Surfaces (other inputs at midpoint)",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_uncertainty(y_true, y_pred, y_std, save_path):
    """7. Prediction uncertainty: ranked predictions with ± 2 σ error bars."""
    # Sort by actual stress for a clear visual
    order = np.argsort(y_true)
    y_true_s = y_true[order]
    y_pred_s = y_pred[order]
    y_std_s  = y_std[order]
    idx = np.arange(len(y_true_s))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(idx, y_true_s, s=30, color="tab:red", zorder=3, label="FEA actual")
    ax.errorbar(
        idx, y_pred_s, yerr=2 * y_std_s,
        fmt="o", markersize=4, color="tab:blue", ecolor="lightblue",
        elinewidth=1.5, capsize=2, label="GP mean ± 2σ",
    )
    ax.set_xlabel("Test sample (sorted by actual stress)")
    ax.set_ylabel("Stress (MPa)")
    ax.set_title("GP Predictions with Uncertainty vs FEA Actual")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_training_data(X_phys, y_stress, feature_names, save_path):
    """8. Training data overview: input histograms + stress distribution."""
    n_dim = X_phys.shape[1]

    fig = plt.figure(figsize=(4 * (n_dim + 1), 4))
    gs = GridSpec(1, n_dim + 1, figure=fig)

    for i in range(n_dim):
        ax = fig.add_subplot(gs[0, i])
        ax.hist(X_phys[:, i], bins=25, edgecolor="k", alpha=0.7, color="steelblue")
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel("Count" if i == 0 else "")
        ax.set_title(f"{feature_names[i]} distribution")

    # Stress distribution
    ax = fig.add_subplot(gs[0, n_dim])
    ax.hist(y_stress, bins=25, edgecolor="k", alpha=0.7, color="salmon")
    ax.set_xlabel("Stress (MPa)")
    ax.set_title("Stress distribution")

    fig.suptitle(f"Training Data Overview  (N = {len(y_stress)})", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_batch_candidates(batch_df, feature_names, save_path):
    """9. Scatter matrix of all BO-proposed candidates coloured by batch."""
    if batch_df is None or len(batch_df) == 0:
        return

    batches = batch_df["Batch"].values
    unique_batches = np.unique(batches)
    cmap = plt.cm.get_cmap("viridis", len(unique_batches))

    n_dim = len(feature_names)
    fig, axes = plt.subplots(n_dim, n_dim, figsize=(3.2 * n_dim, 3.2 * n_dim))

    for i in range(n_dim):
        for j in range(n_dim):
            ax = axes[i, j]
            if i == j:
                # Diagonal: histogram of this feature
                ax.hist(batch_df[feature_names[i]].values,
                        bins=20, edgecolor="k", alpha=0.6, color="steelblue")
            else:
                sc = ax.scatter(
                    batch_df[feature_names[j]].values,
                    batch_df[feature_names[i]].values,
                    c=batches, cmap=cmap, s=15, alpha=0.7, edgecolors="k",
                    linewidths=0.2,
                )
            # Labels only on edges
            if j == 0:
                ax.set_ylabel(feature_names[i], fontsize=9)
            else:
                ax.set_yticklabels([])
            if i == n_dim - 1:
                ax.set_xlabel(feature_names[j], fontsize=9)
            else:
                ax.set_xticklabels([])

    fig.suptitle("BO Candidates — Coloured by Batch", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def print_model_summary(ckpt):
    """Print a concise text summary of the trained GP to the console."""
    model = ckpt["model"]
    print("=" * 60)
    print("  GP MODEL SUMMARY")
    print("=" * 60)
    print(f"  Training samples : {ckpt['X_bo'].shape[0]}")
    print(f"  Input dimensions : {ckpt['input_dim']}")
    print(f"  Features         : {ckpt['feature_names']}")
    print(f"  Raw stress mean  : {ckpt['y_raw_mean']:.4f} MPa")
    print(f"  Raw stress std   : {ckpt['y_raw_std']:.4f} MPa")
    print(f"  Log-stress mean  : {ckpt['ylog_mean']:.4f}")
    print(f"  Log-stress std   : {ckpt['ylog_std']:.4f}")
    print()

    noise = float(model.likelihood.noise.item())
    oscale = float(model.covar_module.outputscale.item())
    mconst = float(model.mean_module.constant.item())
    ls = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()

    print("  Hyperparameters (trained)")
    print(f"    Noise variance : {noise:.6f}")
    print(f"    Output scale   : {oscale:.6f}")
    print(f"    Mean constant  : {mconst:.6f}")
    for name, l in zip(ckpt["feature_names"], ls):
        print(f"    Lengthscale {name:>3s}: {l:.6f}")
    print("=" * 60)
    print()


# =====================================================================
#  Main
# =====================================================================

def main():
    # ── Load checkpoint ──────────────────────────────────────────────
    if not CHECKPOINT_FILE.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_FILE}\n"
            "Run BO_ANSYS_WAIT.py first to train the model."
        )

    print("Loading checkpoint:", CHECKPOINT_FILE.name)
    ckpt = load_checkpoint(CHECKPOINT_FILE)
    model = ckpt["model"]
    feature_names = ckpt["feature_names"]

    print_model_summary(ckpt)

    # ── Create output directory ──────────────────────────────────────
    PLOT_DIR.mkdir(exist_ok=True)

    # ── Recover training data in physical units ──────────────────────
    X_train_unit = ckpt["X_bo"].detach().cpu().numpy()
    Y_train_scaled = ckpt["Y_bo"].detach().cpu().numpy().ravel()

    X_train_phys = unit_unscale_X(X_train_unit, ckpt["lower_bounds"], ckpt["upper_bounds"])
    y_train_log = zunscale_y(Y_train_scaled, ckpt["ylog_mean"], ckpt["ylog_std"])
    y_train_mpa = np.exp(y_train_log)  # Approximate (zero-variance)

    # ── Plot 8: Training data distributions ──────────────────────────
    print("\nGenerating plots...")
    plot_training_data(
        X_train_phys, y_train_mpa, feature_names,
        PLOT_DIR / "08_training_data_overview.png",
    )

    # ── Plot 5: ARD lengthscales ─────────────────────────────────────
    plot_lengthscales(model, feature_names, PLOT_DIR / "05_ard_lengthscales.png")

    # ── Plot 6: 1-D GP slices ────────────────────────────────────────
    plot_gp_1d_slices(model, ckpt, PLOT_DIR / "06_gp_1d_slices.png")

    # ── Test-set evaluation ──────────────────────────────────────────
    if TEST_FILE.exists():
        df_test = pd.read_excel(TEST_FILE, header=None, engine="openpyxl")
        X_test_phys = df_test.iloc[:, :5].apply(pd.to_numeric, errors="coerce").values.astype(float)
        y_test = pd.to_numeric(df_test.iloc[:, 5], errors="coerce").values.astype(float)

        valid = ~(np.isnan(X_test_phys).any(axis=1) | np.isnan(y_test))
        X_test_phys, y_test = X_test_phys[valid], y_test[valid]

        if len(X_test_phys) > 0:
            X_test_unit = unit_scale_X(X_test_phys, ckpt["lower_bounds"], ckpt["upper_bounds"])
            y_pred, y_std = predict_stress(model, X_test_unit, ckpt["ylog_mean"], ckpt["ylog_std"])

            # Metrics summary
            abs_err = np.abs(y_pred - y_test)
            denom = np.maximum(np.abs(y_test), REL_DENOM_FLOOR)
            mae  = float(abs_err.mean())
            rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
            ss_res = float(np.sum((y_test - y_pred) ** 2))
            ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

            print()
            print("  Test-set metrics")
            print(f"    MAE             : {mae:.4f} MPa")
            print(f"    RMSE            : {rmse:.4f} MPa")
            print(f"    Mean rel. error : {float(np.mean(abs_err / denom)):.4%}")
            print(f"    Max  rel. error : {float(np.max(abs_err / denom)):.4%}")
            print(f"    R²              : {r2:.4f}")

            # Plot 1: Parity
            plot_predicted_vs_actual(y_test, y_pred, PLOT_DIR / "01_predicted_vs_actual.png")
            # Plot 2: Residuals
            plot_residuals(y_test, y_pred, PLOT_DIR / "02_residuals.png")
            # Plot 7: Uncertainty
            plot_uncertainty(y_test, y_pred, y_std, PLOT_DIR / "07_prediction_uncertainty.png")
        else:
            print("  No valid test rows found.")
    else:
        print(f"  Test file not found ({TEST_FILE.name}) — skipping test-set plots.")

    # ── Batch history plots (optional) ───────────────────────────────
    if BATCH_HISTORY_FILE.exists():
        hist_df = pd.read_excel(BATCH_HISTORY_FILE, engine="openpyxl")
        # Plot 3: Convergence
        plot_convergence(hist_df, PLOT_DIR / "03_convergence.png")
        # Plot 4: Batch metrics
        plot_batch_metrics(hist_df, PLOT_DIR / "04_batch_metrics.png")
    else:
        print(f"  Batch history not found ({BATCH_HISTORY_FILE.name}) — skipping convergence plots.")

    # ── Batch candidates scatter (optional) ──────────────────────────
    if BATCH_RESULTS_FILE.exists():
        batch_df = pd.read_excel(BATCH_RESULTS_FILE, engine="openpyxl")
        plot_batch_candidates(batch_df, feature_names, PLOT_DIR / "09_batch_candidates_scatter.png")
    else:
        print(f"  Batch results not found ({BATCH_RESULTS_FILE.name}) — skipping candidate scatter.")

    print()
    print(f"All plots saved to: {PLOT_DIR}")
    print("DONE.")


if __name__ == "__main__":
    main()
