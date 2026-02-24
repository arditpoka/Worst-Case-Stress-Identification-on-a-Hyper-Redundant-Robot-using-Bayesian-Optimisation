"""BO_ANSYS_WAIT.py — ANSYS-in-the-Loop Bayesian Optimisation
===========================================================

This script performs iterative Bayesian Optimisation (BO) to explore
load configurations that maximise stress in a structural FEA model.
At every iteration a batch of candidate load vectors is proposed, sent
to ANSYS Mechanical for evaluation, and the results are fed back to
refine a Gaussian Process (GP) surrogate model.

Key design choices
------------------
* **Unit-cube scaling** – Input features are mapped to [0, 1] using known
  physical bounds, which is the recommended representation for BoTorch.
* **Log-stress GP** – Stress values are log-transformed before modelling
  (lognormal assumption), improving fit for strictly-positive, right-skewed
  response data.
* **ARD Matérn-5/2 kernel with priors** – Automatic Relevance Determination
  lengthscales with Gamma priors stabilise training; a noise floor prevents
  the likelihood from collapsing.
* **Batch diversity** – ``optimize_acqf(..., sequential=True)`` generates
  each candidate conditioned on the previous ones, reducing spatial
  clustering.
* **Early stopping** – Optimisation halts when the best predicted stress
  improves by less than 1 % for four consecutive batches.

Inputs
------
* ``results_with_loads.xlsx``  – Initial training data (headerless).
  Columns: Mx, My, Rx, Ry, Rz, Stress  (5 loads + 1 response).
* ``new_loads_results.xlsx``   – Hold-out test set for final evaluation.

ANSYS handshake files
---------------------
* ``ansys_test_data.xlsx``  – Written by this script (candidates for ANSYS).
* ``test_results.xlsx``     – Written by ANSYS (FEA stress results).

Outputs
-------
* ``gp_checkpoint_safe.pt``                           – Resumable GP checkpoint.
* ``all_batch_results.xlsx``                          – Every evaluated candidate.
* ``batch_history.xlsx``                              – Per-batch summary metrics.
* ``new_loads_results_with_actual_and_predicted.xlsx`` – Final test predictions.

Usage
-----
    python BO_ANSYS_WAIT.py

The script automatically detects whether a checkpoint exists and resumes
from it.  ANSYS can be launched manually or via the optional ``ANSYS_CMD``
variable.
"""

# ── Standard library ────────────────────────────────────────────────────
import time
import subprocess
from pathlib import Path

# ── Numerical / data ────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ── PyTorch / GPyTorch ──────────────────────────────────────────────────
import torch
import gpytorch
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior

# ── BoTorch ─────────────────────────────────────────────────────────────
from botorch.models import SingleTaskGP
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler

# `fit_gpytorch_mll` was renamed across BoTorch versions; handle both.
try:
    from botorch.fit import fit_gpytorch_mll as fit_gpytorch_model
except ImportError:
    from botorch.fit import fit_gpytorch_model


# =====================================================================
#  Global configuration
# =====================================================================

# -- Tensor defaults --------------------------------------------------
torch.set_default_dtype(torch.float64)
DTYPE  = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Reproducibility --------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

# -- Paths ------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
EXCEL_TRAIN  = SCRIPT_DIR / "results_with_loads.xlsx"   # Initial training data
TEST_FILE    = SCRIPT_DIR / "new_loads_results.xlsx"     # Hold-out test set
ANSYS_INPUT  = SCRIPT_DIR / "ansys_test_data.xlsx"       # Candidates → ANSYS
ANSYS_OUTPUT = SCRIPT_DIR / "test_results.xlsx"          # ANSYS → stresses
CHECKPOINT   = SCRIPT_DIR / "gp_checkpoint_safe.pt"      # Resumable model state

# Optional: shell command to launch ANSYS automatically (None = manual).
ANSYS_CMD = None

# -- Bayesian optimisation parameters --------------------------------
BATCH_SIZE   = 10    # Candidates per BO iteration
MAX_BATCHES  = 10    # Hard upper limit on iterations
NUM_RESTARTS = 10    # Multi-start restarts for acquisition optimisation
RAW_SAMPLES  = 512   # Sobol points seeding the restarts
ACQ_MAXITER  = 250   # L-BFGS-B iterations inside each restart

# -- Early-stopping parameters ---------------------------------------
IMPROVEMENT_THRESHOLD = 0.01  # Minimum relative improvement (1 %)
PATIENCE              = 4     # Consecutive stagnant batches before stopping

# -- ANSYS file-polling parameters ------------------------------------
ANSYS_WAIT_TIMEOUT  = 60 * 60  # Max wait for ANSYS output (seconds, 1 h)
ANSYS_POLL_INTERVAL = 3        # Seconds between existence checks
FILE_STABLE_SECONDS = 3        # File size must be constant for this long
READ_RETRIES        = 8        # Retry attempts for a locked Excel file
READ_RETRY_DELAY    = 2        # Seconds between read retries

# -- Metrics ----------------------------------------------------------
REL_DENOM_FLOOR = 1.0  # MPa floor in relative-error denominator

# -- Physical domain bounds (unscaled) --------------------------------
#    x = [Mx, My, Rx, Ry, Rz]
LOWER_BOUNDS = np.array([-37.35, -41.79,  658.7, -1295.0, -178.7])
UPPER_BOUNDS = np.array([ 35.14,  46.14, 1872.5,  1185.0,  178.8])
RANGE_BOUNDS = np.where(
    (UPPER_BOUNDS - LOWER_BOUNDS) > 0,
    UPPER_BOUNDS - LOWER_BOUNDS,
    1.0,  # Safeguard against zero-width dimensions
)


# =====================================================================
#  Input scaling helpers  (unit-cube mapping)
# =====================================================================

def unit_scale_X(X_phys: np.ndarray) -> np.ndarray:
    """Map physical-unit inputs to the [0, 1] hypercube.

    Each feature is linearly scaled using the known domain bounds so that
    0 corresponds to ``LOWER_BOUNDS`` and 1 to ``UPPER_BOUNDS``.
    Values that fall outside the bounds are clipped.
    """
    return np.clip((X_phys - LOWER_BOUNDS) / RANGE_BOUNDS, 0.0, 1.0)


def unit_unscale_X(X_unit: np.ndarray) -> np.ndarray:
    """Map [0, 1] hypercube inputs back to physical units."""
    return X_unit * RANGE_BOUNDS + LOWER_BOUNDS


# =====================================================================
#  Output (stress) transforms  –  lognormal assumption
# =====================================================================

LOG_EPS = 1e-6  # Small constant to avoid log(0)


def stress_to_log(y_mpa: np.ndarray) -> np.ndarray:
    """Transform stress values (MPa) to log-space.

    Values below ``LOG_EPS`` are clamped before taking the logarithm to
    avoid numerical issues with zero or negative stresses.
    """
    return np.log(np.maximum(y_mpa, LOG_EPS))


def log_to_stress(
    mu_log: np.ndarray,
    std_log: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert log-space GP predictions back to MPa via lognormal moments.

    If Y ~ LogNormal(mu, sigma²), then:
        E[Y]   = exp(mu + sigma²/2)
        Var[Y] = [exp(sigma²) - 1] · exp(2·mu + sigma²)

    Parameters
    ----------
    mu_log  : Mean in log-space (unscaled).
    std_log : Standard deviation in log-space (unscaled).

    Returns
    -------
    mean_mpa : Lognormal mean in MPa.
    std_mpa  : Lognormal standard deviation in MPa.
    """
    variance = np.maximum(std_log, 0.0) ** 2
    mean_mpa = np.exp(mu_log + 0.5 * variance)
    var_mpa  = (np.exp(variance) - 1.0) * np.exp(2.0 * mu_log + variance)
    std_mpa  = np.sqrt(np.maximum(var_mpa, 0.0))
    return mean_mpa, std_mpa


def zscale_y(y: np.ndarray, mu: float, sig: float) -> np.ndarray:
    """Standardise (z-score) an array given pre-computed mean and std."""
    return (y - mu) / sig


def zunscale_y(y_scaled: np.ndarray, mu: float, sig: float) -> np.ndarray:
    """Reverse z-score standardisation."""
    return y_scaled * sig + mu


# =====================================================================
#  Checkpoint persistence
# =====================================================================

def save_checkpoint(
    path: Path,
    model: SingleTaskGP,
    X_bo: torch.Tensor,
    Y_bo: torch.Tensor,
    y_raw_mean: float,
    y_raw_std: float,
    ylog_mean: float,
    ylog_std: float,
) -> None:
    """Persist a weights-only checkpoint (safe for PyTorch >= 2.6).

    The checkpoint stores the model state-dict together with all
    normalisation constants and the current training tensors so that
    optimisation can be resumed from exactly where it left off.
    """
    payload = {
        "model_state_dict": model.state_dict(),
        "X_bo":       X_bo.detach().cpu(),
        "Y_bo":       Y_bo.detach().cpu(),
        "y_raw_mean": float(y_raw_mean),
        "y_raw_std":  float(y_raw_std),
        "ylog_mean":  float(ylog_mean),
        "ylog_std":   float(ylog_std),
        "input_dim":  int(X_bo.shape[1]),
    }
    torch.save(payload, str(path))


def load_checkpoint(path: Path):
    """Load a weights-only checkpoint and reconstruct the GP model.

    Returns
    -------
    model, X_bo, Y_bo, y_raw_mean, y_raw_std, ylog_mean, ylog_std
    """
    data = torch.load(str(path), map_location="cpu", weights_only=True)

    X_bo = data["X_bo"].to(dtype=DTYPE, device=DEVICE)
    Y_bo = data["Y_bo"].to(dtype=DTYPE, device=DEVICE)

    # Reconstruct the GP architecture and load trained weights
    model = build_gp(X_bo, Y_bo)
    model.load_state_dict(data["model_state_dict"])
    model.eval()

    return (
        model,
        X_bo,
        Y_bo,
        float(data["y_raw_mean"]),
        float(data["y_raw_std"]),
        float(data["ylog_mean"]),
        float(data["ylog_std"]),
    )


# =====================================================================
#  Excel I/O helpers
# =====================================================================

def read_training_excel(path: Path, n_max: int = 200):
    """Read the initial training data from a headerless Excel file.

    Parameters
    ----------
    path  : Path to Excel file (columns: Mx, My, Rx, Ry, Rz, Stress).
    n_max : Maximum number of rows to use (``None`` = all rows).

    Returns
    -------
    X : (N, 5) array of load parameters.
    y : (N,)   array of stress values.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing training file: {path}")

    df = pd.read_excel(path, header=None, engine="openpyxl")

    # Coerce to numeric and drop rows that contain any NaN
    X_df = df.iloc[:, :5].apply(pd.to_numeric, errors="coerce")
    y_df = pd.to_numeric(df.iloc[:, 5], errors="coerce")

    valid = X_df.notna().all(axis=1) & y_df.notna()
    X_df = X_df.loc[valid].reset_index(drop=True)
    y_df = y_df.loc[valid].reset_index(drop=True)

    if len(X_df) == 0:
        raise RuntimeError("No valid numeric rows in training file.")

    # Optionally truncate to the first n_max rows
    if n_max is not None and len(X_df) > n_max:
        X_df = X_df.iloc[:n_max].reset_index(drop=True)
        y_df = y_df.iloc[:n_max].reset_index(drop=True)

    return X_df.values.astype(float), y_df.values.astype(float)


def robust_read_excel(path: Path) -> pd.DataFrame:
    """Read an Excel file with retries to handle transient file locks.

    Tries both with and without a header row, returning the first
    successful read that produces at least one column.
    """
    last_exc = None
    for _ in range(READ_RETRIES):
        try:
            for header in (0, None):
                try:
                    df = pd.read_excel(path, header=header, engine="openpyxl")
                    if df.shape[1] > 0:
                        return df
                except Exception as exc:
                    last_exc = exc
            time.sleep(READ_RETRY_DELAY)
        except Exception as exc:
            last_exc = exc
            time.sleep(READ_RETRY_DELAY)
    raise RuntimeError(f"Failed reading '{path.name}'. Last error: {last_exc}")


def read_ansys_stress(path: Path, expected_rows: int) -> np.ndarray:
    """Extract a 1-D stress array from the ANSYS output Excel file.

    Handles both single-column and multi-column formats.  For multi-column
    files the 6th column (index 5) is assumed to contain stress; otherwise
    the last column is used.
    """
    df = robust_read_excel(path)

    # Single-column file → every value is a stress
    if df.shape[1] == 1:
        stresses = pd.to_numeric(df.iloc[:, 0], errors="coerce").values
        return stresses[~np.isnan(stresses)][:expected_rows]

    # Multi-column: prefer column index 5 (standard layout), else last column
    stress_col = 5 if df.shape[1] >= 6 else (df.shape[1] - 1)
    stresses = pd.to_numeric(df.iloc[:, stress_col], errors="coerce").values
    return stresses[~np.isnan(stresses)][:expected_rows]


def try_delete(path: Path) -> None:
    """Silently delete a file if it exists (best-effort)."""
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass


# =====================================================================
#  ANSYS file-polling
# =====================================================================

def _file_size(path: Path) -> int:
    """Return file size in bytes, or -1 on any error."""
    try:
        return path.stat().st_size
    except OSError:
        return -1


def wait_for_output_file(path: Path, batch_start_time: float) -> None:
    """Block until *path* exists, was modified after *batch_start_time*,
    and its size has been stable for ``FILE_STABLE_SECONDS``.

    This ensures the file is fully written by ANSYS before we attempt to
    read it.  A final parse check via ``robust_read_excel`` guards against
    partially-flushed files.

    Raises ``TimeoutError`` after ``ANSYS_WAIT_TIMEOUT`` seconds.
    """
    t0 = time.time()
    last_size = -1
    stable_since = None

    while True:
        # Timeout guard
        if time.time() - t0 > ANSYS_WAIT_TIMEOUT:
            raise TimeoutError(f"Timeout waiting for '{path.name}'")

        if path.exists():
            # Only accept files written *after* the batch was dispatched
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0

            if mtime >= batch_start_time:
                current_size = _file_size(path)

                if current_size == last_size and current_size > 0:
                    # Size unchanged → start / continue the stability timer
                    if stable_since is None:
                        stable_since = time.time()

                    if time.time() - stable_since >= FILE_STABLE_SECONDS:
                        # Sanity check: can we actually parse the file?
                        try:
                            robust_read_excel(path)
                            return  # File is stable and readable
                        except Exception:
                            stable_since = None  # Reset and keep waiting
                else:
                    # Size changed → file still being written
                    last_size = current_size
                    stable_since = None

        time.sleep(ANSYS_POLL_INTERVAL)


# =====================================================================
#  Gaussian Process model
# =====================================================================

def build_gp(X: torch.Tensor, Y: torch.Tensor) -> SingleTaskGP:
    """Construct an ARD Matern-5/2 GP with informative priors.

    Prior choices
    -------------
    * Lengthscale  – Gamma(3, 6):   mode ~ 0.33, encourages moderate smoothness.
    * Output-scale – Gamma(2, 0.15): mode ~ 6.7, weakly informative.
    * Noise        – Gamma(1.1, 0.05): keeps a sensible noise floor.

    All three have hard lower-bound constraints to prevent degenerate
    solutions during MLL optimisation.
    """
    base_kernel = MaternKernel(
        nu=2.5,
        ard_num_dims=5,
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
        train_X=X,
        train_Y=Y,
        covar_module=covariance,
        mean_module=ConstantMean(),
        likelihood=likelihood,
    ).to(DEVICE)


def fit_gp(X: torch.Tensor, Y: torch.Tensor) -> SingleTaskGP:
    """Build and train a GP model by maximising the log marginal likelihood."""
    model = build_gp(X, Y)
    model.train()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    model.eval()
    return model


def predict_stress(
    model: SingleTaskGP,
    X_unit: np.ndarray,
    ylog_mean: float,
    ylog_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict stress (MPa) from unit-scaled inputs via the log-space GP.

    Pipeline:
      1. Forward pass through the GP → standardised log-space posterior.
      2. Un-standardise to recover log-space mean and variance.
      3. Convert to MPa via lognormal moment formulas.

    Returns
    -------
    mean_mpa : Predicted mean stress in MPa.
    std_mpa  : Predicted standard deviation in MPa.
    """
    X_t = torch.tensor(X_unit, dtype=DTYPE, device=DEVICE)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model.posterior(X_t)
        mu_scaled  = posterior.mean.detach().cpu().numpy().ravel()
        var_scaled = posterior.variance.detach().cpu().numpy().ravel()

    # Un-standardise from z-scored log-space to raw log-space
    mu_log  = zunscale_y(mu_scaled, ylog_mean, ylog_std)
    std_log = np.sqrt(np.maximum(var_scaled, 0.0)) * ylog_std

    # Lognormal moments → physical MPa
    return log_to_stress(mu_log, std_log)


def report_hyperparameters(model: SingleTaskGP) -> None:
    """Print the trained GP hyperparameters to the console."""
    noise_var    = float(model.likelihood.noise.item())
    output_scale = float(model.covar_module.outputscale.item())
    mean_const   = float(model.mean_module.constant.item())
    ls = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()

    print(f"  Noise variance  : {noise_var}")
    print(f"  Output scale    : {output_scale}")
    print(f"  Mean constant   : {mean_const}")
    print(f"  Lengthscales    : {ls.tolist()}")
    print(f"  Lengthscale mean: {float(ls.mean())}")
    print(f"  Lengthscale std : {float(ls.std())}")


# =====================================================================
#  Regression metrics
# =====================================================================

def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float, float, float, float, np.ndarray, np.ndarray]:
    """Compute standard regression error metrics.

    Returns
    -------
    mae       : Mean absolute error (MPa).
    rmse      : Root mean squared error (MPa).
    mean_rel  : Mean relative error (ratio, not %).
    max_rel   : Maximum relative error (ratio).
    r2        : Coefficient of determination (R²).
    abs_err   : Per-point absolute errors.
    rel_err   : Per-point relative errors.
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    abs_err = np.abs(y_pred - y_true)
    mae  = float(abs_err.mean())
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    # Relative error with a floor to avoid blow-up at near-zero stress
    denom   = np.maximum(np.abs(y_true), REL_DENOM_FLOOR)
    rel_err = abs_err / denom
    mean_rel = float(rel_err.mean())
    max_rel  = float(rel_err.max())

    # Coefficient of determination (R²)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return mae, rmse, mean_rel, max_rel, r2, abs_err, rel_err


# =====================================================================
#  Main optimisation loop
# =====================================================================

def main() -> None:
    # ── Startup diagnostics ──────────────────────────────────────────
    print("RUNNING FILE:", Path(__file__).resolve())
    print("WORKING DIR :", Path.cwd())
    print("SCRIPT DIR  :", SCRIPT_DIR)
    print("DEVICE      :", DEVICE)
    print()

    # ── 1. Initialise GP (load checkpoint or train from scratch) ─────
    if CHECKPOINT.exists():
        (model, X_bo, Y_bo,
         y_raw_mean, y_raw_std,
         ylog_mean, ylog_std) = load_checkpoint(CHECKPOINT)
        print("Loaded checkpoint:", CHECKPOINT.name)

    else:
        # Read the first 200 training samples from the initial FEA results
        X_phys, y_phys = read_training_excel(EXCEL_TRAIN, n_max=200)

        # Raw-stress statistics (informational only)
        y_raw_mean = float(y_phys.mean())
        y_raw_std  = float(y_phys.std()) or 1.0

        # Log-transform stress, then compute z-score parameters
        y_log     = stress_to_log(y_phys)
        ylog_mean = float(y_log.mean())
        ylog_std  = float(y_log.std()) or 1.0

        # Scale inputs to [0, 1] and standardise log-stress
        X_unit    = unit_scale_X(X_phys)
        y_scaled  = zscale_y(y_log, ylog_mean, ylog_std).reshape(-1, 1)

        X_bo = torch.tensor(X_unit,   dtype=DTYPE, device=DEVICE)
        Y_bo = torch.tensor(y_scaled, dtype=DTYPE, device=DEVICE)

        model = fit_gp(X_bo, Y_bo)
        save_checkpoint(CHECKPOINT, model, X_bo, Y_bo,
                        y_raw_mean, y_raw_std, ylog_mean, ylog_std)
        print("Trained initial GP.  Saved checkpoint:", CHECKPOINT.name)

    print("Raw stress  –  mean:", y_raw_mean, " std:", y_raw_std)
    print("Log-stress  –  mean:", ylog_mean,  " std:", ylog_std)

    # Unit-cube bounds tensor used by all BoTorch acquisition optimisers
    bounds = torch.tensor(
        np.vstack([np.zeros(5), np.ones(5)]),
        dtype=DTYPE, device=DEVICE,
    )

    # Best observed stress so far (approximated from the GP dataset).
    # Convert best standardised log value back to MPa using a zero-variance
    # approximation: stress ≈ exp(log_value).
    best_log_scaled = float(Y_bo.max().item())
    best_log = zunscale_y(np.array([best_log_scaled]), ylog_mean, ylog_std)[0]
    best_observed_stress = float(np.exp(best_log))
    print("Initial best observed stress ~=", best_observed_stress, "MPa")
    print()

    # ── 2. Iterative BO loop ─────────────────────────────────────────
    all_rows: list[list[float]] = []   # Accumulates per-candidate records
    batch_history: list[dict]   = []   # Accumulates per-batch summaries

    best_predicted_so_far = -np.inf    # Tracks best GP-predicted stress
    no_improve_count      = 0          # Consecutive batches without improvement

    for batch_idx in range(1, MAX_BATCHES + 1):
        print("=" * 60)
        print(f"  BATCH {batch_idx} / {MAX_BATCHES}")
        print("=" * 60)

        # 2a. Refit GP on the current (growing) dataset
        model = fit_gp(X_bo, Y_bo)

        # 2b. Find the current best *predicted* stress by optimising the
        #     posterior mean.  Used only for the stopping criterion.
        pm_acq = PosteriorMean(model=model)
        x_star_t, _ = optimize_acqf(
            acq_function=pm_acq,
            bounds=bounds,
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options={"maxiter": ACQ_MAXITER},
        )
        best_pred_mpa, _ = predict_stress(
            model,
            x_star_t.detach().cpu().numpy().reshape(1, 5),
            ylog_mean, ylog_std,
        )
        best_predicted = float(best_pred_mpa[0])

        # 2c. Check relative improvement for early stopping
        if best_predicted_so_far == -np.inf:
            rel_improve = 1.0               # First batch → always continue
        else:
            rel_improve = (
                (best_predicted - best_predicted_so_far)
                / max(abs(best_predicted_so_far), 1e-12)
            )

        if rel_improve < IMPROVEMENT_THRESHOLD:
            no_improve_count += 1
        else:
            no_improve_count = 0
            best_predicted_so_far = best_predicted

        print(f"  Best predicted stress : {best_predicted:.4f} MPa")
        print(f"  Relative improvement  : {rel_improve:.4%}")
        print(f"  Stagnation counter    : {no_improve_count} / {PATIENCE}")

        # 2d. Propose a batch of candidates via q-Expected Improvement.
        #     ``sequential=True`` generates candidates one at a time,
        #     conditioning each on those already selected, which improves
        #     spatial diversity within the batch.
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1024]))
        qei = qExpectedImprovement(
            model=model,
            best_f=float(Y_bo.max().item()),
            sampler=sampler,
        )
        X_new_unit_t, _ = optimize_acqf(
            acq_function=qei,
            bounds=bounds,
            q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options={"maxiter": ACQ_MAXITER},
            sequential=True,
        )

        X_new_unit = X_new_unit_t.detach().cpu().numpy().reshape(BATCH_SIZE, 5)
        X_new_phys = unit_unscale_X(X_new_unit)   # Convert back to physical units

        # 2e. Dispatch candidates to ANSYS
        try_delete(ANSYS_OUTPUT)                   # Remove stale output
        pd.DataFrame(X_new_phys).to_excel(
            ANSYS_INPUT, index=False, header=False, engine="openpyxl",
        )
        print(f"  Wrote candidates  -> {ANSYS_INPUT.name}")
        print(f"  Waiting for ANSYS -> {ANSYS_OUTPUT.name}")

        batch_start = time.time()
        if ANSYS_CMD:
            subprocess.Popen(ANSYS_CMD, shell=isinstance(ANSYS_CMD, str))

        # Block until ANSYS output is stable and readable
        wait_for_output_file(ANSYS_OUTPUT, batch_start_time=batch_start)
        print(f"  Detected stable output: {ANSYS_OUTPUT.name}")

        # 2f. Read FEA stresses returned by ANSYS
        y_fea = read_ansys_stress(ANSYS_OUTPUT, expected_rows=BATCH_SIZE)
        n = min(len(y_fea), BATCH_SIZE)
        if n == 0:
            raise RuntimeError("No stresses read from test_results.xlsx")

        # 2g. GP predictions for the same batch points (diagnostic only)
        y_pred_mpa, y_pred_std = predict_stress(
            model, X_new_unit[:n], ylog_mean, ylog_std,
        )

        # 2h. Batch-level regression metrics
        mae_b, rmse_b, mean_rel_b, max_rel_b, r2_b, abs_err_b, rel_err_b = (
            regression_metrics(y_fea[:n], y_pred_mpa[:n])
        )
        print(f"  Batch MAE             : {mae_b:.4f} MPa")
        print(f"  Batch RMSE            : {rmse_b:.4f} MPa")
        print(f"  Batch mean rel. error : {mean_rel_b:.4%}")
        print(f"  Batch max  rel. error : {max_rel_b:.4%}")
        print(f"  Batch R^2             : {r2_b:.4f}")

        # 2i. Log individual candidate results
        for i in range(n):
            all_rows.append([
                *X_new_phys[i].tolist(),            # Mx, My, Rx, Ry, Rz
                float(y_fea[i]),                    # FEA stress
                float(y_pred_mpa[i]),               # GP prediction
                float(y_pred_mpa[i] - y_fea[i]),    # Residual
                float(rel_err_b[i]),                # Point-wise relative error
                int(batch_idx),                     # Batch number
            ])

        # 2j. Augment training set with new ANSYS observations
        y_new_log    = stress_to_log(y_fea[:n])
        y_new_scaled = zscale_y(y_new_log, ylog_mean, ylog_std).reshape(-1, 1)

        X_bo = torch.cat([X_bo, torch.tensor(X_new_unit[:n], dtype=DTYPE, device=DEVICE)])
        Y_bo = torch.cat([Y_bo, torch.tensor(y_new_scaled,   dtype=DTYPE, device=DEVICE)])

        batch_best_obs = float(np.max(y_fea[:n]))
        print(f"  Best FEA stress in batch : {batch_best_obs:.4f} MPa")
        print(f"  Total training samples   : {X_bo.shape[0]}")

        # 2k. Persist intermediate results after every batch
        pd.DataFrame(all_rows, columns=[
            "Mx", "My", "Rx", "Ry", "Rz",
            "FEA_Stress", "GP_Pred_Stress",
            "Residual", "Rel_Error_pointwise", "Batch",
        ]).to_excel(
            SCRIPT_DIR / "all_batch_results.xlsx",
            index=False, engine="openpyxl",
        )

        batch_history.append({
            "Batch":                         batch_idx,
            "n_total":                       int(X_bo.shape[0]),
            "best_predicted_stress_MPa":     best_predicted,
            "rel_predicted_improvement":     rel_improve,
            "no_improve_count":              no_improve_count,
            "batch_best_observed_stress_MPa": batch_best_obs,
            "batch_MAE_MPa":                 mae_b,
            "batch_RMSE":                    rmse_b,
            "batch_mean_rel_error":          mean_rel_b,
            "batch_max_rel_error":           max_rel_b,
            "batch_R2":                      r2_b,
        })
        pd.DataFrame(batch_history).to_excel(
            SCRIPT_DIR / "batch_history.xlsx",
            index=False, engine="openpyxl",
        )

        # 2l. Checkpoint after every batch (refit first for clean state)
        model = fit_gp(X_bo, Y_bo)
        save_checkpoint(CHECKPOINT, model, X_bo, Y_bo,
                        y_raw_mean, y_raw_std, ylog_mean, ylog_std)
        print(f"  Saved checkpoint: {CHECKPOINT.name}")

        # 2m. Early-stopping check
        if no_improve_count >= PATIENCE:
            print()
            print(f"STOPPING: best predicted stress improved < "
                  f"{IMPROVEMENT_THRESHOLD:.0%} for {PATIENCE} consecutive "
                  f"batches.")
            break

    # ── 3. Final GP fit and hyperparameter report ────────────────────
    model = fit_gp(X_bo, Y_bo)

    print()
    print("=" * 60)
    print("  FINAL GP HYPERPARAMETERS")
    print("=" * 60)
    report_hyperparameters(model)

    # ── 4. Evaluate on hold-out test set ─────────────────────────────
    if TEST_FILE.exists():
        print()
        print("=" * 60)
        print("  TEST SET EVALUATION  –  new_loads_results.xlsx")
        print("=" * 60)

        df_test = pd.read_excel(TEST_FILE, header=None, engine="openpyxl")
        X_test_phys = (
            df_test.iloc[:, :5]
            .apply(pd.to_numeric, errors="coerce")
            .values.astype(float)
        )
        y_test = pd.to_numeric(df_test.iloc[:, 5], errors="coerce").values.astype(float)

        # Drop rows containing any NaN
        valid = ~(np.isnan(X_test_phys).any(axis=1) | np.isnan(y_test))
        X_test_phys, y_test = X_test_phys[valid], y_test[valid]

        if len(X_test_phys) == 0:
            print("  No valid rows in new_loads_results.xlsx")
        else:
            y_pred, y_std = predict_stress(
                model, unit_scale_X(X_test_phys), ylog_mean, ylog_std,
            )
            mae, rmse, mean_rel, max_rel, r2, _, _ = regression_metrics(
                y_test, y_pred,
            )

            print(f"  MAE             : {mae:.4f} MPa")
            print(f"  RMSE            : {rmse:.4f} MPa")
            print(f"  Mean rel. error : {mean_rel:.4%}")
            print(f"  Max  rel. error : {max_rel:.4%}")
            print(f"  R^2             : {r2:.4f}")

            # Save predictions alongside actual values (headerless, 7 columns)
            out = np.column_stack([X_test_phys, y_test, y_pred])
            out_path = SCRIPT_DIR / "new_loads_results_with_actual_and_predicted.xlsx"
            pd.DataFrame(out).to_excel(
                out_path, index=False, header=False, engine="openpyxl",
            )
            print(f"  Saved: {out_path.name}")
    else:
        print()
        print("Test file not found:", TEST_FILE.name)

    print()
    print("DONE.")


# =====================================================================
#  Entry point
# =====================================================================

if __name__ == "__main__":
    main()
