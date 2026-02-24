# Worst-Case-Stress-Identification-on-a-Hyper-Redundant-Robot-using-Bayesian-Optimisation
Bayesian Optimisation with ANSYS-in-the-Loop: a Gaussian Process surrogate model iteratively proposes load configurations, dispatches them to ANSYS Mechanical for FEA evaluation, and refines predictions to efficiently explore structural stress responses.

# Bayesian Optimisation with ANSYS-in-the-Loop

> **Gaussian-Process-driven iterative optimisation for structural FEA stress analysis.**

This repository implements a closed-loop Bayesian Optimisation (BO) workflow that couples a Gaussian Process (GP) surrogate model with ANSYS Mechanical to efficiently explore loading configurations and predict stress responses. At every iteration the Python script proposes a batch of candidate load vectors, ANSYS evaluates them via FEA, and the results are fed back to refine the surrogate—repeating until convergence.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Prerequisites](#prerequisites)
4. [Data Format](#data-format)
5. [Step 1 — Preparing the Initial Training Data](#step-1--preparing-the-initial-training-data)
6. [Step 2 — Setting Up the ANSYS Automation Script](#step-2--setting-up-the-ansys-automation-script)
7. [Step 3 — Running the Optimisation](#step-3--running-the-optimisation)
8. [Step 4 — How the Batch Loop Works](#step-4--how-the-batch-loop-works)
9. [Step 5 — Plotting Results](#step-5--plotting-results)
10. [Configuration Reference](#configuration-reference)
11. [Methodology](#methodology)
12. [Output Files](#output-files)
13. [Troubleshooting](#troubleshooting)

---

## Overview

The goal is to find loading conditions that produce **maximum stress** in a structural component, using as few expensive ANSYS simulations as possible. The workflow is:

```
┌─────────────────────────────────────────────────────────────────┐
│  Initial FEA dataset (results_with_loads.xlsx, 200 samples)     │
│                            │                                    │
│                    Train GP surrogate                           │
│                            │                                    │
│              ┌─────────────┴──────────────┐                     │
│              │   Bayesian Optimisation     │                     │
│              │         Loop               │                     │
│              │                            │                     │
│              │  1. Propose 10 candidates  │                     │
│              │     (q-Expected Improvement)│                     │
│              │             │              │                     │
│              │  2. Write ansys_test_data   │                     │
│              │             │              │                     │
│              │  3. ANSYS runs FEA ──────► │  test_results.xlsx  │
│              │             │              │                     │
│              │  4. Read FEA stresses      │                     │
│              │             │              │                     │
│              │  5. Augment training data   │                     │
│              │             │              │                     │
│              │  6. Re-train GP            │                     │
│              │             │              │                     │
│              │  7. Check convergence      │                     │
│              │      (< 1% improvement     │                     │
│              │       for 4 batches ──►STOP)│                    │
│              └─────────────┬──────────────┘                     │
│                            │                                    │
│               Final evaluation on test set                      │
│      (new_loads_results.xlsx ──► predictions + metrics)         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
├── BO_ANSYS_WAIT.py              # Main optimisation script
├── plot_results.py               # Post-optimisation plotting (uses saved model)
├── results_with_loads.xlsx       # Initial training data (200 FEA samples)
├── new_loads_results.xlsx        # Hold-out test set (50 samples)
├── Ansys_Simulation_Files.zip    # ANSYS project, CAD model & automation script (see below)
└── README.md                     # This file
```

### ANSYS Simulation Files (`Ansys_Simulation_Files.zip`)

This ZIP archive contains all the files needed to run the ANSYS side of the
optimisation loop. **After downloading, unzip it and place all its contents in
the same directory as the Python scripts and Excel files above.** Your folder
should look like this after extraction:

```
├── BO_ANSYS_WAIT.py
├── plot_results.py
├── results_with_loads.xlsx
├── new_loads_results.xlsx
├── Ansys_Automation_Script.txt       # IronPython script for the ANSYS Mechanical scripting console
├── Ansys_Sim.wbpj                    # ANSYS Workbench project file
├── Ansys_Sim_files/                  # ANSYS project support folder (meshes, solver settings, etc.)
├── .Ansys_Sim_files.backup/          # ANSYS automatic backup folder
└── link.prt.3                        # CAD geometry (Creo/ProE part file)
```

> **Important:** All files must reside in the **same flat directory** — do not
> leave them inside a nested sub-folder after unzipping.

**Files generated after running `BO_ANSYS_WAIT.py`:**

```
├── gp_checkpoint_safe.pt                            # Trained GP model checkpoint
├── ansys_test_data.xlsx                             # Candidates sent to ANSYS (transient)
├── test_results.xlsx                                # FEA stresses from ANSYS (transient)
├── all_batch_results.xlsx                           # Per-candidate results from all batches
├── batch_history.xlsx                               # Per-batch summary metrics
└── new_loads_results_with_actual_and_predicted.xlsx  # Final test-set predictions
```

**Files generated after running `plot_results.py`:**

```
└── plots/
    ├── 01_predicted_vs_actual.png       # Parity plot (predicted vs FEA)
    ├── 02_residuals.png                 # Residual analysis
    ├── 03_convergence.png               # BO convergence curves
    ├── 04_batch_metrics.png             # Per-batch R², relative error, improvement
    ├── 05_ard_lengthscales.png          # Feature importance (ARD lengthscales)
    ├── 06_gp_1d_slices.png             # 1-D GP response surfaces
    ├── 07_prediction_uncertainty.png    # Predictions with ± 2σ error bars
    ├── 08_training_data_overview.png    # Training data distributions
    └── 09_batch_candidates_scatter.png  # BO candidates coloured by batch
```

---

## Prerequisites

**Software:**
- Python 3.9+
- ANSYS Workbench (tested with 2022 R2)

**Python packages:**

```bash
pip install numpy pandas torch gpytorch botorch openpyxl matplotlib
```

| Package    | Purpose                                    |
|------------|--------------------------------------------|
| `numpy`    | Numerical computing                        |
| `pandas`   | Excel I/O                                  |
| `torch`    | Tensor operations, model serialisation     |
| `gpytorch` | Gaussian Process implementation            |
| `botorch`  | Bayesian optimisation (acquisition functions, model fitting) |
| `openpyxl` | Excel file reading/writing engine          |
| `matplotlib` | Plotting (only needed for `plot_results.py`) |

---

## Data Format

All Excel files are **headerless** (no column names in the first row). The expected column layout is:

| Column | 1   | 2   | 3   | 4   | 5   | 6      |
|--------|-----|-----|-----|-----|-----|--------|
| Name   | Mx  | My  | Rx  | Ry  | Rz  | Stress |
| Unit   | N·m | N·m | N   | N   | N   | MPa    |

- **Mx, My** — Bending moments
- **Rx, Ry, Rz** — Reaction forces
- **Stress** — Von Mises stress (or equivalent) from FEA

---

## Step 1 — Preparing the Initial Training Data

### `results_with_loads.xlsx`

This file contains the **initial 200 FEA simulation results** that bootstrap the GP surrogate model. Each row represents one ANSYS Static Structural simulation with a unique combination of 5 load parameters and the resulting stress.

**Where does it come from?**

1. You define a parameterised ANSYS Workbench model with 5 input parameters (Mx, My, Rx, Ry, Rz) representing moments and reaction forces applied to your structural component.
2. You run a Design of Experiments (DoE) or a set of simulations (e.g. Latin Hypercube Sampling) spanning your expected operating range to generate a diverse set of 200+ load combinations.
3. For each combination, ANSYS solves the static structural problem and reports the maximum equivalent (Von Mises) stress.
4. You export the 5 load values and the stress into a 6-column headerless Excel file.

The first 200 rows of this file are used to train the initial GP model. The script reads this file **only once** — after the first run, the trained model is stored in the checkpoint file and subsequent runs load from there.

### `new_loads_results.xlsx`

This is a **hold-out test set** (e.g. 50 samples) with the same 6-column format. It contains load–stress pairs that were **not** used during training, so the GP's predictions can be evaluated against ground truth. These are typically additional ANSYS simulations that you set aside before starting the optimisation.

---

## Step 2 — Setting Up the ANSYS Automation Script

The optimisation loop requires ANSYS to **automatically** read candidate load vectors, run FEA, and write the results back. This is done via an IronPython script that runs inside the ANSYS Mechanical scripting console.

### How to set it up

1. Open your ANSYS Workbench project (`sim_for_BO.wbpj` or your own project).
2. Make sure your Static Structural model is set up with the correct geometry, mesh, boundary conditions, and material properties.
3. Open the **Mechanical Scripting Console** (inside ANSYS Mechanical, go to `Automation` → `Scripting`).
4. Open the file `Ansys_Automation_Script.txt` and **copy-paste its contents** into the scripting console.
5. The script will:
   - **Watch** for the file `ansys_test_data.xlsx` (written by `BO_ANSYS_WAIT.py`)
   - **Read** the 10 candidate load vectors from it
   - **Apply** each load combination to the FEA model and solve
   - **Extract** the maximum stress for each load case
   - **Write** the results to `test_results.xlsx`

> **Important:** The ANSYS script and the Python script must point to the **same working directory** so they can exchange files. By default, both use the directory containing `BO_ANSYS_WAIT.py`.

### File exchange protocol

The two scripts communicate exclusively through Excel files on disk:

```
Python (BO_ANSYS_WAIT.py)                    ANSYS (Automation Script)
        │                                            │
        ├── writes ansys_test_data.xlsx ──────────►  │
        │   (10 rows × 5 columns: Mx My Rx Ry Rz)   │
        │                                            │
        │   ◄────────── reads ansys_test_data.xlsx ──┤
        │                                            ├── runs 10 FEA solves
        │                                            ├── extracts max stress
        │   ◄────────── writes test_results.xlsx ────┤
        │   (10 rows, stress in column 6)            │
        │                                            │
        ├── reads test_results.xlsx                  │
        ├── augments GP training data                │
        ├── proposes next batch                      │
        └── repeats...                               │
```

The Python script **polls** the file system, waiting for `test_results.xlsx` to appear, stabilise in size (no more writes), and become readable. This means you do not need to manually synchronise the two — simply start both and the handshake is automatic.

---

## Step 3 — Running the Optimisation

1. Place `BO_ANSYS_WAIT.py`, `results_with_loads.xlsx`, and `new_loads_results.xlsx` in the same directory.

2. Set up the ANSYS automation script as described in [Step 2](#step-2--setting-up-the-ansys-automation-script).

3. Run the Python script:

```bash
python BO_ANSYS_WAIT.py
```

4. The script will:
   - **First run:** Train the initial GP on 200 samples from `results_with_loads.xlsx` and save a checkpoint.
   - **Subsequent runs:** Load the checkpoint and resume optimisation.
   - Enter the BO loop (see [Step 4](#step-4--how-the-batch-loop-works)).
   - After the loop ends, evaluate the final model on `new_loads_results.xlsx`.

> **Tip:** If you need to start fresh, simply delete `gp_checkpoint_safe.pt` and re-run.

---

## Step 4 — How the Batch Loop Works

Each batch iteration performs the following steps:

### 4.1 — Refit the GP

The GP model is retrained on all accumulated data (initial 200 + all previously evaluated batches). The model uses:
- **ARD Matérn-5/2 kernel** — Automatically learns which input features matter most.
- **Log-stress transform** — Stress values are log-transformed to better fit lognormal distributions.
- **Unit-cube scaling** — Inputs are mapped to [0, 1] using known physical bounds.

### 4.2 — Check for convergence

The script optimises the GP's posterior mean to find the current best-predicted stress. If this value has improved by **less than 1 %** for **4 consecutive batches**, the optimisation stops early.

### 4.3 — Propose candidates (q-Expected Improvement)

The script uses the **q-Expected Improvement (qEI)** acquisition function to propose a batch of 10 candidates that balance:
- **Exploitation** — sampling where the GP predicts high stress.
- **Exploration** — sampling where the GP is uncertain.

The `sequential=True` flag generates candidates one at a time, conditioning each on those already selected. This prevents the batch from clustering in one region of the design space.

### 4.4 — Dispatch to ANSYS

The 10 candidate load vectors (in physical units) are written to `ansys_test_data.xlsx`. The script then deletes any stale `test_results.xlsx` from a previous iteration and begins polling the file system.

### 4.5 — ANSYS evaluates candidates

The ANSYS automation script (running in the Mechanical scripting console) detects the new `ansys_test_data.xlsx`, applies each load combination, solves, extracts the stress, and writes the results to `test_results.xlsx`.

### 4.6 — Ingest results

Once `test_results.xlsx` is detected and stable, the Python script reads the FEA stresses, computes batch-level prediction metrics (MAE, RMSE, R², relative errors), and appends the new observations to the training set.

### 4.7 — Save and repeat

Intermediate results (`all_batch_results.xlsx`, `batch_history.xlsx`) and the model checkpoint are saved after every batch. If the process is interrupted, it can be resumed from the checkpoint.

---

## Step 5 — Plotting Results

After the optimisation finishes, run:

```bash
python plot_results.py
```

This script **loads the saved GP model** from `gp_checkpoint_safe.pt` — it does **not** re-run any optimisation. It produces 9 diagnostic plots in a `plots/` sub-directory:

| # | Plot | Description |
|---|------|-------------|
| 1 | Predicted vs Actual | Parity (45° line) scatter — how close are GP predictions to FEA truth? |
| 2 | Residuals | Residual scatter + histogram — are errors random and unbiased? |
| 3 | Convergence | Best predicted stress and prediction errors over BO iterations. |
| 4 | Batch Metrics | R², relative error, and improvement rate per batch. |
| 5 | ARD Lengthscales | Bar chart of learned lengthscales — shorter = more influential feature. |
| 6 | 1-D GP Slices | Response surface through each input (others at midpoint) with 95 % CI. |
| 7 | Prediction Uncertainty | Sorted predictions with ± 2σ error bars vs actual values. |
| 8 | Training Data Overview | Histograms of all input features and stress in the training set. |
| 9 | Candidate Scatter | Scatter matrix of all BO-proposed candidates, coloured by batch number. |

The script also prints a model summary and test-set metrics to the console.

---

## Configuration Reference

All tuneable parameters are defined as constants at the top of `BO_ANSYS_WAIT.py`:

### Bayesian Optimisation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | 10 | Number of candidates proposed per iteration |
| `MAX_BATCHES` | 10 | Hard upper limit on BO iterations |
| `NUM_RESTARTS` | 10 | Multi-start restarts for acquisition optimisation |
| `RAW_SAMPLES` | 512 | Sobol quasi-random seed points per restart |
| `ACQ_MAXITER` | 250 | L-BFGS-B iterations within each restart |

### Early Stopping

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMPROVEMENT_THRESHOLD` | 0.01 | Minimum relative improvement to reset patience (1 %) |
| `PATIENCE` | 4 | Consecutive stagnant batches before stopping |

### ANSYS File Polling

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ANSYS_WAIT_TIMEOUT` | 3600 s | Maximum wait time for ANSYS output (1 hour) |
| `ANSYS_POLL_INTERVAL` | 3 s | Seconds between file-existence checks |
| `FILE_STABLE_SECONDS` | 3 s | File size must be constant for this long before reading |
| `READ_RETRIES` | 8 | Retry attempts when Excel file is locked |
| `READ_RETRY_DELAY` | 2 s | Seconds between read retries |

### Domain Bounds

| Parameter | Values |Description |
|-----------|--------|------------|
| `LOWER_BOUNDS` | [-37.35, -41.79, 658.7, -1295.0, -178.7] | Lower physical bounds for [Mx, My, Rx, Ry, Rz] |
| `UPPER_BOUNDS` | [35.14, 46.14, 1872.5, 1185.0, 178.8] | Upper physical bounds for [Mx, My, Rx, Ry, Rz] |

> **Adapting to your problem:** Update `LOWER_BOUNDS` and `UPPER_BOUNDS` to match the operating range of your specific structural model. If you have a different number of input parameters, also update the kernel's `ard_num_dims` in the `build_gp` function.

---

## Methodology

### Gaussian Process Surrogate

The GP models stress in **log-space** to respect the strictly-positive nature of stress and to better capture lognormal-like variability:

$$y_{\text{scaled}} = \frac{\log(\sigma_{\text{stress}}) - \mu_{\log}}{\sigma_{\log}}$$

The kernel is an **ARD Matérn-5/2** wrapped in a **ScaleKernel**, which learns a separate lengthscale per input dimension. Features with shorter lengthscales are more influential — the GP response varies more rapidly along those dimensions.

**Priors** (Gamma distributions) are placed on:
- Lengthscales — `Gamma(3.0, 6.0)` — encourages moderate smoothness
- Output scale — `Gamma(2.0, 0.15)` — weakly informative
- Noise variance — `Gamma(1.1, 0.05)` — prevents noise from collapsing to zero

### Acquisition Function

**q-Expected Improvement (qEI)** is used to propose batches. It extends classical EI to the batch setting, estimating the expected improvement over the current best observation across all candidates in the batch simultaneously.

The `sequential=True` mode generates candidates one at a time: after selecting the first candidate, it "fantasises" its outcome and conditions the acquisition function on it before selecting the second, and so on. This significantly improves spatial diversity compared to joint optimisation.

### Predictions in Physical Units

GP predictions are made in standardised log-space and converted back to physical stress (MPa) using the **lognormal moment formulas**:

$$E[Y] = \exp\!\left(\mu_{\log} + \frac{\sigma_{\log}^2}{2}\right)$$

$$\text{Var}[Y] = \left[\exp(\sigma_{\log}^2) - 1\right] \cdot \exp(2\mu_{\log} + \sigma_{\log}^2)$$

---

## Output Files

### `gp_checkpoint_safe.pt`

A PyTorch checkpoint containing:
- Model weights (state dict)
- Full training dataset (unit-cube scaled inputs + standardised log-stress targets)
- All normalisation constants (`ylog_mean`, `ylog_std`, `y_raw_mean`, `y_raw_std`)
- Domain bounds and feature names

This file is everything needed to reconstruct the trained GP without re-running the optimisation. It is used by `plot_results.py`.

### `all_batch_results.xlsx`

One row per candidate evaluated during BO. Columns:

| Mx | My | Rx | Ry | Rz | FEA_Stress | GP_Pred_Stress | Residual | Rel_Error_pointwise | Batch |
|----|----|----|----|----|------------|----------------|----------|---------------------|-------|

### `batch_history.xlsx`

One row per BO batch. Columns include: batch number, total training samples, best predicted stress, relative improvement, stagnation counter, batch-best observed stress, and batch-level error metrics (MAE, RMSE, R², mean/max relative error).

### `new_loads_results_with_actual_and_predicted.xlsx`

Final test-set evaluation. Seven headerless columns:

| Mx | My | Rx | Ry | Rz | Stress_actual | Stress_predicted |
|----|----|----|----|----|---------------|------------------|

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **"Missing training file"** | Ensure `results_with_loads.xlsx` is in the same directory as the script. |
| **"Timeout waiting for test_results.xlsx"** | ANSYS did not produce the output in time. Check that the ANSYS automation script is running and that file paths match. Increase `ANSYS_WAIT_TIMEOUT` if simulations are slow. |
| **"File locked" warnings** | ANSYS or Excel may still have the file open. The script retries automatically — this is usually harmless. |
| **"No stresses read"** | The ANSYS output file format may not match expectations. Ensure stress is in column 6 (index 5) or that the column header contains a keyword like "stress" or "vonmises". |
| **Want to restart from scratch** | Delete `gp_checkpoint_safe.pt` and re-run `BO_ANSYS_WAIT.py`. |
| **Plots fail with "checkpoint not found"** | Run `BO_ANSYS_WAIT.py` first to generate the checkpoint, then run `plot_results.py`. |
| **Different number of input parameters** | Update `LOWER_BOUNDS`, `UPPER_BOUNDS`, `FEATURE_NAMES`, and `ard_num_dims` in `build_gp`. |

---

## License

This project is provided as-is for academic and research purposes.

