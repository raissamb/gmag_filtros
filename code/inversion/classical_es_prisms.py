# Import packages
import sys
#import verde as vd
#import harmonica as hm
import numpy as np
from pathlib import Path
#import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp

# Define paths
CURRENT_FILE_PATH = Path(__file__).resolve()
CODE_ROOT = CURRENT_FILE_PATH.parent.parent

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

PROJECT_HOME = CODE_ROOT.parent
DATA_FOLDER = PROJECT_HOME / "data"
OUTPUT_FOLDER = PROJECT_HOME / "output" 

# import custom modules
#import modules.constants as mc
#import modules.utils as ut
import modules.kernel as kn

"""
# ----------- LINEAR INVERSE SYSTEM
d = G p
f = G \tilde{p}

\tilde{p} = (G^T G + \mu R^T R)^{-1} G^T d

# ---------------------------------------------------------
"""

# ---------------------------------------------------------
# 1. Load Synthetic Data and Subsample for Profiling
# ---------------------------------------------------------

# Read the generated Wang 2015 dataset
df = pd.read_csv(DATA_FOLDER/"wang2015_mag.csv")

# For testing the classical method without memory overflow,
# select a spatial window or decimate the grid (e.g., 50x50 = 2500 points)
# Here we take a centered window of 2500 points:
x_unique = np.unique(df["X"].values)[::13][:50]
y_unique = np.unique(df["Y"].values)[::13][:50]

df_sub = df[df["X"].isin(x_unique) & df["Y"].isin(y_unique)].copy()

easting = df_sub["X"].values
northing = df_sub["Y"].values
upwards = df_sub["Z"].values
d_obs = df_sub["tfa"].values  # Input anomaly data

N = len(d_obs)
print(f"Number of equivalent sources / observations (N): {N}")

# Survey parameters
inc = 90.0
dec = 0.0
dx = float(np.diff(np.sort(np.unique(easting)))[0])
dy = float(np.diff(np.sort(np.unique(northing)))[0])


# ---------------------------------------------------------
# 2. Define the Equivalent Layer of Prisms
# ---------------------------------------------------------
# Place one prism directly below each observation point
# Depth to top (z_top) and bottom (z_bottom) of equivalent layer (negative downwards in Harmonica)
z_top = -500.0       # 500 m depth
z_bottom = -1500.0   # 1000 m thickness
prisms_layer = kn.build_prism_layer(N, dx, dy, easting, northing, z_bottom, z_top)

# ---------------------------------------------------------
# 3. Assemble Jacobian Matrix G and Track Computational Cost
# ---------------------------------------------------------
G, G_assembly_time, G_peak_ram_mb = kn.build_jacobian_with_profiling(N, easting, 
                                                                 northing, upwards, 
                                                                 prisms_layer, inc, dec)


# ---------------------------------------------------------
# 4. Construct the First-Order Tikhonov Matrix (RTR)
# ---------------------------------------------------------
RTR, rtr_time, rtr_mem = kn.build_rtr_with_profiling(df_sub, N, kn)

# # 5. Solve the linear system
# mu = 1e-3 # Placeholder regularization parameter
# p_tilde, solve_time, solve_mem = kn.solve_system_cholesky(G, RTR, d_obs, mu)

# # 6. Evaluate Data Fit
# predicted_data = G @ p_tilde
# residuals = d_obs - predicted_data
# rmse = np.sqrt(np.mean(residuals**2))
# print(f"Inversion RMSE: {rmse:.4f} nT")


# ---------------------------------------------------------
# 5. Run L-Curve Sweep and Solve
# ---------------------------------------------------------

reg_min = 1e-6
reg_max = 1e2
steps = 20

df_metrics, best_solution, solver_sweep_time, solver_peak_ram_mb = kn.run_tikhonov_l_curve_scan_ces(
    G, d_obs, RTR, reg_min, reg_max, steps
)

mu_opt = best_solution["reg_value"]
p_tilde = best_solution["p_est"]
rmse_opt = best_solution["rmse"]

print(f"Optimal Regularization (mu): {mu_opt:.2e}")
print(f"Optimized Inversion RMSE: {rmse_opt:.4f} nT")



















