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
BASE_FOLDER = PROJECT_HOME / "output" 

# import custom modules
#import modules.constants as mc
import modules.utils as ut
import modules.kernel as kn

"""
# ----------- LINEAR INVERSE SYSTEM
d = G p
f = G pe

pe = (G^T G + u R^T R)^{-1} G^T d

# ---------------------------------------------------------
"""

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
metadata = ut.find_read_json_file(DATA_FOLDER/"model1", "inversion_config.json")

# Survey parameters
data_type = metadata["data_type"]
filename = metadata["filename"]
easting_col = metadata["easting_col"]
northing_col = metadata["northing_col"]
upwards_col = metadata["upwards_col"]
field = metadata["field"]
z_top = metadata["z_top"]
z_bottom = metadata["z_bottom"]
output_path = metadata["output_folder"]

dirpath = BASE_FOLDER / output_path
dirpath.mkdir(exist_ok=True, parents=True)


df = pd.read_csv(DATA_FOLDER/filename)

# Enforce strict spatial sorting (Row-major: Y then X) BEFORE extracting arrays
df = df.sort_values(by=["Y", "X"]).reset_index(drop=True)
easting = df[easting_col].to_numpy()
northing = df[northing_col].to_numpy()
upwards = df[upwards_col].to_numpy()
d_obs = df[field].to_numpy()  # Input anomaly data


# ---------------------------------------------------------
# 2. Define the Equivalent Layer of Prisms
# ---------------------------------------------------------
# Place one prism directly below each observation point
# Depth to top (z_top) and bottom (z_bottom) of equivalent layer (negative downwards in Harmonica)
N = len(d_obs)
print(f"Number of equivalent sources / observations (N): {N}")
grid_shape = ( int(np.sqrt(N)), int(np.sqrt(N)))
dx = float(np.diff(np.sort(np.unique(easting)))[0])
dy = float(np.diff(np.sort(np.unique(northing)))[0])
prisms_layer = kn.build_prism_layer(N, dx, dy, easting, northing, z_bottom, z_top)

# ---------------------------------------------------------
# 3. Assemble Jacobian Matrix G and Track Computational Cost
# ---------------------------------------------------------

if data_type in ["magnetic", "mag"]:
    incs = metadata["incs"]
    decs = metadata["decs"]
    incf = metadata["incf"]
    decf = metadata["decf"]
    G, G_assembly_time, G_peak_ram_mb = kn.build_magnetic_jacobian_with_profiling(N, easting, 
                                                                     northing, upwards, 
                                                                     prisms_layer, incs, decs)
elif data_type in ["gravity", "grav"]:
    G, G_assembly_time, G_peak_ram_mb = kn.build_gravity_jacobian_with_profiling(N, easting, 
                                                                     northing, upwards, 
                                                                     prisms_layer)
else:
    raise ValueError("Error: data_type must be either 'magnetic', 'mag', 'gravity', or 'grav'.")


# ---------------------------------------------------------
# 4. Construct the First-Order Tikhonov Matrix (RTR)
# ---------------------------------------------------------
RTR, rtr_time, rtr_mem = kn.build_rtr_with_profiling(df, N, kn)


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


# ---------------------------------------------------------
# 5. Evaluate Data Fit
# ---------------------------------------------------------
predicted_data = G @ p_tilde
residuals = d_obs - predicted_data
rmse = np.sqrt(np.mean(residuals**2))

print(f"Inversion RMSE: {rmse:.4f} nT")
print(f"Optimal Regularization (mu): {mu_opt:.2e}")
print(f"Optimized Inversion RMSE: {rmse_opt:.4f} nT")
print(f"Negative values for magnetization: {np.count_nonzero(p_tilde < 0.0) }")


# ---------------------------------------------------------
# 6. Export data
# ---------------------------------------------------------
# Export equivalent layer parameters
data = pd.DataFrame(prisms_layer, columns=["West", "East", "South", "North", "Bottom", "Top"])
data["P_est"] = p_tilde
data.to_csv(dirpath/f"{data_type}_sources_parameters.csv", index=False)

# Export L-curve metrics (excluding the large p_est arrays for clean CSV formatting)
df_metrics_export = df_metrics.drop(columns=["p_est"])
df_metrics_export.to_csv(dirpath / f"{data_type}_l_curve_metrics.csv", index=False)

# ---------------------------------------------------------
# . Plot results
# ---------------------------------------------------------
predicted_data_grid = predicted_data.reshape(grid_shape)
d_obs_grid = d_obs.reshape(grid_shape)
northing_grid = northing.reshape(grid_shape)
easting_grid = easting.reshape(grid_shape)
p_tilde_grid = p_tilde.reshape(grid_shape)


figname = f"{dirpath}/obs_predicted.png"
ut.compare_eq_layer_results(
    easting_grid, northing_grid,
    d_obs_grid, predicted_data_grid,
    "Observed vs Predicted", figname, mu_opt,
    False)





