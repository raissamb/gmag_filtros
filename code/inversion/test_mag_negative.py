# Import packages
import sys
import verde as vd
import harmonica as hm
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
OUTPUT_FOLDER = PROJECT_HOME / "output" / "mag_negative"
OUTPUT_FOLDER.mkdir()

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
# 1. Create Synthetic Data
# ---------------------------------------------------------
xmin = 0
xmax = 100
ymin = 0
ymax = 100
spacing = 5
height = 0
region = (xmin, xmax, ymin, ymax)
coordinates = vd.grid_coordinates(
        region, spacing=spacing, extra_coords=height)
prism_dimensions = [25.0, 75.0, 25.0, 75.0, -20.0, -10.0]
inc1 = 90
inc2 = 10
dec = 0
intensity = 3

# Add noise
seed = 43
noise_percentage = 2


# ---------------------------------------------------------
# 1. Define sources
# ---------------------------------------------------------
# source at pole
pole_dir = kn.get_direction_cosines_enu(inc1, dec)
pole_mag_vector = kn.get_magnetization_vector_enu(intensity, pole_dir)
pole_b_e, pole_b_n, pole_b_u = hm.prism_magnetic(
        coordinates, prism_dimensions, pole_mag_vector, field="b")
pole_tfa = hm.total_field_anomaly([pole_b_e, pole_b_n, pole_b_u], inc1, dec)
pole_tfa_noisy = ut.add_gaussian_noise(pole_tfa, noise_percentage, seed) 

# source at equator
eq_dir = kn.get_direction_cosines_enu(inc2, dec)
eq_mag_vector = kn.get_magnetization_vector_enu(intensity, eq_dir)
eq_b_e, eq_b_n, eq_b_u = hm.prism_magnetic(
        coordinates, prism_dimensions, eq_mag_vector, field="b")
eq_tfa = hm.total_field_anomaly([eq_b_e, eq_b_n, eq_b_u], inc2, dec)
eq_tfa_noisy = ut.add_gaussian_noise(eq_tfa, noise_percentage, seed) 


# ---------------------------------------------------------
# 2. Equivalent Layer using prisms for the source at equator
# ---------------------------------------------------------
# Place one prism directly below each observation point
# Depth to top (z_top) and bottom (z_bottom) of equivalent layer (negative downwards in Harmonica)
z_top = -10.0       
z_bottom = -15.0   
N = 441

# ---------------------------------------------------------
# 3. Format Data for Strict Spatial Ordering
# ---------------------------------------------------------
# Create a dataframe to keep coordinates and anomalies locked together
df = pd.DataFrame({
    "X": coordinates[0].ravel(),
    "Y": coordinates[1].ravel(),
    "Z": coordinates[2].ravel(),
    "eq_tfa": eq_tfa_noisy.ravel(),
    "pole_tfa": pole_tfa_noisy.ravel()
})

# Enforce strict row-major sorting for the discrete Laplacian (RTR)
df_sub = df.sort_values(by=["Y", "X"]).reset_index(drop=True)

easting = df_sub["X"].to_numpy()
northing = df_sub["Y"].to_numpy()
upwards = df_sub["Z"].to_numpy()
d_obs_eq = df_sub["eq_tfa"].to_numpy()
d_true_pole = df_sub["pole_tfa"].to_numpy()

dx = float(np.diff(np.sort(np.unique(easting)))[0])
dy = float(np.diff(np.sort(np.unique(northing)))[0])

# ---------------------------------------------------------
# 4. Build the Equivalent Layer and Regularization Matrix
# ---------------------------------------------------------
print("\n--- Constructing Equivalent Layer ---")
prisms_layer = kn.build_prism_layer(N, dx, dy, easting, northing, z_bottom, z_top)

RTR, _, _ = kn.build_rtr_with_profiling(df_sub, N, kn)

# ---------------------------------------------------------
# 5. Invert Equator Data (Estimate Magnetization)
# ---------------------------------------------------------
print("\n--- Assembling Equator Jacobian (Inc=10) ---")
# Build G using the Equator magnetic parameters
G_eq, _, _ = kn.build_jacobian_with_profiling(N, easting, northing, upwards, prisms_layer, inc2, dec)

print("\n--- Running L-Curve Inversion ---")
reg_min = 1e-6
reg_max = 1e2
df_lcurve, best_solution, _, _ = kn.run_tikhonov_l_curve_scan_ces(
    G_eq, d_obs_eq, RTR, reg_min, reg_max, steps=20
)

p_est = best_solution["p_est"]
negative_count = np.sum(p_est < 0)
print(f"Optimal Regularization (mu): {best_solution['reg_value']:.2e}")
print(f"Number of negative magnetization values estimated: {negative_count} out of {N}")

# ---------------------------------------------------------
# 6. Apply RTP Transformation
# ---------------------------------------------------------
print("\n--- Assembling Pole Jacobian for RTP (Inc=90) ---")
# Build a new G using the Pole magnetic parameters
G_pole, _, _ = kn.build_jacobian_with_profiling(N, easting, northing, upwards, prisms_layer, inc1, dec)

# Project the estimated magnetization to the pole
rtp_predicted = G_pole @ p_est

# ---------------------------------------------------------
# 7. Evaluate RTP Accuracy
# ---------------------------------------------------------
residuals = rtp_predicted - d_true_pole
rmse = np.sqrt(np.mean(residuals**2))

print(f"\n--- Results ---")
print(f"RMSE between Predicted RTP and True Pole Anomaly: {rmse:.4f} nT")
print(f"Max Absolute Error: {np.max(np.abs(residuals)):.4f} nT")

# Optional: Plot the results side-by-side
figname = f"{OUTPUT_FOLDER}/comparacao_pole_rtp.png"
mu_opt = best_solution["reg_value"]
ut.compare_eq_layer_results(
    easting.reshape((21, 21)), northing.reshape((21, 21)),
    d_true_pole.reshape((21, 21)), rtp_predicted.reshape((21, 21)),
    "Source at Pole vs RTP", figname, mu_opt,
    False)
