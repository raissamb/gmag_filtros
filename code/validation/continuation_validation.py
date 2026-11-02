import numpy as np
import pandas as pd
import harmonica as hm
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Define paths
CURRENT_FILE_PATH = Path(__file__).resolve()
CODE_ROOT = CURRENT_FILE_PATH.parent.parent

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

PROJECT_HOME = CODE_ROOT.parent
DATA_FOLDER = PROJECT_HOME / "data"
BASE_FOLDER = PROJECT_HOME / "output" 


# Import your custom modules
import modules.kernel as kn
import modules.utils as ut

# ---------------------------------------------------------
# 1. Generate Synthetic Data
# ---------------------------------------------------------
# Create a single synthetic prism as our "true" geologic source
true_source = [-2000, 2000, -2000, 2000, -3000, -1000] # W, E, S, N, Bottom, Top
mag_inc, mag_dec = 45.0, -15.0
density = 1000.0 # kg/m^3

# Create observation grids at z=0 and z=500
easting = np.linspace(-5000, 5000, 50)
northing = np.linspace(-5000, 5000, 50)
X, Y = np.meshgrid(easting, northing)
z0 = np.zeros_like(X)
z500 = np.full_like(X, 500.0)

# Calculate true fields using Harmonica
unit_mag = hm.magnetic_angles_to_vec(2.0, mag_inc, mag_dec) # 2.0 A/m intensity
mag_components_0 = hm.prism_magnetic((X, Y, z0), [true_source], magnetization=unit_mag, field="b")
true_tfa_0 = hm.total_field_anomaly(mag_components_0, mag_inc, mag_dec)

mag_components_500 = hm.prism_magnetic((X, Y, z500), [true_source], magnetization=unit_mag, field="b")
true_tfa_500 = hm.total_field_anomaly(mag_components_500, mag_inc, mag_dec)

# Flatten for your pipeline
N = X.size
x_flat, y_flat, z0_flat, true_d0 = X.flatten(), Y.flatten(), z0.flatten(), true_tfa_0.flatten()
true_d500 = true_tfa_500.flatten()

# ---------------------------------------------------------
# 2. Build Equivalent Layer
# ---------------------------------------------------------
# Place equivalent layer deeper than the z=0 observation plane
z_layer_top, z_layer_bottom = -200, -700
dx, dy = easting[1] - easting[0], northing[1] - northing[0]
prisms_layer = kn.build_prism_layer(N, dx, dy, x_flat, y_flat, z_layer_bottom, z_layer_top)

G, _, _ = kn.build_magnetic_jacobian_with_profiling(
    N, x_flat, y_flat, z0_flat, prisms_layer, mag_inc, mag_dec
)

# ---------------------------------------------------------
# 3. Solve (Minimal Regularization for Noise-Free Data)
# ---------------------------------------------------------
# Since we have no noise, we use a tiny mu just for matrix conditioning
df_temp = pd.DataFrame({"X": x_flat, "Y": y_flat})
RTR, _, _ = kn.build_rtr_with_profiling(df_temp, N, kn)

p_est, _, _ = kn.solve_system_cholesky(G, RTR, true_d0, mu=1e-10)

# ---------------------------------------------------------
# 4. Perform Upward Continuation
# ---------------------------------------------------------
dz = 500.0
d_transformed, _ = kn.compute_continuation_transformation(
    N, x_flat, y_flat, z0_flat, prisms_layer, p_est, 
    data_type="magnetic", dz=dz, incs=mag_inc, decs=mag_dec
)

# ---------------------------------------------------------
# 5. Validation Metrics
# ---------------------------------------------------------
# Calculate difference map
diff = true_d500 - d_transformed
rmse_total = np.sqrt(np.mean(diff**2))

print(f"Validation RMSE (Full Grid): {rmse_total:.4f} nT")

# Plot using your utility function
ut.plot_transformation_results(
    X, Y, 
    true_tfa_500, # Original target
    d_transformed.reshape(X.shape), # Our equivalent layer calculation
    f"Validation RMSE: {rmse_total:.4f} nT", 
    "validation_plot.png", 
    "Eq. Layer Upward Cont.",
    show=False
)