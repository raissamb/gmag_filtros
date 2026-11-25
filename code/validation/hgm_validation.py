import numpy as np
import pandas as pd
import harmonica as hm
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
mag_inc, mag_dec = 90.0, 0.0
unit_mag = hm.magnetic_angles_to_vec(2.0, mag_inc, mag_dec) # 2.0 A/m intensity

# Create observation grid at z=0
easting = np.linspace(-5000, 5000, 50)
northing = np.linspace(-5000, 5000, 50)
X, Y = np.meshgrid(easting, northing)
z0 = np.zeros_like(X)

# Calculate true field using Harmonica
mag_components_0 = hm.prism_magnetic((X, Y, z0), [true_source], magnetization=unit_mag, field="b")
true_tfa_0 = hm.total_field_anomaly(mag_components_0, mag_inc, mag_dec)

# --- Calculate TRUE HGM (Analytical Proxy via True Source Forward Model) ---
delta = 1.0 # 1 meter step for finite difference on true model

# True X derivative
b_x_plus = hm.prism_magnetic((X + delta, Y, z0), [true_source], magnetization=unit_mag, field="b")
tfa_x_plus = hm.total_field_anomaly(b_x_plus, mag_inc, mag_dec)
b_x_minus = hm.prism_magnetic((X - delta, Y, z0), [true_source], magnetization=unit_mag, field="b")
tfa_x_minus = hm.total_field_anomaly(b_x_minus, mag_inc, mag_dec)
true_dx = (tfa_x_plus - tfa_x_minus) / (2 * delta)

# True Y derivative
b_y_plus = hm.prism_magnetic((X, Y + delta, z0), [true_source], magnetization=unit_mag, field="b")
tfa_y_plus = hm.total_field_anomaly(b_y_plus, mag_inc, mag_dec)
b_y_minus = hm.prism_magnetic((X, Y - delta, z0), [true_source], magnetization=unit_mag, field="b")
tfa_y_minus = hm.total_field_anomaly(b_y_minus, mag_inc, mag_dec)
true_dy = (tfa_y_plus - tfa_y_minus) / (2 * delta)

# True HGM
true_hgm = np.sqrt(true_dx**2 + true_dy**2)

# Flatten for the equivalent layer pipeline
N = X.size
x_flat, y_flat, z0_flat = X.flatten(), Y.flatten(), z0.flatten()
true_d0 = true_tfa_0.flatten()
true_hgm_flat = true_hgm.flatten()

# ---------------------------------------------------------
# 2. Build Equivalent Layer
# ---------------------------------------------------------
z_layer_top, z_layer_bottom = -200, -700
dx, dy = easting[1] - easting[0], northing[1] - northing[0]
prisms_layer = kn.build_prism_layer(N, dx, dy, x_flat, y_flat, z_layer_bottom, z_layer_top)

G, _, _ = kn.build_magnetic_jacobian_with_profiling(
    N, x_flat, y_flat, z0_flat, prisms_layer, mag_inc, mag_dec
)

# ---------------------------------------------------------
# 3. Solve (Minimal Regularization for Noise-Free Data)
# ---------------------------------------------------------
df_temp = pd.DataFrame({"X": x_flat, "Y": y_flat})
RTR, _, _ = kn.build_rtr_with_profiling(df_temp, N, kn)

p_est, _, _ = kn.solve_system_cholesky(G, RTR, true_d0, mu=1e-10)

# ---------------------------------------------------------
# 4. Perform HGM Transformation
# ---------------------------------------------------------
hgm_transformed = kn.compute_hgm_transformation(
    N, x_flat, y_flat, z0_flat, prisms_layer, p_est, 
    data_type="magnetic", delta=1.0, incs=mag_inc, decs=mag_dec
)

# ---------------------------------------------------------
# 5. Validation Metrics
# ---------------------------------------------------------
# Calculate difference map between true analytical HGM and Eq Layer HGM
diff = true_hgm_flat - hgm_transformed
rmse_total = np.sqrt(np.mean(diff**2))

print(f"HGM Validation RMSE (Full Grid): {rmse_total:.4f} nT/m")

# Plot using your utility function
pathsave = BASE_FOLDER / "hgm_validation"
pathsave.mkdir(exist_ok=True, parents=True)

ut.plot_transformation_results(
    X, Y, 
    true_hgm, # Original analytical target HGM
    hgm_transformed.reshape(X.shape), # Our equivalent layer calculation
    f"HGM Validation RMSE: {rmse_total:.4f} nT/m", 
    pathsave / "hgm_validation_plot.png", 
    "Eq. Layer HGM",
    show=False
)