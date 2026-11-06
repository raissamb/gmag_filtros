"""
Validation of the Poisson ratio by the ratio between equivalent layers.

A single prism with density rho and magnetization M (induced, I = 45°,
D = 12°) generates gravity and magnetic data, without and with 3% noise.
The same steps of the pipeline (prism layer, Jacobians, RTR and the L-curve
run_tikhonov_l_curve_scan_ces over the scaled range of poisson_stage.py)
build the two equivalent layers, and compute_layer_ratio must recover M/rho
over the body. The ratio is also computed with the absolute L-curve range
(1e-6 to 1e2), to show why it is not used.

The magnetization vector comes from hm.magnetic_angles_to_vec (Up = -sin I).
"""

import io
import contextlib
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
OUTPUT_FOLDER = PROJECT_HOME / "output" / "poisson_validation"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

import modules.kernel as kn
import modules.utils as ut

# ---------------------------------------------------------
# 1. Synthetic body and data
# ---------------------------------------------------------
side, spacing = 41, 100.0
inc, dec = 45.0, 12.0
density = 200.0        # kg/m^3
magnetization = 2.0    # A/m
true_ratio = magnetization / density
noise_percentage, seed = 3, 32

g = np.arange(side) * spacing
X, Y = np.meshgrid(g, g)              # rows = Y (north), columns = X (east)
easting, northing = X.ravel(), Y.ravel()
upwards = np.zeros_like(easting)
c = g.mean()
body = [c - 400, c + 400, c - 400, c + 400, -2000.0, -1000.0]   # W, E, S, N, Bottom, Top

gz = hm.prism_gravity((easting, northing, upwards), [body], [density], field="g_z")
me, mn, mu = hm.magnetic_angles_to_vec(magnetization, inc, dec)
b = hm.prism_magnetic((easting, northing, upwards), [body],
                      (np.array([me]), np.array([mn]), np.array([mu])), field="b")
tfa = hm.total_field_anomaly(b, inc, dec)

gz_noisy = ut.add_gaussian_noise(gz, noise_percentage, seed)
tfa_noisy = ut.add_gaussian_noise(tfa, noise_percentage, seed)

# ---------------------------------------------------------
# 2. Jacobians and RTR (same steps as inversion_stage.py)
# ---------------------------------------------------------
N = easting.size
mu0_range = (1.0e-6, 1.0e4, 30)       # as in pipeline_config.json
z_top, z_bottom = -300.0, -800.0
prisms = kn.build_prism_layer(N, spacing, spacing, easting, northing, z_bottom, z_top)
RTR, _, _ = kn.build_rtr_with_profiling(pd.DataFrame({"X": easting, "Y": northing}), N, kn)
G_grav, _, _ = kn.build_gravity_jacobian_with_profiling(N, easting, northing, upwards, prisms)
G_mag, _, _ = kn.build_magnetic_jacobian_with_profiling(N, easting, northing, upwards, prisms, inc, dec)

over_body = ((easting > body[0]) & (easting < body[1])
             & (northing > body[2]) & (northing < body[3]))


def l_curve_layer(G, d, scaled):
    """Her L-curve; scaled=True uses mu0_range times tr(G^T G)/tr(R^T R)."""
    if scaled:
        scale = float(np.einsum("ij,ij->", G, G)) / RTR.diagonal().sum()
        reg_min, reg_max, steps = mu0_range[0] * scale, mu0_range[1] * scale, mu0_range[2]
    else:
        scale, reg_min, reg_max, steps = 1.0, 1e-6, 1e2, 20
    with contextlib.redirect_stdout(io.StringIO()):
        _, best, _, _ = kn.run_tikhonov_l_curve_scan_ces(G, d, RTR, reg_min, reg_max, steps)
    return best["p_est"], best["reg_value"] / scale


def case(label, d_grav, d_mag, scaled):
    p_grav, mu0_grav = l_curve_layer(G_grav, d_grav, scaled)
    p_mag, mu0_mag = l_curve_layer(G_mag, d_mag, scaled)
    ratio = kn.compute_layer_ratio(p_mag, p_grav, prisms, prisms, threshold=0.05)
    valid = np.isfinite(ratio)
    median_body = np.median(ratio[over_body & valid])
    error = abs(median_body - true_ratio) / true_ratio
    name = "mu0" if scaled else "mu"
    print(f"\n{label}")
    print(f"  L-curve {name} (gravity / magnetic): {mu0_grav:.2e} / {mu0_mag:.2e}")
    print(f"  Negative prisms (gravity / magnetic): {np.sum(p_grav < 0)} / {np.sum(p_mag < 0)}")
    print(f"  Median M/rho over the body: {median_body:.4e} A m^2/kg (error {error:.1%})")
    print(f"  Negative ratios: {np.count_nonzero(ratio[valid] < 0.0)} of {valid.sum()} defined")
    return ratio, error


# ---------------------------------------------------------
# 3. Ratio between the layers
# ---------------------------------------------------------
print("")
print("--- Poisson ratio validation ---")
print(f"True M/rho: {true_ratio:.4e} A m^2/kg")

ratio_clean, err_clean = case("No noise, scaled L-curve", gz, tfa, True)
ratio_noisy, err_noisy = case(f"{noise_percentage}% noise, scaled L-curve", gz_noisy, tfa_noisy, True)
case(f"{noise_percentage}% noise, absolute L-curve (1e-6 to 1e2)", gz_noisy, tfa_noisy, False)

print("")
print(f"{'OK' if err_clean < 0.10 else 'FAIL'}: no noise, error over the body below 10%")
print(f"{'OK' if err_noisy < 0.20 else 'FAIL'}: {noise_percentage}% noise, error over the body below 20%")

# Different prisms must raise an error
shifted = prisms.copy()
shifted[:, 4:] -= 100.0
try:
    kn.compute_layer_ratio(np.ones(N), np.ones(N), prisms, shifted)
    print("FAIL: layers with different prisms were accepted")
except ValueError:
    print("OK: layers with different prisms raise an error")

for ratio, tag in ((ratio_clean, "no_noise"), (ratio_noisy, "noise")):
    figname = OUTPUT_FOLDER / f"poisson_layer_ratio_validation_{tag}.png"
    ut.plot_field_contourf(X, Y, ratio.reshape(X.shape),
                           f"M/rho, {tag.replace('_', ' ')} (true = {true_ratio:.3g})",
                           "Easting (m)", "Northing (m)", "M/rho (A m²/kg)",
                           figname, levels=np.linspace(0.0, 0.02, 21), show=False)
    print(f"Figure: {figname}")
