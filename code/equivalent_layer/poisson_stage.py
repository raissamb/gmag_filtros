"""
This file contains the steps to estimate the Poisson ratio (M/rho) from the
ratio between two equivalent layers (Santos, 2019):

1. equivalent layer of the gravity data   -> density p_grav (kg/m^3);
2. equivalent layer of the magnetic data  -> magnetization p_mag (A/m);
3. M/rho = p_mag / p_grav, prism by prism (A m^2/kg).

Both layers must have the same prisms: same grid of observations, same
z_top and z_bottom (taken from "inversion_parameters").

Configuration (pipeline_config.json):
    "poisson_ratio": {
        "execute": true,
        "gravity_file": "model1_grav.csv",
        "magnetic_file": "model1_mag.csv",
        "density_threshold": 0.05,
        "mu0_min": 1e-6,
        "mu0_max": 1e4,
        "steps": 30
    }

Each layer chooses its mu with the L-curve (run_tikhonov_l_curve_scan_ces),
but over the same scaled range, mu = mu0 * tr(G^T G) / tr(R^T R), with mu0
from mu0_min to mu0_max. With absolute limits (1e-6 to 1e2) the gravity
layer is smoothed far more than the magnetic one (their Jacobians differ by
orders of magnitude) and the ratio loses its meaning. mu0_max must be large
(1e4): the L-curve corner depends on the end points of the scanned range.

Output: poisson_layer_ratio.csv and poisson_layer_ratio.png in data/<dataset>/output.
"""

import sys
import numpy as np
from pathlib import Path
import pandas as pd

CURRENT_FILE_PATH = Path(__file__).resolve()
CODE_ROOT = CURRENT_FILE_PATH.parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

PROJECT_HOME = CODE_ROOT.parent
DATA_FOLDER = PROJECT_HOME / "data"

import modules.utils as ut
import modules.kernel as kn
import inversion_stage as inv_stage

PRISM_COLUMNS = ["West", "East", "South", "North", "Bottom", "Top"]


def run_poisson_ratio(dataset_name: str):
    print(f"\n--- Starting Poisson Ratio (layer ratio) for {dataset_name} ---")

    config = ut.find_read_json_file(DATA_FOLDER / dataset_name, "pipeline_config.json")
    
    # Extract from the new nested structures
    params = config.get("joint_analysis", {}).get("poisson_ratio", {})
    proj = config.get("project_settings", {})

    if not params.get("execute", False):
        print("Skipping poisson_ratio (execute=False)")
        return None

    # Fetch filenames from project_settings
    gravity_file = proj.get("gravity_file")
    magnetic_file = proj.get("magnetic_file")

    if not gravity_file or not magnetic_file:
        raise ValueError("Both gravity_file and magnetic_file must be defined in project_settings.")

    threshold = params.get("density_threshold", 0.05)
    mu0_range = (float(params.get("mu0_min", 1.0e-6)),
                 float(params.get("mu0_max", 1.0e4)),
                 int(params.get("steps", 30)))

    dirpath = DATA_FOLDER / dataset_name / "output"

    # # ---------------------------------------------------------
    # # 1 and 2. Equivalent layers: gravity first, then magnetic
    # # ---------------------------------------------------------
    inv_stage.run_inversion(dataset_name, data_type="gravity", filename=gravity_file, mu0_range=mu0_range)
    inv_stage.run_inversion(dataset_name, data_type="magnetic", filename=magnetic_file, mu0_range=mu0_range)

    df_grav = pd.read_csv(dirpath / "gravity_poisson_scaled_sources_parameters.csv")
    df_mag = pd.read_csv(dirpath / "magnetic_poisson_scaled_sources_parameters.csv")

    prisms_grav = df_grav[PRISM_COLUMNS].to_numpy()
    prisms_mag = df_mag[PRISM_COLUMNS].to_numpy()
    p_grav = df_grav["P_est"].to_numpy()
    p_mag = df_mag["P_est"].to_numpy()

    # # ---------------------------------------------------------
    # # 3. Ratio between the layers
    # # ---------------------------------------------------------
    ratio = kn.compute_layer_ratio(p_mag, p_grav, prisms_mag, prisms_grav, threshold)

    valid = np.isfinite(ratio)
    print(f"\nPrisms with defined ratio: {valid.sum()} of {ratio.size} "
          f"(|density| > {threshold:.0%} of the maximum |density|)")
    if valid.any():
        print(f"M/rho median: {np.median(ratio[valid]):.4e} A m^2/kg")
        print(f"M/rho min / max: {ratio[valid].min():.4e} / {ratio[valid].max():.4e} A m^2/kg")
        print(f"Negative ratios: {np.count_nonzero(ratio[valid] < 0.0)}")

    # # ---------------------------------------------------------
    # # Export
    # # ---------------------------------------------------------
    df_export = df_grav[PRISM_COLUMNS].copy()
    df_export["P_grav"] = p_grav
    df_export["P_mag"] = p_mag
    df_export["M_rho"] = ratio
    df_export.to_csv(dirpath / "poisson_layer_ratio.csv", index=False)

    # # ---------------------------------------------------------
    # # Plot (prism centres, grid sorted by Y then X as in the inversion)
    # # ---------------------------------------------------------
    N = ratio.size
    grid_shape = (int(np.sqrt(N)), int(np.sqrt(N)))
    easting = 0.5 * (prisms_grav[:, 0] + prisms_grav[:, 1])
    northing = 0.5 * (prisms_grav[:, 2] + prisms_grav[:, 3])
    ut.plot_field_contourf(
        easting.reshape(grid_shape), northing.reshape(grid_shape),
        ratio.reshape(grid_shape), "Poisson ratio (layer ratio)",
        "Easting (m)", "Northing (m)", "M/rho (A m²/kg)",
        dirpath / "poisson_layer_ratio.png", show=False
    )

    return ratio
