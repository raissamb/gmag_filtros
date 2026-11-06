"""
This file contains the steps necessary to execute the linear inversion that
builds the Equivalent Layer using prisms as sources, one prism beneath each
observation point.

Output: datatype_source_parameters.csv
This file contains the prisms coordinates and their estimated physical property
(density or magnetization intensity). It is necessary information to perform the
transformations later on.

----------- LINEAR INVERSE SYSTEM
d = G p
f = G p_est
p_est = (G^T G + u R^T R)^{-1} G^T d

where:
d = observed data
G = Jacobian matrix
p = true physical property vector
p_est = estimated property vector
R^T R = Smooth constraint matrix
u = regularization parameter
"""

import sys
import numpy as np
from pathlib import Path
import pandas as pd
import scipy.sparse as sp

CURRENT_FILE_PATH = Path(__file__).resolve()
CODE_ROOT = CURRENT_FILE_PATH.parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

PROJECT_HOME = CODE_ROOT.parent
DATA_FOLDER = PROJECT_HOME / "data"

import modules.utils as ut
import modules.kernel as kn


def run_inversion(dataset_name: str,
                  data_type: str | None, 
                  filename: str | None, 
                  mu0_range: tuple | None):
    """
    Builds the equivalent layer of one dataset.

    data_type and filename are optional: when given, they replace the values
    of "project_settings" in pipeline_config.json. This lets the Poisson ratio
    stage invert the gravity and the magnetic data of the same dataset.

    mu0_range is optional, (mu0_min, mu0_max, steps). When given, the
    same L-curve (run_tikhonov_l_curve_scan_ces) scans
    mu = mu0 * tr(G^T G) / tr(R^T R), with mu0 from mu0_min to mu0_max.
    """

    print(f"\n--- Starting Equivalent Layer Inversion for {dataset_name} ---")
    
    # # ---------------------------------------------------------
    # # Read the single unified configuration file
    # # ---------------------------------------------------------
    config = ut.find_read_json_file(DATA_FOLDER / dataset_name, "pipeline_config.json")
    proj = config["project_settings"]
    cols = config["data_columns"]
    inv = config["inversion_parameters"]

    # Determine filename dynamically if not explicitly passed
    if filename is None:
        if data_type in ["mag", "magnetic"]:
            filename = proj.get("magnetic_file")
        elif data_type in ["grav", "gravity"]:
            filename = proj.get("gravity_file")
            
    if not filename or not data_type:
        raise ValueError("Data type and filename must be resolved before proceeding.")
    
    # Survey variables from config file
    easting_col = cols["easting_col"]
    northing_col = cols["northing_col"]
    upwards_col = cols["upwards_col"]
    field = cols["field"]

    # Inversion variables
    z_top = inv["z_top"]
    z_bottom = inv["z_bottom"]
    incs = inv.get("incs", 90.0)
    decs = inv.get("decs", 0.0)

    # Read L-curve parameters from config, with safe fallbacks
    cfg_reg_min = inv.get("reg_min", 1e-6)
    cfg_reg_max = inv.get("reg_max", 1e2)
    cfg_steps = inv.get("steps", 20)

    dirpath = DATA_FOLDER / dataset_name / "output"
    dirpath.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(DATA_FOLDER / dataset_name / filename)
    df = df.sort_values(by=["Y", "X"]).reset_index(drop=True)
    
    easting = df[easting_col].to_numpy()
    northing = df[northing_col].to_numpy()
    upwards = df[upwards_col].to_numpy()
    d_obs = df[field].to_numpy()

    # # ---------------------------------------------------------
    # # Build equivalent layer
    # # ---------------------------------------------------------
    N = len(d_obs)
    print(f"Number of equivalent sources / observations (N): {N}")
    grid_shape = (int(np.sqrt(N)), int(np.sqrt(N)))
    dx = float(np.diff(np.sort(np.unique(easting)))[0])
    dy = float(np.diff(np.sort(np.unique(northing)))[0])
    prisms_layer = kn.build_prism_layer(N, dx, dy, easting, northing, z_bottom, z_top)

    # # ---------------------------------------------------------
    # # Build Jacobian matrix
    # # ---------------------------------------------------------

    if data_type in ["magnetic", "mag"]:
        G, _, _ = kn.build_magnetic_jacobian_with_profiling(
            N, easting, northing, upwards, prisms_layer, incs, decs
        )
    elif data_type in ["gravity", "grav"]:
        G, _, _ = kn.build_gravity_jacobian_with_profiling(
            N, easting, northing, upwards, prisms_layer
        )
    else:
        raise ValueError("Error: data_type must be either 'magnetic', 'mag', 'gravity', or 'grav'.")

    
    # # ---------------------------------------------------------
    # # Build RTR matrix
    # # ---------------------------------------------------------
    RTR, _, _ = kn.build_rtr_with_profiling(df, N, kn)

    # # ---------------------------------------------------------
    # # Solve system
    # # ---------------------------------------------------------
    if mu0_range is None:
        # L-curve: Use absolute limits defined in pipeline_config.json
        reg_min, reg_max, steps = cfg_reg_min, cfg_reg_max, cfg_steps
        scale = 1.0
    else:
        # L-curve with scaled limits: mu = mu0 * tr(G^T G) / tr(R^T R).
        # The same mu0 range gives the same relative smoothness for gravity
        # and magnetic data, whose Jacobians have very different magnitudes.
        mu0_min, mu0_max, steps = mu0_range
        scale = float(np.einsum("ij,ij->", G, G)) / RTR.diagonal().sum()
        reg_min, reg_max = mu0_min * scale, mu0_max * scale
        print(f"Scaled L-curve: mu0 from {mu0_min:.1e} to {mu0_max:.1e} "
              f"({int(steps)} steps), scale tr(G^T G)/tr(R^T R) = {scale:.3e}")

    df_metrics, best_solution, _, _ = kn.run_tikhonov_l_curve_scan_ces(
        G, d_obs, RTR, reg_min, reg_max, int(steps)
    )

    mu_opt = best_solution["reg_value"]
    p_tilde = best_solution["p_est"]
    rmse_opt = best_solution["rmse"]
    if mu0_range is not None:
        print(f"L-curve choice: mu0 = {mu_opt / scale:.2e}")

    
    # # ---------------------------------------------------------
    # # Compare predicted value with observed value
    # # ---------------------------------------------------------
    predicted_data = G @ p_tilde
    residuals = d_obs - predicted_data
    rmse = np.sqrt(np.mean(residuals**2))

    print(f"Inversion RMSE: {rmse:.4f} nT")
    print(f"Optimal Regularization (mu): {mu_opt:.2e}")
    print(f"Negative values for magnetization: {np.count_nonzero(p_tilde < 0.0)}")


    # # ---------------------------------------------------------
    # # Export equivalent layer parameters file
    # # ---------------------------------------------------------
    # Differentiate filenames based on the run type
    run_type = "poisson_scaled" if mu0_range is not None else "standard"
    
    data = pd.DataFrame(prisms_layer, columns=["West", "East", "South", "North", "Bottom", "Top"])
    data["P_est"] = p_tilde
    
    # Export CSVs with the run_type suffix
    data.to_csv(dirpath / f"{data_type}_{run_type}_sources_parameters.csv", index=False)

    df_metrics_export = df_metrics.drop(columns=["p_est"])
    df_metrics_export.to_csv(dirpath / f"{data_type}_{run_type}_l_curve_metrics.csv", index=False)

    # # ---------------------------------------------------------
    # # Plot results
    # # ---------------------------------------------------------
    # Save the figure with the run_type suffix
    figname = dirpath / f"{data_type}_{run_type}_obs_predicted.png"
    ut.compare_eq_layer_results(
        easting.reshape(grid_shape), northing.reshape(grid_shape),
        d_obs.reshape(grid_shape), predicted_data.reshape(grid_shape),
        "Observed vs Predicted", figname, mu_opt, False
    )

