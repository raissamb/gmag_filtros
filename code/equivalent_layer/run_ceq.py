import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Define paths
CURRENT_FILE_PATH = Path(__file__).resolve()
CODE_ROOT = CURRENT_FILE_PATH.parent.parent

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

PROJECT_HOME = CODE_ROOT.parent
DATA_FOLDER = PROJECT_HOME / "data"
BASE_FOLDER = PROJECT_HOME / "output" 

import modules.utils as ut
import modules.kernel as kn

def main():
    # ---------------------------------------------------------
    # 1. Load Data & Configs
    # ---------------------------------------------------------
    dataset_name = "model2"  # WRITE HERE THE FOLDER NAME OF WHERE DATASET IS! 
                            # THEN RUN TO EXECUTE EQ. LAYER BUILDING AND FIELD TRANSFORMATIONS
    
    
    print("==== Start ====\n")
    # Read the single unified configuration file
    config = ut.find_read_json_file(DATA_FOLDER / dataset_name, "pipeline_config.json")
    
    # Extract grouped parameters
    proj = config["project_settings"]
    cols = config["data_columns"]
    inv = config["inversion_parameters"]
    
    data_type = proj["data_type"]
    filename = proj["filename"]
    output_path = proj["output_folder"]
    
    easting_col = cols["easting_col"]
    northing_col = cols["northing_col"]
    upwards_col = cols["upwards_col"]
    field = cols["field"]
    
    z_top = inv["z_top"]
    z_bottom = inv["z_bottom"]
    incs = inv.get("incs", 90.0)
    decs = inv.get("decs", 0.0)

    dirpath = BASE_FOLDER / output_path
    dirpath.mkdir(exist_ok=True, parents=True)

    print(f"Data type: {data_type}")

    # Note: Keep your dataset_name folder path if that's how your data is structured
    df_obs = pd.read_csv(DATA_FOLDER / dataset_name / filename)
    df_obs = df_obs.sort_values(by=["Y", "X"]).reset_index(drop=True)
    
    easting = df_obs[easting_col].to_numpy()
    northing = df_obs[northing_col].to_numpy()
    upwards = df_obs[upwards_col].to_numpy()
    d_obs = df_obs[field].to_numpy()

    N = len(d_obs)
    print(f"Number of equivalent sources / observations (N): {N}")
    grid_shape = (int(np.sqrt(N)), int(np.sqrt(N)))
    dx = float(np.diff(np.sort(np.unique(easting)))[0])
    dy = float(np.diff(np.sort(np.unique(northing)))[0])

    # ---------------------------------------------------------
    # 2. Equivalent Layer Inversion
    # ---------------------------------------------------------
    prisms_layer = kn.build_prism_layer(N, dx, dy, easting, northing, z_bottom, z_top)

    if data_type in ["magnetic", "mag"]:
        G, G_time, G_ram = kn.build_magnetic_jacobian_with_profiling(
            N, easting, northing, upwards, prisms_layer, incs, decs
        )
    elif data_type in ["gravity", "grav"]:
        G, G_time, G_ram = kn.build_gravity_jacobian_with_profiling(
            N, easting, northing, upwards, prisms_layer
        )
    else:
        raise ValueError("Invalid data_type in configuration.")

    RTR, rtr_time, rtr_mem = kn.build_rtr_with_profiling(df_obs, N, kn)

    df_metrics, best_sol, sweep_time, sweep_ram = kn.run_tikhonov_l_curve_scan_ces(
        G, d_obs, RTR, reg_min=1e-6, reg_max=1e2, steps=20
    )

    p_tilde = best_sol["p_est"]
    mu_opt = best_sol["reg_value"]
    
    predicted_data = G @ p_tilde
    rmse = np.sqrt(np.mean((d_obs - predicted_data)**2))
    print(f"Optimized Inversion RMSE: {rmse:.4f} nT")

    # Export Inversion Results
    df_sources = pd.DataFrame(prisms_layer, columns=["West", "East", "South", "North", "Bottom", "Top"])
    df_sources["P_est"] = p_tilde
    df_sources.to_csv(dirpath / f"{data_type}_sources_parameters.csv", index=False)
    df_metrics.drop(columns=["p_est"]).to_csv(dirpath / f"{data_type}_l_curve_metrics.csv", index=False)
    
    ut.compare_eq_layer_results(
        easting.reshape(grid_shape), northing.reshape(grid_shape),
        d_obs.reshape(grid_shape), predicted_data.reshape(grid_shape),
        "Observed vs Predicted", dirpath / f"{data_type}_obs_predicted.png", mu_opt, False
    )

    # ---------------------------------------------------------
    # 3. Field Transformations (In-Memory)
    # ---------------------------------------------------------
    print(" ")
    print("--- Field Transformations ----- \n")
    transformations_dict = config.get("transformations", {})

    for trans_name, params in transformations_dict.items():
        if not params.get("execute", False):
            print(f"Skipping {trans_name} (execute=False)")
            continue
            
        if trans_name in ["upward_continuation", "downward_continuation"]:
            dz = params.get("dz", 0)
            print(f"\nRunning {trans_name} (dz={dz}m)...")
            
            d_transformed, new_upwards = kn.compute_continuation_transformation(
                N=N, easting=easting, northing=northing, upwards=upwards, 
                prisms_layer=prisms_layer, p_est=p_tilde, data_type=data_type,
                dz=dz, incs=incs, decs=decs
            )
            
            df_export = df_obs.copy()
            df_export[upwards_col] = new_upwards
            df_export[f"{field}_transformed"] = d_transformed
            
            output_file = dirpath / f"{data_type}_{trans_name}_{dz}m.csv"
            df_export.to_csv(output_file, index=False)
            
            ut.plot_transformation_results(
                easting.reshape(grid_shape), northing.reshape(grid_shape),
                d_obs.reshape(grid_shape), d_transformed.reshape(grid_shape),
                f"{trans_name}", dirpath / f"{data_type}_{trans_name}.png", "nT", show=False
            )
            
if __name__ == "__main__":
    main()