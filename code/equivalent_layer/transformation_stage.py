"""
This file contains the transformations instructions.

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

---------- TRANSFORMATIONS
t = A p_est

where
t = transformed field
A = Jacobian matrix associated with the transformed field
p_est = estimated physical property vector 
(it was created during the equivalent layer building process)
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

def run_transformations(dataset_name: str):
    print(f"\n--- Starting Field Transformations for {dataset_name} ---")
    
    config = ut.find_read_json_file(DATA_FOLDER / dataset_name, "pipeline_config.json")

    proj = config["project_settings"]
    cols = config["data_columns"]
    inv = config["inversion_parameters"]
    transformations_dict = config.get("transformations", {})

    data_type = proj["data_type"]
    filename = proj["filename"]
    easting_col = cols["easting_col"]
    northing_col = cols["northing_col"]
    upwards_col = cols["upwards_col"]
    field = cols["field"]

    raw_incs = inv.get("incs")
    raw_decs = inv.get("decs")
    incs = float(raw_incs) if raw_incs is not None else None
    decs = float(raw_decs) if raw_decs is not None else None

    dirpath = DATA_FOLDER / dataset_name / "output"
    
    df_obs = pd.read_csv(DATA_FOLDER / dataset_name / filename)
    df_obs = df_obs.sort_values(by=["Y", "X"]).reset_index(drop=True)
    d_obs = df_obs[field].to_numpy()
    easting = df_obs[easting_col].to_numpy()
    northing = df_obs[northing_col].to_numpy()
    upwards = df_obs[upwards_col].to_numpy()

    es_data = f"{data_type}_sources_parameters.csv"
    df_sources = pd.read_csv(dirpath / es_data)

    N = len(df_sources)
    grid_shape = (int(np.sqrt(N)), int(np.sqrt(N)))

    prisms_layer = df_sources[["West", "East", "South", "North", "Bottom", "Top"]].to_numpy()
    p_est = df_sources["P_est"].to_numpy()

    for trans_name, params in transformations_dict.items():
        if not params.get("execute", False):
            print(f"\nSkipping {trans_name} (execute=False)")
            continue
            
        if trans_name in ["upward_continuation", "downward_continuation"]:
            dz = params.get("dz", 0)
            print(f"\nRunning {trans_name} (dz={dz}m)...")
            
            d_transformed, new_upwards = kn.compute_continuation_transformation(
                N=N, easting=easting, northing=northing, upwards=upwards, 
                prisms_layer=prisms_layer, p_est=p_est, data_type=data_type,
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

        elif trans_name == "reduction_to_pole":
            if data_type not in ["magnetic", "mag"]:
                print("Skipping RTP: Data is not magnetic.")
                continue
                
            inc_new = params.get("inc_new", 90.0)
            dec_new = params.get("dec_new", 0.0)
            print(f"\nRunning {trans_name} to Inc:{inc_new}, Dec:{dec_new}...")
            
            d_transformed = kn.compute_rtp_transformation(
                N=N, easting=easting, northing=northing, upwards=upwards, 
                prisms_layer=prisms_layer, p_est=p_est, data_type=data_type,
                inc_new=inc_new, dec_new=dec_new
            )
            
            df_export = df_obs.copy()
            df_export[f"{field}_rtp"] = d_transformed
            output_file = dirpath / f"{data_type}_{trans_name}.csv"
            df_export.to_csv(output_file, index=False)

            ut.plot_transformation_results(
                easting.reshape(grid_shape), northing.reshape(grid_shape),
                d_obs.reshape(grid_shape), d_transformed.reshape(grid_shape),
                "Reduction to Pole", dirpath / f"{data_type}_{trans_name}.png", "nT", show=False
            )