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

#--------------------------

# Suffixes
TRANS_SUFFIXES = {
    "upward_continuation": "_uc",
    "downward_continuation": "_dc",
    "reduction_to_pole": "_rtp",
    "horizontal_gradient_magnitude": "_hgm",
}

def get_output_field_name(input_field: str, trans_type: str, params: dict) -> str:
    if "output_field" in params:
        return params["output_field"]
    base_field = input_field
    for suffix in ("_transformed", "_uc", "_dc", "_rtp", "_hgm"):
        if base_field.endswith(suffix):
            base_field = base_field[: -len(suffix)]
            break
    return f"{base_field}{TRANS_SUFFIXES.get(trans_type, '_transformed')}"


def run_transformations(dataset_name: str, trans_config_name: str):
    print(f"\n--- Starting Field Transformations for {dataset_name} ---")

    trans_config = ut.find_read_json_file(DATA_FOLDER / dataset_name, trans_config_name)
    if not trans_config.get("execute", False):
        print(f"Skipping transformation {trans_config_name} (execute=False)")
        return
        
    eq_layer_name = trans_config["eq_layer_config"]
    trans_type = trans_config["transformation_type"]
    data_type = trans_config["data_type"]
    params = trans_config["parameters"]

    # Read the parent equivalent layer config to get grid metadata
    eq_config = ut.find_read_json_file(DATA_FOLDER / dataset_name / "config", f"{eq_layer_name}.json")
    proj = eq_config["project_settings"]
    cols = eq_config["data_columns"]
    inv = eq_config["inversion_parameters"]
    
    if data_type in ["mag", "magnetic"]:
        filename = proj.get("magnetic_file")
    elif data_type in ["grav", "gravity"]:
        filename = proj.get("gravity_file")
    else:
        raise ValueError("Invalid data_type provided to transformations.")
        
    easting_col = cols["easting_col"]
    northing_col = cols["northing_col"]
    upwards_col = cols["upwards_col"]
    field = cols["field"]

    raw_incs = inv.get("incs")
    raw_decs = inv.get("decs")
    incs = float(raw_incs) if raw_incs is not None else None
    decs = float(raw_decs) if raw_decs is not None else None
        
    # Setup nested output directory
    trans_name = Path(trans_config_name).stem
    dirpath = DATA_FOLDER / dataset_name / eq_layer_name / "transformations" / trans_name
    dirpath.mkdir(exist_ok=True, parents=True)

    # 1. Load the original observation data so we can copy it into df_export
    input_path = ut.resolve_dataset_file(DATA_FOLDER / dataset_name, filename)
    df_obs = pd.read_csv(input_path)
    df_obs = df_obs.sort_values(by=["Y", "X"]).reset_index(drop=True)
    d_obs = df_obs[field].to_numpy()
    easting = df_obs[easting_col].to_numpy()
    northing = df_obs[northing_col].to_numpy()
    upwards = df_obs[upwards_col].to_numpy()

    # 2. Read the estimated property data from the isolated equivalent layer folder
    es_data_path = DATA_FOLDER / dataset_name / eq_layer_name / f"{data_type}_standard_equivalent_sources_parameters.csv"
    df_sources = pd.read_csv(es_data_path)
    
    N = len(df_sources)
    grid_shape = (int(np.sqrt(N)), int(np.sqrt(N)))

    prisms_layer = df_sources[["West", "East", "South", "North", "Bottom", "Top"]].to_numpy()
    p_est = df_sources["P_est"].to_numpy()

    # --- TRANSFORMATION TREE
    
    ## Continuation
    if trans_type in ["upward_continuation", "downward_continuation"]:
        dz = params.get("dz", 0)
        print(f"\nRunning {trans_type} (dz={dz}m)...")
        
        d_transformed, new_upwards = kn.compute_continuation_transformation(
            N=N, easting=easting, northing=northing, upwards=upwards, 
            prisms_layer=prisms_layer, p_est=p_est, data_type=data_type,
            dz=dz, incs=incs, decs=decs
        )
        
        # Define df_export by copying df_obs
        df_export = df_obs.copy()
        df_export[upwards_col] = new_upwards
        out_col = get_output_field_name(field, trans_type, params)
        df_export[out_col] = d_transformed
        
        output_file = dirpath / f"{data_type}_{trans_type}_{dz}m.csv"
        df_export.to_csv(output_file, index=False)
        
        ut.plot_transformation_results(
            easting.reshape(grid_shape), northing.reshape(grid_shape),
            d_obs.reshape(grid_shape), d_transformed.reshape(grid_shape),
            f"{trans_type}", dirpath / f"{data_type}_{trans_type}_{dz}m.png", "nT", show=False
        )

    ## HGM
    elif trans_type == "horizontal_gradient_magnitude":
        delta = float(params.get("delta", 1.0))
        print(f"\nRunning {trans_type} (delta={delta}m)...")

        d_transformed = kn.compute_hgm_transformation(
            N=N, easting=easting, northing=northing, upwards=upwards,
            prisms_layer=prisms_layer, p_est=p_est, data_type=data_type,
            delta=delta, incs=incs, decs=decs
        )

        df_export = df_obs.copy()
        out_col = get_output_field_name(field, trans_type, params)
        df_export[out_col] = d_transformed

        output_file = dirpath / f"{data_type}_{trans_type}.csv"
        df_export.to_csv(output_file, index=False)

        unit = "nT/m" if data_type in ["mag", "magnetic"] else "mGal/m"
        ut.plot_transformation_results(
            easting.reshape(grid_shape), northing.reshape(grid_shape),
            d_obs.reshape(grid_shape), d_transformed.reshape(grid_shape),
            "Horizontal Gradient Magnitude", dirpath / f"{data_type}_{trans_type}.png", unit, show=False
        )

    ## RTP
    elif trans_type == "reduction_to_pole":
        solver_used = inv.get("solver", "cholesky").lower()
        neg_count = int(np.count_nonzero(p_est < 0.0))
        if solver_used not in ["bvls", "trf"] or neg_count > 0:
            print(
                f"WARNING: Running RTP on equivalent layer '{eq_layer_name}' "
                f"(solver='{solver_used}', negative magnetizations={neg_count}). "
                f"At low inclinations, use solver='bvls' to avoid striation artifacts."
            )
        
        inc_new = params.get("inc_new", 90.0)
        dec_new = params.get("dec_new", 0.0)
        print(f"\nRunning {trans_type} to Inc:{inc_new}, Dec:{dec_new}...")
        
        d_transformed = kn.compute_rtp_transformation(
            N=N, easting=easting, northing=northing, upwards=upwards, 
            prisms_layer=prisms_layer, p_est=p_est, data_type=data_type,
            inc_new=inc_new, dec_new=dec_new
        )
        
        # Define df_export by copying df_obs
        df_export = df_obs.copy()
        out_col = get_output_field_name(field, trans_type, params)
        df_export[out_col] = d_transformed
        
        output_file = dirpath / f"{data_type}_{trans_type}.csv"
        df_export.to_csv(output_file, index=False)
        
        ut.plot_transformation_results(
            easting.reshape(grid_shape), northing.reshape(grid_shape),
            d_obs.reshape(grid_shape), d_transformed.reshape(grid_shape),
            "Reduction to Pole", dirpath / f"{data_type}_{trans_type}.png", "nT", show=False
        )

