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

# Inversion parameters
metadata_inv = ut.find_read_json_file(DATA_FOLDER/"model1", "inversion_config.json")
data_type = metadata_inv["data_type"]
filename = metadata_inv["filename"]
easting_col = metadata_inv["easting_col"]
northing_col = metadata_inv["northing_col"]
upwards_col = metadata_inv["upwards_col"]
field = metadata_inv["field"]
z_top = metadata_inv["z_top"]
z_bottom = metadata_inv["z_bottom"]
output_path = metadata_inv["output_folder"]

# Transformation parameters
metadata_transf = ut.find_read_json_file(DATA_FOLDER/"model1", "processing_config.json")
es_data = metadata_transf["input_data"]



dirpath = BASE_FOLDER / output_path
dirpath.mkdir(exist_ok=True, parents=True)

# Read the original survey data to get the exact observation coordinates
df_obs = pd.read_csv(DATA_FOLDER / filename)
df_obs = df_obs.sort_values(by=["Y", "X"]).reset_index(drop=True)
d_obs = df_obs[field].to_numpy()  # Input anomaly data
easting = df_obs[easting_col].to_numpy()
northing = df_obs[northing_col].to_numpy()
upwards = df_obs[upwards_col].to_numpy()

# Read Equivalent source parameters using the path from processing_config.json
df_sources = pd.read_csv(BASE_FOLDER / es_data)

N = len(df_sources)
print(f"Number of equivalent sources / observations (N): {N}")
grid_shape = (int(np.sqrt(N)), int(np.sqrt(N)))

# Extract NumPy arrays for high-performance computation
prisms_layer = df_sources[["West", "East", "South", "North", "Bottom", "Top"]].to_numpy()
p_est = df_sources["P_est"].to_numpy()

# ---------------------------------------------------------
# 2. Build transformation matrix and execute
# ---------------------------------------------------------
# Extract specific fields for magnetic transformations safely
raw_incs = metadata_inv.get("incs")
raw_decs = metadata_inv.get("decs")

incs = float(raw_incs) if raw_incs is not None else None
decs = float(raw_decs) if raw_decs is not None else None

transformations_dict = metadata_transf["transformations"]

# Loop through the dictionary of requested transformations
for trans_name, params in transformations_dict.items():
    
    # Skip the transformation if 'execute' is False
    if not params.get("execute", False):
        print(f"\nSkipping {trans_name} (execute=False)")
        continue
        
    if trans_name in ["upward_continuation", "downward_continuation"]:
        dz = params.get("dz", 0)
        print(f"\nRunning {trans_name} (dz={dz}m)...")
        
        d_transformed, new_upwards = kn.compute_continuation_transformation(
            N=N, 
            easting=easting, 
            northing=northing, 
            upwards=upwards, 
            prisms_layer=prisms_layer, 
            p_est=p_est,
            data_type=data_type,
            dz=dz, 
            incs=incs, 
            decs=decs
        )
        
        # Export the transformed data to a new CSV file
        df_export = df_obs.copy()
        df_export[upwards_col] = new_upwards
        df_export[f"{field}_transformed"] = d_transformed
        
        output_file = dirpath / f"{data_type}_{trans_name}_{dz}m.csv"
        df_export.to_csv(output_file, index=False)
        print(f"Successfully saved transformed field to: {output_file}")
        
        # Plot
        ut.plot_transformation_results(
            easting.reshape(grid_shape), 
            northing.reshape(grid_shape),
            d_obs.reshape(grid_shape), 
            d_transformed.reshape(grid_shape),
            f"{trans_name}", 
            dirpath/f"{trans_name}.png", 
            "nT",
            show = False
        )

    elif trans_name == "reduction_to_pole":
        # Placeholder for your future RTP implementation
        inc_new = params.get("inc_new")
        dec_new = params.get("dec_new")
        print(f"\nRunning {trans_name} to Inc:{inc_new}, Dec:{dec_new}...")
        # d_transformed = kn.compute_rtp_transformation(...)