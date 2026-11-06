"""
Criaçao de modelo sintético
Fonte: prisma
Unidades:
 - Sistema de referencia: Cartesiano
 - Distância: m
 - Densidade: kg/m3
 - Intensidade de magnetização: A/m 

"""

import sys
import verde as vd
import harmonica as hm
import numpy as np
import json
from pathlib import Path
from typing import List, Union

# Define paths
CURRENT_FILE_PATH = Path(__file__).resolve()
CODE_ROOT = CURRENT_FILE_PATH.parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

PROJECT_HOME = CODE_ROOT.parent
DATA_FOLDER = PROJECT_HOME / "data"

import modules.kernel as kn
import modules.utils as ut

# =========================================================
# 1. MODEL CONFIGURATION (Edit this section)
# =========================================================
MODEL_NAME = "model3"

# Grid parameters
REGION = (0, 5000, 0, 5000)  # (west, east, south, north)
SPACING = 100
OBS_HEIGHT = 0.0

# Prism Definitions [West, East, South, North, Bottom, Top]
# You can define 1 to N prisms here
PRISMS = [
    [1000, 2000, 1000, 2000, -8000, -1000],  # Prism 1
    [3000, 4000, 3000, 4000, -5000, -500],   # Prism 2
]

# Physical Properties (must match the length of PRISMS)
DENSITIES = [2170.0, 2600.0]        # kg/m3
INTENSITIES = [3.0, 1.5]            # A/m
INC, DEC = 45.0, 12.0               # Magnetization angles

# Noise injection
NOISE_PERCENTAGE = 3.0
SEED = 32

# Equivalent Layer config for the generated metadata file
EQ_LAYER_Z_TOP = -200.0
EQ_LAYER_Z_BOTTOM = -1000.0

# Plot synthetic sources, define values for Z axis, positive upwards, meters
plot_z_top = 0
plot_z_bottom = -12000
plot_elevation = 20
plot_azimuth = -60

# =========================================================
# 2. CORE FUNCTIONS
# =========================================================
# Available at utils module

#=========================================================
# 2. EXECUTION WORKFLOW
# =========================================================
if __name__ == "__main__":
    
    # 1. Build Grid
    coords = ut.build_observation_grid(REGION, SPACING, OBS_HEIGHT)
    easting, northing, _ = coords
    
    # 2. Compute Fields
    gz, tfa = ut.generate_synthetic_fields(coords, PRISMS, DENSITIES, INTENSITIES, INC, DEC)
    
    # 3. Add Noise
    gz_noisy = ut.add_gaussian_noise(gz, NOISE_PERCENTAGE, SEED) 
    tfa_noisy = ut.add_gaussian_noise(tfa, NOISE_PERCENTAGE, SEED) 
    
    # 4. Export CSVs and Configs (Updated to pass true source parameters)
    output_dir = ut.export_synthetic_data(
        MODEL_NAME, coords, gz_noisy, tfa_noisy, 
        PRISMS, DENSITIES, INTENSITIES, INC, DEC,
        DATA_FOLDER, EQ_LAYER_Z_TOP, EQ_LAYER_Z_BOTTOM,
    )
    
    # 5. Visualizations
    print("Generating plots...")
    ut.plot_field_contourf(easting, northing, tfa_noisy, "TFA (Noisy)", "Easting (m)", "Northing (m)", "nT", 
                           Path(output_dir / f"{MODEL_NAME}_tfa_noisy.png"), show=False)
                           
    ut.plot_field_contourf(easting, northing, gz_noisy, "g_z (Noisy)", "Easting (m)", "Northing (m)", "mGal", 
                           Path(output_dir / f"{MODEL_NAME}_gz_noisy.png"), show=False)

    # Plot the 3D source model showing the entire survey region
    ut.plot_prisms_3d(
        PRISMS, 
        figname=Path(output_dir / f"{MODEL_NAME}_3d_source_full.png"), 
        region_bounds=REGION,
        z_bounds=(plot_z_bottom, plot_z_top),
        elev = plot_elevation,
        azim = plot_azimuth, 
        show=False
    )
    print("Synthetic data generation complete.")
