"""
Criaçao de modelo sintético - Dados Irregulares (Scattered)
Fonte: prisma
"""

import sys
import numpy as np
import harmonica as hm
import json
from pathlib import Path
from typing import List

# Define paths
CURRENT_FILE_PATH = Path(__file__).resolve()
CODE_ROOT = CURRENT_FILE_PATH.parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

PROJECT_HOME = CODE_ROOT.parent
DATA_FOLDER = PROJECT_HOME / "data"

import modules.kernel as kn
import modules.utils as ut
import modules.constants as ct

# =========================================================
# 1. MODEL CONFIGURATION
# =========================================================
MODEL_NAME = "model_scattered"

# Scattered Survey Parameters
REGION = (0, 5000, 0, 5000)  # (west, east, south, north)
N_POINTS = 1500              # Total number of random observation points
BASE_CLEARANCE = 150.0       # Base observation height (m)
SEED = 42                    # Seed for random coordinate generation

# Prism Definitions [West, East, South, North, Bottom, Top]
PRISMS = [
    [1000, 2000, 1000, 2000, -8000, -1000],  # Prism 1
    [3000, 4000, 3000, 4000, -5000, -500],   # Prism 2
]

# Physical Properties
DENSITIES = [2170.0, 2600.0]        # kg/m3
INTENSITIES = [3.0, 1.5]            # A/m
INC, DEC = 45.0, 12.0               # Magnetization angles

# Noise injection
NOISE_PERCENTAGE = 3.0

# Equivalent Layer config for the generated metadata file
EQ_LAYER_Z_TOP = -200.0
EQ_LAYER_Z_BOTTOM = -1000.0

# =========================================================
# 2. CORE FUNCTIONS
# =========================================================
# Available at utils module


# =========================================================
# 3. EXECUTION WORKFLOW
# =========================================================
if __name__ == "__main__":
    
    # 1. Build Grid
    coords = ut.build_scattered_observation_surface(REGION, N_POINTS, BASE_CLEARANCE, SEED)
    
    # 2. Compute Fields
    gz, tfa = ut.generate_synthetic_fields(coords, PRISMS, DENSITIES, INTENSITIES, INC, DEC)

    # 3. Add Noise
    gz_noisy = ut.add_gaussian_noise(gz, NOISE_PERCENTAGE, SEED) 
    tfa_noisy = ut.add_gaussian_noise(tfa, NOISE_PERCENTAGE, SEED) 
    
    # 4. Export CSVs and Configs
    output_dir = ut.export_synthetic_data(MODEL_NAME, coords, gz_noisy, tfa_noisy, 
                                          PRISMS, DENSITIES, INTENSITIES, INC, DEC,
                                          DATA_FOLDER)

    ut.export_survey_parameters(
        model_name=MODEL_NAME,
        survey_type="scattered",
        coords=coords,
        region=REGION,
        prisms=PRISMS,
        noise_percentage=NOISE_PERCENTAGE,
        seed=SEED,
        output_dir=output_dir,
        base_clearance=BASE_CLEARANCE,
        topography_type="sin_cos_draped",
    )

    # 5. Visualizations
    print("Generating plots...")
    ut.plot_scattered_data(
        coords[0], coords[1], tfa_noisy, 
        "TFA (Scattered)", "Easting (m)", "Northing (m)", "nT", 
        Path(output_dir / f"{MODEL_NAME}_tfa_scatter.png"), 
        show=False
    )
                           
    ut.plot_scattered_data(
        coords[0], coords[1], gz_noisy, 
        "g_z (Scattered)", "Easting (m)", "Northing (m)", "mGal", 
        Path(output_dir / f"{MODEL_NAME}_gz_scatter.png"), 
        show=False
    )

    ut.plot_scattered_data(
        coords[0], coords[1], coords[2], 
        "Observation Surface Topography", "Easting (m)", "Northing (m)", "Height (m)", 
        Path(output_dir / f"{MODEL_NAME}_topography.png"), 
        cmap="terrain",
        show=False
    )

    ut.plot_scattered_surface_3d(
        coords[0], coords[1], coords[2], 
        "3D Observation Surface", 
        Path(output_dir / f"{MODEL_NAME}_3d_surface.png"),
        show=False
    )

    ut.plot_source_footprint_and_points(
        coords, PRISMS,
        figname=output_dir / f"{MODEL_NAME}_source_footprint.png",
        region_bounds=REGION,
        show=False,
    )

    ut.plot_prisms_and_surface_3d(
        coords, PRISMS,
        figname=output_dir / f"{MODEL_NAME}_3d_sources_and_surface.png",
        region_bounds=REGION,
        show=False,
    )


    ut.plot_experiment_summary_slide(
        model_name=MODEL_NAME,
        survey_type="scattered",
        coords=coords,
        tfa=tfa_noisy,
        gz=gz_noisy,
        prisms=PRISMS,
        inc=INC,
        dec=DEC,
        noise_percentage=NOISE_PERCENTAGE,
        figname=output_dir / f"{MODEL_NAME}_slide_summary.png",
        region_bounds=REGION,
        show=False,
    )

    
    
    # # 6. Create config folder with a default equivalent layer config file
    # config_folder = DATA_FOLDER / MODEL_NAME / "config"
    # config_folder.mkdir(exist_ok=True, parents=True)
    # eq_layer_config = ct.eq_layer_config
    # savepath = config_folder / "eq_layer1.json"
    # json_str = json.dumps(eq_layer_config, indent=4)
    # with open(savepath, "w") as f:
    #     f.write(json_str)

    
    print("Synthetic scattered data generation complete.")