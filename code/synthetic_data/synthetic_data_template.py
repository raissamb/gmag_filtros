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

def build_observation_grid(region: tuple, spacing: float, height: float):
    """Generates the 3D coordinate arrays for the survey using Verde."""
    print(f"Building grid over region {region} with spacing {spacing}m...")
    coordinates = vd.grid_coordinates(region, spacing=spacing, extra_coords=height)
    return coordinates

def generate_synthetic_fields(
    coords: tuple, 
    prisms: List[list], 
    densities: List[float], 
    intensities: List[float], 
    inc: float, 
    dec: float
):
    """Computes gravity and magnetic fields for N prisms using Harmonica."""
    print(f"Computing forward models for {len(prisms)} prisms...")
    
    # Calculate Gravity (g_z)
    # Harmonica accepts lists of prisms and lists of densities directly
    gz = hm.prism_gravity(coords, prisms, density=densities, field="g_z")
    
    # Calculate Magnetic (TFA)
    mag_e = []
    mag_n = []
    mag_u = []
    
    # Build component arrays across all prisms
    for intensity in intensities:
        cosine_dir = kn.get_direction_cosines_enu(inc, dec)
        vector = kn.get_magnetization_vector_enu(intensity, cosine_dir)
        mag_e.append(vector[0])
        mag_n.append(vector[1])
        mag_u.append(vector[2])
    
    # Harmonica expects a tuple of 1D arrays for multiple varying magnetizations
    mag_vectors = (np.array(mag_e), np.array(mag_n), np.array(mag_u))
    
    b_e, b_n, b_u = hm.prism_magnetic(coords, prisms, magnetization=mag_vectors, field="b")
    tfa = hm.total_field_anomaly([b_e, b_n, b_u], inc, dec)
    
    return gz, tfa

def export_synthetic_data(
    model_name: str, 
    coords: tuple, 
    gz_noisy: np.ndarray, 
    tfa_noisy: np.ndarray,
    prisms: List[list], 
    densities: List[float], 
    intensities: List[float], 
    inc: float, 
    dec: float
):
    """Saves arrays to CSV files and generates the metadata and true source configs."""
    datapath = DATA_FOLDER / model_name
    datapath.mkdir(exist_ok=True, parents=True)
    
    easting, northing, upwards = coords
    header = "X,Y,Z,obs"
    
    print(f"Exporting data to {datapath}...")
    
    # Gravity export
    grav_array = np.column_stack((easting.ravel(), northing.ravel(), upwards.ravel(), gz_noisy.ravel()))
    np.savetxt(datapath / f"{model_name}_grav.csv", grav_array, delimiter=",", header=header, comments="", fmt="%.6f")
    
    # Magnetic export
    mag_array = np.column_stack((easting.ravel(), northing.ravel(), upwards.ravel(), tfa_noisy.ravel()))
    np.savetxt(datapath / f"{model_name}_mag.csv", mag_array, delimiter=",", header=header, comments="", fmt="%.6f")
    
    # Create Pipeline Config
    ut.create_pipeline_config(
        savepath=datapath,
        data_type="magnetic", 
        filename=f"{model_name}_mag.csv",
        z_top=EQ_LAYER_Z_TOP,
        z_bottom=EQ_LAYER_Z_BOTTOM,
        incf=inc, decf=dec, incs=inc, decs=dec
    )
    
    # Create True Source Config
    true_source_config = {
        "prisms": prisms,
        "densities": densities,
        "intensities": intensities,
        "inc": inc,
        "dec": dec
    }
    
    source_filepath = datapath / "true_source_config.json"
    with open(source_filepath, "w", encoding="utf-8") as f:
        json.dump(true_source_config, f, indent=4)
    print(f"True source config successfully saved to: {source_filepath}")
        
    return datapath

#=========================================================
# 3. EXECUTION WORKFLOW
# =========================================================
if __name__ == "__main__":
    
    # 1. Build Grid
    coords = build_observation_grid(REGION, SPACING, OBS_HEIGHT)
    easting, northing, _ = coords
    
    # 2. Compute Fields
    gz, tfa = generate_synthetic_fields(coords, PRISMS, DENSITIES, INTENSITIES, INC, DEC)
    
    # 3. Add Noise
    gz_noisy = ut.add_gaussian_noise(gz, NOISE_PERCENTAGE, SEED) 
    tfa_noisy = ut.add_gaussian_noise(tfa, NOISE_PERCENTAGE, SEED) 
    
    # 4. Export CSVs and Configs (Updated to pass true source parameters)
    output_dir = export_synthetic_data(
        MODEL_NAME, coords, gz_noisy, tfa_noisy, 
        PRISMS, DENSITIES, INTENSITIES, INC, DEC
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