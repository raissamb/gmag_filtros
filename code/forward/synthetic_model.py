"""
Criaçao de modelo sintético
Fonte: prisma
Unidades:
 - Sistema de referencia: Cartesiano
 - Distância: m
 - Densidade: kg/m3
 - Intensidade de magnetização: A/m 

"""

# Import packages
import sys
import verde as vd
import harmonica as hm
import numpy as np
from pathlib import Path

# Define paths
CURRENT_FILE_PATH = Path(__file__).resolve()
CODE_ROOT = CURRENT_FILE_PATH.parent.parent

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

PROJECT_HOME = CODE_ROOT.parent
DATA_FOLDER = PROJECT_HOME / "data"
OUTPUT_FOLDER = PROJECT_HOME / "output" 

# import custom modules
import modules.kernel as kn
import modules.utils as ut


# ==================== Start =============
# Observation Grid coordinates
# Verde and Harmonica use ENU direction: X axis is Easting,  Y axis is Northing and Z axis is Upwards, positive up.
# extra_coords: observation height
height = 0

# Define region = (west, east, south, north)
region =(0, 5000, 0, 5000)

# # Option 1: define grid by shape (10x10, 35x35...)
# shape = (40,40)
# coordinates = vd.grid_coordinates(
#     region=region, shape=shape, extra_coords=height
# )


##Option 2: define grid by spacing
spacing = 100
coordinates = vd.grid_coordinates(
        region, spacing=spacing, extra_coords=height)

# Define Prism properties
# prism dimensions: west, east, south, north, bottom, top in Cartesian 
# coordinates and in meters 
modelname = "model1"
prism_dimensions = (2000, 3000, 2000, 3000, -10000, -1000)
density = 2670

# magnetic properties
inc = 90
dec = 0
intensity = 3
cosine_dir = kn.get_direction_cosines_enu(inc, dec)
mag_vector = kn.get_magnetization_vector_enu(intensity, cosine_dir)

# Example for multiple prisms
# prisms = [
#     [2e3, 3e3, 2e3, 3e3, -10e3, -1e3],
#     [3e3, 4e3, 7e3, 8e3, -9e3, -1e3],
#     [7e3, 8e3, 1e3, 2e3, -7e3, -1e3],
#     [8e3, 9e3, 6e3, 7e3, -8e3, -1e3],
# ]
# densities = [2670, 3300, 2900, 2980]


# Calculate gz
gz = hm.prism_gravity(coordinates, prism_dimensions, density, field="g_z")

# Calculate mag field
b_e, b_n, b_u = hm.prism_magnetic(
        coordinates, prism_dimensions, mag_vector, field="b")
tfa = hm.total_field_anomaly([b_e, b_n, b_u], inc, dec)


# Add noise
seed = 32
noise_percentage = 3
gz_noisy = ut.add_gaussian_noise(gz, noise_percentage, seed) 
tfa_noisy = ut.add_gaussian_noise(tfa, noise_percentage, seed) 


# Save data as file
easting, northing, upwards = coordinates
header_text = "X,Y,Z,gz_exact,gz_noisy,tfa_exact,tfa_noisy"
data = np.column_stack((easting.ravel(), northing.ravel(), upwards.ravel(), 
                        gz.ravel(), gz_noisy.ravel(), 
                        tfa.ravel(), tfa_noisy.ravel(),  
                        ))


np.savetxt(DATA_FOLDER / f"{modelname}_data.csv",
    data,
    delimiter=",",
    header=header_text,
    comments="",  # Prevents NumPy from adding a default '#' character before the header
    fmt="%.6f",  # Adjust format if needed (e.g., '%d' for integers, '%.6f' for floats)
)


# Plot data
pathname = Path(OUTPUT_FOLDER/ modelname)
pathname.mkdir(exist_ok=True, parents=True)

ut.plot_field_contourf(easting, 
                        northing,
                        tfa_noisy,
                        "TFA",
                        "Easting (m)",
                        "Northing (m)",
                        "nT",
                        Path(pathname / f"{modelname}_tfa_noisy.png"),
                        show=False) # Explicitly name the 'show' parameter

ut.plot_field_contourf(easting, 
                        northing,
                        gz_noisy,
                        "gz",
                        "Easting (m)",
                        "Northing (m)",
                        "mGal",
                        Path(pathname / f"{modelname}_gz_noisy.png"),
                        show=False) # Explicitly name the 'show' parameter