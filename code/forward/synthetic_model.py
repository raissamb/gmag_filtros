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
import modules.constants as mc
import modules.utils as ut


# ==================== Start =============
# Observation Grid coordinates
# Verde and Harmonica use ENU direction: X axis is Easting,  Y axis is Northing and Z axis is Upwards, positive up.
# extra_coords: observation height
height = 0

# Define region = (west, east, south, north)
region =(0, 10e3, 0, 10e3)

# Option 1: define grid by shape (10x10, 35x35...)
shape = (40,40)
coordinates = vd.grid_coordinates(
    region=region, shape=shape, extra_coords=height
)

# Option 2: define grid by spacing
# spacing = 100
# coordinates = vd.grid_coordinates(
#         region, spacing=spacing, extra_coords=height)


# Define prism dimensions: west, east, south, north, bottom, top in Cartesian coordinates and in meters and physical properties
modelname = "model1"
prism = (2e3, 3e3, 2e3, 3e3, -10e3, -1e3)
density = 2670

# Example for multiple prisms
# prisms = [
#     [2e3, 3e3, 2e3, 3e3, -10e3, -1e3],
#     [3e3, 4e3, 7e3, 8e3, -9e3, -1e3],
#     [7e3, 8e3, 1e3, 2e3, -7e3, -1e3],
#     [8e3, 9e3, 6e3, 7e3, -8e3, -1e3],
# ]
# densities = [2670, 3300, 2900, 2980]


# Calculate gz
g_z = hm.prism_gravity(coordinates, prism, density, field="g_z")

# Save data as file 
#coords_flat = np.column_stack([c.ravel() for c in coordinates])
#gz_flat = g_z.ravel()

# Add noise
seeds = mc.seeds
noise_percentage = mc.noise_percentage

field_data = []
for seed in seeds:
    noisy_field = ut.add_gaussian_noise(g_z, noise_percentage, seed)
    field_data.append(noisy_field)


# Save data
# Unpack the tuple to get individual arrays
easting, northing, upwards = coordinates

# Save to CSV
for i in range(5):
    field = field_data[i]
    
    data = np.column_stack((easting.ravel(), northing.ravel(), upwards.ravel(), field.ravel()))

    header_text = f"X,Y,Z,gz_{i}"
    np.savetxt(DATA_FOLDER / f"{modelname}_seed_{i}.csv",
        data,
        delimiter=",",
        header=header_text,
        comments="",  # Prevents NumPy from adding a default '#' character before the header
        fmt="%.6f",  # Adjust format if needed (e.g., '%d' for integers, '%.6f' for floats)
    )



# Plot data
pathname = Path(OUTPUT_FOLDER/ modelname)
pathname.mkdir(exist_ok=True, parents=True)
ut.plot_field_contourf(coordinates[0], 
                        coordinates[1],
                        field_data[0],
                        "gz_1",
                        "Easting (m)",
                        "Northing (m)",
                        "mGal",
                        Path(pathname / f"{modelname}_gz_1.png"),
                        False)

