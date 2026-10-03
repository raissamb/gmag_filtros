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
import modules.utils as ut


# ==================== Start =============
# Observation Grid coordinates
# Verde and Harmonica use ENU direction: X axis is Easting,  Y axis is Northing and Z axis is Upwards, positive up.
# extra_coords: observation height
height = 0

# Define region = (west, east, south, north)
region =(0, 64000, 0, 64000)

# Option 1: define grid by shape (10x10, 35x35...)
#shape = (40,40)
#coordinates = vd.grid_coordinates(
#    region=region, shape=shape, extra_coords=height
#)

# Option 2: define grid by spacing
spacing = 100
coordinates = vd.grid_coordinates(
         region, spacing=spacing, extra_coords=height)


# Define prism dimensions: west, east, south, north, bottom, top in Cartesian coordinates and in meters and physical properties
modelname = "model_wang2015"

# Example for multiple prisms
prisms = [
    [20000, 36000, 40000, 44000, -6000, -2000],   # A1
    [27000, 39000, 28000, 32000, -12000, -8000],  # A2
    [31000, 47000, 16000, 20000, -16000, -12000], # A3 (Shifted East)
]
densities = [2670, 3300, 2900]

# angles, induced magnetization
inc = 90
dec = 0
F = 50000

# Regional field to Teslas then to H field in A/m
H = (F * 10**(-9)) / (4 * np.pi * 10**(-7))

# Magnetic suscp. k
#ka1 = 0.1
# Lowered from 0.1 to match the ~280 nT peak shown in the paper's Profile P4
ka1 = 0.035
ka2 = 0.15
ka3 = 0.2
k = np.array([ka1, ka2, ka3])

# Magnetization: M = k * H
ma1 = ka1 * H
ma2 = ka2 * H
ma3 = ka3 * H
mag = np.array([ma1, ma2, ma3])

# # Magnetization vector: m = mx + my + mz
mx = []
my = []
mz = []
for m in mag:
    mag_e, mag_n, mag_u = hm.magnetic_angles_to_vec(m, inc, dec)
    mx.append(mag_e)
    my.append(mag_n)
    mz.append(mag_u)

magnetization_easting = np.array(mx)
magnetization_northing = np.array(my)
magnetization_upward = np.array(mz)
magnetization = (
   magnetization_easting, magnetization_northing, magnetization_upward
)

# Calculate gz
gz = hm.prism_gravity(coordinates, prisms, densities, field="g_z")

#Calculate mag field and tfa
b_e, b_n, b_u = hm.prism_magnetic(coordinates, prisms, magnetization, field="b")
tfa = hm.total_field_anomaly([b_e, b_n, b_u], inc, dec)

# # Add noise
seed = 43
noise_percentage = 2

# add noise
gz_noisy = ut.add_gaussian_noise(gz, noise_percentage, seed)
tfa_noisy = ut.add_gaussian_noise(tfa, noise_percentage, seed) 

# Save data
# Unpack the tuple to get individual arrays
easting, northing, upwards = coordinates
data_grav = np.column_stack((easting.ravel(), northing.ravel(), upwards.ravel(), gz_noisy.ravel(), gz.ravel() ))
data_mag = np.column_stack((easting.ravel(), northing.ravel(), upwards.ravel(), tfa_noisy.ravel(), tfa.ravel() ))

# Save grav data
header_text = "X,Y,Z,gz,gz_exact"
np.savetxt(DATA_FOLDER / "wang2015_grav.csv",
    data_grav,
    delimiter=",",
    header=header_text,
    comments="",  # Prevents NumPy from adding a default '#' character before the header
    #fmt="%.6f",  # Adjust format if needed (e.g., '%d' for integers, '%.6f' for floats)
)


# save mag data
header_text = "X,Y,Z,tfa,tfa_exact"
np.savetxt(DATA_FOLDER / "wang2015_mag.csv",
    data_mag,
    delimiter=",",
    header=header_text,
    comments="",  # Prevents NumPy from adding a default '#' character before the header
    #fmt="%.6f",  # Adjust format if needed (e.g., '%d' for integers, '%.6f' for floats)
)




# # Plot data
pathname = Path(OUTPUT_FOLDER/ "wang2015")
pathname.mkdir(exist_ok=True, parents=True)
levels = [-40, -10, 20, 50, 80, 110, 140, 170, 200, 230, 260, 290, 320, 360, 380, 410]
ut.plot_field_contourf(coordinates[0], 
                       coordinates[1], 
                       tfa_noisy,
                       "TFA",
                      "Easting (m)",
                      "Northing (m)",
                      "nT",
                      Path(pathname / "wang2015_tfa_noisy.png"),
                      levels,
                      #"nipy_spectral",
                      #"rainbow",
                      "turbo",
                      False)




























