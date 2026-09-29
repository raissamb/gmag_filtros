# Import packages
import sys
import verde as vd
import harmonica as hm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

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
import modules.kernel as kn


# ============================================================
# Observation grid
# ============================================================
lim_min = 0.0
lim_max = 64.0
spacing = 1.0

east  = np.arange(lim_min, lim_max + spacing, spacing)
north= np.arange(lim_min, lim_max + spacing, spacing)
E, N = np.meshgrid(east, north)


# Observation elevation
z0 = 0.0


# ============================================================
# Prism geometry
# ============================================================

# Prism dimensions:
# width  = 20 m
# length = 20 m
# center = (30, 30) m

x1 = 20.0
x2 = 40.0
y1 = 20.0
y2 = 40.0
# Top and bottom depths
z1 = 1.0
z2 = 3.0

# ============================================================
# Magnetic parameters
# ============================================================

# Geomagnetic field
fi = 5.0       # inclination [degrees]
fd = 10.0    # declination [degrees]

# Induced magnetization
# Since there is no remanent magnetization, the direction
# of the magnetization is the same as the geomagnetic field.

mi = fi
md = fd

# Magnetization intensity [A/m]
m = 0.35

# Azimuth of the x-axis.
# The original MBOX formulation uses x pointing toward North
# when theta = 0 deg. Therefore, inside MBOX we use:
#
#     x -> North
#     y -> East
#
# The final map is plotted as East versus North.
theta = 0.0


# ============================================================
# Compute magnetic anomaly
# ============================================================

magnetic_anomaly = np.zeros_like(E, dtype=float)

for i in range(len(north)):
    for j in range(len(east)):
        x0 = N[i, j]
        y0 = E[i, j]
        magnetic_anomaly[i, j] = kn.mbox(x0, y0, z0, x1, y1, z1, x2, y2, z2, mi, md, fi, fd, m, theta)



# Save data

# Add noise
seed = 38
noise_percentage = 2
noisy_field = ut.add_gaussian_noise(magnetic_anomaly, noise_percentage, seed)


easting, northing, upwards = np.meshgrid(east, north, z0)
data = np.column_stack( (easting.ravel(), northing.ravel(), upwards.ravel(), magnetic_anomaly.ravel(), noisy_field.ravel() ))

header_text = "X,Y,Z,TFA_exact,TFA_field"
np.savetxt(DATA_FOLDER / "test_model.csv",
    data,
    delimiter=",",
    header=header_text,
    comments="",  # Prevents NumPy from adding a default '#' character before the header
    fmt="%.6f",  # Adjust format if needed (e.g., '%d' for integers, '%.6f' for floats)
)


# ============================================================
# 2-D map
# ============================================================

plt.figure(figsize=(7, 6))
#contour = plt.contourf(E, N, magnetic_anomaly, levels=30, cmap="jet")
contour = plt.contourf(E, N, noisy_field, levels=30, cmap="jet")
plt.colorbar(contour, label="Magnetic anomaly (nT)")

# Prism outline
plt.plot([20, 40, 40, 20, 20], [20, 20, 40, 40, 20],"k-", linewidth=1.5)
plt.xlabel("E (m)")
plt.ylabel("N (m)")
plt.xlim(0, 64)
plt.ylim(0, 64)
plt.gca().set_aspect("equal")
plt.title("Magnetic Anomaly")
plt.tight_layout()
plt.show()