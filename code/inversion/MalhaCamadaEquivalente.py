"""
Module containing functions for building equivalent-layer grids.
"""

# Import packages
import numpy as np
import matplotlib.pyplot as plt

# Functions
def build_equivalent_layer_grid(nprism_x: int,
                                                        nprism_y: int,
                                                        xmin: float,
                                                        xmax: float,
                                                        ymin: float,
                                                        ymax: float,
                                                        z_top: float,
                                                        z_bottom: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a regular grid of rectangular prisms representing an equivalent layer.

    Parameters
    ----------
    nprism_x : int
        Number of prisms in the equivalent layer along the x-direction.
    nprism_y : int
        Number of prisms in the equivalent layer along the y-direction.
    xmin : float
        Minimum x-coordinate of the equivalent layer grid.
    xmax : float
        Maximum x-coordinate of the equivalent layer grid.
    ymin : float
        Minimum y-coordinate of the equivalent layer grid.
    ymax : float
        Maximum y-coordinate of the equivalent layer grid.
    z_top : float
        Z-coordinate of the top surface of the equivalent layer.
    z_bottom : float
        Z-coordinate of the bottom surface of the equivalent layer.

    Returns
    -------
    vertices : np.ndarray
        Array with shape (nvertices, 3) containing the x, y and z
        coordinates of the grid vertices.
    prism_corner_ids : np.ndarray
        Array with shape (nprisms, 2) containing the indices of two
        diagonally opposite vertices defining each prism.

    """

    # Check input parameters
    if nprism_x <= 0:
        raise ValueError("nprism_x must be greater than zero.")

    if nprism_y <= 0:
        raise ValueError("nprism_y must be greater than zero.")

    if xmax <= xmin:
        raise ValueError("xmax must be greater than xmin.")

    if ymax <= ymin:
        raise ValueError("ymax must be greater than ymin.")

    # Grid spacing
    dx = (xmax - xmin) / nprism_x
    dy = (ymax - ymin) / nprism_y

    # Number of vertices and prisms
    nvertices_surface = (nprism_x + 1) * (nprism_y + 1)
    nvertices = 2 * nvertices_surface
    nprisms = nprism_x * nprism_y

    # Allocate arrays
    vertices = np.empty((nvertices, 3), dtype=float)
    prism_corner_ids = np.empty((nprisms, 2), dtype=int)

    # Build grid vertices
    k = 0

    for j in range(nprism_y + 1):
        y = ymin + j * dy
        for i in range(nprism_x + 1):
            x = xmin + i * dx
            # Top surface
            vertices[k, 0] = x
            vertices[k, 1] = y
            vertices[k, 2] = z_top
            # Bottom surface
            vertices[k + nvertices_surface, 0] = x
            vertices[k + nvertices_surface, 1] = y
            vertices[k + nvertices_surface, 2] = z_bottom
            k += 1

    # Build prism connectivity
    k = 0

    for j in range(nprism_y):
        for i in range(nprism_x):
            # Top vertex
            vertex_top = (nprism_x + 1) * j + i
            # Diagonally opposite bottom vertex
            vertex_bottom = (nvertices_surface  + (nprism_x + 1) * (j + 1) + i  + 1)
            prism_corner_ids[k, 0] = vertex_top
            prism_corner_ids[k, 1] = vertex_bottom
            k += 1

    return vertices, prism_corner_ids



def plot_equivalent_layer_grid(vertices: np.ndarray,
                               prism_corner_ids: np.ndarray) -> None:
    """
    Plot a discretized equivalent layer composed of rectangular prisms.

    Parameters
    ----------
    vertices : np.ndarray
        Array with shape (nvertices, 3) containing the x, y and z
        coordinates of the grid vertices.
    prism_corner_ids : np.ndarray
        Array with shape (nprisms, 2) containing the indices of two
        diagonally opposite vertices defining each prism.

    Returns
    -------
    None.

    """

    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot each prism
    for prism in prism_corner_ids:

        # Get diagonally opposite vertices
        vertex_1 = vertices[prism[0]]
        vertex_2 = vertices[prism[1]]

        # Prism limits
        xmin = min(vertex_1[0], vertex_2[0])
        xmax = max(vertex_1[0], vertex_2[0])

        ymin = min(vertex_1[1], vertex_2[1])
        ymax = max(vertex_1[1], vertex_2[1])

        zmin = min(vertex_1[2], vertex_2[2])
        zmax = max(vertex_1[2], vertex_2[2])

        # Prism vertices
        prism_vertices = np.array([
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ])

        # Prism edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        # Plot edges
        for edge in edges:
            p1 = prism_vertices[edge[0]]
            p2 = prism_vertices[edge[1]]

            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                "k-",
                linewidth=0.8,
            )

    # Axis labels
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.set_zlabel("Upward")

    # Equal aspect ratio
    ax.set_box_aspect((
        np.ptp(vertices[:, 0]),
        np.ptp(vertices[:, 1]),
        np.ptp(vertices[:, 2]),
    ))

    plt.show()



# Main

def main() -> None:
    """Build and display an example equivalent-layer grid."""

    # Number of prisms
    nprism_x = 3
    nprism_y = 4

    # Grid limits
    xmin = -3.0
    xmax = 3.0

    ymin = -4.0
    ymax = 4.0

    # Equivalent-layer depth
    z_top = 4.0
    z_bottom = 5.0

    # Build equivalent-layer grid
    vertices, prism_corner_ids = build_equivalent_layer_grid(nprism_x, nprism_y, xmin, xmax, ymin, ymax, z_top,  z_bottom)
    plot_equivalent_layer_grid(vertices, prism_corner_ids)
    


if __name__ == "__main__":
    main()