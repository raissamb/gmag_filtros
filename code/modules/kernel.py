"""
Module containing functions for forward calculation and transforming operations.

"""

# Import packages
import verde as vd
import harmonica as hm
import numpy as np
import time
import scipy.sparse as sp
import pandas as pd
import tracemalloc
import time
from scipy.linalg import cho_factor, cho_solve

def _dircos(incl, decl, azim):
    """
    Compute the direction cosines from inclination,
    declination, and azimuth.
    
    Parameters
    ----------
    incl : float
        Inclination in degrees, positive downward from the horizontal.
    
    decl : float
        Declination in degrees, positive east of true north.
    
    azim : float
        Azimuth of the x-axis in degrees, positive east of true north.
    
    Returns
    -------
    a, b, c : float
        Direction cosines.
    """

    d2rad = np.pi / 180.0
    xincl = incl * d2rad
    xdecl = decl * d2rad
    xazim = azim * d2rad

    a = np.cos(xincl) * np.cos(xdecl - xazim)
    b = np.cos(xincl) * np.sin(xdecl - xazim)
    c = np.sin(xincl)

    return a, b, c



def _infinity_z_box(x0, y0, z0, x1, y1, z1, x2, y2, mi, md, fi, fd, theta):
    """
    Compute the total-field magnetic anomaly produced by
    a rectangular prism that extends infinitely downward.

    MODIFICADA (REMOVEU m) PARA USO COM DMBOX PARA CALCULO DA JACOBIANA.
    
    Parameters
    ----------
    x0, y0, z0 : float
        Coordinates of the observation point.
    
    x1, x2 : float
        Prism limits along the x-direction.
    
    y1, y2 : float
        Prism limits along the y-direction.
    
    z1 : float
        Coordinate of the top of the prism.
    
    mi, md : float
        Inclination and declination of the magnetization, in degrees.
    
    fi, fd : float
        Inclination and declination of the ambient magnetic field,
        in degrees.
    
    theta : float
        Azimuth of the x-axis, in degrees.
    
    Returns
    -------
    t : float
        Total-field magnetic anomaly in nT.
    """

    cm = 1.0e-7
    t2nt = 1.0e9

    # Cossenos diretores da magnetização
    ma, mb, mc = _dircos(mi, md, theta)

    # Cossenos diretores do campo magnético ambiente
    fa, fb, fc = _dircos(fi, fd, theta)

    # Combinações dos cossenos diretores
    fm1 = ma * fb + mb * fa
    fm2 = ma * fc + mc * fa
    fm3 = mb * fc + mc * fb

    fm4 = ma * fa
    fm5 = mb * fb
    fm6 = mc * fc

    # Distâncias horizontais entre os limites do prisma
    # e o ponto de observação
    alpha = np.array([
        x1 - x0,
        x2 - x0
    ], dtype=float)

    beta = np.array([
        y1 - y0,
        y2 - y0
    ], dtype=float)

    # Distância vertical
    h = z1 - z0

    hsq = h**2

    t = 0.0

    for i in range(2):

        alphasq = alpha[i]**2

        for j in range(2):

            if i == j:
                sign = 1.0
            else:
                sign = -1.0

            r0sq = alphasq + beta[j]**2 + hsq

            r0 = np.sqrt(r0sq)

            r0h = r0 * h

            alphabeta = alpha[i] * beta[j]

            arg1 = (r0 - alpha[i]) / (r0 + alpha[i])

            arg2 = (r0 - beta[j]) / (r0 + beta[j])

            arg3 = alphasq + r0h + hsq

            arg4 = r0sq + r0h - alphasq

            # Termos logarítmicos
            tlog = (
                0.5 * fm3 * np.log(arg1)
                + 0.5 * fm2 * np.log(arg2)
                - fm1 * np.log(r0 + h)
            )

            # Termos arco-tangente
            tatan = (
                -fm4 * np.arctan2(alphabeta, arg3)
                -fm5 * np.arctan2(alphabeta, arg4)
                +fm6 * np.arctan2(alphabeta, r0h)
            )

            t += sign * (tlog + tatan)

    # Conversão para nT
    t *= cm * t2nt

    return t

def mbox(x0, y0, z0, x1, y1, z1, x2, y2, z2, mi, md, fi, fd, m, theta):
    """
    Compute the total-field magnetic anomaly produced by a finite
    rectangular prism.

    The prism is bounded by x1 and x2 along the x-direction, by y1
    and y2 along the y-direction, and by z_top and z_bottom along the
    vertical direction. The z-axis is assumed positive downward.
    
    The total-field magnetic anomaly of a finite rectangular prism is
    computed by calling _infinity_z_box twice. The first call evaluates
    the anomaly produced by a semi-infinite prism extending downward
    from the top of the finite prism, while the second call evaluates
    the anomaly produced by a semi-infinite prism extending downward
    from its bottom. The response of the finite prism is then obtained
    by subtracting:
        t = t_top - t_bottom
    where 
        t_top       = _infinity_z_box(..., z_top=z1, ...) and
        t_bottom = _infinity_z_box(..., z_bottom=z2, ...).

    Parameters
    ----------
    x0, y0, z0 : float
        Coordinates of the observation point.

    x1, x2 : float
        Minimum and maximum x-coordinates of the prism.

    y1, y2 : float
        Minimum and maximum y-coordinates of the prism.

    z_top : float
        Depth of the top surface of the prism.

    z_bottom : float
        Depth of the bottom surface of the prism.

    mi : float
        Inclination of the prism magnetization, in degrees.

    md : float
        Declination of the prism magnetization, in degrees.

    fi : float
        Inclination of the ambient magnetic field, in degrees.

    fd : float
        Declination of the ambient magnetic field, in degrees.

    m : float
        Magnetization intensity, in A/m.

    theta : float
        Azimuth of the x-axis, in degrees.

    Returns
    -------
    t : float
        Total-field magnetic anomaly produced by the finite prism,
        in nT.
    """
    
    if z2 <= z1:
        raise ValueError(
            "z_bottom must be greater than z_top "
            "because the z-axis is positive downward." )
    
    t_top      =_infinity_z_box(x0, y0, z0, x1, y1, z1, x2, y2, mi, md, fi, fd, theta)
    t_bottom=_infinity_z_box(x0, y0, z0, x1, y1, z2, x2, y2, mi, md, fi, fd, theta)
    return m*(t_top-t_bottom)

def dmbox(x0, y0, z0, x1, y1, z1, x2, y2, z2, mi, md, fi, fd, theta):
    """
    Compute the total-field magnetic anomaly produced by a finite
    rectangular prism.

    USAR PARA CALCULAR JACOBIANA.

    The prism is bounded by x1 and x2 along the x-direction, by y1
    and y2 along the y-direction, and by z_top and z_bottom along the
    vertical direction. The z-axis is assumed positive downward.
    
    The total-field magnetic anomaly of a finite rectangular prism is
    computed by calling _infinity_z_box twice. The first call evaluates
    the anomaly produced by a semi-infinite prism extending downward
    from the top of the finite prism, while the second call evaluates
    the anomaly produced by a semi-infinite prism extending downward
    from its bottom. The response of the finite prism is then obtained
    by subtracting:
        t = t_top - t_bottom
    where 
        t_top       = _infinity_z_box(..., z_top=z1, ...) and
        t_bottom = _infinity_z_box(..., z_bottom=z2, ...).

    Parameters
    ----------
    x0, y0, z0 : float
        Coordinates of the observation point.

    x1, x2 : float
        Minimum and maximum x-coordinates of the prism.

    y1, y2 : float
        Minimum and maximum y-coordinates of the prism.

    z_top : float
        Depth of the top surface of the prism.

    z_bottom : float
        Depth of the bottom surface of the prism.

    mi : float
        Inclination of the prism magnetization, in degrees.

    md : float
        Declination of the prism magnetization, in degrees.

    fi : float
        Inclination of the ambient magnetic field, in degrees.

    fd : float
        Declination of the ambient magnetic field, in degrees.

    theta : float
        Azimuth of the x-axis, in degrees.

    Returns
    -------
    t : float
        Total-field magnetic anomaly produced by the finite prism,
        in nT.
    """
    
    if z2 <= z1:
        raise ValueError(
            "z_bottom must be greater than z_top "
            "because the z-axis is positive downward." )
    
    t_top      =_infinity_z_box(x0, y0, z0, x1, y1, z1, x2, y2, mi, md, fi, fd, theta)
    t_bottom=_infinity_z_box(x0, y0, z0, x1, y1, z2, x2, y2, mi, md, fi, fd, theta)
    return (t_top-t_bottom)


def build_prism_layer(N:int, 
                      dx:float, 
                      dy:float, 
                      easting:np.ndarray, 
                      northing:np.ndarray, 
                      z_bottom: float, 
                      z_top:float):

    # Construct the prism boundaries directly below each observation point
    prisms = np.zeros((N, 6))
    prisms[:, 0] = easting - dx / 2  # West
    prisms[:, 1] = easting + dx / 2  # East
    prisms[:, 2] = northing - dy / 2 # South
    prisms[:, 3] = northing + dy / 2 # North
    prisms[:, 4] = z_bottom          # Bottom
    prisms[:, 5] = z_top             # Top
    
    return prisms


def build_jacobian_with_profiling(N: int, 
                                  easting: np.ndarray, 
                                  northing: np.ndarray, 
                                  upwards:np.ndarray, 
                                  prisms_layer: np.ndarray, 
                                  inc:float, 
                                  dec: float):

    # # ---------------------------------------------------------
    # Assemble Jacobian Matrix G and Track Computational Cost
    # ---------------------------------------------------------
    tracemalloc.start()
    t0 = time.perf_counter()
    
    # Allocate the dense N x N Jacobian matrix
    G = np.empty((N, N), dtype=np.float64)
    obs_coords = (easting, northing, upwards)
    
    # Define unit magnetization (1 A/m) aligned with regional induction
    mag_e, mag_n, mag_u = hm.magnetic_angles_to_vec(1.0, inc, dec)
    unit_mag = (np.array([mag_e]), np.array([mag_n]), np.array([mag_u]))
    
    # Assemble G column-by-column:
    # Column j contains the TFA produced by the j-th prism at all observation points
    for j in range(N):
        # Compute the analytical field of the single j-th prism
        b_e, b_n, b_u = hm.prism_magnetic(
            obs_coords, 
            [prisms_layer[j]], 
            magnetization=unit_mag, 
            field="b"
        )
        # Project B-components to Total Field Anomaly (TFA)
        G[:, j] = hm.total_field_anomaly([b_e, b_n, b_u], inc, dec)
    
    t1 = time.perf_counter()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    assembly_time = t1 - t0
    peak_ram_mb = peak_mem / (1024 ** 2)
    
    print(f"Jacobian G shape: {G.shape}")
    print(f"G Assembly Time: {assembly_time:.2f} s")
    print(f"Peak Memory Allocated: {peak_ram_mb:.2f} MB")
    
    return G, assembly_time, peak_ram_mb


def find_all_connections(grid):
    """
    Loops through a grid and finds all unique, canonical neighbor pairs.

    A canonical pair is [smaller_value, larger_value, direction].
    This function depends on get_neighbors_with_directions().

    Args:
        grid (list of lists): The input grid of values.

    Returns:
        list: A list of all connection pairs, e.g., [[1, 2, 'LX'], [1, 4, 'LY'], ...]
    """
    all_connections = []
    n_rows, n_cols = len(grid), len(grid[0])

    for r in range(n_rows):
        for c in range(n_cols):
            # Get the VALUE of the source cell
            source_value = grid[r][c]

            # Get its neighbors (coordinates and directions)
            neighbor_coords, neighbor_dirs = get_neighbors_with_directions(
                grid, r, c)

            # Create a connection for each neighbor
            for (neighbor_r, neighbor_c), direction in zip(neighbor_coords, neighbor_dirs):

                # Get the VALUE of the neighbor cell
                neighbor_value = grid[neighbor_r][neighbor_c]

                # Create the canonical connection
                if source_value < neighbor_value:
                    connection = [source_value, neighbor_value, direction]
                else:
                    connection = [neighbor_value, source_value, direction]

                all_connections.append(connection)

    # We can even put the de-duplication step right in here!
    final_connections = remove_duplicate_sublists(all_connections)

    return final_connections



def get_neighbors_with_directions(grid, row, col, include_diagonals=False):
    """
    Finds the valid neighbors of a cell (row, col) in a grid and their directions.

    Args:
        grid (list of lists): The 2D grid.
        row (int): The row index of the cell.
        col (int): The column index of the cell.
        include_diagonals (bool): If True, includes diagonal neighbors.

    Returns:
        tuple: A tuple containing two lists:
            neighbors (list): A list of (row, col) tuples for each valid neighbor.
            neighbor_dirs (list): A list of strings (e.g., 'LX', 'LY') 
                                  corresponding to each neighbor.
    """
    rows, cols = len(grid), len(grid[0])
    neighbors = []
    neighbor_dirs = []

    # Define the directional offsets AND their string names
    # Suggestion: Consider more descriptive names like 'vertical' and 'horizontal'
    cardinal_dirs = [
        ((-1, 0), 'LY'),
        ((1, 0), 'LY'),
        ((0, -1), 'LX'),
        ((0, 1), 'LX')
    ]

    diagonal_dirs = [
        ((-1, -1), 'up-left'),
        ((-1, 1), 'up-right'),
        ((1, -1), 'down-left'),
        ((1, 1), 'down-right')
    ]

    directions = cardinal_dirs
    if include_diagonals:
        directions += diagonal_dirs

    for (d_row, d_col), direction_name in directions:
        new_row, new_col = row + d_row, col + d_col

        if 0 <= new_row < rows and 0 <= new_col < cols:
            neighbors.append((new_row, new_col))
            neighbor_dirs.append(direction_name)

    return neighbors, neighbor_dirs



def remove_duplicate_sublists(list_of_lists):
    """
    Removes duplicate sublists from a list of lists.
    This function is efficient and correct.
    """
    unique_sublists = []
    seen_tuples = set()

    for sublist in list_of_lists:
        # Convert to tuple to make it hashable for the set
        sublist_as_tuple = tuple(sublist)

        if sublist_as_tuple not in seen_tuples:
            seen_tuples.add(sublist_as_tuple)
            unique_sublists.append(sublist)

    return unique_sublists





def build_rtr_with_profiling(df_sub: pd.DataFrame, N: int, kn_module) -> tuple:
    """
    Builds the First-Order Tikhonov regularization matrix (RTR) 
    using neighbor connections, tracking time and peak memory.
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    
    # 1. Sort to guarantee a strict spatial grid order (Row-major: Y then X)
    df_sorted = df_sub.sort_values(by=["Y", "X"]).reset_index(drop=True)
    nx = len(np.unique(df_sorted["X"].to_numpy()))
    ny = len(np.unique(df_sorted["Y"].to_numpy()))
    
    # 2. Map the 1D tags to the 2D grid
    tags_grid = np.arange(N).reshape((ny, nx)).tolist()
    connections = kn_module.find_all_connections(tags_grid)
    
    # 3. Assemble the sparse Graph Laplacian
    RTR = sp.lil_matrix((N, N), dtype=np.float64)
    for src, nbr, direction in connections:
        RTR[src, nbr] = -1.0
        RTR[nbr, src] = -1.0
        RTR[src, src] += 1.0
        RTR[nbr, nbr] += 1.0
        
    RTR = RTR.tocsr()
    
    t1 = time.perf_counter()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    assembly_time = t1 - t0
    peak_ram_mb = peak_mem / (1024 ** 2)
    
    print(f"RTR Assembly Time: {assembly_time:.4f} s")
    print(f"RTR Peak Memory: {peak_ram_mb:.2f} MB")
    
    return RTR, assembly_time, peak_ram_mb


def solve_system_cholesky(G: np.ndarray, RTR: sp.csr_matrix, d_obs: np.ndarray, mu: float) -> tuple:
    """
    Solves the regularized normal equations using Cholesky decomposition,
    tracking time and peak memory.
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    
    # 1. Set up the Normal Equations: A * p = b
    # G.T @ G creates a dense matrix. We convert RTR to dense to sum them.
    A = (G.T @ G) + (mu * RTR.toarray())
    b = G.T @ d_obs
    
    # 2. Perform Cholesky Factorization and Solve
    # cho_factor returns the lower/upper triangular matrix, cho_solve computes the final vector
    c, lower = cho_factor(A, lower=True)
    p_tilde = cho_solve((c, lower), b)
    
    t1 = time.perf_counter()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    solve_time = t1 - t0
    peak_ram_mb = peak_mem / (1024 ** 2)
    
    print(f"Cholesky Solver Time: {solve_time:.2f} s")
    print(f"Solver Peak Memory: {peak_ram_mb:.2f} MB")
    
    return p_tilde, solve_time, peak_ram_mb



def run_tikhonov_l_curve_scan_ces(mat_g: np.ndarray, d: np.ndarray, mat_rtr, 
                                 reg_min: float, reg_max: float, steps: int) -> tuple[pd.DataFrame, dict, float, float]:
    """
    Sweeps a logarithmic range of regularization values to find the best parameter 
    via a fast vectorized L-curve maximum curvature sweep, tracking time and peak memory.
    """
    tracemalloc.start()
    t0 = time.perf_counter()

    reg_range = np.logspace(np.log10(reg_min), np.log10(reg_max), int(steps))
    
    # Pre-calculate normal equations products once
    mat_gtg = mat_g.T @ mat_g
    mat_gtd = mat_g.T @ d
    mat_rtr_dense = mat_rtr.toarray() if hasattr(mat_rtr, 'toarray') else mat_rtr
    
    records = []
    print(f"📐 Scanning CES L-curve across {len(reg_range)} regularization steps...")
    
    for mu in reg_range:
        mat_a = mat_gtg + (mu * mat_rtr_dense)
    
        try:
            c, lower = cho_factor(mat_a, lower=True)
            p_est = cho_solve((c, lower), mat_gtd)
        except np.linalg.LinAlgError:
            p_est = np.linalg.solve(mat_a, mat_gtd)
            
        pred_d = mat_g @ p_est
        residual_norm = np.linalg.norm(pred_d - d)
        model_norm = np.sqrt(float(p_est.T @ mat_rtr @ p_est))
        rmse = np.sqrt(np.mean((pred_d - d) ** 2))
        
        records.append({
            "reg_value": mu,
            "residual_norm": float(residual_norm),
            "model_norm": float(model_norm),
            "rmse": float(rmse),
            "p_est": p_est
        })
        
    df_metrics = pd.DataFrame(records)
    
    # Vectorized geometric maximum curvature extraction
    x = np.log10(df_metrics["residual_norm"].to_numpy())
    y = np.log10(df_metrics["model_norm"].to_numpy())
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    
    coords = np.column_stack((x, y))
    vec_line = p2 - p1
    vec_points = p1 - coords
    
    cross_2d = (vec_line[0] * vec_points[:, 1]) - (vec_line[1] * vec_points[:, 0])
    distances = np.abs(cross_2d) / np.linalg.norm(vec_line)

    best_idx = np.argmax(distances)
    best_run = df_metrics.iloc[best_idx].to_dict()
    
    t1 = time.perf_counter()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    sweep_time = t1 - t0
    peak_ram_mb = peak_mem / (1024 ** 2)
    
    print(f"L-Curve Sweep Time: {sweep_time:.2f} s")
    print(f"L-Curve Peak Memory: {peak_ram_mb:.2f} MB")
    
    return df_metrics, best_run, sweep_time, peak_ram_mb



def get_direction_cosines_enu(inc: float, dec: float) -> np.ndarray:
    """
    Calculates direction cosines for an East-North-Up (ENU) coordinate system.

    Parameters
    ----------
    inc : float
        Inclination in degrees (positive DOWN towards the Earth's center).
    dec : float
        Declination in degrees (angle measured clockwise from true NORTH).

    Returns
    -------
    np.ndarray
        A 1D array of shape (3,) containing the unit vector components: 
        [u_east, u_north, u_up].
        
    Harmonica ENU orientation:
        Harmonica X is Easting, East component = cos(I) * sin(D)
        Harmonica Y is Northing, North component = cos(I) * cos(D)
        Harmonica Z is Up, Vertical component = -sin(I) (because I is positive down)
    """
    rinc = np.deg2rad(inc)
    rdec = np.deg2rad(dec)

    # Harmonica ENU system mapping alignment
    ux = np.cos(rinc) * np.sin(rdec)  # Easting component
    uy = np.cos(rinc) * np.cos(rdec)  # Northing component
    uz = -np.sin(rinc)                # Upward component (negative due to positive down inc)

    return np.array([ux, uy, uz])


def get_magnetization_vector_enu(intensity: float, cosine_dir: np.ndarray):
    # Conversion to Cartesian (ENU)
    # Z is Up, Inc is positive Down -> Mz = -sin(I)
    mag_e = intensity * cosine_dir[0]
    mag_n = intensity * cosine_dir[1]
    mag_u = -intensity * cosine_dir[2]
    magnetization = (mag_e, mag_n, mag_u)

    return magnetization
