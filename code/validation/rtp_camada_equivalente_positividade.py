"""
RTP via camada equivalente — VERSÃO COM POSITIVIDADE.

Evolução do rtp_camada_equivalente_corrigido.py. Roda as duas soluções sobre o
mesmo dado e compara:

  A) ORIGINAL   — mínimos quadrados sem restrição, mu escolhido pela curva L
                  (exatamente o que o script corrigido fazia: erro 8,99%).
  B) PROPOSTA   — magnetização restrita a valores positivos (p >= 0),
                  mu escolhido pelo princípio da discrepância.

O que mudou em relação ao script corrigido (marcado com  # >>> NOVO):
  1. Sinal do resíduo padronizado: em TODO lugar (terminal, figuras, CSV)
     resíduo = RTP estimado - RTP verdadeiro. Positivo = superestimou.
     No script anterior a figura usava o sinal oposto ao do CSV.
  2. Princípio da discrepância: escolhe o maior mu cujo ajuste aos dados
     (RMS) ainda fica dentro do nível do ruído. A curva L escolhia um mu com
     ajuste de 1,8 nT para um ruído de 5,5 nT, ou seja, ajustava o ruído.
  3. Restrição de positividade: com magnetização induzida e contraste de
     susceptibilidade positivo, a magnetização é paralela ao campo, logo
     positiva. Impor p >= 0 elimina a oscilação de sinal entre prismas
     vizinhos que gerava as listras do resíduo.
     Referência da ideia: Li, Nabighian & Oldenburg (2014), Geophysics 79(6),
     "Using an equivalent source with positivity for low-latitude reduction
     to the pole without striation".
     Ressalva: só vale para magnetização induzida com contraste positivo. Com
     remanência forte ou corpos menos magnéticos que a encaixante, a restrição
     distorce o resultado.

A regularização é a MESMA nas duas soluções (o RTR de primeira ordem da sua
colega). Para a versão com restrição o problema é escrito como
    min || [G ; sqrt(mu) R] p - [d ; 0] ||²   sujeito a  p >= 0,
onde R é a matriz de diferenças entre vizinhos, com R^T R = RTR (o script
confere essa igualdade ao rodar). Resolvido com scipy.optimize.lsq_linear
(método BVLS, mínimos quadrados com limites).

As funções da colega (jacobiana, RTR, curva L) continuam intactas.

Referências do método:
  - Lawson, C. L.; Hanson, R. J. (1974). Solving Least Squares Problems.
    Prentice-Hall. Origem do NNLS (mínimos quadrados não negativos).
  - Stark, P. B.; Parker, R. L. (1995). Bounded-variable least-squares: an
    algorithm and applications. Computational Statistics, 10, 129-141.
    É o algoritmo usado por lsq_linear(method="bvls").
  - Li, Y.; Nabighian, M.; Oldenburg, D. W. (2014). Using an equivalent source
    with positivity for low-latitude reduction to the pole without striation.
    Geophysics, 79(6).
  (Conferir volume e páginas antes de citar.)

Como rodar: abra a pasta no VS Code, selecione o interpretador do .venv e
aperte Ctrl+F5. Leva cerca de 1 minuto (a varredura com restrição resolve
um problema com limites para cada mu).
"""


# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import verde as vd
import harmonica as hm
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import lsq_linear   # >>> NOVO: mínimos quadrados com limites

# Define paths
CURRENT_FILE_PATH = Path(__file__).resolve()
CODE_ROOT = CURRENT_FILE_PATH.parent.parent

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

PROJECT_HOME = CODE_ROOT.parent
OUTPUT_FOLDER = PROJECT_HOME / "output" / "rtp_positivity_validation"
OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)


# =========================================================================
# BLOCO 1 — funções vindas de code/modules/kernel.py (verbatim)
# =========================================================================


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
    # >>> CORRIGIDO (1): era  mag_u = -intensity * cosine_dir[2]
    # cosine_dir vem de get_direction_cosines_enu e JÁ está em ENU, com a
    # componente Up igual a -sin(I). O sinal extra invertia o vetor na vertical.
    # Confere com harmonica.magnetic_angles_to_vec(intensity, inc, dec).
    cosine_dir = np.asarray(cosine_dir, dtype=float)
    mag_e = intensity * cosine_dir[0]
    mag_n = intensity * cosine_dir[1]
    mag_u = intensity * cosine_dir[2]
    magnetization = (mag_e, mag_n, mag_u)

    return magnetization


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



def build_magnetic_jacobian_with_profiling(N: int, 
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


# =========================================================================
# BLOCO 2 — funções vindas de code/modules/utils.py (verbatim)
# =========================================================================


def add_gaussian_noise(field: np.ndarray,
                       noise_percentage: float,
                       seed: int) -> np.ndarray:
    """
    Function to create and add noise to a given field.

    Parameters
    ----------
    field : np.ndarray
        Input field.
    noise_percentage : float
        Noise percentage to setup noise amplitude in relation to maximum value of input field.
    seed : int
        Integer to start seed.

    Returns
    -------
    noisy_field : np.ndarray
        Field with noise.
    noise : np.ndarray
        Noise sequence.
    sigma : float
        Parameter to define noise scale.

    """
    rng = np.random.default_rng(seed)
    max_amplitude = np.max(np.abs(field))
    sigma = (noise_percentage / 100.0) * max_amplitude
    noise = rng.normal(loc=0.0, scale=sigma, size=field.shape)
    noisy_field = field + noise

    return noisy_field



def calculate_rmse(targets, predictions):
    """
    Calculates RMSE using NumPy.

    Args:
        predictions (np.array): Array of predicted values.
        targets (np.array): Array of actual (observed) values.

    Returns:
        float: The Root Mean Square Error.
    """
    return np.sqrt(np.mean((predictions - targets) ** 2))



def compare_eq_layer_results(
    xcoord, ycoord,
    obs, predicted,
    modelname, figname, reg,
    show=False
):
    """
    Plots Target vs. Predicted data and the Residuals with consistent scaling.

    Parameters
    ----------
    xcoord, ycoord : np.ndarray
        Coordinates for the grid (usually meshgrid).
    obs : np.ndarray
        Observed data grid.
    predicted : np.ndarray
        Predicted (Calculated) data grid.
    modelname : str
        Name of the model for the title.
    rmse : float
        Root Mean Square Error value.
    figname : str or Path
        Path to save the figure.
    reg : float or str
        Regularization parameter value.
    show : bool
        Whether to display the plot window.
    """

    # 1. Calculate Residuals
    residual = obs - predicted
    rmse = calculate_rmse(obs, predicted)

    # 2. Determine Global Scales for Obs/Pred
    # This ensures "Yellow" means the same nT value in both plots.
    global_min = min(np.min(obs), np.min(predicted))
    global_max = max(np.max(obs), np.max(predicted))
    # 21 levels for smooth contour
    levels = np.linspace(global_min, global_max, 21)

    # 3. Setup Figure (1 row, 3 columns)
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(16, 5), constrained_layout=True)

    # --- Plot 1: Observed ---
    # We use 'vmin' and 'vmax' to lock the scale
    c1 = ax1.contourf(xcoord, ycoord, obs, levels=levels, cmap='jet',
                      vmin=global_min, vmax=global_max)
    fig.colorbar(c1, ax=ax1, label='nT', orientation='vertical')
    ax1.set_title('Input Data')
    ax1.set_xlabel('Easting (m)')
    ax1.set_ylabel('Northing (m)')

    # --- Plot 2: Predicted ---
    # Must use same levels and cmap as Observed!
    c2 = ax2.contourf(xcoord, ycoord, predicted, levels=levels, cmap='jet',
                      vmin=global_min, vmax=global_max)
    fig.colorbar(c2, ax=ax2, label='nT', orientation='vertical')
    ax2.set_title(f'Predicted \nReg: {reg}')
    ax2.set_xlabel('Easting (m)')
    # Y-label optional here to save space

    # --- Plot 3: Residuals (Obs - Pred) ---
    #
    # Residuals usually center on 0, so we use a DIVERGING colormap (blue-white-red).
    # We create a symmetric scale centered on 0.
    res_abs_max = np.max(np.abs(residual))
    c3 = ax3.contourf(xcoord, ycoord, residual, cmap='seismic',
                      levels=21, vmin=-res_abs_max, vmax=res_abs_max)
    fig.colorbar(c3, ax=ax3, label='Residual (nT)', orientation='vertical')
    ax3.set_title(f'Residuals (Obs - Pred)\nRMSE: {rmse:.4f}')
    ax3.set_xlabel('Easting (m)')

    # Main Title
    fig.suptitle(f'Inversion Results: {
                 modelname}', fontsize=16, fontweight='bold')

    # Save
    fig.savefig(figname, dpi=300, bbox_inches="tight")
    print(f"Comparison plot saved to: {figname}")

    # Memory Management
    if show:
        plt.show()
    else:
        plt.close(fig)


# =========================================================================
# BLOCO 3 — funções novas: positividade, discrepância e figuras   # >>> NOVO
# =========================================================================
#
# VISÃO GERAL DO MÉTODO
# ---------------------
# A camada equivalente procura magnetizações p (uma por prisma, N = 441) tais
# que G p ≈ d, onde G é a jacobiana no equador e d é o dado observado. Como o
# problema é mal condicionado, a colega acrescenta regularização de primeira
# ordem e resolve
#
#     minimizar  ||G p - d||²  +  mu ||R p||²                          (1)
#
# A solução A resolve (1) sem restrição. A solução B resolve (1) com p >= 0.
# A regularização é a mesma nas duas; o que muda é a restrição (e o critério
# de escolha do mu, que o quadro comparativo também separa).
#
# Por que impor p >= 0
#   Física: com magnetização induzida e contraste de susceptibilidade positivo,
#   M = Δχ F / mu0 aponta no mesmo sentido do campo. Cada prisma da camada já é
#   modelado magnetizado na direção do campo (é assim que a jacobiana é
#   montada), então a intensidade p_i só pode ser positiva.
#   Numérica: sem restrição, a camada usa oscilações de sinal entre prismas
#   vizinhos (+20, -20, +20 A/m...) para ajustar o ruído. Em I = 10° os campos
#   de vizinhos com sinais opostos quase se cancelam a distância, então essas
#   oscilações custam pouco no ajuste. Projetadas no polo com G_pole, elas
#   deixam de se cancelar e viram as listras do resíduo. A positividade torna
#   essas oscilações impossíveis: um prisma pode ser zero, mas não negativo
#   para compensar o vizinho. É uma informação a priori que a suavidade sozinha
#   não dá, porque ||R p||² controla o tamanho dos saltos, não o sinal.
#   (Essa explicação numérica é uma interpretação dos resultados, coerente com
#   o mapa de magnetização e com a direção das listras; não é um teorema.)
#
# Limitações
#   - Validade física: só vale para magnetização induzida com contraste
#     positivo. Com remanência significativa, ou fontes menos magnéticas que a
#     encaixante, a magnetização equivalente pode precisar de sinal negativo,
#     e a restrição força um resultado errado.
#   - Custo: o BVLS trabalha com matrizes densas. Com 441 incógnitas leva menos
#     de 1 s por mu; com dezenas de milhares de pontos fica inviável, e é
#     preciso recorrer a métodos iterativos (gradiente projetado, por exemplo).
#   - Escolha do mu: o princípio da discrepância exige conhecer o nível do
#     ruído sigma. No sintético ele é conhecido; em dado real, é estimado.


def build_difference_matrix(connections: list, N: int) -> sp.csr_matrix:
    """
    Matriz de diferenças R entre prismas vizinhos.

    Cada linha corresponde a um par de vizinhos (i, j), com +1 na coluna i e
    -1 na coluna j. No grid de 21 x 21 há 420 pares na direção Leste e 420 na
    direção Norte, então R tem 840 linhas e 441 colunas.

    Multiplicando por p:
        ||R p||² = soma de (p_i - p_j)² sobre todos os pares de vizinhos,
    ou seja, o termo de regularização pune saltos entre prismas vizinhos.
    Quanto maior o mu, mais suave a camada.

    R^T R é o laplaciano do grid: na diagonal fica o número de vizinhos de cada
    prisma (2, 3 ou 4) e fora dela -1 para cada par de vizinhos. É exatamente a
    matriz RTR montada por build_rtr_with_profiling. O bloco principal confere
    essa igualdade ao rodar: se as duas fossem diferentes, as soluções A e B
    usariam regularizações distintas e a comparação não isolaria o efeito da
    positividade.

    Por que precisamos de R e não só de R^T R: a solução com restrição é
    escrita como um sistema aumentado (ver solve_positive_layer), que usa R
    diretamente.
    """
    R = sp.lil_matrix((len(connections), N), dtype=np.float64)
    for k, (src_id, nbr_id, _) in enumerate(connections):
        R[k, src_id] = 1.0
        R[k, nbr_id] = -1.0
    return R.tocsr()


def solve_positive_layer(G: np.ndarray, d: np.ndarray, R_dense: np.ndarray,
                         mu: float) -> np.ndarray:
    """
    Resolve  min ||G p - d||² + mu ||R p||²  sujeito a  p >= 0.

    1) Por que não há fórmula fechada
       Sem restrição, basta derivar e igualar a zero, o que dá as equações
       normais  (G^T G + mu R^T R) p = G^T d,  um sistema linear resolvido de
       uma vez (é o que run_tikhonov_l_curve_scan_ces faz, com Cholesky).
       Com p >= 0 isso deixa de valer: o mínimo pode cair na fronteira, com
       alguns p_i = 0, e nesse ponto a derivada não precisa ser zero. A solução
       passa a depender de QUAIS variáveis ficam travadas em zero, e isso não
       se sabe de antemão.

    2) O sistema aumentado
       Empilhando as matrizes,

           A = [   G     ]        b = [ d ]
               [ √mu · R ]            [ 0 ]

       a função objetivo vira um único problema de mínimos quadrados:

           ||A p - b||² = ||G p - d||² + ||√mu R p - 0||²
                        = ||G p - d||² + mu ||R p||²

       É a mesma função de (1), agora na forma padrão ||A p - b||², com A de
       1281 linhas (441 de dados + 840 de regularização) e 441 colunas.

    3) O algoritmo BVLS (Bounded-Variable Least Squares, Stark e Parker, 1995)
       Generaliza o NNLS de Lawson e Hanson (1974). É um método de CONJUNTO
       ATIVO: mantém as variáveis em dois grupos,
           livres   — estritamente positivas, calculadas por mínimos quadrados
                      comuns;
           travadas — fixas em zero;
       e repete:
           a) resolve mínimos quadrados só com as livres (travadas em zero);
           b) se alguma livre ficou negativa, recua na direção da solução
              anterior até a primeira delas tocar o zero e a move para as
              travadas;
           c) para cada travada, olha o gradiente g = A^T (A p - b). Se g_i < 0,
              aumentar p_i reduziria o erro, então ela é liberada.
       Para quando nenhuma livre fica negativa e nenhuma travada quer sair.
       Essas são as condições de Karush-Kuhn-Tucker (KKT) do problema:
           p_i > 0  ->  g_i = 0   (não há como melhorar mexendo nela)
           p_i = 0  ->  g_i >= 0  (aumentá-la só pioraria o ajuste)
       Como a função objetivo e o conjunto p >= 0 são convexos, o mínimo
       encontrado é o MÍNIMO GLOBAL: o resultado não depende de chute inicial.
    """
    # Sistema aumentado: dados em cima, regularização embaixo
    A = np.vstack([G, np.sqrt(mu) * R_dense])
    b = np.concatenate([d, np.zeros(R_dense.shape[0])])

    # Mínimos quadrados com limites: 0 <= p_i < infinito para todo i
    result = lsq_linear(A, b, bounds=(0.0, np.inf), method="bvls")

    # O BVLS pode devolver valores como -1e-18 por arredondamento de máquina;
    # zerá-los não altera a solução e garante p >= 0 estritamente.
    return np.maximum(result.x, 0.0)


def scan_positive_layer(G: np.ndarray, d: np.ndarray, R_dense: np.ndarray,
                        mus: np.ndarray) -> pd.DataFrame:
    """
    Resolve o problema com positividade para cada mu da lista e registra o
    RMS do ajuste aos dados. Com restrição não há como aproveitar a fatoração
    de um mu para outro, então cada mu é um BVLS completo (< 1 s cada aqui).
    O resultado alimenta select_by_discrepancy e a figura escolha_mu.png.
    """
    records = []
    t0 = time.perf_counter()
    print(f"Varrendo {len(mus)} valores de mu com p >= 0...")
    for mu in mus:
        p = solve_positive_layer(G, d, R_dense, mu)
        records.append({"reg_value": mu,
                        "rmse": float(np.sqrt(np.mean((G @ p - d) ** 2))),
                        "p_est": p})
    print(f"Varredura com positividade: {time.perf_counter() - t0:.1f} s")
    return pd.DataFrame(records)


def select_by_discrepancy(df_scan: pd.DataFrame, sigma: float) -> dict:
    """
    Princípio da discrepância (Morozov): escolhe o MAIOR mu cujo RMS do ajuste
    ainda fica dentro do nível do ruído (rmse <= sigma).

    A ideia: o dado contém sinal + ruído. Um ajuste com RMS menor que sigma
    significa que a camada está reproduzindo o ruído como se fosse sinal (foi
    o que a curva L fez na solução A: ajuste de 1,8 nT para ruído de 5,5 nT).
    Um ajuste muito acima de sigma significa que parte do sinal ficou de fora.
    O equilíbrio é RMS ≈ sigma. Entre os mu que respeitam rmse <= sigma, o
    maior dá a solução mais suave que ainda explica o sinal.

    Como o RMS cresce com mu, isso equivale a pegar o último mu da varredura
    antes de o ajuste ultrapassar o ruído (linha tracejada em escolha_mu.png).
    Se nenhum mu atingir o ruído, devolve o de menor RMS e avisa.
    """
    dentro = df_scan[df_scan["rmse"] <= sigma]
    if dentro.empty:
        print("AVISO: nenhum mu ajustou os dados ao nível do ruído; usando o de menor RMS.")
        return df_scan.loc[df_scan["rmse"].idxmin()].to_dict()
    return dentro.loc[dentro["reg_value"].idxmax()].to_dict()


def metrics(estimated: np.ndarray, reference: np.ndarray) -> dict:
    """Métricas de comparação. Resíduo = estimado - referência."""
    res = estimated - reference
    amp = reference.max() - reference.min()
    rmse = float(np.sqrt(np.mean(res ** 2)))
    return {"rmse": rmse,
            "erro_rel": 100.0 * rmse / amp,
            "max_abs": float(np.max(np.abs(res))),
            "corr": float(np.corrcoef(estimated, reference)[0, 1]),
            "min": float(estimated.min()),
            "max": float(estimated.max())}


def plot_rtp_two_solutions(X, Y, obs, true_pole, rtp_a, rtp_b,
                           mu_a, mu_b, figname):
    """
    Linha 1: RTP verdadeiro | RTP solução A | resíduo A
    Linha 2: dado observado | RTP solução B | resíduo B
    Os três mapas de RTP compartilham a escala; os dois resíduos também.
    Resíduo = estimado - verdadeiro.
    """
    res_a, res_b = rtp_a - true_pole, rtp_b - true_pole
    vmin = min(true_pole.min(), rtp_a.min(), rtp_b.min())
    vmax = max(true_pole.max(), rtp_a.max(), rtp_b.max())
    lv = np.linspace(vmin, vmax, 21)
    rmax = max(np.abs(res_a).max(), np.abs(res_b).max())
    lr = np.linspace(-rmax, rmax, 21)

    fig, ax = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)

    def mapa(a, Z, levels, cmap, titulo):
        c = a.contourf(X, Y, Z, levels=levels, cmap=cmap)
        fig.colorbar(c, ax=a, label="nT")
        a.set_title(titulo)
        a.set_aspect("equal")
        a.set_xlabel("Easting (m)")

    m_a, m_b = metrics(rtp_a, true_pole), metrics(rtp_b, true_pole)
    mapa(ax[0, 0], true_pole, lv, "jet", "RTP verdadeiro\n(resposta direta com I = 90°)")
    mapa(ax[0, 1], rtp_a, lv, "jet", f"A) Sem restrição, curva L\nμ = {mu_a:.2e}")
    mapa(ax[0, 2], res_a, lr, "seismic",
         f"Resíduo A (estimado − verdadeiro)\nRMSE {m_a['rmse']:.2f} nT · erro {m_a['erro_rel']:.2f}%")
    c = ax[1, 0].contourf(X, Y, obs, levels=21, cmap="jet")
    fig.colorbar(c, ax=ax[1, 0], label="nT")
    ax[1, 0].set_title("Dado observado (entrada)\nI = 10°, com ruído")
    ax[1, 0].set_aspect("equal"); ax[1, 0].set_xlabel("Easting (m)")
    mapa(ax[1, 1], rtp_b, lv, "jet", f"B) Positividade, discrepância\nμ = {mu_b:.2e}")
    mapa(ax[1, 2], res_b, lr, "seismic",
         f"Resíduo B (estimado − verdadeiro)\nRMSE {m_b['rmse']:.2f} nT · erro {m_b['erro_rel']:.2f}%")
    for a in ax[:, 0]:
        a.set_ylabel("Northing (m)")

    fig.suptitle("RTP via camada equivalente de prismas — sem restrição vs positividade",
                 fontsize=16, fontweight="bold")
    fig.savefig(figname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura salva em: {figname}")


def plot_magnetizations(X, Y, p_a, p_b, prism, figname):
    """Magnetização estimada na camada pelas duas soluções, mesma escala."""
    vmax = max(np.abs(p_a).max(), np.abs(p_b).max())
    lv = np.linspace(-vmax, vmax, 21)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for a, p, tit in [(ax[0], p_a, "A) Sem restrição"), (ax[1], p_b, "B) Positividade")]:
        c = a.contourf(X, Y, p, levels=lv, cmap="RdBu_r")
        fig.colorbar(c, ax=a, label="A/m")
        w, e, s, n = prism[:4]
        a.plot([w, e, e, w, w], [s, s, n, n, s], "k--", lw=1.2, label="contorno do corpo")
        a.set_title(f"{tit}\n{(p < 0).sum()} negativos · faixa {p.min():.1f} a {p.max():.1f} A/m")
        a.set_aspect("equal"); a.set_xlabel("Easting (m)")
    ax[0].set_ylabel("Northing (m)")
    ax[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Magnetização estimada na camada equivalente", fontsize=14, fontweight="bold")
    fig.savefig(figname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura salva em: {figname}")


def plot_mu_selection(df_a, df_b, sigma, mu_lcurve, mu_disc_a, mu_disc_b, figname):
    """Ajuste aos dados versus mu nas duas varreduras, com o nível do ruído."""
    fig, a = plt.subplots(figsize=(8, 5), constrained_layout=True)
    a.loglog(df_a["reg_value"], df_a["rmse"], "o-", color="#c0392b", label="sem restrição")
    a.loglog(df_b["reg_value"], df_b["rmse"], "s-", color="#2c5aa0", label="positividade")
    a.axhline(sigma, color="k", ls="--", lw=1, label=f"nível do ruído σ = {sigma:.2f} nT")
    a.axvline(mu_lcurve, color="#c0392b", ls=":", lw=1.2, label=f"curva L (A): μ = {mu_lcurve:.1e}")
    a.axvline(mu_disc_b, color="#2c5aa0", ls=":", lw=1.2, label=f"discrepância (B): μ = {mu_disc_b:.1e}")
    a.set_xlabel("μ (parâmetro de regularização)")
    a.set_ylabel("RMS do ajuste aos dados (nT)")
    a.set_title("Escolha do parâmetro de regularização")
    a.grid(alpha=0.3, which="both")
    a.legend(fontsize=8)
    fig.savefig(figname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura salva em: {figname}")


# =========================================================================
# BLOCO 4 — aplicação
# =========================================================================

# ---------------------------------------------------------
# 1. Dados sintéticos (idênticos ao script corrigido)
# ---------------------------------------------------------
xmin, xmax, ymin, ymax = 0, 100, 0, 100
spacing = 5
height = 0
region = (xmin, xmax, ymin, ymax)
coordinates = vd.grid_coordinates(region, spacing=spacing, extra_coords=height)
prism_dimensions = [25.0, 75.0, 25.0, 75.0, -20.0, -10.0]
inc1 = 90
inc2 = 10
dec = 0
intensity = 3

seed_eq = 43
seed_pole = 44
noise_percentage = 2

grid_shape = coordinates[0].shape
N = coordinates[0].size

pole_mag_vector = get_magnetization_vector_enu(intensity, get_direction_cosines_enu(inc1, dec))
pole_b = hm.prism_magnetic(coordinates, prism_dimensions, pole_mag_vector, field="b")
pole_tfa = hm.total_field_anomaly(list(pole_b), inc1, dec)

eq_mag_vector = get_magnetization_vector_enu(intensity, get_direction_cosines_enu(inc2, dec))
eq_b = hm.prism_magnetic(coordinates, prism_dimensions, eq_mag_vector, field="b")
eq_tfa = hm.total_field_anomaly(list(eq_b), inc2, dec)
eq_tfa_noisy = add_gaussian_noise(eq_tfa, noise_percentage, seed_eq)

# >>> NOVO: nível do ruído, com a mesma regra de add_gaussian_noise.
# Em dado real ele não é conhecido e precisa ser estimado (repetibilidade das
# leituras, estação base, ou o resíduo de um ajuste regional).
sigma_noise = (noise_percentage / 100.0) * np.max(np.abs(eq_tfa))

print(f"Dado observado (I={inc2}): {eq_tfa.min():8.2f} a {eq_tfa.max():8.2f} nT")
print(f"RTP verdadeiro (I={inc1}): {pole_tfa.min():8.2f} a {pole_tfa.max():8.2f} nT")
print(f"Nível do ruído: sigma = {sigma_noise:.2f} nT")

# ---------------------------------------------------------
# 2. Ordenação espacial e camada equivalente
# ---------------------------------------------------------
df = pd.DataFrame({
    "X": coordinates[0].ravel(),
    "Y": coordinates[1].ravel(),
    "Z": coordinates[2].ravel(),
    "eq_tfa": eq_tfa_noisy.ravel(),
    "pole_tfa_clean": pole_tfa.ravel(),
})
df_sub = df.sort_values(by=["Y", "X"]).reset_index(drop=True)

easting = df_sub["X"].to_numpy()
northing = df_sub["Y"].to_numpy()
upwards = df_sub["Z"].to_numpy()
d_obs_eq = df_sub["eq_tfa"].to_numpy()
d_true_pole = df_sub["pole_tfa_clean"].to_numpy()

dx = float(np.diff(np.sort(np.unique(easting)))[0])
dy = float(np.diff(np.sort(np.unique(northing)))[0])

z_top = -10.0
z_bottom = -15.0

print("\n--- Construindo a camada equivalente ---")
prisms_layer = build_prism_layer(N, dx, dy, easting, northing, z_bottom, z_top)
RTR, _, _ = build_rtr_with_profiling(df_sub, N, sys.modules[__name__])

# >>> NOVO: matriz de diferenças R, com conferência de R^T R = RTR
# Os pares de vizinhos vêm da mesma find_all_connections que a colega usa para
# montar o RTR. A conferência abaixo garante que as soluções A e B usam
# exatamente a mesma regularização.
tags_grid = np.arange(N).reshape(grid_shape).tolist()
connections = find_all_connections(tags_grid)
R = build_difference_matrix(connections, N)
dif_rtr = abs(R.T @ R - RTR).max()
print(f"Conferência R^T R = RTR: max |diferença| = {dif_rtr:.1e}")
assert dif_rtr < 1e-12, "R^T R não reproduz o RTR"
R_dense = R.toarray()

print(f"\n--- Jacobiana no equador (I={inc2}) ---")
G_eq, _, _ = build_magnetic_jacobian_with_profiling(
    N, easting, northing, upwards, prisms_layer, inc2, dec)
print(f"\n--- Jacobiana no polo (I={inc1}) ---")
G_pole, _, _ = build_magnetic_jacobian_with_profiling(
    N, easting, northing, upwards, prisms_layer, inc1, dec)

# ---------------------------------------------------------
# 3. Solução A — sem restrição, curva L (como no script corrigido)
# ---------------------------------------------------------
print("\n--- Solução A: sem restrição ---")
df_a, best_lcurve, _, _ = run_tikhonov_l_curve_scan_ces(
    G_eq, d_obs_eq, RTR, reg_min=1e-6, reg_max=1e2, steps=20)
mu_a = best_lcurve["reg_value"]
p_a = best_lcurve["p_est"]
rtp_a = G_pole @ p_a

# ---------------------------------------------------------
# 4. Solução B — positividade, princípio da discrepância
# ---------------------------------------------------------
# Sequência:
#   a) para cada mu da lista, resolve (1) com p >= 0 pelo BVLS;
#   b) escolhe o maior mu com RMS do ajuste <= sigma (discrepância);
#   c) projeta a magnetização escolhida no polo: rtp_b = G_pole @ p_b,
#      exatamente como na solução A.
# A faixa de mu vai mais alto que a da solução A porque, com a restrição, o
# ajuste só alcança o nível do ruído para mu da ordem de 10 a 100.
print("\n--- Solução B: positividade ---")
mus_b = np.logspace(-3, 4, 15)
df_b = scan_positive_layer(G_eq, d_obs_eq, R_dense, mus_b)
best_b = select_by_discrepancy(df_b, sigma_noise)
mu_b = best_b["reg_value"]
p_b = best_b["p_est"]
rtp_b = G_pole @ p_b

# Para o quadro comparativo: discrepância também na solução sem restrição.
# Essa linha intermediária separa os dois efeitos: quanto do ganho vem só de
# trocar o critério do mu, e quanto vem da positividade.
best_a_disc = select_by_discrepancy(df_a, sigma_noise)

# ---------------------------------------------------------
# 5. Quadro comparativo
# ---------------------------------------------------------
casos = [
    ("sem restrição, curva L (A)", mu_a, p_a, best_lcurve["rmse"]),
    ("sem restrição, discrepância", best_a_disc["reg_value"], best_a_disc["p_est"], best_a_disc["rmse"]),
    ("positividade, discrepância (B)", mu_b, p_b, best_b["rmse"]),
]
linhas = []
print("\n--- Resultados (resíduo = estimado - verdadeiro) ---")
print(f"{'solução':32s} {'mu':>9s} {'ajuste':>8s} {'RMSE':>8s} {'erro':>7s} {'corr':>7s} "
      f"{'neg':>4s} {'faixa do RTP (nT)':>19s}")
for nome, mu, p, aj in casos:
    m = metrics(G_pole @ p, d_true_pole)
    print(f"{nome:32s} {mu:9.2e} {aj:6.2f}nT {m['rmse']:6.2f}nT {m['erro_rel']:6.2f}% "
          f"{m['corr']:7.4f} {int((p < 0).sum()):4d} {m['min']:8.1f} a {m['max']:6.1f}")
    linhas.append({"solucao": nome, "mu": mu, "ajuste_rms_nT": aj, **m,
                   "magnetizacoes_negativas": int((p < 0).sum()),
                   "mag_min_Am": float(p.min()), "mag_max_Am": float(p.max())})
print(f"{'RTP verdadeiro':32s} {'':9s} {'':8s} {'':8s} {'':7s} {'':7s} {'':4s} "
      f"{d_true_pole.min():8.1f} a {d_true_pole.max():6.1f}")
pd.DataFrame(linhas).to_csv(OUTPUT_FOLDER / "quadro_comparativo.csv", index=False)

# ---------------------------------------------------------
# 6. Figuras e tabela
# ---------------------------------------------------------
Xg, Yg = easting.reshape(grid_shape), northing.reshape(grid_shape)
print()
plot_rtp_two_solutions(Xg, Yg, d_obs_eq.reshape(grid_shape), d_true_pole.reshape(grid_shape),
                       rtp_a.reshape(grid_shape), rtp_b.reshape(grid_shape),
                       mu_a, mu_b, OUTPUT_FOLDER / "comparacao_rtp.png")
plot_magnetizations(Xg, Yg, p_a.reshape(grid_shape), p_b.reshape(grid_shape),
                    prism_dimensions, OUTPUT_FOLDER / "magnetizacao_camada.png")
plot_mu_selection(df_a, df_b, sigma_noise, mu_a, best_a_disc["reg_value"], mu_b,
                  OUTPUT_FOLDER / "escolha_mu.png")

# >>> NOVO: resíduo com o mesmo sinal em todo lugar (estimado - verdadeiro)
saida = pd.DataFrame({
    "X": easting, "Y": northing, "Z": upwards,
    "obs_equador": d_obs_eq,
    "rtp_verdadeiro": d_true_pole,
    "rtp_A_sem_restricao": rtp_a,
    "residuo_A": rtp_a - d_true_pole,
    "mag_A_Am": p_a,
    "rtp_B_positividade": rtp_b,
    "residuo_B": rtp_b - d_true_pole,
    "mag_B_Am": p_b,
})
saida.to_csv(OUTPUT_FOLDER / "resultados_rtp.csv", index=False)
print(f"Tabelas salvas em: {OUTPUT_FOLDER}")
