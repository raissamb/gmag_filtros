"""
This module contains utility functions like plotting.
"""


# Imports
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, Union, Any, Literal, List
import json

# Functions

# I/O Functions




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



def plot_field_contourf(xcoords: np.ndarray, 
                        ycoords: np.ndarray,
                        field: np.ndarray,
                        title: str,
                        xlabel: str,
                        ylabel: str,
                        unit: str,
                        figname: Union[str, Path],
                        levels: Optional[Union[list, np.ndarray, int]] = None,
                        cmap: str = "viridis",
                        show: bool = False) -> None:
    """
    Function to plot field data.

    Parameters
    ----------
    xcoords : np.ndarray
        2D array containing the coordinates in the X axis, Cartesian System.
    ycoords : np.ndarray
        2D array containing the coordinates in the Y axis, Cartesian System.
    field : np.ndarray
        2D array of the field data to be plotted.
    title : str
        Plot title.
    xlabel : str
        Label for the X axis.
    ylabel : str
        Label for the Y axis.
    unit : str
        Unit label for the colorbar.
    figname : str | Path
        Filename or Path object to save the figure.
    levels : list | np.ndarray | int | None, optional
        Determines the number and positions of the contour lines / regions.
        If None, matplotlib automatically determines the levels.
    cmap : str, optional
        Colormap to use (default is 'viridis').
    show : bool, optional
        If True, displays the figure. If False, closes the figure to save memory (default is False).
    """

    fig = plt.figure()    
    # Explicitly type the dictionary to accept mixed value types
    contour_kwargs: dict[str, Any] = {"cmap": cmap}
    
    # Bundle keyword arguments to handle optional custom levels
    if levels is not None:
        contour_kwargs["levels"] = levels
        contour_kwargs["extend"] = "both"
        
    plt.contourf(xcoords, ycoords, field, **contour_kwargs)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.gca().set_aspect("equal")
    plt.colorbar(label=unit)
    
    fig.savefig(figname, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)



def plot_transformation_results(
    xcoord: np.ndarray, 
    ycoord: np.ndarray,
    original: np.ndarray, 
    transformed: np.ndarray,
    transformation_name: str, 
    figname: str | Path, 
    transformation_label: str,
    show: bool = False
):
    """
    Plots Original vs. Transformed data and their difference with consistent scaling.

    Parameters
    ----------
    xcoord, ycoord : np.ndarray
        Coordinates for the grid (usually meshgrid or 2D reshaped array).
    original : np.ndarray
        Original observed data grid.
    transformed : np.ndarray
        Transformed (e.g., Upward Continued) data grid.
    modelname : str
        Name of the model for the overarching title.
    figname : str or Path
        Path to save the figure.
    transformation_label : str
        Label for the transformation plot (e.g., 'Upward Cont. (500m)').
    show : bool
        Whether to display the plot window.
    """

    # 1. Calculate Difference
    difference = original - transformed

    # 2. Determine Global Scales for Original/Transformed
    # Sharing the scale accurately visualizes signal attenuation or amplification
    global_min = min(np.min(original), np.min(transformed))
    global_max = max(np.max(original), np.max(transformed))
    levels = np.linspace(global_min, global_max, 21)

    # 3. Setup Figure (1 row, 3 columns)
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(16, 5), constrained_layout=True)

    # --- Plot 1: Original Data ---
    c1 = ax1.contourf(xcoord, ycoord, original, levels=levels, cmap='jet',
                      vmin=global_min, vmax=global_max)
    fig.colorbar(c1, ax=ax1, label='nT', orientation='vertical')
    ax1.set_title('Original Data')
    ax1.set_xlabel('Easting (m)')
    ax1.set_ylabel('Northing (m)')

    # --- Plot 2: Transformed Data ---
    c2 = ax2.contourf(xcoord, ycoord, transformed, levels=levels, cmap='jet',
                      vmin=global_min, vmax=global_max)
    fig.colorbar(c2, ax=ax2, label='nT', orientation='vertical')
    ax2.set_title(f'Transformed\n{transformation_label}')
    ax2.set_xlabel('Easting (m)')

    # --- Plot 3: Difference (Original - Transformed) ---
    # Diverging colormap centered on 0
    diff_abs_max = np.max(np.abs(difference))
    c3 = ax3.contourf(xcoord, ycoord, difference, cmap='seismic',
                      levels=21, vmin=-diff_abs_max, vmax=diff_abs_max)
    fig.colorbar(c3, ax=ax3, label='Difference (nT)', orientation='vertical')
    ax3.set_title('Difference (Original - Transformed)')
    ax3.set_xlabel('Easting (m)')

    # Main Title
    fig.suptitle(f'Transformation Results: {transformation_name}', fontsize=16, fontweight='bold')

    # Save
    fig.savefig(figname, dpi=300, bbox_inches="tight")
    print(f"Transformation plot saved to: {figname}")

    # Memory Management
    if show:
        plt.show()
    else:
        plt.close(fig)


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


def plot_matrix(
    matrix: np.ndarray,
    title: str,
    figname: Union[str, Path],
    show: bool = False,
    origin: Literal['upper', 'lower'] = 'lower',  # FIXED: Restricts strings to exact valid types
    cmap: str = 'viridis'
) -> None:
    """
    Plots a matrix as a heatmap and saves it to a file.

    This function uses the Object-Oriented Matplotlib API for better control
    and memory management.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix to be plotted.
    title : str
        Title for the figure.
    figname : str or Path
        Output path for the saved figure.
    show : bool, optional
        If True, displays the plot. If False, just saves it. Default is False.
    origin : {'lower', 'upper'}, optional
        'lower' (default) places (0,0) at bottom-left (Cartesian/Map).
        'upper' places (0,0) at top-left (Matrix standard).
    cmap : str, optional
        Colormap to use. Default is 'viridis'.

    Returns
    -------
    None
    """
    # 1. Use Object-Oriented API (Safer)
    fig, ax = plt.subplots(figsize=(8, 6))

    # 2. Plot data - Pylance is completely happy now because the literal sets match perfectly!
    im = ax.imshow(matrix, cmap=cmap, origin=origin, aspect='auto')

    # 3. Add Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Value')

    # 4. Labels and Title
    ax.set_title(f'{title}\nShape: {matrix.shape}')
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')

    # 5. Save the figure
    save_path = Path(figname)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {save_path}")

    # 6. Handling Display vs. Memory
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_physical_property_distribution_matrix(easting: np.ndarray, 
                              northing: np.ndarray, 
                              p_est: np.ndarray, 
                              title: str,
                              figname: Union[str, Path],
                              cmap: str = "turbo") -> None:
    """
    Reshapes a 1D estimated parameter vector into a 2D matrix and plots 
    the discrete physical property distribution of the equivalent layer.
    """
    # 1. Determine the grid dimensions dynamically
    nx = len(np.unique(easting))
    ny = len(np.unique(northing))
    
    # 2. Reshape the 1D arrays into 2D spatial matrices
    X = easting.reshape((ny, nx))
    Y = northing.reshape((ny, nx))
    P = p_est.reshape((ny, nx))
    
    # 3. Generate the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # pcolormesh visually represents the discrete blocks (prisms) perfectly
    pc = ax.pcolormesh(X, Y, P, cmap=cmap, shading="nearest")
    
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    fig.colorbar(pc, label="Magnetization (A/m)")
    
    fig.savefig(figname, dpi=300, bbox_inches="tight")
    plt.close(fig)


def find_read_json_file(main_dir: Path, pattern: str) -> dict:
    """
    Searches for a JSON file matching the pattern and returns its contents as a dictionary.
    Raises FileNotFoundError or ValueError if it fails, ensuring it never returns None.
    """
    matching_files = list(main_dir.rglob(pattern))

    if not matching_files:
        raise FileNotFoundError(f"No matching JSON file found for pattern: '{pattern}' in {main_dir}")
    
    # Take the first matching file found
    target_file = matching_files[0]
    print(f"Found file: {target_file}")
    
    # Open and read the JSON content
    try:
        data = json.loads(target_file.read_text(encoding='utf-8'))
        print("Successfully read JSON data")
        return data
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Error: The file {target_file} is not a valid JSON document.\n{e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while reading {target_file}: {e}")


def create_pipeline_config(
    savepath: Path,
    data_type: str,
    filename: str,
    easting_col: str = "X",
    northing_col: str = "Y",
    upwards_col: str = "Z",
    field: str = "obs",
    z_top: float = -300.0,
    z_bottom: float = -800.0,
    incf: float = 90.0,
    decf: float = 0.0,
    incs: float = 90.0,
    decs: float = 0.0,
    transformations: dict = None
) -> None:
    """
    Creates and saves the unified pipeline configuration (pipeline_config.json).
    """
    if transformations is None:
        # Default transformations block if none provided
        transformations = {
            "upward_continuation": {"execute": False, "dz": 500},
            "downward_continuation": {"execute": False, "dz": -150},
            "reduction_to_pole": {"execute": False, "inc_new": 90, "dec_new": 0}
        }

    config = {
        "project_settings": {
            "data_type": data_type,
            "filename": filename
        },
        "data_columns": {
            "easting_col": easting_col,
            "northing_col": northing_col,
            "upwards_col": upwards_col,
            "field": field
        },
        "inversion_parameters": {
            "z_top": z_top,
            "z_bottom": z_bottom,
            "incf": incf,
            "decf": decf,
            "incs": incs,
            "decs": decs
        },
        "transformations": transformations
    }
    
    filepath = savepath / "pipeline_config.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"Pipeline config successfully saved to: {filepath}")



def plot_prisms_3d(
    prisms: Union[list, tuple, np.ndarray],
    figname: Union[str, Path],
    title: str = "Synthetic Source Model",
    region_bounds: Optional[tuple] = None,
    z_bounds: Optional[tuple] = None,
    elev: float = 20,
    azim: float = -60,  # Changed default to -60 for a better standard view
    show: bool = False
) -> None:
    """
    Plots one or multiple 3D rectangular prisms.

    Parameters
    ----------
    prisms : list, tuple, or np.ndarray
        A single prism defined as [west, east, south, north, bottom, top]
        OR a list of prisms e.g., [[W, E, S, N, B, T], [W, E, S, N, B, T]].
    figname : str | Path
        Filename or Path object to save the figure.
    title : str
        Plot title.
    region_bounds : tuple, optional
        Tuple defining the lateral limits of the plot (West, East, South, North). 
        If None, the plot focuses tightly around the source.
    z_bounds : tuple, optional
        Tuple defining the vertical limits (Bottom, Top).
        If None, auto-scales to the source.
    show : bool, optional
        If True, displays the figure. If False, closes the figure to save memory (default is False).
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Check if the input is a single prism (1D) or a list of prisms (2D)
    prisms_array = np.atleast_2d(prisms)

    for p in prisms_array:
        west, east, south, north, bottom, top = p
        
        # Define the 8 vertices of the prism
        x = [west, east, east, west, west, east, east, west]
        y = [south, south, north, north, south, south, north, north]
        z = [bottom, bottom, bottom, bottom, top, top, top, top]
        
        # Define the edges that connect the vertices to form a box
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0], # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4], # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]  # Vertical pillars
        ]
        
        # Plot each edge
        for edge in edges:
            ax.plot(
                [x[edge[0]], x[edge[1]]], 
                [y[edge[0]], y[edge[1]]], 
                [z[edge[0]], z[edge[1]]], 
                color='red', linewidth=1.5
            )

    ax.set_title(title)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_zlabel("Depth / Upwards (m)")
    
    # Apply optional axis limits if provided
    if region_bounds is not None:
        ax.set_xlim(region_bounds[0], region_bounds[1])
        ax.set_ylim(region_bounds[2], region_bounds[3])
        
    if z_bounds is not None:
        ax.set_zlim(z_bounds[0], z_bounds[1])
    
    # Adjust viewing angle for better 3D perspective
    ax.view_init(elev=elev, azim=azim)
    
    # Save and handle display
    fig.savefig(figname, dpi=300, bbox_inches="tight")
    print(f"3D Prism plot saved to: {figname}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)