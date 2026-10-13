"""
This module contains utility functions like plotting.
"""


# Imports
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, Union, Any, Literal
import json

# Functions

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
    figname: str, 
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