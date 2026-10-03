"""
This module contains utility functions like plotting.
"""


# Imports
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, Union, Any


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

    
    # # Bundle keyword arguments to handle optional custom levels
    # contour_kwargs = {"cmap": cmap}
    # if levels is not None:
    #     contour_kwargs["levels"] = levels
    #     contour_kwargs["extend"] = "both"
        
    # plt.contourf(xcoords, ycoords, field, **contour_kwargs)
    
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